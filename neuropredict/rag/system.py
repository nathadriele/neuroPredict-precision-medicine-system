"""
Sistema de Retrieval-Augmented Generation (RAG) para recomendações clínicas.
Integra busca semântica em literatura médica com LLMs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from loguru import logger


# ============================================================================
# Estruturas de Dados
# ============================================================================

@dataclass
class Document:
    """Documento na base de conhecimento."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Resultado de busca semântica."""
    
    documents: List[Document]
    scores: List[float]
    query: str


@dataclass
class RAGResponse:
    """Resposta do sistema RAG."""
    
    answer: str
    sources: List[Document]
    confidence: float
    reasoning: str


# ============================================================================
# Document Loader
# ============================================================================

class MedicalDocumentLoader:
    """Carregador de documentos médicos."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Inicializa loader.
        
        Args:
            chunk_size: Tamanho dos chunks
            chunk_overlap: Overlap entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        self.logger = logger.bind(component="MedicalDocumentLoader")
    
    def load_from_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Carrega PDF médico.
        
        Args:
            pdf_path: Caminho do PDF
            
        Returns:
            Lista de documentos chunked
        """
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Chunking
        chunks = self.text_splitter.split_documents(pages)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"{pdf_path.stem}_{i}",
                content=chunk.page_content,
                metadata={
                    "source": str(pdf_path),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": i,
                },
            )
            documents.append(doc)
        
        self.logger.info(f"Carregados {len(documents)} chunks de {pdf_path.name}")
        
        return documents
    
    def load_from_pubmed(
        self,
        query: str,
        max_results: int = 50,
    ) -> List[Document]:
        """
        Busca e carrega artigos do PubMed.
        
        Args:
            query: Query de busca
            max_results: Número máximo de resultados
            
        Returns:
            Lista de documentos
        """
        from Bio import Entrez
        
        Entrez.email = "seu-email@exemplo.com"
        
        # Busca IDs
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
        )
        record = Entrez.read(handle)
        handle.close()
        
        ids = record["IdList"]
        
        # Busca detalhes
        documents = []
        
        if ids:
            handle = Entrez.efetch(
                db="pubmed",
                id=ids,
                rettype="abstract",
                retmode="text",
            )
            
            abstracts = handle.read().split("\n\n")
            handle.close()
            
            for i, abstract in enumerate(abstracts):
                if abstract.strip():
                    doc = Document(
                        id=f"pubmed_{ids[i]}",
                        content=abstract,
                        metadata={
                            "source": "PubMed",
                            "pmid": ids[i],
                            "query": query,
                        },
                    )
                    documents.append(doc)
        
        self.logger.info(f"Carregados {len(documents)} artigos do PubMed")
        
        return documents
    
    def load_clinical_guidelines(
        self,
        guidelines_path: Path,
    ) -> List[Document]:
        """
        Carrega guidelines clínicos.
        
        Args:
            guidelines_path: Caminho dos guidelines
            
        Returns:
            Lista de documentos
        """
        documents = []
        
        # Carrega múltiplos arquivos
        for file_path in guidelines_path.glob("*.txt"):
            loader = TextLoader(str(file_path))
            content = loader.load()
            
            chunks = self.text_splitter.split_documents(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    id=f"{file_path.stem}_{i}",
                    content=chunk.page_content,
                    metadata={
                        "source": str(file_path),
                        "type": "guideline",
                        "chunk_index": i,
                    },
                )
                documents.append(doc)
        
        self.logger.info(
            f"Carregados {len(documents)} chunks de guidelines"
        )
        
        return documents


# ============================================================================
# Vector Store
# ============================================================================

class MedicalVectorStore:
    """Vector store para documentos médicos usando ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "medical_docs",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """
        Inicializa vector store.
        
        Args:
            persist_directory: Diretório de persistência
            collection_name: Nome da coleção
            embedding_model: Modelo de embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Inicializa ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Cria ou carrega coleção
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        self.logger = logger.bind(component="MedicalVectorStore")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> None:
        """
        Adiciona documentos ao vector store.
        
        Args:
            documents: Lista de documentos
            batch_size: Tamanho do batch
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Gera embeddings
            texts = [doc.content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Adiciona à coleção
            self.collection.add(
                ids=[doc.id for doc in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in batch],
            )
        
        self.logger.info(f"Adicionados {len(documents)} documentos")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Busca semântica.
        
        Args:
            query: Query de busca
            top_k: Número de resultados
            filter_metadata: Filtros de metadata
            
        Returns:
            RetrievalResult
        """
        # Gera embedding da query
        query_embedding = self.embeddings.embed_query(query)
        
        # Busca
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )
        
        # Converte para Documents
        documents = []
        scores = []
        
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                doc = Document(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                )
                documents.append(doc)
                scores.append(results["distances"][0][i])
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
        )
    
    def delete_collection(self) -> None:
        """Deleta coleção."""
        self.client.delete_collection(self.collection_name)
        self.logger.info(f"Coleção {self.collection_name} deletada")


# ============================================================================
# RAG System
# ============================================================================

class EpilepsyRAGSystem:
    """Sistema RAG para recomendações de tratamento em epilepsia."""
    
    def __init__(
        self,
        vector_store: MedicalVectorStore,
        llm_provider: str = "openai",
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
    ) -> None:
        """
        Inicializa sistema RAG.
        
        Args:
            vector_store: Vector store
            llm_provider: Provider do LLM (openai, anthropic)
            model_name: Nome do modelo
            temperature: Temperatura
        """
        self.vector_store = vector_store
        
        # Inicializa LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
            )
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Provider {llm_provider} não suportado")
        
        self.logger = logger.bind(component="EpilepsyRAGSystem")
        
        # Template de prompt
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Cria template de prompt."""
        template = """Você é um especialista em neurologia e epilepsia. 
Sua tarefa é fornecer recomendações de tratamento baseadas em evidências científicas.

Contexto relevante da literatura médica:
{context}

Informações do paciente:
- Tipo de epilepsia: {seizure_type}
- Frequência de crises: {seizure_frequency}
- Genótipo: {genotype}
- Tratamentos anteriores: {previous_treatments}
- Resposta aos tratamentos: {treatment_response}

Pergunta: {question}

Forneça uma recomendação detalhada que inclua:
1. Tratamento sugerido e dosagem
2. Justificativa baseada em evidências
3. Potenciais efeitos colaterais
4. Monitoramento recomendado
5. Alternativas terapêuticas

Importante: Base suas recomendações apenas nas evidências fornecidas no contexto.
Se não houver informação suficiente, indique isso claramente.

Resposta:"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "seizure_type",
                "seizure_frequency",
                "genotype",
                "previous_treatments",
                "treatment_response",
                "question",
            ],
        )
    
    def generate_recommendation(
        self,
        patient_data: Dict[str, Any],
        question: Optional[str] = None,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Gera recomendação de tratamento.
        
        Args:
            patient_data: Dados do paciente
            question: Pergunta específica
            top_k: Número de documentos para recuperar
            
        Returns:
            RAGResponse com recomendação
        """
        # Constrói query para busca
        search_query = self._construct_search_query(patient_data)
        
        # Recupera documentos relevantes
        retrieval_result = self.vector_store.search(
            query=search_query,
            top_k=top_k,
        )
        
        # Prepara contexto
        context = "\n\n".join([
            f"Fonte {i+1}: {doc.content}"
            for i, doc in enumerate(retrieval_result.documents)
        ])
        
        # Prepara prompt
        if question is None:
            question = "Qual o melhor tratamento para este paciente?"
        
        prompt_input = {
            "context": context,
            "seizure_type": patient_data.get("seizure_type", "não especificado"),
            "seizure_frequency": patient_data.get("seizure_frequency", "não especificada"),
            "genotype": ", ".join(patient_data.get("genetic_variants", [])) or "não disponível",
            "previous_treatments": ", ".join(patient_data.get("previous_treatments", [])) or "nenhum",
            "treatment_response": patient_data.get("treatment_response", "não disponível"),
            "question": question,
        }
        
        prompt = self.prompt_template.format(**prompt_input)
        
        # Gera resposta
        self.logger.info("Gerando recomendação com LLM...")
        response = self.llm.invoke(prompt)
        
        # Calcula confiança baseado nos scores de similaridade
        confidence = self._calculate_confidence(retrieval_result.scores)
        
        return RAGResponse(
            answer=response.content,
            sources=retrieval_result.documents,
            confidence=confidence,
            reasoning="Baseado em evidências da literatura médica",
        )
    
    def _construct_search_query(
        self,
        patient_data: Dict[str, Any],
    ) -> str:
        """Constrói query de busca baseada nos dados do paciente."""
        query_parts = []
        
        if "seizure_type" in patient_data:
            query_parts.append(f"epilepsy {patient_data['seizure_type']}")
        
        if "genetic_variants" in patient_data:
            genes = patient_data["genetic_variants"][:3]  # Top 3 genes
            query_parts.append(" ".join(genes))
        
        if "previous_treatments" in patient_data:
            query_parts.append("treatment resistance")
        
        query_parts.append("treatment guidelines recommendations")
        
        return " ".join(query_parts)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calcula confiança baseada nos scores de similaridade."""
        if not scores:
            return 0.0
        
        # Normaliza scores (assumindo distância cosine)
        # Quanto menor a distância, maior a similaridade
        similarities = [1 - score for score in scores]
        
        # Média ponderada (mais peso para os primeiros resultados)
        weights = np.array([1 / (i + 1) for i in range(len(similarities))])
        weights = weights / weights.sum()
        
        confidence = np.average(similarities, weights=weights)
        
        return float(confidence)
    
    def explain_recommendation(
        self,
        recommendation: RAGResponse,
    ) -> str:
        """
        Gera explicação detalhada da recomendação.
        
        Args:
            recommendation: Resposta RAG
            
        Returns:
            Explicação em texto
        """
        explanation = f"""
Recomendação de Tratamento
==========================

{recommendation.answer}

Confiança: {recommendation.confidence:.2%}

Fontes Consultadas:
-------------------
"""
        
        for i, source in enumerate(recommendation.sources, 1):
            explanation += f"\n{i}. {source.metadata.get('source', 'Desconhecida')}"
            explanation += f"\n   PMID: {source.metadata.get('pmid', 'N/A')}"
            explanation += f"\n   Excerto: {source.content[:200]}...\n"
        
        return explanation


# ============================================================================
# Knowledge Base Builder
# ============================================================================

class KnowledgeBaseBuilder:
    """Construtor de base de conhecimento."""
    
    def __init__(
        self,
        vector_store: MedicalVectorStore,
        document_loader: MedicalDocumentLoader,
    ) -> None:
        """Inicializa builder."""
        self.vector_store = vector_store
        self.document_loader = document_loader
        self.logger = logger.bind(component="KnowledgeBaseBuilder")
    
    def build_from_sources(
        self,
        pdf_paths: List[Path],
        pubmed_queries: List[str],
        guidelines_path: Optional[Path] = None,
    ) -> None:
        """
        Constrói base de conhecimento de múltiplas fontes.
        
        Args:
            pdf_paths: Caminhos de PDFs
            pubmed_queries: Queries para PubMed
            guidelines_path: Caminho dos guidelines
        """
        all_documents = []
        
        # Carrega PDFs
        for pdf_path in pdf_paths:
            self.logger.info(f"Carregando {pdf_path.name}...")
            docs = self.document_loader.load_from_pdf(pdf_path)
            all_documents.extend(docs)
        
        # Busca PubMed
        for query in pubmed_queries:
            self.logger.info(f"Buscando PubMed: {query}")
            docs = self.document_loader.load_from_pubmed(query)
            all_documents.extend(docs)
        
        # Carrega guidelines
        if guidelines_path:
            self.logger.info("Carregando guidelines...")
            docs = self.document_loader.load_clinical_guidelines(guidelines_path)
            all_documents.extend(docs)
        
        # Adiciona ao vector store
        self.logger.info(
            f"Adicionando {len(all_documents)} documentos ao vector store..."
        )
        self.vector_store.add_documents(all_documents)
        
        self.logger.info("Base de conhecimento construída com sucesso!")