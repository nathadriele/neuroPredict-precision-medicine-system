"""
API REST para o sistema NeuroPredict.
Fornece endpoints para predição, RAG e análise de pacientes.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from neuropredict.config import get_settings
from neuropredict.knowledge_graph.graph import Neo4jKnowledgeGraph
from neuropredict.models.ensemble import EnsembleModel
from neuropredict.rag.system import EpilepsyRAGSystem, MedicalVectorStore

# ============================================================================
# Configuração
# ============================================================================

settings = get_settings()

app = FastAPI(
    title="NeuroPredict API",
    description="API para medicina de precisão em epilepsia",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Schemas Pydantic
# ============================================================================

class GeneticVariant(BaseModel):
    """Variante genética."""
    
    gene: str = Field(..., description="Símbolo do gene")
    variant: str = Field(..., description="Nomenclatura da variante")
    variant_type: str = Field(..., description="Tipo de variante")
    
    @field_validator("variant_type")
    @classmethod
    def validate_variant_type(cls, v: str) -> str:
        """Valida tipo de variante."""
        valid_types = ["missense", "nonsense", "frameshift", "splice_site", "synonymous"]
        if v not in valid_types:
            raise ValueError(f"Tipo de variante deve ser um de: {valid_types}")
        return v


class PatientData(BaseModel):
    """Dados do paciente."""
    
    patient_id: str = Field(..., description="ID do paciente")
    age: int = Field(..., ge=0, le=120, description="Idade")
    sex: str = Field(..., description="Sexo (M/F/Other)")
    seizure_type: str = Field(..., description="Tipo de crise")
    seizure_frequency_per_month: float = Field(..., ge=0, description="Frequência mensal")
    age_at_onset: int = Field(..., ge=0, description="Idade de início")
    epilepsy_duration_years: float = Field(..., ge=0, description="Duração da epilepsia")
    previous_treatments: List[str] = Field(
        default_factory=list,
        description="Tratamentos anteriores"
    )
    genetic_variants: List[GeneticVariant] = Field(
        default_factory=list,
        description="Variantes genéticas"
    )
    eeg_features: Optional[Dict[str, float]] = Field(
        None,
        description="Features do EEG"
    )
    mri_features: Optional[Dict[str, float]] = Field(
        None,
        description="Features da MRI"
    )
    
    @field_validator("seizure_type")
    @classmethod
    def validate_seizure_type(cls, v: str) -> str:
        """Valida tipo de crise."""
        valid_types = [
            "focal_aware",
            "focal_impaired_awareness",
            "focal_to_bilateral_tonic_clonic",
            "generalized_tonic_clonic",
            "absence",
            "myoclonic",
            "atonic",
        ]
        if v not in valid_types:
            raise ValueError(f"Tipo de crise deve ser um de: {valid_types}")
        return v


class PredictionRequest(BaseModel):
    """Request para predição."""
    
    patient: PatientData
    explain: bool = Field(default=True, description="Incluir explicações SHAP")


class PredictionResponse(BaseModel):
    """Response de predição."""
    
    patient_id: str
    predicted_treatment: str
    response_probability: float
    confidence: float
    alternative_treatments: List[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]] = None
    timestamp: datetime


class RAGRequest(BaseModel):
    """Request para RAG."""
    
    patient: PatientData
    question: Optional[str] = Field(
        None,
        description="Pergunta específica"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Número de documentos")


class RAGResponse(BaseModel):
    """Response do RAG."""
    
    patient_id: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime


class SimilarPatientsRequest(BaseModel):
    """Request para pacientes similares."""
    
    patient_id: str
    top_k: int = Field(default=10, ge=1, le=50)


class SimilarPatientsResponse(BaseModel):
    """Response de pacientes similares."""
    
    patient_id: str
    similar_patients: List[Dict[str, Any]]
    timestamp: datetime


class HealthCheckResponse(BaseModel):
    """Response de health check."""
    
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]


# ============================================================================
# Gerenciamento de Estado
# ============================================================================

class AppState:
    """Estado global da aplicação."""
    
    def __init__(self) -> None:
        """Inicializa estado."""
        self.ensemble_model: Optional[EnsembleModel] = None
        self.rag_system: Optional[EpilepsyRAGSystem] = None
        self.knowledge_graph: Optional[Neo4jKnowledgeGraph] = None
        self.vector_store: Optional[MedicalVectorStore] = None


state = AppState()


@app.on_event("startup")
async def startup_event() -> None:
    """Inicializa recursos na startup."""
    logger.info("Iniciando NeuroPredict API...")
    
    try:
        # Carrega modelo ensemble
        model_path = Path("models/ensemble_model_v1.pkl")
        if model_path.exists():
            state.ensemble_model = EnsembleModel.load(model_path)
            logger.info("Modelo ensemble carregado")
        else:
            logger.warning(f"Modelo não encontrado em {model_path}")
        
        # Inicializa vector store
        state.vector_store = MedicalVectorStore(
            persist_directory=settings.llm.chroma_persist_directory,
            embedding_model=settings.llm.embedding_model,
        )
        logger.info("Vector store inicializado")
        
        # Inicializa sistema RAG
        state.rag_system = EpilepsyRAGSystem(
            vector_store=state.vector_store,
            llm_provider=settings.llm.provider,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
        )
        logger.info("Sistema RAG inicializado")
        
        # Inicializa grafo de conhecimento
        state.knowledge_graph = Neo4jKnowledgeGraph(
            uri=settings.neo4j.uri,
            user=settings.neo4j.user,
            password=settings.neo4j.password,
            database=settings.neo4j.database,
        )
        logger.info("Grafo de conhecimento conectado")
        
        logger.info("API iniciada com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Limpa recursos no shutdown."""
    logger.info("Encerrando NeuroPredict API...")
    
    if state.knowledge_graph:
        state.knowledge_graph.close()
        logger.info("Conexão com Neo4j encerrada")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthCheckResponse)
async def root() -> HealthCheckResponse:
    """Health check endpoint."""
    services = {
        "model": "loaded" if state.ensemble_model else "not_loaded",
        "rag": "initialized" if state.rag_system else "not_initialized",
        "knowledge_graph": "connected" if state.knowledge_graph else "disconnected",
    }
    
    return HealthCheckResponse(
        status="healthy" if all(v != "not_loaded" for v in services.values()) else "degraded",
        version=settings.version,
        timestamp=datetime.now(),
        services=services,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_treatment(request: PredictionRequest) -> PredictionResponse:
    """
    Prediz melhor tratamento para o paciente.
    
    Args:
        request: Dados do paciente
        
    Returns:
        Predição com probabilidades e explicações
    """
    if state.ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível",
        )
    
    try:
        # Prepara features
        features = _prepare_features(request.patient)
        
        # Faz predição
        prediction = state.ensemble_model.predict(features)
        probabilities = state.ensemble_model.predict_proba(features)
        
        # Mapeia classes para tratamentos
        treatment_map = {
            0: "levetiracetam",
            1: "lamotrigine",
            2: "oxcarbazepine",
        }
        
        predicted_treatment = treatment_map[int(prediction[0])]
        response_probability = float(np.max(probabilities[0]))
        
        # Tratamentos alternativos
        sorted_indices = np.argsort(probabilities[0])[::-1]
        alternative_treatments = [
            {
                "treatment": treatment_map[int(idx)],
                "probability": float(probabilities[0][idx]),
            }
            for idx in sorted_indices[1:3]
        ]
        
        # Explicações SHAP (se solicitado)
        explanation = None
        if request.explain:
            # Implementar SHAP explanation
            explanation = {
                "feature_importance": {},
                "force_plot_data": {},
            }
        
        return PredictionResponse(
            patient_id=request.patient.patient_id,
            predicted_treatment=predicted_treatment,
            response_probability=response_probability,
            confidence=_calculate_confidence(probabilities[0]),
            alternative_treatments=alternative_treatments,
            explanation=explanation,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na predição: {str(e)}",
        )


@app.post(
    "/rag/recommend",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
)
async def rag_recommendation(request: RAGRequest) -> RAGResponse:
    """
    Gera recomendação usando RAG.
    
    Args:
        request: Dados do paciente e pergunta
        
    Returns:
        Recomendação baseada em literatura
    """
    if state.rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sistema RAG não disponível",
        )
    
    try:
        # Prepara dados do paciente
        patient_data = request.patient.model_dump()
        
        # Gera recomendação
        recommendation = state.rag_system.generate_recommendation(
            patient_data=patient_data,
            question=request.question,
            top_k=request.top_k,
        )
        
        # Formata sources
        sources = [
            {
                "id": source.id,
                "content": source.content[:500],  # Trunca para API
                "metadata": source.metadata,
            }
            for source in recommendation.sources
        ]
        
        return RAGResponse(
            patient_id=request.patient.patient_id,
            answer=recommendation.answer,
            sources=sources,
            confidence=recommendation.confidence,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no RAG: {str(e)}",
        )


@app.post(
    "/graph/similar-patients",
    response_model=SimilarPatientsResponse,
    status_code=status.HTTP_200_OK,
)
async def find_similar_patients(
    request: SimilarPatientsRequest,
) -> SimilarPatientsResponse:
    """
    Encontra pacientes similares no grafo de conhecimento.
    
    Args:
        request: ID do paciente e top K
        
    Returns:
        Lista de pacientes similares
    """
    if state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grafo de conhecimento não disponível",
        )
    
    try:
        similar = state.knowledge_graph.find_similar_patients(
            patient_id=request.patient_id,
            top_k=request.top_k,
        )
        
        similar_patients = [
            {
                "patient_id": patient_id,
                "similarity_score": score,
            }
            for patient_id, score in similar
        ]
        
        return SimilarPatientsResponse(
            patient_id=request.patient_id,
            similar_patients=similar_patients,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro ao buscar pacientes similares: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro: {str(e)}",
        )


# ============================================================================
# Funções Auxiliares
# ============================================================================

def _prepare_features(patient: PatientData) -> np.ndarray:
    """
    Prepara features para o modelo.
    
    Args:
        patient: Dados do paciente
        
    Returns:
        Array de features
    """
    # Implementação simplificada
    # Na prática, deve seguir o mesmo pipeline de feature engineering do treinamento
    
    features = [
        patient.age,
        patient.seizure_frequency_per_month,
        patient.age_at_onset,
        patient.epilepsy_duration_years,
        len(patient.previous_treatments),
        len(patient.genetic_variants),
    ]
    
    # Adiciona features dummy para completar
    features.extend([0.0] * 20)  # Preenche até ter features suficientes
    
    return np.array(features).reshape(1, -1)


def _calculate_confidence(probabilities: np.ndarray) -> float:
    """
    Calcula confiança da predição.
    
    Args:
        probabilities: Array de probabilidades
        
    Returns:
        Score de confiança
    """
    # Usa entropia normalizada como proxy de confiança
    # Menor entropia = maior confiança
    
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = -np.log(1 / len(probabilities))
    
    # Normaliza entre 0 e 1 (inverte para que maior seja melhor)
    confidence = 1 - (entropy / max_entropy)
    
    return float(confidence)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    """Handler para HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handler para exceções gerais."""
    logger.error(f"Erro não tratado: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erro interno do servidor",
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "neuropredict.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
    )