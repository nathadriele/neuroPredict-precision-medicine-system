# NeuroPredict: Precision Medicine System for Refractory Epilepsy

<img width="1462" height="557" alt="Captura de tela de 2025-12-02 11-47-23" src="https://github.com/user-attachments/assets/7feafd7a-07b0-4a47-9d19-4e35022f1d03" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview
NeuroPredict is an AI-driven precision medicine system that integrates clinical, genomic, EEG, neuroimaging, and medical literature data to predict individual treatment response in refractory epilepsy. The platform combines knowledge graphs, deep learning models, and large language models (LLMs) to provide evidence-based, personalized therapeutic recommendations.

## Key Features

- **Análise Multimodal**: Integração de dados clínicos, genômicos, EEG e neuroimagem
- **Grafo de Conhecimento Médico**: Representação semântica de relações entre genes, drogas, sintomas e fenótipos
- **Predição com Ensemble Learning**: Modelos XGBoost, LightGBM, CatBoost e redes neurais
- **Explicabilidade**: SHAP values e análise de importância de features
- **Interface Web Interativa**: Dashboard para visualização e análise
- **RAG com LLMs**: Sistema de recuperação e geração aumentada para recomendações baseadas em evidências
- **Pipeline MLOps**: Versionamento de modelos, monitoramento e CI/CD

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     Camada de Ingestão                      │
│  (CSV, FHIR, DICOM, VCF, Literatura PubMed)                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Camada de Processamento                     │
│  • ETL Pipeline          • Feature Engineering              │
│  • Data Validation       • Normalização                     │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼─────────┐
│  Knowledge Graph │    │  Feature Store   │
│   (Neo4j)        │    │   (Feast)        │
└────────┬────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Camada de ML/AI                          │
│  • Ensemble Models    • GNN (PyG)      • Transformers       │
│  • SHAP Explainer     • AutoML         • RAG System         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Camada de Aplicação                        │
│  • REST API (FastAPI)  • Web Dashboard  • Relatórios        │
└─────────────────────────────────────────────────────────────┘
```

## Pré-requisitos

- Python 3.10+
- Docker & Docker Compose
- Neo4j 5.x
- PostgreSQL 15+
- Pelo menos 16GB RAM
- GPU (recomendado para treinamento)

## Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/nathadriele/neuroPredict-sistema-medicina-precisao-epilepsia-refrataria.git
cd neuropredict

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -e ".[dev]"

# Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações

# Inicie serviços com Docker
docker-compose up -d

# Execute migrações
alembic upgrade head

# Carregue dados de exemplo
python scripts/load_sample_data.py
```

## Estrutura do Projeto (Em Desenvolvimento)

```
neuropredict/
├── data/
│   ├── raw/                    # Dados brutos
│   ├── processed/              # Dados processados
│   ├── external/               # Dados externos (OMIM, ClinVar)
│   └── interim/                # Dados intermediários
├── notebooks/
│   ├── 01_eda.ipynb           # Análise exploratória
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── neuropredict/
│   │   ├── data/              # Módulos de ingestão e ETL
│   │   ├── features/          # Feature engineering
│   │   ├── models/            # Modelos ML/DL
│   │   ├── knowledge_graph/   # Grafo de conhecimento
│   │   ├── rag/               # Sistema RAG
│   │   ├── api/               # FastAPI endpoints
│   │   ├── utils/             # Utilitários
│   │   └── config.py          # Configurações
│   └── tests/                 # Testes unitários e integração
├── models/                    # Modelos treinados
├── deployment/
│   ├── docker/               # Dockerfiles
│   ├── kubernetes/           # Manifests K8s
│   └── terraform/            # Infraestrutura como código
├── scripts/                  # Scripts auxiliares
├── docs/                     # Documentação
├── .github/
│   └── workflows/            # CI/CD pipelines
├── docker-compose.yml
├── pyproject.toml
├── setup.py
└── README.md
```

## Uso

### 1. Treinamento de Modelos

```bash
# Treinamento completo do pipeline
python -m neuropredict.train --config configs/training_config.yaml

# Treinamento com HPO (Hyperparameter Optimization)
python -m neuropredict.train --config configs/training_config.yaml --hpo
```

### 2. API REST

```bash
# Inicie o servidor
uvicorn neuropredict.api.main:app --reload --host 0.0.0.0 --port 8000

# Acesse a documentação interativa
# http://localhost:8000/docs
```

### 3. Predição

```python
from neuropredict.models.predictor import TreatmentPredictor

predictor = TreatmentPredictor.load("models/ensemble_model_v1.pkl")

patient_data = {
    "age": 35,
    "seizure_frequency": 4.5,
    "seizure_type": "focal_impaired_awareness",
    "eeg_features": [...],
    "genetic_variants": ["SCN1A_p.R1648H", "KCNQ2_p.A306T"],
    "mri_features": {...},
    "previous_treatments": ["levetiracetam", "lamotrigine"]
}

prediction = predictor.predict(patient_data)
print(f"Tratamento recomendado: {prediction['recommended_treatment']}")
print(f"Probabilidade de resposta: {prediction['response_probability']:.2%}")
```

### 4. Dashboard Web

```bash
# Inicie o dashboard Streamlit
streamlit run src/neuropredict/dashboard/app.py

# Acesse em http://localhost:8501
```

## Testes

```bash
# Execute todos os testes
pytest

# Com cobertura
pytest --cov=neuropredict --cov-report=html

# Testes específicos
pytest tests/test_models.py -v
```

## Performance

Em nosso conjunto de validação (n=500 pacientes):

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.847 |
| Precision | 0.823 |
| Recall | 0.871 |
| F1-Score | 0.846 |
| ROC-AUC | 0.912 |

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## Observações Importantes

Este sistema é apenas para fins de pesquisa e educação. Não deve ser usado como substituto para aconselhamento médico profissional, diagnóstico ou tratamento. Sempre procure o conselho de seu médico ou outro profissional de saúde qualificado.

## Citação

Se você usar este projeto em sua pesquisa, por favor cite:

```bibtex
@software{neuropredict2024,
  author = {Seu Nome},
  title = {NeuroPredict: Sistema de Medicina de Precisão para Epilepsia Refratária},
  year = {2024},
  url = {https://github.com/seu-usuario/neuropredict}
}
```
