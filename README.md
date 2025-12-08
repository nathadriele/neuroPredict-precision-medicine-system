# NeuroPredict: Precision Medicine System for Refractory Epilepsy

<img width="1462" height="557" alt="Captura de tela de 2025-12-02 11-47-23" src="https://github.com/user-attachments/assets/7feafd7a-07b0-4a47-9d19-4e35022f1d03" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview
NeuroPredict is an AI-driven precision medicine system that integrates clinical, genomic, EEG, neuroimaging, and medical literature data to predict individual treatment response in refractory epilepsy. The platform combines knowledge graphs, deep learning models, and large language models (LLMs) to provide evidence-based, personalized therapeutic recommendations.

## Key Features
- Multimodal Analysis: Integration of clinical, genomic, EEG, and neuroimaging data
- Medical Knowledge Graph: Semantic representation of relationships between genes, drugs, symptoms, and phenotypes
- Ensemble Learning Prediction: XGBoost, LightGBM, CatBoost, and neural network models
- Explainability: SHAP values and feature importance analysis
- Interactive Web Interface: Dashboard for visualization and analysis
- RAG with LLMs: Retrieval-augmented generation system for evidence-based recommendations
- MLOps Pipeline: Model versioning, monitoring, and CI/CD

## Architecture

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

## Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Neo4j 5.x
- PostgreSQL 15+
- At least 16GB RAM
- GPU

## Quick Installation
```bash
git clone https://github.com/nathadriele/neuroPredict-precision-medicine-system.git
cd neuropredict

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

pip install -e ".[dev]"

cp .env.example .env
docker-compose up -d

alembic upgrade head

python scripts/load_sample_data.py
```

## Usage

### 1. Model Training
```bash
python -m neuropredict.train --config configs/training_config.yaml

python -m neuropredict.train --config configs/training_config.yaml --hpo
```

### 2. API REST
```bash
uvicorn neuropredict.api.main:app --reload --host 0.0.0.0 --port 8000

# http://localhost:8000/docs
```

### 3. Prediction
<img width="954" height="668" alt="image" src="https://github.com/user-attachments/assets/45a0b3e7-9a8a-4fe3-b117-5badfa8041bd" />

### 4. Web Dashboard
<img width="726" height="178" alt="image" src="https://github.com/user-attachments/assets/925003b1-e3a1-44e5-af22-4bc93b32b073" />

## Testing
<img width="718" height="317" alt="image" src="https://github.com/user-attachments/assets/55625255-a250-4c04-beba-e4935a128e39" />

## Performance
<img width="719" height="406" alt="image" src="https://github.com/user-attachments/assets/f01c13f0-8a21-4b7f-a31a-81934e1491b1" />

## Contributing
Contributions are welcome: 

1. Fork the project
2. Create a branch for your feature (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Important Notes
This system is for research and educational purposes only. It must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified healthcare professional.

## Quote
If you use this project in your research, cite:

```bibtex
@software{neuropredict2024,
  author = {Your Name},
  title = {NeuroPredict: Precision Medicine System for Refractory Epilepsy},
  year = {2024},
  url = {https://github.com/your-username/neuroPredict-precision-medicine-system}
}
```
