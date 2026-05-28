# NeuroPredict: Precision Medicine System for Refractory Epilepsy

<img width="1462" height="557" alt="NeuroPredict banner" src="https://github.com/user-attachments/assets/7feafd7a-07b0-4a47-9d19-4e35022f1d03" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**NeuroPredict** is an AI-driven precision medicine system designed to support research on treatment response prediction in refractory epilepsy.

The system integrates clinical, genomic, EEG, neuroimaging, and medical literature data to generate personalized, evidence-based therapeutic insights. It combines machine learning, knowledge graphs, explainability methods, and retrieval-augmented generation with large language models.

This project is intended for research, education, experimentation, and technical demonstration in precision medicine and AI-assisted clinical decision support.

## Key Features

* **Multimodal Data Integration:** combines clinical, genomic, EEG, neuroimaging, and literature-based data.
* **Medical Knowledge Graph:** models relationships between genes, drugs, symptoms, phenotypes, biomarkers, and clinical outcomes.
* **Predictive Modeling:** uses ensemble learning with XGBoost, LightGBM, CatBoost, and neural network models.
* **Explainability:** provides SHAP values and feature importance analysis to support model interpretation.
* **RAG with LLMs:** retrieves scientific evidence and generates context-aware therapeutic recommendations.
* **Interactive Dashboard:** enables visualization, analysis, and exploration of patient-level predictions.
* **MLOps Pipeline:** supports model versioning, monitoring, testing, and CI/CD practices.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        Ingestion Layer                      │
│        CSV, FHIR, DICOM, VCF, and PubMed Literature         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                       Processing Layer                      │
│       ETL Pipeline, Feature Engineering, Validation,         │
│                    and Data Normalization                    │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌─────────▼────────┐
│ Knowledge Graph │    │   Feature Store  │
│     Neo4j       │    │      Feast       │
└────────┬────────┘    └─────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                         ML/AI Layer                         │
│      Ensemble Models, GNNs, Transformers, SHAP, AutoML,     │
│                         and RAG System                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                      Application Layer                      │
│              FastAPI REST API, Web Dashboard,               │
│                    and Analytical Reports                   │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Data and Machine Learning

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* PyTorch Geometric
* SHAP

### Data Storage and Knowledge Representation

* PostgreSQL
* Neo4j
* Feast Feature Store

### API and Application

* FastAPI
* Streamlit or web dashboard interface
* Pydantic
* Uvicorn

### MLOps and DevOps

* Docker
* Docker Compose
* Alembic
* CI/CD workflows
* Model monitoring
* Experiment tracking

## Prerequisites

Before running the project, make sure you have:

* Python 3.10 or higher
* Docker and Docker Compose
* Neo4j 5.x
* PostgreSQL 15 or higher
* At least 16 GB RAM
* GPU support recommended for deep learning experiments

## Installation

Clone the repository:

```bash
git clone https://github.com/nathadriele/neuroPredict-precision-medicine-system.git
cd neuroPredict-precision-medicine-system
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -e ".[dev]"
```

Create the environment configuration file:

```bash
cp .env.example .env
```

Start the required services:

```bash
docker-compose up -d
```

Run database migrations:

```bash
alembic upgrade head
```

Load sample data:

```bash
python scripts/load_sample_data.py
```

## Usage

### 1. Model Training

Run model training using the default configuration:

```bash
python -m neuropredict.train --config configs/training_config.yaml
```

Run model training with hyperparameter optimization:

```bash
python -m neuropredict.train --config configs/training_config.yaml --hpo
```

### 2. REST API

Start the API locally:

```bash
uvicorn neuropredict.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation:

```text
http://localhost:8000/docs
```

### 3. Prediction Workflow

The prediction workflow receives patient-level multimodal features and returns treatment response probabilities, model explanations, and recommendation support.

<img width="954" height="668" alt="Prediction workflow" src="https://github.com/user-attachments/assets/45a0b3e7-9a8a-4fe3-b117-5badfa8041bd" />

### 4. Web Dashboard

The dashboard provides an interactive interface for exploring predictions, patient profiles, model explanations, and treatment recommendation outputs.

<img width="726" height="178" alt="Web dashboard" src="https://github.com/user-attachments/assets/925003b1-e3a1-44e5-af22-4bc93b32b073" />

<img width="1980" height="1079" alt="image" src="https://github.com/user-attachments/assets/c79063a2-dcc6-49ad-8a11-b2438e62a115" />

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=neuropredict --cov-report=term-missing
```

Example testing output:

<img width="718" height="317" alt="Testing results" src="https://github.com/user-attachments/assets/55625255-a250-4c04-beba-e4935a128e39" />

## Performance

The system supports model evaluation using classification metrics, calibration metrics, explainability reports, and validation workflows.

Example performance report:

<img width="719" height="406" alt="Performance results" src="https://github.com/user-attachments/assets/f01c13f0-8a21-4b7f-a31a-81934e1491b1" />

## Project Structure

```text
neuroPredict-precision-medicine-system/
├── configs/
│   └── training_config.yaml
├── data/
├── docs/
├── neuropredict/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── graph/
│   ├── models/
│   ├── rag/
│   └── training/
├── scripts/
│   └── load_sample_data.py
├── tests/
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Roadmap

Planned improvements include:

* Integration with real-world clinical and genomic datasets.
* Extended support for FHIR-compatible data ingestion.
* Advanced graph neural network models.
* Improved RAG pipeline with biomedical literature retrieval.
* Bias and fairness evaluation across demographic groups.
* Model monitoring and drift detection.
* Deployment-ready CI/CD pipeline.
* Clinical validation with independent cohorts.

## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

3. Commit your changes:

```bash
git commit -m "Add your feature description"
```

4. Push your branch:

```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request.

## Important Notes

This system is intended for research and educational purposes only.

It must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek guidance from a qualified healthcare professional for medical decisions.

## Citation

If you use this project in research or academic work, please cite:

```bibtex
@software{neuropredict2026,
  author = {Nathalia Adriele},
  title = {NeuroPredict: Precision Medicine System for Refractory Epilepsy},
  year = {2026},
  url = {https://github.com/nathadriele/neuroPredict-precision-medicine-system}
}
```

## License

This project is distributed under the MIT License.
