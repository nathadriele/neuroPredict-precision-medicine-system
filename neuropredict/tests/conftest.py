"""
Configuração de fixtures para testes pytest.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuropredict.config import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings para testes."""
    return Settings(
        debug=True,
        database={"host": "localhost", "port": 5432, "name": "test_db"},
        log={"level": "DEBUG"},
    )


@pytest.fixture
def sample_clinical_data() -> pd.DataFrame:
    """Dados clínicos de exemplo."""
    return pd.DataFrame({
        "patient_id": [f"PAT{i:03d}" for i in range(100)],
        "age": np.random.randint(18, 80, 100),
        "sex": np.random.choice(["M", "F"], 100),
        "seizure_type": np.random.choice(
            ["focal_aware", "focal_impaired_awareness", "generalized_tonic_clonic"],
            100,
        ),
        "seizure_frequency_per_month": np.random.uniform(0.5, 20, 100),
        "age_at_onset": np.random.randint(5, 40, 100),
        "epilepsy_duration_years": np.random.uniform(1, 30, 100),
        "previous_treatments": ["levetiracetam;lamotrigine"] * 100,
        "treatment_response": np.random.choice(
            ["responder", "non_responder", "partial_responder"],
            100,
        ),
    })


@pytest.fixture
def sample_genetic_data() -> pd.DataFrame:
    """Dados genéticos de exemplo."""
    genes = ["SCN1A", "SCN2A", "KCNQ2", "KCNQ3", "GABRA1"]
    
    data = []
    for i in range(50):
        patient_id = f"PAT{i:03d}"
        for _ in range(np.random.randint(0, 3)):
            data.append({
                "patient_id": patient_id,
                "gene": np.random.choice(genes),
                "variant": f"p.R{np.random.randint(100, 2000)}H",
                "variant_type": np.random.choice(
                    ["missense", "nonsense", "frameshift"]
                ),
                "allele_frequency": np.random.uniform(0, 0.1),
                "clinvar_significance": np.random.choice(
                    ["Pathogenic", "Likely pathogenic", "Uncertain significance"]
                ),
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_ml_data() -> tuple:
    """Dados para ML de exemplo."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """Caminho temporário para salvar modelos."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir / "test_model.pkl"


@pytest.fixture(scope="session")
def docker_services():
    """
    Garante que serviços Docker estão rodando.
    Requer docker-compose.
    """
    import subprocess
    
    # Check if running in CI
    if os.getenv("CI"):
        pytest.skip("Pulando testes que requerem Docker em CI")
    
    # Start services
    subprocess.run(["docker-compose", "up", "-d", "postgres", "neo4j", "redis"])
    
    yield
    
    # Cleanup (opcional)
    # subprocess.run(["docker-compose", "down"])


@pytest.fixture
def mock_api_client():
    """Mock client para API."""
    from unittest.mock import Mock
    
    client = Mock()
    client.post.return_value.json.return_value = {
        "patient_id": "PAT001",
        "predicted_treatment": "levetiracetam",
        "response_probability": 0.85,
        "confidence": 0.90,
    }
    
    return client


@pytest.fixture
def patient_data_dict() -> dict:
    """Dicionário de dados de paciente."""
    return {
        "patient_id": "PAT001",
        "age": 35,
        "sex": "M",
        "seizure_type": "focal_impaired_awareness",
        "seizure_frequency_per_month": 4.5,
        "age_at_onset": 12,
        "epilepsy_duration_years": 23.0,
        "previous_treatments": ["levetiracetam", "lamotrigine"],
        "genetic_variants": [
            {"gene": "SCN1A", "variant": "p.R1648H", "variant_type": "missense"}
        ],
    }


# Markers
def pytest_configure(config):
    """Configura markers customizados."""
    config.addinivalue_line(
        "markers", "slow: marca testes que são lentos"
    )
    config.addinivalue_line(
        "markers", "integration: marca testes de integração"
    )
    config.addinivalue_line(
        "markers", "unit: marca testes unitários"
    )
    config.addinivalue_line(
        "markers", "requires_docker: marca testes que requerem Docker"
    )


# Hooks
def pytest_collection_modifyitems(config, items):
    """Modifica itens coletados."""
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)