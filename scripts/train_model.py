"""
Script de treinamento completo do modelo NeuroPredict.
Implementa pipeline end-to-end: ETL → Feature Engineering → Training → Evaluation.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

from neuropredict.config import get_settings
from neuropredict.data.pipeline import IntegratedDataPipeline
from neuropredict.models.ensemble import (
    CatBoostModel,
    EnsembleModel,
    HyperparameterOptimizer,
    LightGBMModel,
    NeuralNetModel,
    XGBoostModel,
)


def setup_logging(log_dir: Path) -> None:
    """Configura sistema de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    
    logger.info("Logging configurado")


def load_and_prepare_data(
    data_sources: Dict[str, Path],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega e prepara dados.
    
    Args:
        data_sources: Dicionário com caminhos dos dados
        
    Returns:
        Tuple (features, target)
    """
    logger.info("Iniciando pipeline de dados...")
    
    pipeline = IntegratedDataPipeline()
    
    # Processa todos os tipos de dados
    results = pipeline.process_all(data_sources)
    
    # Merge de dados
    merged_data = pipeline.merge_data(results)
    
    logger.info(f"Dados carregados: {merged_data.shape}")
    
    # Separa features e target
    target_col = "treatment_response_encoded"
    
    if target_col not in merged_data.columns:
        raise ValueError(f"Coluna target {target_col} não encontrada")
    
    # Remove colunas não-features
    feature_cols = [
        col for col in merged_data.columns
        if col not in [
            "patient_id",
            "treatment_response",
            target_col,
            "previous_treatments",
        ]
    ]
    
    X = merged_data[feature_cols]
    y = merged_data[target_col]
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribuição: {y.value_counts().to_dict()}")
    
    return X, y


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Otimiza hiperparâmetros de cada modelo.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação
        n_trials: Número de trials Optuna
        
    Returns:
        Dicionário com melhores parâmetros por modelo
    """
    logger.info("Iniciando otimização de hiperparâmetros...")
    
    best_params = {}
    
    models_to_optimize = [
        ("XGBoost", XGBoostModel),
        ("LightGBM", LightGBMModel),
    ]
    
    for model_name, model_class in models_to_optimize:
        logger.info(f"Otimizando {model_name}...")
        
        optimizer = HyperparameterOptimizer(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
        )
        
        params = optimizer.optimize()
        best_params[model_name] = params
        
        logger.info(f"{model_name} melhores params: {params}")
    
    return best_params


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict[str, Dict[str, Any]],
) -> EnsembleModel:
    """
    Treina ensemble de modelos.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação
        best_params: Melhores hiperparâmetros
        
    Returns:
        EnsembleModel treinado
    """
    logger.info("Treinando ensemble de modelos...")
    
    # Cria modelos individuais
    models = [
        XGBoostModel(**best_params.get("XGBoost", {})),
        LightGBMModel(**best_params.get("LightGBM", {})),
        CatBoostModel(),
        NeuralNetModel(
            input_dim=X_train.shape[1],
            hidden_dims=[256, 128, 64],
            n_classes=len(np.unique(y_train)),
        ),
    ]
    
    # Cria ensemble
    ensemble = EnsembleModel(
        models=models,
        method="weighted_voting",
        weights=[0.3, 0.3, 0.2, 0.2],  # Pesos ajustáveis
    )
    
    # Treina
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    logger.info("Ensemble treinado com sucesso!")
    
    return ensemble


def evaluate_model(
    model: EnsembleModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Avalia modelo no conjunto de teste.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        Dicionário com métricas
    """
    logger.info("Avaliando modelo no conjunto de teste...")
    
    metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"Métricas de teste: {metrics}")
    
    return metrics.to_dict()


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict[str, Dict[str, Any]],
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Validação cruzada estratificada.
    
    Args:
        X: Features
        y: Target
        best_params: Melhores hiperparâmetros
        n_folds: Número de folds
        
    Returns:
        Métricas médias
    """
    logger.info(f"Iniciando validação cruzada com {n_folds} folds...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Treinando fold {fold}/{n_folds}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Treina ensemble
        ensemble = train_ensemble(
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            best_params,
        )
        
        # Avalia
        metrics = evaluate_model(ensemble, X_val_fold, y_val_fold)
        
        # Armazena métricas
        for metric, value in metrics.items():
            cv_metrics[metric].append(value)
    
    # Calcula médias
    avg_metrics = {
        metric: float(np.mean(values))
        for metric, values in cv_metrics.items()
    }
    
    logger.info(f"Métricas CV médias: {avg_metrics}")
    
    return avg_metrics


def save_model(
    model: EnsembleModel,
    model_dir: Path,
    version: str,
) -> Path:
    """
    Salva modelo treinado.
    
    Args:
        model: Modelo a salvar
        model_dir: Diretório de modelos
        version: Versão do modelo
        
    Returns:
        Caminho do arquivo salvo
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"ensemble_model_{version}.pkl"
    model.save(model_path)
    
    logger.info(f"Modelo salvo em {model_path}")
    
    return model_path


def track_with_mlflow(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Dict[str, Path],
) -> None:
    """
    Registra experimento no MLFlow.
    
    Args:
        experiment_name: Nome do experimento
        params: Parâmetros do modelo
        metrics: Métricas
        artifacts: Artifacts a registrar
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parâmetros
        mlflow.log_params(params)
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        for artifact_name, artifact_path in artifacts.items():
            mlflow.log_artifact(str(artifact_path), artifact_name)
        
        logger.info("Experimento registrado no MLFlow")


def main(args: argparse.Namespace) -> None:
    """
    Função principal de treinamento.
    
    Args:
        args: Argumentos de linha de comando
    """
    # Configuração
    settings = get_settings()
    setup_logging(Path("logs"))
    
    logger.info("=" * 80)
    logger.info("INICIANDO TREINAMENTO DO NEUROPREDICT")
    logger.info("=" * 80)
    
    # Carrega dados
    data_sources = {
        "clinical": Path(args.clinical_data),
        "genetic": Path(args.genetic_data) if args.genetic_data else None,
        "neuroimaging": Path(args.neuroimaging_data) if args.neuroimaging_data else None,
    }
    
    # Remove fontes None
    data_sources = {k: v for k, v in data_sources.items() if v is not None}
    
    X, y = load_and_prepare_data(data_sources)
    
    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values,
        y.values,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Otimização de hiperparâmetros (se solicitado)
    if args.hpo:
        best_params = optimize_hyperparameters(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=args.hpo_trials,
        )
    else:
        best_params = {}
    
    # Validação cruzada (se solicitado)
    if args.cv:
        cv_metrics = cross_validate(
            X_train,
            y_train,
            best_params,
            n_folds=args.cv_folds,
        )
    
    # Treina modelo final
    final_model = train_ensemble(X_train, y_train, X_val, y_val, best_params)
    
    # Avalia
    test_metrics = evaluate_model(final_model, X_test, y_test)
    
    # Salva modelo
    version = f"v{datetime.now():%Y%m%d_%H%M%S}"
    model_path = save_model(final_model, Path("models"), version)
    
    # MLFlow tracking
    if args.mlflow:
        track_with_mlflow(
            experiment_name="neuropredict_training",
            params={
                "version": version,
                "n_features": X_train.shape[1],
                "n_samples": X_train.shape[0],
                **best_params,
            },
            metrics=test_metrics,
            artifacts={
                "model": model_path,
            },
        )
    
    # Salva metadados
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "n_features": int(X_train.shape[1]),
        "n_samples": int(X_train.shape[0]),
        "best_params": best_params,
        "test_metrics": test_metrics,
    }
    
    metadata_path = Path("models") / f"metadata_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info(f"Modelo salvo em: {model_path}")
    logger.info(f"Métricas de teste: {test_metrics}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina modelo NeuroPredict"
    )
    
    parser.add_argument(
        "--clinical-data",
        type=str,
        required=True,
        help="Caminho dos dados clínicos",
    )
    
    parser.add_argument(
        "--genetic-data",
        type=str,
        help="Caminho dos dados genéticos",
    )
    
    parser.add_argument(
        "--neuroimaging-data",
        type=str,
        help="Caminho dos dados de neuroimagem",
    )
    
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Executar otimização de hiperparâmetros",
    )
    
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=100,
        help="Número de trials para HPO",
    )
    
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Executar validação cruzada",
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Número de folds para CV",
    )
    
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Registrar no MLFlow",
    )
    
    args = parser.parse_args()
    
    main(args)