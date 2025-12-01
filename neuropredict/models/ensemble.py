"""
Ensemble de modelos para predição de resposta a tratamento.
Integra XGBoost, LightGBM, CatBoost e Redes Neurais.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Métricas e Avaliação
# ============================================================================

@dataclass
class ModelMetrics:
    """Métricas de avaliação de modelo."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1: {self.f1_score:.4f}, "
            f"ROC-AUC: {self.roc_auc:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> ModelMetrics:
    """
    Computa métricas de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        y_proba: Probabilidades (para ROC-AUC)
        
    Returns:
        ModelMetrics
    """
    metrics = ModelMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        roc_auc=roc_auc_score(
            y_true,
            y_proba if y_proba is not None else y_pred,
            multi_class="ovr",
            average="weighted",
        ) if y_proba is not None else 0.0,
    )
    
    return metrics


# ============================================================================
# Interface Base
# ============================================================================

class BaseModel(ABC):
    """Interface base para modelos."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Inicializa modelo."""
        self.model: Any = None
        self.is_fitted = False
        self.feature_importance_: Optional[np.ndarray] = None
        self.logger = logger.bind(model=self.__class__.__name__)
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Treina modelo."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades."""
        pass
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ModelMetrics:
        """Avalia modelo."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return compute_metrics(y, y_pred, y_proba)
    
    def save(self, path: Path) -> None:
        """Salva modelo."""
        joblib.dump(self, path)
        self.logger.info(f"Modelo salvo em {path}")
    
    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Carrega modelo."""
        model = joblib.load(path)
        logger.info(f"Modelo carregado de {path}")
        return model


# ============================================================================
# Modelos Específicos
# ============================================================================

class XGBoostModel(BaseModel):
    """Wrapper para XGBoost."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Inicializa XGBoost."""
        super().__init__()
        
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "enable_categorical": True,
            **kwargs,
        }
        
        self.model = xgb.XGBClassifier(**self.params)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        """Treina XGBoost."""
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predições."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades."""
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """Wrapper para LightGBM."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Inicializa LightGBM."""
        super().__init__()
        
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "multiclass",
            "verbosity": -1,
            **kwargs,
        }
        
        self.model = lgb.LGBMClassifier(**self.params)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMModel":
        """Treina LightGBM."""
        callbacks = [lgb.early_stopping(10), lgb.log_evaluation(0)]
        
        self.model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            callbacks=callbacks,
        )
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predições."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades."""
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    """Wrapper para CatBoost."""
    
    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Inicializa CatBoost."""
        super().__init__()
        
        self.params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": "MultiClass",
            "verbose": False,
            **kwargs,
        }
        
        self.model = cb.CatBoostClassifier(**self.params)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CatBoostModel":
        """Treina CatBoost."""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = cb.Pool(X_val, y_val)
        
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=10,
        )
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predições."""
        return self.model.predict(X).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades."""
        return self.model.predict_proba(X)


class NeuralNetModel(BaseModel):
    """Rede Neural Feedforward."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        n_classes: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
    ) -> None:
        """Inicializa rede neural."""
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Constrói arquitetura
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.model = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.scaler = StandardScaler()
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NeuralNetModel":
        """Treina rede neural."""
        # Normaliza features
        X_scaled = self.scaler.fit_transform(X)
        
        # Converte para tensores
        X_train = torch.FloatTensor(X_scaled).to(self.device)
        y_train = torch.LongTensor(y).to(self.device)
        
        # DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        # Otimizador e loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Treinamento
        self.model.train()
        best_loss = float("inf")
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}"
                )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predições."""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades."""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()


# ============================================================================
# Ensemble
# ============================================================================

class EnsembleModel(BaseEstimator, ClassifierMixin):
    """Ensemble de múltiplos modelos."""
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        method: str = "voting",
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Inicializa ensemble.
        
        Args:
            models: Lista de modelos
            method: Método de ensemble (voting, weighted_voting, stacking)
            weights: Pesos para weighted voting
        """
        self.models = models or []
        self.method = method
        self.weights = weights
        self.logger = logger.bind(component="EnsembleModel")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EnsembleModel":
        """Treina todos os modelos do ensemble."""
        for i, model in enumerate(self.models):
            self.logger.info(f"Treinando modelo {i + 1}/{len(self.models)}")
            model.fit(X, y, X_val, y_val)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predições via ensemble."""
        if self.method == "voting":
            return self._voting_predict(X)
        elif self.method == "weighted_voting":
            return self._weighted_voting_predict(X)
        else:
            raise ValueError(f"Método {self.method} não suportado")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades via ensemble."""
        all_probas = np.array([model.predict_proba(X) for model in self.models])
        
        if self.method == "voting" or self.method == "weighted_voting":
            if self.weights:
                weights = np.array(self.weights).reshape(-1, 1, 1)
                return np.average(all_probas, axis=0, weights=weights.squeeze())
            else:
                return np.mean(all_probas, axis=0)
        
        return np.mean(all_probas, axis=0)
    
    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Voting simples."""
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Voto majoritário
        from scipy import stats
        votes = stats.mode(predictions, axis=0, keepdims=False)
        return votes.mode
    
    def _weighted_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Voting ponderado."""
        if not self.weights:
            return self._voting_predict(X)
        
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Avalia ensemble."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return compute_metrics(y, y_pred, y_proba)


# ============================================================================
# Hyperparameter Optimization
# ============================================================================

class HyperparameterOptimizer:
    """Otimização de hiperparâmetros com Optuna."""
    
    def __init__(
        self,
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        timeout: int = 3600,
    ) -> None:
        """Inicializa otimizador."""
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.timeout = timeout
        self.logger = logger.bind(component="HyperparameterOptimizer")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Função objetivo para Optuna."""
        # Define espaço de busca baseado no modelo
        if self.model_class == XGBoostModel:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        elif self.model_class == LightGBMModel:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            }
        else:
            params = {}
        
        # Treina modelo
        model = self.model_class(**params)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Avalia
        metrics = model.evaluate(self.X_val, self.y_val)
        
        return metrics.f1_score
    
    def optimize(self) -> Dict[str, Any]:
        """Executa otimização."""
        study = optuna.create_study(direction="maximize")
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        self.logger.info(f"Melhor F1-Score: {study.best_value:.4f}")
        self.logger.info(f"Melhores parâmetros: {study.best_params}")
        
        return study.best_params