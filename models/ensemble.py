from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import torch
import torch.nn as nn
import xgboost as xgb
from loguru import logger
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
        }

    def __repr__(self) -> str:
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
    roc_auc = 0.0

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            roc_auc = 0.0

    return ModelMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        roc_auc=roc_auc,
    )


class BaseModel(ABC):
    def __init__(self, **kwargs: Any) -> None:
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
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        return compute_metrics(
            y_true=y,
            y_pred=self.predict(X),
            y_proba=self.predict_proba(X),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "eval_metric": "mlogloss",
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
        eval_set = [(X, y)]

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X, y, eval_set=eval_set, verbose=False)

        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
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
        callbacks = [lgb.log_evaluation(0)]

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(10))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
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
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = cb.Pool(X_val, y_val)

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set is not None else None,
        )

        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class NeuralNetModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        n_classes: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.n_classes = n_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        previous_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, self.n_classes))

        return nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NeuralNetModel":
        X_scaled = self.scaler.fit_transform(X)

        X_train = torch.FloatTensor(X_scaled).to(self.device)
        y_train = torch.LongTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

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
                    f"Epoch {epoch + 1}/{self.epochs}, loss: {avg_loss:.4f}"
                )

        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        method: str = "voting",
        weights: Optional[List[float]] = None,
    ) -> None:
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
        if not self.models:
            raise ValueError("At least one model must be provided.")

        for index, model in enumerate(self.models, start=1):
            self.logger.info(f"Training model {index}/{len(self.models)}")
            model.fit(X, y, X_val, y_val)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.method == "voting":
            return self._voting_predict(X)

        if self.method == "weighted_voting":
            return self._weighted_voting_predict(X)

        raise ValueError(f"Unsupported ensemble method: {self.method}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("The ensemble has no models.")

        all_probabilities = np.array([model.predict_proba(X) for model in self.models])

        if self.method == "weighted_voting" and self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)

            if len(weights) != len(self.models):
                raise ValueError("The number of weights must match the number of models.")

            return np.average(all_probabilities, axis=0, weights=weights)

        return np.mean(all_probabilities, axis=0)

    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        votes = stats.mode(predictions, axis=0, keepdims=False)

        return votes.mode

    def _weighted_voting_predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)

        return np.argmax(probabilities, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        return compute_metrics(
            y_true=y,
            y_pred=self.predict(X),
            y_proba=self.predict_proba(X),
        )


class HyperparameterOptimizer:
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
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.timeout = timeout
        self.logger = logger.bind(component="HyperparameterOptimizer")

    def objective(self, trial: optuna.Trial) -> float:
        params = self._get_trial_params(trial)

        model = self.model_class(**params)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        metrics = model.evaluate(self.X_val, self.y_val)

        return metrics.f1_score

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.model_class == XGBoostModel:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

        if self.model_class == LightGBMModel:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            }

        return {}

    def optimize(self) -> Dict[str, Any]:
        study = optuna.create_study(direction="maximize")

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        self.logger.info(f"Best F1-score: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")

        return study.best_params
