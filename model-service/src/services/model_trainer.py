"""
ML model trainer service.

Trains XGBoost and scikit-learn models, supports batch retraining,
handles model serialization using joblib for scikit-learn and XGBoost
native JSON format.
"""

from typing import Dict, Any, Optional, Literal
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier, XGBRegressor

from ..models.training_dataset import TrainingDataset
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains ML models from training datasets."""

    def __init__(self):
        """Initialize model trainer."""
        self.supported_model_types = {
            "xgboost": {"classifier": XGBClassifier, "regressor": XGBRegressor},
            "random_forest": {"classifier": RandomForestClassifier, "regressor": RandomForestRegressor},
            "logistic_regression": {"classifier": LogisticRegression, "regressor": None},
            "sgd_classifier": {"classifier": SGDClassifier, "regressor": None},
        }

    def train_model(
        self,
        dataset: TrainingDataset,
        model_type: Literal["xgboost", "random_forest", "logistic_regression", "sgd_classifier"],
        task_type: Literal["classification", "regression"] = "classification",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Train a model from a training dataset.

        Args:
            dataset: TrainingDataset with features and labels
            model_type: Type of model to train
            task_type: Type of task ('classification' or 'regression')
            hyperparameters: Optional hyperparameters for the model

        Returns:
            Trained model object

        Raises:
            ValueError: If model_type or task_type is not supported
        """
        logger.info(
            "Starting model training",
            model_type=model_type,
            task_type=task_type,
            dataset_size=dataset.get_record_count(),
            feature_count=len(dataset.get_feature_names()),
        )

        # Validate dataset
        dataset.validate_consistency()

        # Get model class
        if model_type not in self.supported_model_types:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_classes = self.supported_model_types[model_type]
        model_class = model_classes.get(task_type)
        if model_class is None:
            raise ValueError(f"Model type {model_type} does not support {task_type}")

        # Prepare data
        X = dataset.features
        y = dataset.labels

        # Handle missing values
        X = X.fillna(X.mean())  # Fill with mean for numeric columns

        # Prepare hyperparameters
        if hyperparameters is None:
            hyperparameters = {}

        # Set default hyperparameters based on model type
        default_hyperparameters = self._get_default_hyperparameters(model_type, task_type)
        hyperparameters = {**default_hyperparameters, **hyperparameters}

        # Create and train model
        try:
            model = model_class(**hyperparameters)
            model.fit(X, y)

            logger.info(
                "Model training completed",
                model_type=model_type,
                task_type=task_type,
                dataset_size=dataset.get_record_count(),
            )

            return model

        except Exception as e:
            logger.error("Model training failed", model_type=model_type, error=str(e), exc_info=True)
            raise

    def save_model(
        self,
        model: Any,
        model_type: str,
        file_path: str,
    ) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model object
            model_type: Type of model ('xgboost' or scikit-learn type)
            file_path: Path to save the model file

        Raises:
            ValueError: If model_type is not supported for saving
        """
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            if model_type == "xgboost":
                # Save XGBoost model in native JSON format
                model.save_model(file_path)
                logger.info("Saved XGBoost model", file_path=file_path, format="json")
            else:
                # Save scikit-learn models using joblib
                joblib.dump(model, file_path)
                logger.info("Saved scikit-learn model", file_path=file_path, format="joblib")

        except Exception as e:
            logger.error("Failed to save model", model_type=model_type, file_path=file_path, error=str(e), exc_info=True)
            raise

    def load_model(
        self,
        model_type: str,
        file_path: str,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_type: Type of model ('xgboost' or scikit-learn type)
            file_path: Path to the model file
            task_type: Type of task ('classification' or 'regression')

        Returns:
            Loaded model object

        Raises:
            ValueError: If model_type is not supported for loading
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            if model_type == "xgboost":
                # Load XGBoost model from JSON
                model_classes = self.supported_model_types[model_type]
                model_class = model_classes.get(task_type)
                if model_class is None:
                    raise ValueError(f"XGBoost does not support {task_type}")
                model = model_class()
                model.load_model(file_path)
                logger.info("Loaded XGBoost model", file_path=file_path, format="json")
            else:
                # Load scikit-learn models using joblib
                model = joblib.load(file_path)
                logger.info("Loaded scikit-learn model", file_path=file_path, format="joblib")

            return model

        except Exception as e:
            logger.error("Failed to load model", model_type=model_type, file_path=file_path, error=str(e), exc_info=True)
            raise

    def _get_default_hyperparameters(
        self, model_type: str, task_type: Literal["classification", "regression"]
    ) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model type.

        Args:
            model_type: Type of model
            task_type: Type of task

        Returns:
            Dictionary of default hyperparameters
        """
        defaults = {
            "xgboost": {
                "classification": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
            },
            "random_forest": {
                "classification": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
            "logistic_regression": {
                "classification": {
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
            "sgd_classifier": {
                "classification": {
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
        }

        return defaults.get(model_type, {}).get(task_type, {})


# Global model trainer instance
model_trainer = ModelTrainer()

