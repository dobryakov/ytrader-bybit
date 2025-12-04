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
        # Map task_type to dictionary key: "classification" -> "classifier", "regression" -> "regressor"
        model_key = "classifier" if task_type == "classification" else "regressor"
        model_class = model_classes.get(model_key)
        if model_class is None:
            raise ValueError(f"Model type {model_type} does not support {task_type}")

        # Prepare data
        X = dataset.features
        y = dataset.labels

        # Handle missing values
        X = X.fillna(X.mean())  # Fill with mean for numeric columns

        # Check for label diversity (critical for XGBoost with logistic loss)
        unique_labels = y.unique()
        
        # Calculate class weights for balancing (for classification tasks)
        if task_type == "classification" and model_type == "xgboost" and len(unique_labels) > 1:
            # Calculate class distribution
            label_counts = y.value_counts()
            # Calculate scale_pos_weight: ratio of negative class to positive class
            # For binary classification: 0 (negative) vs 1 (positive)
            if 0 in label_counts.index and 1 in label_counts.index:
                negative_count = label_counts[0]
                positive_count = label_counts[1]
                if positive_count > 0:
                    scale_pos_weight = negative_count / positive_count
                    logger.info(
                        "Calculated class balance for XGBoost",
                        negative_count=int(negative_count),
                        positive_count=int(positive_count),
                        scale_pos_weight=float(scale_pos_weight),
                    )
                    # Add to hyperparameters if not already set (will be merged later)
                    if hyperparameters is None:
                        hyperparameters = {}
                    if "scale_pos_weight" not in hyperparameters:
                        hyperparameters["scale_pos_weight"] = scale_pos_weight
        if len(unique_labels) == 1:
            # All labels are the same - XGBoost will fail with logistic loss
            # For XGBoost, we need to handle this case
            if model_type == "xgboost" and task_type == "classification":
                logger.warning(
                    "All labels are identical - cannot train XGBoost classifier with logistic loss",
                    unique_label=unique_labels[0],
                    dataset_size=len(y),
                )
                # Set base_score explicitly to avoid XGBoost error
                # base_score should be in (0,1) for logistic loss
                # If all labels are 0, set base_score to 0.1 (low probability)
                # If all labels are 1, set base_score to 0.9 (high probability)
                if unique_labels[0] == 0:
                    base_score = 0.1
                elif unique_labels[0] == 1:
                    base_score = 0.9
                else:
                    # For other values, normalize to (0,1)
                    base_score = max(0.1, min(0.9, float(unique_labels[0])))
                
                logger.info(
                    "Setting base_score for XGBoost to handle identical labels",
                    base_score=base_score,
                )
            else:
                # For other models, log warning but proceed
                logger.warning(
                    "All labels are identical - model may not learn effectively",
                    unique_label=unique_labels[0],
                    dataset_size=len(y),
                )

        # Prepare hyperparameters
        if hyperparameters is None:
            hyperparameters = {}

        # Set default hyperparameters based on model type
        default_hyperparameters = self._get_default_hyperparameters(model_type, task_type)
        hyperparameters = {**default_hyperparameters, **hyperparameters}

        # Add base_score for XGBoost if all labels are identical
        if model_type == "xgboost" and task_type == "classification" and len(unique_labels) == 1:
            if "base_score" not in hyperparameters:
                if unique_labels[0] == 0:
                    hyperparameters["base_score"] = 0.1
                elif unique_labels[0] == 1:
                    hyperparameters["base_score"] = 0.9
                else:
                    hyperparameters["base_score"] = max(0.1, min(0.9, float(unique_labels[0])))

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

