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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier, XGBRegressor
from itertools import product

# Optional import for SMOTE - handle import errors gracefully
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    SMOTE_AVAILABLE = False
    SMOTE = None

from ..models.training_dataset import TrainingDataset
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains ML models from training datasets."""

    def __init__(self):
        """Initialize model trainer."""
        self._label_mapping_for_inference = None  # Store label mapping for inference
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

        # Handle missing values - only for numeric columns
        # Exclude non-numeric columns (like 'symbol', 'timestamp') from mean calculation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        # For non-numeric columns, fill with forward fill or drop if needed
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            # Drop non-numeric columns that are not needed for training (like 'symbol', 'timestamp')
            # These should be excluded from features before training
            X = X.drop(columns=non_numeric_cols, errors='ignore')

        # Check for label diversity (critical for XGBoost with logistic loss)
        unique_labels = y.unique()
        
        # Normalize class labels for XGBoost (must start from 0 and be consecutive)
        # XGBoost requires classes to be [0, 1, 2, ...], but we might have [-1, 0, 1]
        label_mapping = None
        if task_type == "classification" and model_type == "xgboost":
            sorted_unique = sorted(unique_labels)
            original_label_dist = y.value_counts().to_dict()
            original_label_dist_pct = {k: (v / len(y) * 100) for k, v in original_label_dist.items()}
            
            if sorted_unique[0] < 0 or sorted_unique != list(range(len(sorted_unique))):
                # Need to remap labels to [0, 1, 2, ...]
                # Convert numpy types to Python native types for JSON serialization
                sorted_unique_py = [int(x) if isinstance(x, (np.integer, np.int64, np.int32)) else float(x) if isinstance(x, (np.floating, np.float64, np.float32)) else x for x in sorted_unique]
                label_mapping = {int(old_label) if isinstance(old_label, (np.integer, np.int64, np.int32)) else old_label: new_label for new_label, old_label in enumerate(sorted_unique)}
                reverse_mapping = {new_label: int(old_label) if isinstance(old_label, (np.integer, np.int64, np.int32)) else old_label for old_label, new_label in label_mapping.items()}
                y = y.map(label_mapping)
                # Update unique_labels after remapping to reflect new label values
                unique_labels = sorted(y.unique())
                
                # Log distribution after remapping
                remapped_label_dist = y.value_counts().to_dict()
                remapped_label_dist_pct = {k: (v / len(y) * 100) for k, v in remapped_label_dist.items()}
                
                logger.info(
                    "Remapped class labels for XGBoost",
                    original_labels=sorted_unique_py,
                    mapped_labels=list(range(len(sorted_unique))),
                    mapping={str(k): v for k, v in label_mapping.items()},  # Convert keys to strings for JSON
                    original_distribution=original_label_dist,
                    original_distribution_percentage={k: round(v, 2) for k, v in original_label_dist_pct.items()},
                    remapped_distribution=remapped_label_dist,
                    remapped_distribution_percentage={k: round(v, 2) for k, v in remapped_label_dist_pct.items()},
                )
                # Store reverse mapping separately (not in hyperparameters - XGBoost doesn't accept it)
                # We'll store it in model metadata after training
                self._label_mapping_for_inference = {int(k): int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v for k, v in reverse_mapping.items()}
            else:
                logger.info(
                    "Class labels already normalized for XGBoost",
                    labels=unique_labels.tolist(),
                    distribution=original_label_dist,
                    distribution_percentage={k: round(v, 2) for k, v in original_label_dist_pct.items()},
                )
        
        # Apply SMOTE oversampling for minority classes if enabled
        original_class_distribution = None
        if (
            task_type == "classification"
            and settings.model_training_use_smote
            and SMOTE_AVAILABLE
            and len(unique_labels) > 1
        ):
            # Log original class distribution
            original_label_counts = y.value_counts()
            original_class_distribution = {
                int(k): int(v) for k, v in original_label_counts.items()
            }
            original_total = len(y)
            
            # Check if we have enough samples for SMOTE (requires at least k neighbors, default k=5)
            min_class_count = original_label_counts.min()
            if min_class_count < 5:
                logger.warning(
                    "Skipping SMOTE: insufficient samples for minority classes",
                    min_class_count=int(min_class_count),
                    required_min_samples=5,
                    note="SMOTE requires at least k=5 neighbors for each minority class",
                )
            else:
                try:
                    # Apply SMOTE to balance classes
                    # SMOTE generates synthetic samples for minority classes
                    smote = SMOTE(random_state=42, k_neighbors=5)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # Convert back to pandas DataFrame/Series
                    X = pd.DataFrame(X_resampled, columns=X.columns, index=pd.RangeIndex(len(X_resampled)))
                    y = pd.Series(y_resampled, name=y.name)
                    
                    # Log new class distribution
                    new_label_counts = y.value_counts()
                    new_class_distribution = {
                        int(k): int(v) for k, v in new_label_counts.items()
                    }
                    new_total = len(y)
                    synthetic_samples = new_total - original_total
                    
                    logger.info(
                        "Applied SMOTE oversampling for class balancing",
                        original_class_distribution=original_class_distribution,
                        new_class_distribution=new_class_distribution,
                        original_total_samples=original_total,
                        new_total_samples=new_total,
                        synthetic_samples_generated=synthetic_samples,
                        smote_parameters={"k_neighbors": 5, "random_state": 42},
                        note="SMOTE increases training time and memory usage for large datasets",
                    )
                    
                    # Update unique_labels after SMOTE
                    unique_labels = sorted(y.unique())
                except Exception as e:
                    logger.warning(
                        "SMOTE application failed, continuing without SMOTE",
                        error=str(e),
                        exc_info=True,
                    )
        
        # Calculate class weights for balancing (for classification tasks)
        sample_weight = None
        if task_type == "classification" and model_type == "xgboost" and len(unique_labels) > 1:
            # Calculate class distribution (use current y values after remapping)
            label_counts = y.value_counts()
            
            # For binary classification: use scale_pos_weight
            if len(unique_labels) == 2 and 0 in label_counts.index and 1 in label_counts.index:
                negative_count = label_counts[0]
                positive_count = label_counts[1]
                if positive_count > 0:
                    scale_pos_weight = negative_count / positive_count
                    total_samples = len(y)
                    negative_pct = (negative_count / total_samples) * 100
                    positive_pct = (positive_count / total_samples) * 100
                    logger.info(
                        "Calculated class balance for binary XGBoost",
                        negative_count=int(negative_count),
                        positive_count=int(positive_count),
                        negative_percentage=round(negative_pct, 2),
                        positive_percentage=round(positive_pct, 2),
                        scale_pos_weight=float(scale_pos_weight),
                        formula="scale_pos_weight = negative_count / positive_count",
                        note="scale_pos_weight helps with class imbalance and improves recall for minority classes",
                    )
                    # Add to hyperparameters if not already set (will be merged later)
                    if hyperparameters is None:
                        hyperparameters = {}
                    if "scale_pos_weight" not in hyperparameters:
                        hyperparameters["scale_pos_weight"] = scale_pos_weight
            # For multi-class classification: use sample_weight
            elif len(unique_labels) > 2:
                total_samples = len(y)
                num_classes = len(unique_labels)
                class_distribution_pct = {}
                for label in unique_labels:
                    class_count = label_counts[label]
                    class_distribution_pct[label] = (class_count / total_samples) * 100
                
                # Get class weight method from configuration
                weight_method = settings.model_training_class_weight_method
                
                # Calculate class weights based on selected method
                if weight_method == "balanced":
                    # Use sklearn's balanced class weights
                    # Formula: n_samples / (n_classes * np.bincount(y))
                    sklearn_weights = compute_class_weight(
                        "balanced",
                        classes=np.array(sorted(unique_labels)),
                        y=y.values,
                    )
                    class_weights = {label: weight for label, weight in zip(sorted(unique_labels), sklearn_weights)}
                elif weight_method == "inverse_frequency":
                    # Original inverse frequency method (default)
                    # Formula: class_weight = total_samples / (number_of_classes * class_count)
                    class_weights = {}
                    for label in unique_labels:
                        class_count = label_counts[label]
                        class_weights[label] = total_samples / (num_classes * class_count)
                elif weight_method == "custom":
                    # Custom weights from configuration (not yet implemented)
                    # For now, fall back to inverse_frequency
                    logger.warning(
                        "Custom class weights not yet implemented, using inverse_frequency",
                        weight_method=weight_method,
                    )
                    class_weights = {}
                    for label in unique_labels:
                        class_count = label_counts[label]
                        class_weights[label] = total_samples / (num_classes * class_count)
                else:
                    # Fallback to inverse_frequency
                    logger.warning(
                        "Unknown class weight method, using inverse_frequency",
                        weight_method=weight_method,
                    )
                    class_weights = {}
                    for label in unique_labels:
                        class_count = label_counts[label]
                        class_weights[label] = total_samples / (num_classes * class_count)
                
                # Verify that minority classes receive higher weights than majority class
                max_class = max(class_distribution_pct.items(), key=lambda x: x[1])
                min_class = min(class_distribution_pct.items(), key=lambda x: x[1])
                max_weight = class_weights[max_class[0]]
                min_weight = class_weights[min_class[0]]
                
                # Calculate weight statistics for monitoring
                weight_values = list(class_weights.values())
                min_weight_val = min(weight_values)
                max_weight_val = max(weight_values)
                mean_weight_val = sum(weight_values) / len(weight_values)
                
                # Create sample weights array
                sample_weight = y.map(class_weights).values
                
                logger.info(
                    "Calculated sample weights for multi-class XGBoost",
                    weight_method=weight_method,
                    class_distribution={int(k): int(v) for k, v in label_counts.items()},
                    class_distribution_percentage={int(k): round(v, 2) for k, v in class_distribution_pct.items()},
                    class_weights={int(k): round(float(v), 4) for k, v in class_weights.items()},
                    weight_statistics={
                        "min": round(float(min_weight_val), 4),
                        "max": round(float(max_weight_val), 4),
                        "mean": round(float(mean_weight_val), 4),
                    },
                    majority_class={"label": int(max_class[0]), "percentage": round(max_class[1], 2), "weight": round(float(max_weight), 4)},
                    minority_class={"label": int(min_class[0]), "percentage": round(min_class[1], 2), "weight": round(float(min_weight), 4)},
                    weight_ratio=round(float(min_weight / max_weight), 2) if max_weight > 0 else None,
                    note="sample_weight helps with multi-class imbalance (current use case: 3 classes - flat, up, down)",
                )
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

        # Remove _label_mapping from hyperparameters before training (XGBoost doesn't accept it)
        # Store it separately for inference
        label_mapping_for_inference = hyperparameters.pop("_label_mapping", None)
        if label_mapping_for_inference is not None:
            self._label_mapping_for_inference = label_mapping_for_inference

        # Perform hyperparameter tuning if enabled
        if settings.model_training_hyperparameter_tuning and model_type == "xgboost":
            # Remove _label_mapping from base_hyperparameters before optimization
            base_hyperparameters = {k: v for k, v in hyperparameters.items() if k != "_label_mapping"}
            hyperparameters = self._optimize_hyperparameters(
                X=X,
                y=y,
                model_type=model_type,
                task_type=task_type,
                base_hyperparameters=base_hyperparameters,
                sample_weight=sample_weight,
            )
            # Ensure _label_mapping is not in optimized hyperparameters
            hyperparameters.pop("_label_mapping", None)

        # Create and train model
        try:
            # Remove _label_mapping if it somehow got back in
            hyperparameters.pop("_label_mapping", None)
            model = model_class(**hyperparameters)
            # Use sample_weight for multi-class balancing if calculated
            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
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
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "eval_metric": "mlogloss",
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

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        task_type: Literal["classification", "regression"],
        base_hyperparameters: Dict[str, Any],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Grid Search or Bayesian Optimization.

        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model
            task_type: Type of task ('classification' or 'regression')
            base_hyperparameters: Base hyperparameters to start from
            sample_weight: Optional sample weights for training

        Returns:
            Dictionary of optimized hyperparameters
        """
        from datetime import datetime

        tuning_start_time = datetime.utcnow()
        tuning_method = settings.model_training_tuning_method
        max_iterations = settings.model_training_tuning_max_iterations

        logger.info(
            "Starting hyperparameter optimization",
            tuning_method=tuning_method,
            max_iterations=max_iterations,
            model_type=model_type,
            task_type=task_type,
        )

        if tuning_method == "grid_search":
            # Grid Search implementation
            # Define parameter grid for XGBoost
            param_grid = {
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 200, 500],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1, 3, 5],
            }

            # Calculate total combinations
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)

            logger.info(
                "Hyperparameter grid search configuration",
                total_combinations=total_combinations,
                max_iterations=max_iterations,
                parameter_grid=param_grid,
            )

            # Limit combinations if exceeds max_iterations
            if total_combinations > max_iterations:
                logger.warning(
                    "Parameter grid exceeds max_iterations, limiting search space",
                    total_combinations=total_combinations,
                    max_iterations=max_iterations,
                )
                # Reduce grid size by taking fewer values
                param_grid = {
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "min_child_weight": [1, 3],
                }
                # Recalculate total combinations after reduction
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                logger.info(
                    "Reduced parameter grid",
                    new_total_combinations=total_combinations,
                    reduced_grid=param_grid,
                )

            # Split data for cross-validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
            )

            # Prepare sample weights for validation split if provided
            train_sample_weight = None
            val_sample_weight = None
            if sample_weight is not None:
                train_indices = X_train.index
                val_indices = X_val.index
                train_sample_weight = sample_weight[X.index.isin(train_indices)]
                val_sample_weight = sample_weight[X.index.isin(val_indices)]

            # Get model class
            model_classes = self.supported_model_types[model_type]
            model_key = "classifier" if task_type == "classification" else "regressor"
            model_class = model_classes.get(model_key)

            # Use F1-score for classification, R2 for regression
            scoring = "f1_macro" if task_type == "classification" else "r2"

            # Perform grid search with cross-validation
            best_score = -np.inf
            best_params = base_hyperparameters.copy()
            iteration = 0

            # Generate parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Use total_combinations calculated above (already includes reduction if needed)
            # Limit to max_iterations for progress tracking
            effective_total = min(total_combinations, max_iterations)
            
            # Log progress every 10% or every 10 iterations, whichever is more frequent
            progress_log_interval = max(1, min(10, effective_total // 10))
            last_logged_iteration = -1
            
            logger.info(
                "Starting grid search iterations",
                effective_total_combinations=effective_total,
                progress_log_interval=progress_log_interval,
            )

            for param_combo in product(*param_values):
                if iteration >= max_iterations:
                    logger.warning(
                        "Reached max_iterations limit, stopping search",
                        iteration=iteration,
                        max_iterations=max_iterations,
                    )
                    break

                # Log progress periodically
                if iteration % progress_log_interval == 0 or iteration == last_logged_iteration + 1:
                    progress_pct = (iteration / effective_total * 100) if effective_total > 0 else 0
                    estimated_remaining = None
                    if iteration > 0:
                        elapsed = (datetime.utcnow() - tuning_start_time).total_seconds()
                        avg_time_per_iter = elapsed / iteration
                        remaining_iterations = effective_total - iteration
                        estimated_remaining = round(avg_time_per_iter * remaining_iterations, 1)
                    
                    logger.info(
                        "Hyperparameter optimization progress",
                        iteration=iteration,
                        total_combinations=effective_total,
                        progress_percent=round(progress_pct, 1),
                        current_best_score=round(float(best_score), 4) if best_score > -np.inf else None,
                        elapsed_seconds=round((datetime.utcnow() - tuning_start_time).total_seconds(), 1),
                        estimated_remaining_seconds=estimated_remaining,
                    )
                    last_logged_iteration = iteration

                # Create parameter dict
                params = dict(zip(param_names, param_combo))
                # Merge with base hyperparameters (ensure _label_mapping is not included)
                test_params = {k: v for k, v in {**base_hyperparameters, **params}.items() if k != "_label_mapping"}

                # Create and train model
                model = model_class(**test_params)
                if train_sample_weight is not None:
                    model.fit(X_train, y_train, sample_weight=train_sample_weight)
                else:
                    model.fit(X_train, y_train)

                # Evaluate on validation set
                if task_type == "classification":
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average="macro")
                else:
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, y_pred)

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    logger.debug(
                        "New best parameters found",
                        iteration=iteration,
                        score=round(float(score), 4),
                        parameters=test_params,
                    )

                iteration += 1

            tuning_duration = (datetime.utcnow() - tuning_start_time).total_seconds()

            # Remove _label_mapping from best_params before returning (XGBoost doesn't accept it)
            best_params_clean = {k: v for k, v in best_params.items() if k != "_label_mapping"}

            logger.info(
                "Hyperparameter optimization completed",
                tuning_method=tuning_method,
                iterations_completed=iteration,
                best_score=round(float(best_score), 4),
                best_parameters=best_params_clean,
                search_duration_seconds=round(tuning_duration, 2),
            )

            return best_params_clean

        elif tuning_method == "bayesian":
            # Bayesian optimization placeholder
            # Requires scikit-optimize or optuna library
            logger.warning(
                "Bayesian optimization not yet implemented, using base hyperparameters",
                tuning_method=tuning_method,
                note="Bayesian optimization requires scikit-optimize>=0.9.0 or optuna>=3.0.0",
            )
            return base_hyperparameters

        else:
            logger.warning(
                "Unknown tuning method, using base hyperparameters",
                tuning_method=tuning_method,
            )
            return base_hyperparameters


# Global model trainer instance
model_trainer = ModelTrainer()

