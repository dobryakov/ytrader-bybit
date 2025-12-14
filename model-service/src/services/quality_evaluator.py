"""
Model quality evaluator.

Calculates accuracy, precision, recall, f1_score, sharpe_ratio, profit_factor,
and other metrics for model evaluation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    confusion_matrix,
)

from ..config.logging import get_logger

logger = get_logger(__name__)


class QualityEvaluator:
    """Evaluates model quality using various metrics."""

    def __init__(self):
        """Initialize quality evaluator."""
        pass

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: Optional[pd.Series] = None,
        task_type: str = "classification",
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model quality using appropriate metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            task_type: Type of task ('classification' or 'regression')
            performance_metrics: Optional trading performance metrics (sharpe_ratio, profit_factor, etc.)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        if task_type == "classification":
            metrics.update(self._evaluate_classification(y_true, y_pred, y_pred_proba))
        elif task_type == "regression":
            metrics.update(self._evaluate_regression(y_true, y_pred))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Add trading performance metrics if provided
        if performance_metrics:
            metrics.update(performance_metrics)

        logger.info("Model quality evaluation completed", metric_count=len(metrics), task_type=task_type)
        return metrics

    def _evaluate_classification(
        self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of classification metrics
        """
        metrics = {}

        # Log class distribution for debugging
        true_dist = y_true.value_counts().to_dict()
        pred_dist = pd.Series(y_pred).value_counts().to_dict()
        true_dist_pct = {k: (v / len(y_true) * 100) for k, v in true_dist.items()}
        pred_dist_pct = {k: (v / len(y_pred) * 100) for k, v in pred_dist.items()}
        
        logger.info(
            "classification_evaluation_distribution",
            total_samples=len(y_true),
            true_class_distribution=true_dist,
            true_class_distribution_percentage={k: round(v, 2) for k, v in true_dist_pct.items()},
            pred_class_distribution=pred_dist,
            pred_class_distribution_percentage={k: round(v, 2) for k, v in pred_dist_pct.items()},
        )

        # Compute confusion matrix
        unique_classes = sorted(list(set(y_true.unique().tolist() + pd.Series(y_pred).unique().tolist())))
        try:
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
            
            # Convert confusion matrix to dictionary format for logging
            # Format: {f"{true_class}_{pred_class}": count}
            cm_dict = {}
            cm_dict_pct = {}
            total_samples = len(y_true)
            
            for i, true_class in enumerate(unique_classes):
                for j, pred_class in enumerate(unique_classes):
                    count = int(cm[i, j])
                    pct = (count / total_samples * 100) if total_samples > 0 else 0.0
                    key = f"true_{true_class}_pred_{pred_class}"
                    cm_dict[key] = count
                    cm_dict_pct[key] = round(pct, 2)
            
            # Also log row-normalized (percentage of true class predicted as each class)
            cm_row_pct = {}
            for i, true_class in enumerate(unique_classes):
                row_total = cm[i, :].sum()
                if row_total > 0:
                    for j, pred_class in enumerate(unique_classes):
                        count = int(cm[i, j])
                        pct = (count / row_total * 100) if row_total > 0 else 0.0
                        key = f"true_{true_class}_pred_{pred_class}"
                        cm_row_pct[key] = round(pct, 2)
                else:
                    # No samples of this true class
                    for j, pred_class in enumerate(unique_classes):
                        key = f"true_{true_class}_pred_{pred_class}"
                        cm_row_pct[key] = 0.0
            
            logger.info(
                "confusion_matrix",
                total_samples=total_samples,
                class_labels=unique_classes,
                matrix_counts=cm_dict,
                matrix_percentages_of_total=cm_dict_pct,
                matrix_percentages_of_true_class=cm_row_pct,
                note="matrix_counts shows absolute counts, matrix_percentages_of_total shows % of total samples, matrix_percentages_of_true_class shows % of each true class",
            )
        except Exception as e:
            logger.warning("Failed to calculate confusion matrix", error=str(e), exc_info=True)

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Precision, recall, F1 (handle binary and multi-class)
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            
            # Log per-class metrics for binary classification
            # Use same unique_classes as defined earlier for confusion matrix
            if len(unique_classes) == 2:
                try:
                    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
                    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                    
                    logger.info(
                        "classification_per_class_metrics",
                        class_0_precision=float(precision_per_class[0]) if len(precision_per_class) > 0 else None,
                        class_0_recall=float(recall_per_class[0]) if len(recall_per_class) > 0 else None,
                        class_0_f1=float(f1_per_class[0]) if len(f1_per_class) > 0 else None,
                        class_1_precision=float(precision_per_class[1]) if len(precision_per_class) > 1 else None,
                        class_1_recall=float(recall_per_class[1]) if len(recall_per_class) > 1 else None,
                        class_1_f1=float(f1_per_class[1]) if len(f1_per_class) > 1 else None,
                        class_labels=unique_classes,
                    )
                except Exception as e:
                    logger.debug("Failed to calculate per-class metrics", error=str(e))
        except Exception as e:
            logger.warning("Failed to calculate precision/recall/f1", error=str(e))
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1_score"] = 0.0

        # ROC AUC (for binary and multi-class classification with probabilities)
        if y_pred_proba is not None:
            try:
                unique_labels = sorted(y_true.unique().tolist())
                unique_classes = len(unique_labels)
                
                # Convert to numpy array if needed
                if isinstance(y_pred_proba, pd.Series):
                    y_pred_proba_array = y_pred_proba.values
                elif isinstance(y_pred_proba, np.ndarray):
                    y_pred_proba_array = y_pred_proba
                else:
                    y_pred_proba_array = np.array(y_pred_proba)
                
                # Log information for debugging
                logger.debug(
                    "Calculating ROC AUC",
                    unique_classes=unique_classes,
                    unique_labels=unique_labels,
                    y_pred_proba_shape=y_pred_proba_array.shape if hasattr(y_pred_proba_array, 'shape') else None,
                    y_pred_proba_ndim=y_pred_proba_array.ndim if hasattr(y_pred_proba_array, 'ndim') else None,
                )
                
                # Check if it's 1D or 2D
                if y_pred_proba_array.ndim == 1:
                    # 1D array: binary classification probabilities
                    if unique_classes == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba_array))
                    else:
                        logger.warning(
                            "ROC AUC not calculated: 1D probabilities but more than 2 classes",
                            unique_classes=unique_classes,
                            unique_labels=unique_labels,
                        )
                        metrics["roc_auc"] = 0.0
                elif y_pred_proba_array.ndim == 2:
                    # 2D array: probabilities for each class (n_samples, n_classes)
                    if unique_classes == 2:
                        # Binary classification: use probabilities for positive class (last column)
                        # Ensure labels are 0 and 1 for binary classification
                        y_true_array = np.array(y_true)
                        if y_pred_proba_array.shape[1] == 2:
                            # Two columns: probabilities for class 0 and class 1
                            # Use probabilities for positive class (class 1)
                            metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_pred_proba_array[:, 1]))
                        else:
                            logger.warning(
                                "ROC AUC not calculated: binary classification but wrong number of probability columns",
                                expected_columns=2,
                                actual_columns=y_pred_proba_array.shape[1],
                            )
                            metrics["roc_auc"] = 0.0
                    elif unique_classes > 2:
                        # Multi-class classification: use one-vs-rest (ovr) or one-vs-one (ovo)
                        # Use one-vs-rest averaging with weighted average
                        # Note: y_true should contain remapped labels [0, 1, 2, ...] after XGBoost remapping
                        # If some classes are missing in y_true, sklearn will handle it automatically
                        y_true_array = np.array(y_true)
                        
                        # Check if probability array has correct number of columns
                        # XGBoost should output probabilities for all classes [0, 1, 2, ...] even if some are missing in y_true
                        if y_pred_proba_array.shape[1] >= unique_classes:
                            metrics["roc_auc"] = float(roc_auc_score(
                                y_true_array,
                                y_pred_proba_array,
                                multi_class='ovr',
                                average='weighted',
                            ))
                        else:
                            logger.warning(
                                "ROC AUC not calculated: multi-class but probability array has fewer columns than unique classes",
                                unique_classes=unique_classes,
                                unique_labels=unique_labels,
                                probability_columns=y_pred_proba_array.shape[1],
                            )
                            metrics["roc_auc"] = 0.0
                    else:
                        logger.warning(
                            "ROC AUC not calculated: insufficient classes",
                            unique_classes=unique_classes,
                            unique_labels=unique_labels,
                        )
                        metrics["roc_auc"] = 0.0
                else:
                    logger.warning(
                        "ROC AUC not calculated: unexpected probability array shape",
                        ndim=y_pred_proba_array.ndim if hasattr(y_pred_proba_array, 'ndim') else None,
                    )
                    metrics["roc_auc"] = 0.0
            except Exception as e:
                logger.warning("Failed to calculate ROC AUC", error=str(e), exc_info=True)
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0

        return metrics

    def _evaluate_regression(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Evaluate regression model.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of regression metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))

        # R-squared
        try:
            metrics["r2_score"] = float(r2_score(y_true, y_pred))
        except Exception as e:
            logger.warning("Failed to calculate R2 score", error=str(e))
            metrics["r2_score"] = 0.0

        return metrics

    def calculate_trading_metrics(
        self,
        execution_events: List[Any],
        predictions: pd.Series,
        y_true: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics.

        **NOTE**: This method is deprecated for training pipeline. Training now uses only ML metrics
        (accuracy, precision, recall, F1, MSE, MAE, R2) from market data predictions.
        This method is kept for backtesting purposes only.

        Args:
            execution_events: List of execution events (with performance data)
            predictions: Model predictions
            y_true: True labels (optional, for comparison)

        Returns:
            Dictionary of trading performance metrics
        """
        if not execution_events:
            return {}

        metrics = {}

        # Calculate win rate
        profitable_trades = sum(
            1
            for event in execution_events
            if event.performance.realized_pnl is not None and event.performance.realized_pnl > 0
        )
        total_trades = len(execution_events)
        metrics["win_rate"] = float(profitable_trades / total_trades) if total_trades > 0 else 0.0

        # Calculate total PnL
        total_pnl = sum(
            event.performance.realized_pnl
            for event in execution_events
            if event.performance.realized_pnl is not None
        )
        metrics["total_pnl"] = float(total_pnl)

        # Calculate average PnL
        pnl_values = [
            event.performance.realized_pnl
            for event in execution_events
            if event.performance.realized_pnl is not None
        ]
        metrics["avg_pnl"] = float(np.mean(pnl_values)) if pnl_values else 0.0

        # Calculate Sharpe ratio
        returns = [
            event.performance.return_percent
            for event in execution_events
            if event.performance.return_percent is not None
        ]
        if returns and len(returns) > 1:
            returns_array = np.array(returns)
            if returns_array.std() > 0:
                metrics["sharpe_ratio"] = float(returns_array.mean() / returns_array.std())
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0

        # Calculate profit factor
        profits = [pnl for pnl in pnl_values if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_values if pnl < 0]
        total_profit = sum(profits) if profits else 0.0
        total_loss = sum(losses) if losses else 0.0
        metrics["profit_factor"] = float(total_profit / total_loss) if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnl_values) if pnl_values else np.array([0.0])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        metrics["max_drawdown"] = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        logger.info("Trading metrics calculated", metrics=metrics)
        return metrics


# Global quality evaluator instance
    def calibrate_prediction_thresholds(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        target_recall: float = 0.5,
    ) -> Dict[int, float]:
        """
        Calibrate prediction thresholds for each class to improve recall for minority classes.
        
        Uses ROC curve to find optimal thresholds that achieve target recall for each class.
        Lower thresholds improve recall but may decrease precision.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array: n_samples, n_classes)
            target_recall: Target recall to achieve (default: 0.5)
            
        Returns:
            Dictionary mapping class index to optimal threshold
        """
        if y_pred_proba.ndim != 2:
            logger.warning(
                "Cannot calibrate thresholds: probabilities must be 2D array",
                shape=y_pred_proba.shape if hasattr(y_pred_proba, 'shape') else None,
            )
            return {}
        
        unique_labels = sorted(y_true.unique().tolist())
        num_classes = len(unique_labels)
        
        if num_classes != y_pred_proba.shape[1]:
            logger.warning(
                "Cannot calibrate thresholds: number of classes doesn't match probability columns",
                num_classes=num_classes,
                prob_columns=y_pred_proba.shape[1],
            )
            return {}
        
        thresholds = {}
        
        # For each class, find threshold that achieves target recall
        for class_idx, class_label in enumerate(unique_labels):
            # Create binary labels: 1 for this class, 0 for others
            y_binary = (y_true == class_label).astype(int)
            
            # Get probabilities for this class
            class_probs = y_pred_proba[:, class_idx]
            
            # Calculate ROC curve
            try:
                fpr, tpr, threshold_candidates = roc_curve(y_binary, class_probs)
                
                # Find threshold that achieves target recall (TPR)
                # Find closest threshold to target recall
                recall_diff = np.abs(tpr - target_recall)
                best_idx = np.argmin(recall_diff)
                
                if best_idx < len(threshold_candidates):
                    optimal_threshold = float(threshold_candidates[best_idx])
                    actual_recall = float(tpr[best_idx])
                    
                    thresholds[class_label] = optimal_threshold
                    
                    logger.info(
                        "Threshold calibrated for class",
                        class_label=class_label,
                        class_idx=class_idx,
                        optimal_threshold=optimal_threshold,
                        target_recall=target_recall,
                        actual_recall=actual_recall,
                    )
                else:
                    # Fallback: use threshold that maximizes TPR - FPR (Youden's J statistic)
                    youden_j = tpr - fpr
                    best_idx = np.argmax(youden_j)
                    optimal_threshold = float(threshold_candidates[best_idx])
                    thresholds[class_label] = optimal_threshold
                    
                    logger.info(
                        "Threshold calibrated using Youden's J statistic",
                        class_label=class_label,
                        class_idx=class_idx,
                        optimal_threshold=optimal_threshold,
                        recall=float(tpr[best_idx]),
                        precision=float(1 - fpr[best_idx]) if best_idx < len(fpr) else None,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to calibrate threshold for class",
                    class_label=class_label,
                    class_idx=class_idx,
                    error=str(e),
                )
                # Use default threshold based on class distribution
                class_freq = (y_true == class_label).sum() / len(y_true)
                # Lower threshold for minority classes
                default_threshold = max(0.1, min(0.5, class_freq * 2))
                thresholds[class_label] = default_threshold
                logger.info(
                    "Using default threshold for class",
                    class_label=class_label,
                    default_threshold=default_threshold,
                    class_frequency=class_freq,
                )
        
        return thresholds


quality_evaluator = QualityEvaluator()

