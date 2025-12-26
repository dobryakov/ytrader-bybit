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
    average_precision_score,
    precision_recall_curve,
    balanced_accuracy_score,
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
        # Get all unique classes from both true and predicted labels
        true_unique = y_true.unique().tolist()
        pred_unique = pd.Series(y_pred).unique().tolist()
        unique_classes = sorted(list(set(true_unique + pred_unique)))
        
        try:
            # Always pass labels parameter to prevent sklearn warnings
            # This ensures confusion matrix has correct shape even when some classes are missing
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes if len(unique_classes) > 0 else None)
            
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
            # Balanced accuracy (average of recall per class)
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            
            # Per-class metrics for binary classification (and expose them in metrics)
            # Use same unique_classes as defined earlier for confusion matrix
            if len(unique_classes) == 2:
                try:
                    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
                    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                    # Map per-class metrics into metrics dict with explicit labels
                    # unique_classes is sorted list of label values (e.g. [-1, 1] for candle color)
                    for idx, class_label in enumerate(unique_classes):
                        if idx < len(precision_per_class):
                            metrics[f"precision_class_{class_label}"] = float(precision_per_class[idx])
                        if idx < len(recall_per_class):
                            metrics[f"recall_class_{class_label}"] = float(recall_per_class[idx])
                        if idx < len(f1_per_class):
                            metrics[f"f1_class_{class_label}"] = float(f1_per_class[idx])
                    
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
            logger.warning("Failed to calculate precision/recall/f1/balanced_accuracy", error=str(e))
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1_score"] = 0.0
            metrics["balanced_accuracy"] = 0.0

        # ROC AUC and PR-AUC (for binary and multi-class classification with probabilities)
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

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
                        # ROC-AUC
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba_array))
                        # PR-AUC (average precision)
                        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_proba_array))
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
                            pos_proba = y_pred_proba_array[:, 1]
                            metrics["roc_auc"] = float(roc_auc_score(y_true_array, pos_proba))
                            # PR-AUC (average precision) for positive class
                            metrics["pr_auc"] = float(average_precision_score(y_true_array, pos_proba))
                        else:
                            logger.warning(
                                "ROC AUC not calculated: binary classification but wrong number of probability columns",
                                expected_columns=2,
                                actual_columns=y_pred_proba_array.shape[1],
                            )
                            metrics["roc_auc"] = 0.0
                    elif unique_classes > 2:
                        # Multi-class classification: use one-vs-rest (ovr) or one-vs-one (ovo)
                        # Use one-vs-rest averaging with weighted average for ROC-AUC.
                        # Note: y_true should contain remapped labels [0, 1, 2, ...] after XGBoost remapping.
                        # If some classes are missing in y_true, sklearn will handle it automatically.
                        y_true_array = np.array(y_true)
                        
                        # Check if probability array has correct number of columns
                        # XGBoost should output probabilities for all classes [0, 1, 2, ...] even if some are missing in y_true
                        if y_pred_proba_array.shape[1] >= unique_classes:
                            metrics["roc_auc"] = float(
                                roc_auc_score(
                                    y_true_array,
                                    y_pred_proba_array,
                                    multi_class="ovr",
                                    average="weighted",
                                )
                            )
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

    def calibrate_prediction_thresholds(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        target_recall: float = 0.5,
        optimization_metric: str = "f1",
    ) -> Dict[int, float]:
        """
        Calibrate prediction thresholds for each class using specified optimization metric.
        
        Supports multiple optimization strategies:
        - 'f1': Maximize F1-score (default, recommended for imbalanced datasets)
        - 'pr_auc': Maximize PR-AUC (area under precision-recall curve)
        - 'balanced_accuracy': Maximize balanced accuracy
        - 'recall': Achieve target recall (legacy method, fallback)
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array: n_samples, n_classes)
            target_recall: Target recall to achieve (used only for 'recall' method)
            optimization_metric: Metric to optimize ('f1', 'pr_auc', 'balanced_accuracy', 'recall')
            
        Returns:
            Dictionary mapping class label to optimal threshold
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
        
        optimization_metric = optimization_metric.lower()
        
        # Try new optimization methods first, fallback to legacy method
        try:
            if optimization_metric == "f1":
                thresholds = self._optimize_thresholds_by_f1(y_true, y_pred_proba, unique_labels)
            elif optimization_metric == "pr_auc":
                thresholds = self._optimize_thresholds_by_pr_auc(y_true, y_pred_proba, unique_labels)
            elif optimization_metric == "balanced_accuracy":
                thresholds = self._optimize_thresholds_by_balanced_accuracy(y_true, y_pred_proba, unique_labels)
            elif optimization_metric == "recall":
                # Legacy method: use target_recall
                thresholds = self._calibrate_by_target_recall(y_true, y_pred_proba, unique_labels, target_recall)
            else:
                logger.warning(
                    "Unknown optimization metric, falling back to target_recall method",
                    optimization_metric=optimization_metric,
                )
                thresholds = self._calibrate_by_target_recall(y_true, y_pred_proba, unique_labels, target_recall)
        except Exception as e:
            logger.warning(
                "Threshold optimization failed, falling back to target_recall method",
                optimization_metric=optimization_metric,
                error=str(e),
                exc_info=True,
            )
            thresholds = self._calibrate_by_target_recall(y_true, y_pred_proba, unique_labels, target_recall)
        
        return thresholds
    
    def _optimize_thresholds_by_f1(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        unique_labels: List[int],
    ) -> Dict[int, float]:
        """
        Optimize thresholds by maximizing F1-score for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array)
            unique_labels: List of unique class labels
            
        Returns:
            Dictionary mapping class label to optimal threshold
        """
        thresholds = {}
        
        for class_idx, class_label in enumerate(unique_labels):
            # Create binary labels: 1 for this class, 0 for others
            y_binary = (y_true == class_label).astype(int)
            
            # Get probabilities for this class
            class_probs = y_pred_proba[:, class_idx]
            
            # Calculate precision-recall curve
            try:
                precision, recall, threshold_candidates = precision_recall_curve(y_binary, class_probs)
                
                # Calculate F1-score for each threshold
                f1_scores = []
                for i in range(len(threshold_candidates)):
                    y_pred_binary = (class_probs >= threshold_candidates[i]).astype(int)
                    f1 = f1_score(y_binary, y_pred_binary, zero_division=0)
                    f1_scores.append(f1)
                
                # Find threshold with maximum F1-score
                if f1_scores:
                    best_idx = np.argmax(f1_scores)
                    optimal_threshold = float(threshold_candidates[best_idx])
                    best_f1 = float(f1_scores[best_idx])
                    
                    thresholds[class_label] = optimal_threshold
                    
                    logger.info(
                        "Threshold optimized by F1-score for class",
                        class_label=class_label,
                        class_idx=class_idx,
                        optimal_threshold=optimal_threshold,
                        best_f1_score=best_f1,
                    )
                else:
                    # Fallback: use median threshold
                    optimal_threshold = float(np.median(threshold_candidates)) if len(threshold_candidates) > 0 else 0.5
                    thresholds[class_label] = optimal_threshold
                    logger.warning(
                        "No F1 scores calculated, using median threshold",
                        class_label=class_label,
                        threshold=optimal_threshold,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to optimize threshold by F1-score for class",
                    class_label=class_label,
                    class_idx=class_idx,
                    error=str(e),
                )
                # Fallback: use default threshold
                class_freq = (y_true == class_label).sum() / len(y_true)
                default_threshold = max(0.1, min(0.5, class_freq * 2))
                thresholds[class_label] = default_threshold
        
        return thresholds
    
    def _optimize_thresholds_by_pr_auc(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        unique_labels: List[int],
    ) -> Dict[int, float]:
        """
        Optimize thresholds by maximizing PR-AUC for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array)
            unique_labels: List of unique class labels
            
        Returns:
            Dictionary mapping class label to optimal threshold
        """
        thresholds = {}
        
        for class_idx, class_label in enumerate(unique_labels):
            # Create binary labels: 1 for this class, 0 for others
            y_binary = (y_true == class_label).astype(int)
            
            # Get probabilities for this class
            class_probs = y_pred_proba[:, class_idx]
            
            try:
                precision, recall, threshold_candidates = precision_recall_curve(y_binary, class_probs)
                
                # Calculate PR-AUC for each threshold (approximate by integrating)
                # We'll use the threshold that maximizes the area under the precision-recall curve
                # by finding the point with best precision-recall trade-off
                best_idx = 0
                best_score = 0.0
                
                for i in range(len(threshold_candidates)):
                    # Calculate approximate PR-AUC up to this threshold
                    # Use precision * recall as a proxy for PR-AUC contribution
                    score = precision[i] * recall[i] if i < len(precision) and i < len(recall) else 0.0
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                optimal_threshold = float(threshold_candidates[best_idx])
                thresholds[class_label] = optimal_threshold
                
                logger.info(
                    "Threshold optimized by PR-AUC proxy for class",
                    class_label=class_label,
                    class_idx=class_idx,
                    optimal_threshold=optimal_threshold,
                    best_score=best_score,
                )
            except Exception as e:
                logger.warning(
                    "Failed to optimize threshold by PR-AUC for class",
                    class_label=class_label,
                    class_idx=class_idx,
                    error=str(e),
                )
                # Fallback: use default threshold
                class_freq = (y_true == class_label).sum() / len(y_true)
                default_threshold = max(0.1, min(0.5, class_freq * 2))
                thresholds[class_label] = default_threshold
        
        return thresholds
    
    def _optimize_thresholds_by_balanced_accuracy(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        unique_labels: List[int],
    ) -> Dict[int, float]:
        """
        Optimize thresholds by maximizing balanced accuracy for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array)
            unique_labels: List of unique class labels
            
        Returns:
            Dictionary mapping class label to optimal threshold
        """
        thresholds = {}
        
        for class_idx, class_label in enumerate(unique_labels):
            # Create binary labels: 1 for this class, 0 for others
            y_binary = (y_true == class_label).astype(int)
            
            # Get probabilities for this class
            class_probs = y_pred_proba[:, class_idx]
            
            try:
                fpr, tpr, threshold_candidates = roc_curve(y_binary, class_probs)
                
                # Calculate balanced accuracy for each threshold
                balanced_acc_scores = []
                for i in range(len(threshold_candidates)):
                    y_pred_binary = (class_probs >= threshold_candidates[i]).astype(int)
                    balanced_acc = balanced_accuracy_score(y_binary, y_pred_binary)
                    balanced_acc_scores.append(balanced_acc)
                
                # Find threshold with maximum balanced accuracy
                if balanced_acc_scores:
                    best_idx = np.argmax(balanced_acc_scores)
                    optimal_threshold = float(threshold_candidates[best_idx])
                    best_balanced_acc = float(balanced_acc_scores[best_idx])
                    
                    thresholds[class_label] = optimal_threshold
                    
                    logger.info(
                        "Threshold optimized by balanced accuracy for class",
                        class_label=class_label,
                        class_idx=class_idx,
                        optimal_threshold=optimal_threshold,
                        best_balanced_accuracy=best_balanced_acc,
                    )
                else:
                    # Fallback: use median threshold
                    optimal_threshold = float(np.median(threshold_candidates)) if len(threshold_candidates) > 0 else 0.5
                    thresholds[class_label] = optimal_threshold
            except Exception as e:
                logger.warning(
                    "Failed to optimize threshold by balanced accuracy for class",
                    class_label=class_label,
                    class_idx=class_idx,
                    error=str(e),
                )
                # Fallback: use default threshold
                class_freq = (y_true == class_label).sum() / len(y_true)
                default_threshold = max(0.1, min(0.5, class_freq * 2))
                thresholds[class_label] = default_threshold
        
        return thresholds
    
    def _calibrate_by_target_recall(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        unique_labels: List[int],
        target_recall: float = 0.5,
    ) -> Dict[int, float]:
        """
        Legacy method: Calibrate thresholds to achieve target recall for each class.
        
        This is the original implementation, kept as fallback.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array)
            unique_labels: List of unique class labels
            target_recall: Target recall to achieve
            
        Returns:
            Dictionary mapping class label to optimal threshold
        """
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

    def calculate_baseline_metrics(self, y_true: pd.Series) -> Dict[str, float]:
        """
        Calculate baseline metrics for majority class strategy.
        
        Baseline strategy: always predict the majority class.
        
        Args:
            y_true: True labels
            
        Returns:
            Dictionary of baseline metrics with 'baseline_' prefix
        """
        if len(y_true) == 0:
            logger.warning("Cannot calculate baseline metrics: empty y_true")
            return {}
        
        # Find majority class
        majority_class = y_true.value_counts().idxmax()
        majority_count = int(y_true.value_counts().max())
        total_count = len(y_true)
        majority_pct = (majority_count / total_count * 100) if total_count > 0 else 0.0
        
        logger.info(
            "Calculating baseline metrics (majority class strategy)",
            majority_class=int(majority_class) if isinstance(majority_class, (int, np.integer)) else majority_class,
            majority_count=majority_count,
            total_count=total_count,
            majority_percentage=round(majority_pct, 2),
        )
        
        # Create baseline predictions (always predict majority class)
        y_pred_baseline = pd.Series([majority_class] * len(y_true))
        
        # Calculate metrics using existing evaluation method
        baseline_metrics = self._evaluate_classification(y_true, y_pred_baseline, y_pred_proba=None)
        
        # Add 'baseline_' prefix to all metric names
        prefixed_metrics = {f"baseline_{k}": v for k, v in baseline_metrics.items()}
        
        logger.info(
            "Baseline metrics calculated",
            baseline_accuracy=prefixed_metrics.get("baseline_accuracy"),
            baseline_f1_score=prefixed_metrics.get("baseline_f1_score"),
            baseline_balanced_accuracy=prefixed_metrics.get("baseline_balanced_accuracy"),
        )
        
        return prefixed_metrics

    def analyze_top_k_performance(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        k_values: List[int] = [10, 20, 30, 50],
    ) -> Dict[str, Any]:
        """
        Analyze model performance for top-k% predictions without filters.
        
        Sorts predictions by confidence (max probability) and calculates metrics
        for top-k% samples without applying confidence threshold or hysteresis.
        Also calculates and returns the confidence threshold for each k value,
        which can be used in production to filter predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (2D array: n_samples, n_classes)
            k_values: List of k percentages to analyze (default: [10, 20, 30, 50])
            
        Returns:
            Dictionary with metrics for each k value, keys like:
            - 'top_k_{k}_accuracy': Accuracy for top-k% predictions
            - 'top_k_{k}_confidence_threshold': Minimum confidence threshold for top-k%
            - Other metrics: f1_score, precision, recall, coverage, etc.
        """
        if len(y_true) == 0:
            logger.warning("Cannot analyze top-k performance: empty y_true")
            return {}
        
        if y_pred_proba is None or y_pred_proba.ndim != 2:
            logger.warning(
                "Cannot analyze top-k performance: invalid y_pred_proba",
                shape=y_pred_proba.shape if y_pred_proba is not None else None,
            )
            return {}
        
        results = {}
        
        # Calculate confidence for each sample (max probability)
        confidence = np.max(y_pred_proba, axis=1)
        
        # Get predicted class (argmax)
        y_pred = pd.Series(np.argmax(y_pred_proba, axis=1))
        
        # Get unique classes for per-class metrics
        unique_classes = sorted(y_true.unique().tolist())
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence)[::-1]
        
        total_samples = len(y_true)
        
        logger.info(
            "Starting top-k performance analysis",
            total_samples=total_samples,
            k_values=k_values,
            unique_classes=unique_classes,
        )
        
        for k in k_values:
            if k <= 0 or k >= 100:
                logger.warning(f"Skipping invalid k value: {k} (must be between 1 and 99)")
                continue
            
            # Calculate number of samples in top-k%
            k_samples = max(1, int(total_samples * k / 100))
            
            # Get top-k% indices
            top_k_indices = sorted_indices[:k_samples]
            
            # Get y_true and y_pred for top-k%
            y_true_top_k = y_true.iloc[top_k_indices]
            y_pred_top_k = y_pred.iloc[top_k_indices]
            y_pred_proba_top_k = y_pred_proba[top_k_indices]
            
            # Calculate metrics for top-k%
            top_k_metrics = self._evaluate_classification(y_true_top_k, y_pred_top_k, y_pred_proba_top_k)
            
            # Add per-class metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # Calculate per-class metrics with all known classes from original dataset
                # This ensures we get metrics for all classes, even if they're not in top-k
                # Using labels parameter prevents sklearn warnings when some classes are missing
                if len(unique_classes) > 0:
                    precision_per_class = precision_score(
                        y_true_top_k, y_pred_top_k, average=None, zero_division=0, labels=unique_classes
                    )
                    recall_per_class = recall_score(
                        y_true_top_k, y_pred_top_k, average=None, zero_division=0, labels=unique_classes
                    )
                    f1_per_class = f1_score(
                        y_true_top_k, y_pred_top_k, average=None, zero_division=0, labels=unique_classes
                    )
                    
                    for idx, class_label in enumerate(unique_classes):
                        if idx < len(precision_per_class):
                            top_k_metrics[f"precision_class_{class_label}"] = float(precision_per_class[idx])
                        if idx < len(recall_per_class):
                            top_k_metrics[f"recall_class_{class_label}"] = float(recall_per_class[idx])
                        if idx < len(f1_per_class):
                            top_k_metrics[f"f1_class_{class_label}"] = float(f1_per_class[idx])
            except Exception as e:
                logger.warning(f"Failed to calculate per-class metrics for top-{k}%", error=str(e))
            
            # Add coverage (should be k/100)
            coverage = k_samples / total_samples if total_samples > 0 else 0.0
            top_k_metrics["coverage"] = coverage
            
            # Calculate confidence threshold for top-k%
            # Threshold is the minimum confidence among top-k% samples
            # This threshold can be used in production to filter predictions
            top_k_confidence_values = confidence[top_k_indices]
            confidence_threshold = float(np.min(top_k_confidence_values)) if len(top_k_confidence_values) > 0 else 0.0
            top_k_metrics["confidence_threshold"] = confidence_threshold
            
            # Add prefix to all metrics
            for metric_name, metric_value in top_k_metrics.items():
                results[f"top_k_{k}_{metric_name}"] = metric_value
            
            logger.info(
                f"Top-{k}% analysis completed",
                k=k,
                k_samples=k_samples,
                coverage=round(coverage, 4),
                accuracy=top_k_metrics.get("accuracy"),
                f1_score=top_k_metrics.get("f1_score"),
                confidence_threshold=round(confidence_threshold, 4),
            )
        
        return results


# Global quality evaluator instance
quality_evaluator = QualityEvaluator()

