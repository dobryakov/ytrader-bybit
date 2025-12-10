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

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Precision, recall, F1 (handle binary and multi-class)
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception as e:
            logger.warning("Failed to calculate precision/recall/f1", error=str(e))
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1_score"] = 0.0

        # ROC AUC (for binary and multi-class classification with probabilities)
        if y_pred_proba is not None:
            try:
                unique_classes = len(y_true.unique())
                
                # Convert to numpy array if needed
                if isinstance(y_pred_proba, pd.Series):
                    y_pred_proba_array = y_pred_proba.values
                elif isinstance(y_pred_proba, np.ndarray):
                    y_pred_proba_array = y_pred_proba
                else:
                    y_pred_proba_array = np.array(y_pred_proba)
                
                # Check if it's 1D or 2D
                if y_pred_proba_array.ndim == 1:
                    # 1D array: binary classification probabilities
                    if unique_classes == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba_array))
                    else:
                        logger.warning("ROC AUC not calculated: 1D probabilities but more than 2 classes")
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
                            logger.warning("ROC AUC not calculated: binary classification but wrong number of probability columns")
                            metrics["roc_auc"] = 0.0
                    elif unique_classes > 2:
                        # Multi-class classification: use one-vs-rest (ovr) or one-vs-one (ovo)
                        # Use one-vs-rest averaging
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba_array, multi_class='ovr', average='weighted'))
                    else:
                        metrics["roc_auc"] = 0.0
                else:
                    logger.warning("ROC AUC not calculated: unexpected probability array shape")
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
quality_evaluator = QualityEvaluator()

