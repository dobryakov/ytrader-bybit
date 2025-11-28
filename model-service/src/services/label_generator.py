"""
Label generation service.

Extracts labels from execution events for model training.
Supports binary classification, multi-class, and regression targets.
"""

from typing import List, Literal, Optional
import pandas as pd
import numpy as np

from ..models.execution_event import OrderExecutionEvent
from ..config.logging import get_logger

logger = get_logger(__name__)


class LabelGenerator:
    """Generates labels from execution events for model training."""

    def __init__(self):
        """Initialize label generator."""
        pass

    def generate_labels(
        self,
        execution_events: List[OrderExecutionEvent],
        label_type: Literal["binary", "multi_class", "regression"] = "binary",
        profit_threshold: float = 0.0,
    ) -> pd.Series:
        """
        Generate labels from execution events.

        Args:
            execution_events: List of order execution events
            label_type: Type of labels to generate ('binary', 'multi_class', 'regression')
            profit_threshold: Minimum profit to consider a trade profitable (for binary classification)

        Returns:
            Series with labels (one per execution event)
        """
        if not execution_events:
            logger.warning("No execution events provided for label generation")
            return pd.Series(dtype=float)

        labels = []

        for event in execution_events:
            if label_type == "binary":
                label = self._generate_binary_label(event, profit_threshold)
            elif label_type == "multi_class":
                label = self._generate_multiclass_label(event)
            elif label_type == "regression":
                label = self._generate_regression_label(event)
            else:
                raise ValueError(f"Unknown label type: {label_type}")

            labels.append(label)

        labels_series = pd.Series(labels, name="label")
        logger.info(
            "Generated labels",
            event_count=len(execution_events),
            label_type=label_type,
            label_stats=labels_series.describe().to_dict(),
        )
        return labels_series

    def _generate_binary_label(self, event: OrderExecutionEvent, profit_threshold: float = 0.0) -> int:
        """
        Generate binary classification label (profitable vs unprofitable).

        Args:
            event: Order execution event
            profit_threshold: Minimum profit to consider a trade profitable

        Returns:
            1 if profitable, 0 if unprofitable
        """
        # Use realized_pnl if available, otherwise calculate from return_percent
        if event.performance.realized_pnl is not None:
            is_profitable = event.performance.realized_pnl > profit_threshold
        elif event.performance.return_percent is not None:
            is_profitable = event.performance.return_percent > profit_threshold
        else:
            # Fallback: use slippage as proxy (negative slippage is good for buys, positive for sells)
            if event.side == "buy":
                is_profitable = event.performance.slippage_percent < 0  # Negative slippage = better execution
            else:  # sell
                is_profitable = event.performance.slippage_percent > 0  # Positive slippage = better execution

        return 1 if is_profitable else 0

    def _generate_multiclass_label(self, event: OrderExecutionEvent) -> int:
        """
        Generate multi-class classification label (buy signal, sell signal, hold).

        Args:
            event: Order execution event

        Returns:
            0 for buy, 1 for sell, 2 for hold (if applicable)
        """
        # For now, we'll use the side as the class
        # In a more sophisticated system, we might classify based on performance
        if event.side == "buy":
            return 0
        elif event.side == "sell":
            return 1
        else:
            return 2  # hold (shouldn't happen with current data model)

    def _generate_regression_label(self, event: OrderExecutionEvent) -> float:
        """
        Generate regression target (expected return, risk-adjusted return).

        Args:
            event: Order execution event

        Returns:
            Regression target value (return percentage or risk-adjusted return)
        """
        # Use return_percent if available, otherwise calculate from realized_pnl
        if event.performance.return_percent is not None:
            return event.performance.return_percent
        elif event.performance.realized_pnl is not None:
            # Calculate return percentage from PnL
            # PnL / (execution_price * execution_quantity) * 100
            if event.execution_price > 0 and event.execution_quantity > 0:
                return (event.performance.realized_pnl / (event.execution_price * event.execution_quantity)) * 100
            else:
                return 0.0
        else:
            # Fallback: use negative slippage as proxy for return
            # Negative slippage is better, so we'll use -slippage_percent as return proxy
            return -event.performance.slippage_percent

    def generate_risk_adjusted_labels(
        self, execution_events: List[OrderExecutionEvent], risk_free_rate: float = 0.0
    ) -> pd.Series:
        """
        Generate risk-adjusted return labels (Sharpe ratio-like metric).

        Args:
            execution_events: List of order execution events
            risk_free_rate: Risk-free rate for Sharpe ratio calculation

        Returns:
            Series with risk-adjusted return labels
        """
        if not execution_events:
            return pd.Series(dtype=float)

        labels = []

        for event in execution_events:
            # Calculate return
            return_pct = self._generate_regression_label(event)

            # Calculate risk (volatility at execution time)
            risk = event.market_conditions.volatility

            # Risk-adjusted return = (return - risk_free_rate) / risk
            # Avoid division by zero
            if risk > 0:
                risk_adjusted_return = (return_pct - risk_free_rate) / risk
            else:
                risk_adjusted_return = return_pct - risk_free_rate

            labels.append(risk_adjusted_return)

        labels_series = pd.Series(labels, name="risk_adjusted_return")
        logger.info("Generated risk-adjusted labels", event_count=len(execution_events))
        return labels_series


# Global label generator instance
label_generator = LabelGenerator()

