"""
Metrics tracking for signal generation skipping.

Tracks counts of signals skipped due to open orders for observability.
"""

from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime
from threading import Lock

from ..config.logging import get_logger

logger = get_logger(__name__)


class SignalSkipMetrics:
    """Tracks metrics for signal generation skipping."""

    def __init__(self):
        """Initialize metrics tracker."""
        self._metrics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = Lock()
        self._last_reset = datetime.utcnow()

    def record_skip(
        self,
        asset: str,
        strategy_id: str,
        reason: str,
    ) -> None:
        """
        Record a signal skip event.

        Args:
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            reason: Reason for skipping (e.g., "open_order_exists", "opposite_order_exists")
        """
        with self._lock:
            key = f"{strategy_id}:{asset}"
            self._metrics[key][reason] += 1
            logger.debug(
                "Recorded signal skip",
                asset=asset,
                strategy_id=strategy_id,
                reason=reason,
            )

    def get_metrics(
        self,
        asset: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Get skip metrics.

        Args:
            asset: Optional asset filter
            strategy_id: Optional strategy filter

        Returns:
            Dictionary with metrics by asset/strategy and reason
        """
        with self._lock:
            result = {
                "total_skips": 0,
                "by_asset_strategy": {},
                "by_reason": defaultdict(int),
                "last_reset": self._last_reset.isoformat() + "Z",
            }

            for key, reasons in self._metrics.items():
                strat_asset = key.split(":", 1)
                if len(strat_asset) != 2:
                    continue
                strat, asset_name = strat_asset

                # Apply filters
                if strategy_id and strat != strategy_id:
                    continue
                if asset and asset_name != asset:
                    continue

                # Sum all reasons for this asset/strategy
                total_skips = sum(reasons.values())
                result["total_skips"] += total_skips

                # Add to by_asset_strategy
                result["by_asset_strategy"][key] = {
                    "strategy_id": strat,
                    "asset": asset_name,
                    "total_skips": total_skips,
                    "by_reason": dict(reasons),
                }

                # Add to by_reason
                for reason, count in reasons.items():
                    result["by_reason"][reason] += count

            # Convert defaultdict to dict for JSON serialization
            result["by_reason"] = dict(result["by_reason"])

            return result

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._last_reset = datetime.utcnow()
            logger.info("Reset signal skip metrics")


# Global metrics instance
signal_skip_metrics = SignalSkipMetrics()

