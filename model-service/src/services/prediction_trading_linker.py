"""
Prediction Trading Linker service.

Links predictions with actual trading PnL for model quality evaluation.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal

from ..database.repositories.prediction_target_repo import PredictionTargetRepository
from ..database.repositories.prediction_trading_results_repo import PredictionTradingResultsRepository
from ..config.logging import get_logger

logger = get_logger(__name__)


class PredictionTradingLinker:
    """Service for linking predictions with trading results."""

    def __init__(self):
        """Initialize prediction trading linker."""
        self.prediction_target_repo = PredictionTargetRepository()
        self.prediction_trading_results_repo = PredictionTradingResultsRepository()

    async def link_prediction_to_trading(
        self,
        signal_id: str,
        entry_signal_id: Optional[str] = None,
        entry_price: Optional[Decimal] = None,
        entry_timestamp: Optional[datetime] = None,
        position_size_at_entry: Optional[Decimal] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Link prediction to trading result when signal opens a position.

        Args:
            signal_id: Trading signal UUID
            entry_signal_id: Entry signal UUID (optional, defaults to signal_id)
            entry_price: Entry price (optional)
            entry_timestamp: Entry timestamp (optional)
            position_size_at_entry: Position size at entry (optional)

        Returns:
            Created prediction trading result record or None if prediction not found
        """
        try:
            # Get prediction target by signal_id
            prediction_target = await self.prediction_target_repo.get_by_signal_id(signal_id)
            if not prediction_target:
                logger.debug(
                    "No prediction target found for signal, skipping prediction trading result creation",
                    signal_id=signal_id,
                )
                return None

            # Check if trading result already exists
            # Convert asyncpg UUID to Python UUID via string
            prediction_target_id = prediction_target["id"]
            if not isinstance(prediction_target_id, UUID):
                prediction_target_id = UUID(str(prediction_target_id))
            existing_result = await self.prediction_trading_results_repo.get_by_prediction_target_id(
                prediction_target_id
            )
            if existing_result:
                logger.debug(
                    "Prediction trading result already exists, returning existing result",
                    signal_id=signal_id,
                    prediction_target_id=prediction_target["id"],
                    result_id=existing_result["id"],
                )
                return existing_result

            # Create new trading result
            logger.debug(
                "Creating prediction trading result",
                signal_id=signal_id,
                prediction_target_id=prediction_target["id"],
                entry_price=str(entry_price) if entry_price else None,
            )
            result = await self.prediction_trading_results_repo.create(
                prediction_target_id=prediction_target_id,
                signal_id=signal_id,
                entry_signal_id=entry_signal_id or signal_id,
                entry_price=entry_price,
                entry_timestamp=entry_timestamp,
                position_size_at_entry=position_size_at_entry,
            )

            logger.info(
                "Prediction linked to trading result successfully",
                signal_id=signal_id,
                prediction_target_id=prediction_target["id"],
                result_id=result["id"],
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to link prediction to trading",
                signal_id=signal_id,
                error=str(e),
                exc_info=True,
            )
            return None

    async def update_trading_result_on_order_fill(
        self,
        signal_id: str,
        order_id: UUID,
        execution_price: Decimal,
        execution_quantity: Decimal,
        realized_pnl_delta: Decimal,
        relationship_type: str,
    ) -> None:
        """
        Update trading result when order is filled.

        Args:
            signal_id: Trading signal UUID
            order_id: Order UUID
            execution_price: Execution price
            execution_quantity: Execution quantity
            realized_pnl_delta: Realized PnL delta
            relationship_type: Relationship type ('closed', 'decreased', etc.)
        """
        try:
            # Get prediction trading results for this signal
            results = await self.prediction_trading_results_repo.get_by_signal_id(signal_id)
            if not results:
                logger.debug(
                    "No prediction trading results found for signal",
                    signal_id=signal_id,
                )
                return

            # Update each open result
            for result in results:
                if result.get("is_closed"):
                    continue

                # Update realized PnL
                current_realized_pnl = Decimal(str(result.get("realized_pnl", 0)))
                new_realized_pnl = current_realized_pnl + realized_pnl_delta

                # Update total PnL (assuming unrealized stays the same for now)
                current_unrealized_pnl = Decimal(str(result.get("unrealized_pnl", 0)))
                new_total_pnl = new_realized_pnl + current_unrealized_pnl

                # Determine if position is closed
                is_closed = relationship_type == "closed"

                # Convert asyncpg UUID to Python UUID if needed
                result_id = result["id"]
                if not isinstance(result_id, UUID):
                    result_id = UUID(str(result_id))
                
                await self.prediction_trading_results_repo.update(
                    result_id=result_id,
                    realized_pnl=new_realized_pnl,
                    total_pnl=new_total_pnl,
                    exit_price=execution_price if is_closed else None,
                    exit_timestamp=datetime.utcnow() if is_closed else None,
                    is_closed=is_closed,
                )

                logger.info(
                    "Prediction trading result updated on order fill",
                    signal_id=signal_id,
                    result_id=result["id"],
                    relationship_type=relationship_type,
                    is_closed=is_closed,
                )

        except Exception as e:
            logger.error(
                "Failed to update trading result on order fill",
                signal_id=signal_id,
                order_id=str(order_id),
                error=str(e),
                exc_info=True,
            )

    async def compute_signal_pnl(
        self,
        signal_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute aggregated PnL for a signal.

        Args:
            signal_id: Trading signal UUID

        Returns:
            Aggregated PnL metrics or None if not found
        """
        return await self.prediction_trading_results_repo.aggregate_pnl_by_signal(signal_id)


# Global instance
prediction_trading_linker = PredictionTradingLinker()

