"""
Prediction chain API.

Позволяет получить цепочку сущностей для конкретного signal_id:
- сигнал
- prediction_target
- prediction_trading_results
- последний execution_event

Полная цепочка ордеров/позиций может быть расширена позже.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from ..database.connection import db_pool
from ..database.repositories.trading_signal_repo import TradingSignalRepository
from ..database.repositories.prediction_target_repo import PredictionTargetRepository
from ..database.repositories.prediction_trading_results_repo import (
    PredictionTradingResultsRepository,
)
from ..database.repositories.execution_event_repo import ExecutionEventRepository
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["prediction-chain"])


@router.get("/{signal_id}/chain")
async def get_prediction_chain(signal_id: str) -> Dict[str, Any]:
    """
    Get prediction → target → trading result chain for a given signal_id.
    """
    signal_repo = TradingSignalRepository()
    target_repo = PredictionTargetRepository()
    trading_results_repo = PredictionTradingResultsRepository()
    execution_repo = ExecutionEventRepository()

    # Trading signal
    signal = await signal_repo.get_by_signal_id(signal_id)
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal {signal_id} not found",
        )

    # Prediction target (may be None)
    prediction_target: Optional[Dict[str, Any]] = await target_repo.get_by_signal_id(signal_id)

    # Trading result (linked via prediction_target_id, if present)
    trading_result: Optional[Dict[str, Any]] = None
    if prediction_target:
        try:
            trading_result = await trading_results_repo.get_by_prediction_target_id(
                UUID(str(prediction_target["id"]))
            )
        except Exception as e:
            logger.error(
                "Failed to load prediction trading result",
                signal_id=signal_id,
                prediction_target_id=str(prediction_target.get("id")),
                error=str(e),
                exc_info=True,
            )

    # Last execution event for this signal (if any)
    execution_event = await execution_repo.get_by_signal_id(signal_id)
    
    # Orders for this signal
    orders: List[Dict[str, Any]] = []
    positions: List[Dict[str, Any]] = []
    position_orders: List[Dict[str, Any]] = []

    try:
        pool = await db_pool.get_pool()
        async with pool.acquire() as conn:
            # Orders linked to this signal_id
            order_rows = await conn.fetch(
                """
                SELECT id, order_id, asset, side, status, quantity, filled_quantity,
                       average_price, fees, created_at, updated_at
                FROM orders
                WHERE signal_id = $1
                """,
                signal["signal_id"],
            )
            orders = [dict(row) for row in order_rows]

            # Position_orders and positions via join (if any orders exist)
            # Note: By the time this API is called, position_orders.order_id should already be filled
            # (order-manager updates position_orders.order_id after creating the order in DB)
            if orders:
                order_ids = [row["id"] for row in order_rows]
                po_rows = await conn.fetch(
                    """
                    SELECT po.*, p.asset, p.size, p.average_entry_price,
                           p.realized_pnl, p.unrealized_pnl
                    FROM position_orders po
                    JOIN positions p ON po.position_id = p.id
                    WHERE po.order_id = ANY($1::uuid[])
                    """,
                    order_ids,
                )
                position_orders = []
                seen_positions: Dict[UUID, Dict[str, Any]] = {}
                for row in po_rows:
                    d = dict(row)
                    position_orders.append(d)
                    pos_id = d["position_id"]
                    if pos_id not in seen_positions:
                        seen_positions[pos_id] = {
                            "id": d["position_id"],
                            "asset": d.get("asset"),
                            "size": d.get("size"),
                            "average_entry_price": d.get("average_entry_price"),
                            "realized_pnl": d.get("realized_pnl"),
                            "unrealized_pnl": d.get("unrealized_pnl"),
                        }
                positions = list(seen_positions.values())
    except Exception as e:
        logger.error(
            "Failed to load orders/positions for prediction chain",
            signal_id=signal_id,
            error=str(e),
            exc_info=True,
        )

    return {
        "prediction": {
            "signal_id": signal_id,
            "prediction_timestamp": prediction_target.get("prediction_timestamp") if prediction_target else None,
            "target_timestamp": prediction_target.get("target_timestamp") if prediction_target else None,
            "predicted_values": prediction_target.get("predicted_values") if prediction_target else None,
            "actual_values": prediction_target.get("actual_values") if prediction_target else None,
        }
        if prediction_target
        else None,
        "trading_result": trading_result,
        "signal": signal,
        "execution_events": [execution_event] if execution_event else [],
        "orders": orders,
        "positions": positions,
        "position_orders": position_orders,
        "target": None,
    }


