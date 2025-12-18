"""
Model quality API (trading metrics).

На данном этапе реализуется заглушка для эндпоинта:
- GET /api/v1/model-quality/trading-metrics

Полная агрегация по PnL и метрикам будет добавлена позже.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from ..config.logging import get_logger
from ..database.connection import db_pool

logger = get_logger(__name__)

router = APIRouter(prefix="/model-quality", tags=["model-quality"])


@router.get("/trading-metrics")
async def get_trading_metrics(
    model_version: Optional[str] = Query(
        None,
        description="Model version filter (optional)",
    ),
    strategy_id: Optional[str] = Query(
        None,
        description="Strategy ID filter (optional)",
    ),
    start_date: Optional[datetime] = Query(
        None,
        description="Start date filter (optional)",
    ),
    end_date: Optional[datetime] = Query(
        None,
        description="End date filter (optional)",
    ),
) -> Dict[str, Any]:
    """
    Aggregate trading PnL metrics for prediction_targets + prediction_trading_results.
    """
    try:
        pool = await db_pool.get_pool()
        params: List[Any] = []
        conditions: List[str] = []

        if model_version:
            conditions.append("pt.model_version = $%d" % (len(params) + 1))
            params.append(model_version)
        if strategy_id:
            conditions.append("ts.strategy_id = $%d" % (len(params) + 1))
            params.append(strategy_id)
        if start_date:
            conditions.append("ts.timestamp >= $%d" % (len(params) + 1))
            params.append(start_date)
        if end_date:
            conditions.append("ts.timestamp <= $%d" % (len(params) + 1))
            params.append(end_date)

        where_sql = ""
        if conditions:
            where_sql = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT 
                pt.model_version,
                ts.strategy_id,
                COUNT(DISTINCT pt.id) AS total_predictions,
                COUNT(DISTINCT CASE WHEN ptr.is_closed THEN ptr.id END) AS closed_positions,
                COUNT(DISTINCT CASE WHEN NOT ptr.is_closed THEN ptr.id END) AS open_positions,
                COALESCE(SUM(ptr.realized_pnl), 0) AS total_realized_pnl,
                COALESCE(SUM(ptr.unrealized_pnl), 0) AS total_unrealized_pnl,
                COALESCE(SUM(ptr.total_pnl), 0) AS total_pnl,
                COALESCE(SUM(ptr.fees), 0) AS total_fees,
                -- Win rate среди закрытых сделок
                COALESCE(AVG(
                    CASE 
                        WHEN ptr.is_closed AND ptr.total_pnl > 0 THEN 1.0
                        WHEN ptr.is_closed AND ptr.total_pnl <= 0 THEN 0.0
                        ELSE NULL
                    END
                ), 0) AS win_rate,
                -- Средний выигрыш/убыток по закрытым сделкам
                COALESCE(AVG(CASE WHEN ptr.is_closed AND ptr.total_pnl > 0 THEN ptr.total_pnl END), 0) AS average_win,
                COALESCE(AVG(CASE WHEN ptr.is_closed AND ptr.total_pnl < 0 THEN ptr.total_pnl END), 0) AS average_loss
            FROM prediction_targets pt
            JOIN trading_signals ts ON pt.signal_id = ts.signal_id
            LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
            {where_sql}
            GROUP BY pt.model_version, ts.strategy_id
            ORDER BY pt.model_version, ts.strategy_id
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return {
                "model_version": model_version,
                "strategy_id": strategy_id,
                "metrics": [],
            }

        metrics: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            mv = d.pop("model_version")
            sid = d.pop("strategy_id")
            metrics.append(
                {
                    "model_version": mv,
                    "strategy_id": sid,
                    "metrics": d,
                }
            )

        return {
            "model_version": model_version,
            "strategy_id": strategy_id,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error("Failed to compute trading metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute trading metrics",
        ) from e

