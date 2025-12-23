"""Chart data endpoints."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/charts/pnl")
async def get_pnl_chart(
    date_from: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="End date (ISO 8601)"),
    interval: str = Query("1h", description="Time interval (1h, 4h, 1d)"),
):
    """Get PnL chart data."""
    trace_id = get_or_create_trace_id()
    logger.info("chart_pnl_request", date_from=date_from, date_to=date_to, interval=interval, trace_id=trace_id)

    try:
        # Validate interval
        valid_intervals = {"1h", "4h", "1d"}
        if interval not in valid_intervals:
            raise HTTPException(status_code=400, detail=f"Invalid interval. Must be one of {valid_intervals}")

        # Map interval to PostgreSQL date_trunc interval
        interval_map = {"1h": "hour", "4h": "hour", "1d": "day"}

        # Build query
        query = """
            SELECT 
                DATE_TRUNC($1, snapshot_timestamp) as time_bucket,
                SUM(unrealized_pnl) as unrealized_pnl,
                SUM(realized_pnl) as realized_pnl
            FROM position_snapshots
            WHERE 1=1
        """
        params = [interval_map[interval]]
        param_idx = 2

        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                query += f" AND snapshot_timestamp >= ${param_idx}::timestamptz"
                params.append(date_from_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO 8601 format.")

        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                query += f" AND snapshot_timestamp <= ${param_idx}::timestamptz"
                params.append(date_to_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO 8601 format.")

        query += " GROUP BY time_bucket ORDER BY time_bucket"

        rows = await DatabaseConnection.fetch(query, *params)

        chart_data = []
        for row in rows:
            chart_dict = {
                "time": row["time_bucket"].isoformat() + "Z",
                "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else "0",
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else "0",
            }
            chart_data.append(chart_dict)

        logger.info("chart_pnl_completed", count=len(chart_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "data": chart_data,
                "count": len(chart_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chart_pnl_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart data: {str(e)}")


@router.get("/charts/signals-confidence")
async def get_signals_confidence_chart(
    asset: Optional[str] = Query(None, description="Filter by asset"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    date_from: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="End date (ISO 8601)"),
):
    """Get signals confidence chart data."""
    trace_id = get_or_create_trace_id()
    logger.info("chart_signals_confidence_request", asset=asset, trace_id=trace_id)

    try:
        query = """
            SELECT 
                timestamp,
                AVG(confidence) as avg_confidence,
                COUNT(*) as signal_count
            FROM trading_signals
            WHERE confidence IS NOT NULL
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        if strategy_id:
            query += f" AND strategy_id = ${param_idx}"
            params.append(strategy_id)
            param_idx += 1

        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                query += f" AND timestamp >= ${param_idx}::timestamptz"
                params.append(date_from_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO 8601 format.")

        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                query += f" AND timestamp <= ${param_idx}::timestamptz"
                params.append(date_to_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO 8601 format.")

        query += " GROUP BY timestamp ORDER BY timestamp"

        rows = await DatabaseConnection.fetch(query, *params)

        chart_data = []
        for row in rows:
            chart_dict = {
                "time": row["timestamp"].isoformat() + "Z",
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else None,
                "signal_count": row["signal_count"],
            }
            chart_data.append(chart_dict)

        logger.info("chart_signals_confidence_completed", count=len(chart_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "data": chart_data,
                "count": len(chart_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chart_signals_confidence_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart data: {str(e)}")

