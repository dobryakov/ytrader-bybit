"""Trading signal query endpoints."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/signals")
async def list_signals(
    signal_type: Optional[str] = Query(None, description="Filter by signal type (buy, sell)"),
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    date_from: Optional[str] = Query(None, description="Filter signals from date (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="Filter signals until date (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List trading signals with filtering and pagination."""
    trace_id = get_or_create_trace_id()
    logger.info("signal_list_request", asset=asset, strategy_id=strategy_id, trace_id=trace_id)

    try:
        query = """
            SELECT 
                ts.signal_id, ts.side, ts.asset, ts.price, ts.confidence,
                ts.strategy_id, ts.model_version, ts.timestamp, ts.is_warmup, ts.prediction_horizon_seconds,
                pt.predicted_values->>'direction' as model_prediction,
                ts.market_data_snapshot->>'price' as price_from,
                pt.actual_values->>'candle_close' as price_to,
                pt.actual_values->>'direction' as actual_direction,
                pt.actual_values->>'return_value' as actual_return
            FROM trading_signals ts
            LEFT JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if signal_type:
            if signal_type.lower() not in {"buy", "sell"}:
                raise HTTPException(status_code=400, detail="Invalid signal_type. Must be 'buy' or 'sell'")
            query += f" AND side = ${param_idx}"
            params.append(signal_type.lower())
            param_idx += 1

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

        query += " ORDER BY timestamp DESC"

        # Add pagination
        offset = (page - 1) * page_size
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([page_size, offset])

        rows = await DatabaseConnection.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) as count
            FROM trading_signals
            WHERE 1=1
        """
        count_params = []
        count_param_idx = 1

        if signal_type:
            count_query += f" AND side = ${count_param_idx}"
            count_params.append(signal_type.lower())
            count_param_idx += 1

        if asset:
            count_query += f" AND asset = ${count_param_idx}"
            count_params.append(asset)
            count_param_idx += 1

        if strategy_id:
            count_query += f" AND strategy_id = ${count_param_idx}"
            count_params.append(strategy_id)
            count_param_idx += 1

        if date_from:
            date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            count_query += f" AND timestamp >= ${count_param_idx}::timestamptz"
            count_params.append(date_from_dt)
            count_param_idx += 1

        if date_to:
            date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            count_query += f" AND timestamp <= ${count_param_idx}::timestamptz"
            count_params.append(date_to_dt)
            count_param_idx += 1

        total_count = await DatabaseConnection.fetchval(count_query, *count_params)

        signals_data = []
        for row in rows:
            # Map model prediction: "green" -> "UP", "red" -> "DOWN", None -> None
            model_prediction = None
            if row.get("model_prediction"):
                if row["model_prediction"] == "green":
                    model_prediction = "UP"
                elif row["model_prediction"] == "red":
                    model_prediction = "DOWN"
            
            # Parse actual price movement
            price_from = None
            price_to = None
            actual_direction = None
            actual_return = None
            
            # Try to get price_from from market_data_snapshot first, fallback to signal price
            if row.get("price_from"):
                try:
                    price_from = float(row["price_from"])
                except (ValueError, TypeError):
                    pass
            
            if price_from is None and row.get("price"):
                try:
                    price_from = float(row["price"])
                except (ValueError, TypeError):
                    pass
            
            if row.get("price_to"):
                try:
                    price_to = float(row["price_to"])
                except (ValueError, TypeError):
                    pass
            
            if row.get("actual_direction"):
                if row["actual_direction"] == "green":
                    actual_direction = "UP"
                elif row["actual_direction"] == "red":
                    actual_direction = "DOWN"
            
            if row.get("actual_return"):
                try:
                    actual_return = float(row["actual_return"])
                except (ValueError, TypeError):
                    pass
            
            # Fallback: calculate return_value from price_from and price_to if it's missing or zero
            # but we have both prices (this handles cases where return_value was incorrectly stored as 0.0)
            if actual_return == 0.0 and price_from is not None and price_to is not None and price_from != 0:
                try:
                    calculated_return = (price_to - price_from) / price_from
                    # Only use calculated return if it's significantly different from 0 (more than 0.0001%)
                    if abs(calculated_return) > 0.000001:
                        actual_return = calculated_return
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            signal_dict = {
                "signal_id": str(row["signal_id"]),
                "signal_type": row["side"],  # Map 'side' to 'signal_type' for API compatibility
                "asset": row["asset"],
                "amount": str(row["price"]),  # Map 'price' to 'amount' for API compatibility
                "confidence": float(row["confidence"]) if row["confidence"] else None,
                "strategy_id": row["strategy_id"],
                "model_version": row["model_version"],
                "timestamp": row["timestamp"].isoformat() + "Z",
                "is_warmup": row["is_warmup"],
                "horizon": row.get("prediction_horizon_seconds"),  # May be None
                "model_prediction": model_prediction,  # "UP", "DOWN", or None
                "actual_movement": {
                    "price_from": price_from,
                    "price_to": price_to,
                    "direction": actual_direction,  # "UP", "DOWN", or None
                    "return_value": actual_return,  # Decimal value (e.g., 0.00002 for 0.002%)
                } if price_from is not None or price_to is not None else None,
            }
            signals_data.append(signal_dict)

        logger.info("signal_list_completed", count=len(signals_data), total=total_count, trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "signals": signals_data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("signal_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signals: {str(e)}")

