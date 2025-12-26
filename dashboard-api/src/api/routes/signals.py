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
                ts.metadata,
                ts.is_rejected,
                ts.rejection_reason,
                ts.effective_threshold,
                pt.predicted_values->>'direction' as model_prediction,
                COALESCE(pt.actual_values->>'candle_open', ts.market_data_snapshot->>'price') as price_from,
                pt.actual_values->>'candle_close' as price_to,
                pt.actual_values->>'direction' as actual_direction,
                pt.actual_values->>'return_value' as actual_return,
                pt.is_obsolete,
                pt.actual_values_computed_at,
                pt.target_timestamp,
                COALESCE(SUM((ee.performance->>'realized_pnl')::numeric) FILTER (WHERE ee.performance->>'realized_pnl' IS NOT NULL), 0) as total_pnl,
                CASE 
                    WHEN ts.model_version IS NULL THEN false
                    ELSE EXISTS (
                        SELECT 1 
                        FROM model_versions mv
                        WHERE mv.version = ts.model_version
                            AND mv.is_active = true
                            AND (mv.strategy_id = ts.strategy_id OR (mv.strategy_id IS NULL AND ts.strategy_id IS NULL))
                            AND (mv.symbol = ts.asset OR mv.symbol IS NULL)
                    )
                END as is_model_active
            FROM trading_signals ts
            LEFT JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
            LEFT JOIN execution_events ee ON ts.signal_id = ee.signal_id
            WHERE 1=1
            GROUP BY ts.signal_id, ts.side, ts.asset, ts.price, ts.confidence,
                     ts.strategy_id, ts.model_version, ts.timestamp, ts.is_warmup, ts.prediction_horizon_seconds,
                     ts.metadata, ts.is_rejected, ts.rejection_reason, ts.effective_threshold,
                     pt.predicted_values->>'direction',
                     pt.actual_values->>'candle_open',
                     ts.market_data_snapshot->>'price',
                     pt.actual_values->>'candle_close',
                     pt.actual_values->>'direction',
                     pt.actual_values->>'return_value',
                     pt.is_obsolete,
                     pt.actual_values_computed_at,
                     pt.target_timestamp
        """
        params = []
        param_idx = 1

        if signal_type:
            if signal_type.lower() not in {"buy", "sell"}:
                raise HTTPException(status_code=400, detail="Invalid signal_type. Must be 'buy' or 'sell'")
            query += f" AND ts.side = ${param_idx}"
            params.append(signal_type.lower())
            param_idx += 1

        if asset:
            query += f" AND ts.asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        if strategy_id:
            query += f" AND ts.strategy_id = ${param_idx}"
            params.append(strategy_id)
            param_idx += 1

        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                query += f" AND ts.timestamp >= ${param_idx}::timestamptz"
                params.append(date_from_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO 8601 format.")

        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                query += f" AND ts.timestamp <= ${param_idx}::timestamptz"
                params.append(date_to_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO 8601 format.")

        query += " ORDER BY ts.timestamp DESC"

        # Add pagination
        offset = (page - 1) * page_size
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([page_size, offset])

        rows = await DatabaseConnection.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) as count
            FROM trading_signals ts
            WHERE 1=1
        """
        count_params = []
        count_param_idx = 1

        if signal_type:
            count_query += f" AND ts.side = ${count_param_idx}"
            count_params.append(signal_type.lower())
            count_param_idx += 1

        if asset:
            count_query += f" AND ts.asset = ${count_param_idx}"
            count_params.append(asset)
            count_param_idx += 1

        if strategy_id:
            count_query += f" AND ts.strategy_id = ${count_param_idx}"
            count_params.append(strategy_id)
            count_param_idx += 1

        if date_from:
            date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            count_query += f" AND ts.timestamp >= ${count_param_idx}::timestamptz"
            count_params.append(date_from_dt)
            count_param_idx += 1

        if date_to:
            date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            count_query += f" AND ts.timestamp <= ${count_param_idx}::timestamptz"
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
            
            # Get price_from: priority is candle_open from actual_values (used for direction calculation),
            # fallback to market_data_snapshot->>'price' (from feature_vector at signal creation)
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
            
            # Validate and correct direction if it doesn't match actual price movement
            # This handles cases where direction was computed based on different prices than displayed
            if price_from is not None and price_to is not None and price_from != 0:
                try:
                    calculated_return = (price_to - price_from) / price_from
                    calculated_direction = "UP" if calculated_return > 0 else "DOWN" if calculated_return < 0 else None
                    
                    # If direction exists but doesn't match calculated direction, use calculated one
                    # This ensures consistency between displayed prices and direction
                    if actual_direction is not None and calculated_direction is not None:
                        if actual_direction != calculated_direction:
                            logger.debug(
                                "Direction mismatch corrected",
                                signal_id=str(row.get("signal_id")),
                                stored_direction=actual_direction,
                                calculated_direction=calculated_direction,
                                price_from=price_from,
                                price_to=price_to,
                                calculated_return=calculated_return,
                            )
                            actual_direction = calculated_direction
                            # Also update return_value if it was inconsistent
                            if actual_return is None or abs(actual_return - calculated_return) > 0.0001:
                                actual_return = calculated_return
                    elif calculated_direction is not None and actual_direction is None:
                        # If direction is missing but we can calculate it, use calculated one
                        actual_direction = calculated_direction
                        if actual_return is None:
                            actual_return = calculated_return
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Get total PnL from execution events
            total_pnl = None
            if row.get("total_pnl") is not None:
                try:
                    total_pnl = float(row["total_pnl"])
                except (ValueError, TypeError):
                    pass
            
            # Determine actual_movement status
            actual_movement_status = None
            is_obsolete = row.get("is_obsolete", False) if row.get("is_obsolete") is not None else False
            actual_values_computed_at = row.get("actual_values_computed_at")
            target_timestamp = row.get("target_timestamp")
            
            # Check if target exists
            if target_timestamp is not None:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                target_ts = target_timestamp
                if isinstance(target_ts, str):
                    target_ts = datetime.fromisoformat(target_ts.replace("Z", "+00:00"))
                elif hasattr(target_ts, 'tzinfo'):
                    if target_ts.tzinfo is None:
                        target_ts = target_ts.replace(tzinfo=timezone.utc)
                    else:
                        target_ts = target_ts.astimezone(timezone.utc)
                
                if is_obsolete:
                    actual_movement_status = "obsolete"  # Попытки прекращены
                elif actual_values_computed_at is not None:
                    actual_movement_status = "computed"  # Вычислено
                elif target_ts <= now:
                    actual_movement_status = "pending"  # Ожидается вычисление
                else:
                    actual_movement_status = "waiting"  # Ожидается target_timestamp
            elif price_from is not None or price_to is not None:
                # If we have prices but no target_timestamp, assume computed
                actual_movement_status = "computed"
            
            # Extract raw prediction data from metadata
            raw_prediction_data = None
            metadata = row.get("metadata")
            
            # Handle JSONB: asyncpg returns JSONB as string, need to parse it
            if metadata:
                # asyncpg returns JSONB as string, parse it
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Failed to parse metadata as JSON", signal_id=str(row.get("signal_id")), error=str(e))
                        metadata = None
                
                # Now metadata should be a dict (or None if parsing failed)
                if metadata and isinstance(metadata, dict):
                    prediction_result = metadata.get("prediction_result")
                    effective_threshold = metadata.get("effective_threshold")
                    threshold_source = metadata.get("threshold_source")
                    
                    if prediction_result or effective_threshold is not None:
                        raw_prediction_data = {
                            "prediction_result": prediction_result,
                            "effective_threshold": float(effective_threshold) if effective_threshold is not None else None,
                            "threshold_source": threshold_source,
                        }
                elif metadata:
                    logger.debug("Metadata is not a dict after parsing", signal_id=str(row.get("signal_id")), metadata_type=type(metadata).__name__)
            
            # Fallback: if metadata doesn't have prediction_result, try to get from effective_threshold column
            if raw_prediction_data is None or raw_prediction_data.get("prediction_result") is None:
                effective_threshold_col = row.get("effective_threshold")
                if effective_threshold_col is not None:
                    if raw_prediction_data is None:
                        raw_prediction_data = {}
                    raw_prediction_data["effective_threshold"] = float(effective_threshold_col)
                    if not raw_prediction_data.get("threshold_source"):
                        raw_prediction_data["threshold_source"] = "static"  # Default if not in metadata
            
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
                "raw_prediction_data": raw_prediction_data,  # Raw prediction data from metadata
                "is_rejected": bool(row.get("is_rejected", False)),
                "rejection_reason": row.get("rejection_reason"),
                "actual_movement": {
                    "price_from": price_from,
                    "price_to": price_to,
                    "direction": actual_direction,  # "UP", "DOWN", or None
                    "return_value": actual_return,  # Decimal value (e.g., 0.00002 for 0.002%)
                    "status": actual_movement_status,  # "computed", "pending", "waiting", "obsolete", or None
                } if price_from is not None or price_to is not None or actual_movement_status else None,
                "total_pnl": str(total_pnl) if total_pnl is not None else None,  # Total PnL from execution events
                "is_model_active": bool(row.get("is_model_active", False)),  # Whether the model is currently active
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

