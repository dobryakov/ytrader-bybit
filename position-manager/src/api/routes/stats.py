"""Position statistics API endpoints for Position Manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...utils.tracing import get_or_create_trace_id
from ..middleware.auth import api_key_auth


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["stats"])


@router.get(
    "/stats",
    dependencies=[Depends(api_key_auth)],
    summary="Get detailed position statistics including PnL, holding duration, predictions, and close reasons",
)
async def get_position_stats(
    asset: Optional[str] = Query(None, description="Filter by trading pair (e.g., BTCUSDT)"),
    mode: Optional[str] = Query(None, description="Filter by trading mode (one-way, hedge)"),
    status: Optional[str] = Query(None, description="Filter by status (open, closed, all)"),
    from_date: Optional[datetime] = Query(None, description="Start date filter (for closed positions)"),
    to_date: Optional[datetime] = Query(None, description="End date filter (for closed positions)"),
    group_by: Optional[str] = Query(None, description="Group results by (asset, status, none)"),
    include_details: bool = Query(False, description="Include detailed information per position (predictions, close reasons)"),
) -> JSONResponse:
    """Get comprehensive position statistics including PnL metrics, holding duration, model predictions, and close reasons.
    
    Returns aggregated statistics about positions including:
    - Summary counts (total, open, closed)
    - PnL statistics (total, average, win rate)
    - Duration statistics (average, min, max, median holding time)
    - Grouped statistics by asset or status (if requested)
    - Detailed position information with predictions and close reasons (if include_details=true)
    """
    trace_id = get_or_create_trace_id()
    
    logger.info(
        "position_stats_request",
        asset=asset,
        mode=mode,
        status=status,
        from_date=from_date.isoformat() if from_date else None,
        to_date=to_date.isoformat() if to_date else None,
        group_by=group_by,
        include_details=include_details,
        trace_id=trace_id,
    )
    
    # Validate mode
    if mode is not None:
        mode_lower = mode.lower()
        if mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")
    else:
        mode_lower = None
    
    # Validate status
    if status is not None:
        status_lower = status.lower()
        if status_lower not in {"open", "closed", "all"}:
            raise HTTPException(status_code=400, detail="Invalid status. Must be 'open', 'closed', or 'all'")
    else:
        status_lower = "all"
    
    # Validate group_by
    if group_by is not None:
        group_by_lower = group_by.lower()
        if group_by_lower not in {"asset", "status", "none"}:
            raise HTTPException(status_code=400, detail="Invalid group_by. Must be 'asset', 'status', or 'none'")
    else:
        group_by_lower = None
    
    try:
        pool = await DatabaseConnection.get_pool()
        
        # Build base query with position stats
        base_query = """
            WITH position_stats AS (
                SELECT 
                    p.id,
                    p.asset,
                    p.mode,
                    p.size,
                    p.realized_pnl,
                    p.unrealized_pnl,
                    p.created_at,
                    p.closed_at,
                    p.last_updated,
                    CASE 
                        WHEN p.size = 0 AND p.closed_at IS NOT NULL THEN 'closed'
                        WHEN p.size != 0 THEN 'open'
                        ELSE 'unknown'
                    END as status,
                    CASE 
                        WHEN p.closed_at IS NOT NULL AND p.created_at IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (p.closed_at - p.created_at)) / 60
                        WHEN p.size != 0 AND p.created_at IS NOT NULL
                        THEN EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 60
                        ELSE NULL
                    END as duration_minutes
                FROM positions p
                WHERE 1=1
        """
        
        params: List[Any] = []
        param_idx = 1
        
        # Add filters
        if mode_lower:
            base_query += f" AND p.mode = ${param_idx}"
            params.append(mode_lower)
            param_idx += 1
        
        if asset:
            base_query += f" AND p.asset = ${param_idx}"
            params.append(asset.upper())
            param_idx += 1
        
        if status_lower != "all":
            if status_lower == "closed":
                base_query += f" AND p.size = 0 AND p.closed_at IS NOT NULL"
            elif status_lower == "open":
                base_query += f" AND p.size != 0"
        
        if from_date:
            base_query += f" AND (p.closed_at IS NULL OR p.closed_at >= ${param_idx})"
            params.append(from_date)
            param_idx += 1
        
        if to_date:
            base_query += f" AND (p.closed_at IS NULL OR p.closed_at <= ${param_idx})"
            params.append(to_date)
            param_idx += 1
        
        # Summary statistics
        summary_query = base_query + """
            )
            SELECT 
                COUNT(*) as total_positions,
                COUNT(*) FILTER (WHERE status = 'open') as open_positions,
                COUNT(*) FILTER (WHERE status = 'closed') as closed_positions,
                COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
                COALESCE(SUM(realized_pnl) + SUM(unrealized_pnl), 0) as total_pnl,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl > 0) as winning_positions,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl < 0) as losing_positions,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl = 0) as breakeven_positions
            FROM position_stats
        """
        
        summary_row = await pool.fetchrow(summary_query, *params)
        
        # Duration statistics for closed positions
        duration_query = base_query + """
            )
            SELECT 
                COUNT(*) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as closed_positions_count,
                AVG(duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as avg_duration_minutes,
                MIN(duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as min_duration_minutes,
                MAX(duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as max_duration_minutes,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as p25_duration_minutes,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as median_duration_minutes,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as p75_duration_minutes,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as p90_duration_minutes,
                SUM(duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as total_duration_minutes
            FROM position_stats
        """
        
        duration_row = await pool.fetchrow(duration_query, *params)
        
        # PnL statistics
        pnl_query = base_query + """
            )
            SELECT 
                COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
                AVG(realized_pnl) FILTER (WHERE status = 'closed') as avg_realized_pnl_per_closed_position,
                AVG(unrealized_pnl) FILTER (WHERE status = 'open') as avg_unrealized_pnl_per_open_position,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl > 0) as winning_positions,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl < 0) as losing_positions,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl != 0) as total_closed_with_pnl
            FROM position_stats
        """
        
        pnl_row = await pool.fetchrow(pnl_query, *params)
        
        # Calculate win rate
        total_closed_with_pnl = pnl_row["total_closed_with_pnl"] or 0
        winning_positions = pnl_row["winning_positions"] or 0
        win_rate = float(winning_positions / total_closed_with_pnl) if total_closed_with_pnl > 0 else None
        
        # Extract duration values for analysis (convert Decimal to float)
        def to_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            return float(value)
        
        p25_min = to_float(duration_row["p25_duration_minutes"])
        p75_min = to_float(duration_row["p75_duration_minutes"])
        median_min = to_float(duration_row["median_duration_minutes"])
        p90_min = to_float(duration_row["p90_duration_minutes"])
        min_min = to_float(duration_row["min_duration_minutes"])
        max_min = to_float(duration_row["max_duration_minutes"])
        avg_min = to_float(duration_row["avg_duration_minutes"])
        
        # Helper function to convert minutes to hours and days
        def format_duration(minutes: Optional[float]) -> Optional[Dict[str, float]]:
            if minutes is None:
                return None
            return {
                "minutes": round(minutes, 2),
                "hours": round(minutes / 60, 2),
                "days": round(minutes / 1440, 2),
            }
        
        # Calculate IQR (Interquartile Range) for outlier detection
        iqr_data = None
        if p25_min is not None and p75_min is not None:
            iqr = p75_min - p25_min
            lower_bound = p25_min - 1.5 * iqr
            upper_bound = p75_min + 1.5 * iqr
            has_outliers = max_min is not None and (max_min < lower_bound or max_min > upper_bound)
            
            iqr_data = {
                "iqr_minutes": round(iqr, 2),
                "outlier_detection": {
                    "lower_bound_minutes": round(lower_bound, 2),
                    "upper_bound_minutes": round(upper_bound, 2),
                    "has_outliers": has_outliers,
                },
            }
        
        # Build response
        response: Dict[str, Any] = {
            "summary": {
                "total_positions": summary_row["total_positions"] or 0,
                "open_positions": summary_row["open_positions"] or 0,
                "closed_positions": summary_row["closed_positions"] or 0,
                "total_realized_pnl": float(summary_row["total_realized_pnl"] or 0),
                "total_unrealized_pnl": float(summary_row["total_unrealized_pnl"] or 0),
                "total_pnl": float(summary_row["total_pnl"] or 0),
            },
            "duration_stats": {
                "closed_positions_count": duration_row["closed_positions_count"] or 0,
                # Raw values in minutes
                "avg_duration_minutes": round(float(avg_min or 0), 2) if avg_min is not None else None,
                "min_duration_minutes": round(float(min_min or 0), 2) if min_min is not None else None,
                "max_duration_minutes": round(float(max_min or 0), 2) if max_min is not None else None,
                "p25_duration_minutes": round(float(p25_min or 0), 2) if p25_min is not None else None,
                "median_duration_minutes": round(float(median_min or 0), 2) if median_min is not None else None,
                "p75_duration_minutes": round(float(p75_min or 0), 2) if p75_min is not None else None,
                "p90_duration_minutes": round(float(p90_min or 0), 2) if p90_min is not None else None,
                "total_duration_minutes": round(float(duration_row["total_duration_minutes"] or 0), 2) if duration_row["total_duration_minutes"] is not None else None,
                # Formatted values (minutes, hours, days)
                "avg_duration": format_duration(avg_min),
                "min_duration": format_duration(min_min),
                "max_duration": format_duration(max_min),
                "p25_duration": format_duration(p25_min),
                "median_duration": format_duration(median_min),
                "p75_duration": format_duration(p75_min),
                "p90_duration": format_duration(p90_min),
                # IQR analysis for outlier detection
                "iqr_analysis": iqr_data,
                # Recommendations
                "recommendations": {
                    "use_p75_or_p90": "Для исключения аномалий рекомендуется использовать P75 или P90 вместо среднего арифметического",
                    "use_median": "Медиана (P50) более устойчива к выбросам, чем среднее арифметическое",
                    "percentile_meaning": {
                        "p25": "25% позиций закрываются быстрее этого времени",
                        "p50": "50% позиций закрываются быстрее этого времени (медиана)",
                        "p75": "75% позиций закрываются быстрее этого времени",
                        "p90": "90% позиций закрываются быстрее этого времени (исключает большинство аномалий)",
                    },
                },
            },
            "pnl_stats": {
                "total_realized_pnl": float(pnl_row["total_realized_pnl"] or 0),
                "total_unrealized_pnl": float(pnl_row["total_unrealized_pnl"] or 0),
                "avg_realized_pnl_per_closed_position": round(float(pnl_row["avg_realized_pnl_per_closed_position"] or 0), 2) if pnl_row["avg_realized_pnl_per_closed_position"] is not None else None,
                "avg_unrealized_pnl_per_open_position": round(float(pnl_row["avg_unrealized_pnl_per_open_position"] or 0), 2) if pnl_row["avg_unrealized_pnl_per_open_position"] is not None else None,
                "winning_positions": pnl_row["winning_positions"] or 0,
                "losing_positions": pnl_row["losing_positions"] or 0,
                "breakeven_positions": summary_row["breakeven_positions"] or 0,
                "win_rate": round(win_rate, 4) if win_rate is not None else None,
            },
        }
        
        # Add detailed position information if requested
        if include_details:
            details_query = base_query + """
            )
            SELECT 
                ps.id as position_id,
                ps.asset,
                ps.size,
                ps.created_at,
                ps.closed_at,
                ps.realized_pnl,
                ps.unrealized_pnl,
                ps.status,
                ps.duration_minutes,
                -- Предсказания через prediction_trading_results
                ptr.entry_signal_id,
                ptr.exit_signal_id,
                ptr.entry_price,
                ptr.exit_price,
                ptr.position_size_at_entry,
                ptr.position_size_at_exit,
                ptr.entry_timestamp,
                ptr.exit_timestamp,
                ptr.realized_pnl as ptr_realized_pnl,
                ptr.total_pnl as ptr_total_pnl,
                entry_ts.metadata->>'prediction_result' as entry_prediction_json,
                entry_pt.predicted_values as entry_predicted_values,
                entry_pt.actual_values as entry_actual_values,
                entry_pt.target_timestamp,
                entry_pt.prediction_timestamp,
                -- Причина закрытия через exit signal
                exit_ts.metadata->>'exit_reason' as exit_reason,
                exit_ts.metadata->>'rule_triggered' as exit_rule_triggered,
                -- Альтернативный путь через orders (если exit_signal_id NULL)
                o_closed.signal_id as closing_order_signal_id,
                ts_closed.metadata->>'exit_reason' as closing_exit_reason,
                ts_closed.metadata->>'rule_triggered' as closing_rule_triggered,
                -- Ордера для входа
                sor_entry.order_id as entry_order_id,
                o_entry.side as entry_order_side,
                o_entry.status as entry_order_status,
                o_entry.price as entry_order_price,
                o_entry.quantity as entry_order_quantity,
                o_entry.created_at as entry_order_created
            FROM position_stats ps
            LEFT JOIN position_orders po_entry ON po_entry.position_id = ps.id
            LEFT JOIN signal_order_relationships sor_entry ON sor_entry.order_id = po_entry.order_id
            LEFT JOIN orders o_entry ON o_entry.id = sor_entry.order_id
            LEFT JOIN prediction_trading_results ptr ON ptr.signal_id = sor_entry.signal_id
            LEFT JOIN trading_signals entry_ts ON entry_ts.signal_id = ptr.entry_signal_id
            LEFT JOIN trading_signals exit_ts ON exit_ts.signal_id = ptr.exit_signal_id
            LEFT JOIN prediction_targets entry_pt ON entry_pt.signal_id = entry_ts.signal_id
            LEFT JOIN position_orders po_closed ON po_closed.position_id = ps.id AND po_closed.relationship_type = 'closed'
            LEFT JOIN orders o_closed ON o_closed.id = po_closed.order_id
            LEFT JOIN trading_signals ts_closed ON ts_closed.signal_id = o_closed.signal_id
            ORDER BY ps.closed_at DESC NULLS LAST, ps.created_at DESC
            """
            
            details_rows = await pool.fetch(details_query, *params)
            
            position_details = []
            for row in details_rows:
                # Определяем причину закрытия (приоритет: exit_signal, затем closing_order_signal)
                close_reason = None
                close_rule = None
                if row["exit_reason"]:
                    close_reason = row["exit_reason"]
                    close_rule = row["exit_rule_triggered"]
                elif row["closing_exit_reason"]:
                    close_reason = row["closing_exit_reason"]
                    close_rule = row["closing_rule_triggered"]
                elif row["status"] == "closed":
                    # Если позиция закрыта, но нет exit_reason, возможно это встречный ордер
                    close_reason = "closed_by_opposite_order"
                
                # Парсим prediction_result из JSON
                entry_prediction = None
                if row["entry_prediction_json"]:
                    try:
                        entry_prediction = json.loads(row["entry_prediction_json"]) if isinstance(row["entry_prediction_json"], str) else row["entry_prediction_json"]
                    except Exception:
                        entry_prediction = row["entry_prediction_json"]
                
                # Парсим predicted_values и actual_values
                predicted_values = None
                actual_values = None
                if row["entry_predicted_values"]:
                    try:
                        predicted_values = json.loads(row["entry_predicted_values"]) if isinstance(row["entry_predicted_values"], str) else row["entry_predicted_values"]
                    except Exception:
                        predicted_values = row["entry_predicted_values"]
                
                if row["entry_actual_values"]:
                    try:
                        actual_values = json.loads(row["entry_actual_values"]) if isinstance(row["entry_actual_values"], str) else row["entry_actual_values"]
                    except Exception:
                        actual_values = row["entry_actual_values"]
                
                # Анализ сравнения предсказания с реальностью
                market_analysis = None
                if predicted_values and actual_values:
                    predicted_dir = predicted_values.get("direction") if isinstance(predicted_values, dict) else None
                    actual_dir = actual_values.get("direction") if isinstance(actual_values, dict) else None
                    predicted_conf = predicted_values.get("confidence") if isinstance(predicted_values, dict) else None
                    actual_open = actual_values.get("candle_open") if isinstance(actual_values, dict) else None
                    actual_close = actual_values.get("candle_close") if isinstance(actual_values, dict) else None
                    actual_return = actual_values.get("return_value") if isinstance(actual_values, dict) else None
                    
                    prediction_correct = predicted_dir == actual_dir if predicted_dir and actual_dir else None
                    price_change = None
                    if actual_open and actual_close:
                        try:
                            price_change = float(actual_close) - float(actual_open)
                            price_change_pct = (price_change / float(actual_open)) * 100 if float(actual_open) != 0 else 0
                        except (ValueError, TypeError):
                            pass
                    
                    market_analysis = {
                        "prediction_direction": predicted_dir,
                        "actual_direction": actual_dir,
                        "prediction_correct": prediction_correct,
                        "predicted_confidence": float(predicted_conf) if predicted_conf else None,
                        "actual_price_open": float(actual_open) if actual_open else None,
                        "actual_price_close": float(actual_close) if actual_close else None,
                        "price_change": price_change,
                        "price_change_percent": round(price_change_pct, 4) if price_change is not None else None,
                        "actual_return": float(actual_return) if actual_return else None,
                    }
                
                # Информация об ордерах
                entry_order = None
                if row["entry_order_id"]:
                    entry_order = {
                        "order_id": str(row["entry_order_id"]),
                        "side": row["entry_order_side"],
                        "status": row["entry_order_status"],
                        "price": float(row["entry_order_price"]) if row["entry_order_price"] else None,
                        "quantity": float(row["entry_order_quantity"]) if row["entry_order_quantity"] else None,
                        "created_at": row["entry_order_created"].isoformat() + "Z" if row["entry_order_created"] else None,
                    }
                
                position_details.append({
                    "position_id": str(row["position_id"]),
                    "asset": row["asset"],
                    "size": float(row["size"] or 0),
                    "status": row["status"],
                    "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                    "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
                    "realized_pnl": float(row["realized_pnl"] or 0),
                    "unrealized_pnl": float(row["unrealized_pnl"] or 0),
                    "duration_minutes": round(float(row["duration_minutes"] or 0), 2) if row["duration_minutes"] is not None else None,
                    "prediction": {
                        "entry_signal_id": str(row["entry_signal_id"]) if row["entry_signal_id"] else None,
                        "entry_prediction": entry_prediction,
                        "predicted_values": predicted_values,
                        "actual_values": actual_values,
                        "target_timestamp": row["target_timestamp"].isoformat() + "Z" if row["target_timestamp"] else None,
                        "prediction_timestamp": row["prediction_timestamp"].isoformat() + "Z" if row["prediction_timestamp"] else None,
                    } if row["entry_signal_id"] or row["entry_predicted_values"] else None,
                    "trading": {
                        "entry_price": float(row["entry_price"]) if row["entry_price"] else None,
                        "exit_price": float(row["exit_price"]) if row["exit_price"] else None,
                        "entry_timestamp": row["entry_timestamp"].isoformat() + "Z" if row["entry_timestamp"] else None,
                        "exit_timestamp": row["exit_timestamp"].isoformat() + "Z" if row["exit_timestamp"] else None,
                        "position_size_at_entry": float(row["position_size_at_entry"]) if row["position_size_at_entry"] else None,
                        "position_size_at_exit": float(row["position_size_at_exit"]) if row["position_size_at_exit"] else None,
                        "realized_pnl": float(row["ptr_realized_pnl"]) if row["ptr_realized_pnl"] else None,
                        "total_pnl": float(row["ptr_total_pnl"]) if row["ptr_total_pnl"] else None,
                        "entry_order": entry_order,
                    } if row["entry_signal_id"] else None,
                    "market_analysis": market_analysis,
                    "close_reason": {
                        "reason": close_reason,
                        "rule_triggered": close_rule,
                        "exit_signal_id": str(row["exit_signal_id"]) if row["exit_signal_id"] else None,
                        "closing_order_signal_id": str(row["closing_order_signal_id"]) if row["closing_order_signal_id"] else None,
                    } if close_reason or row["status"] == "closed" else None,
                })
            
            response["position_details"] = position_details
        
        # Add grouped statistics if requested
        if group_by_lower == "asset":
            by_asset_query = base_query + """
                )
                SELECT 
                    asset,
                    COUNT(*) as total_positions,
                    COUNT(*) FILTER (WHERE status = 'open') as open_positions,
                    COUNT(*) FILTER (WHERE status = 'closed') as closed_positions,
                    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                    COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
                    AVG(duration_minutes) FILTER (WHERE status = 'closed' AND duration_minutes IS NOT NULL) as avg_duration_minutes
                FROM position_stats
                GROUP BY asset
                ORDER BY asset
            """
            
            by_asset_rows = await pool.fetch(by_asset_query, *params)
            response["by_asset"] = [
                {
                    "asset": row["asset"],
                    "total_positions": row["total_positions"],
                    "open_positions": row["open_positions"],
                    "closed_positions": row["closed_positions"],
                    "total_realized_pnl": float(row["total_realized_pnl"] or 0),
                    "total_unrealized_pnl": float(row["total_unrealized_pnl"] or 0),
                    "avg_duration_minutes": round(float(row["avg_duration_minutes"] or 0), 2) if row["avg_duration_minutes"] is not None else None,
                }
                for row in by_asset_rows
            ]
        
        elif group_by_lower == "status":
            by_status_query = base_query + """
                )
                SELECT 
                    status,
                    COUNT(*) as count,
                    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                    COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
                    AVG(duration_minutes) FILTER (WHERE duration_minutes IS NOT NULL) as avg_time_held_minutes
                FROM position_stats
                WHERE status IN ('open', 'closed')
                GROUP BY status
                ORDER BY status
            """
            
            by_status_rows = await pool.fetch(by_status_query, *params)
            response["by_status"] = {
                row["status"]: {
                    "count": row["count"],
                    "total_realized_pnl": float(row["total_realized_pnl"] or 0),
                    "total_unrealized_pnl": float(row["total_unrealized_pnl"] or 0),
                    "avg_time_held_minutes": round(float(row["avg_time_held_minutes"] or 0), 2) if row["avg_time_held_minutes"] is not None else None,
                }
                for row in by_status_rows
            }
        
        logger.info(
            "position_stats_completed",
            total_positions=response["summary"]["total_positions"],
            trace_id=trace_id,
        )
        
        return JSONResponse(status_code=200, content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "position_stats_failed",
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position statistics: {e}") from e

