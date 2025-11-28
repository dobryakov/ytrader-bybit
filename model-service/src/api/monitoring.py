"""
Monitoring and observability API endpoints.

Provides REST API endpoints for system monitoring:
- Model performance metrics
- System health details
- Active models count
- Strategy performance time-series
"""

from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.connection import db_pool
from ..config.logging import get_logger
from ..services.signal_skip_metrics import signal_skip_metrics

logger = get_logger(__name__)

router = APIRouter(tags=["monitoring"])


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics response."""

    active_models_count: int
    total_models_count: int
    models_by_strategy: dict = Field(default_factory=dict)
    models_by_type: dict = Field(default_factory=dict)


class SystemHealthDetails(BaseModel):
    """System health details response."""

    database_connected: bool
    message_queue_connected: bool
    model_storage_accessible: bool
    active_models: int
    warmup_mode_enabled: bool


class StrategyPerformanceDataPoint(BaseModel):
    """Strategy performance data point."""

    timestamp: str
    success_rate: float
    total_pnl: float
    avg_pnl: float
    total_orders: int
    successful_orders: int


class StrategyPerformanceTimeSeries(BaseModel):
    """Strategy performance time-series response."""

    strategy_id: str
    granularity: str
    start_time: str
    end_time: str
    data_points: List[StrategyPerformanceDataPoint]


class SignalSkipMetricsResponse(BaseModel):
    """Signal skip metrics response."""

    total_skips: int
    by_asset_strategy: dict
    by_reason: dict
    last_reset: str


@router.get("/monitoring/models/performance", response_model=ModelPerformanceMetrics)
async def get_model_performance_metrics() -> ModelPerformanceMetrics:
    """
    Get model performance metrics.

    Returns:
        Model performance metrics including counts and breakdowns
    """
    try:
        repo = ModelVersionRepository()

        # Get total models count
        total_count_record = await repo._fetchrow("SELECT COUNT(*) FROM model_versions")
        total_models_count = total_count_record[0] if total_count_record else 0

        # Get active models count
        active_count_record = await repo._fetchrow("SELECT COUNT(*) FROM model_versions WHERE is_active = true")
        active_models_count = active_count_record[0] if active_count_record else 0

        # Get models by strategy
        strategy_records = await repo._fetch(
            """
            SELECT strategy_id, COUNT(*) as count
            FROM model_versions
            GROUP BY strategy_id
        """
        )
        models_by_strategy = {str(record["strategy_id"]) if record["strategy_id"] else "default": record["count"] for record in strategy_records}

        # Get models by type
        type_records = await repo._fetch(
            """
            SELECT model_type, COUNT(*) as count
            FROM model_versions
            GROUP BY model_type
        """
        )
        models_by_type = {record["model_type"]: record["count"] for record in type_records}

        logger.info("Retrieved model performance metrics", active_models_count=active_models_count, total_models_count=total_models_count)

        return ModelPerformanceMetrics(
            active_models_count=active_models_count,
            total_models_count=total_models_count,
            models_by_strategy=models_by_strategy,
            models_by_type=models_by_type,
        )
    except Exception as e:
        logger.error("Failed to get model performance metrics", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/monitoring/health", response_model=SystemHealthDetails)
async def get_system_health_details() -> SystemHealthDetails:
    """
    Get detailed system health information.

    Returns:
        System health details including connectivity and status
    """
    try:
        # Check database connection
        database_connected = False
        try:
            async with db_pool.get_connection() as conn:
                await conn.fetchval("SELECT 1")
                database_connected = True
        except Exception:
            pass

        # Check message queue connection (from rabbitmq_manager)
        from ..config.rabbitmq import rabbitmq_manager

        message_queue_connected = rabbitmq_manager.is_connected() if hasattr(rabbitmq_manager, "is_connected") else False

        # Check model storage accessibility
        from pathlib import Path
        from ..config.settings import settings

        model_storage_accessible = Path(settings.model_storage_path).exists()

        # Get active models count
        repo = ModelVersionRepository()
        active_count_record = await repo._fetchrow("SELECT COUNT(*) FROM model_versions WHERE is_active = true")
        active_models = active_count_record[0] if active_count_record else 0

        # Check warmup mode
        from ..config.settings import settings

        warmup_mode_enabled = settings.warmup_mode_enabled

        logger.info(
            "Retrieved system health details",
            database_connected=database_connected,
            message_queue_connected=message_queue_connected,
            model_storage_accessible=model_storage_accessible,
        )

        return SystemHealthDetails(
            database_connected=database_connected,
            message_queue_connected=message_queue_connected,
            model_storage_accessible=model_storage_accessible,
            active_models=active_models,
            warmup_mode_enabled=warmup_mode_enabled,
        )
    except Exception as e:
        logger.error("Failed to get system health details", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/strategies/{strategy_id}/performance/time-series", response_model=StrategyPerformanceTimeSeries)
async def get_strategy_performance_time_series(
    strategy_id: str,
    granularity: str = Query("hour", regex="^(hour|day|week)$", description="Time granularity for aggregation"),
    start_time: Optional[str] = Query(None, description="Start time (ISO 8601 format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO 8601 format)"),
) -> StrategyPerformanceTimeSeries:
    """
    Get strategy performance time-series data.

    Args:
        strategy_id: Trading strategy identifier
        granularity: Time granularity (hour, day, week)
        start_time: Start time in ISO 8601 format (default: 7 days ago)
        end_time: End time in ISO 8601 format (default: now)

    Returns:
        Strategy performance time-series data

    Raises:
        HTTPException: If invalid parameters or database error
    """
    try:
        # Parse time range
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        else:
            end_dt = datetime.utcnow()

        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        else:
            # Default to 7 days ago
            start_dt = end_dt - timedelta(days=7)

        if start_dt >= end_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_time must be before end_time",
            )

        # Determine time truncation based on granularity
        truncate_map = {
            "hour": "hour",
            "day": "day",
            "week": "week",
        }
        truncate_expr = truncate_map.get(granularity, "hour")

        # Check if execution_events table exists
        async with db_pool.get_connection() as conn:
            # Check table existence
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'execution_events'
                )
            """
            )

            if not table_exists:
                logger.warning("execution_events table does not exist", strategy_id=strategy_id)
                return StrategyPerformanceTimeSeries(
                    strategy_id=strategy_id,
                    granularity=granularity,
                    start_time=start_dt.isoformat(),
                    end_time=end_dt.isoformat(),
                    data_points=[],
                )

            # Query execution events aggregated by time granularity
            query = f"""
                SELECT 
                    date_trunc('{truncate_expr}', executed_at) as timestamp,
                    COUNT(*) as total_orders,
                    COUNT(*) FILTER (WHERE (performance->>'realized_pnl')::numeric > 0) as successful_orders,
                    AVG((performance->>'realized_pnl')::numeric) as avg_pnl,
                    SUM((performance->>'realized_pnl')::numeric) as total_pnl
                FROM execution_events
                WHERE strategy_id = $1
                    AND executed_at >= $2
                    AND executed_at <= $3
                GROUP BY date_trunc('{truncate_expr}', executed_at)
                ORDER BY timestamp ASC
            """

            records = await conn.fetch(query, strategy_id, start_dt, end_dt)

        # Build response data points
        data_points = []
        for record in records:
            total_orders = record["total_orders"] or 0
            successful_orders = record["successful_orders"] or 0
            success_rate = (successful_orders / total_orders) if total_orders > 0 else 0.0

            data_points.append(
                StrategyPerformanceDataPoint(
                    timestamp=record["timestamp"].isoformat() if hasattr(record["timestamp"], "isoformat") else str(record["timestamp"]),
                    success_rate=success_rate,
                    total_pnl=float(record["total_pnl"] or 0),
                    avg_pnl=float(record["avg_pnl"] or 0),
                    total_orders=total_orders,
                    successful_orders=successful_orders,
                )
            )

        logger.info(
            "Retrieved strategy performance time-series",
            strategy_id=strategy_id,
            granularity=granularity,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            data_points_count=len(data_points),
        )

        return StrategyPerformanceTimeSeries(
            strategy_id=strategy_id,
            granularity=granularity,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            data_points=data_points,
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Invalid time format", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid time format: {str(e)}")
    except Exception as e:
        logger.error("Failed to get strategy performance time-series", strategy_id=strategy_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/monitoring/signals/skip-metrics", response_model=SignalSkipMetricsResponse)
async def get_signal_skip_metrics(
    asset: Optional[str] = Query(None, description="Filter by asset"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
) -> SignalSkipMetricsResponse:
    """
    Get metrics for signal generation skipping.

    Args:
        asset: Optional asset filter
        strategy_id: Optional strategy filter

    Returns:
        Signal skip metrics including counts by asset/strategy and reason
    """
    try:
        metrics = signal_skip_metrics.get_metrics(asset=asset, strategy_id=strategy_id)
        logger.info(
            "Retrieved signal skip metrics",
            total_skips=metrics["total_skips"],
            asset=asset,
            strategy_id=strategy_id,
        )
        return SignalSkipMetricsResponse(**metrics)
    except Exception as e:
        logger.error("Failed to get signal skip metrics", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

