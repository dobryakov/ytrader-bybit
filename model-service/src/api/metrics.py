"""
Model quality metrics API endpoints.

Provides REST API endpoints for model quality metrics:
- Get quality metrics for a model version
- Get time-series metrics data for charting
"""

from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..config.logging import get_logger
from .middleware.security import validate_version_string

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


class MetricResponse(BaseModel):
    """Quality metric response model."""

    id: str
    model_version_id: str
    metric_name: str
    metric_value: float
    metric_type: str
    evaluated_at: str
    evaluation_dataset_size: Optional[int] = None
    metadata: Optional[dict] = None


class TimeSeriesDataPoint(BaseModel):
    """Time-series data point."""

    timestamp: str
    value: float
    metric_name: str


class TimeSeriesResponse(BaseModel):
    """Time-series metrics response."""

    model_version: str
    metric_name: Optional[str] = None
    granularity: str
    start_time: str
    end_time: str
    data_points: List[TimeSeriesDataPoint]


@router.get("/models/{version}/metrics", response_model=List[MetricResponse])
async def get_model_quality_metrics(
    version: str,
    metric_type: Optional[str] = Query(None, description="Filter by metric type (classification, regression, trading_performance)"),
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
) -> List[MetricResponse]:
    """
    Get quality metrics for a model version.

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')
        metric_type: Filter by metric type
        metric_name: Filter by metric name

    Returns:
        List of quality metrics

    Raises:
        HTTPException: If model version not found
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        version_repo = ModelVersionRepository()
        model_version = await version_repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        metrics_repo = ModelQualityMetricsRepository()
        metrics = await metrics_repo.get_by_model_version(
            UUID(model_version["id"]),
            metric_name=metric_name,
            metric_type=metric_type,
        )

        logger.info(
            "Retrieved quality metrics",
            version=version,
            metric_type=metric_type,
            metric_name=metric_name,
            count=len(metrics),
        )

        return [MetricResponse(**metric) for metric in metrics]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get quality metrics", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/{version}/metrics/time-series", response_model=TimeSeriesResponse)
async def get_time_series_metrics(
    version: str,
    granularity: str = Query("hour", regex="^(hour|day|week)$", description="Time granularity for aggregation"),
    start_time: Optional[str] = Query(None, description="Start time (ISO 8601 format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO 8601 format)"),
    metric_names: Optional[str] = Query(None, description="Comma-separated list of metric names to include"),
) -> TimeSeriesResponse:
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    """
    Get time-series metrics data for a model version.

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')
        granularity: Time granularity (hour, day, week)
        start_time: Start time in ISO 8601 format (default: 7 days ago)
        end_time: End time in ISO 8601 format (default: now)
        metric_names: Comma-separated list of metric names to include

    Returns:
        Time-series metrics data

    Raises:
        HTTPException: If model version not found or invalid parameters
    """
    try:
        version_repo = ModelVersionRepository()
        model_version = await version_repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

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

        # Parse metric names
        metric_name_list = None
        if metric_names:
            metric_name_list = [name.strip() for name in metric_names.split(",")]

        # Determine time truncation based on granularity
        truncate_map = {
            "hour": "hour",
            "day": "day",
            "week": "week",
        }
        truncate_expr = truncate_map.get(granularity, "hour")

        metrics_repo = ModelQualityMetricsRepository()

        # Build query for time-series data
        conditions = ["model_version_id = $1", "evaluated_at >= $2", "evaluated_at <= $3"]
        params = [UUID(model_version["id"]), start_dt, end_dt]
        param_index = 4

        if metric_name_list:
            # Use ANY for array matching
            conditions.append(f"metric_name = ANY(${param_index})")
            params.append(metric_name_list)
            param_index += 1

        # Aggregate by time granularity
        query = f"""
            SELECT 
                date_trunc('{truncate_expr}', evaluated_at) as timestamp,
                metric_name,
                AVG(metric_value) as value
            FROM model_quality_metrics
            WHERE {' AND '.join(conditions)}
            GROUP BY date_trunc('{truncate_expr}', evaluated_at), metric_name
            ORDER BY timestamp ASC, metric_name ASC
        """

        records = await metrics_repo._fetch(query, *params)

        # Build response data points
        data_points = []
        for record in records:
            data_points.append(
                TimeSeriesDataPoint(
                    timestamp=record["timestamp"].isoformat() if hasattr(record["timestamp"], "isoformat") else str(record["timestamp"]),
                    value=float(record["value"]),
                    metric_name=record["metric_name"],
                )
            )

        logger.info(
            "Retrieved time-series metrics",
            version=version,
            granularity=granularity,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            data_points_count=len(data_points),
        )

        return TimeSeriesResponse(
            model_version=version,
            metric_name=metric_names,
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
        logger.error("Failed to get time-series metrics", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

