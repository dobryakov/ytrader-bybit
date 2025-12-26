"""
Model versions API endpoints.

Provides REST API endpoints for model version management:
- List model versions with filtering and pagination
- Get model version details with quality metrics
- Activate/deactivate model versions
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
import json
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..database.repositories.model_prediction_repo import ModelPredictionRepository
from ..services.model_version_manager import model_version_manager
from ..config.logging import get_logger
from ..config.settings import settings
from .middleware.security import validate_version_string

logger = get_logger(__name__)

router = APIRouter(tags=["models"])


class ModelVersionResponse(BaseModel):
    """Model version response model."""

    id: str
    version: str
    file_path: str
    model_type: str
    strategy_id: Optional[str] = None
    trained_at: str
    training_duration_seconds: Optional[int] = None
    training_dataset_size: Optional[int] = None
    training_config: Optional[dict] = None
    is_active: bool
    is_warmup_mode: bool
    auto_activation_disabled: bool = False
    created_at: str
    updated_at: str


class ModelVersionListResponse(BaseModel):
    """Model version list response with pagination."""

    items: List[ModelVersionResponse]
    total: int
    limit: Optional[int] = None
    offset: int


class ConfidenceThresholdInfo(BaseModel):
    """Information about confidence threshold used for signal generation."""
    
    threshold_value: float
    threshold_source: str  # 'top_k' or 'static'
    top_k_percentage: Optional[int] = None  # Only set if source is 'top_k'
    static_threshold: Optional[float] = None  # Only set if source is 'static'
    metric_name: Optional[str] = None  # e.g., 'top_k_10_confidence_threshold'


class ConfidenceThresholdInfo(BaseModel):
    """Information about confidence threshold used for signal generation."""
    
    threshold_value: float
    threshold_source: str  # 'top_k' or 'static'
    top_k_percentage: Optional[int] = None  # Only set if source is 'top_k'
    static_threshold: Optional[float] = None  # Only set if source is 'static'
    metric_name: Optional[str] = None  # e.g., 'top_k_10_confidence_threshold'


class ModelVersionDetailResponse(ModelVersionResponse):
    """Model version detail response with quality metrics."""

    quality_metrics: List[dict] = Field(default_factory=list)
    confidence_threshold_info: Optional[ConfidenceThresholdInfo] = None


class ModelActivationRequest(BaseModel):
    """Model activation request model."""

    strategy_id: Optional[str] = None


class TopKMetrics(BaseModel):
    """Top-k metrics for a specific k value."""

    k: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    lift: Optional[float] = None
    coverage: Optional[float] = None
    precision_class_1: Optional[float] = None
    recall_class_1: Optional[float] = None
    f1_class_1: Optional[float] = None


class BaselineMetrics(BaseModel):
    """Baseline metrics (majority class strategy)."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


class ModelMetrics(BaseModel):
    """Main model metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


class PredictionInfo(BaseModel):
    """Information about saved predictions."""

    split: str
    count: int
    dataset_id: Optional[str] = None
    created_at: Optional[str] = None


class ModelAnalysisResponse(BaseModel):
    """Detailed model analysis response."""

    model_version: str
    model_id: str
    predictions: List[PredictionInfo]
    model_metrics: ModelMetrics
    baseline_metrics: BaselineMetrics
    top_k_metrics: List[TopKMetrics]
    comparison: Dict[str, Any]
    confidence_threshold_info: Optional[ConfidenceThresholdInfo] = None
    optimal_top_k_percentage: Optional[int] = Field(
        default=None,
        description="Optimal top-k percentage selected for this model (from training_config)"
    )


@router.get("/models", response_model=ModelVersionListResponse)
async def list_model_versions(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> ModelVersionListResponse:
    """
    List model versions with filtering and pagination.

    Args:
        strategy_id: Filter by trading strategy identifier
        is_active: Filter by active status
        limit: Maximum number of results (1-1000)
        offset: Number of results to skip

    Returns:
        Model version list with pagination metadata
    """
    try:
        repo = ModelVersionRepository()

        # Build query conditions
        conditions = []
        params = []
        param_index = 1

        if strategy_id is not None:
            conditions.append(f"strategy_id = ${param_index}")
            params.append(strategy_id)
            param_index += 1

        if is_active is not None:
            conditions.append(f"is_active = ${param_index}")
            params.append(is_active)
            param_index += 1

        # Get total count
        count_query = f"SELECT COUNT(*) FROM model_versions"
        if conditions:
            count_query += f" WHERE {' AND '.join(conditions)}"
        count_record = await repo._fetchrow(count_query, *params)
        total = count_record[0] if count_record else 0

        # Get paginated results
        query = "SELECT * FROM model_versions"
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        query += " ORDER BY trained_at DESC"
        query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
        params.extend([limit, offset])

        records = await repo._fetch(query, *params)
        items = [ModelVersionResponse(**repo._record_to_dict(record)) for record in records]

        logger.info(
            "Listed model versions",
            strategy_id=strategy_id,
            is_active=is_active,
            count=len(items),
            total=total,
        )

        return ModelVersionListResponse(items=items, total=total, limit=limit, offset=offset)
    except Exception as e:
        logger.error("Failed to list model versions", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/{version}", response_model=ModelVersionDetailResponse)
async def get_model_version_details(version: str) -> ModelVersionDetailResponse:
    """
    Get model version details with quality metrics.

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')

    Returns:
        Model version details with quality metrics

    Raises:
        HTTPException: If model version not found
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        repo = ModelVersionRepository()
        model_version = await repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        # Get quality metrics
        metrics_repo = ModelQualityMetricsRepository()
        quality_metrics = await metrics_repo.get_by_model_version(UUID(model_version["id"]))

        # Get confidence threshold information
        confidence_threshold_info = await _get_confidence_threshold_info(
            model_version_id=UUID(model_version["id"]),
            version=version,
        )

        logger.info("Retrieved model version details", version=version, metrics_count=len(quality_metrics))

        return ModelVersionDetailResponse(
            **model_version,
            quality_metrics=quality_metrics,
            confidence_threshold_info=confidence_threshold_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model version details", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def _get_confidence_threshold_info(
    model_version_id: UUID,
    version: str,
) -> Optional[ConfidenceThresholdInfo]:
    """
    Get confidence threshold information for a model version.
    
    Tries to get top-k confidence threshold from quality metrics.
    Falls back to static threshold from settings if not available.
    
    Args:
        model_version_id: Model version UUID
        version: Model version string (for logging)
        
    Returns:
        ConfidenceThresholdInfo or None if model is not active
    """
    try:
        # Get model version to check for optimal_top_k_percentage in training_config
        version_repo = ModelVersionRepository()
        model_version = await version_repo.get_by_id(model_version_id)
        
        # Get top-k percentage: first try from model's training_config (optimal for this model),
        # then fallback to settings (global default)
        top_k_percentage = None
        
        if model_version and model_version.get("training_config"):
            training_config = model_version["training_config"]
            if isinstance(training_config, str):
                try:
                    training_config = json.loads(training_config)
                except (json.JSONDecodeError, TypeError):
                    training_config = None
            
            if training_config and isinstance(training_config, dict):
                optimal_k = training_config.get("optimal_top_k_percentage")
                if optimal_k is not None:
                    try:
                        top_k_percentage = int(optimal_k)
                        logger.debug(
                            "Using optimal top-k percentage from model training_config",
                            version=version,
                            optimal_top_k_percentage=top_k_percentage,
                        )
                    except (ValueError, TypeError):
                        top_k_percentage = None
        
        # Fallback to settings if not found in training_config
        if top_k_percentage is None:
            top_k_percentage = getattr(settings, "model_signal_top_k_percentage", 10)
            logger.debug(
                "Using top-k percentage from settings (not found in model training_config)",
                version=version,
                top_k_percentage=top_k_percentage,
            )
        
        static_threshold = getattr(settings, "model_activation_threshold", 0.75)
        
        # Try to get top-k confidence threshold from quality metrics
        metrics_repo = ModelQualityMetricsRepository()
        metric_name = f"top_k_{top_k_percentage}_confidence_threshold"
        
        logger.debug(
            "Fetching top-k confidence threshold from quality metrics",
            version=version,
            model_version_id=str(model_version_id),
            metric_name=metric_name,
            top_k_percentage=top_k_percentage,
        )
        
        metrics = await metrics_repo.get_by_model_version(
            model_version_id=model_version_id,
            metric_name=metric_name,
            dataset_split="test",
        )
        
        logger.debug(
            "Retrieved metrics from repository",
            version=version,
            metrics_count=len(metrics) if metrics else 0,
            metric_name=metric_name,
        )
        
        if metrics and len(metrics) > 0:
            # Get the latest metric (first in list, sorted by evaluated_at DESC)
            threshold_value = metrics[0].get("metric_value")
            logger.debug(
                "Processing threshold value",
                version=version,
                threshold_value=threshold_value,
                threshold_value_type=type(threshold_value).__name__,
            )
            if threshold_value is not None and isinstance(threshold_value, (int, float)):
                threshold = float(threshold_value)
                if 0.0 <= threshold <= 1.0:
                    logger.info(
                        "Found top-k confidence threshold in quality metrics",
                        version=version,
                        top_k_percentage=top_k_percentage,
                        threshold=threshold,
                    )
                    return ConfidenceThresholdInfo(
                        threshold_value=threshold,
                        threshold_source="top_k",
                        top_k_percentage=top_k_percentage,
                        metric_name=metric_name,
                    )
                else:
                    logger.warning(
                        "Threshold value out of range [0.0, 1.0]",
                        version=version,
                        threshold=threshold,
                    )
            else:
                logger.warning(
                    "Threshold value is None or invalid type",
                    version=version,
                    threshold_value=threshold_value,
                    threshold_value_type=type(threshold_value).__name__ if threshold_value is not None else None,
                )
        else:
            logger.debug(
                "No metrics found for top-k confidence threshold",
                version=version,
                metric_name=metric_name,
            )
        
        # Fallback to static threshold
        logger.info(
            "Using static confidence threshold (top-k threshold not found or invalid)",
            version=version,
            static_threshold=static_threshold,
        )
        return ConfidenceThresholdInfo(
            threshold_value=static_threshold,
            threshold_source="static",
            static_threshold=static_threshold,
        )
    except Exception as e:
        logger.warning(
            "Failed to get confidence threshold info",
            version=version,
            error=str(e),
            exc_info=True,
        )
        # Return static threshold as fallback
        static_threshold = getattr(settings, "model_activation_threshold", 0.75)
        return ConfidenceThresholdInfo(
            threshold_value=static_threshold,
            threshold_source="static",
            static_threshold=static_threshold,
        )


@router.post("/models/{version}/activate", response_model=ModelVersionResponse)
async def activate_model_version(
    version: str,
    request: Optional[ModelActivationRequest] = None,
) -> ModelVersionResponse:
    """
    Activate a model version (deactivates previous active model for the strategy).

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')
        request: Optional activation request with strategy_id

    Returns:
        Activated model version record

    Raises:
        HTTPException: If model version not found or activation fails
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        repo = ModelVersionRepository()
        model_version = await repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found for activation", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        strategy_id = request.strategy_id if request else model_version.get("strategy_id")
        activated_version = await model_version_manager.activate_version(UUID(model_version["id"]), strategy_id)

        if not activated_version:
            logger.error("Failed to activate model version", version=version)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to activate model version {version}",
            )

        logger.info("Model version activated", version=version, strategy_id=strategy_id)

        return ModelVersionResponse(**activated_version)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to activate model version", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/{version}/deactivate", response_model=ModelVersionResponse)
async def deactivate_model_version(version: str) -> ModelVersionResponse:
    """
    Deactivate a model version and block automatic reactivation.

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')

    Returns:
        Deactivated model version record

    Raises:
        HTTPException: If model version not found or deactivation fails
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        repo = ModelVersionRepository()
        model_version = await repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found for deactivation", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        deactivated_version = await model_version_manager.deactivate_version(UUID(model_version["id"]))

        if not deactivated_version:
            logger.error("Failed to deactivate model version", version=version)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to deactivate model version {version}",
            )

        logger.info("Model version deactivated", version=version)

        return ModelVersionResponse(**deactivated_version)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to deactivate model version", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/{version}/analysis", response_model=ModelAnalysisResponse)
async def get_model_analysis(version: str) -> ModelAnalysisResponse:
    """
    Get detailed analysis for a model version including predictions, baseline, and top-k metrics.

    Returns:
    - Predictions information (count, split)
    - Model metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
    - Baseline metrics (majority class strategy)
    - Top-k metrics for k=10,20,30,50
    - Comparison between model and baseline

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')

    Returns:
        Detailed model analysis with all metrics

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

        model_id = UUID(model_version["id"])
        metrics_repo = ModelQualityMetricsRepository()
        pred_repo = ModelPredictionRepository()

        # Get predictions
        predictions_data = await pred_repo.get_by_model_version(version)
        predictions_info = []
        for pred in predictions_data:
            # Get count from predictions JSONB array length
            # predictions is stored as JSONB, need to parse it
            predictions_json = pred.get("predictions")
            if isinstance(predictions_json, str):
                predictions_json = json.loads(predictions_json)
            pred_count = len(predictions_json) if isinstance(predictions_json, list) else 0
            # Handle created_at - it might be datetime or string
            created_at = pred.get("created_at")
            if created_at:
                if hasattr(created_at, "isoformat"):
                    created_at = created_at.isoformat()
                elif isinstance(created_at, str):
                    created_at = created_at
                else:
                    created_at = str(created_at)
            else:
                created_at = None
            
            predictions_info.append(
                PredictionInfo(
                    split=pred.get("split", ""),
                    count=pred_count,
                    dataset_id=str(pred.get("dataset_id", "")),
                    created_at=created_at,
                )
            )

        # Get all metrics
        all_metrics = await metrics_repo.get_by_model_version(model_id, dataset_split="test")

        # Extract model metrics (without baseline_ and top_k_ prefixes)
        model_metrics_dict = {}
        baseline_metrics_dict = {}
        top_k_metrics_dict = {}

        for metric in all_metrics:
            name = metric.get("metric_name", "")
            value = metric.get("metric_value")
            
            if name.startswith("baseline_"):
                # Remove baseline_ prefix
                key = name.replace("baseline_", "")
                baseline_metrics_dict[key] = value
            elif name.startswith("top_k_"):
                # Parse top_k_{k}_{metric_name}
                parts = name.split("_")
                if len(parts) >= 3 and parts[2].isdigit():
                    k = int(parts[2])
                    metric_name = "_".join(parts[3:])
                    if k not in top_k_metrics_dict:
                        top_k_metrics_dict[k] = {}
                    top_k_metrics_dict[k][metric_name] = value
            else:
                # Regular model metric
                model_metrics_dict[name] = value

        # Build response models
        model_metrics = ModelMetrics(
            accuracy=model_metrics_dict.get("accuracy"),
            precision=model_metrics_dict.get("precision"),
            recall=model_metrics_dict.get("recall"),
            f1_score=model_metrics_dict.get("f1_score"),
            balanced_accuracy=model_metrics_dict.get("balanced_accuracy"),
            roc_auc=model_metrics_dict.get("roc_auc"),
            pr_auc=model_metrics_dict.get("pr_auc"),
        )

        baseline_metrics = BaselineMetrics(
            accuracy=baseline_metrics_dict.get("accuracy"),
            precision=baseline_metrics_dict.get("precision"),
            recall=baseline_metrics_dict.get("recall"),
            f1_score=baseline_metrics_dict.get("f1_score"),
            balanced_accuracy=baseline_metrics_dict.get("balanced_accuracy"),
            roc_auc=baseline_metrics_dict.get("roc_auc"),
            pr_auc=baseline_metrics_dict.get("pr_auc"),
        )

        # Build top-k metrics
        top_k_list = []
        for k in sorted(top_k_metrics_dict.keys()):
            k_metrics = top_k_metrics_dict[k]
            top_k_list.append(
                TopKMetrics(
                    k=k,
                    accuracy=k_metrics.get("accuracy"),
                    precision=k_metrics.get("precision"),
                    recall=k_metrics.get("recall"),
                    f1_score=k_metrics.get("f1_score"),
                    balanced_accuracy=k_metrics.get("balanced_accuracy"),
                    roc_auc=k_metrics.get("roc_auc"),
                    pr_auc=k_metrics.get("pr_auc"),
                    lift=k_metrics.get("lift"),
                    coverage=k_metrics.get("coverage"),
                    precision_class_1=k_metrics.get("precision_class_1"),
                    recall_class_1=k_metrics.get("recall_class_1"),
                    f1_class_1=k_metrics.get("f1_class_1"),
                )
            )

        # Build comparison
        def calc_diff(model_val, baseline_val):
            """Calculate difference between model and baseline values."""
            if model_val is not None and baseline_val is not None:
                return model_val - baseline_val
            return None
        
        comparison = {
            "accuracy": {
                "model": model_metrics.accuracy,
                "baseline": baseline_metrics.accuracy,
                "difference": calc_diff(model_metrics.accuracy, baseline_metrics.accuracy),
            },
            "f1_score": {
                "model": model_metrics.f1_score,
                "baseline": baseline_metrics.f1_score,
                "difference": calc_diff(model_metrics.f1_score, baseline_metrics.f1_score),
            },
            "pr_auc": {
                "model": model_metrics.pr_auc,
                "baseline": baseline_metrics.pr_auc,
                "difference": calc_diff(model_metrics.pr_auc, baseline_metrics.pr_auc),
            },
            "roc_auc": {
                "model": model_metrics.roc_auc,
                "baseline": baseline_metrics.roc_auc,
                "difference": calc_diff(model_metrics.roc_auc, baseline_metrics.roc_auc),
            },
        }

        # Get confidence threshold information
        confidence_threshold_info = await _get_confidence_threshold_info(
            model_version_id=model_id,
            version=version,
        )

        # Extract optimal_top_k_percentage from training_config
        optimal_top_k_percentage = None
        training_config = model_version.get("training_config")
        if training_config:
            if isinstance(training_config, str):
                try:
                    training_config = json.loads(training_config)
                except (json.JSONDecodeError, TypeError):
                    training_config = None
            
            if isinstance(training_config, dict):
                optimal_k = training_config.get("optimal_top_k_percentage")
                if optimal_k is not None:
                    try:
                        optimal_top_k_percentage = int(optimal_k)
                    except (ValueError, TypeError):
                        optimal_top_k_percentage = None

        logger.info(
            "Retrieved model analysis",
            version=version,
            predictions_count=len(predictions_info),
            top_k_count=len(top_k_list),
            optimal_top_k_percentage=optimal_top_k_percentage,
        )

        return ModelAnalysisResponse(
            model_version=version,
            model_id=str(model_id),
            predictions=predictions_info,
            model_metrics=model_metrics,
            baseline_metrics=baseline_metrics,
            top_k_metrics=top_k_list,
            comparison=comparison,
            confidence_threshold_info=confidence_threshold_info,
            optimal_top_k_percentage=optimal_top_k_percentage,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model analysis", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

