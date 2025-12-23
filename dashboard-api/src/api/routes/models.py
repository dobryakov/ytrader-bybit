"""Model query endpoints."""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
):
    """List model versions with optional filtering."""
    trace_id = get_or_create_trace_id()
    logger.info("model_list_request", symbol=symbol, strategy_id=strategy_id, trace_id=trace_id)

    try:
        query = """
            SELECT 
                mv.id, mv.version, mv.file_path, mv.model_type, mv.strategy_id,
                mv.symbol, mv.trained_at, mv.training_duration_seconds,
                mv.training_dataset_size, mv.training_config, mv.is_active
            FROM model_versions mv
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if symbol:
            query += f" AND mv.symbol = ${param_idx}"
            params.append(symbol)
            param_idx += 1

        if strategy_id:
            query += f" AND mv.strategy_id = ${param_idx}"
            params.append(strategy_id)
            param_idx += 1

        if is_active is not None:
            query += f" AND mv.is_active = ${param_idx}"
            params.append(is_active)
            param_idx += 1

        query += " ORDER BY mv.trained_at DESC"

        rows = await DatabaseConnection.fetch(query, *params)

        models_data = []
        for row in rows:
            # Get quality metrics for this model
            # Metrics are stored as rows with metric_name and metric_value columns
            metrics_query = """
                SELECT 
                    metric_name, metric_value
                FROM model_quality_metrics
                WHERE model_version_id = $1
                ORDER BY evaluated_at DESC
            """
            metrics_rows = await DatabaseConnection.fetch(metrics_query, row["id"])

            # Build metrics dictionary from rows
            metrics_dict = {}
            if metrics_rows:
                # Get the latest evaluation (first row after ORDER BY DESC)
                # Group by metric_name and take the first value for each metric
                seen_metrics = set()
                for metric_row in metrics_rows:
                    metric_name = metric_row["metric_name"]
                    if metric_name not in seen_metrics:
                        seen_metrics.add(metric_name)
                        # Map metric names to API field names
                        api_field_name = metric_name  # Use same name by default
                        if metric_name == "roc_auc":
                            api_field_name = "roc_auc_score"
                        elif metric_name == "pr_auc":
                            api_field_name = "pr_auc_score"
                        elif metric_name == "balanced_accuracy":
                            api_field_name = "balanced_accuracy"
                        
                        metrics_dict[api_field_name] = float(metric_row["metric_value"])

            model_dict = {
                "id": str(row["id"]),
                "version": row["version"],
                "file_path": row["file_path"],
                "model_type": row["model_type"],
                "strategy_id": row["strategy_id"],
                "symbol": row["symbol"],
                "trained_at": row["trained_at"].isoformat() + "Z",
                "training_duration_seconds": row["training_duration_seconds"],
                "training_dataset_size": row["training_dataset_size"],
                "training_config": row["training_config"],
                "is_active": row["is_active"],
                "metrics": {
                    "f1_score": metrics_dict.get("f1_score"),
                    "precision_score": metrics_dict.get("precision_score"),
                    "recall_score": metrics_dict.get("recall_score"),
                    "accuracy_score": metrics_dict.get("accuracy_score"),
                    "roc_auc_score": metrics_dict.get("roc_auc_score"),
                    "pr_auc_score": metrics_dict.get("pr_auc_score"),
                    "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
                } if metrics_dict else None,
            }
            models_data.append(model_dict)

        logger.info("model_list_completed", count=len(models_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "models": models_data,
                "count": len(models_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("model_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")

