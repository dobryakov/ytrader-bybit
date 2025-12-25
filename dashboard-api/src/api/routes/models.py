"""Model query endpoints."""

from typing import Optional
import json

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...config.settings import settings
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

            # Build metrics dictionary from rows - keep original metric names from DB
            metrics_dict = {}
            if metrics_rows:
                # Get the latest evaluation (first row after ORDER BY DESC)
                # Group by metric_name and take the first value for each metric
                seen_metrics = set()
                for metric_row in metrics_rows:
                    metric_name = metric_row["metric_name"]
                    if metric_name not in seen_metrics:
                        seen_metrics.add(metric_name)
                        # Keep original metric names from database
                        metrics_dict[metric_name] = float(metric_row["metric_value"])

            # Unified metrics structure - all metrics from DB
            # Classification: accuracy, balanced_accuracy, f1_score, pr_auc, precision, recall, roc_auc
            # Regression: mae, mse, r2_score, rmse
            # Trading Performance: avg_pnl, max_drawdown, profit_factor, sharpe_ratio, total_pnl, win_rate
            unified_metrics = {
                # Classification metrics
                "accuracy": metrics_dict.get("accuracy"),
                "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
                "f1_score": metrics_dict.get("f1_score"),
                "pr_auc": metrics_dict.get("pr_auc"),
                "precision": metrics_dict.get("precision"),
                "recall": metrics_dict.get("recall"),
                "roc_auc": metrics_dict.get("roc_auc"),
                # Regression metrics
                "mae": metrics_dict.get("mae"),
                "mse": metrics_dict.get("mse"),
                "r2_score": metrics_dict.get("r2_score"),
                "rmse": metrics_dict.get("rmse"),
                # Trading performance metrics
                "avg_pnl": metrics_dict.get("avg_pnl"),
                "max_drawdown": metrics_dict.get("max_drawdown"),
                "profit_factor": metrics_dict.get("profit_factor"),
                "sharpe_ratio": metrics_dict.get("sharpe_ratio"),
                "total_pnl": metrics_dict.get("total_pnl"),
                "win_rate": metrics_dict.get("win_rate"),
            } if metrics_dict else None

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
                "metrics": unified_metrics,
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


@router.get("/models/history")
async def list_model_training_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results"),
):
    """List all model training history with detailed information."""
    trace_id = get_or_create_trace_id()
    logger.info("model_training_history_request", symbol=symbol, strategy_id=strategy_id, limit=limit, trace_id=trace_id)

    try:
        query = """
            SELECT 
                mv.id, mv.version, mv.model_type, mv.strategy_id, mv.symbol,
                mv.trained_at, mv.training_duration_seconds, mv.training_dataset_size,
                mv.training_config, mv.is_active, mv.created_at
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

        query += " ORDER BY mv.trained_at DESC"
        
        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        rows = await DatabaseConnection.fetch(query, *params)

        history_data = []
        for row in rows:
            # Get quality metrics for this model
            metrics_query = """
                SELECT 
                    metric_name, metric_value, metric_type, evaluated_at, evaluation_dataset_size
                FROM model_quality_metrics
                WHERE model_version_id = $1
                ORDER BY evaluated_at DESC
            """
            metrics_rows = await DatabaseConnection.fetch(metrics_query, row["id"])

            # Parse training_config to extract dataset_id and other info
            training_config = row["training_config"]
            dataset_id = None
            feature_registry_version = None
            target_registry_version = None
            feature_count = None
            
            if isinstance(training_config, str):
                try:
                    training_config = json.loads(training_config)
                except (json.JSONDecodeError, TypeError):
                    training_config = None
            
            if isinstance(training_config, dict):
                dataset_id = training_config.get("dataset_id")
                feature_registry_version = training_config.get("feature_registry_version")
                target_registry_version = training_config.get("target_registry_version")
                feature_count = training_config.get("feature_count")

            # Build metrics dictionary (get latest values) - same format as /models endpoint
            metrics_dict = {}
            if metrics_rows:
                seen_metrics = set()
                for metric_row in metrics_rows:
                    metric_name = metric_row["metric_name"]
                    if metric_name not in seen_metrics:
                        seen_metrics.add(metric_name)
                        # Store as flat dictionary like /models endpoint for consistency
                        metrics_dict[metric_name] = float(metric_row["metric_value"])

            # Unified metrics structure - all metrics from DB (same as /models endpoint)
            # Classification: accuracy, balanced_accuracy, f1_score, pr_auc, precision, recall, roc_auc
            # Regression: mae, mse, r2_score, rmse
            # Trading Performance: avg_pnl, max_drawdown, profit_factor, sharpe_ratio, total_pnl, win_rate
            unified_metrics = {
                # Classification metrics
                "accuracy": metrics_dict.get("accuracy"),
                "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
                "f1_score": metrics_dict.get("f1_score"),
                "pr_auc": metrics_dict.get("pr_auc"),
                "precision": metrics_dict.get("precision"),
                "recall": metrics_dict.get("recall"),
                "roc_auc": metrics_dict.get("roc_auc"),
                # Regression metrics
                "mae": metrics_dict.get("mae"),
                "mse": metrics_dict.get("mse"),
                "r2_score": metrics_dict.get("r2_score"),
                "rmse": metrics_dict.get("rmse"),
                # Trading performance metrics
                "avg_pnl": metrics_dict.get("avg_pnl"),
                "max_drawdown": metrics_dict.get("max_drawdown"),
                "profit_factor": metrics_dict.get("profit_factor"),
                "sharpe_ratio": metrics_dict.get("sharpe_ratio"),
                "total_pnl": metrics_dict.get("total_pnl"),
                "win_rate": metrics_dict.get("win_rate"),
            } if metrics_dict else None

            history_item = {
                "id": str(row["id"]),
                "version": row["version"],
                "model_type": row["model_type"],
                "strategy_id": row["strategy_id"],
                "symbol": row["symbol"],
                "trained_at": row["trained_at"].isoformat() + "Z",
                "training_duration_seconds": row["training_duration_seconds"],
                "training_dataset_size": row["training_dataset_size"],
                "dataset_id": dataset_id,
                "feature_registry_version": feature_registry_version,
                "target_registry_version": target_registry_version,
                "feature_count": feature_count,
                "is_active": row["is_active"],
                "created_at": row["created_at"].isoformat() + "Z",
                "metrics": unified_metrics,
            }
            history_data.append(history_item)

        logger.info("model_training_history_completed", count=len(history_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content=history_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("model_training_history_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model training history: {str(e)}")


@router.post("/training/dataset/build")
async def request_dataset_build(request: Request):
    """Proxy dataset build request to model-service."""
    trace_id = get_or_create_trace_id()
    
    try:
        # Get request body
        body = await request.json()
        logger.info(
            "dataset_build_request",
            symbol=body.get("symbol"),
            strategy_id=body.get("strategy_id"),
            trace_id=trace_id
        )
        
        # Forward request to model-service
        model_service_url = f"http://{settings.model_service_host}:{settings.model_service_port}/api/v1/training/dataset/build"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                model_service_url,
                json=body,
                headers={
                    "X-API-Key": settings.model_service_api_key,
                    "Content-Type": "application/json",
                }
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                "dataset_build_completed",
                dataset_id=result.get("dataset_id"),
                trace_id=trace_id
            )
            
            return JSONResponse(status_code=response.status_code, content=result)
            
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response.text else str(e)
        logger.error(
            "dataset_build_failed",
            status_code=e.response.status_code,
            error=error_detail,
            trace_id=trace_id
        )
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        logger.error("dataset_build_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to request dataset build: {str(e)}")

