"""Model query endpoints."""

from typing import Optional
import json
import uuid

from fastapi import APIRouter, Query, HTTPException, Request, Body
from fastapi.responses import JSONResponse
import httpx
import aio_pika

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


@router.get("/models/signal-success-rate")
async def get_signal_success_rate(
    model_version: str = Query(..., description="Model version (e.g., 'v1.0')"),
    asset: str = Query(..., description="Trading asset (e.g., 'BTCUSDT')"),
    strategy_id: str = Query(..., description="Strategy ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format, optional)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format, optional)"),
):
    """Get signal success rate statistics grouped by hour."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "signal_success_rate_request",
        model_version=model_version,
        asset=asset,
        strategy_id=strategy_id,
        trace_id=trace_id
    )

    try:
        query = """
            SELECT 
                DATE_TRUNC('hour', ts.timestamp) as hour,
                COUNT(*) as total_signals,
                COUNT(CASE WHEN pt.actual_values IS NOT NULL AND pt.actual_values != '{}'::jsonb THEN 1 END) as evaluated_signals,
                COUNT(CASE 
                    WHEN pt.predicted_values->>'direction' IS NOT NULL 
                     AND pt.actual_values->>'direction' IS NOT NULL
                     AND pt.predicted_values->>'direction' = pt.actual_values->>'direction'
                    THEN 1 
                END) as successful_by_direction,
                COUNT(CASE 
                    WHEN ptr.total_pnl IS NOT NULL AND ptr.total_pnl > 0 
                    THEN 1 
                END) as successful_by_pnl,
                ROUND(
                    100.0 * COUNT(CASE 
                        WHEN pt.predicted_values->>'direction' IS NOT NULL 
                         AND pt.actual_values->>'direction' IS NOT NULL
                         AND pt.predicted_values->>'direction' = pt.actual_values->>'direction'
                        THEN 1 
                    END)::numeric / 
                    NULLIF(COUNT(CASE 
                        WHEN pt.predicted_values->>'direction' IS NOT NULL 
                         AND pt.actual_values->>'direction' IS NOT NULL
                        THEN 1 
                    END), 0),
                    2
                ) as success_rate_direction_percent,
                ROUND(
                    100.0 * COUNT(CASE 
                        WHEN ptr.total_pnl IS NOT NULL AND ptr.total_pnl > 0 
                        THEN 1 
                    END)::numeric / 
                    NULLIF(COUNT(CASE WHEN ptr.total_pnl IS NOT NULL THEN 1 END), 0),
                    2
                ) as success_rate_pnl_percent,
                AVG(ts.confidence) as avg_confidence,
                COUNT(CASE WHEN ts.side = 'buy' THEN 1 END) as buy_signals,
                COUNT(CASE WHEN ts.side = 'sell' THEN 1 END) as sell_signals,
                SUM(COALESCE(ptr.total_pnl, 0)) as total_pnl_sum,
                AVG(ptr.total_pnl) FILTER (WHERE ptr.total_pnl IS NOT NULL) as avg_pnl
            FROM trading_signals ts
            INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
            LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
            WHERE 
                ts.model_version = $1
                AND ts.asset = $2
                AND ts.strategy_id = $3
                AND pt.actual_values IS NOT NULL
                AND pt.actual_values != '{}'::jsonb
                AND pt.actual_values_computed_at IS NOT NULL
                AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
        """
        params = [model_version, asset, strategy_id]
        param_idx = 4

        if start_date:
            query += f" AND ts.timestamp >= ${param_idx}::timestamptz"
            params.append(start_date)
            param_idx += 1

        if end_date:
            query += f" AND ts.timestamp <= ${param_idx}::timestamptz"
            params.append(end_date)
            param_idx += 1

        query += """
            GROUP BY 
                DATE_TRUNC('hour', ts.timestamp),
                ts.model_version,
                ts.asset,
                ts.strategy_id
            ORDER BY 
                hour DESC
        """

        rows = await DatabaseConnection.fetch(query, *params)

        result_data = []
        for row in rows:
            result_data.append({
                "hour": row["hour"].isoformat() + "Z" if row["hour"] else None,
                "total_signals": row["total_signals"],
                "evaluated_signals": row["evaluated_signals"],
                "successful_by_direction": row["successful_by_direction"],
                "successful_by_pnl": row["successful_by_pnl"],
                "success_rate_direction_percent": float(row["success_rate_direction_percent"]) if row["success_rate_direction_percent"] is not None else None,
                "success_rate_pnl_percent": float(row["success_rate_pnl_percent"]) if row["success_rate_pnl_percent"] is not None else None,
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] is not None else None,
                "buy_signals": row["buy_signals"],
                "sell_signals": row["sell_signals"],
                "total_pnl_sum": float(row["total_pnl_sum"]) if row["total_pnl_sum"] is not None else None,
                "avg_pnl": float(row["avg_pnl"]) if row["avg_pnl"] is not None else None,
            })

        logger.info("signal_success_rate_completed", count=len(result_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "data": result_data,
                "count": len(result_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("signal_success_rate_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signal success rate: {str(e)}")


@router.get("/models/available-assets")
async def get_available_assets():
    """Get list of unique assets from trading signals."""
    trace_id = get_or_create_trace_id()
    logger.info("available_assets_request", trace_id=trace_id)

    try:
        query = """
            SELECT DISTINCT asset
            FROM trading_signals
            WHERE asset IS NOT NULL
            ORDER BY asset
        """
        rows = await DatabaseConnection.fetch(query)
        assets = [row["asset"] for row in rows]

        logger.info("available_assets_completed", count=len(assets), trace_id=trace_id)
        return JSONResponse(status_code=200, content={"assets": assets})

    except Exception as e:
        logger.error("available_assets_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available assets: {str(e)}")


@router.get("/models/available-strategies")
async def get_available_strategies():
    """Get list of unique strategy IDs from trading signals."""
    trace_id = get_or_create_trace_id()
    logger.info("available_strategies_request", trace_id=trace_id)

    try:
        query = """
            SELECT DISTINCT strategy_id
            FROM trading_signals
            WHERE strategy_id IS NOT NULL
            ORDER BY strategy_id
        """
        rows = await DatabaseConnection.fetch(query)
        strategies = [row["strategy_id"] for row in rows]

        logger.info("available_strategies_completed", count=len(strategies), trace_id=trace_id)
        return JSONResponse(status_code=200, content={"strategies": strategies})

    except Exception as e:
        logger.error("available_strategies_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available strategies: {str(e)}")


@router.get("/models/by-dataset/{dataset_id}")
async def get_models_by_dataset(dataset_id: str):
    """Get models trained on a specific dataset."""
    trace_id = get_or_create_trace_id()
    logger.info("models_by_dataset_request", dataset_id=dataset_id, trace_id=trace_id)
    
    try:
        query = """
            SELECT 
                mv.id, mv.version, mv.file_path, mv.model_type, mv.strategy_id,
                mv.symbol, mv.trained_at, mv.training_duration_seconds,
                mv.training_dataset_size, mv.training_config, mv.is_active
            FROM model_versions mv
            WHERE mv.training_config->>'dataset_id' = $1
            ORDER BY mv.trained_at DESC
        """
        rows = await DatabaseConnection.fetch(query, dataset_id)
        
        models_data = []
        for row in rows:
            # Parse training_config to verify dataset_id matches
            training_config = row["training_config"]
            if isinstance(training_config, str):
                try:
                    training_config = json.loads(training_config)
                except (json.JSONDecodeError, TypeError):
                    training_config = None
            
            # Verify dataset_id matches
            if isinstance(training_config, dict):
                config_dataset_id = training_config.get("dataset_id")
                if config_dataset_id != dataset_id:
                    continue
            
            # Get quality metrics for this model
            metrics_query = """
                SELECT 
                    metric_name, metric_value
                FROM model_quality_metrics
                WHERE model_version_id = $1
                ORDER BY evaluated_at DESC
            """
            metrics_rows = await DatabaseConnection.fetch(metrics_query, row["id"])
            
            # Build metrics dictionary
            metrics_dict = {}
            if metrics_rows:
                seen_metrics = set()
                for metric_row in metrics_rows:
                    metric_name = metric_row["metric_name"]
                    if metric_name not in seen_metrics:
                        seen_metrics.add(metric_name)
                        metrics_dict[metric_name] = float(metric_row["metric_value"])
            
            # Unified metrics structure
            unified_metrics = {
                "accuracy": metrics_dict.get("accuracy"),
                "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
                "f1_score": metrics_dict.get("f1_score"),
                "pr_auc": metrics_dict.get("pr_auc"),
                "precision": metrics_dict.get("precision"),
                "recall": metrics_dict.get("recall"),
                "roc_auc": metrics_dict.get("roc_auc"),
                "mae": metrics_dict.get("mae"),
                "mse": metrics_dict.get("mse"),
                "r2_score": metrics_dict.get("r2_score"),
                "rmse": metrics_dict.get("rmse"),
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
                "training_config": training_config,
                "is_active": row["is_active"],
                "metrics": unified_metrics,
            }
            models_data.append(model_dict)
        
        logger.info("models_by_dataset_completed", dataset_id=dataset_id, count=len(models_data), trace_id=trace_id)
        
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
        logger.error("models_by_dataset_failed", dataset_id=dataset_id, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get("/models/active-version")
async def get_active_model_version(
    asset: str = Query(..., description="Trading asset (e.g., 'BTCUSDT')"),
    strategy_id: str = Query(..., description="Strategy ID"),
):
    """Get active model version for a specific asset and strategy."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "active_model_version_request",
        asset=asset,
        strategy_id=strategy_id,
        trace_id=trace_id
    )

    try:
        query = """
            SELECT version
            FROM model_versions
            WHERE symbol = $1
              AND strategy_id = $2
              AND is_active = true
            ORDER BY trained_at DESC
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, asset, strategy_id)

        if row:
            version = row["version"]
            logger.info(
                "active_model_version_found",
                asset=asset,
                strategy_id=strategy_id,
                version=version,
                trace_id=trace_id
            )
            return JSONResponse(status_code=200, content={"version": version})
        else:
            logger.info(
                "active_model_version_not_found",
                asset=asset,
                strategy_id=strategy_id,
                trace_id=trace_id
            )
            return JSONResponse(status_code=200, content={"version": None})

    except Exception as e:
        logger.error("active_model_version_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve active model version: {str(e)}")


@router.get("/models/{version}/analysis")
async def get_model_analysis(version: str):
    """Proxy model analysis request to model-service."""
    trace_id = get_or_create_trace_id()
    logger.info("model_analysis_request", version=version, trace_id=trace_id)
    
    try:
        # Forward request to model-service
        model_service_url = f"http://{settings.model_service_host}:{settings.model_service_port}/api/v1/models/{version}/analysis"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                model_service_url,
                headers={
                    "X-API-Key": settings.model_service_api_key,
                    "Content-Type": "application/json",
                }
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                "model_analysis_completed",
                version=version,
                trace_id=trace_id
            )
            
            return JSONResponse(status_code=response.status_code, content=result)
            
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response.text else str(e)
        logger.error(
            "model_analysis_failed",
            version=version,
            status_code=e.response.status_code,
            error=error_detail,
            trace_id=trace_id
        )
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        logger.error("model_analysis_failed", version=version, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model analysis: {str(e)}")


@router.post("/models/relearn")
async def relearn_model(
    dataset_id: str = Body(..., description="Dataset ID"),
    symbol: str = Body(..., description="Trading symbol (e.g., BTCUSDT)"),
    strategy_id: Optional[str] = Body(None, description="Strategy ID (optional)"),
):
    """
    Publish a dataset.ready message to RabbitMQ to trigger model retraining on the same dataset.
    
    This mimics the behavior of publish_dataset_ready.py script.
    """
    trace_id = get_or_create_trace_id()
    logger.info(
        "relearn_model_request",
        dataset_id=dataset_id,
        symbol=symbol,
        strategy_id=strategy_id,
        trace_id=trace_id
    )
    
    try:
        # Create RabbitMQ connection
        connection = await aio_pika.connect_robust(
            f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}@{settings.rabbitmq_host}:{settings.rabbitmq_port}/"
        )
        
        try:
            channel = await connection.channel()
            
            # Prepare message (same format as publish_dataset_ready.py)
            message_data = {
                "dataset_id": dataset_id,
                "symbol": symbol,
                "status": "ready",
                "train_records": 0,
                "validation_records": 0,
                "test_records": 0,
                "trace_id": trace_id or f"dashboard-relearn-{uuid.uuid4().hex[:8]}",
            }
            
            # Add strategy_id if provided
            if strategy_id is not None:
                message_data["strategy_id"] = strategy_id
            
            message_body = json.dumps(message_data).encode()
            
            # Publish message to features.dataset.ready queue
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key="features.dataset.ready",
            )
            
            logger.info(
                "relearn_model_published",
                dataset_id=dataset_id,
                symbol=symbol,
                strategy_id=strategy_id,
                trace_id=trace_id,
                queue="features.dataset.ready",
            )
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Dataset ready signal published to RabbitMQ",
                    "dataset_id": dataset_id,
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "trace_id": message_data["trace_id"],
                }
            )
            
        finally:
            await connection.close()
            
    except Exception as e:
        logger.error(
            "relearn_model_failed",
            dataset_id=dataset_id,
            symbol=symbol,
            strategy_id=strategy_id,
            error=str(e),
            trace_id=trace_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish dataset ready signal: {str(e)}"
        )

