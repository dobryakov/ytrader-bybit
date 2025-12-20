"""
Feature Service REST API client.

Provides access to feature vectors and dataset building from Feature Service.
"""

from typing import Optional, Dict, Any
from uuid import UUID
from pathlib import Path
from datetime import datetime
import asyncio
import httpx

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.feature_vector import FeatureVector
from ..models.dataset import Dataset, DatasetBuildRequest

logger = get_logger(__name__)


class FeatureServiceClient:
    """Client for Feature Service REST API."""

    def __init__(self):
        """Initialize Feature Service client."""
        self.base_url = settings.feature_service_url
        self.api_key = settings.feature_service_api_key
        self.timeout = settings.feature_service_feature_timeout_seconds  # Configurable timeout for feature requests
        self.dataset_timeout = settings.feature_service_dataset_build_timeout_seconds
        self.dataset_metadata_timeout = settings.feature_service_dataset_metadata_timeout_seconds
        self.dataset_download_timeout = settings.feature_service_dataset_download_timeout_seconds

    async def get_latest_features(self, symbol: str, trace_id: Optional[str] = None) -> Optional[FeatureVector]:
        """
        Get latest computed features for a symbol from Feature Service.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            trace_id: Optional trace ID for request flow tracking

        Returns:
            FeatureVector or None if features unavailable or error
        """
        url = f"{self.base_url}/features/latest"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {"symbol": symbol}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Parse FeatureVector from response
                feature_vector = FeatureVector(**data)
                if trace_id:
                    feature_vector.trace_id = trace_id
                
                logger.debug(
                    "Retrieved latest features from Feature Service",
                    symbol=symbol,
                    feature_count=len(feature_vector.features),
                    trace_id=trace_id,
                )
                return feature_vector

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Features not available - this is normal if data hasn't arrived yet
                logger.debug("Features not available for symbol", symbol=symbol, trace_id=trace_id)
                return None
            logger.error(
                "Feature Service API error",
                symbol=symbol,
                status_code=e.response.status_code,
                error=str(e),
                trace_id=trace_id,
            )
            return None
        except httpx.TimeoutException:
            logger.warning("Feature Service API timeout", symbol=symbol, timeout=self.timeout, trace_id=trace_id)
            return None
        except Exception as e:
            logger.error(
                "Failed to query Feature Service",
                symbol=symbol,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def get_historical_price(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_seconds: int = 60,
        trace_id: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get historical price for a symbol at (or near) a given timestamp.

        NOTE: Делегирует запрос в feature-service /api/v1/historical/price.
        Эндпоинт пока может возвращать 501/404 – в этом случае возвращаем None.
        """
        from datetime import timezone

        url = f"{self.base_url}/api/v1/historical/price"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        # Ensure UTC ISO format
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        params = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "lookback_seconds": lookback_seconds,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code in (404, 501):
                    logger.debug(
                        "Historical price not available",
                        symbol=symbol,
                        timestamp=params["timestamp"],
                        status_code=response.status_code,
                        trace_id=trace_id,
                    )
                    return None
                response.raise_for_status()
                data = response.json()
                # Ожидаем, что ответ содержит поле price (float)
                price = data.get("price")
                if price is None:
                    logger.warning(
                        "Historical price response without 'price' field",
                        symbol=symbol,
                        timestamp=params["timestamp"],
                        payload=data,
                        trace_id=trace_id,
                    )
                    return None
                return float(price)
        except httpx.TimeoutException:
            logger.warning(
                "Feature Service historical price API timeout",
                symbol=symbol,
                timestamp=params["timestamp"],
                timeout=self.timeout,
                trace_id=trace_id,
            )
            return None
        except httpx.HTTPError as e:
            logger.error(
                "Feature Service historical price API error",
                symbol=symbol,
                timestamp=params["timestamp"],
                error=str(e),
                trace_id=trace_id,
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to query historical price from Feature Service",
                symbol=symbol,
                timestamp=params["timestamp"],
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def build_dataset(self, request: Dict[str, Any], trace_id: Optional[str] = None) -> Optional[UUID]:
        """
        Request dataset build from Feature Service.

        Args:
            request: Dataset build request dictionary with fields:
                - symbol: str
                - split_strategy: str ('time_based' or 'walk_forward')
                - train_period_start: Optional[datetime]
                - train_period_end: Optional[datetime]
                - validation_period_start: Optional[datetime]
                - validation_period_end: Optional[datetime]
                - test_period_start: Optional[datetime]
                - test_period_end: Optional[datetime]
                - walk_forward_config: Optional[Dict]
                - target_registry_version: str
                - feature_registry_version: str
                - output_format: str (default: 'parquet')
            trace_id: Optional trace ID for request flow tracking

        Returns:
            Dataset ID (UUID) or None if request failed
        """
        url = f"{self.base_url}/dataset/build"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.dataset_timeout) as client:
                response = await client.post(url, headers=headers, json=request)
                response.raise_for_status()
                data = response.json()
                
                dataset_id = UUID(data["dataset_id"])
                logger.info(
                    "Dataset build requested from Feature Service",
                    dataset_id=str(dataset_id),
                    symbol=request.get("symbol"),
                    trace_id=trace_id,
                )
                return dataset_id

        except httpx.HTTPStatusError as e:
            logger.error(
                "Feature Service dataset build API error",
                status_code=e.response.status_code,
                error=str(e),
                request=request,
                trace_id=trace_id,
            )
            return None
        except httpx.TimeoutException:
            logger.warning(
                "Feature Service dataset build API timeout",
                timeout=self.dataset_timeout,
                request=request,
                trace_id=trace_id,
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to request dataset build from Feature Service",
                error=str(e),
                request=request,
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def get_dataset(self, dataset_id: UUID, trace_id: Optional[str] = None, max_retries: int = 3) -> Optional[Dataset]:
        """
        Get dataset metadata by ID from Feature Service with retry logic for transient failures.

        Args:
            dataset_id: Dataset UUID identifier
            trace_id: Optional trace ID for request flow tracking
            max_retries: Maximum number of retry attempts for transient failures (default: 3)

        Returns:
            Dataset model or None if not found/error
        """
        url = f"{self.base_url}/dataset/{dataset_id}"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        delay = 1.0  # Initial delay in seconds
        max_delay = 10.0  # Maximum delay in seconds

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.dataset_metadata_timeout) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Parse Dataset from response
                    dataset = Dataset(**data)
                    if attempt > 0:
                        logger.info(
                            "Retrieved dataset metadata from Feature Service after retry",
                            dataset_id=str(dataset_id),
                            attempt=attempt + 1,
                            trace_id=trace_id,
                        )
                    else:
                        logger.debug(
                            "Retrieved dataset metadata from Feature Service",
                            dataset_id=str(dataset_id),
                            status=dataset.status.value if hasattr(dataset.status, 'value') else str(dataset.status),
                            trace_id=trace_id,
                        )
                    return dataset

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Dataset not found - don't retry
                    logger.debug("Dataset not found", dataset_id=str(dataset_id), trace_id=trace_id)
                    return None
                # Other HTTP errors - retry if not last attempt
                if attempt < max_retries and e.response.status_code >= 500:
                    logger.warning(
                        "Feature Service dataset API error, will retry",
                        dataset_id=str(dataset_id),
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                        max_retries=max_retries + 1,
                        delay=delay,
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, max_delay)
                    continue
                logger.error(
                    "Feature Service dataset API error",
                    dataset_id=str(dataset_id),
                    status_code=e.response.status_code,
                    error=str(e),
                    trace_id=trace_id,
                )
                return None
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                # Network/timeout errors - retry if not last attempt
                if attempt < max_retries:
                    logger.warning(
                        "Feature Service dataset API timeout/network error, will retry",
                        dataset_id=str(dataset_id),
                        timeout=self.dataset_metadata_timeout,
                        attempt=attempt + 1,
                        max_retries=max_retries + 1,
                        delay=delay,
                        error_type=type(e).__name__,
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, max_delay)
                    continue
                logger.warning(
                    "Feature Service dataset API timeout/network error after retries",
                    dataset_id=str(dataset_id),
                    timeout=self.dataset_metadata_timeout,
                    max_retries=max_retries + 1,
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                )
                return None
            except Exception as e:
                # Other errors - don't retry
                logger.error(
                    "Failed to query dataset from Feature Service",
                    dataset_id=str(dataset_id),
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )
                return None

        return None

    async def download_dataset(
        self, dataset_id: UUID, split: str = "train", trace_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download dataset split file from Feature Service.

        Args:
            dataset_id: Dataset UUID identifier
            split: Dataset split to download ('train', 'validation', 'test')
            trace_id: Optional trace ID for request flow tracking

        Returns:
            Path to downloaded file or None if download failed
        """
        url = f"{self.base_url}/dataset/{dataset_id}/download"
        headers = {
            "X-API-Key": self.api_key,
        }
        params = {"split": split}

        # Determine output directory from settings
        storage_path = Path(settings.feature_service_dataset_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Determine file extension from dataset metadata
        dataset_meta = await self.get_dataset(dataset_id, trace_id)
        if not dataset_meta:
            logger.error(
                "Cannot download dataset - metadata not found",
                dataset_id=str(dataset_id),
                split=split,
                trace_id=trace_id,
            )
            return None

        output_format = dataset_meta.output_format
        file_path = storage_path / f"{dataset_id}_{split}.{output_format}"

        try:
            async with httpx.AsyncClient(timeout=self.dataset_download_timeout) as client:
                async with client.stream("GET", url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    
                    # Download file in chunks
                    with open(file_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    
                    logger.info(
                        "Downloaded dataset split from Feature Service",
                        dataset_id=str(dataset_id),
                        split=split,
                        file_path=str(file_path),
                        file_size=file_path.stat().st_size,
                        trace_id=trace_id,
                    )
                    return file_path

        except httpx.HTTPStatusError as e:
            logger.error(
                "Feature Service dataset download API error",
                dataset_id=str(dataset_id),
                split=split,
                status_code=e.response.status_code,
                error=str(e),
                trace_id=trace_id,
            )
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            return None
        except httpx.TimeoutException:
            logger.warning(
                "Feature Service dataset download timeout",
                dataset_id=str(dataset_id),
                split=split,
                trace_id=trace_id,
            )
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            return None
        except Exception as e:
            logger.error(
                "Failed to download dataset from Feature Service",
                dataset_id=str(dataset_id),
                split=split,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            return None

    async def compute_target(
        self,
        symbol: str,
        prediction_timestamp: datetime,
        target_timestamp: datetime,
        target_registry_version: str,
        horizon_seconds: Optional[int] = None,
        max_lookback_seconds: int = 300,
        trace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Вычислить фактическое значение таргета через Feature Service API.
        
        Args:
            symbol: Trading pair symbol
            prediction_timestamp: Timestamp when prediction was made
            target_timestamp: Timestamp for target computation
            target_registry_version: Target Registry version (config will be loaded from registry)
            horizon_seconds: Optional horizon override (if None, uses horizon from registry config)
            max_lookback_seconds: Maximum lookback for data availability fallback
            trace_id: Optional trace ID
        
        Returns:
            Dict с результатами вычисления или None при ошибке
        """
        from datetime import timezone
        
        url = f"{self.base_url}/api/v1/targets/compute"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Normalize timestamps to UTC
        if prediction_timestamp.tzinfo is None:
            prediction_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc)
        else:
            prediction_timestamp = prediction_timestamp.astimezone(timezone.utc)
        
        if target_timestamp.tzinfo is None:
            target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
        else:
            target_timestamp = target_timestamp.astimezone(timezone.utc)
        
        payload = {
            "symbol": symbol,
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "target_timestamp": target_timestamp.isoformat(),
            "target_registry_version": target_registry_version,
            "max_lookback_seconds": max_lookback_seconds,
        }
        
        if horizon_seconds is not None:
            payload["horizon_seconds"] = horizon_seconds
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 404:
                    # Data unavailable - try to extract error details from response
                    error_detail = None
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            error_detail = error_data.get("detail")
                            if isinstance(error_detail, dict):
                                error_detail = error_detail.get("message", str(error_detail))
                            elif isinstance(error_detail, str):
                                pass  # Use as-is
                    except Exception:
                        pass  # Ignore JSON parsing errors
                    
                    logger.warning(
                        "Target computation data unavailable",
                        symbol=symbol,
                        prediction_timestamp=prediction_timestamp.isoformat(),
                        target_timestamp=target_timestamp.isoformat(),
                        target_registry_version=target_registry_version,
                        horizon_seconds=horizon_seconds,
                        max_lookback_seconds=max_lookback_seconds,
                        error_detail=error_detail,
                        trace_id=trace_id,
                    )
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                logger.debug(
                    "Computed target from Feature Service",
                    symbol=symbol,
                    target_type=data.get("target_type"),
                    preset=data.get("preset"),
                    trace_id=trace_id,
                )
                return data
                
        except httpx.TimeoutException:
            logger.warning(
                "Feature Service target computation API timeout",
                symbol=symbol,
                prediction_timestamp=prediction_timestamp.isoformat(),
                target_timestamp=target_timestamp.isoformat(),
                timeout=self.timeout,
                trace_id=trace_id,
            )
            return None
        except httpx.HTTPError as e:
            logger.error(
                "Feature Service target computation API error",
                symbol=symbol,
                prediction_timestamp=prediction_timestamp.isoformat(),
                target_timestamp=target_timestamp.isoformat(),
                error=str(e),
                trace_id=trace_id,
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to compute target from Feature Service",
                symbol=symbol,
                prediction_timestamp=prediction_timestamp.isoformat(),
                target_timestamp=target_timestamp.isoformat(),
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None


# Global client instance
feature_service_client = FeatureServiceClient()

