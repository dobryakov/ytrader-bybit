"""
Feature Service REST API client.

Provides access to feature vectors and dataset building from Feature Service.
"""

from typing import Optional, Dict, Any
from uuid import UUID
from pathlib import Path
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
        self.timeout = 10.0  # 10 second timeout for feature requests
        self.dataset_timeout = settings.feature_service_dataset_build_timeout_seconds

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
                - target_config: Dict
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

    async def get_dataset(self, dataset_id: UUID, trace_id: Optional[str] = None) -> Optional[Dataset]:
        """
        Get dataset metadata by ID from Feature Service.

        Args:
            dataset_id: Dataset UUID identifier
            trace_id: Optional trace ID for request flow tracking

        Returns:
            Dataset model or None if not found/error
        """
        url = f"{self.base_url}/dataset/{dataset_id}"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Parse Dataset from response
                dataset = Dataset(**data)
                logger.debug(
                    "Retrieved dataset metadata from Feature Service",
                    dataset_id=str(dataset_id),
                    status=dataset.status.value if hasattr(dataset.status, 'value') else str(dataset.status),
                    trace_id=trace_id,
                )
                return dataset

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("Dataset not found", dataset_id=str(dataset_id), trace_id=trace_id)
                return None
            logger.error(
                "Feature Service dataset API error",
                dataset_id=str(dataset_id),
                status_code=e.response.status_code,
                error=str(e),
                trace_id=trace_id,
            )
            return None
        except httpx.TimeoutException:
            logger.warning(
                "Feature Service dataset API timeout",
                dataset_id=str(dataset_id),
                timeout=self.timeout,
                trace_id=trace_id,
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to query dataset from Feature Service",
                dataset_id=str(dataset_id),
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
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
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for large downloads
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


# Global client instance
feature_service_client = FeatureServiceClient()

