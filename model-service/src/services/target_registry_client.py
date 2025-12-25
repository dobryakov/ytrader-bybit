"""
Target Registry Client.

Provides access to target registry configurations via Feature Service REST API.
After migration 024, config is stored in YAML files managed by feature-service.
Model-service should not access files directly - use feature-service API instead.
"""

from typing import Optional, Dict, Any
import httpx

from ..config.settings import settings
from ..config.logging import get_logger
from ..config.retry import retry_async
from .target_registry_cache import target_registry_cache

logger = get_logger(__name__)


class TargetRegistryClient:
    """Client for accessing target registry configurations via Feature Service API."""

    def __init__(self):
        """Initialize Target Registry client."""
        self.base_url = settings.feature_service_url
        self.api_key = settings.feature_service_api_key
        self.timeout = 10.0  # 10 second timeout for target registry requests

    async def get_target_config(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get target registry configuration by version via Feature Service API.

        Uses cache first, then falls back to API with retry logic.
        Updates cache after successful API response.

        Args:
            version: Target registry version identifier

        Returns:
            Target registry configuration dict or None if not found
        """
        # Check cache first
        cached_config = await target_registry_cache.get(version)
        if cached_config is not None:
            logger.debug("Using cached target registry config", version=version)
            return cached_config

        # Cache miss - fetch from API with retry
        async def _fetch_config():
            url = f"{self.base_url}/target-registry/versions/{version}"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 404:
                    logger.debug("Target registry version not found", version=version)
                    return None
                
                if response.status_code == 503:
                    logger.warning(
                        "Feature Service target registry not initialized",
                        version=version,
                    )
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # API returns {"version": str, "config": dict, ...}
                config = data.get("config")
                if not isinstance(config, dict):
                    logger.error(
                        "Invalid target registry config format from API",
                        version=version,
                        config_type=type(config).__name__,
                    )
                    return None
                
                # Update cache after successful API response
                await target_registry_cache.set(version, config)
                logger.debug(
                    "Target registry config loaded from Feature Service API and cached",
                    version=version,
                )
                return config

        try:
            # Retry with exponential backoff for timeout and network errors
            # Include OSError for network-related errors (ConnectionRefusedError, etc.)
            config = await retry_async(
                _fetch_config,
                max_retries=2,  # 2 retries = 3 total attempts
                initial_delay=1.0,
                max_delay=5.0,
                backoff_multiplier=2.0,
                retryable_exceptions=(httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError, ConnectionError, TimeoutError),
                operation_name="get_target_config",
            )
            return config
        except (httpx.TimeoutException, TimeoutError, ConnectionError) as e:
            logger.error(
                "Timeout or connection error requesting target registry config from Feature Service (after retries)",
                version=version,
                timeout=self.timeout,
                error_type=type(e).__name__,
                error=str(e),
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error requesting target registry config",
                version=version,
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to get target registry config from Feature Service",
                version=version,
                error=str(e),
                exc_info=True,
            )
            # Don't raise - return None to allow graceful degradation
            return None

    async def get_active_target_config(self) -> Optional[Dict[str, Any]]:
        """
        Get active target registry configuration via Feature Service API.

        Returns:
            Active target registry configuration dict or None if not found
        """
        url = f"{self.base_url}/target-registry"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 404:
                    logger.debug("No active target registry version found")
                    return None
                
                if response.status_code == 503:
                    logger.warning("Feature Service target registry not initialized")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # API returns {"active": True, "config": dict}
                config = data.get("config")
                if not isinstance(config, dict):
                    logger.error(
                        "Invalid target registry config format from API",
                        config_type=type(config).__name__,
                    )
                    return None
                
                logger.debug("Active target registry config loaded from Feature Service API")
                return config

        except httpx.TimeoutException:
            logger.error(
                "Timeout requesting active target registry config from Feature Service",
                timeout=self.timeout,
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error requesting active target registry config",
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to get active target registry config from Feature Service",
                error=str(e),
                exc_info=True,
            )
            # Don't raise - return None to allow graceful degradation
            return None

    async def get_target_registry_version(self) -> Optional[str]:
        """
        Get active target registry version identifier via Feature Service API.

        Returns:
            Active target registry version string or None if not found
        """
        # Query versions list and find active one
        url = f"{self.base_url}/target-registry/versions"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 503:
                    logger.warning("Feature Service target registry not initialized")
                    return None
                
                if response.status_code == 404:
                    logger.debug("No target registry versions found")
                    return None
                
                response.raise_for_status()
                versions = response.json()
                
                # Find active version
                for version_record in versions:
                    if version_record.get("is_active"):
                        version = version_record.get("version")
                        if version:
                            logger.debug(
                                "Active target registry version found",
                                version=version,
                            )
                            return version
                
                logger.debug("No active target registry version found")
                return None
                
        except httpx.TimeoutException:
            logger.error(
                "Timeout requesting target registry versions from Feature Service",
                timeout=self.timeout,
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error requesting target registry versions",
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to get target registry version from Feature Service",
                error=str(e),
                exc_info=True,
            )
            return None


# Global instance
target_registry_client = TargetRegistryClient()

