"""Bybit REST API client wrapper with HMAC-SHA256 authentication and retry logic."""

import asyncio
import hashlib
import hmac
import time
from typing import Any, Dict, Optional
import httpx
from httpx import AsyncClient, Response

from ..config.settings import settings
from ..exceptions import BybitAPIError
from ..config.logging import get_logger
from .tracing import get_or_create_trace_id

logger = get_logger(__name__)


class BybitClient:
    """Async HTTP client for Bybit REST API with authentication and retry logic."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_multiplier: float = 2.0,
    ):
        """
        Initialize Bybit REST API client.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            base_url: Bybit API base URL (testnet or mainnet)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_base_delay: Base delay for exponential backoff (seconds)
            retry_max_delay: Maximum delay between retries (seconds)
            retry_multiplier: Exponential backoff multiplier
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.retry_multiplier = retry_multiplier
        self._client: Optional[AsyncClient] = None

    async def _get_client(self) -> AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """
        Generate HMAC-SHA256 signature for Bybit REST API authentication.

        Bybit REST API v5 uses the following signature format:
        - Sort parameters alphabetically
        - Create query string: key1=value1&key2=value2
        - Create signature string: timestamp + api_key + recv_window + query_string
        - Generate HMAC-SHA256 signature

        Args:
            params: Request parameters (excluding timestamp, api_key, sign)
            timestamp: Request timestamp in milliseconds

        Returns:
            Hex-encoded signature string
        """
        # Sort parameters alphabetically
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params if v is not None])

        # Create signature string: timestamp + api_key + recv_window + query_string
        recv_window = "5000"  # Standard receive window (5 seconds)
        signature_string = f"{timestamp}{self.api_key}{recv_window}{query_string}"

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def _prepare_auth_params(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare authenticated request parameters.

        Args:
            params: Additional request parameters

        Returns:
            Dictionary with authentication parameters added
        """
        if params is None:
            params = {}

        # Add authentication parameters
        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
        auth_params = {
            "api_key": self.api_key,
            "timestamp": timestamp,
            "recv_window": "5000",
            **params,
        }

        # Generate signature
        signature = self._generate_signature(params, timestamp)
        auth_params["sign"] = signature

        return auth_params

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> Response:
        """
        Make HTTP request with exponential backoff retry logic for 429 errors.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body data
            authenticated: Whether to include authentication parameters

        Returns:
            HTTP response

        Raises:
            BybitAPIError: If request fails after all retries
        """
        trace_id = get_or_create_trace_id()
        client = await self._get_client()

        # Prepare parameters
        if authenticated:
            request_params = self._prepare_auth_params(params)
        else:
            request_params = params or {}

        # Prepare request data
        request_kwargs: Dict[str, Any] = {}
        if json_data:
            request_kwargs["json"] = json_data

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Log request
                logger.debug(
                    "bybit_api_request",
                    method=method,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    trace_id=trace_id,
                )

                # Make request
                response = await client.request(
                    method=method,
                    url=endpoint,
                    params=request_params,
                    **request_kwargs,
                )

                # Log response
                logger.debug(
                    "bybit_api_response",
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                    trace_id=trace_id,
                )

                # Check for rate limit (429)
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        # Calculate exponential backoff delay
                        delay = min(
                            self.retry_base_delay * (self.retry_multiplier ** attempt),
                            self.retry_max_delay,
                        )
                        logger.warning(
                            "bybit_api_rate_limit",
                            endpoint=endpoint,
                            attempt=attempt + 1,
                            retry_after=delay,
                            trace_id=trace_id,
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Max retries reached
                        logger.error(
                            "bybit_api_rate_limit_max_retries",
                            endpoint=endpoint,
                            max_retries=self.max_retries,
                            trace_id=trace_id,
                        )
                        raise BybitAPIError(
                            f"Rate limit exceeded after {self.max_retries} attempts"
                        )

                # Check for other errors
                if response.status_code >= 400:
                    error_msg = f"Bybit API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("ret_msg", error_msg)
                    except Exception:
                        pass

                    logger.error(
                        "bybit_api_error",
                        method=method,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        error=error_msg,
                        trace_id=trace_id,
                    )
                    raise BybitAPIError(error_msg)

                return response

            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.retry_base_delay * (self.retry_multiplier ** attempt),
                        self.retry_max_delay,
                    )
                    logger.warning(
                        "bybit_api_request_failed",
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        error=str(e),
                        retry_after=delay,
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "bybit_api_request_failed_max_retries",
                        endpoint=endpoint,
                        max_retries=self.max_retries,
                        error=str(e),
                        trace_id=trace_id,
                    )
                    raise BybitAPIError(f"Request failed after {self.max_retries} attempts: {e}") from e

        # Should not reach here, but handle edge case
        if last_error:
            raise BybitAPIError(f"Request failed: {last_error}") from last_error
        raise BybitAPIError("Request failed: unknown error")

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, authenticated: bool = True) -> Dict[str, Any]:
        """Make GET request to Bybit API."""
        response = await self._request_with_retry("GET", endpoint, params=params, authenticated=authenticated)
        return response.json()

    async def post(self, endpoint: str, params: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None, authenticated: bool = True) -> Dict[str, Any]:
        """Make POST request to Bybit API."""
        response = await self._request_with_retry("POST", endpoint, params=params, json_data=json_data, authenticated=authenticated)
        return response.json()

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("bybit_client_closed")


# Global client instance
_bybit_client: Optional[BybitClient] = None


def get_bybit_client() -> BybitClient:
    """Get or create global Bybit client instance."""
    global _bybit_client
    if _bybit_client is None:
        _bybit_client = BybitClient(
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            base_url=settings.bybit_api_base_url,
            timeout=settings.order_manager_order_execution_timeout,
            max_retries=settings.order_manager_bybit_api_retry_max_attempts,
            retry_base_delay=settings.order_manager_bybit_api_retry_base_delay,
            retry_max_delay=settings.order_manager_bybit_api_retry_max_delay,
            retry_multiplier=settings.order_manager_bybit_api_retry_multiplier,
        )
    return _bybit_client


async def close_bybit_client() -> None:
    """Close global Bybit client instance."""
    global _bybit_client
    if _bybit_client is not None:
        await _bybit_client.close()
        _bybit_client = None

