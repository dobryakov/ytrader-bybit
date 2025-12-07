"""
Bybit REST API client for fetching historical market data.

Adapted from order-manager for Feature Service needs (public endpoints only).
"""
import asyncio
import hashlib
import hmac
import time
from typing import Any, Dict, Optional
import httpx
from httpx import AsyncClient, Response

from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class BybitAPIError(Exception):
    """Exception raised for Bybit API errors."""
    pass


class BybitClient:
    """Async HTTP client for Bybit REST API v5 with retry logic."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_multiplier: float = 2.0,
        rate_limit_delay_ms: int = 100,
    ):
        """
        Initialize Bybit REST API client.
        
        Args:
            api_key: Bybit API key (optional for public endpoints)
            api_secret: Bybit API secret (optional for public endpoints)
            base_url: Bybit API base URL (testnet or mainnet)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_base_delay: Base delay for exponential backoff (seconds)
            retry_max_delay: Maximum delay between retries (seconds)
            retry_multiplier: Exponential backoff multiplier
            rate_limit_delay_ms: Delay between requests in milliseconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (base_url or config.bybit_rest_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.retry_multiplier = retry_multiplier
        self.rate_limit_delay_ms = rate_limit_delay_ms
        self._client: Optional[AsyncClient] = None
    
    async def _get_client(self) -> AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    def _generate_signature_for_get(
        self, params: Dict[str, Any], timestamp: int, recv_window: str
    ) -> str:
        """
        Generate HMAC-SHA256 signature for GET requests.
        
        Args:
            params: Request parameters (excluding auth params)
            timestamp: Request timestamp in milliseconds
            recv_window: Receive window value
            
        Returns:
            Hex-encoded signature string
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for authenticated requests")
        
        # Sort parameters alphabetically and create query string
        sorted_params = sorted([(k, str(v)) for k, v in params.items() if v is not None])
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Create signature string: timestamp + api_key + recv_window + query_string
        signature_string = f"{timestamp}{self.api_key}{recv_window}{query_string}"
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        
        return signature
    
    def _prepare_auth_headers(
        self, timestamp: int, recv_window: str, signature: str
    ) -> Dict[str, str]:
        """
        Prepare authentication headers for Bybit API v5.
        
        Args:
            timestamp: Request timestamp in milliseconds
            recv_window: Receive window value
            signature: Generated signature
            
        Returns:
            Dictionary with authentication headers
        """
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
    ) -> Response:
        """
        Make HTTP request with exponential backoff retry logic for 429 errors.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            authenticated: Whether to include authentication parameters
            
        Returns:
            HTTP response
            
        Raises:
            BybitAPIError: If request fails after all retries
        """
        client = await self._get_client()
        
        # Prepare authentication
        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
        recv_window = "5000"  # Standard receive window (5 seconds)
        
        # Prepare request parameters and headers
        request_params = params or {}
        request_headers: Dict[str, str] = {}
        
        if authenticated and self.api_key and self.api_secret:
            # Generate signature for GET requests
            signature = self._generate_signature_for_get(request_params, timestamp, recv_window)
            # Add authentication headers
            request_headers.update(self._prepare_auth_headers(timestamp, recv_window, signature))
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Rate limit delay (except for first request)
                if attempt > 0 or self.rate_limit_delay_ms > 0:
                    await asyncio.sleep(self.rate_limit_delay_ms / 1000.0)
                
                # Log request
                logger.debug(
                    "bybit_api_request",
                    method=method,
                    endpoint=endpoint,
                    params_keys=list(request_params.keys()) if request_params else None,
                    has_auth=authenticated,
                    attempt=attempt + 1,
                )
                
                # Make request
                response = await client.request(
                    method=method,
                    url=endpoint,
                    params=request_params,
                    headers=request_headers if request_headers else None,
                )
                
                # Log response
                logger.debug(
                    "bybit_api_response",
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    attempt=attempt + 1,
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
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Max retries reached
                        logger.error(
                            "bybit_api_rate_limit_max_retries",
                            endpoint=endpoint,
                            max_retries=self.max_retries,
                        )
                        raise BybitAPIError(
                            f"Rate limit exceeded after {self.max_retries} attempts"
                        )
                
                # Check for other errors
                if response.status_code >= 400:
                    error_msg = f"Bybit API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("retMsg", error_data.get("ret_msg", error_msg))
                        if "result" in error_data and error_data["result"]:
                            error_msg = str(error_data["result"])
                    except Exception:
                        pass
                    
                    logger.error(
                        "bybit_api_error",
                        method=method,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        error=error_msg,
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
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "bybit_api_request_failed_max_retries",
                        endpoint=endpoint,
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    raise BybitAPIError(f"Request failed after {self.max_retries} attempts: {e}") from e
        
        # Should not reach here, but handle edge case
        if last_error:
            raise BybitAPIError(f"Request failed: {last_error}") from last_error
        raise BybitAPIError("Request failed: unknown error")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, authenticated: bool = False) -> Dict[str, Any]:
        """
        Make GET request to Bybit API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            authenticated: Whether to use authentication (default: False for public endpoints)
            
        Returns:
            Response JSON data
        """
        response = await self._request_with_retry("GET", endpoint, params=params, authenticated=authenticated)
        return response.json()
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("bybit_client_closed")

