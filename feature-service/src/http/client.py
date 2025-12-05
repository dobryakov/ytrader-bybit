"""
HTTP client for ws-gateway REST API integration.
"""
import httpx
from typing import Optional, Dict, Any
from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class HTTPClient:
    """HTTP client for ws-gateway API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        connect_timeout: float = 5.0,
    ):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL for API (defaults to config)
            api_key: API key for authentication (defaults to config)
            timeout: Request timeout in seconds (default: 10.0)
            connect_timeout: Connection timeout in seconds (default: 5.0)
        """
        self._base_url = base_url or config.ws_gateway_api_url
        self._api_key = api_key or config.ws_gateway_api_key
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.debug(
            "HTTPClient initialized",
            base_url=self._base_url,
            timeout=self._timeout,
            connect_timeout=self._connect_timeout,
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers: Dict[str, str] = {}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            
            # Use separate timeouts for connect and read
            timeout = httpx.Timeout(
                connect=self._connect_timeout,
                read=self._timeout,
                write=self._timeout,
                pool=self._connect_timeout,
            )
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.debug("HTTP client created", base_url=self._base_url)
        return self._client
    
    async def get(self, path: str, **kwargs) -> httpx.Response:
        """
        Make GET request.
        
        Args:
            path: Request path
            **kwargs: Additional arguments for httpx
            
        Returns:
            httpx.Response: Response object
        """
        client = await self._get_client()
        logger.debug("HTTP GET", path=path)
        return await client.get(path, **kwargs)
    
    async def post(self, path: str, **kwargs) -> httpx.Response:
        """
        Make POST request.
        
        Args:
            path: Request path
            **kwargs: Additional arguments for httpx
            
        Returns:
            httpx.Response: Response object
            
        Raises:
            httpx.ConnectError: If connection to server fails
            httpx.TimeoutException: If request times out
            httpx.HTTPStatusError: If server returns error status
        """
        client = await self._get_client()
        logger.debug("HTTP POST", path=path, base_url=self._base_url)
        try:
            return await client.post(path, **kwargs)
        except httpx.ConnectError as e:
            logger.error(
                "HTTP connection error",
                path=path,
                base_url=self._base_url,
                error=str(e),
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(
                "HTTP timeout",
                path=path,
                base_url=self._base_url,
                timeout=self._timeout,
                error=str(e),
            )
            raise
    
    async def put(self, path: str, **kwargs) -> httpx.Response:
        """
        Make PUT request.
        
        Args:
            path: Request path
            **kwargs: Additional arguments for httpx
            
        Returns:
            httpx.Response: Response object
        """
        client = await self._get_client()
        logger.debug("HTTP PUT", path=path)
        return await client.put(path, **kwargs)
    
    async def delete(self, path: str, **kwargs) -> httpx.Response:
        """
        Make DELETE request.
        
        Args:
            path: Request path
            **kwargs: Additional arguments for httpx
            
        Returns:
            httpx.Response: Response object
        """
        client = await self._get_client()
        logger.debug("HTTP DELETE", path=path)
        return await client.delete(path, **kwargs)
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("HTTP client closed")

