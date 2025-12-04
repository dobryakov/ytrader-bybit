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
        timeout: float = 30.0,
    ):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL for API (defaults to config)
            api_key: API key for authentication (defaults to config)
            timeout: Request timeout in seconds
        """
        self._base_url = base_url or config.ws_gateway_api_url
        self._api_key = api_key or config.ws_gateway_api_key
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers: Dict[str, str] = {}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
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
        """
        client = await self._get_client()
        logger.debug("HTTP POST", path=path)
        return await client.post(path, **kwargs)
    
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

