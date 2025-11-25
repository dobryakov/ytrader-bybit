"""Bybit WebSocket authentication logic."""

import hashlib
import hmac
import time
from typing import Dict

from ..config.settings import settings


def generate_auth_signature(api_key: str, api_secret: str, expires: int) -> str:
    """
    Generate HMAC-SHA256 signature for Bybit WebSocket authentication.

    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        expires: Expiration timestamp (Unix timestamp in milliseconds)

    Returns:
        Base64-encoded signature string
    """
    # Bybit uses HMAC-SHA256 with specific format: GET/realtime{expires}
    message = f"GET/realtime{expires}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


def generate_auth_message() -> Dict[str, str]:
    """
    Generate Bybit WebSocket authentication message.

    Returns:
        Dictionary containing authentication parameters
    """
    # Expires timestamp: current time + 5 minutes (in milliseconds)
    expires = int((time.time() + 300) * 1000)

    # Generate signature
    signature = generate_auth_signature(
        settings.bybit_api_key, settings.bybit_api_secret, expires
    )

    # Return authentication message in Bybit format
    return {
        "op": "auth",
        "args": [settings.bybit_api_key, expires, signature],
    }


def validate_auth_response(response: Dict) -> bool:
    """
    Validate authentication response from Bybit.

    Args:
        response: Response dictionary from Bybit WebSocket

    Returns:
        True if authentication successful, False otherwise
    """
    # Bybit returns {"success": true} on successful authentication
    # or {"ret_msg": "error message"} on failure
    if isinstance(response, dict):
        return response.get("success", False) is True
    return False

