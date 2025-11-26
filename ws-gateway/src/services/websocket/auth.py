"""Bybit WebSocket authentication logic."""

import hashlib
import hmac
import time
from typing import Dict

from ...config.logging import get_logger
from ...config.settings import settings
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


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

    Handles authentication failures and validates credential format (EC4: Handle authentication failures).

    Args:
        response: Response dictionary from Bybit WebSocket

    Returns:
        True if authentication successful, False otherwise
    """
    trace_id = get_or_create_trace_id()

    # Validate response structure
    if not isinstance(response, dict):
        logger.error(
            "auth_response_invalid_structure",
            response_type=type(response).__name__,
            trace_id=trace_id,
        )
        return False

    # Bybit returns {"success": true} on successful authentication
    # or {"ret_msg": "error message"} on failure
    success = response.get("success", False) is True

    if not success:
        error_msg = response.get("ret_msg", "Unknown authentication error")
        logger.error(
            "auth_response_failed",
            error=error_msg,
            response=response,  # Log full response for debugging
            trace_id=trace_id,
        )

        # Check for specific authentication error types
        error_msg_lower = error_msg.lower()
        if "invalid" in error_msg_lower or "expired" in error_msg_lower:
            logger.error(
                "auth_credentials_invalid_or_expired",
                error=error_msg,
                trace_id=trace_id,
            )
        elif "timeout" in error_msg_lower:
            logger.error(
                "auth_timeout",
                error=error_msg,
                trace_id=trace_id,
            )

    return success


def validate_credentials() -> bool:
    """
    Validate that API credentials are present and properly formatted (EC4: Handle authentication failures).

    Returns:
        True if credentials appear valid, False otherwise
    """
    trace_id = get_or_create_trace_id()

    if not settings.bybit_api_key or not settings.bybit_api_secret:
        logger.error(
            "auth_credentials_missing",
            has_api_key=bool(settings.bybit_api_key),
            has_api_secret=bool(settings.bybit_api_secret),
            trace_id=trace_id,
        )
        return False

    # Basic format validation
    if len(settings.bybit_api_key) < 10 or len(settings.bybit_api_secret) < 10:
        logger.warning(
            "auth_credentials_suspicious_length",
            api_key_length=len(settings.bybit_api_key),
            api_secret_length=len(settings.bybit_api_secret),
            trace_id=trace_id,
        )

    return True

