"""Input validation and sanitization utilities."""

import re
from typing import Optional
from uuid import UUID


def sanitize_string(value: Optional[str], max_length: Optional[int] = None) -> Optional[str]:
    """Sanitize string input by stripping whitespace and limiting length.

    Args:
        value: String value to sanitize
        max_length: Maximum allowed length (None for no limit)

    Returns:
        Sanitized string or None if value is None
    """
    if value is None:
        return None

    # Strip leading/trailing whitespace
    sanitized = value.strip()

    # Limit length if specified
    if max_length is not None and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def validate_asset_symbol(asset: str) -> str:
    """Validate and sanitize trading pair symbol.

    Args:
        asset: Trading pair symbol (e.g., BTCUSDT)

    Returns:
        Uppercase sanitized asset symbol

    Raises:
        ValueError: If asset symbol is invalid
    """
    if not asset:
        raise ValueError("Asset symbol cannot be empty")

    # Sanitize and uppercase
    asset = sanitize_string(asset, max_length=20).upper()

    # Validate format: alphanumeric, typically 6-20 characters
    if not re.match(r"^[A-Z0-9]{2,20}$", asset):
        raise ValueError(f"Invalid asset symbol format: {asset}")

    return asset


def validate_uuid(value: str, field_name: str = "ID") -> str:
    """Validate UUID format.

    Args:
        value: UUID string to validate
        field_name: Name of the field for error messages

    Returns:
        Validated UUID string

    Raises:
        ValueError: If UUID format is invalid
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    value = sanitize_string(value)

    try:
        # Validate UUID format
        UUID(value)
    except ValueError as e:
        raise ValueError(f"Invalid {field_name} format: {value}") from e

    return value


def validate_order_status(status: str) -> str:
    """Validate order status value.

    Args:
        status: Order status string

    Returns:
        Lowercase validated status

    Raises:
        ValueError: If status is invalid
    """
    valid_statuses = {
        "pending",
        "partially_filled",
        "filled",
        "cancelled",
        "rejected",
        "dry_run",
    }

    if not status:
        raise ValueError("Status cannot be empty")

    status = sanitize_string(status).lower()

    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of: {', '.join(sorted(valid_statuses))}")

    return status


def validate_order_side(side: str) -> str:
    """Validate order side value.

    Args:
        side: Order side string (Buy/Sell)

    Returns:
        Uppercase validated side

    Raises:
        ValueError: If side is invalid
    """
    valid_sides = {"BUY", "SELL"}

    if not side:
        raise ValueError("Side cannot be empty")

    side = sanitize_string(side).upper()

    if side not in valid_sides:
        raise ValueError(f"Invalid side. Must be one of: {', '.join(sorted(valid_sides))}")

    return side


def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier to prevent SQL injection.

    Args:
        identifier: SQL identifier (table name, column name)

    Returns:
        Sanitized identifier

    Raises:
        ValueError: If identifier contains invalid characters
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")

    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")

    return identifier

