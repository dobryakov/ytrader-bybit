"""Unit tests for validation utilities."""

import pytest
from src.utils.validation import (
    sanitize_string,
    validate_asset_symbol,
    validate_uuid,
    validate_order_status,
    validate_order_side,
    sanitize_sql_identifier,
)


def test_sanitize_string():
    """Test string sanitization."""
    assert sanitize_string("  test  ") == "test"
    assert sanitize_string("test", max_length=2) == "te"
    assert sanitize_string(None) is None


def test_validate_asset_symbol():
    """Test asset symbol validation."""
    assert validate_asset_symbol("BTCUSDT") == "BTCUSDT"
    assert validate_asset_symbol("btcusdt") == "BTCUSDT"
    assert validate_asset_symbol("  ETHUSDT  ") == "ETHUSDT"

    with pytest.raises(ValueError):
        validate_asset_symbol("")

    with pytest.raises(ValueError):
        validate_asset_symbol("INVALID@SYMBOL")


def test_validate_uuid():
    """Test UUID validation."""
    valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
    assert validate_uuid(valid_uuid) == valid_uuid

    with pytest.raises(ValueError):
        validate_uuid("")

    with pytest.raises(ValueError):
        validate_uuid("not-a-uuid")


def test_validate_order_status():
    """Test order status validation."""
    assert validate_order_status("pending") == "pending"
    assert validate_order_status("FILLED") == "filled"

    with pytest.raises(ValueError):
        validate_order_status("invalid_status")


def test_validate_order_side():
    """Test order side validation."""
    assert validate_order_side("buy") == "BUY"
    assert validate_order_side("SELL") == "SELL"

    with pytest.raises(ValueError):
        validate_order_side("invalid_side")


def test_sanitize_sql_identifier():
    """Test SQL identifier sanitization."""
    assert sanitize_sql_identifier("orders") == "orders"
    assert sanitize_sql_identifier("order_id") == "order_id"

    with pytest.raises(ValueError):
        sanitize_sql_identifier("order; DROP TABLE orders;--")

