"""Unit tests for channel type classification."""

import pytest

from src.services.websocket.channel_types import (
    PUBLIC_CHANNELS,
    PRIVATE_CHANNELS,
    get_endpoint_type_for_channel,
    is_private_channel,
    is_public_channel,
)


def test_public_channels_are_classified_correctly():
    """Test that public channels are correctly identified."""
    assert is_public_channel("trades")
    assert is_public_channel("ticker")
    assert is_public_channel("orderbook")
    assert is_public_channel("kline")
    assert is_public_channel("liquidation")
    assert is_public_channel("funding")


def test_private_channels_are_classified_correctly():
    """Test that private channels are correctly identified."""
    assert is_private_channel("balance")
    assert is_private_channel("order")


def test_public_channels_not_private():
    """Test that public channels are not classified as private."""
    assert not is_private_channel("trades")
    assert not is_private_channel("ticker")
    assert not is_private_channel("orderbook")
    assert not is_private_channel("kline")
    assert not is_private_channel("liquidation")
    assert not is_private_channel("funding")


def test_private_channels_not_public():
    """Test that private channels are not classified as public."""
    assert not is_public_channel("balance")
    assert not is_public_channel("order")


def test_get_endpoint_type_for_public_channels():
    """Test that public channels return 'public' endpoint type."""
    assert get_endpoint_type_for_channel("trades") == "public"
    assert get_endpoint_type_for_channel("ticker") == "public"
    assert get_endpoint_type_for_channel("orderbook") == "public"
    assert get_endpoint_type_for_channel("kline") == "public"
    assert get_endpoint_type_for_channel("liquidation") == "public"
    assert get_endpoint_type_for_channel("funding") == "public"


def test_get_endpoint_type_for_private_channels():
    """Test that private channels return 'private' endpoint type."""
    assert get_endpoint_type_for_channel("balance") == "private"
    assert get_endpoint_type_for_channel("order") == "private"


def test_get_endpoint_type_for_unknown_channel():
    """Test that unknown channel types raise ValueError."""
    with pytest.raises(ValueError, match="Unknown channel type"):
        get_endpoint_type_for_channel("unknown_channel")


def test_public_channels_set_contents():
    """Test that PUBLIC_CHANNELS set contains expected channels."""
    expected_public = {"trades", "ticker", "orderbook", "kline", "liquidation", "funding"}
    assert PUBLIC_CHANNELS == expected_public


def test_private_channels_set_contents():
    """Test that PRIVATE_CHANNELS set contains expected channels."""
    expected_private = {"balance", "order", "position"}
    assert PRIVATE_CHANNELS == expected_private


def test_channel_sets_are_disjoint():
    """Test that public and private channel sets don't overlap."""
    assert PUBLIC_CHANNELS.isdisjoint(PRIVATE_CHANNELS)

