"""Channel type classification for public and private WebSocket endpoints."""

# Public channels that can be subscribed via /v5/public endpoint (no authentication required)
PUBLIC_CHANNELS = {
    "trades",      # trade.BTCUSDT
    "ticker",      # tickers.BTCUSDT
    "orderbook",   # orderbook.1.BTCUSDT
    "kline",       # kline.1.BTCUSDT
    "liquidation",  # liquidation
}

# Private channels that require /v5/private endpoint (authentication required)
PRIVATE_CHANNELS = {
    "balance",     # wallet
    "order",       # order
    "position",    # position updates
}


def is_public_channel(channel_type: str) -> bool:
    """
    Check if a channel type is public.

    Args:
        channel_type: Channel type string (e.g., "trades", "balance")

    Returns:
        True if channel is public, False if private
    """
    return channel_type in PUBLIC_CHANNELS


def is_private_channel(channel_type: str) -> bool:
    """
    Check if a channel type is private.

    Args:
        channel_type: Channel type string (e.g., "trades", "balance")

    Returns:
        True if channel is private, False if public
    """
    return channel_type in PRIVATE_CHANNELS


def get_endpoint_type_for_channel(channel_type: str) -> str:
    """
    Get the endpoint type (public or private) for a given channel type.

    Args:
        channel_type: Channel type string (e.g., "trades", "balance")

    Returns:
        "public" or "private"

    Raises:
        ValueError: If channel_type is not recognized
    """
    if is_public_channel(channel_type):
        return "public"
    elif is_private_channel(channel_type):
        return "private"
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

