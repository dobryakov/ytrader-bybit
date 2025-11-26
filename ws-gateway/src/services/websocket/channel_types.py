"""Channel type classification for public vs private endpoints."""

# Public channels that should use /v5/public endpoint
PUBLIC_CHANNELS = {
    "trades",
    "ticker",
    "orderbook",
    "kline",
    "liquidation",
}

# Private channels that require authentication and use /v5/private endpoint
PRIVATE_CHANNELS = {
    "order",
    "balance",
    "position",
}


def is_public_channel(channel_type: str) -> bool:
    """Check if a channel type is public."""
    return channel_type in PUBLIC_CHANNELS


def is_private_channel(channel_type: str) -> bool:
    """Check if a channel type is private."""
    return channel_type in PRIVATE_CHANNELS


def get_endpoint_type(channel_type: str) -> str:
    """Get endpoint type (public or private) for a channel type."""
    if is_public_channel(channel_type):
        return "public"
    elif is_private_channel(channel_type):
        return "private"
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

