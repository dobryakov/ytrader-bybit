"""Unit tests for ConnectionManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.subscription import Subscription
from src.services.websocket.connection_manager import ConnectionManager, get_connection_manager


@pytest.fixture
def mock_connection():
    """Create a mock WebSocketConnection."""
    conn = MagicMock()
    conn.is_connected = True
    conn.connect = AsyncMock()
    conn.disconnect = AsyncMock()
    conn._endpoint_type = "private"
    return conn


@pytest.fixture
def connection_manager():
    """Create a ConnectionManager instance."""
    return ConnectionManager()


@pytest.mark.asyncio
async def test_get_public_connection_creates_new_connection(connection_manager):
    """Test that get_public_connection creates a new connection if none exists."""
    with patch(
        "src.services.websocket.connection_manager.WebSocketConnection"
    ) as mock_ws_class:
        mock_conn = MagicMock()
        mock_conn.is_connected = False
        mock_conn.connect = AsyncMock()
        mock_ws_class.return_value = mock_conn

        result = await connection_manager.get_public_connection()

        assert result == mock_conn
        mock_ws_class.assert_called_once_with(endpoint_type="public")
        mock_conn.connect.assert_called_once()


@pytest.mark.asyncio
async def test_get_public_connection_reuses_existing_connection(connection_manager, mock_connection):
    """Test that get_public_connection reuses existing connected connection."""
    connection_manager._public_connection = mock_connection

    result = await connection_manager.get_public_connection()

    assert result == mock_connection
    mock_connection.connect.assert_not_called()


@pytest.mark.asyncio
async def test_get_public_connection_reconnects_if_disconnected(connection_manager, mock_connection):
    """Test that get_public_connection reconnects if connection is disconnected."""
    mock_connection.is_connected = False
    connection_manager._public_connection = mock_connection

    result = await connection_manager.get_public_connection()

    assert result == mock_connection
    mock_connection.connect.assert_called_once()


@pytest.mark.asyncio
async def test_get_private_connection_creates_new_connection(connection_manager):
    """Test that get_private_connection creates a new connection if none exists."""
    with patch(
        "src.services.websocket.connection_manager.WebSocketConnection"
    ) as mock_ws_class:
        mock_conn = MagicMock()
        mock_conn.is_connected = False
        mock_conn.connect = AsyncMock()
        mock_ws_class.return_value = mock_conn

        result = await connection_manager.get_private_connection()

        assert result == mock_conn
        mock_ws_class.assert_called_once_with(endpoint_type="private")
        mock_conn.connect.assert_called_once()


@pytest.mark.asyncio
async def test_get_private_connection_reuses_existing_connection(connection_manager, mock_connection):
    """Test that get_private_connection reuses existing connected connection."""
    connection_manager._private_connection = mock_connection

    result = await connection_manager.get_private_connection()

    assert result == mock_connection
    mock_connection.connect.assert_not_called()


@pytest.mark.asyncio
async def test_get_connection_for_subscription_public_channel(connection_manager):
    """Test that get_connection_for_subscription returns public connection for public channels."""
    subscription = Subscription.create(
        channel_type="trades",
        topic="publicTrade.BTCUSDT",  # Bybit v5 uses publicTrade format
        requesting_service="test-service",
        symbol="BTCUSDT",
    )

    with patch.object(
        connection_manager, "get_public_connection", new_callable=AsyncMock
    ) as mock_get_public:
        mock_conn = MagicMock()
        mock_get_public.return_value = mock_conn

        result = await connection_manager.get_connection_for_subscription(subscription)

        assert result == mock_conn
        mock_get_public.assert_called_once()


@pytest.mark.asyncio
async def test_get_connection_for_subscription_private_channel(connection_manager):
    """Test that get_connection_for_subscription returns private connection for private channels."""
    subscription = Subscription.create(
        channel_type="balance",
        topic="wallet",
        requesting_service="test-service",
        symbol=None,
    )

    with patch.object(
        connection_manager, "get_private_connection", new_callable=AsyncMock
    ) as mock_get_private:
        mock_conn = MagicMock()
        mock_get_private.return_value = mock_conn

        result = await connection_manager.get_connection_for_subscription(subscription)

        assert result == mock_conn
        mock_get_private.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_all_disconnects_both_connections(connection_manager, mock_connection):
    """Test that disconnect_all disconnects both public and private connections."""
    connection_manager._public_connection = mock_connection
    connection_manager._private_connection = mock_connection

    await connection_manager.disconnect_all()

    assert connection_manager._public_connection is None
    assert connection_manager._private_connection is None
    assert mock_connection.disconnect.call_count == 2


@pytest.mark.asyncio
async def test_disconnect_all_handles_errors_gracefully(connection_manager, mock_connection):
    """Test that disconnect_all handles disconnection errors gracefully."""
    mock_connection.disconnect.side_effect = Exception("Connection error")
    connection_manager._public_connection = mock_connection
    connection_manager._private_connection = mock_connection

    # Should not raise
    await connection_manager.disconnect_all()

    assert connection_manager._public_connection is None
    assert connection_manager._private_connection is None


def test_get_connection_manager_returns_singleton():
    """Test that get_connection_manager returns a singleton instance."""
    manager1 = get_connection_manager()
    manager2 = get_connection_manager()

    assert manager1 is manager2


def test_get_public_connection_sync_returns_connection(connection_manager, mock_connection):
    """Test that get_public_connection_sync returns connection if it exists."""
    connection_manager._public_connection = mock_connection

    result = connection_manager.get_public_connection_sync()

    assert result == mock_connection


def test_get_public_connection_sync_returns_none_if_not_exists(connection_manager):
    """Test that get_public_connection_sync returns None if connection doesn't exist."""
    result = connection_manager.get_public_connection_sync()

    assert result is None


def test_get_private_connection_sync_returns_connection(connection_manager, mock_connection):
    """Test that get_private_connection_sync returns connection if it exists."""
    connection_manager._private_connection = mock_connection

    result = connection_manager.get_private_connection_sync()

    assert result == mock_connection


def test_get_private_connection_sync_returns_none_if_not_exists(connection_manager):
    """Test that get_private_connection_sync returns None if connection doesn't exist."""
    result = connection_manager.get_private_connection_sync()

    assert result is None

