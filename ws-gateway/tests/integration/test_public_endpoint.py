"""Integration tests for public endpoint connection (testnet)."""

import pytest

from src.services.websocket.connection import WebSocketConnection
from src.services.websocket.connection_manager import ConnectionManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_endpoint_connection_establishes():
    """
    Test that public endpoint connection can be established to Bybit testnet.
    
    This test requires:
    - Network access to Bybit testnet
    - No API credentials needed (public endpoint)
    """
    connection = WebSocketConnection(endpoint_type="public")
    
    try:
        await connection.connect()
        assert connection.is_connected
        
        # Verify connection state
        assert connection.state.status.value == "connected"
        assert connection._endpoint_type == "public"
    finally:
        await connection.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_endpoint_no_authentication_required():
    """
    Test that public endpoint does not require authentication.
    
    This test verifies that public connections can be established
    without API credentials.
    """
    connection = WebSocketConnection(endpoint_type="public")
    
    try:
        # Should connect without authentication
        await connection.connect()
        assert connection.is_connected
        
        # Verify no authentication was attempted
        # (This is implicit - if auth was required, connection would fail)
    finally:
        await connection.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_manager_public_connection():
    """
    Test that ConnectionManager can create and manage public connections.
    """
    manager = ConnectionManager()
    
    try:
        connection = await manager.get_public_connection()
        assert connection is not None
        assert connection.is_connected
        assert connection._endpoint_type == "public"
        
        # Verify reusing same connection
        connection2 = await manager.get_public_connection()
        assert connection is connection2
    finally:
        await manager.disconnect_all()

