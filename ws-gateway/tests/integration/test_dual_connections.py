"""Integration tests for dual connection simultaneous operation."""

import pytest

from src.services.websocket.connection_manager import ConnectionManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dual_connections_simultaneous_operation():
    """
    Test that both public and private connections can operate simultaneously.
    
    This test requires:
    - Network access to Bybit testnet
    - Valid API credentials for private endpoint
    """
    manager = ConnectionManager()
    
    try:
        # Get both connections
        public_conn = await manager.get_public_connection()
        private_conn = await manager.get_private_connection()
        
        # Verify both are connected
        assert public_conn.is_connected
        assert private_conn.is_connected
        
        # Verify they are different connections
        assert public_conn is not private_conn
        
        # Verify endpoint types
        assert public_conn._endpoint_type == "public"
        assert private_conn._endpoint_type == "private"
        
        # Both should maintain their connections independently
        assert public_conn.state.status.value == "connected"
        assert private_conn.state.status.value == "connected"
    finally:
        await manager.disconnect_all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_and_private_connections_independent():
    """
    Test that public and private connections operate independently.
    
    This test verifies that disconnecting one connection does not
    affect the other.
    """
    manager = ConnectionManager()
    
    try:
        public_conn = await manager.get_public_connection()
        private_conn = await manager.get_private_connection()
        
        # Disconnect public connection
        await public_conn.disconnect()
        
        # Private connection should still be connected
        assert not public_conn.is_connected
        assert private_conn.is_connected
        
        # Reconnect public
        await public_conn.connect()
        assert public_conn.is_connected
        assert private_conn.is_connected
    finally:
        await manager.disconnect_all()

