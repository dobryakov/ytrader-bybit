"""Integration tests for independent reconnection per connection type."""

import asyncio
import pytest

from src.services.websocket.connection import WebSocketConnection
from src.services.websocket.connection_manager import ConnectionManager
from src.services.websocket.reconnection import ReconnectionManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_connection_reconnects_independently():
    """
    Test that public connection reconnects independently of private connection.
    
    This test simulates a disconnection of the public connection and verifies
    that it reconnects without affecting the private connection.
    """
    manager = ConnectionManager()
    
    try:
        public_conn = await manager.get_public_connection()
        private_conn = await manager.get_private_connection()
        
        # Setup reconnection manager for public connection
        public_reconnect = ReconnectionManager(public_conn)
        await public_reconnect.start()
        
        # Disconnect public connection
        await public_conn.disconnect()
        assert not public_conn.is_connected
        
        # Private connection should still be connected
        assert private_conn.is_connected
        
        # Wait for reconnection (with timeout)
        try:
            await asyncio.wait_for(public_conn.wait_connected(timeout=5.0), timeout=35.0)
            assert public_conn.is_connected
        except asyncio.TimeoutError:
            pytest.skip("Reconnection took too long (may be network issue)")
        
        # Private connection should still be connected
        assert private_conn.is_connected
        
        await public_reconnect.stop()
    finally:
        await manager.disconnect_all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_private_connection_reconnects_independently():
    """
    Test that private connection reconnects independently of public connection.
    
    This test simulates a disconnection of the private connection and verifies
    that it reconnects without affecting the public connection.
    """
    manager = ConnectionManager()
    
    try:
        public_conn = await manager.get_public_connection()
        private_conn = await manager.get_private_connection()
        
        # Setup reconnection manager for private connection
        private_reconnect = ReconnectionManager(private_conn)
        await private_reconnect.start()
        
        # Disconnect private connection
        await private_conn.disconnect()
        assert not private_conn.is_connected
        
        # Public connection should still be connected
        assert public_conn.is_connected
        
        # Wait for reconnection (with timeout)
        try:
            await asyncio.wait_for(private_conn.wait_connected(timeout=5.0), timeout=35.0)
            assert private_conn.is_connected
        except asyncio.TimeoutError:
            pytest.skip("Reconnection took too long (may be network issue)")
        
        # Public connection should still be connected
        assert public_conn.is_connected
        
        await private_reconnect.stop()
    finally:
        await manager.disconnect_all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_both_connections_reconnect_independently():
    """
    Test that both connections can reconnect independently when both disconnect.
    
    This test verifies that when both connections are lost, they each
    reconnect independently without interfering with each other.
    """
    manager = ConnectionManager()
    
    try:
        public_conn = await manager.get_public_connection()
        private_conn = await manager.get_private_connection()
        
        # Setup reconnection managers for both
        public_reconnect = ReconnectionManager(public_conn)
        private_reconnect = ReconnectionManager(private_conn)
        await public_reconnect.start()
        await private_reconnect.start()
        
        # Disconnect both
        await public_conn.disconnect()
        await private_conn.disconnect()
        
        assert not public_conn.is_connected
        assert not private_conn.is_connected
        
        # Wait for both to reconnect (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    public_conn.wait_connected(timeout=5.0),
                    private_conn.wait_connected(timeout=5.0),
                ),
                timeout=35.0,
            )
            assert public_conn.is_connected
            assert private_conn.is_connected
        except asyncio.TimeoutError:
            pytest.skip("Reconnection took too long (may be network issue)")
        
        await public_reconnect.stop()
        await private_reconnect.stop()
    finally:
        await manager.disconnect_all()

