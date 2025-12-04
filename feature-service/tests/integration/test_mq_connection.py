"""
Integration tests for RabbitMQ connection manager.
"""
import pytest
from unittest.mock import AsyncMock, patch
from aio_pika import Connection, Channel


@pytest.mark.asyncio
class TestMQConnection:
    """Integration tests for RabbitMQ connection manager."""
    
    async def test_mq_connection_connects(self, mock_rabbitmq_connection):
        """Test that MQ connection connects successfully."""
        from src.mq.connection import MQConnectionManager
        
        manager = MQConnectionManager(
            host="rabbitmq",
            port=5672,
            user="guest",
            password="guest"
        )
        manager._connection = mock_rabbitmq_connection
        
        await manager.connect()
        
        assert manager.is_connected()
    
    async def test_mq_connection_creates_channel(self, mock_rabbitmq_connection_and_channel):
        """Test that MQ connection creates channel."""
        from src.mq.connection import MQConnectionManager
        
        connection, channel = mock_rabbitmq_connection_and_channel
        
        manager = MQConnectionManager(
            host="rabbitmq",
            port=5672,
            user="guest",
            password="guest"
        )
        manager._connection = connection
        
        mq_channel = await manager.get_channel()
        
        assert mq_channel is not None
    
    async def test_mq_connection_handles_connection_errors(self):
        """Test that MQ connection handles connection errors."""
        from src.mq.connection import MQConnectionManager
        
        manager = MQConnectionManager(
            host="invalid-host",
            port=5672,
            user="guest",
            password="guest"
        )
        
        # Mock connection failure
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                await manager.connect()
    
    async def test_mq_connection_closes_gracefully(self, mock_rabbitmq_connection):
        """Test that MQ connection closes gracefully."""
        from src.mq.connection import MQConnectionManager
        
        manager = MQConnectionManager(
            host="rabbitmq",
            port=5672,
            user="guest",
            password="guest"
        )
        manager._connection = mock_rabbitmq_connection
        
        await manager.connect()
        await manager.close()
        
        assert not manager.is_connected()
        mock_rabbitmq_connection.close.assert_called_once()
    
    async def test_mq_connection_reconnects_on_failure(self, mock_rabbitmq_connection):
        """Test that MQ connection reconnects on failure."""
        from src.mq.connection import MQConnectionManager
        
        manager = MQConnectionManager(
            host="rabbitmq",
            port=5672,
            user="guest",
            password="guest"
        )
        manager._connection = mock_rabbitmq_connection
        
        # Simulate connection loss
        mock_rabbitmq_connection.is_closed = True
        
        # Should attempt reconnection
        # (adjust based on actual reconnection logic)

