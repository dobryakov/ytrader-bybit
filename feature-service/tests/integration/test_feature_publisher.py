"""
Integration tests for feature publisher (with mocked RabbitMQ).
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.publishers.feature_publisher import FeaturePublisher
from src.models.feature_vector import FeatureVector
from src.mq.connection import MQConnectionManager


class TestFeaturePublisher:
    """Test feature publisher integration."""
    
    @pytest.fixture
    def mock_mq_manager(self):
        """Create mocked MQ connection manager."""
        manager = MagicMock(spec=MQConnectionManager)
        manager.get_channel = AsyncMock()
        return manager
    
    @pytest.fixture
    def publisher(self, mock_mq_manager):
        """Create feature publisher."""
        return FeaturePublisher(mq_manager=mock_mq_manager)
    
    @pytest.fixture
    def sample_feature_vector(self):
        """Create sample feature vector."""
        return FeatureVector(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            features={"mid_price": 50000.0, "spread_abs": 1.0},
            feature_registry_version="1.0.0",
            trace_id="test-trace",
        )
    
    @pytest.mark.asyncio
    async def test_publish_feature_vector(self, publisher, sample_feature_vector, mock_mq_manager):
        """Test publishing feature vector."""
        # Mock channel and queue
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
        mock_channel.default_exchange = MagicMock()
        mock_channel.default_exchange.publish = AsyncMock()
        mock_mq_manager.get_channel = AsyncMock(return_value=mock_channel)
        
        await publisher.initialize()
        await publisher.publish(sample_feature_vector)
        
        # Verify publish was called
        assert mock_channel.default_exchange.publish.called
    
    @pytest.mark.asyncio
    async def test_publisher_initialization(self, publisher, mock_mq_manager):
        """Test publisher initialization."""
        # Mock channel and queue
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
        mock_mq_manager.get_channel = AsyncMock(return_value=mock_channel)
        
        await publisher.initialize()
        
        assert publisher._queue is not None
        assert mock_channel.declare_queue.called

