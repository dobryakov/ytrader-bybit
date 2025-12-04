"""
Integration tests for market data consumer (with mocked RabbitMQ).
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.consumers.market_data_consumer import MarketDataConsumer
from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager
from src.mq.connection import MQConnectionManager
from src.http.client import HTTPClient


class TestMarketDataConsumer:
    """Test market data consumer integration."""
    
    @pytest.fixture
    def mock_mq_manager(self):
        """Create mocked MQ connection manager."""
        manager = MagicMock(spec=MQConnectionManager)
        manager.get_channel = AsyncMock()
        manager.is_connected = MagicMock(return_value=True)
        return manager
    
    @pytest.fixture
    def mock_http_client(self):
        """Create mocked HTTP client."""
        client = MagicMock(spec=HTTPClient)
        client.post = AsyncMock()
        return client
    
    @pytest.fixture
    def orderbook_manager(self):
        """Create orderbook manager."""
        return OrderbookManager()
    
    @pytest.fixture
    def feature_computer(self, orderbook_manager):
        """Create feature computer."""
        return FeatureComputer(orderbook_manager)
    
    @pytest.fixture
    def consumer(
        self,
        mock_mq_manager,
        mock_http_client,
        feature_computer,
        orderbook_manager,
    ):
        """Create market data consumer."""
        return MarketDataConsumer(
            mq_manager=mock_mq_manager,
            http_client=mock_http_client,
            feature_computer=feature_computer,
            orderbook_manager=orderbook_manager,
            symbols=["BTCUSDT"],
        )
    
    @pytest.mark.asyncio
    async def test_create_subscriptions(self, consumer, mock_http_client):
        """Test creating subscriptions via ws-gateway API."""
        # Mock successful subscription response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"subscription_id": "sub-123"})
        mock_http_client.post.return_value = mock_response
        
        await consumer._create_subscriptions()
        
        # Verify subscriptions were created
        assert len(consumer._subscriptions) > 0
        assert mock_http_client.post.called
    
    @pytest.mark.asyncio
    async def test_process_market_data_event_trade(self, consumer, sample_trade):
        """Test processing trade event."""
        event = sample_trade.copy()
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        await consumer._process_market_data_event(event, "ws-gateway.trades")
        
        # Verify event was processed (rolling windows updated)
        rw = consumer._feature_computer.get_rolling_windows("BTCUSDT")
        assert rw is not None
    
    @pytest.mark.asyncio
    async def test_process_market_data_event_orderbook_snapshot(
        self,
        consumer,
        sample_orderbook_snapshot,
    ):
        """Test processing orderbook snapshot."""
        event = sample_orderbook_snapshot.copy()
        event["timestamp"] = datetime.now(timezone.utc)
        
        await consumer._process_market_data_event(event, "ws-gateway.orderbook")
        
        # Verify orderbook was updated
        orderbook = consumer._orderbook_manager.get_orderbook("BTCUSDT")
        assert orderbook is not None
        assert orderbook.sequence == event["sequence"]
    
    @pytest.mark.asyncio
    async def test_consumer_start_stop(self, consumer, mock_mq_manager):
        """Test starting and stopping consumer."""
        # Mock channel and queue
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
        mock_queue.consume = AsyncMock()
        mock_mq_manager.get_channel = AsyncMock(return_value=mock_channel)
        
        # Start consumer
        start_task = asyncio.create_task(consumer.start())
        await asyncio.sleep(0.1)  # Let it start
        
        # Stop consumer
        await consumer.stop()
        await start_task
        
        # Verify consumer stopped
        assert not consumer._running

