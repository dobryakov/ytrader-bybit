"""Unit tests for SignalProcessor with FIFO queue processing."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from uuid import uuid4
from datetime import datetime

from src.services.signal_processor import SignalProcessor
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.models.order import Order
from src.exceptions import OrderExecutionError
from src.config.settings import settings


@pytest.fixture
def sample_signal():
    """Create a sample trading signal for testing."""
    return TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="BTCUSDT",
        amount=Decimal("1000.0"),
        confidence=Decimal("0.85"),
        timestamp=datetime.utcnow(),
        strategy_id="test-strategy",
        model_version="v1.0",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("50000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id="test-trace-001",
    )


@pytest.fixture
def mock_order():
    """Create a mock order for testing."""
    return Order(
        id=uuid4(),
        order_id="bybit-test-123",
        signal_id=uuid4(),
        asset="BTCUSDT",
        side="Buy",
        order_type="Limit",
        quantity=Decimal("0.02"),
        price=Decimal("50000.0"),
        status="pending",
        filled_quantity=Decimal("0"),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        trace_id="test-trace",
        is_dry_run=False,
    )


@pytest.mark.asyncio
async def test_process_signal_creates_fifo_queue(sample_signal):
    """Test that process_signal creates FIFO queue for new asset."""
    processor = SignalProcessor()
    
    # Mock all dependencies
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', new_callable=AsyncMock) as mock_process:
        
        mock_process.return_value = None
        
        # Start processing in background
        task = asyncio.create_task(processor.process_signal(sample_signal))
        
        # Wait a bit for queue to be created
        await asyncio.sleep(0.1)
        
        # Check that queue was created
        assert sample_signal.asset in processor._signal_queues
        assert sample_signal.asset in processor._processing_tasks
        
        # Cancel task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_process_signal_uses_fifo_queue(sample_signal, mock_order):
    """Test that signals are processed through FIFO queue."""
    processor = SignalProcessor()
    
    # Mock dependencies
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', new_callable=AsyncMock) as mock_process:
        
        mock_process.return_value = mock_order
        
        # Process signal
        result = await processor.process_signal(sample_signal)
        
        # Verify signal was processed
        assert result == mock_order
        mock_process.assert_called_once()
        
        # Check that future was cleaned up
        assert sample_signal.signal_id not in processor._signal_futures
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_sequential_processing_same_asset(sample_signal, mock_order):
    """Test that signals for the same asset are processed sequentially."""
    processor = SignalProcessor()
    
    # Create second signal with same asset
    signal2 = TradingSignal(
        signal_id=uuid4(),
        signal_type="sell",
        asset=sample_signal.asset,  # Same asset
        amount=Decimal("500.0"),
        confidence=Decimal("0.75"),
        timestamp=datetime.utcnow(),
        strategy_id="test-strategy",
        model_version="v1.0",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("51000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id="test-trace-002",
    )
    
    processing_order = []
    
    async def mock_process_with_order(signal):
        processing_order.append(signal.signal_id)
        # Simulate processing time
        await asyncio.sleep(0.1)
        return mock_order
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', side_effect=mock_process_with_order):
        
        # Process both signals concurrently
        task1 = asyncio.create_task(processor.process_signal(sample_signal))
        await asyncio.sleep(0.01)  # Small delay to ensure first signal enters queue first
        task2 = asyncio.create_task(processor.process_signal(signal2))
        
        results = await asyncio.gather(task1, task2)
        
        # Verify both completed
        assert all(r == mock_order for r in results)
        
        # Verify sequential processing - first signal should be processed first
        assert len(processing_order) == 2
        assert processing_order[0] == sample_signal.signal_id
        assert processing_order[1] == signal2.signal_id
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_parallel_processing_different_assets(sample_signal, mock_order):
    """Test that signals for different assets are processed in parallel."""
    processor = SignalProcessor()
    
    # Create signal with different asset
    signal2 = TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="ETHUSDT",  # Different asset
        amount=Decimal("500.0"),
        confidence=Decimal("0.75"),
        timestamp=datetime.utcnow(),
        strategy_id="test-strategy",
        model_version="v1.0",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("3000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id="test-trace-002",
    )
    
    processing_started = {}
    
    async def mock_process_with_timing(signal):
        processing_started[signal.signal_id] = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        return mock_order
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', side_effect=mock_process_with_timing):
        
        # Process both signals concurrently
        start_time = asyncio.get_event_loop().time()
        task1 = asyncio.create_task(processor.process_signal(sample_signal))
        task2 = asyncio.create_task(processor.process_signal(signal2))
        
        results = await asyncio.gather(task1, task2)
        end_time = asyncio.get_event_loop().time()
        
        # Verify both completed
        assert all(r == mock_order for r in results)
        
        # Verify parallel processing - total time should be ~0.1s, not ~0.2s
        # (with some tolerance for overhead)
        assert end_time - start_time < 0.15
        
        # Both should start processing around the same time
        assert abs(processing_started[sample_signal.signal_id] - processing_started[signal2.signal_id]) < 0.05
        
        # Clean up
        for asset in [sample_signal.asset, signal2.asset]:
            if asset in processor._processing_tasks:
                processor._processing_tasks[asset].cancel()
                try:
                    await processor._processing_tasks[asset]
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_process_signal_timeout(sample_signal):
    """Test that process_signal raises timeout error when processing exceeds timeout."""
    processor = SignalProcessor()
    
    # Set short timeout for testing
    original_timeout = settings.order_manager_signal_processing_timeout_seconds
    settings.order_manager_signal_processing_timeout_seconds = 0.1
    
    try:
        async def slow_process(signal):
            await asyncio.sleep(1.0)  # Longer than timeout
            return None
        
        with patch.object(processor, '_validate_signal'), \
             patch.object(processor, '_process_signal_internal', side_effect=slow_process):
            
            with pytest.raises(OrderExecutionError) as exc_info:
                await processor.process_signal(sample_signal)
            
            assert "timeout" in str(exc_info.value).lower()
            
            # Verify future was cleaned up
            assert sample_signal.signal_id not in processor._signal_futures
            
    finally:
        settings.order_manager_signal_processing_timeout_seconds = original_timeout
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_process_signal_exception_propagation(sample_signal):
    """Test that exceptions during processing are propagated correctly."""
    processor = SignalProcessor()
    
    test_exception = ValueError("Test error")
    
    async def failing_process(signal):
        raise test_exception
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', side_effect=failing_process):
        
        with pytest.raises(ValueError) as exc_info:
            await processor.process_signal(sample_signal)
        
        assert exc_info.value == test_exception
        
        # Verify future was cleaned up
        assert sample_signal.signal_id not in processor._signal_futures
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_future_cleanup_on_success(sample_signal, mock_order):
    """Test that Future is cleaned up after successful processing."""
    processor = SignalProcessor()
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', new_callable=AsyncMock) as mock_process:
        
        mock_process.return_value = mock_order
        
        # Verify future exists before processing
        assert sample_signal.signal_id not in processor._signal_futures
        
        # Process signal
        await processor.process_signal(sample_signal)
        
        # Verify future was cleaned up after processing
        assert sample_signal.signal_id not in processor._signal_futures
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_future_cleanup_on_exception(sample_signal):
    """Test that Future is cleaned up even when exception occurs."""
    processor = SignalProcessor()
    
    test_exception = RuntimeError("Processing failed")
    
    async def failing_process(signal):
        raise test_exception
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', side_effect=failing_process):
        
        try:
            await processor.process_signal(sample_signal)
        except RuntimeError:
            pass
        
        # Verify future was cleaned up even after exception
        assert sample_signal.signal_id not in processor._signal_futures
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_multiple_signals_fifo_order(sample_signal, mock_order):
    """Test that multiple signals are processed in FIFO order."""
    processor = SignalProcessor()
    
    # Create multiple signals with same asset
    signals = [sample_signal]
    for i in range(3):
        sig = TradingSignal(
            signal_id=uuid4(),
            signal_type="buy" if i % 2 == 0 else "sell",
            asset=sample_signal.asset,
            amount=Decimal(f"{100 + i * 100}.0"),
            confidence=Decimal("0.75"),
            timestamp=datetime.utcnow(),
            strategy_id="test-strategy",
            model_version="v1.0",
            is_warmup=False,
            market_data_snapshot=MarketDataSnapshot(
                price=Decimal("50000.0"),
                spread=Decimal("0.01"),
                volume_24h=Decimal("1000000.0"),
                volatility=Decimal("0.02"),
            ),
            metadata=None,
            trace_id=f"test-trace-{i+3}",
        )
        signals.append(sig)
    
    processing_order = []
    
    async def mock_process_with_order(signal):
        processing_order.append(signal.signal_id)
        await asyncio.sleep(0.05)  # Small delay to ensure ordering
        return mock_order
    
    with patch.object(processor, '_validate_signal'), \
         patch.object(processor, '_process_signal_internal', side_effect=mock_process_with_order):
        
        # Process all signals concurrently (they should queue)
        tasks = [asyncio.create_task(processor.process_signal(sig)) for sig in signals]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert all(r == mock_order for r in results)
        
        # Verify FIFO order - signals should be processed in the order they were submitted
        assert len(processing_order) == len(signals)
        assert processing_order == [sig.signal_id for sig in signals]
        
        # Clean up
        if sample_signal.asset in processor._processing_tasks:
            processor._processing_tasks[sample_signal.asset].cancel()
            try:
                await processor._processing_tasks[sample_signal.asset]
            except asyncio.CancelledError:
                pass

