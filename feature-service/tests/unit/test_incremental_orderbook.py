"""
Unit tests for IncrementalOrderbookManager.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.services.optimized_dataset.incremental_orderbook import IncrementalOrderbookManager
from src.models.orderbook_state import OrderbookState


@pytest.fixture
def orderbook_manager():
    """Create IncrementalOrderbookManager instance."""
    return IncrementalOrderbookManager(
        symbol="BTCUSDT",
        snapshot_refresh_interval=3600,
        max_delta_count=10000,
    )


@pytest.fixture
def sample_snapshot():
    """Create sample orderbook snapshot."""
    return pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
        "sequence": [1000],
        "bids": [[[50000.0, 1.0], [49999.0, 2.0]]],
        "asks": [[[50001.0, 1.0], [50002.0, 2.0]]],
    })


@pytest.fixture
def sample_deltas():
    """Create sample orderbook deltas."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i) for i in range(1, 6)],
        "sequence": [1000 + i for i in range(1, 6)],
        "delta_type": ["update"] * 5,
        "side": ["bid"] * 5,
        "price": [50000.0] * 5,
        "quantity": [1.5] * 5,
    })


def test_initialization(orderbook_manager):
    """Test orderbook manager initialization."""
    assert orderbook_manager.symbol == "BTCUSDT"
    assert orderbook_manager.snapshot_refresh_interval == 3600
    assert orderbook_manager.max_delta_count == 10000
    assert orderbook_manager.current_state is None
    assert orderbook_manager.last_processed_timestamp is None


def test_update_to_timestamp_initial(orderbook_manager, sample_snapshot, sample_deltas):
    """Test initial update to timestamp."""
    target_time = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
    
    state = orderbook_manager.update_to_timestamp(
        timestamp=target_time,
        snapshots=sample_snapshot,
        deltas=sample_deltas,
    )
    
    assert state is not None
    assert isinstance(state, OrderbookState)
    assert orderbook_manager.current_state is not None
    assert orderbook_manager.last_processed_timestamp == target_time


def test_update_to_timestamp_incremental(orderbook_manager, sample_snapshot, sample_deltas):
    """Test incremental update to timestamp."""
    # Initial update
    target_time1 = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
    state1 = orderbook_manager.update_to_timestamp(
        timestamp=target_time1,
        snapshots=sample_snapshot,
        deltas=sample_deltas,
    )
    
    assert state1 is not None
    
    # Incremental update
    new_deltas = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 12, 0, 6, tzinfo=timezone.utc)],
        "sequence": [1006],
        "delta_type": ["update"],
        "side": ["bid"],
        "price": [50000.0],
        "quantity": [2.0],
    })
    
    target_time2 = datetime(2024, 1, 1, 12, 0, 6, tzinfo=timezone.utc)
    state2 = orderbook_manager.update_to_timestamp(
        timestamp=target_time2,
        snapshots=sample_snapshot,
        deltas=pd.concat([sample_deltas, new_deltas]),
    )
    
    assert state2 is not None
    assert orderbook_manager.last_processed_timestamp == target_time2


def test_snapshot_refresh_needed(orderbook_manager, sample_snapshot, sample_deltas):
    """Test snapshot refresh when interval exceeded."""
    # Initial update
    target_time1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    orderbook_manager.update_to_timestamp(
        timestamp=target_time1,
        snapshots=sample_snapshot,
        deltas=sample_deltas,
    )
    
    # Update after refresh interval (1 hour)
    target_time2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    new_snapshot = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)],
        "sequence": [2000],
        "bids": [[[51000.0, 1.0], [50999.0, 2.0]]],
        "asks": [[[51001.0, 1.0], [51002.0, 2.0]]],
    })
    
    state = orderbook_manager.update_to_timestamp(
        timestamp=target_time2,
        snapshots=new_snapshot,
        deltas=pd.DataFrame(),  # Empty deltas
    )
    
    assert state is not None
    # Should have refreshed from snapshot
    assert orderbook_manager.current_state.last_snapshot_at == datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)


def test_max_delta_count_refresh(orderbook_manager, sample_snapshot):
    """Test snapshot refresh when delta count exceeds limit."""
    # Create many deltas
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    many_deltas = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i) for i in range(10001)],
        "sequence": [1000 + i for i in range(10001)],
        "delta_type": ["update"] * 10001,
        "side": ["bid"] * 10001,
        "price": [50000.0] * 10001,
        "quantity": [1.0] * 10001,
    })
    
    target_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Initial update
    orderbook_manager.update_to_timestamp(
        timestamp=target_time,
        snapshots=sample_snapshot,
        deltas=many_deltas.iloc[:10000],
    )
    
    # Add one more delta to exceed limit
    target_time2 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    state = orderbook_manager.update_to_timestamp(
        timestamp=target_time2,
        snapshots=sample_snapshot,
        deltas=many_deltas,
    )
    
    # Should have refreshed due to delta count
    assert state is not None

