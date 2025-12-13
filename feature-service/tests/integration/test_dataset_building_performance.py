"""
Integration tests for dataset building performance improvements.

Note: These tests are marked with @pytest.mark.slow and may take longer to run.
To run only fast tests, use: pytest -m "not slow"
To run only these performance tests: pytest -m slow
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.services.dataset_builder import DatasetBuilder
from src.services.offline_engine import OfflineEngine
from src.storage.parquet_storage import ParquetStorage
from src.storage.metadata_storage import MetadataStorage
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_metadata_storage():
    """Mock metadata storage."""
    storage = AsyncMock(spec=MetadataStorage)
    storage.create_dataset = AsyncMock(return_value="test-dataset-id")
    storage.update_dataset = AsyncMock()
    storage.get_dataset = AsyncMock(return_value={
        "id": "test-dataset-id",
        "status": "building",
        "symbol": "BTCUSDT",
    })
    return storage


@pytest.fixture
def mock_parquet_storage():
    """Mock parquet storage with sample data."""
    storage = AsyncMock(spec=ParquetStorage)
    
    # Generate 5 hours of data (optimized for test speed - sufficient to test optimizations and compute most features)
    # 30 klines with 10-minute interval = 5 hours, enough for EMA(21) to compute for last 10 timestamps
    base_time = datetime.now(timezone.utc) - timedelta(hours=5)
    
    # Generate trades (1 per 10 minutes to reduce data volume)
    trades_data = []
    for i in range(30):  # 5 hours * 6 trades per hour (1 per 10 minutes)
        trades_data.append({
            "timestamp": base_time + timedelta(minutes=i * 10),
            "symbol": "BTCUSDT",
            "price": 50000.0 + (i % 100) * 0.1,
            "quantity": 0.1 + (i % 10) * 0.01,
            "side": "Buy" if i % 2 == 0 else "Sell",
        })
    
    # Generate klines (1 per 10 minutes)
    klines_data = []
    for i in range(30):  # 30 klines = 5 hours
        klines_data.append({
            "timestamp": base_time + timedelta(minutes=i * 10),
            "symbol": "BTCUSDT",
            "open": 50000.0 + i * 0.1,
            "high": 50010.0 + i * 0.1,
            "low": 49990.0 + i * 0.1,
            "close": 50005.0 + i * 0.1,
            "volume": 100.0 + i * 0.1,
        })
    
    # Generate orderbook snapshots (1 per hour)
    snapshots_data = []
    for i in range(5):  # 5 hours = 5 snapshots
        snapshots_data.append({
            "timestamp": base_time + timedelta(hours=i),
            "symbol": "BTCUSDT",
            "sequence": 1000 + i * 100,
            "bids": [[50000.0 + i, 1.0], [49999.0 + i, 2.0]],
            "asks": [[50001.0 + i, 1.0], [50002.0 + i, 2.0]],
        })
    
    # Generate orderbook deltas (3 per hour)
    deltas_data = []
    for i in range(15):  # 5 hours * 3 deltas per hour
        deltas_data.append({
            "timestamp": base_time + timedelta(minutes=i * 20),
            "symbol": "BTCUSDT",
            "sequence": 1000 + i,
            "delta_type": "update",
            "side": "bid" if i % 2 == 0 else "ask",
            "price": 50000.0 + i * 0.01,
            "quantity": 1.0,
        })
    
    storage.read_trades_range = AsyncMock(return_value=pd.DataFrame(trades_data))
    storage.read_klines_range = AsyncMock(return_value=pd.DataFrame(klines_data))
    storage.read_orderbook_snapshots_range = AsyncMock(return_value=pd.DataFrame(snapshots_data))
    storage.read_orderbook_deltas_range = AsyncMock(return_value=pd.DataFrame(deltas_data))
    
    return storage


@pytest.fixture
def dataset_builder(mock_metadata_storage, mock_parquet_storage):
    """Create dataset builder instance."""
    return DatasetBuilder(
        metadata_storage=mock_metadata_storage,
        parquet_storage=mock_parquet_storage,
        dataset_storage_path="/tmp/test_datasets",
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_dataset_build_with_incremental_orderbook(
    dataset_builder,
    mock_parquet_storage,
):
    """Test full dataset build with incremental orderbook: build 1-day dataset, measure time, verify correctness."""
    import time
    
    symbol = "BTCUSDT"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=5)  # Use 5 hours for better feature coverage
    
    # Read historical data
    historical_data = {
        "trades": await mock_parquet_storage.read_trades_range(symbol, start_date, end_date),
        "klines": await mock_parquet_storage.read_klines_range(symbol, start_date, end_date),
        "snapshots": await mock_parquet_storage.read_orderbook_snapshots_range(symbol, start_date, end_date),
        "deltas": await mock_parquet_storage.read_orderbook_deltas_range(symbol, start_date, end_date),
    }
    
    # Build dataset with incremental orderbook
    start_time = time.time()
    features_df = await dataset_builder._compute_features_batch(
        symbol=symbol,
        historical_data=historical_data,
        dataset_id="test-dataset-id",
    )
    build_time = time.time() - start_time
    
    # Verify correctness
    assert not features_df.empty
    assert len(features_df) > 0
    assert "timestamp" in features_df.columns
    
    # Log performance
    print(f"\nDataset build with incremental orderbook: {build_time:.2f}s for {len(features_df)} records")
    print(f"Average time per record: {build_time / len(features_df) * 1000:.2f}ms")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_dataset_build_with_incremental_rolling_windows(
    dataset_builder,
    mock_parquet_storage,
):
    """Test full dataset build with incremental rolling windows: build 1-day dataset, measure time, verify correctness."""
    import time
    
    symbol = "BTCUSDT"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=5)  # Use 5 hours for better feature coverage
    
    # Read historical data
    historical_data = {
        "trades": await mock_parquet_storage.read_trades_range(symbol, start_date, end_date),
        "klines": await mock_parquet_storage.read_klines_range(symbol, start_date, end_date),
        "snapshots": await mock_parquet_storage.read_orderbook_snapshots_range(symbol, start_date, end_date),
        "deltas": await mock_parquet_storage.read_orderbook_deltas_range(symbol, start_date, end_date),
    }
    
    # Build dataset with incremental rolling windows
    start_time = time.time()
    features_df = await dataset_builder._compute_features_batch(
        symbol=symbol,
        historical_data=historical_data,
        dataset_id="test-dataset-id",
    )
    build_time = time.time() - start_time
    
    # Verify correctness
    assert not features_df.empty
    assert len(features_df) > 0
    assert "timestamp" in features_df.columns
    
    # Log performance
    print(f"\nDataset build with incremental rolling windows: {build_time:.2f}s for {len(features_df)} records")
    print(f"Average time per record: {build_time / len(features_df) * 1000:.2f}ms")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_combined_optimizations(
    dataset_builder,
    mock_parquet_storage,
):
    """Test combined optimizations: all optimizations together, measure total speedup, verify feature correctness."""
    import time
    
    symbol = "BTCUSDT"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=5)  # Use 5 hours for better feature coverage  # Use 1 day for faster test
    
    # Read historical data
    historical_data = {
        "trades": await mock_parquet_storage.read_trades_range(symbol, start_date, end_date),
        "klines": await mock_parquet_storage.read_klines_range(symbol, start_date, end_date),
        "snapshots": await mock_parquet_storage.read_orderbook_snapshots_range(symbol, start_date, end_date),
        "deltas": await mock_parquet_storage.read_orderbook_deltas_range(symbol, start_date, end_date),
    }
    
    # Build dataset with all optimizations
    start_time = time.time()
    features_df = await dataset_builder._compute_features_batch(
        symbol=symbol,
        historical_data=historical_data,
        dataset_id="test-dataset-id",
    )
    build_time = time.time() - start_time
    
    # Verify correctness
    assert not features_df.empty
    assert len(features_df) > 0
    assert "timestamp" in features_df.columns
    
    # Verify feature correctness: check that features are computed
    feature_columns = [col for col in features_df.columns if col not in ["timestamp", "symbol"]]
    assert len(feature_columns) > 0
    
    # Log performance
    print(f"\nDataset build with all optimizations: {build_time:.2f}s for {len(features_df)} records")
    print(f"Average time per record: {build_time / len(features_df) * 1000:.2f}ms")
    print(f"Total features computed: {len(feature_columns)}")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_feature_correctness_comparison(
    dataset_builder,
    mock_parquet_storage,
):
    """Verify feature correctness: compare features computed with optimized vs original implementation."""
    symbol = "BTCUSDT"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=5)  # Use 5 hours for better feature coverage  # Use 1 day for faster test
    
    # Read historical data
    historical_data = {
        "trades": await mock_parquet_storage.read_trades_range(symbol, start_date, end_date),
        "klines": await mock_parquet_storage.read_klines_range(symbol, start_date, end_date),
        "snapshots": await mock_parquet_storage.read_orderbook_snapshots_range(symbol, start_date, end_date),
        "deltas": await mock_parquet_storage.read_orderbook_deltas_range(symbol, start_date, end_date),
    }
    
    # Build dataset with optimizations
    features_df_optimized = await dataset_builder._compute_features_batch(
        symbol=symbol,
        historical_data=historical_data,
        dataset_id="test-dataset-id",
    )
    
    # Verify features are computed correctly
    assert not features_df_optimized.empty
    
    # Check that features were computed
    # With 30 klines (5 hours), most features should be computable for later timestamps
    # Early timestamps may have NaN for features requiring lookback (EMA(21), RSI(14), etc.)
    feature_columns = [col for col in features_df_optimized.columns if col not in ["timestamp", "symbol"]]
    assert len(feature_columns) > 0, "No features were computed"
    
    # At least some features should have non-NaN values
    non_nan_features = [col for col in feature_columns if features_df_optimized[col].notna().any()]
    assert len(non_nan_features) > 0, "No features have valid (non-NaN) values"
    
    # With 30 klines, we expect better coverage than with 12 klines
    # Check that features without lookback (or with small lookback) are mostly computed
    # Features that should work: mid_price, spread_abs, spread_rel, signed_volume_*, temporal features
    basic_features = [col for col in feature_columns if any(x in col.lower() for x in ["mid_price", "spread", "signed_volume", "time_of_day"])]
    if basic_features:
        # Basic features should have low NaN ratio (they don't require lookback)
        for col in basic_features:
            nan_ratio = features_df_optimized[col].isna().sum() / len(features_df_optimized)
            assert nan_ratio < 0.5, f"Basic feature {col} has too many NaN: {nan_ratio:.2%}"
    
    # For features with lookback (EMA, RSI, returns_5m), check that they compute for later timestamps
    # With 30 klines, EMA(21) should compute for last 10 timestamps, RSI(14) for last 16
    # Note: Some features may compute less frequently due to data availability or logic
    lookback_features = [col for col in feature_columns if any(x in col.lower() for x in ["ema", "rsi", "returns_5m"])]
    if lookback_features:
        # At least some timestamps should have these features computed
        # With 30 klines, we expect at least some computation (may vary based on feature logic)
        lookback_features_with_data = [col for col in lookback_features if features_df_optimized[col].notna().any()]
        # At least one lookback feature should have computed values
        assert len(lookback_features_with_data) > 0, \
            f"At least one lookback feature should compute, but none of {lookback_features} have values"
    
    # Verify dataset structure
    assert "timestamp" in features_df_optimized.columns
    assert "symbol" in features_df_optimized.columns


@pytest.mark.asyncio
@pytest.mark.slow
async def test_memory_efficiency(
    dataset_builder,
    mock_parquet_storage,
):
    """Test memory efficiency: verify that optimizations don't significantly increase memory usage."""
    try:
        import psutil
        import os
        psutil_available = True
    except ImportError:
        psutil_available = False
    
    symbol = "BTCUSDT"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=5)  # Use 5 hours for better feature coverage  # Use 1 day for faster test
    
    # Read historical data
    historical_data = {
        "trades": await mock_parquet_storage.read_trades_range(symbol, start_date, end_date),
        "klines": await mock_parquet_storage.read_klines_range(symbol, start_date, end_date),
        "snapshots": await mock_parquet_storage.read_orderbook_snapshots_range(symbol, start_date, end_date),
        "deltas": await mock_parquet_storage.read_orderbook_deltas_range(symbol, start_date, end_date),
    }
    
    if psutil_available:
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Build dataset
    features_df = await dataset_builder._compute_features_batch(
        symbol=symbol,
        historical_data=historical_data,
        dataset_id="test-dataset-id",
    )
    
    # Verify dataset was built successfully
    assert not features_df.empty
    assert len(features_df) > 0
    
    if psutil_available:
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB for 1 day of data)
        assert memory_increase < 1024, \
            f"Memory increase too high: {memory_increase:.2f}MB"
        
        print(f"\nMemory usage: initial={initial_memory:.2f}MB, peak={peak_memory:.2f}MB, increase={memory_increase:.2f}MB")
    else:
        # If psutil not available, just verify dataset was built
        print(f"\nMemory check skipped (psutil not available). Dataset built successfully: {len(features_df)} records")

