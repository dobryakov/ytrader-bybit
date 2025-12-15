"""
Integration tests for OptimizedDatasetBuilder.

Tests the complete pipeline from data loading to dataset creation.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.models.dataset import SplitStrategy, DatasetStatus
from src.models.feature_registry import FeatureRegistry, FeatureDefinition
from src.models.dataset import TargetConfig
from src.services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.services.target_registry_version_manager import TargetRegistryVersionManager


@pytest.fixture
def mock_metadata_storage():
    """Create mock metadata storage."""
    storage = AsyncMock(spec=MetadataStorage)
    storage.create_dataset = AsyncMock(return_value="test-dataset-id")
    storage.get_dataset = AsyncMock(return_value={
        "dataset_id": "test-dataset-id",
        "symbol": "BTCUSDT",
        "status": DatasetStatus.BUILDING.value,
        "train_period_start": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "train_period_end": datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        "validation_period_start": datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        "validation_period_end": datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        "test_period_start": datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        "test_period_end": datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        "split_strategy": SplitStrategy.TIME_BASED.value,
        "output_format": "parquet",
    })
    storage.update_dataset = AsyncMock()
    storage.list_datasets = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_parquet_storage(tmp_path):
    """Create mock parquet storage with sample data."""
    storage = AsyncMock(spec=ParquetStorage)
    
    # Create sample klines data
    # NOTE: Keep data volume small to make the integration test fast.
    # We only need enough data to verify the pipeline wiring, not performance.
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    klines_data = []
    for day in range(1):  # 1 day of data is enough for tests
        for minute in range(60):  # 60 minutes per day (instead of full 1440)
            timestamp = base_time + timedelta(days=day, minutes=minute)
            klines_data.append({
                "timestamp": timestamp,
                "open": 50000.0 + day * 100,
                "high": 50100.0 + day * 100,
                "low": 49900.0 + day * 100,
                "close": 50050.0 + day * 100,
                "volume": 100.0,
            })
    
    klines_df = pd.DataFrame(klines_data)
    
    # Mock read_klines to return data for any date
    async def read_klines(symbol, date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        start_time = datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc)
        end_time = datetime.combine(date_obj, datetime.max.time(), tzinfo=timezone.utc)
        return klines_df[
            (klines_df["timestamp"] >= start_time) &
            (klines_df["timestamp"] <= end_time)
        ].copy()
    
    async def read_klines_range(symbol, start_date, end_date):
        start_time = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        return klines_df[
            (klines_df["timestamp"] >= start_time) &
            (klines_df["timestamp"] <= end_time)
        ].copy()
    
    storage.read_klines = AsyncMock(side_effect=read_klines)
    storage.read_klines_range = AsyncMock(side_effect=read_klines_range)
    storage.read_trades = AsyncMock(return_value=pd.DataFrame())
    storage.read_orderbook_snapshots = AsyncMock(return_value=pd.DataFrame())
    storage.read_orderbook_deltas = AsyncMock(return_value=pd.DataFrame())
    storage.read_funding = AsyncMock(return_value=pd.DataFrame())
    storage.read_ticker = AsyncMock(return_value=pd.DataFrame())
    
    return storage


@pytest.fixture
def mock_feature_registry_loader():
    """Create mock feature registry loader."""
    loader = AsyncMock()
    
    # Create sample feature registry
    features = [
        FeatureDefinition(
            name="returns_1m",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
        FeatureDefinition(
            name="ema_21",
            input_sources=["kline"],
            lookback_window="21m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
    ]
    
    registry = FeatureRegistry(version="1.0.0", features=features)
    
    loader.load_async = AsyncMock()
    loader._registry_model = registry
    loader.get_config = MagicMock(return_value={"version": "1.0.0", "features": []})
    
    return loader


@pytest.fixture
def mock_target_registry_version_manager():
    """Create mock target registry version manager."""
    manager = AsyncMock(spec=TargetRegistryVersionManager)
    
    target_config_dict = {
        "type": "regression",
        "horizon": 300,  # 5 minutes
        "computation": {
            "preset": "returns",
            "options": {},
        },
    }
    
    manager.get_version = AsyncMock(return_value=target_config_dict)
    
    return manager


@pytest.fixture
def mock_dataset_publisher():
    """Create mock dataset publisher."""
    publisher = AsyncMock()
    publisher.publish_dataset_ready = AsyncMock()
    return publisher


@pytest.fixture
def optimized_builder(
    mock_metadata_storage,
    mock_parquet_storage,
    mock_feature_registry_loader,
    mock_target_registry_version_manager,
    mock_dataset_publisher,
    tmp_path,
):
    """Create OptimizedDatasetBuilder instance."""
    dataset_storage_path = str(tmp_path / "datasets")
    
    return OptimizedDatasetBuilder(
        metadata_storage=mock_metadata_storage,
        parquet_storage=mock_parquet_storage,
        dataset_storage_path=dataset_storage_path,
        cache_service=None,  # Disable cache for tests
        feature_registry_loader=mock_feature_registry_loader,
        target_registry_version_manager=mock_target_registry_version_manager,
        dataset_publisher=mock_dataset_publisher,
        batch_size=100,
    )


@pytest.mark.asyncio
async def test_build_dataset_time_based(optimized_builder):
    """Test building dataset with time-based split strategy."""
    dataset_id = await optimized_builder.build_dataset(
        symbol="BTCUSDT",
        split_strategy=SplitStrategy.TIME_BASED,
        target_registry_version="1.0.0",
        train_period_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        train_period_end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        validation_period_start=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        validation_period_end=datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        test_period_start=datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        test_period_end=datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        output_format="parquet",
        feature_registry_version="1.0.0",
    )
    
    assert dataset_id == "test-dataset-id"
    
    # We don't wait for background build completion here.
    # Just verify that build was initiated and metadata record created.
    optimized_builder._metadata_storage.create_dataset.assert_called_once()


@pytest.mark.asyncio
async def test_build_dataset_walk_forward(optimized_builder):
    """Test building dataset with walk-forward split strategy."""
    walk_forward_config = {
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-04T00:00:00Z",
        "train_window_days": 1,
        "validation_window_days": 1,
        "test_window_days": 1,
        "step_days": 1,
    }
    
    dataset_id = await optimized_builder.build_dataset(
        symbol="BTCUSDT",
        split_strategy=SplitStrategy.WALK_FORWARD,
        target_registry_version="1.0.0",
        walk_forward_config=walk_forward_config,
        output_format="parquet",
        feature_registry_version="1.0.0",
    )
    
    assert dataset_id == "test-dataset-id"
    optimized_builder._metadata_storage.create_dataset.assert_called_once()


@pytest.mark.asyncio
async def test_get_build_progress(optimized_builder):
    """Test getting build progress."""
    progress = await optimized_builder.get_build_progress("test-dataset-id")
    
    assert progress is not None
    assert "status" in progress
    assert progress["status"] == DatasetStatus.BUILDING.value


@pytest.mark.asyncio
async def test_build_dataset_with_cache(
    mock_metadata_storage,
    mock_parquet_storage,
    mock_feature_registry_loader,
    mock_target_registry_version_manager,
    mock_dataset_publisher,
    tmp_path,
):
    """Test building dataset with cache enabled."""
    # Create mock cache service
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    
    dataset_storage_path = str(tmp_path / "datasets")
    
    builder = OptimizedDatasetBuilder(
        metadata_storage=mock_metadata_storage,
        parquet_storage=mock_parquet_storage,
        dataset_storage_path=dataset_storage_path,
        cache_service=mock_cache,
        feature_registry_loader=mock_feature_registry_loader,
        target_registry_version_manager=mock_target_registry_version_manager,
        dataset_publisher=mock_dataset_publisher,
        batch_size=100,
    )
    
    dataset_id = await builder.build_dataset(
        symbol="BTCUSDT",
        split_strategy=SplitStrategy.TIME_BASED,
        target_registry_version="1.0.0",
        train_period_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        train_period_end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        validation_period_start=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        validation_period_end=datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        test_period_start=datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        test_period_end=datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        output_format="parquet",
        feature_registry_version="1.0.0",
    )
    
    assert dataset_id == "test-dataset-id"
    # Cache should be used if enabled
    # (In real scenario, cache.get would be called during data loading)

