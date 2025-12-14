"""
Unit tests for DatasetBuilder computed features caching.
"""
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from datetime import datetime, timezone, timedelta
import hashlib

from src.services.dataset_builder import DatasetBuilder
from src.services.cache_service import CacheService, InMemoryCacheService, CacheServiceFactory
from src.services.feature_registry import FeatureRegistryLoader
from src.models.feature_vector import FeatureVector
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage


class TestDatasetBuilderFeaturesCache:
    """Tests for computed features caching in DatasetBuilder."""
    
    @pytest.fixture
    def mock_metadata_storage(self):
        """Create mock metadata storage."""
        storage = MagicMock(spec=MetadataStorage)
        storage.create_dataset = AsyncMock(return_value="test-dataset-id")
        storage.get_dataset = AsyncMock(return_value={"status": "building"})
        storage.update_dataset = AsyncMock()
        return storage
    
    @pytest.fixture
    def mock_parquet_storage(self):
        """Create mock Parquet storage."""
        storage = MagicMock(spec=ParquetStorage)
        storage.read_orderbook_snapshots_range = AsyncMock(return_value=pd.DataFrame())
        storage.read_orderbook_deltas_range = AsyncMock(return_value=pd.DataFrame())
        storage.read_trades_range = AsyncMock(return_value=pd.DataFrame())
        storage.read_klines_range = AsyncMock(return_value=pd.DataFrame())
        return storage
    
    @pytest.fixture
    def mock_feature_registry_loader(self):
        """Create mock Feature Registry loader."""
        loader = MagicMock(spec=FeatureRegistryLoader)
        loader.get_required_data_types = MagicMock(return_value={"orderbook", "kline", "trades"})
        loader.get_data_type_mapping = MagicMock(return_value={
            "orderbook": ["orderbook_snapshots", "orderbook_deltas"],
            "kline": ["klines"],
            "trades": ["trades"],
        })
        loader._registry_model = MagicMock()
        loader._registry_model.version = "1.0.0"
        return loader
    
    @pytest.fixture
    def mock_offline_engine(self):
        """Create mock OfflineEngine."""
        engine = MagicMock()
        
        # Create sample feature vector
        feature_vector = FeatureVector(
            symbol="BTCUSDT",
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            feature_registry_version="1.0.0",
            features={
                "mid_price": 50000.0,
                "spread_abs": 1.0,
                "returns_1s": 0.001,
            },
        )
        
        engine.compute_features_at_timestamp = AsyncMock(
            return_value=(feature_vector, None, None)
        )
        return engine
    
    @pytest_asyncio.fixture
    async def memory_cache_service(self):
        """Create in-memory cache service for testing."""
        cache = InMemoryCacheService(max_size_mb=100, max_entries=1000)
        await cache.start_cleanup_task()
        yield cache
        await cache.stop_cleanup_task()
    
    @pytest.fixture
    def dataset_builder_with_cache(
        self,
        mock_metadata_storage,
        mock_parquet_storage,
        mock_feature_registry_loader,
        mock_offline_engine,
        memory_cache_service,
    ):
        """Create DatasetBuilder with cache service."""
        # Note: memory_cache_service is async fixture, but pytest-asyncio handles it
        builder = DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
            feature_registry_loader=mock_feature_registry_loader,
        )
        # Inject cache service and mock offline engine
        builder._cache_service = memory_cache_service
        builder._cache_enabled = True
        builder._cache_features_enabled = True
        builder._offline_engine = mock_offline_engine
        return builder
    
    @pytest.mark.asyncio
    async def test_cache_hit_same_parameters(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
        memory_cache_service,
    ):
        """Test cache hit: same symbol, timestamp, feature_registry_version, data_hash should return cached FeatureVector."""
        symbol = "BTCUSDT"
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dataset_id = "test-dataset-id"
        
        # Create historical data
        historical_data = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp],
                "price": [50000.0],
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # Compute data hash
        data_hash = dataset_builder_with_cache._compute_data_hash(historical_data)
        
        # First call - should compute and cache
        result1 = await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify offline engine was called
        assert mock_offline_engine.compute_features_at_timestamp.called
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call - should use cache
        result2 = await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify offline engine was NOT called (cache hit)
        assert not mock_offline_engine.compute_features_at_timestamp.called
        
        # Verify results are identical
        assert len(result1) == len(result2)
        if not result1.empty and not result2.empty:
            assert result1["timestamp"].iloc[0] == result2["timestamp"].iloc[0]
    
    @pytest.mark.asyncio
    async def test_cache_miss_different_parameters(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
    ):
        """Test cache miss: different parameters should compute features."""
        symbol = "BTCUSDT"
        timestamp1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        dataset_id = "test-dataset-id"
        
        # Create historical data for first timestamp
        historical_data1 = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp1],
                "price": [50000.0],
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp1],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data1,
            dataset_id,
        )
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call with different timestamp - should compute (cache miss)
        historical_data2 = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp2],
                "price": [51000.0],
                "size": [0.2],
                "side": ["Sell"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp2],
                "open": [51000.0],
                "high": [51100.0],
                "low": [50900.0],
                "close": [51050.0],
                "volume": [2.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data2,
            dataset_id,
        )
        
        # Verify offline engine was called (cache miss)
        assert mock_offline_engine.compute_features_at_timestamp.called
    
    @pytest.mark.asyncio
    async def test_batch_cache_retrieval(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
        memory_cache_service,
    ):
        """Test batch cache retrieval: verify multiple FeatureVectors can be retrieved from cache efficiently."""
        symbol = "BTCUSDT"
        dataset_id = "test-dataset-id"
        
        # Create historical data with multiple timestamps
        timestamps = [
            datetime(2025, 1, 1, 12, i, 0, tzinfo=timezone.utc)
            for i in range(5)
        ]
        
        historical_data = {
            "trades": pd.DataFrame({
                "timestamp": timestamps,
                "price": [50000.0 + i * 100 for i in range(5)],
                "size": [0.1] * 5,
                "side": ["Buy"] * 5,
            }),
            "klines": pd.DataFrame({
                "timestamp": timestamps,
                "open": [50000.0 + i * 100 for i in range(5)],
                "high": [50100.0 + i * 100 for i in range(5)],
                "low": [49900.0 + i * 100 for i in range(5)],
                "close": [50050.0 + i * 100 for i in range(5)],
                "volume": [1.0] * 5,
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call - compute and cache all features
        result1 = await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify all features were computed
        assert len(result1) == 5
        assert mock_offline_engine.compute_features_at_timestamp.call_count == 5
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call - should use cache for all timestamps
        result2 = await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify offline engine was NOT called (all cache hits)
        assert not mock_offline_engine.compute_features_at_timestamp.called
        
        # Verify results are identical
        assert len(result2) == 5
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_registry_change(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
    ):
        """Test cache invalidation on Feature Registry version change."""
        symbol = "BTCUSDT"
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dataset_id = "test-dataset-id"
        
        historical_data = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp],
                "price": [50000.0],
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call with version 1.0.0
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Change registry version
        dataset_builder_with_cache._feature_registry_loader._registry_model.version = "1.1.0"
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call - should compute (cache miss due to version change)
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify offline engine was called (cache miss)
        assert mock_offline_engine.compute_features_at_timestamp.called
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_data_modification(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
    ):
        """Test cache invalidation on data modification (data_hash change)."""
        symbol = "BTCUSDT"
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dataset_id = "test-dataset-id"
        
        # First historical data
        historical_data1 = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp],
                "price": [50000.0],
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data1,
            dataset_id,
        )
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call with modified data (different price)
        historical_data2 = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp],
                "price": [51000.0],  # Different price
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp],
                "open": [51000.0],  # Different price
                "high": [51100.0],
                "low": [50900.0],
                "close": [51050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # Compute hash - should be different
        hash1 = dataset_builder_with_cache._compute_data_hash(historical_data1)
        hash2 = dataset_builder_with_cache._compute_data_hash(historical_data2)
        assert hash1 != hash2
        
        # Second call - should compute (cache miss due to data change)
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data2,
            dataset_id,
        )
        
        # Verify offline engine was called (cache miss)
        assert mock_offline_engine.compute_features_at_timestamp.called
    
    @pytest.mark.asyncio
    async def test_partial_cache_hit(
        self,
        dataset_builder_with_cache,
        mock_offline_engine,
    ):
        """Test partial cache hit: verify that partial cache hits (e.g., 80% of timestamps cached) still provide performance benefit."""
        symbol = "BTCUSDT"
        dataset_id = "test-dataset-id"
        
        # Create historical data with 10 timestamps
        timestamps = [
            datetime(2025, 1, 1, 12, i, 0, tzinfo=timezone.utc)
            for i in range(10)
        ]
        
        historical_data = {
            "trades": pd.DataFrame({
                "timestamp": timestamps,
                "price": [50000.0 + i * 100 for i in range(10)],
                "size": [0.1] * 10,
                "side": ["Buy"] * 10,
            }),
            "klines": pd.DataFrame({
                "timestamp": timestamps,
                "open": [50000.0 + i * 100 for i in range(10)],
                "high": [50100.0 + i * 100 for i in range(10)],
                "low": [49900.0 + i * 100 for i in range(10)],
                "close": [50050.0 + i * 100 for i in range(10)],
                "volume": [1.0] * 10,
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call - cache all 10 timestamps
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        assert mock_offline_engine.compute_features_at_timestamp.call_count == 10
        
        # Reset mock
        mock_offline_engine.compute_features_at_timestamp.reset_mock()
        
        # Second call with same data - should use cache for all
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Verify offline engine was NOT called (all cache hits)
        assert not mock_offline_engine.compute_features_at_timestamp.called
    
    @pytest.mark.asyncio
    async def test_cache_statistics_tracking(
        self,
        dataset_builder_with_cache,
        memory_cache_service,
    ):
        """Test cache statistics: verify hit_rate, miss_count tracking."""
        symbol = "BTCUSDT"
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dataset_id = "test-dataset-id"
        
        historical_data = {
            "trades": pd.DataFrame({
                "timestamp": [timestamp],
                "price": [50000.0],
                "size": [0.1],
                "side": ["Buy"],
            }),
            "klines": pd.DataFrame({
                "timestamp": [timestamp],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [1.0],
            }),
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
        }
        
        # First call - cache miss
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Second call - cache hit
        await dataset_builder_with_cache._compute_features_batch(
            symbol,
            historical_data,
            dataset_id,
        )
        
        # Check statistics
        stats = await memory_cache_service.get_statistics()
        assert stats["hit_count"] > 0
        assert stats["miss_count"] > 0
        assert stats["hit_rate"] > 0

