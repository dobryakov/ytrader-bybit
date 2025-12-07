"""
Unit tests for DatasetBuilder data type optimization.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from datetime import datetime, timezone, date

from src.services.dataset_builder import DatasetBuilder
from src.services.feature_registry import FeatureRegistryLoader
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage


class TestDatasetBuilderDataTypes:
    """Tests for DatasetBuilder data type optimization."""
    
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
        return loader
    
    @pytest.fixture
    def dataset_builder_with_registry(self, mock_metadata_storage, mock_parquet_storage, mock_feature_registry_loader):
        """Create DatasetBuilder with Feature Registry loader."""
        return DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
            feature_registry_loader=mock_feature_registry_loader,
        )
    
    @pytest.fixture
    def dataset_builder_without_registry(self, mock_metadata_storage, mock_parquet_storage):
        """Create DatasetBuilder without Feature Registry loader."""
        return DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
        )
    
    @pytest.mark.asyncio
    async def test_read_historical_data_with_registry(
        self,
        dataset_builder_with_registry,
        mock_parquet_storage,
        mock_feature_registry_loader,
    ):
        """Test _read_historical_data only loads required data types when Feature Registry provided."""
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        # Setup mock to return empty DataFrames
        mock_parquet_storage.read_orderbook_snapshots_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_orderbook_deltas_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_trades_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_klines_range = AsyncMock(return_value=pd.DataFrame())
        
        result = await dataset_builder_with_registry._read_historical_data(
            "BTCUSDT",
            start_date,
            end_date,
        )
        
        # Verify only required data types were loaded
        mock_parquet_storage.read_orderbook_snapshots_range.assert_called_once()
        mock_parquet_storage.read_orderbook_deltas_range.assert_called_once()
        mock_parquet_storage.read_trades_range.assert_called_once()
        mock_parquet_storage.read_klines_range.assert_called_once()
        # Ticker and funding should not be loaded (not in required types)
        
        assert "snapshots" in result
        assert "deltas" in result
        assert "trades" in result
        assert "klines" in result
    
    @pytest.mark.asyncio
    async def test_read_historical_data_without_registry(
        self,
        dataset_builder_without_registry,
        mock_parquet_storage,
    ):
        """Test _read_historical_data loads all data types when Feature Registry not provided."""
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        # Setup mock to return empty DataFrames
        mock_parquet_storage.read_orderbook_snapshots_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_orderbook_deltas_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_trades_range = AsyncMock(return_value=pd.DataFrame())
        mock_parquet_storage.read_klines_range = AsyncMock(return_value=pd.DataFrame())
        
        result = await dataset_builder_without_registry._read_historical_data(
            "BTCUSDT",
            start_date,
            end_date,
        )
        
        # Verify all data types were loaded (fallback behavior)
        mock_parquet_storage.read_orderbook_snapshots_range.assert_called_once()
        mock_parquet_storage.read_orderbook_deltas_range.assert_called_once()
        mock_parquet_storage.read_trades_range.assert_called_once()
        mock_parquet_storage.read_klines_range.assert_called_once()
        
        assert "snapshots" in result
        assert "deltas" in result
        assert "trades" in result
        assert "klines" in result
    
    @pytest.mark.asyncio
    async def test_read_historical_data_subset_types(
        self,
        dataset_builder_with_registry,
        mock_parquet_storage,
        mock_feature_registry_loader,
    ):
        """Test _read_historical_data with subset of data types."""
        # Mock Feature Registry to only require klines
        mock_feature_registry_loader.get_required_data_types = MagicMock(return_value={"kline"})
        mock_feature_registry_loader.get_data_type_mapping = MagicMock(return_value={
            "kline": ["klines"],
        })
        
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        mock_parquet_storage.read_klines_range = AsyncMock(return_value=pd.DataFrame())
        
        result = await dataset_builder_with_registry._read_historical_data(
            "BTCUSDT",
            start_date,
            end_date,
        )
        
        # Verify only klines were loaded
        mock_parquet_storage.read_klines_range.assert_called_once()
        # Other data types should not be loaded
        mock_parquet_storage.read_orderbook_snapshots_range.assert_not_called()
        mock_parquet_storage.read_orderbook_deltas_range.assert_not_called()
        mock_parquet_storage.read_trades_range.assert_not_called()
        
        assert "klines" in result

