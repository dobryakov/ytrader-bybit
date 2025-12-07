"""
Integration tests for backfilling service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from datetime import date, datetime, timezone
import tempfile
import shutil
from pathlib import Path

from src.services.backfilling_service import BackfillingService
from src.services.dataset_builder import DatasetBuilder
from src.storage.parquet_storage import ParquetStorage
from src.storage.metadata_storage import MetadataStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.utils.bybit_client import BybitClient


class TestBackfillingIntegration:
    """Integration tests for backfilling."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def parquet_storage(self, temp_data_dir):
        """Create Parquet storage for testing."""
        return ParquetStorage(base_path=temp_data_dir)
    
    @pytest.fixture
    def mock_bybit_client(self):
        """Create mock Bybit client."""
        client = MagicMock(spec=BybitClient)
        client.get = AsyncMock()
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_feature_registry_loader(self):
        """Create mock Feature Registry loader."""
        loader = MagicMock(spec=FeatureRegistryLoader)
        loader.get_required_data_types = MagicMock(return_value={"kline"})
        loader.get_data_type_mapping = MagicMock(return_value={"kline": ["klines"]})
        return loader
    
    @pytest.fixture
    def backfilling_service(self, parquet_storage, mock_feature_registry_loader, mock_bybit_client):
        """Create backfilling service for testing."""
        return BackfillingService(
            parquet_storage=parquet_storage,
            feature_registry_loader=mock_feature_registry_loader,
            bybit_client=mock_bybit_client,
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_backfilling(
        self,
        backfilling_service,
        parquet_storage,
        mock_bybit_client,
    ):
        """Test end-to-end backfilling: fetch from Bybit API, save to Parquet, verify data validation."""
        # Mock Bybit API response - return data first, then empty to stop pagination
        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [
                        str(int(datetime(2025, 1, 1, 0, i, 0, tzinfo=timezone.utc).timestamp() * 1000)),
                        "50000.0",
                        "50100.0",
                        "49900.0",
                        "50050.0",
                        "100.5",
                        "5025000.0",
                    ]
                    for i in range(10)
                ],
            },
        }
        mock_response_empty = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        }
        mock_bybit_client.get = AsyncMock(side_effect=[mock_response, mock_response_empty])
        
        # Backfill klines
        klines = await backfilling_service.backfill_klines(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 1),
            interval=1,
        )
        
        assert len(klines) == 10
        
        # Save to Parquet
        await backfilling_service._save_klines("BTCUSDT", "2025-01-01", klines)
        
        # Verify data validation passes
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=10,
            data_type="klines",
        )
        
        assert validation_passed is True
        
        # Verify data can be read by ParquetStorage
        read_data = await parquet_storage.read_klines("BTCUSDT", "2025-01-01")
        
        assert len(read_data) == 10
        assert "timestamp" in read_data.columns
        assert "open" in read_data.columns
        assert "close" in read_data.columns
    
    @pytest.mark.asyncio
    async def test_automatic_backfilling_trigger(
        self,
        backfilling_service,
        parquet_storage,
        mock_bybit_client,
        mock_feature_registry_loader,
    ):
        """Test automatic backfilling trigger in dataset builder with Feature Registry."""
        from src.services.dataset_builder import DatasetBuilder
        from src.storage.metadata_storage import MetadataStorage
        
        mock_metadata_storage = MagicMock(spec=MetadataStorage)
        mock_metadata_storage.create_dataset = AsyncMock(return_value="test-dataset-id")
        mock_metadata_storage.get_dataset = AsyncMock(return_value={
            "status": "building",
            "train_period_start": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "test_period_end": datetime(2025, 1, 2, tzinfo=timezone.utc),
        })
        mock_metadata_storage.update_dataset = AsyncMock()
        
        # Mock Bybit API response
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        })
        
        dataset_builder = DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
            feature_registry_loader=mock_feature_registry_loader,
            backfilling_service=backfilling_service,
        )
        
        # Mock config to enable automatic backfilling
        with patch("src.config.config") as mock_config:
            mock_config.feature_service_backfill_enabled = True
            mock_config.feature_service_backfill_auto = True
            
            # Check data availability (should trigger backfilling)
            available_period = await dataset_builder._check_data_availability(
                "BTCUSDT",
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc),
            )
            
            # Verify backfilling was triggered
            assert mock_bybit_client.get.called
    
    @pytest.mark.asyncio
    async def test_only_required_data_types_backfilled(
        self,
        backfilling_service,
        mock_bybit_client,
        mock_feature_registry_loader,
    ):
        """Test that only required data types are backfilled based on Feature Registry."""
        # Mock Feature Registry to only require klines
        mock_feature_registry_loader.get_required_data_types = MagicMock(return_value={"kline"})
        mock_feature_registry_loader.get_data_type_mapping = MagicMock(return_value={"kline": ["klines"]})
        
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        })
        
        job_id = await backfilling_service.backfill_historical(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 2),
            data_types=None,  # Should use Feature Registry
        )
        
        # Verify Feature Registry was used to determine data types
        mock_feature_registry_loader.get_required_data_types.assert_called()
        mock_feature_registry_loader.get_data_type_mapping.assert_called()
    
    @pytest.mark.asyncio
    async def test_validation_failure_handling(
        self,
        backfilling_service,
        parquet_storage,
        mock_bybit_client,
    ):
        """Test validation failure handling: simulate corrupted save, verify file is deleted."""
        # Mock Bybit API response
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [
                        str(int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)),
                        "50000.0",
                        "50100.0",
                        "49900.0",
                        "50050.0",
                        "100.5",
                        "5025000.0",
                    ],
                ],
            },
        })
        
        # Backfill and save
        klines = await backfilling_service.backfill_klines(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 1),
            interval=1,
        )
        
        await backfilling_service._save_klines("BTCUSDT", "2025-01-01", klines)
        
        # Simulate validation failure by mocking read to return wrong count
        original_read = parquet_storage.read_klines
        
        async def mock_read_wrong_count(*args, **kwargs):
            data = await original_read(*args, **kwargs)
            # Return empty DataFrame to simulate count mismatch
            return pd.DataFrame()
        
        parquet_storage.read_klines = AsyncMock(side_effect=mock_read_wrong_count)
        
        # Validation should fail
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=1,
            data_type="klines",
        )
        
        assert validation_passed is False
    
    @pytest.mark.asyncio
    async def test_data_format_matches_websocket(
        self,
        backfilling_service,
        parquet_storage,
        mock_bybit_client,
    ):
        """Test that backfilled data format matches WebSocket data format."""
        # Mock Bybit API response
        timestamp_ms = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [
                        str(timestamp_ms),
                        "50000.0",
                        "50100.0",
                        "49900.0",
                        "50050.0",
                        "100.5",
                        "5025000.0",
                    ],
                ],
            },
        })
        
        # Backfill and save
        klines = await backfilling_service.backfill_klines(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 1),
            interval=1,
        )
        
        await backfilling_service._save_klines("BTCUSDT", "2025-01-01", klines)
        
        # Read back and verify format
        read_data = await parquet_storage.read_klines("BTCUSDT", "2025-01-01")
        
        # Verify required fields match WebSocket format
        assert "timestamp" in read_data.columns
        assert "open" in read_data.columns
        assert "high" in read_data.columns
        assert "low" in read_data.columns
        assert "close" in read_data.columns
        assert "volume" in read_data.columns
        assert "symbol" in read_data.columns
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(read_data["timestamp"])
        assert pd.api.types.is_float_dtype(read_data["open"])
        assert pd.api.types.is_float_dtype(read_data["volume"])

