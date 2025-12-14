"""
Unit tests for DatasetBuilder feature filtering by Feature Registry.
Tests that features are correctly filtered according to Feature Registry version 1.2.0.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta, date

from src.services.dataset_builder import DatasetBuilder
from src.services.feature_registry import FeatureRegistryLoader
from src.services.offline_engine import OfflineEngine
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.models.feature_registry import FeatureRegistry


class TestDatasetBuilderFeatureRegistryFiltering:
    """Tests for feature filtering by Feature Registry in DatasetBuilder."""
    
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
        # Return klines data for Feature Registry 1.2.0
        # Need at least 21 minutes for EMA(21), 14 for RSI(14), 20 for volume_ratio_20, 5 for returns_5m/volatility_5m
        # Use 30 minutes to have enough data for all features
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        klines_data = []
        for i in range(30):  # 30 minutes of klines
            timestamp = base_time + timedelta(minutes=i)
            klines_data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open": 50000.0 + (i * 0.1),
                "high": 50010.0 + (i * 0.1),
                "low": 49990.0 + (i * 0.1),
                "close": 50005.0 + (i * 0.1),
                "volume": 10.0 + (i * 0.01),
            })
        
        klines_df = pd.DataFrame(klines_data)
        storage.read_klines_range = AsyncMock(return_value=klines_df)
        storage.read_trades_range = AsyncMock(return_value=pd.DataFrame())  # Empty
        storage.read_orderbook_snapshots_range = AsyncMock(return_value=pd.DataFrame())  # Empty
        storage.read_orderbook_deltas_range = AsyncMock(return_value=pd.DataFrame())  # Empty
        storage.read_ticker = AsyncMock(return_value=pd.DataFrame())  # Empty
        storage.read_funding = AsyncMock(return_value=pd.DataFrame())  # Empty
        
        # For data availability check
        storage.read_klines = AsyncMock(return_value=klines_df)
        storage.read_trades = AsyncMock(return_value=pd.DataFrame())
        return storage
    
    @pytest.fixture
    def synthetic_klines_with_enough_data(self):
        """Create synthetic klines with enough data for all features."""
        # Need enough data for all features:
        # - EMA(21): needs 21 minutes
        # - RSI(14): needs 14+1 minutes  
        # - volume_ratio_20: needs 20 minutes
        # - returns_5m, volatility_5m: need 5 minutes
        # Use 30 minutes to be safe
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        klines_data = []
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i)
            klines_data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open": 50000.0 + (i * 0.1),
                "high": 50010.0 + (i * 0.1),
                "low": 49990.0 + (i * 0.1),
                "close": 50005.0 + (i * 0.1),
                "volume": 10.0 + (i * 0.01),
            })
        return pd.DataFrame(klines_data)
    
    @pytest.fixture
    def feature_registry_v1_2_0_config(self):
        """Feature Registry version 1.2.0 configuration."""
        return {
            "version": "1.2.0",
            "features": [
                {
                    "name": "returns_5m",
                    "input_sources": ["kline"],
                    "lookback_window": "5m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "volatility_5m",
                    "input_sources": ["kline"],
                    "lookback_window": "5m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "rsi_14",
                    "input_sources": ["kline"],
                    "lookback_window": "14m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "ema_21",
                    "input_sources": ["kline"],
                    "lookback_window": "21m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "price_ema21_ratio",
                    "input_sources": ["kline"],
                    "lookback_window": "21m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "volume_ratio_20",
                    "input_sources": ["kline"],
                    "lookback_window": "20m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [
                        {"source": "kline", "timestamp_required": True}
                    ]
                },
                {
                    "name": "funding_rate",
                    "input_sources": ["funding"],
                    "lookback_window": "0s",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 0,
                    "data_sources": [
                        {"source": "funding", "timestamp_required": True}
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def mock_feature_registry_loader(self, feature_registry_v1_2_0_config):
        """Create mock Feature Registry loader with version 1.2.0."""
        loader = MagicMock(spec=FeatureRegistryLoader)
        
        # Create FeatureRegistry model from config
        registry_model = FeatureRegistry(**feature_registry_v1_2_0_config)
        loader._registry_model = registry_model
        
        # Mock methods
        loader.get_config = MagicMock(return_value=feature_registry_v1_2_0_config)
        loader.get_required_data_types = MagicMock(return_value={"kline", "funding"})
        loader.get_data_type_mapping = MagicMock(return_value={
            "kline": ["klines"],
            "funding": ["funding"],
        })
        
        return loader
    
    @pytest.fixture
    def dataset_builder(
        self,
        mock_metadata_storage,
        mock_parquet_storage,
        mock_feature_registry_loader,
    ):
        """Create DatasetBuilder instance with Feature Registry."""
        return DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
            feature_registry_version="1.2.0",
            feature_registry_loader=mock_feature_registry_loader,
        )
    
    @pytest.mark.asyncio
    async def test_compute_features_batch_with_feature_registry_filtering(
        self,
        dataset_builder,
        synthetic_klines_with_enough_data,
    ):
        """Test that features are correctly filtered by Feature Registry version 1.2.0."""
        klines_df = synthetic_klines_with_enough_data
        
        # Add funding data for funding_rate feature
        base_time = klines_df["timestamp"].min()
        funding_data = []
        for i in range(4):
            timestamp = base_time + timedelta(hours=i * 8)
            funding_data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "funding_rate": 0.0001 + (i * 0.00001),
                "next_funding_time": (timestamp + timedelta(hours=8)).timestamp() * 1000,
            })
        funding_df = pd.DataFrame(funding_data)
        
        historical_data = {
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "klines": klines_df,
            "ticker": pd.DataFrame(),
            "funding": funding_df,
        }
        
        # Compute features
        features_df = await dataset_builder._compute_features_batch(
            symbol="BTCUSDT",
            historical_data=historical_data,
            dataset_id="test-dataset-id",
        )
        
        # Verify features DataFrame is not empty
        assert not features_df.empty, "Features DataFrame should not be empty"
        
        # Identify feature columns (exclude metadata columns)
        metadata_columns = {"timestamp", "symbol"}
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        # Verify that we have feature columns
        assert len(feature_columns) > 0, (
            f"No feature columns found in DataFrame. "
            f"Columns: {list(features_df.columns)}, "
            f"DataFrame shape: {features_df.shape}"
        )
        
        # Expected features from Feature Registry 1.2.0
        expected_features = {
            "returns_5m",
            "volatility_5m",
            "rsi_14",
            "ema_21",
            "price_ema21_ratio",
            "volume_ratio_20",
            "funding_rate",
        }
        
        # Verify all expected features are present
        missing_features = expected_features - set(feature_columns)
        assert len(missing_features) == 0, (
            f"Missing expected features: {missing_features}. "
            f"Available features: {feature_columns}. "
            f"All columns: {list(features_df.columns)}. "
            f"DataFrame shape: {features_df.shape}"
        )
        
        # Note: Some features may return None for first few rows due to lookback window requirements
        # (e.g., EMA(21) needs 21 minutes, RSI(14) needs 14+1 minutes, etc.)
        # But later rows should have valid values.
        
        # Check if features have at least some valid values in later rows
        # (skip first few rows which may have None due to lookback requirements)
        for feature_name in expected_features:
            if feature_name != "funding_rate":  # funding_rate doesn't need lookback
                feature_values = features_df[feature_name]
                valid_count = feature_values.notna().sum()
                # At least some rows should have valid values (not all None)
                # For features requiring 21 minutes lookback, first 21 rows may be None
                assert valid_count > 0, (
                    f"Feature {feature_name} has no valid (non-NaN) values. "
                    f"Total rows: {len(features_df)}, Valid: {valid_count}, "
                    f"Feature values (first 10): {feature_values.head(10).tolist()}"
                )
    
    @pytest.mark.asyncio
    async def test_compute_features_batch_no_feature_registry_filtering(
        self,
        mock_metadata_storage,
        mock_parquet_storage,
    ):
        """Test that without Feature Registry, all computed features are present."""
        # Create DatasetBuilder without Feature Registry
        dataset_builder = DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
            feature_registry_version="1.0.0",
            feature_registry_loader=None,  # No filtering
        )
        
        # Prepare historical data
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        klines_data = []
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i)
            klines_data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open": 50000.0 + (i * 0.1),
                "high": 50010.0 + (i * 0.1),
                "low": 49990.0 + (i * 0.1),
                "close": 50005.0 + (i * 0.1),
                "volume": 10.0 + (i * 0.01),
            })
        klines_df = pd.DataFrame(klines_data)
        
        historical_data = {
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "klines": klines_df,
            "ticker": pd.DataFrame(),
            "funding": pd.DataFrame(),
        }
        
        # Compute features
        features_df = await dataset_builder._compute_features_batch(
            symbol="BTCUSDT",
            historical_data=historical_data,
            dataset_id="test-dataset-id",
        )
        
        # Without Feature Registry filtering, should have more features
        metadata_columns = {"timestamp", "symbol"}
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        # Should have many features (price, orderflow, orderbook, etc.)
        assert len(feature_columns) > 7, (
            f"Without Feature Registry filtering, should have more than 7 features. "
            f"Got: {len(feature_columns)}, Features: {feature_columns}"
        )
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_with_registry_filtered_features(
        self,
        dataset_builder,
        mock_parquet_storage,
    ):
        """Test that validation works correctly with Feature Registry filtered features."""
        # Prepare historical data
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        klines_data = []
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i)
            klines_data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open": 50000.0 + (i * 0.1),
                "high": 50010.0 + (i * 0.1),
                "low": 49990.0 + (i * 0.1),
                "close": 50005.0 + (i * 0.1),
                "volume": 10.0 + (i * 0.01),
            })
        klines_df = pd.DataFrame(klines_data)
        
        funding_data = [{
            "timestamp": base_time,
            "symbol": "BTCUSDT",
            "funding_rate": 0.0001,
            "next_funding_time": (base_time + timedelta(hours=8)).timestamp() * 1000,
        }]
        funding_df = pd.DataFrame(funding_data)
        
        historical_data = {
            "snapshots": pd.DataFrame(),
            "deltas": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "klines": klines_df,
            "ticker": pd.DataFrame(),
            "funding": funding_df,
        }
        
        # Compute features
        features_df = await dataset_builder._compute_features_batch(
            symbol="BTCUSDT",
            historical_data=historical_data,
            dataset_id="test-dataset-id",
        )
        
        # Verify features DataFrame is not empty
        assert not features_df.empty, "Features DataFrame should not be empty"
        
        # Verify we have feature columns
        metadata_columns = {"timestamp", "symbol"}
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]
        assert len(feature_columns) > 0, f"No feature columns found. Columns: {list(features_df.columns)}"
        
        # Validate features quality
        validation_result = await dataset_builder._validate_features_quality(
            features_df,
            "test-dataset-id",
        )
        
        # Validation should pass (features should have valid values)
        assert validation_result["valid"] is True, (
            f"Features validation should pass. "
            f"Error: {validation_result.get('error_message')}, "
            f"Feature columns: {feature_columns}"
        )
        
        # Verify filtered DataFrame has features
        filtered_df = validation_result["filtered_features_df"]
        filtered_feature_columns = [col for col in filtered_df.columns if col not in metadata_columns]
        assert len(filtered_feature_columns) > 0, (
            f"Filtered DataFrame should have feature columns. "
            f"Columns: {list(filtered_df.columns)}"
        )
    
    def test_compute_max_lookback_minutes_with_version_1_2_0(
        self,
        dataset_builder,
        mock_feature_registry_loader,
    ):
        """Test that _compute_max_lookback_minutes correctly computes max lookback from Feature Registry version 1.2.0."""
        # Version 1.2.0 has features with large lookback: ema_21 (26 min), rsi_14 (19 min), volume_ratio_20 (20 min)
        max_lookback = dataset_builder._compute_max_lookback_minutes(mock_feature_registry_loader)
        
        # Should compute max lookback from FEATURE_LOOKBACK_MAPPING for ema_21 = 26 minutes
        assert max_lookback == 26, (
            f"Expected max_lookback=26 (from ema_21), got {max_lookback}. "
            f"Feature Registry version: {mock_feature_registry_loader._registry_model.version if mock_feature_registry_loader._registry_model else 'None'}"
        )
    
    def test_compute_max_lookback_minutes_without_loader(self, dataset_builder):
        """Test that _compute_max_lookback_minutes returns default when loader is None."""
        max_lookback = dataset_builder._compute_max_lookback_minutes(None)
        
        # Should return default 30 minutes when loader is None
        assert max_lookback == 30, (
            f"Expected max_lookback=30 (default), got {max_lookback}"
        )
    
    def test_compute_max_lookback_minutes_with_small_lookback(
        self,
        dataset_builder,
    ):
        """Test that _compute_max_lookback_minutes uses default when computed value is too small."""
        # Create mock loader with features that have small lookback (like version 1.0.0)
        mock_loader = MagicMock(spec=FeatureRegistryLoader)
        
        # Create FeatureRegistry model with features that have small lookback (max 6 minutes from volatility_5m)
        from src.models.feature_registry import FeatureRegistry, FeatureDefinition
        
        small_lookback_config = {
            "version": "1.0.0",
            "features": [
                {
                    "name": "volatility_5m",
                    "input_sources": ["kline"],
                    "lookback_window": "5m",
                    "lookahead_forbidden": True,
                    "max_lookback_days": 1,
                    "data_sources": [{"source": "kline", "timestamp_required": True}],
                },
            ],
        }
        registry_model = FeatureRegistry(**small_lookback_config)
        mock_loader._registry_model = registry_model
        
        max_lookback = dataset_builder._compute_max_lookback_minutes(mock_loader)
        
        # Should return default 30 minutes because computed value (6) < 26
        assert max_lookback == 30, (
            f"Expected max_lookback=30 (default, because computed 6 < 26), got {max_lookback}"
        )

