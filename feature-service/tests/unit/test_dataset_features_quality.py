"""
Unit tests for DatasetBuilder features quality validation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.services.dataset_builder import DatasetBuilder
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage


class TestDatasetFeaturesQuality:
    """Tests for DatasetBuilder features quality validation."""
    
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
        return storage
    
    @pytest.fixture
    def dataset_builder(self, mock_metadata_storage, mock_parquet_storage):
        """Create DatasetBuilder instance."""
        return DatasetBuilder(
            metadata_storage=mock_metadata_storage,
            parquet_storage=mock_parquet_storage,
            dataset_storage_path="/tmp/test_datasets",
        )
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_no_nan(self, dataset_builder):
        """Test validation passes when no NaN values in features."""
        # Create features DataFrame with no NaN
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 10,
            "symbol": ["BTCUSDT"] * 10,
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "feature3": [3.0] * 10,
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is True
        assert result["filtered_rows"] == 0
        assert len(result["filtered_features_df"]) == 10
        assert result["nan_stats"]["overall_nan_ratio"] == 0.0
        assert len(result["nan_stats"]["high_nan_features"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_some_nan_in_features(self, dataset_builder):
        """Test validation filters rows with high NaN ratio."""
        # Create features DataFrame with some NaN values
        # Row 0: 0% NaN (all valid)
        # Row 1: 33% NaN (1 out of 3 features)
        # Row 2: 67% NaN (2 out of 3 features) - should be filtered
        # Row 3: 100% NaN (all features) - should be filtered
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 4,
            "symbol": ["BTCUSDT"] * 4,
            "feature1": [1.0, 1.0, np.nan, np.nan],
            "feature2": [2.0, 2.0, 2.0, np.nan],
            "feature3": [3.0, np.nan, np.nan, np.nan],
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is True
        assert result["rows_before"] == 4
        # Rows with >80% NaN (default max_row_nan_ratio) should be filtered
        # Row 2 has 67% NaN, Row 3 has 100% NaN
        # With default max_row_nan_ratio=0.8, Row 3 should be filtered
        # With default min_valid_features_ratio=0.3, Row 2 (67% NaN = 33% valid) should pass
        assert result["rows_after"] >= 3  # At least rows 0, 1, 2 should remain
        assert result["nan_stats"]["overall_nan_ratio"] > 0.0
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_empty_dataframe(self, dataset_builder):
        """Test validation fails when DataFrame is empty."""
        features_df = pd.DataFrame()
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is False
        assert "empty" in result["error_message"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_no_feature_columns(self, dataset_builder):
        """Test validation fails when no feature columns found."""
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["BTCUSDT"],
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is False
        assert "no feature columns" in result["error_message"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_high_nan_feature(self, dataset_builder):
        """Test validation logs warning for features with high NaN ratio."""
        # Create features DataFrame where one feature has >50% NaN (default threshold)
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 10,
            "symbol": ["BTCUSDT"] * 10,
            "feature1": [1.0] * 10,  # 0% NaN
            "feature2": [np.nan] * 6 + [2.0] * 4,  # 60% NaN - exceeds default 50% threshold
            "feature3": [3.0] * 10,  # 0% NaN
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is True
        assert len(result["nan_stats"]["high_nan_features"]) == 1
        assert "feature2" in result["nan_stats"]["high_nan_features"]
        assert result["nan_stats"]["high_nan_features"]["feature2"] > 0.5
    
    @pytest.mark.asyncio
    @patch("src.services.dataset_builder.config")
    async def test_validate_features_quality_fail_on_high_nan(self, mock_config, dataset_builder):
        """Test validation fails when fail_on_high_nan_ratio is enabled."""
        # Configure to fail on high NaN - use PropertyMock for attributes
        from unittest.mock import PropertyMock
        type(mock_config).dataset_max_feature_nan_ratio = PropertyMock(return_value=0.5)
        type(mock_config).dataset_max_row_nan_ratio = PropertyMock(return_value=0.8)
        type(mock_config).dataset_min_valid_features_ratio = PropertyMock(return_value=0.3)
        type(mock_config).dataset_fail_on_high_nan_ratio = PropertyMock(return_value=True)
        
        # Create features DataFrame with feature having >50% NaN
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 10,
            "symbol": ["BTCUSDT"] * 10,
            "feature1": [1.0] * 10,
            "feature2": [np.nan] * 6 + [2.0] * 4,  # 60% NaN
            "feature3": [3.0] * 10,
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is False
        assert "failed" in result["error_message"].lower()
        assert "feature2" in result["error_message"] or "feature" in result["error_message"]
    
    @pytest.mark.asyncio
    @patch("src.services.dataset_builder.config")
    async def test_validate_features_quality_all_rows_filtered(self, mock_config, dataset_builder):
        """Test validation fails when all rows are filtered out."""
        # Configure strict thresholds - use PropertyMock for attributes
        from unittest.mock import PropertyMock
        type(mock_config).dataset_max_row_nan_ratio = PropertyMock(return_value=0.1)  # Very strict: max 10% NaN per row
        type(mock_config).dataset_min_valid_features_ratio = PropertyMock(return_value=0.9)  # Very strict: min 90% valid features
        type(mock_config).dataset_fail_on_high_nan_ratio = PropertyMock(return_value=False)
        type(mock_config).dataset_max_feature_nan_ratio = PropertyMock(return_value=0.5)
        
        # Create features DataFrame where all rows have >10% NaN
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 5,
            "symbol": ["BTCUSDT"] * 5,
            "feature1": [1.0] * 5,
            "feature2": [2.0] * 5,
            "feature3": [np.nan] * 5,  # 33% NaN in each row
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is False
        assert "all rows were filtered" in result["error_message"].lower()
        assert result["nan_stats"]["rows_after"] == 0
    
    @pytest.mark.asyncio
    @patch("src.services.dataset_builder.config")
    async def test_validate_features_quality_custom_thresholds(self, mock_config, dataset_builder):
        """Test validation respects custom threshold configuration."""
        # Configure custom thresholds - use PropertyMock for attributes
        from unittest.mock import PropertyMock
        type(mock_config).dataset_max_row_nan_ratio = PropertyMock(return_value=0.5)  # Allow up to 50% NaN per row
        type(mock_config).dataset_min_valid_features_ratio = PropertyMock(return_value=0.4)  # Require at least 40% valid features
        type(mock_config).dataset_max_feature_nan_ratio = PropertyMock(return_value=0.3)  # Warn if feature has >30% NaN
        type(mock_config).dataset_fail_on_high_nan_ratio = PropertyMock(return_value=False)
        
        # Create features DataFrame
        # Row 0: 0% NaN (should keep)
        # Row 1: 33% NaN (should keep, <50%)
        # Row 2: 67% NaN (should filter, >50%)
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 3,
            "symbol": ["BTCUSDT"] * 3,
            "feature1": [1.0, 1.0, np.nan],
            "feature2": [2.0, 2.0, np.nan],
            "feature3": [3.0, np.nan, np.nan],
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is True
        assert result["rows_before"] == 3
        # Row 2 has 67% NaN, which exceeds 50% threshold, so it should be filtered
        assert result["rows_after"] == 2
        assert result["filtered_rows"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_features_quality_nan_statistics(self, dataset_builder):
        """Test validation provides detailed NaN statistics."""
        features_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 10,
            "symbol": ["BTCUSDT"] * 10,
            "feature1": [1.0] * 10,  # 0% NaN
            "feature2": [np.nan] * 3 + [2.0] * 7,  # 30% NaN
            "feature3": [np.nan] * 7 + [3.0] * 3,  # 70% NaN (should be in high_nan_features)
        })
        
        result = await dataset_builder._validate_features_quality(features_df, "test-dataset-id")
        
        assert result["valid"] is True
        assert "nan_stats" in result
        
        # Check statistics structure
        nan_stats = result["nan_stats"]
        assert "overall_nan_ratio" in nan_stats
        assert "high_nan_features" in nan_stats
        assert "nan_counts_per_feature" in nan_stats
        assert "nan_ratios_per_feature" in nan_stats
        
        # Check feature3 is in high_nan_features (70% > 50% default threshold)
        assert "feature3" in nan_stats["high_nan_features"]
        assert nan_stats["high_nan_features"]["feature3"] == 0.7
        
        # Check counts
        assert nan_stats["nan_counts_per_feature"]["feature1"] == 0
        assert nan_stats["nan_counts_per_feature"]["feature2"] == 3
        assert nan_stats["nan_counts_per_feature"]["feature3"] == 7
        
        # Check ratios
        assert nan_stats["nan_ratios_per_feature"]["feature1"] == 0.0
        assert nan_stats["nan_ratios_per_feature"]["feature2"] == 0.3
        assert nan_stats["nan_ratios_per_feature"]["feature3"] == 0.7

