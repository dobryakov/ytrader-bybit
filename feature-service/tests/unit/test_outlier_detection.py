"""
Unit tests for outlier detection and clipping in OptimizedDatasetBuilder.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.models.dataset import TargetConfig
from src.services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from unittest.mock import MagicMock


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for OptimizedDatasetBuilder."""
    metadata_storage = MagicMock(spec=MetadataStorage)
    parquet_storage = MagicMock(spec=ParquetStorage)
    
    return {
        "metadata_storage": metadata_storage,
        "parquet_storage": parquet_storage,
        "dataset_storage_path": "/tmp/test_datasets",
        "cache_service": None,
        "feature_registry_loader": None,
        "target_registry_version_manager": None,
        "dataset_publisher": None,
    }


@pytest.fixture
def builder(mock_dependencies):
    """Create OptimizedDatasetBuilder instance for testing."""
    return OptimizedDatasetBuilder(**mock_dependencies)


@pytest.fixture
def regression_target_config():
    """Create regression target config."""
    return TargetConfig(
        type="regression",
        horizon=3600,
        threshold=None,
        computation=None,
    )


@pytest.fixture
def classification_target_config():
    """Create classification target config."""
    return TargetConfig(
        type="classification",
        horizon=3600,
        threshold=0.005,
        computation=None,
    )


def test_apply_outlier_detection_regression_target(builder, regression_target_config):
    """Test outlier detection is applied to regression targets."""
    # Create splits with outliers
    # Use normal distribution with clear outliers
    train_targets = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0, -10.0])  # 10.0 and -10.0 are outliers
    train_mean = train_targets.mean()
    train_std = train_targets.std()
    threshold = 3.0 * train_std
    
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(len(train_targets))],
            "target": train_targets,
        }),
        "validation": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0.0, 5.0, -5.0],  # 5.0 and -5.0 might be outliers
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0.0, 15.0, -15.0],  # 15.0 and -15.0 are definitely outliers
        }),
    }
    
    # Apply outlier detection
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    # Verify config was computed
    assert outlier_config is not None
    assert "threshold" in outlier_config
    assert "train_mean" in outlier_config
    assert "train_std" in outlier_config
    assert "lower_bound" in outlier_config
    assert "upper_bound" in outlier_config
    assert outlier_config["method"] == "3sigma_clipping"
    
    # Verify threshold is 3σ
    assert abs(outlier_config["threshold"] - threshold) < 1e-6
    
    # Verify is_outlier column was added
    assert "is_outlier" in splits["train"].columns
    assert "is_outlier" in splits["validation"].columns
    assert "is_outlier" in splits["test"].columns
    
    # Verify outliers were detected (at least in test split where values are extreme)
    test_outliers = splits["test"]["is_outlier"].sum()
    assert test_outliers > 0, "Should detect outliers in test split with extreme values"
    
    # Verify targets were clipped
    train_clipped = splits["train"]["target"]
    assert train_clipped.max() <= outlier_config["upper_bound"]
    assert train_clipped.min() >= outlier_config["lower_bound"]
    
    # Verify validation and test use same threshold
    val_outliers = splits["validation"]["is_outlier"].sum()
    test_outliers = splits["test"]["is_outlier"].sum()
    
    # Verify targets in validation and test were clipped
    val_clipped = splits["validation"]["target"]
    test_clipped = splits["test"]["target"]
    assert val_clipped.max() <= outlier_config["upper_bound"]
    assert val_clipped.min() >= outlier_config["lower_bound"]
    assert test_clipped.max() <= outlier_config["upper_bound"]
    assert test_clipped.min() >= outlier_config["lower_bound"]


def test_apply_outlier_detection_classification_skipped(builder, classification_target_config):
    """Test outlier detection is skipped for classification targets."""
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(5)],
            "target": [0, 1, 0, 1, 0],
        }),
        "validation": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0, 1, 0],
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0, 1, 0],
        }),
    }
    
    # Apply outlier detection
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, classification_target_config)
    
    # Verify it was skipped
    assert outlier_config is None
    
    # Verify is_outlier column was NOT added
    assert "is_outlier" not in splits["train"].columns
    assert "is_outlier" not in splits["validation"].columns
    assert "is_outlier" not in splits["test"].columns


def test_apply_outlier_detection_empty_train(builder, regression_target_config):
    """Test outlier detection handles empty train split."""
    splits = {
        "train": pd.DataFrame(),
        "validation": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, i, 0, 0, tzinfo=timezone.utc) for i in range(3)],
            "target": [0.0, 5.0, -5.0],
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3, i, 0, 0, tzinfo=timezone.utc) for i in range(3)],
            "target": [0.0, 15.0, -15.0],
        }),
    }
    
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    assert outlier_config is None
    assert "is_outlier" not in splits["validation"].columns
    assert "is_outlier" not in splits["test"].columns


def test_apply_outlier_detection_zero_std(builder, regression_target_config):
    """Test outlier detection handles zero std (no variance)."""
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(5)],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0],  # All same value, std = 0
        }),
        "validation": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [1.0, 1.0, 1.0],
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [1.0, 1.0, 1.0],
        }),
    }
    
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    assert outlier_config is None
    assert "is_outlier" not in splits["train"].columns


def test_apply_outlier_detection_nan_values(builder, regression_target_config):
    """Test outlier detection handles NaN values correctly."""
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(5)],
            "target": [0.0, 0.1, 0.2, np.nan, 0.4],
        }),
        "validation": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0.0, np.nan, 5.0],
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [0.0, np.nan, -5.0],
        }),
    }
    
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    assert outlier_config is not None
    
    # Verify NaN values are not marked as outliers (is_outlier = 0 for NaN)
    train_nan_mask = splits["train"]["target"].isna()
    train_is_outlier_nan = splits["train"]["is_outlier"][train_nan_mask]
    assert (train_is_outlier_nan == 0).all(), "NaN values should not be marked as outliers"
    
    # Verify NaN values remain NaN after clipping
    assert splits["train"]["target"].isna().any(), "NaN values should remain NaN"


def test_apply_outlier_detection_clipping_bounds(builder, regression_target_config):
    """Test that clipping correctly bounds values."""
    # Create train with normal distribution
    np.random.seed(42)
    train_targets = pd.Series(np.random.normal(0.0, 1.0, 100))
    
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(len(train_targets))],
            "target": train_targets,
        }),
        "test": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(3)],
            "target": [100.0, -100.0, 0.0],  # Extreme outliers
        }),
    }
    
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    assert outlier_config is not None
    
    # Verify test targets were clipped to bounds
    test_clipped = splits["test"]["target"]
    assert test_clipped.max() <= outlier_config["upper_bound"]
    assert test_clipped.min() >= outlier_config["lower_bound"]
    
    # Verify extreme values were actually clipped (not equal to original)
    assert test_clipped.iloc[0] < 100.0, "Extreme positive value should be clipped"
    assert test_clipped.iloc[1] > -100.0, "Extreme negative value should be clipped"
    
    # Verify normal value was not clipped
    assert test_clipped.iloc[2] == 0.0, "Normal value should not be clipped"


def test_apply_outlier_detection_outlier_counting(builder, regression_target_config):
    """Test that outlier counting is correct."""
    # Create train with known outliers
    train_targets = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0, -10.0])
    train_mean = train_targets.mean()
    train_std = train_targets.std()
    threshold = 3.0 * train_std
    
    splits = {
        "train": pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(len(train_targets))],
            "target": train_targets,
        }),
    }
    
    outlier_config = builder._apply_outlier_detection_and_clipping(splits, regression_target_config)
    
    assert outlier_config is not None
    
    # Manually count outliers
    expected_outliers = ((train_targets - train_mean).abs() > threshold).sum()
    actual_outliers = splits["train"]["is_outlier"].sum()
    
    assert actual_outliers == expected_outliers, f"Outlier count mismatch: expected {expected_outliers}, got {actual_outliers}"
    assert outlier_config["total_outliers_detected"] == expected_outliers

