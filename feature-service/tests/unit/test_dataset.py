"""
Unit tests for Dataset model.
"""
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pydantic import ValidationError

# Model is imported in each test function


def test_dataset_model_creation_time_based(sample_dataset_metadata):
    """Test creating Dataset model with time-based split."""
    from src.models.dataset import Dataset, DatasetStatus, SplitStrategy
    
    dataset = Dataset(**sample_dataset_metadata)
    assert dataset.id == sample_dataset_metadata["id"]
    assert dataset.symbol == "BTCUSDT"
    assert dataset.status == DatasetStatus.READY
    assert dataset.split_strategy == SplitStrategy.TIME_BASED
    assert dataset.train_records == 10000
    assert dataset.validation_records == 2000
    assert dataset.test_records == 1000


def test_dataset_model_creation_walk_forward(sample_dataset_metadata_walk_forward):
    """Test creating Dataset model with walk-forward split."""
    from src.models.dataset import Dataset, DatasetStatus, SplitStrategy
    
    dataset = Dataset(**sample_dataset_metadata_walk_forward)
    assert dataset.split_strategy == SplitStrategy.WALK_FORWARD
    assert dataset.walk_forward_config is not None
    assert dataset.walk_forward_config["train_window_days"] == 30


def test_dataset_model_validation_periods_order():
    """Test Dataset model validates period order."""
    from src.models.dataset import Dataset
    from pydantic import ValidationError
    
    base_time = datetime.now(timezone.utc)
    
    invalid_data = {
        "id": uuid4(),
        "symbol": "BTCUSDT",
        "status": "building",
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=10),
        "train_period_end": base_time - timedelta(days=20),  # End before start - invalid
        "validation_period_start": base_time - timedelta(days=5),
        "validation_period_end": base_time,
        "test_period_start": base_time - timedelta(days=3),
        "test_period_end": base_time,
        "target_config": {
            "type": "regression",
            "horizon": 60,
        },
        "feature_registry_version": "1.0.0",
        "output_format": "parquet",
    }
    
    # Should raise ValidationError (Pydantic converts ValueError to ValidationError)
    with pytest.raises((ValidationError, ValueError)):
        Dataset(**invalid_data)


def test_dataset_model_validation_status_enum():
    """Test Dataset model validates status enum."""
    from src.models.dataset import Dataset
    from pydantic import ValidationError
    
    base_time = datetime.now(timezone.utc)
    
    invalid_data = {
        "id": uuid4(),
        "symbol": "BTCUSDT",
        "status": "invalid_status",  # Invalid status
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=30),
        "train_period_end": base_time - timedelta(days=10),
        "validation_period_start": base_time - timedelta(days=10),
        "validation_period_end": base_time - timedelta(days=5),
        "test_period_start": base_time - timedelta(days=5),
        "test_period_end": base_time,
        "target_config": {
            "type": "regression",
            "horizon": 60,
        },
        "feature_registry_version": "1.0.0",
        "output_format": "parquet",
    }
    
    # Should raise ValidationError (Pydantic converts ValueError to ValidationError)
    with pytest.raises((ValidationError, ValueError)):
        Dataset(**invalid_data)


def test_dataset_model_validation_split_strategy_enum():
    """Test Dataset model validates split strategy enum."""
    from src.models.dataset import Dataset
    from pydantic import ValidationError
    
    base_time = datetime.now(timezone.utc)
    
    invalid_data = {
        "id": uuid4(),
        "symbol": "BTCUSDT",
        "status": "building",
        "split_strategy": "invalid_strategy",  # Invalid strategy
        "train_period_start": base_time - timedelta(days=30),
        "train_period_end": base_time - timedelta(days=10),
        "target_config": {
            "type": "regression",
            "horizon": 60,
        },
        "feature_registry_version": "1.0.0",
        "output_format": "parquet",
    }
    
    # Should raise ValidationError (Pydantic converts ValueError to ValidationError)
    with pytest.raises((ValidationError, ValueError)):
        Dataset(**invalid_data)


def test_dataset_model_validation_target_config_required():
    """Test Dataset model requires target_config."""
    from src.models.dataset import Dataset
    from pydantic import ValidationError
    
    base_time = datetime.now(timezone.utc)
    
    invalid_data = {
        "id": uuid4(),
        "symbol": "BTCUSDT",
        "status": "building",
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=30),
        "train_period_end": base_time - timedelta(days=10),
        # Missing target_config
        "feature_registry_version": "1.0.0",
        "output_format": "parquet",
    }
    
    # Should raise ValidationError (Pydantic converts ValueError to ValidationError)
    with pytest.raises((ValidationError, ValueError)):
        Dataset(**invalid_data)


def test_dataset_model_building_status(sample_dataset_metadata_building):
    """Test Dataset model with building status."""
    from src.models.dataset import Dataset, DatasetStatus
    
    dataset = Dataset(**sample_dataset_metadata_building)
    assert dataset.status == DatasetStatus.BUILDING
    assert dataset.storage_path is None
    assert dataset.completed_at is None
    assert dataset.estimated_completion is not None


def test_dataset_model_failed_status(sample_dataset_metadata_failed):
    """Test Dataset model with failed status."""
    from src.models.dataset import Dataset, DatasetStatus
    
    dataset = Dataset(**sample_dataset_metadata_failed)
    assert dataset.status == DatasetStatus.FAILED
    assert dataset.error_message is not None
    assert "Insufficient" in dataset.error_message
