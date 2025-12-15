"""
Test fixtures for dataset metadata.
"""
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from uuid import uuid4


@pytest.fixture
def sample_dataset_metadata():
    """Sample dataset metadata with time-based split."""
    base_time = datetime.now(timezone.utc)
    dataset_id = uuid4()
    
    return {
        "id": dataset_id,
        "symbol": "BTCUSDT",
        "status": "ready",
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=30),
        "train_period_end": base_time - timedelta(days=11),
        "validation_period_start": base_time - timedelta(days=10),
        "validation_period_end": base_time - timedelta(days=6),
        "test_period_start": base_time - timedelta(days=5),
        "test_period_end": base_time,
        "walk_forward_config": None,
        "target_config": {
            "type": "regression",
            "horizon": 60,
            "threshold": None,
        },
        "target_registry_version": "1.0.0",
        "feature_registry_version": "1.0.0",
        "train_records": 10000,
        "validation_records": 2000,
        "test_records": 1000,
        "output_format": "parquet",
        "storage_path": "/data/datasets/test-dataset-id",
        "created_at": base_time - timedelta(days=31),
        "completed_at": base_time - timedelta(days=30),
        "estimated_completion": None,
        "error_message": None,
    }


@pytest.fixture
def sample_dataset_metadata_walk_forward():
    """Sample dataset metadata with walk-forward split."""
    base_time = datetime.now(timezone.utc)
    dataset_id = uuid4()
    
    return {
        "id": dataset_id,
        "symbol": "BTCUSDT",
        "status": "ready",
        "split_strategy": "walk_forward",
        "train_period_start": None,
        "train_period_end": None,
        "validation_period_start": None,
        "validation_period_end": None,
        "test_period_start": None,
        "test_period_end": None,
        "walk_forward_config": {
            "train_window_days": 30,
            "validation_window_days": 5,
            "test_window_days": 5,
            "step_days": 5,
            "start_date": (base_time - timedelta(days=60)).isoformat(),
            "end_date": base_time.isoformat(),
        },
        "target_config": {
            "type": "classification",
            "horizon": 300,
            "threshold": 0.001,
        },
        "target_registry_version": "1.0.0",
        "feature_registry_version": "1.0.0",
        "train_records": 15000,
        "validation_records": 2500,
        "test_records": 1200,
        "output_format": "parquet",
        "storage_path": "/data/datasets/walk-forward-dataset-id",
        "created_at": base_time - timedelta(days=61),
        "completed_at": base_time - timedelta(days=60),
        "estimated_completion": None,
        "error_message": None,
    }


@pytest.fixture
def sample_dataset_metadata_building():
    """Sample dataset metadata with building status."""
    base_time = datetime.now(timezone.utc)
    dataset_id = uuid4()
    
    return {
        "id": dataset_id,
        "symbol": "ETHUSDT",
        "status": "building",
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=20),
        "train_period_end": base_time - timedelta(days=8),
        "validation_period_start": base_time - timedelta(days=7),
        "validation_period_end": base_time - timedelta(days=4),
        "test_period_start": base_time - timedelta(days=3),
        "test_period_end": base_time,
        "walk_forward_config": None,
        "target_config": {
            "type": "regression",
            "horizon": 900,
            "threshold": None,
        },
        "target_registry_version": "1.0.0",
        "feature_registry_version": "1.0.0",
        "train_records": 0,
        "validation_records": 0,
        "test_records": 0,
        "output_format": "parquet",
        "storage_path": None,
        "created_at": base_time - timedelta(minutes=10),
        "completed_at": None,
        "estimated_completion": base_time + timedelta(minutes=50),
        "error_message": None,
    }


@pytest.fixture
def sample_dataset_metadata_failed():
    """Sample dataset metadata with failed status."""
    base_time = datetime.now(timezone.utc)
    dataset_id = uuid4()
    
    return {
        "id": dataset_id,
        "symbol": "BTCUSDT",
        "status": "failed",
        "split_strategy": "time_based",
        "train_period_start": base_time - timedelta(days=30),
        "train_period_end": base_time - timedelta(days=11),
        "validation_period_start": base_time - timedelta(days=10),
        "validation_period_end": base_time - timedelta(days=6),
        "test_period_start": base_time - timedelta(days=5),
        "test_period_end": base_time,
        "walk_forward_config": None,
        "target_config": {
            "type": "regression",
            "horizon": 60,
            "threshold": None,
        },
        "target_registry_version": "1.0.0",
        "feature_registry_version": "1.0.0",
        "train_records": 0,
        "validation_records": 0,
        "test_records": 0,
        "output_format": "parquet",
        "storage_path": None,
        "created_at": base_time - timedelta(hours=2),
        "completed_at": base_time - timedelta(hours=1),
        "estimated_completion": None,
        "error_message": "Insufficient historical data for requested period",
    }


@pytest.fixture
def sample_dataset_list():
    """List of sample dataset metadata."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "id": uuid4(),
            "symbol": "BTCUSDT",
            "status": "ready",
            "split_strategy": "time_based",
            "train_period_start": base_time - timedelta(days=30),
            "train_period_end": base_time - timedelta(days=11),
            "validation_period_start": base_time - timedelta(days=10),
            "validation_period_end": base_time - timedelta(days=6),
            "test_period_start": base_time - timedelta(days=5),
            "test_period_end": base_time,
            "walk_forward_config": None,
            "target_config": {
                "type": "regression",
                "horizon": 60,
            },
            "feature_registry_version": "1.0.0",
            "train_records": 10000,
            "validation_records": 2000,
            "test_records": 1000,
            "output_format": "parquet",
            "storage_path": "/data/datasets/dataset-1",
            "created_at": base_time - timedelta(days=31),
            "completed_at": base_time - timedelta(days=30),
        },
        {
            "id": uuid4(),
            "symbol": "BTCUSDT",
            "status": "ready",
            "split_strategy": "walk_forward",
            "train_period_start": None,
            "train_period_end": None,
            "validation_period_start": None,
            "validation_period_end": None,
            "test_period_start": None,
            "test_period_end": None,
            "walk_forward_config": {
                "train_window_days": 30,
                "validation_window_days": 5,
                "test_window_days": 5,
                "step_days": 5,
            },
            "target_config": {
                "type": "classification",
                "horizon": 300,
                "threshold": 0.001,
            },
            "feature_registry_version": "1.0.0",
            "train_records": 15000,
            "validation_records": 2500,
            "test_records": 1200,
            "output_format": "parquet",
            "storage_path": "/data/datasets/dataset-2",
            "created_at": base_time - timedelta(days=40),
            "completed_at": base_time - timedelta(days=39),
        },
        {
            "id": uuid4(),
            "symbol": "ETHUSDT",
            "status": "building",
            "split_strategy": "time_based",
            "train_period_start": base_time - timedelta(days=20),
            "train_period_end": base_time - timedelta(days=8),
            "validation_period_start": base_time - timedelta(days=7),
            "validation_period_end": base_time - timedelta(days=4),
            "test_period_start": base_time - timedelta(days=3),
            "test_period_end": base_time,
            "walk_forward_config": None,
            "target_config": {
                "type": "regression",
                "horizon": 900,
            },
            "feature_registry_version": "1.0.0",
            "train_records": 0,
            "validation_records": 0,
            "test_records": 0,
            "output_format": "parquet",
            "storage_path": None,
            "created_at": base_time - timedelta(minutes=10),
            "completed_at": None,
        },
    ]
