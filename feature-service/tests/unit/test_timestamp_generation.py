"""
Unit tests for timestamp generation with configurable intervals.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.services.optimized_dataset.streaming_builder import StreamingDatasetBuilder
from src.services.optimized_dataset.requirements_analyzer import DataRequirements


class TestTimestampGeneration:
    """Tests for timestamp generation with different intervals."""
    
    @pytest.fixture
    def builder(self):
        """Create StreamingDatasetBuilder instance."""
        return StreamingDatasetBuilder(
            cache_service=None,
            parquet_storage=MagicMock(),
            feature_registry_loader=None,
            batch_size=1000,
        )
    
    @pytest.fixture
    def requirements(self):
        """Create sample DataRequirements."""
        from src.services.optimized_dataset.requirements_analyzer import TimestampStrategy
        return DataRequirements(
            required_data_types={"kline"},
            max_lookback_minutes=15,
            timestamp_strategy=TimestampStrategy.KLINES_ONLY,
            required_kline_columns={"open", "close", "high", "low", "volume"},
            feature_groups={},
            storage_types={},
            needs_klines=True,
            needs_trades=False,
            needs_orderbook=False,
            needs_ticker=False,
            needs_funding=False,
        )
    
    def test_generate_timestamps_1_minute_interval(self, builder, requirements):
        """Test timestamp generation with 1-minute interval."""
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=1,
        )
        
        assert len(timestamps) == 6  # 00:00, 00:01, 00:02, 00:03, 00:04, 00:05
        assert timestamps.iloc[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[-1] == datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        
        # Check alignment: all timestamps should be on minute boundaries
        for ts in timestamps:
            assert ts.second == 0
            assert ts.microsecond == 0
    
    def test_generate_timestamps_5_minute_interval(self, builder, requirements):
        """Test timestamp generation with 5-minute interval."""
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc)
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=5,
        )
        
        assert len(timestamps) == 4  # 00:00, 00:05, 00:10, 00:15
        assert timestamps.iloc[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[1] == datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[2] == datetime(2024, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[3] == datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc)
        
        # Check alignment: all timestamps should be on 5-minute boundaries
        for ts in timestamps:
            assert ts.minute % 5 == 0
            assert ts.second == 0
            assert ts.microsecond == 0
    
    def test_generate_timestamps_5_minute_interval_alignment(self, builder, requirements):
        """Test that 5-minute interval aligns timestamps correctly even if start is not aligned."""
        # Start at 16:42:30 - should align to 16:40:00
        start_date = datetime(2024, 1, 1, 16, 42, 30, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 16, 50, 0, tzinfo=timezone.utc)
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=5,
        )
        
        # Should start at 16:40 (aligned down), then 16:45, 16:50
        assert len(timestamps) == 3
        assert timestamps.iloc[0] == datetime(2024, 1, 1, 16, 40, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[1] == datetime(2024, 1, 1, 16, 45, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[2] == datetime(2024, 1, 1, 16, 50, 0, tzinfo=timezone.utc)
    
    def test_generate_timestamps_15_minute_interval(self, builder, requirements):
        """Test timestamp generation with 15-minute interval."""
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=15,
        )
        
        assert len(timestamps) == 5  # 00:00, 00:15, 00:30, 00:45, 01:00
        assert timestamps.iloc[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[1] == datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[2] == datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[3] == datetime(2024, 1, 1, 0, 45, 0, tzinfo=timezone.utc)
        assert timestamps.iloc[4] == datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        
        # Check alignment: all timestamps should be on 15-minute boundaries
        for ts in timestamps:
            assert ts.minute % 15 == 0
            assert ts.second == 0
            assert ts.microsecond == 0
    
    def test_generate_timestamps_empty_period(self, builder, requirements):
        """Test timestamp generation for empty period."""
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Same as start
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=1,
        )
        
        assert len(timestamps) == 1  # Should include the start timestamp
        assert timestamps.iloc[0] == start_date
    
    def test_generate_timestamps_timezone_aware(self, builder, requirements):
        """Test that timestamps are timezone-aware."""
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=1,
        )
        
        for ts in timestamps:
            assert ts.tzinfo is not None
            assert ts.tzinfo == timezone.utc
    
    def test_generate_timestamps_timezone_naive_converted(self, builder, requirements):
        """Test that timezone-naive timestamps are converted to UTC."""
        start_date = datetime(2024, 1, 1, 0, 0, 0)  # No timezone
        end_date = datetime(2024, 1, 1, 0, 5, 0)  # No timezone
        
        timestamps = builder._generate_timestamps_for_period(
            start_date=start_date,
            end_date=end_date,
            all_klines=pd.DataFrame(),
            all_trades=pd.DataFrame(),
            requirements=requirements,
            timestamp_interval_minutes=1,
        )
        
        # All timestamps should be timezone-aware UTC
        for ts in timestamps:
            assert ts.tzinfo is not None
            assert ts.tzinfo == timezone.utc

