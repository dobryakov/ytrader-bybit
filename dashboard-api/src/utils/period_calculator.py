"""Utility functions for calculating dataset periods from durations."""

from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, Optional


def calculate_periods_from_durations(
    train_duration_days: int,
    validation_duration_days: int,
    test_duration_days: int,
    reference_time: Optional[datetime] = None
) -> Dict[str, datetime]:
    """
    Calculate train/validation/test periods from durations.
    
    Uses rolling window approach:
    - Test period: most recent data (test_duration_days)
    - Validation period: before test period (validation_duration_days)
    - Train period: before validation period (train_duration_days)
    
    Periods are rounded to start/end of day to ensure full day coverage.
    
    Args:
        train_duration_days: Training period duration in days
        validation_duration_days: Validation period duration in days
        test_duration_days: Test period duration in days
        reference_time: Reference time for calculation (defaults to UTC now - 1 day)
    
    Returns:
        Dictionary with keys:
        - train_period_start: datetime (start of day)
        - train_period_end: datetime (end of day)
        - validation_period_start: datetime (start of day)
        - validation_period_end: datetime (end of day)
        - test_period_start: datetime (start of day)
        - test_period_end: datetime (end of day)
    """
    if reference_time is None:
        # Use 1 day ago as reference to exclude most recent data (which may be incomplete)
        reference_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Ensure reference_time is timezone-aware
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    # Calculate periods backwards from reference time
    # Round reference_time to end of day (23:59:59) to include all data for that day
    reference_date = reference_time.date()
    test_period_end = datetime.combine(reference_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
    
    # For N days period, we need N days: from (reference_date - N + 1) to reference_date
    test_period_start_date = reference_date - timedelta(days=test_duration_days - 1)
    test_period_start = datetime.combine(test_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)
    
    # Validation period ends 1 day before test period starts
    validation_period_end_date = test_period_start_date - timedelta(days=1)
    validation_period_end = datetime.combine(validation_period_end_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
    
    # For N days period, we need N days: from (validation_period_end_date - N + 1) to validation_period_end_date
    validation_period_start_date = validation_period_end_date - timedelta(days=validation_duration_days - 1)
    validation_period_start = datetime.combine(validation_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)
    
    # Train period ends 1 day before validation period starts
    train_period_end_date = validation_period_start_date - timedelta(days=1)
    train_period_end = datetime.combine(train_period_end_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
    
    # For N days period, we need N days: from (train_period_end_date - N + 1) to train_period_end_date
    train_period_start_date = train_period_end_date - timedelta(days=train_duration_days - 1)
    train_period_start = datetime.combine(train_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)
    
    periods = {
        "train_period_start": train_period_start,
        "train_period_end": train_period_end,
        "validation_period_start": validation_period_start,
        "validation_period_end": validation_period_end,
        "test_period_start": test_period_start,
        "test_period_end": test_period_end,
    }
    
    return periods

