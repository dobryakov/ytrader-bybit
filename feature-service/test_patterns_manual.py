#!/usr/bin/env python3
"""
Manual test script for compute_all_candle_patterns_15m on specific timestamp.
Tests second day of dataset at 18:00.
"""
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.rolling_windows import RollingWindows
from src.features.candle_patterns import compute_all_candle_patterns_15m
from src.services.optimized_dataset.optimized_rolling_window import OptimizedRollingWindow
from src.services.data_storage import DataStorageService
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


async def test_patterns_at_timestamp():
    """Test compute_all_candle_patterns_15m at specific timestamp."""
    # Second day of dataset, 18:00 UTC
    # Assuming dataset starts from 2025-12-21, second day would be 2025-12-22
    target_timestamp = datetime(2025, 12, 22, 18, 0, 0, tzinfo=timezone.utc)
    
    print(f"\n{'='*80}")
    print(f"Testing compute_all_candle_patterns_15m at timestamp: {target_timestamp.isoformat()}")
    print(f"{'='*80}\n")
    
    # Initialize data storage service
    data_storage = DataStorageService()
    
    # Get symbol (assuming BTCUSDT)
    symbol = "BTCUSDT"
    
    # Load klines for the time window needed (15 minutes before target)
    start_time = target_timestamp - timedelta(minutes=20)  # Extra buffer
    end_time = target_timestamp
    
    print(f"Loading klines from {start_time.isoformat()} to {end_time.isoformat()}")
    
    # Load 1m klines
    klines_1m = await data_storage.get_klines(
        symbol=symbol,
        interval="1m",
        start_time=start_time,
        end_time=end_time,
    )
    
    if klines_1m is None or len(klines_1m) == 0:
        print(f"ERROR: No klines found for {symbol} in time range")
        return
    
    print(f"Loaded {len(klines_1m)} 1m klines")
    print(f"First kline: {klines_1m.iloc[0]['timestamp'] if len(klines_1m) > 0 else 'N/A'}")
    print(f"Last kline: {klines_1m.iloc[-1]['timestamp'] if len(klines_1m) > 0 else 'N/A'}")
    
    # Create RollingWindows
    rolling_windows = RollingWindows(
        symbol=symbol,
        windows={
            "1m": klines_1m.copy(),
        },
        last_update=target_timestamp,
    )
    
    # Try to get 5m klines too
    klines_5m = await data_storage.get_klines(
        symbol=symbol,
        interval="5m",
        start_time=start_time,
        end_time=end_time,
    )
    
    if klines_5m is not None and len(klines_5m) > 0:
        print(f"Loaded {len(klines_5m)} 5m klines")
        rolling_windows.windows["5m"] = klines_5m.copy()
    
    # Call compute_all_candle_patterns_15m
    print(f"\n{'='*80}")
    print("Calling compute_all_candle_patterns_15m...")
    print(f"{'='*80}\n")
    
    try:
        patterns = compute_all_candle_patterns_15m(rolling_windows)
        
        print(f"Result type: {type(patterns)}")
        print(f"Result keys count: {len(patterns.keys()) if patterns else 0}")
        
        if patterns:
            print(f"\n{'='*80}")
            print("Patterns result:")
            print(f"{'='*80}\n")
            
            # Count None values
            none_count = sum(1 for v in patterns.values() if v is None)
            filled_count = len(patterns) - none_count
            
            print(f"Total features: {len(patterns)}")
            print(f"Filled features: {filled_count}")
            print(f"None features: {none_count}")
            
            if none_count > 0:
                print(f"\n{'='*80}")
                print("WARNING: Found None values in patterns!")
                print(f"{'='*80}\n")
                none_features = [k for k, v in patterns.items() if v is None]
                print(f"None features ({len(none_features)}):")
                for feature in none_features[:20]:  # Show first 20
                    print(f"  - {feature}")
                if len(none_features) > 20:
                    print(f"  ... and {len(none_features) - 20} more")
            
            # Show all features
            print(f"\n{'='*80}")
            print("All features:")
            print(f"{'='*80}\n")
            for feature_name, value in sorted(patterns.items()):
                value_str = "None" if value is None else str(value)
                print(f"  {feature_name}: {value_str}")
        else:
            print("ERROR: patterns is None or empty!")
            
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR during computation:")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_patterns_at_timestamp())

