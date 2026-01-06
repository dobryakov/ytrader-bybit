import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.services.target_computation import TargetComputationEngine


def _make_price_series(n: int = 100) -> pd.DataFrame:
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    timestamps = [base_time + timedelta(seconds=i) for i in range(n)]
    # Smooth upward trend with some noise so returns are non-constant
    prices = np.linspace(50000.0, 50100.0, n) + np.random.normal(scale=5.0, size=n)
    return pd.DataFrame({"timestamp": timestamps, "close": prices})


def test_postprocess_quantile_clipping_and_sharpe_normalization():
    # Для юнита нам не важен сам горизонт, важно получить ненулевой набор таргетов.
    # Поэтому используем относительно короткий горизонт, чтобы future_timestamp
    # всегда попадал в доступный диапазон данных.
    data = _make_price_series(300)
    config = {
        "formula": "returns",
        "price_source": "close",
        "future_price_source": "close",
        "lookup_method": "nearest_forward",
        "tolerance_seconds": 60,
        # post-processing options mirroring target_registry_v1.6.0
        "clip_method": "quantile",
        "clip_q_low": 0.01,
        "clip_q_high": 0.99,
        "normalize": "sharpe",
        "sharpe_window": 20,
    }

    result = TargetComputationEngine.compute_target(
        data=data,
        horizon=60,
        computation_config=config,
        historical_price_data=None,
    )

    # Basic sanity checks
    assert not result.empty
    assert "timestamp" in result.columns
    assert "target" in result.columns

    y = result["target"]
    # No NaN / inf after post-processing
    assert not y.isna().any()
    assert not np.isinf(y).any()

    # After Sharpe-like normalization, scale should be O(1)
    # (std not exploding, not vanishing to ~0)
    std = float(y.std())
    assert 0.01 < std < 10.0


def test_postprocess_fixed_clipping_and_log_normalization():
    data = _make_price_series(300)
    # Make some extreme returns by injecting a large jump
    data.loc[50, "close"] *= 10.0

    base_config = {
        "formula": "returns",
        "price_source": "close",
        "future_price_source": "close",
        "lookup_method": "nearest_forward",
        "tolerance_seconds": 60,
    }

    # Without clipping, log-normalization should still handle extremes but we
    # check that fixed clipping actually constrains the range to [-max_abs, max_abs]
    config = {
        **base_config,
        "clip_method": "fixed",
        "clip_abs_max": 0.05,
        "normalize": "log",
    }

    result = TargetComputationEngine.compute_target(
        data=data,
        horizon=60,
        computation_config=config,
        historical_price_data=None,
    )

    assert not result.empty
    y = result["target"]

    # No NaN / inf
    assert not y.isna().any()
    assert not np.isinf(y).any()

    # After fixed clipping and log-normalization absolute values
    # should not explode and should be bounded by log1p(clip_abs_max)
    max_abs_log = float(np.log1p(0.05))
    assert (np.abs(y) <= max_abs_log + 1e-6).all()


def _make_candle_data(n: int = 100) -> pd.DataFrame:
    """Create sample candle data with open and close prices."""
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    timestamps = [base_time + timedelta(seconds=i * 60) for i in range(n)]
    # Create candles with varying colors (green/red)
    opens = [50000.0 + i * 0.1 for i in range(n)]
    # Alternate between green and red candles
    closes = []
    for i in range(n):
        if i % 2 == 0:
            # Green candle: close > open
            closes.append(opens[i] + 1.0 + np.random.normal(scale=0.1))
        else:
            # Red candle: close < open
            closes.append(opens[i] - 1.0 + np.random.normal(scale=0.1))
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "close": closes,
    })


def test_candle_direction_computation():
    """Test candle_direction formula computes price change from current_open to future_close correctly."""
    data = _make_candle_data(200)
    
    config = {
        "formula": "candle_direction",
        "price_source": "open",  # Used for current_open (at timestamp)
        "future_price_source": "close",  # Used for future_close (at timestamp + horizon)
        "lookup_method": "nearest_forward",
        "tolerance_seconds": 60,
    }
    
    result = TargetComputationEngine.compute_target(
        data=data,
        horizon=60,  # 1 minute horizon
        computation_config=config,
        historical_price_data=None,
    )
    
    # Basic sanity checks
    assert not result.empty
    assert "timestamp" in result.columns
    assert "target" in result.columns
    
    y = result["target"]
    # No NaN / inf
    assert not y.isna().any()
    assert not np.isinf(y).any()
    
    # Check that targets reflect price changes
    # Positive targets should indicate price increase (future_close > current_open)
    # Negative targets should indicate price decrease (future_close < current_open)
    positive_count = (y > 0).sum()
    negative_count = (y < 0).sum()
    zero_count = (y == 0).sum()
    
    # Should have both positive and negative price changes
    assert positive_count > 0, "Should have some positive price changes (positive targets)"
    assert negative_count > 0, "Should have some negative price changes (negative targets)"
    
    # Verify the formula: (future_close - current_open) / current_open
    # For each row, check that target matches the price change
    for idx, row in result.iterrows():
        # Find the corresponding candle in data
        timestamp = row["timestamp"]
        future_timestamp = timestamp + timedelta(seconds=60)
        
        # Get current_open from data at timestamp
        current_candles = data[data["timestamp"] <= timestamp]
        if len(current_candles) > 0:
            current_candle = current_candles.iloc[-1]
            current_open = current_candle["open"]
        else:
            continue
        
        # Find the candle at future_timestamp (or nearest)
        future_candles = data[data["timestamp"] >= future_timestamp]
        if len(future_candles) > 0:
            future_candle = future_candles.iloc[0]
            future_close = future_candle["close"]
            expected_target = (future_close - current_open) / current_open
            # Allow small floating point differences
            assert abs(row["target"] - expected_target) < 1e-6, \
                f"Target {row['target']} should match expected {expected_target} (current_open={current_open}, future_close={future_close})"


def test_next_candle_direction_preset():
    """Test that next_candle_direction preset uses candle_direction formula."""
    from src.services.target_computation import TargetComputationPresets
    
    preset = TargetComputationPresets.get_preset("next_candle_direction")
    
    # Verify preset configuration
    assert preset["formula"] == "candle_direction"
    assert preset["price_source"] == "open"  # Used for current_open
    assert preset["future_price_source"] == "close"  # Used for future_close
    assert "price change" in preset["description"].lower() or "current open" in preset["description"].lower()


