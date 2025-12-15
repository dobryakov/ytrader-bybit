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


