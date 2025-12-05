"""
Test fixtures for target variables (regression, classification, risk-adjusted).
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any


@pytest.fixture
def sample_targets_regression():
    """Sample regression targets (returns) for different horizons."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        # Simulate returns at different horizons
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "target_1m": 0.0001 * (i % 10),  # 1-minute return
            "target_5m": 0.0005 * (i % 10),  # 5-minute return
            "target_15m": 0.0015 * (i % 10),  # 15-minute return
            "target_1h": 0.006 * (i % 10),   # 1-hour return
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_targets_classification():
    """Sample classification targets (direction) with threshold."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    threshold = 0.001  # 0.1%
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        # Simulate returns
        return_value = 0.0001 * (i % 20) - 0.001  # Range from -0.001 to 0.001
        
        # Classify: 1 for positive above threshold, -1 for negative below -threshold, 0 otherwise
        if return_value > threshold:
            direction = 1
        elif return_value < -threshold:
            direction = -1
        else:
            direction = 0
        
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "target_1m": direction,
            "target_5m": direction,
            "target_15m": direction,
            "target_1h": direction,
            "threshold": threshold,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_targets_risk_adjusted():
    """Sample risk-adjusted targets (Sharpe ratio, Sortino ratio)."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        # Simulate risk-adjusted metrics
        return_value = 0.0001 * (i % 10)
        volatility = 0.0005 + (i % 5) * 0.0001
        sharpe_ratio = return_value / volatility if volatility > 0 else 0.0
        
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "target_sharpe_1m": sharpe_ratio,
            "target_sharpe_5m": sharpe_ratio * 2,
            "target_sortino_1m": sharpe_ratio * 1.2,
            "target_sortino_5m": sharpe_ratio * 2.4,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_targets_with_leakage():
    """Sample targets that would cause data leakage (using future data)."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        # Simulate leakage: target uses price from future timestamp
        future_price = 50000.0 + (i + 60) * 0.1  # Using price 60 seconds in future
        current_price = 50000.0 + i * 0.1
        return_value = (future_price - current_price) / current_price
        
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "target_1m": return_value,  # This would be leakage
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_targets_no_leakage():
    """Sample targets computed correctly without data leakage."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    prices = []
    for i in range(200):  # Generate more data to compute targets correctly
        timestamp = base_time + timedelta(seconds=i)
        price = 50000.0 + i * 0.1
        prices.append(price)
        
        if i >= 60:  # Can compute 1-minute return after 60 seconds
            current_price = prices[i]
            past_price = prices[i - 60]  # Price 60 seconds ago
            return_value = (current_price - past_price) / past_price
            
            data.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "target_1m": return_value,  # Correct: uses only past data
            })
    
    return pd.DataFrame(data)
