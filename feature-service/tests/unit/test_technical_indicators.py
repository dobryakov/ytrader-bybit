"""
Unit tests for technical indicators computation.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np

from src.models.rolling_windows import RollingWindows


class TestTechnicalIndicators:
    """Test technical indicators computation."""
    
    def test_compute_ema_21_sufficient_data(self, sample_rolling_windows_klines):
        """Test EMA(21) computation with sufficient price history."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        base_time = rw.last_update
        
        # Add 21+ klines for EMA(21)
        closes = [49900.0 + i * 10.0 for i in range(25)]
        for i, close_price in enumerate(closes):
            kline = {
                "timestamp": base_time - timedelta(minutes=25-i),
                "open": close_price - 5.0,
                "high": close_price + 5.0,
                "low": close_price - 10.0,
                "close": close_price,
                "volume": 5.0 + i,
            }
            rw.add_kline(kline)
        
        ema = compute_ema_21(rw)
        
        # Should compute EMA(21)
        assert ema is not None
        assert isinstance(ema, float)
        assert ema > 0
    
    def test_compute_ema_21_insufficient_data(self, sample_rolling_windows_klines):
        """Test EMA(21) with insufficient data."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        # Only 2 klines, need at least 21
        ema = compute_ema_21(rw)
        
        assert ema is None
    
    def test_compute_ema_21_formula(self, sample_rolling_windows_klines):
        """Test EMA(21) calculation matches expected formula."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        base_time = rw.last_update
        
        # Create 21 klines with known prices
        closes = [50000.0] * 21  # All same price
        for i, close_price in enumerate(closes):
            kline = {
                "timestamp": base_time - timedelta(minutes=21-i),
                "open": close_price - 5.0,
                "high": close_price + 5.0,
                "low": close_price - 10.0,
                "close": close_price,
                "volume": 5.0,
            }
            rw.add_kline(kline)
        
        ema = compute_ema_21(rw)
        
        # EMA of constant prices should equal the price
        assert ema is not None
        assert abs(ema - 50000.0) < 0.01
    
    def test_compute_ema_21_smoothing(self, sample_rolling_windows_klines):
        """Test EMA smoothing: recent prices should have more weight."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        base_time = rw.last_update
        
        # Create 21 klines: first 20 at 50000.0, last at 51000.0
        closes = [50000.0] * 20 + [51000.0]
        for i, close_price in enumerate(closes):
            kline = {
                "timestamp": base_time - timedelta(minutes=21-i),
                "open": close_price - 5.0,
                "high": close_price + 5.0,
                "low": close_price - 10.0,
                "close": close_price,
                "volume": 5.0,
            }
            rw.add_kline(kline)
        
        ema = compute_ema_21(rw)
        
        # EMA should be closer to 51000.0 than 50000.0 (recent price has more weight)
        assert ema is not None
        assert ema > 50000.0
        assert ema < 51000.0
        # EMA should be closer to recent price
        assert abs(ema - 51000.0) < abs(ema - 50000.0)
    
    def test_compute_ema_21_all_prices_equal(self, sample_rolling_windows_klines):
        """Test EMA(21) with all prices equal."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        base_time = rw.last_update
        
        # Create 21 klines with same price
        price = 50000.0
        for i in range(21):
            kline = {
                "timestamp": base_time - timedelta(minutes=21-i),
                "open": price - 5.0,
                "high": price + 5.0,
                "low": price - 10.0,
                "close": price,
                "volume": 5.0,
            }
            rw.add_kline(kline)
        
        ema = compute_ema_21(rw)
        
        # EMA of constant prices should equal the price
        assert ema is not None
        assert abs(ema - price) < 0.01
    
    def test_compute_ema_21_single_price(self, sample_rolling_windows_klines):
        """Test EMA(21) with single price value."""
        from src.features.technical_indicators import compute_ema_21
        
        rw = RollingWindows(**sample_rolling_windows_klines)
        base_time = rw.last_update
        
        # Add only 1 kline
        kline = {
            "timestamp": base_time,
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50000.0,
            "volume": 5.0,
        }
        rw.add_kline(kline)
        
        ema = compute_ema_21(rw)
        
        # Should return None (insufficient data)
        assert ema is None

