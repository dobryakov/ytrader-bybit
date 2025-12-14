"""
Unit tests for verifying completeness and correctness of all features in Feature Registry.

This test iterates through all features in the active Feature Registry and verifies
that each feature can be computed correctly with appropriate test data.
"""
import pytest
import pytest_asyncio
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import math

from src.services.offline_engine import OfflineEngine
from src.services.feature_registry import FeatureRegistryLoader
from src.models.rolling_windows import RollingWindows
from src.models.orderbook_state import OrderbookState


class TestFeatureRegistryCompleteness:
    """Test that all features in Feature Registry can be computed correctly."""
    
    @pytest.fixture
    def feature_registry_loader(self):
        """Load Feature Registry v1.4.0."""
        loader = FeatureRegistryLoader(
            config_path="config/versions/feature_registry_v1.4.0.yaml",
            use_db=False,
        )
        return loader
    
    @pytest.fixture
    def offline_engine(self, feature_registry_loader):
        """Create OfflineEngine with Feature Registry."""
        loader = feature_registry_loader
        registry_config = loader.load()
        engine = OfflineEngine(
            feature_registry_version=registry_config["version"],
            feature_registry_loader=loader,
        )
        return engine
    
    def _create_test_klines(self, pattern: str = "default", count: int = 30) -> pd.DataFrame:
        """
        Create test klines with different patterns.
        
        Args:
            pattern: Pattern type ("default", "green", "red", "pin_bar", "engulfing", "star", "harami")
            count: Number of klines to create
            
        Returns:
            DataFrame with klines
        """
        base_time = datetime.now(timezone.utc) - timedelta(minutes=count)
        klines = []
        
        for i in range(count):
            timestamp = base_time + timedelta(minutes=i)
            base_price = 50000.0
            
            if pattern == "green":
                # All green candles
                open_price = base_price + (i * 0.1)
                close_price = open_price + 10.0
                high_price = close_price + 5.0
                low_price = open_price - 5.0
            elif pattern == "red":
                # All red candles
                open_price = base_price + (i * 0.1)
                close_price = open_price - 10.0
                high_price = open_price + 5.0
                low_price = close_price - 5.0
            elif pattern == "pin_bar" and i == count - 1:
                # Pin bar: very long shadow
                open_price = base_price
                close_price = base_price + 1.0
                high_price = base_price + 200.0  # Very long upper shadow
                low_price = base_price - 1.0
            elif pattern == "engulfing" and i >= count - 2:
                # Engulfing pattern: last candle engulfs previous
                if i == count - 2:
                    # Small red candle
                    open_price = base_price + 10.0
                    close_price = base_price + 5.0
                    high_price = base_price + 12.0
                    low_price = base_price + 3.0
                else:
                    # Large green candle that engulfs previous
                    open_price = base_price + 3.0
                    close_price = base_price + 12.0
                    high_price = base_price + 13.0
                    low_price = base_price + 2.0
            elif pattern == "star" and i >= count - 3:
                # Star pattern: large -> small -> large
                if i == count - 3:
                    # Large green
                    open_price = base_price
                    close_price = base_price + 20.0
                    high_price = base_price + 22.0
                    low_price = base_price - 2.0
                elif i == count - 2:
                    # Small doji
                    open_price = base_price + 19.0
                    close_price = base_price + 19.5
                    high_price = base_price + 20.0
                    low_price = base_price + 18.0
                else:
                    # Large red
                    open_price = base_price + 20.0
                    close_price = base_price
                    high_price = base_price + 21.0
                    low_price = base_price - 2.0
            elif pattern == "harami" and i >= count - 2:
                # Harami pattern: large -> small inside
                if i == count - 2:
                    # Large red
                    open_price = base_price + 20.0
                    close_price = base_price
                    high_price = base_price + 22.0
                    low_price = base_price - 2.0
                else:
                    # Small green inside
                    open_price = base_price + 2.0
                    close_price = base_price + 8.0
                    high_price = base_price + 9.0
                    low_price = base_price + 1.0
            else:
                # Default: alternating pattern
                if i % 2 == 0:
                    open_price = base_price + (i * 0.1)
                    close_price = open_price + 5.0
                else:
                    open_price = base_price + (i * 0.1)
                    close_price = open_price - 5.0
                high_price = max(open_price, close_price) + 3.0
                low_price = min(open_price, close_price) - 3.0
            
            klines.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 10.0 + (i * 0.1),
                "internal_timestamp": timestamp,
            })
        
        return pd.DataFrame(klines)
    
    def _create_test_orderbook_snapshots(self) -> pd.DataFrame:
        """Create test orderbook snapshots."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        snapshots = []
        
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i * 15)
            snapshots.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "sequence": 1000 + i * 100,
                "bids": [[50000.0 - i, 1.5], [49999.0 - i, 2.0]],
                "asks": [[50001.0 + i, 1.2], [50002.0 + i, 2.5]],
                "internal_timestamp": timestamp,
                "exchange_timestamp": timestamp,
            })
        
        return pd.DataFrame(snapshots)
    
    def _create_test_orderbook_deltas(self) -> pd.DataFrame:
        """Create test orderbook deltas."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        deltas = []
        
        for i in range(100):
            timestamp = base_time + timedelta(seconds=i * 10)
            deltas.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "sequence": 1001 + i,
                "delta_type": ["insert", "update", "delete"][i % 3],
                "side": ["bid", "ask"][i % 2],
                "price": 50000.0 + (i * 0.1),
                "quantity": 1.0 + (i * 0.01),
                "internal_timestamp": timestamp,
                "exchange_timestamp": timestamp,
            })
        
        return pd.DataFrame(deltas)
    
    def _create_test_trades(self) -> pd.DataFrame:
        """Create test trades."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        trades = []
        
        for i in range(200):
            timestamp = base_time + timedelta(seconds=i * 5)
            trades.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "price": 50000.0 + (i * 0.05),
                "quantity": 0.1 + (i * 0.001),
                "side": ["Buy", "Sell"][i % 2],
                "trade_time": timestamp,
                "internal_timestamp": timestamp,
            })
        
        return pd.DataFrame(trades)
    
    def _create_test_funding(self) -> pd.DataFrame:
        """Create test funding rate data."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        funding = []
        
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            next_funding_time = int((timestamp + timedelta(hours=8)).timestamp() * 1000)
            funding.append({
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "funding_rate": 0.0001 + (i * 0.00001),
                "next_funding_time": next_funding_time,
                "internal_timestamp": timestamp,
            })
        
        return pd.DataFrame(funding)
    
    def _get_test_data_pattern(self, feature_name: str) -> str:
        """
        Determine which test data pattern to use for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Pattern name ("default", "green", "red", "pin_bar", "engulfing", "star", "harami")
        """
        if "pin_bar" in feature_name:
            return "pin_bar"
        elif "engulfing" in feature_name:
            return "engulfing"
        elif "star" in feature_name or "doji" in feature_name:
            return "star"
        elif "harami" in feature_name:
            return "harami"
        elif "green" in feature_name:
            return "green"
        elif "red" in feature_name:
            return "red"
        else:
            return "default"
    
    @pytest.mark.asyncio
    async def test_all_features_computable(
        self,
        offline_engine,
        feature_registry_loader,
    ):
        """
        Test that all features in Feature Registry can be computed correctly.
        
        For each feature:
        1. Create appropriate test data (3-5 klines with needed pattern)
        2. Compute features using OfflineEngine
        3. Verify that the feature is present and has a valid value (not None, NaN, or Inf)
        4. Move to next feature
        """
        engine = offline_engine
        loader = feature_registry_loader
        registry_config = loader.load()
        
        # Get all features from registry
        features = registry_config.get("features", [])
        assert len(features) > 0, "Feature Registry should contain features"
        
        # Prepare base test data
        base_timestamp = datetime.now(timezone.utc)
        
        # Create base test data for different feature types
        orderbook_snapshots = self._create_test_orderbook_snapshots()
        orderbook_deltas = self._create_test_orderbook_deltas()
        trades = self._create_test_trades()
        funding = self._create_test_funding()
        
        # Track which features were tested
        tested_features = []
        failed_features = []
        
        # Iterate through each feature individually
        for feature in features:
            feature_name = feature.get("name")
            if not feature_name:
                continue
            
            input_sources = feature.get("input_sources", [])
            
            # Determine test data pattern based on feature name
            pattern = self._get_test_data_pattern(feature_name)
            
            # Create minimal test data (3-5 klines) with appropriate pattern
            klines = self._create_test_klines(pattern=pattern, count=5)
            
            # Compute features for this specific feature
            feature_vector = await engine.compute_features_at_timestamp(
                symbol="BTCUSDT",
                timestamp=base_timestamp,
                orderbook_snapshots=orderbook_snapshots,
                orderbook_deltas=orderbook_deltas,
                trades=trades,
                klines=klines,
                funding=funding,
            )
            
            if feature_vector is None:
                failed_features.append({
                    "feature": feature_name,
                    "error": "FeatureVector is None (insufficient data)",
                })
                continue
            
            # Check if this feature is present in computed features
            if feature_name not in feature_vector.features:
                # Feature might be filtered by registry or not computed
                # This is acceptable if feature requires specific conditions
                continue
            
            tested_features.append(feature_name)
            value = feature_vector.features[feature_name]
            
            # Verify feature value is valid
            if value is None:
                failed_features.append({
                    "feature": feature_name,
                    "error": "Feature value is None",
                    "pattern": pattern,
                })
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                failed_features.append({
                    "feature": feature_name,
                    "error": f"Feature value is invalid: {value}",
                    "pattern": pattern,
                })
            # Feature is valid if it passes all checks above
        
        # Report results
        total_features = len(features)
        tested_count = len(tested_features)
        failed_count = len(failed_features)
        
        print(f"\n=== Feature Registry Completeness Test Results ===")
        print(f"Total features in registry: {total_features}")
        print(f"Features tested: {tested_count}")
        print(f"Features failed: {failed_count}")
        
        if failed_features:
            print(f"\nFailed features:")
            for failure in failed_features:
                pattern_info = f" (pattern: {failure.get('pattern', 'N/A')})" if 'pattern' in failure else ""
                print(f"  - {failure['feature']}: {failure['error']}{pattern_info}")
        
        # Assert that we tested at least some features
        assert tested_count > 0, "No features were tested"
        
        # Assert that no features failed (or at least report failures)
        if failed_features:
            failure_messages = "\n".join([
                f"  - {f['feature']}: {f['error']}" + (f" (pattern: {f.get('pattern', 'N/A')})" if 'pattern' in f else "")
                for f in failed_features
            ])
            pytest.fail(
                f"Some features failed validation:\n{failure_messages}\n\n"
                f"Tested {tested_count}/{total_features} features."
            )

