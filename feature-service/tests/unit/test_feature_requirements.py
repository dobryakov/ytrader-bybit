"""
Unit tests for FeatureRequirementsAnalyzer.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.services.feature_requirements import FeatureRequirementsAnalyzer, WindowRequirements


class TestFeatureRequirementsAnalyzer:
    """Tests for FeatureRequirementsAnalyzer."""

    def _make_loader(self, features):
        """Create a fake FeatureRegistryLoader with given features."""
        loader = MagicMock()

        # Простая модель реестра с атрибутом features
        class RegistryModel:
            def __init__(self, feats):
                self.features = feats
                self.version = "test-version"

        class FeatureObj:
            def __init__(self, name, lookback_window=None, input_sources=None):
                self.name = name
                self.lookback_window = lookback_window
                self.input_sources = input_sources or []

        loader._registry_model = RegistryModel(
            [FeatureObj(**f) for f in features]
        )
        return loader

    def test_defaults_when_loader_is_none(self):
        """Analyzer with None loader should return safe defaults."""
        analyzer = FeatureRequirementsAnalyzer(loader=None)
        req = analyzer.compute_requirements()

        assert isinstance(req, WindowRequirements)
        # Без реестра используем консервативный дефолт: все онлайн-окна
        assert req.trade_intervals == {"1s", "3s", "15s", "1m"}
        assert req.max_lookback_minutes_1m == 30

    def test_only_kline_long_term_features(self):
        """Если в реестре только длинные kline-фичи, трейдовые окна не нужны."""
        features = [
            {"name": "returns_5m", "lookback_window": "5m", "input_sources": ["kline"]},
            {"name": "volatility_10m", "lookback_window": "10m", "input_sources": ["kline"]},
        ]
        loader = self._make_loader(features)
        analyzer = FeatureRequirementsAnalyzer(loader)

        req = analyzer.compute_requirements()

        # Должно быть только минутное трейд-окно (для совместимости)
        assert req.trade_intervals == {"1m"}
        # lookback должен быть не меньше маппинга для этих фич (см. _FEATURE_LOOKBACK_MAPPING)
        assert req.max_lookback_minutes_1m >= 12

    def test_includes_short_trade_windows_when_needed(self):
        """Если в реестре есть короткие trade-фичи, должны появиться 1s/3s/15s окна."""
        features = [
            {
                "name": "returns_1s",
                "lookback_window": "1s",
                "input_sources": ["trades"],
            },
            {
                "name": "vwap_15s",
                "lookback_window": "15s",
                "input_sources": ["trades"],
            },
        ]
        loader = self._make_loader(features)
        analyzer = FeatureRequirementsAnalyzer(loader)

        req = analyzer.compute_requirements()

        assert "1s" in req.trade_intervals
        assert "15s" in req.trade_intervals
        assert "1m" in req.trade_intervals  # 1m всегда добавляется


