"""
Feature requirements analyzer for online feature computation.

Определяет:
- какие интервалы rolling windows (trades / klines) реально нужны онлайн-движку;
- максимальный lookback по 1m-клайнам, чтобы все фичи могли посчитаться.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set, Iterable

import structlog

from src.services.feature_registry import FeatureRegistryLoader

logger = structlog.get_logger(__name__)


@dataclass
class WindowRequirements:
    """
    Требования к окнам для онлайн-FeatureComputer.

    - trade_intervals: интервалы trades, которые нужно поддерживать в RollingWindows.windows
    - max_lookback_minutes_1m: максимальный lookback по 1m-клайнам, с запасом
    """

    trade_intervals: Set[str]
    max_lookback_minutes_1m: int


class FeatureRequirementsAnalyzer:
    """
    Анализирует активный Feature Registry и вычисляет требования к окнам.

    Логика:
    - по именам фич и их lookback_window определяем, нужны ли короткие трейд-окна
      (1s/3s/15s) или достаточно минутных/многоминутных окон;
    - считаем максимальный lookback по 1m-клайнам для ценовых/технических фич.
    """

    # Жёстко заданные требования к lookback по именам фич
    _FEATURE_LOOKBACK_MAPPING: Dict[str, int] = {
        # Technical indicators
        "ema_21": 26,  # 21 минут + 5 минут буфер
        "rsi_14": 19,  # 14 минут + буфер
        # Price features
        "price_ema21_ratio": 26,  # зависит от ema_21
        "volume_ratio_20": 20,  # 20 минут
        "volatility_5m": 6,  # 5 минут + буфер
        "volatility_10m": 12,
        "volatility_15m": 17,
        "returns_5m": 6,
        "returns_3m": 4,
        "returns_1m": 2,
        # Default buffer
        "_default_buffer": 5,
    }

    # Фичи, которые требуют именно трейдовых коротких окон (< 60s)
    _TRADE_WINDOW_FEATURES: Set[str] = {
        "returns_1s",
        "returns_3s",
        "vwap_3s",
        "vwap_15s",
        "volume_3s",
        "volume_15s",
        "signed_volume_1s",
        "signed_volume_3s",
        "signed_volume_15s",
    }

    # Какие трейдовые интервалы соответствуют каким фичам
    _TRADE_INTERVALS_BY_FEATURE: Dict[str, Set[str]] = {
        "returns_1s": {"1s"},
        "returns_3s": {"3s"},
        "vwap_3s": {"3s"},
        "vwap_15s": {"15s"},
        "volume_3s": {"3s"},
        "volume_15s": {"15s"},
        "signed_volume_1s": {"1s"},
        "signed_volume_3s": {"3s"},
        "signed_volume_15s": {"15s"},
    }

    def __init__(self, loader: Optional[FeatureRegistryLoader]) -> None:
        self._loader = loader

    def _iter_feature_defs(self) -> Iterable:
        """
        Итератор по объектам фич из FeatureRegistry (model-based API).
        """
        if self._loader is None:
            return []

        registry_model = getattr(self._loader, "_registry_model", None)
        if registry_model is None or not getattr(registry_model, "features", None):
            # Попробуем лениво загрузить конфиг, если модель ещё не инициализирована
            try:
                config = self._loader.get_config()
                if not config:
                    return []
                # _validate_and_store_config уже вызывалась при загрузке; если нет – это
                # значит, что registry использовался только как dict. Для требований
                # достаточно пройти по dict'у.
                for feat in config.get("features", []):
                    yield type("FeatureLike", (), feat)
                return
            except Exception as e:
                logger.warning(
                    "feature_requirements_iter_failed_config",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return []

        return registry_model.features

    def compute_requirements(self) -> WindowRequirements:
        """
        Основной метод: возвращает WindowRequirements для онлайн-движка.

        Если реестр недоступен / пустой – используем безопасные дефолты:
        - trade_intervals = {"1m"} (только минутное окно)
        - max_lookback_minutes_1m = 30
        """
        trade_intervals: Set[str] = set()
        max_lookback_minutes_1m: int = 0

        # Специальный случай: loader отсутствует → используем консервативный
        # дефолт (все онлайн-окна, как в исходной реализации).
        if self._loader is None:
            logger.info(
                "feature_requirements_no_loader_using_defaults",
                trade_intervals=["1s", "3s", "15s", "1m"],
                max_lookback_minutes_1m=30,
            )
            return WindowRequirements(
                trade_intervals={"1s", "3s", "15s", "1m"},
                max_lookback_minutes_1m=30,
            )

        try:
            for feature in self._iter_feature_defs():
                feature_name = getattr(feature, "name", None)
                lookback_window = getattr(feature, "lookback_window", None)
                input_sources = getattr(feature, "input_sources", None)

                if not feature_name:
                    continue

                # 1) трейдовые фичи → включаем короткие окна
                if feature_name in self._TRADE_WINDOW_FEATURES:
                    trade_intervals.update(
                        self._TRADE_INTERVALS_BY_FEATURE.get(feature_name, set())
                    )

                # 2) lookback по 1m-клайнам
                mapped = self._FEATURE_LOOKBACK_MAPPING.get(feature_name)
                if mapped is not None:
                    max_lookback_minutes_1m = max(max_lookback_minutes_1m, mapped)
                else:
                    parsed = self._parse_lookback_window(lookback_window)
                    if parsed is not None:
                        buf = self._FEATURE_LOOKBACK_MAPPING.get("_default_buffer", 5)
                        max_lookback_minutes_1m = max(
                            max_lookback_minutes_1m, parsed + buf
                        )

                # 3) Если фича использует только trades и очень короткий lookback в секундах,
                # то тоже считаем, что нужны трейдовые окна.
                if input_sources and isinstance(input_sources, list):
                    if "trades" in input_sources and lookback_window:
                        parsed_sec = self._parse_lookback_seconds(lookback_window)
                        if parsed_sec is not None and parsed_sec < 60:
                            # по умолчанию включим 1s/3s/15s, если явно не указано обратное
                            trade_intervals.update({"1s", "3s", "15s"})

        except Exception as e:
            logger.warning(
                "feature_requirements_compute_failed",
                error=str(e),
                error_type=type(e).__name__,
                fallback="using_defaults",
            )

        # дефолтный безопасный lookback – 30 минут (как в OfflineEngine)
        if max_lookback_minutes_1m <= 0 or max_lookback_minutes_1m < 26:
            # минимум, чтобы покрыть ema_21
            if max_lookback_minutes_1m > 0 and max_lookback_minutes_1m < 26:
                logger.warning(
                    "feature_requirements_lookback_too_small",
                    computed=max_lookback_minutes_1m,
                    minimum_required=26,
                    using_default=30,
                )
            max_lookback_minutes_1m = 30

        # Если трейдовые окна не нужны – оставляем только минутное трейд-окно
        if not trade_intervals:
            trade_intervals = {"1m"}
        else:
            # всегда добавляем 1m, чтобы compute_vwap/volume_1m и др. могли работать
            trade_intervals.add("1m")

        logger.info(
            "feature_requirements_computed",
            trade_intervals=sorted(trade_intervals),
            max_lookback_minutes_1m=max_lookback_minutes_1m,
        )

        return WindowRequirements(
            trade_intervals=trade_intervals,
            max_lookback_minutes_1m=max_lookback_minutes_1m,
        )

    @staticmethod
    def _parse_lookback_window(lookback_window: Optional[str]) -> Optional[int]:
        """
        Разбор lookback_window в минутах.
        Поддерживает суффиксы: s, m, h, d.
        """
        if not lookback_window:
            return None

        try:
            unit = lookback_window[-1]
            value = int(lookback_window[:-1])

            if unit == "s":
                return max(0, value // 60)
            if unit == "m":
                return value
            if unit == "h":
                return value * 60
            if unit == "d":
                return value * 24 * 60
        except (ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _parse_lookback_seconds(lookback_window: Optional[str]) -> Optional[int]:
        """
        Разбор lookback_window в секундах (для трейдовых коротких окон).
        """
        if not lookback_window:
            return None

        try:
            unit = lookback_window[-1]
            value = int(lookback_window[:-1])

            if unit == "s":
                return value
            if unit == "m":
                return value * 60
            if unit == "h":
                return value * 3600
            if unit == "d":
                return value * 86400
        except (ValueError, IndexError):
            return None
        return None


