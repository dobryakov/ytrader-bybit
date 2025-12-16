"""
Price features computation module.
"""
from typing import Dict, Optional, Iterable, Optional as Opt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows


def _ensure_datetime(value) -> datetime:
    """Ensure value is a datetime object."""
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        from dateutil.parser import parse
        return parse(value)
    else:
        return datetime.now(timezone.utc)


def compute_mid_price(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute mid price from orderbook."""
    if orderbook is None:
        return None
    
    return orderbook.get_mid_price()


def compute_spread_abs(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute absolute spread (ask - bid)."""
    if orderbook is None:
        return None
    
    return orderbook.get_spread_abs()


def compute_spread_rel(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute relative spread (spread / mid_price)."""
    if orderbook is None:
        return None
    
    return orderbook.get_spread_rel()


def compute_returns(
    rolling_windows: RollingWindows,
    window_seconds: int,
    current_price: Optional[float],
) -> Optional[float]:
    """
    Compute returns over specified window.

    Для коротких окон (< 60 секунд) используем trade-окна (price из trades),
    для длинных окон (>= 60 секунд, включая 3m/5m) используем klines ("1m"),
    чтобы быть консистентными с оффлайн-обучением и Feature Registry
    (input_sources: ["kline"] для returns_3m/returns_5m).
    """
    if current_price is None:
        return None

    # IMPORTANT: Use current time, not rolling_windows.last_update
    # This ensures that we compute returns over the correct time window,
    # even if last_update is stale (e.g., from old klines in queue)
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(seconds=window_seconds)

    # Для длинных окон используем klines (источник kline, как в обучении и реестре).
    if window_seconds >= 60:
        # Берём 1m-клайны за нужный интервал
        klines = rolling_windows.get_klines_for_window("1m", start_time, now)
        total_klines = len(rolling_windows.get_window_data("1m"))
        
        import structlog
        logger = structlog.get_logger(__name__)
        logger.info(
            "compute_returns_klines_check",
            window_seconds=window_seconds,
            window_minutes=window_seconds / 60,
            klines_in_window=len(klines),
            total_klines_available=total_klines,
            has_close="close" in klines.columns if len(klines) > 0 else False,
            start_time=start_time.isoformat(),
            now=now.isoformat(),
            time_diff_seconds=(now - start_time).total_seconds(),
        )
        
        if len(klines) == 0 or "close" not in klines.columns:
            logger.warning(
                "compute_returns_no_klines",
                window_seconds=window_seconds,
                window_minutes=window_seconds / 60,
                klines_count=len(klines),
                has_close="close" in klines.columns if len(klines) > 0 else False,
                start_time=start_time.isoformat(),
                now=now.isoformat(),
                total_klines_in_window=total_klines,
            )
            return None

        klines_sorted = klines.sort_values("timestamp")
        if len(klines_sorted) == 0:
            logger.warning("compute_returns_empty_after_sort", window_seconds=window_seconds)
            return None

        # Первая цена в окне — исходная точка для доходности
        first_close_raw = klines_sorted.iloc[0]["close"]
        first_price = pd.to_numeric(first_close_raw, errors="coerce")
        
        # Проверим несколько первых свечей, если первая имеет close=0
        if pd.isna(first_price) or first_price == 0:
            # Попробуем найти первую свечу с ненулевым close
            non_zero_closes = klines_sorted[klines_sorted["close"].notna()]
            non_zero_closes = non_zero_closes[pd.to_numeric(non_zero_closes["close"], errors="coerce") > 0]
            
            logger.warning(
                "compute_returns_invalid_first_price",
                window_seconds=window_seconds,
                first_price=first_price,
                first_price_raw=first_close_raw,
                first_price_type=type(first_close_raw).__name__,
                first_timestamp=klines_sorted.iloc[0]["timestamp"] if "timestamp" in klines_sorted.columns else None,
                total_klines=len(klines_sorted),
                non_zero_closes_count=len(non_zero_closes),
                sample_closes=list(klines_sorted["close"].head(5).tolist()) if "close" in klines_sorted.columns else [],
            )
            
            # Если есть свечи с ненулевым close, используем первую из них
            if len(non_zero_closes) > 0:
                first_price = pd.to_numeric(non_zero_closes.iloc[0]["close"], errors="coerce")
                logger.info(
                    "compute_returns_using_non_zero_close",
                    window_seconds=window_seconds,
                    found_at_index=non_zero_closes.index[0],
                    first_price=first_price,
                )
            else:
                return None
        
        logger.debug(
            "compute_returns_success",
            window_seconds=window_seconds,
            klines_count=len(klines_sorted),
            first_price=first_price,
            current_price=current_price,
        )
    else:
        # Для очень коротких окон (< 60 секунд) используем trades-окна
        window_data = rolling_windows.get_window_data(f"{window_seconds}s")
        if len(window_data) == 0 or "price" not in window_data.columns:
            return None

        window_data_sorted = window_data.sort_values("timestamp")
        if len(window_data_sorted) == 0:
            return None

        # Ensure price is numeric (convert from string if needed)
        first_price = pd.to_numeric(window_data_sorted.iloc[0]["price"], errors="coerce")
        if pd.isna(first_price) or first_price == 0:
            return None

    # Compute return: (current - first) / first
    return float((current_price - first_price) / first_price)


def compute_vwap(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute Volume-Weighted Average Price (VWAP) over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return None
    
    if "price" not in trades.columns or "volume" not in trades.columns:
        return None
    
    # Ensure numeric types (convert from string if needed)
    prices = pd.to_numeric(trades["price"], errors='coerce').fillna(0.0)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    
    # VWAP = sum(price * volume) / sum(volume)
    total_value = (prices * volumes).sum()
    total_volume = volumes.sum()
    
    if total_volume == 0:
        return None
    
    return total_value / total_volume


def compute_volume(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute total volume over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return 0.0
    
    if "volume" not in trades.columns:
        return 0.0
    
    # Ensure numeric type (convert from string if needed)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    return float(volumes.sum())


def compute_volatility(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute volatility (standard deviation of returns) over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get klines for window
    klines = rolling_windows.get_klines_for_window("1m", start_time, now)
    total_klines = len(rolling_windows.get_window_data("1m"))
    
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info(
        "compute_volatility_klines_check",
        window_seconds=window_seconds,
        window_minutes=window_seconds / 60,
        klines_in_window=len(klines),
        total_klines_available=total_klines,
        required_count=2,
        start_time=start_time.isoformat(),
        now=now.isoformat(),
        time_diff_seconds=(now - start_time).total_seconds(),
    )
    
    if len(klines) < 2:
        logger.warning(
            "compute_volatility_insufficient_klines",
            window_seconds=window_seconds,
            window_minutes=window_seconds / 60,
            klines_count=len(klines),
            required_count=2,
            start_time=start_time.isoformat(),
            now=now.isoformat(),
            total_klines_in_window=total_klines,
        )
        return None
    
    if "close" not in klines.columns:
        logger.warning(
            "compute_volatility_no_close_column",
            window_seconds=window_seconds,
            klines_count=len(klines),
            columns=list(klines.columns) if len(klines) > 0 else [],
        )
        return None
    
    # Compute returns from close prices
    # Ensure closes are numeric (convert from string if needed)
    # First, get the close column and convert to numeric
    close_col = klines["close"]
    
    # Convert to numeric, handling any string values
    closes_series = pd.to_numeric(close_col, errors='coerce')
    
    # Fill NaN with 0.0 and convert to float
    closes_series = closes_series.fillna(0.0).astype(float)
    
    # Convert to numpy array with explicit float dtype
    closes = np.array(closes_series.values, dtype=float)
    
    # Additional safety check: ensure we have a proper float array
    if closes.dtype != np.float64:
        closes = closes.astype(float)
    
    if len(closes) < 2:
        logger.warning(
            "compute_volatility_insufficient_closes_after_conversion",
            window_seconds=window_seconds,
            closes_count=len(closes),
            required_count=2,
        )
        return None
    
    # Filter out zero values to avoid division by zero
    closes_before_filter = len(closes)
    closes = closes[closes > 0]
    closes_after_filter = len(closes)
    
    # Ensure dtype is still float after filtering
    if len(closes) > 0 and closes.dtype != np.float64:
        closes = closes.astype(float)
    
    if len(closes) < 2:
        logger.warning(
            "compute_volatility_insufficient_closes_after_zero_filter",
            window_seconds=window_seconds,
            closes_before_filter=closes_before_filter,
            closes_after_filter=closes_after_filter,
            required_count=2,
            zero_count=closes_before_filter - closes_after_filter,
        )
        return None
    
    # Compute returns with explicit float conversion
    closes_float = closes.astype(float)
    returns = np.diff(closes_float) / closes_float[:-1]
    
    if len(returns) == 0:
        logger.warning(
            "compute_volatility_empty_returns",
            window_seconds=window_seconds,
            closes_count=len(closes),
        )
        return None
    
    # Compute standard deviation of returns
    volatility = np.std(returns)
    
    logger.debug(
        "compute_volatility_success",
        window_seconds=window_seconds,
        klines_count=len(klines),
        closes_count=len(closes),
        returns_count=len(returns),
        volatility=volatility,
    )
    
    return float(volatility)


def compute_price_ema21_ratio(
    rolling_windows: RollingWindows,
    current_price: Optional[float],
) -> Optional[float]:
    """
    Compute price to EMA21 ratio.
    
    Computes EMA(21) using technical_indicators.compute_ema_21() and
    computes ratio as current_price / ema_21.
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        current_price: Current price (from latest kline close or provided parameter)
        
    Returns:
        Ratio value or None if EMA is None or zero
    """
    if current_price is None:
        return None
    
    # Import here to avoid circular dependency
    from src.features.technical_indicators import compute_ema_21
    
    # Compute EMA(21)
    ema_21 = compute_ema_21(rolling_windows)
    
    if ema_21 is None or ema_21 == 0:
        return None
    
    # Compute ratio: current_price / ema_21
    return float(current_price / ema_21)


def compute_volume_ratio_20(
    rolling_windows: RollingWindows,
    current_volume: Optional[float],
    candle_interval: str = "1m",
) -> Optional[float]:
    """
    Compute volume to 20-period MA ratio.
    
    Gets historical volumes from last 20 candles, computes volume_ma_20 as
    simple moving average of volumes over 20 periods, and computes ratio as
    current_volume / volume_ma_20.
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        current_volume: Current volume (from latest kline or provided parameter)
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        Ratio value or None if insufficient data or zero average volume
    """
    if current_volume is None:
        return None
    
    now = _ensure_datetime(rolling_windows.last_update)
    # Get enough history for 20 completed candles (excluding current incomplete candle)
    # Request 21 minutes to ensure we have 20 completed candles even if current one is included
    start_time = now - timedelta(minutes=21)
    
    # Get klines for window (may include current incomplete candle)
    klines = rolling_windows.get_klines_for_window(candle_interval, start_time, now)
    
    if len(klines) < 20:
        return None
    
    if "volume" not in klines.columns:
        return None
    
    # Sort by timestamp
    klines_sorted = klines.sort_values("timestamp")
    if len(klines_sorted) < 20:
        return None
    
    # Get last 20 COMPLETED candles (exclude current incomplete candle if present)
    # If we have more than 20 candles, take the 20 before the last one
    if len(klines_sorted) > 20:
        # Exclude the last (current) candle and take previous 20
        completed_klines = klines_sorted.iloc[:-1].tail(20)
    else:
        # Exactly 20 candles - use all of them (assuming they're all completed)
        completed_klines = klines_sorted.tail(20)
    
    # Get volumes from completed candles only
    volumes = pd.to_numeric(completed_klines["volume"], errors='coerce').fillna(0.0)
    
    # Compute volume_ma_20 as simple moving average
    volume_ma_20 = float(volumes.mean())
    
    if volume_ma_20 == 0:
        return None
    
    # Compute ratio: current_volume / volume_ma_20
    return float(current_volume / volume_ma_20)


def _should_compute(name: str, allowed: Opt[Iterable[str]]) -> bool:
    """
    Вспомогательная функция: решает, нужно ли вообще считать фичу.

    Если allowed is None → считаем все (старое поведение).
    """
    if allowed is None:
        return True
    return name in allowed


def compute_all_price_features(
    orderbook: Optional[OrderbookState],
    rolling_windows: RollingWindows,
    current_price: Optional[float],
    allowed_feature_names: Opt[Iterable[str]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute all price features.

    allowed_feature_names — опциональный список имён фич из Feature Registry.
    Если он передан, мы не только фильтруем по нему, но и не считаем лишние фичи.
    """
    features = {}
    
    # Basic price features from orderbook
    if _should_compute("mid_price", allowed_feature_names):
        features["mid_price"] = compute_mid_price(orderbook)
    if _should_compute("spread_abs", allowed_feature_names):
        features["spread_abs"] = compute_spread_abs(orderbook)
    if _should_compute("spread_rel", allowed_feature_names):
        features["spread_rel"] = compute_spread_rel(orderbook)
    
    # Returns
    if _should_compute("returns_1s", allowed_feature_names):
        features["returns_1s"] = compute_returns(rolling_windows, 1, current_price)
    if _should_compute("returns_3s", allowed_feature_names):
        features["returns_3s"] = compute_returns(rolling_windows, 3, current_price)
    if _should_compute("returns_1m", allowed_feature_names):
        features["returns_1m"] = compute_returns(rolling_windows, 60, current_price)
    # Long-term returns
    if _should_compute("returns_3m", allowed_feature_names):
        result_3m = compute_returns(rolling_windows, 180, current_price)
        features["returns_3m"] = result_3m
        import structlog
        logger = structlog.get_logger(__name__)
        if result_3m is None:
            logger.warning(
                "returns_3m_computed_as_none",
                current_price=current_price,
                total_klines=len(rolling_windows.get_window_data("1m")),
            )
        else:
            logger.debug(
                "returns_3m_computed_success",
                value=result_3m,
                current_price=current_price,
            )
    if _should_compute("returns_5m", allowed_feature_names):
        result_5m = compute_returns(rolling_windows, 300, current_price)
        features["returns_5m"] = result_5m
        import structlog
        logger = structlog.get_logger(__name__)
        if result_5m is None:
            logger.warning(
                "returns_5m_computed_as_none",
                current_price=current_price,
                total_klines=len(rolling_windows.get_window_data("1m")),
            )
        else:
            logger.debug(
                "returns_5m_computed_success",
                value=result_5m,
                current_price=current_price,
            )
    
    # VWAP
    if _should_compute("vwap_3s", allowed_feature_names):
        features["vwap_3s"] = compute_vwap(rolling_windows, 3)
    if _should_compute("vwap_15s", allowed_feature_names):
        features["vwap_15s"] = compute_vwap(rolling_windows, 15)
    if _should_compute("vwap_1m", allowed_feature_names):
        features["vwap_1m"] = compute_vwap(rolling_windows, 60)
    # Long-term VWAP
    if _should_compute("vwap_3m", allowed_feature_names):
        features["vwap_3m"] = compute_vwap(rolling_windows, 180)
    if _should_compute("vwap_5m", allowed_feature_names):
        features["vwap_5m"] = compute_vwap(rolling_windows, 300)
    
    # Volume
    if _should_compute("volume_3s", allowed_feature_names):
        features["volume_3s"] = compute_volume(rolling_windows, 3)
    if _should_compute("volume_15s", allowed_feature_names):
        features["volume_15s"] = compute_volume(rolling_windows, 15)
    if _should_compute("volume_1m", allowed_feature_names):
        features["volume_1m"] = compute_volume(rolling_windows, 60)
    # Long-term volume
    if _should_compute("volume_3m", allowed_feature_names):
        features["volume_3m"] = compute_volume(rolling_windows, 180)
    if _should_compute("volume_5m", allowed_feature_names):
        features["volume_5m"] = compute_volume(rolling_windows, 300)
    
    # Volatility
    if _should_compute("volatility_1m", allowed_feature_names):
        features["volatility_1m"] = compute_volatility(rolling_windows, 60)
    if _should_compute("volatility_5m", allowed_feature_names):
        features["volatility_5m"] = compute_volatility(rolling_windows, 300)
    # Long-term volatility
    if _should_compute("volatility_10m", allowed_feature_names):
        result_10m = compute_volatility(rolling_windows, 600)
        features["volatility_10m"] = result_10m
        import structlog
        logger = structlog.get_logger(__name__)
        if result_10m is None:
            logger.warning(
                "volatility_10m_computed_as_none",
                total_klines=len(rolling_windows.get_window_data("1m")),
            )
        else:
            logger.debug(
                "volatility_10m_computed_success",
                value=result_10m,
            )
    if _should_compute("volatility_15m", allowed_feature_names):
        result_15m = compute_volatility(rolling_windows, 900)
        features["volatility_15m"] = result_15m
        import structlog
        logger = structlog.get_logger(__name__)
        if result_15m is None:
            logger.warning(
                "volatility_15m_computed_as_none",
                total_klines=len(rolling_windows.get_window_data("1m")),
            )
        else:
            logger.debug(
                "volatility_15m_computed_success",
                value=result_15m,
            )
    
    # Price to EMA21 ratio
    if _should_compute("price_ema21_ratio", allowed_feature_names):
        features["price_ema21_ratio"] = compute_price_ema21_ratio(
            rolling_windows, current_price
        )
    
    # Note: rsi_14 and ema_21 are computed in compute_all_technical_indicators()
    # to avoid duplication. They will be added separately.
    
    # Volume ratio
    # Get current volume from latest kline or use volume_1m
    current_volume = features.get("volume_1m")
    if current_volume is None:
        # Try to get from latest kline
        now = _ensure_datetime(rolling_windows.last_update)
        klines = rolling_windows.get_klines_for_window("1m", now - timedelta(minutes=1), now)
        if len(klines) > 0 and "volume" in klines.columns:
            klines_sorted = klines.sort_values("timestamp")
            current_volume = pd.to_numeric(klines_sorted.iloc[-1]["volume"], errors='coerce')
            if pd.isna(current_volume):
                current_volume = None
    
    if _should_compute("volume_ratio_20", allowed_feature_names):
        features["volume_ratio_20"] = compute_volume_ratio_20(
            rolling_windows, current_volume
        )
    
    return features

