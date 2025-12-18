"""Unit tests for FeeRateManager (Bybit fee rate cache service)."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from src.services.fee_rate_manager import FeeRateManager, FeeRate
from src.config.settings import settings


@pytest.mark.asyncio
async def test_get_fee_rate_uses_cached_value_when_fresh(monkeypatch):
    manager = FeeRateManager()

    now = datetime.now(timezone.utc)
    cached = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("0.0001"),
        taker_fee_rate=Decimal("0.0006"),
        last_synced_at=now,
    )

    monkeypatch.setattr(
        manager,
        "_get_fee_rate_from_db",
        AsyncMock(return_value=cached),
    )
    api_mock = AsyncMock()
    monkeypatch.setattr(manager, "_fetch_fee_rate_from_api", api_mock)

    with patch.object(settings, "order_manager_fee_data_ttl_seconds", 3600):
        result = await manager.get_fee_rate("BTCUSDT", "linear", trace_id="test-trace")

    assert result is cached
    api_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_fee_rate_fallbacks_to_api_when_cache_missing(monkeypatch):
    manager = FeeRateManager()

    now = datetime.now(timezone.utc)
    api_rate = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("0.0001"),
        taker_fee_rate=Decimal("0.0006"),
        last_synced_at=now,
    )

    monkeypatch.setattr(
        manager,
        "_get_fee_rate_from_db",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        manager,
        "_fetch_fee_rate_from_api",
        AsyncMock(return_value=api_rate),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(manager, "_upsert_fee_rate", upsert_mock)

    result = await manager.get_fee_rate("BTCUSDT", "linear", trace_id="test-trace")

    assert result is api_rate
    upsert_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_fee_rate_respects_ttl_and_calls_api_when_stale(monkeypatch):
    manager = FeeRateManager()

    stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
    cached = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("0.0001"),
        taker_fee_rate=Decimal("0.0006"),
        last_synced_at=stale_time,
    )
    api_rate = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("0.0002"),
        taker_fee_rate=Decimal("0.0007"),
        last_synced_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(
        manager,
        "_get_fee_rate_from_db",
        AsyncMock(return_value=cached),
    )
    monkeypatch.setattr(
        manager,
        "_fetch_fee_rate_from_api",
        AsyncMock(return_value=api_rate),
    )

    with patch.object(settings, "order_manager_fee_data_ttl_seconds", 60):
        result = await manager.get_fee_rate("BTCUSDT", "linear", trace_id="test-trace")

    assert result is api_rate



