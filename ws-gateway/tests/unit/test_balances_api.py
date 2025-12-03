"""Unit tests for balance REST API endpoints."""

from datetime import datetime, timedelta
from decimal import Decimal

from fastapi.testclient import TestClient

from src.main import app
from src.models.account_balance import AccountBalance
from src.models.account_margin_balance import AccountMarginBalance
from src.services.database.account_margin_balance_repository import (
    AccountMarginBalanceRepository,
)
from src.services.database.balance_repository import BalanceRepository
from src.config.settings import settings


def _make_balance(coin: str = "USDT") -> AccountBalance:
    now = datetime.utcnow()
    return AccountBalance.create(
        coin=coin,
        wallet_balance=Decimal("100"),
        available_balance=Decimal("80"),
        frozen=Decimal("20"),
        event_timestamp=now - timedelta(seconds=10),
        trace_id="test-trace",
    )


def _make_margin_balance() -> AccountMarginBalance:
    now = datetime.utcnow()
    return AccountMarginBalance.create(
        account_type="UNIFIED",
        total_equity=Decimal("100"),
        total_wallet_balance=Decimal("100"),
        total_margin_balance=Decimal("80"),
        total_available_balance=Decimal("20"),
        total_initial_margin=Decimal("50"),
        total_maintenance_margin=Decimal("10"),
        total_order_im=Decimal("5"),
        base_currency="USDT",
        event_timestamp=now - timedelta(seconds=5),
    )


def _client() -> TestClient:
    return TestClient(app)


def test_get_latest_balances(monkeypatch):
    async def fake_list_latest_balances(coin=None, limit=None, offset=None):
        return [_make_balance("USDT"), _make_balance("BTC")]

    async def fake_get_latest_margin_balance():
        return _make_margin_balance()

    monkeypatch.setattr(
        BalanceRepository,
        "list_latest_balances",
        staticmethod(fake_list_latest_balances),
    )
    monkeypatch.setattr(
        AccountMarginBalanceRepository,
        "get_latest_margin_balance",
        staticmethod(fake_get_latest_margin_balance),
    )

    client = _client()
    headers = {"X-API-Key": settings.ws_gateway_api_key}

    response = client.get("/api/v1/balances", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 2
    assert len(body["balances"]) == 2
    coins = {b["coin"] for b in body["balances"]}
    assert coins == {"USDT", "BTC"}
    assert body["margin_balance"]["account_type"] == "UNIFIED"
    assert body["margin_balance"]["base_currency"] == "USDT"


def test_get_balance_history(monkeypatch):
    balances = [_make_balance("USDT"), _make_balance("BTC")]

    async def fake_list_balances(
        coin=None, limit=None, offset=None, start_time=None, end_time=None
    ):
        return balances

    monkeypatch.setattr(
        BalanceRepository,
        "list_balances",
        staticmethod(fake_list_balances),
    )

    client = _client()
    headers = {"X-API-Key": settings.ws_gateway_api_key}

    response = client.get("/api/v1/balances/history", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 2
    assert len(body["balances"]) == 2
    coins = {b["coin"] for b in body["balances"]}
    assert coins == {"USDT", "BTC"}


def test_get_balance_history_invalid_range(monkeypatch):
    client = _client()
    headers = {"X-API-Key": settings.ws_gateway_api_key}

    now = datetime.utcnow()
    earlier = (now - timedelta(hours=1)).isoformat()
    later = now.isoformat()

    # from > to should fail
    response = client.get(
        f"/api/v1/balances/history?from={later}&to={earlier}",
        headers=headers,
    )
    assert response.status_code == 400


def test_sync_balances_not_implemented(monkeypatch):
    async def fake_sync_from_rest():
        return {"updated_coins": ["USDT"], "updated_count": 3}

    from src.services.database.balance_service import BalanceService

    monkeypatch.setattr(
        BalanceService,
        "sync_from_rest",
        staticmethod(fake_sync_from_rest),
    )

    client = _client()
    headers = {"X-API-Key": settings.ws_gateway_api_key}

    response = client.post("/api/v1/balances/sync", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["updated_coins"] == ["USDT"]
    assert body["updated_count"] == 3


