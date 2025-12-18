"""
Historical price API.

Минимальная реализация для соответствия спецификации trading-chain.md:
- эндпоинт GET /api/v1/historical/price
- пока возвращает 501 Not Implemented, чтобы не ломать существующий функционал.

Дальше можно доработать, используя Parquet-хранилище и DataStorageService.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/historical", tags=["historical"])


@router.get("/price")
async def get_historical_price(
    symbol: str = Query(..., description="Trading symbol, e.g. BTCUSDT"),
    timestamp: datetime = Query(..., description="Target timestamp (UTC)"),
    lookback_seconds: int = Query(
        60,
        description="Lookback window in seconds for searching closest price",
        gt=0,
    ),
) -> dict:
    """
    Placeholder endpoint for historical price lookup.

    TODO:
    - Использовать DataStorageService/ParquetStorage для поиска цены.
    - Вернуть структуру с ценой(ами), на основе которых можно считать actual targets.
    """
    logger.info(
        "historical_price_not_implemented_yet",
        symbol=symbol,
        timestamp=timestamp.isoformat(),
        lookback_seconds=lookback_seconds,
    )
    raise HTTPException(
        status_code=501,
        detail="Historical price API not implemented yet",
    )


