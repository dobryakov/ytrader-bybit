"""
Helper script to publish a manual dataset.ready notification to RabbitMQ.

Usage (inside feature-service container):

    python -m scripts.publish_dataset_ready 733efdcb-2d50-4fc0-90be-465bba385778 BTCUSDT

If symbol is omitted, it will be retrieved from dataset metadata in DB.
If strategy_id is omitted, it will be retrieved from dataset metadata in DB.
"""
import asyncio
import json
import sys
from typing import Optional, Tuple

import aio_pika

from src.config import config
from src.logging import get_logger
from src.mq.connection import MQConnectionManager
from src.storage.metadata_storage import MetadataStorage

logger = get_logger(__name__)


async def publish_dataset_ready(
    dataset_id: str,
    symbol: str,
    status: str = "ready",
    train_records: int = 0,
    validation_records: int = 0,
    test_records: int = 0,
    trace_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> None:
    """Publish a dataset.ready message to RabbitMQ."""
    mq_manager = MQConnectionManager()
    channel = await mq_manager.get_channel()

    message = {
        "dataset_id": dataset_id,
        "symbol": symbol,
        "status": status,
        "train_records": train_records,
        "validation_records": validation_records,
        "test_records": test_records,
        "trace_id": trace_id,
    }
    # Add strategy_id if provided
    if strategy_id is not None:
        message["strategy_id"] = strategy_id

    body = json.dumps(message).encode()

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key="features.dataset.ready",
    )

    logger.info(
        "manual_dataset_ready_published",
        dataset_id=dataset_id,
        symbol=symbol,
        strategy_id=strategy_id,
        status=status,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        trace_id=trace_id,
        queue="features.dataset.ready",
    )


async def get_dataset_metadata(dataset_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get symbol and strategy_id from dataset metadata in DB.
    
    Returns:
        Tuple of (symbol, strategy_id) or (None, None) if dataset not found
    """
    try:
        storage = MetadataStorage()
        dataset = await storage.get_dataset(dataset_id)
        if dataset:
            symbol = dataset.get("symbol")
            strategy_id = dataset.get("strategy_id")
            logger.info(
                "Retrieved dataset metadata from DB",
                dataset_id=dataset_id,
                symbol=symbol,
                strategy_id=strategy_id,
            )
            return symbol, strategy_id
        else:
            logger.warning(
                "Dataset not found in DB",
                dataset_id=dataset_id,
            )
            return None, None
    except Exception as e:
        logger.warning(
            "Failed to retrieve dataset metadata from DB",
            dataset_id=dataset_id,
            error=str(e),
        )
        return None, None


async def main_async() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.publish_dataset_ready <DATASET_ID> [SYMBOL] [STRATEGY_ID]")
        sys.exit(1)

    dataset_id = sys.argv[1]
    symbol_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    strategy_id_arg = sys.argv[3] if len(sys.argv) >= 4 else None

    # Try to get symbol and strategy_id from DB if not provided
    symbol = symbol_arg
    strategy_id = strategy_id_arg
    
    if symbol is None or strategy_id is None:
        db_symbol, db_strategy_id = await get_dataset_metadata(dataset_id)
        
        if symbol is None:
            symbol = db_symbol or "BTCUSDT"  # Fallback to BTCUSDT if not in DB
            if db_symbol:
                logger.info(
                    "Using symbol from dataset metadata",
                    dataset_id=dataset_id,
                    symbol=symbol,
                )
            else:
                logger.warning(
                    "Symbol not found in dataset metadata, using default BTCUSDT",
                    dataset_id=dataset_id,
                )
        
        if strategy_id is None:
            strategy_id = db_strategy_id
            if db_strategy_id:
                logger.info(
                    "Using strategy_id from dataset metadata",
                    dataset_id=dataset_id,
                    strategy_id=strategy_id,
                )
            else:
                logger.info(
                    "strategy_id not found in dataset metadata, will not include it in message",
                    dataset_id=dataset_id,
                )

    # We don't know exact record counts here; set them to 0 so consumer focuses on dataset_id.
    await publish_dataset_ready(
        dataset_id=dataset_id,
        symbol=symbol,
        status="ready",
        train_records=0,
        validation_records=0,
        test_records=0,
        trace_id=f"manual-cli-{dataset_id}",
        strategy_id=strategy_id,
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


