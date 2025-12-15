"""
Helper script to publish a manual dataset.ready notification to RabbitMQ.

Usage (inside feature-service container):

    python -m scripts.publish_dataset_ready 733efdcb-2d50-4fc0-90be-465bba385778 BTCUSDT

If symbol is omitted, BTCUSDT will be used by default.
"""
import asyncio
import json
import sys
from typing import Optional

import aio_pika

from src.config import config
from src.logging import get_logger
from src.mq.connection import MQConnectionManager

logger = get_logger(__name__)


async def publish_dataset_ready(
    dataset_id: str,
    symbol: str,
    status: str = "ready",
    train_records: int = 0,
    validation_records: int = 0,
    test_records: int = 0,
    trace_id: Optional[str] = None,
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
        status=status,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        trace_id=trace_id,
        queue="features.dataset.ready",
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.publish_dataset_ready <DATASET_ID> [SYMBOL]")
        sys.exit(1)

    dataset_id = sys.argv[1]
    symbol = sys.argv[2] if len(sys.argv) >= 3 else "BTCUSDT"

    # We don't know exact record counts here; set them to 0 so consumer focuses on dataset_id.
    asyncio.run(
        publish_dataset_ready(
            dataset_id=dataset_id,
            symbol=symbol,
            status="ready",
            train_records=0,
            validation_records=0,
            test_records=0,
            trace_id=f"manual-cli-{dataset_id}",
        )
    )


if __name__ == "__main__":
    main()


