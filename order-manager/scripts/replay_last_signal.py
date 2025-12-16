#!/usr/bin/env python3
"""Script to replay the last trading signal from database to RabbitMQ queue."""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import aio_pika
from aio_pika import Message

# Import settings from order-manager
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import settings


async def replay_last_signal():
    """Replay the last trading signal from database."""
    # Last signal data from database
    # signal_id: 64d49563-d832-48b6-a846-5ce24ed40843
    # side: sell
    # asset: BTCUSDT
    # price: 94653.70
    # confidence: 0.6763
    # strategy_id: test-strategy
    # timestamp: 2025-12-16 18:35:40.194252
    # model_version: v1765879286
    # is_warmup: false
    
    # Generate new signal_id to avoid duplicate
    signal_id = str(uuid4())
    current_time = datetime.utcnow()
    
    signal_data = {
        "signal_id": signal_id,
        "signal_type": "sell",  # From last signal
        "asset": "BTCUSDT",
        "amount": "100.0",  # Default amount (not stored in DB)
        "confidence": "0.6763",
        "timestamp": current_time.isoformat() + "Z",
        "strategy_id": "test-strategy",
        "model_version": "v1765879286",
        "is_warmup": False,
        "market_data_snapshot": {
            "price": "94653.70",
            "spread": "0.0",
            "volatility": "0.0",
            "volume_24h": "0.0",
            "orderbook_depth": None,
            "technical_indicators": None,
        },
        "metadata": {
            "reasoning": "Model prediction: -1",
            "model_version": "v1765879286",
            "prediction_result": {
                "prediction": -1,
                "buy_probability": 0.32367271184921265,
                "sell_probability": 0.6763272881507874
            },
            "inference_timestamp": current_time.isoformat() + "Z",
            "feature_registry_version": "1.5.0"
        },
        "trace_id": f"replay-{signal_id}",
    }
    
    # Connect to RabbitMQ
    rabbitmq_url = f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}@{settings.rabbitmq_host}:{settings.rabbitmq_port}/"
    
    print(f"Connecting to RabbitMQ at {settings.rabbitmq_host}:{settings.rabbitmq_port}...")
    connection = await aio_pika.connect_robust(rabbitmq_url)
    
    try:
        channel = await connection.channel()
        
        # Queue name for trading signals (from model-service to order-manager)
        queue_name = "model-service.trading_signals"
        
        # Declare queue (ensure it exists)
        queue = await channel.declare_queue(queue_name, durable=True)
        
        # Publish signal
        message_body = json.dumps(signal_data)
        message = Message(
            message_body.encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"trace_id": f"replay-{signal_id}"},
        )
        
        await channel.default_exchange.publish(message, routing_key=queue_name)
        
        print(f"✅ Signal replayed successfully!")
        print(f"   Signal ID: {signal_id}")
        print(f"   Asset: {signal_data['asset']}")
        print(f"   Type: {signal_data['signal_type']}")
        print(f"   Amount: {signal_data['amount']} USDT")
        print(f"   Confidence: {signal_data['confidence']}")
        print(f"   Queue: {queue_name}")
        print(f"\nCheck order-manager logs to see if order was created:")
        print(f"   docker compose logs order-manager --tail 50 --follow")
        
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(replay_last_signal())

