#!/usr/bin/env python3
"""Script to send a test trading signal to RabbitMQ queue."""

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


async def send_test_signal():
    """Send a test trading signal to RabbitMQ queue."""
    # Create trading signal
    signal_id = uuid4()
    current_time = datetime.utcnow()
    
    # Get current price from Bybit API (or use a reasonable test value)
    # For BTCUSDT, use a test price around 90,000 USDT
    test_price = Decimal("90000.0")
    
    signal_data = {
        "signal_id": str(signal_id),
        "signal_type": "buy",  # Start with buy order
        "asset": "BTCUSDT",
        "amount": str(Decimal("100.0")),  # 100 USDT order
        "confidence": str(Decimal("0.85")),
        "timestamp": current_time.isoformat(),
        "strategy_id": "test-strategy",
        "model_version": None,
        "is_warmup": False,
        "market_data_snapshot": {
            "price": str(test_price),
            "spread": str(Decimal("0.01")),  # 1% spread
            "volume_24h": str(Decimal("1000000.0")),
            "volatility": str(Decimal("0.02")),
            "orderbook_depth": None,
            "technical_indicators": None,
        },
        "metadata": {
            "test": True,
            "source": "manual_test_script",
        },
        "trace_id": f"test-{signal_id}",
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
        )
        
        await channel.default_exchange.publish(message, routing_key=queue_name)
        
        print(f"✅ Test signal sent successfully!")
        print(f"   Signal ID: {signal_id}")
        print(f"   Asset: {signal_data['asset']}")
        print(f"   Type: {signal_data['signal_type']}")
        print(f"   Amount: {signal_data['amount']} USDT")
        print(f"   Queue: {queue_name}")
        print(f"\nCheck order-manager logs to see if order was created:")
        print(f"   docker compose logs order-manager --tail 50 --follow")
        
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(send_test_signal())

