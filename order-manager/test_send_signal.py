#!/usr/bin/env python3
"""Test script to send a trading signal to RabbitMQ queue for testing order-manager service."""

import json
import sys
import uuid
from datetime import datetime, timezone

import pika

# Test signal data
test_signal = {
    "signal_id": str(uuid.uuid4()),
    "signal_type": "buy",
    "asset": "BTCUSDT",
    "amount": "1000.0",
    "confidence": "0.85",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy_id": "test-strategy",
    "model_version": None,
    "is_warmup": True,
    "market_data_snapshot": {
        "price": "50000.0",
        "spread": "0.0015",  # 0.15%
        "volume_24h": "1000000.0",
        "volatility": "0.02",
        "orderbook_depth": {
            "bid_depth": "100.0",
            "ask_depth": "120.0"
        },
        "technical_indicators": None
    },
    "metadata": {
        "reasoning": "Test signal for order-manager",
        "risk_score": "0.3"
    },
    "trace_id": f"test-trace-{uuid.uuid4()}"
}

def send_signal():
    """Send test signal to RabbitMQ queue."""
    # Connection parameters - adjust if needed
    connection_params = pika.ConnectionParameters(
        host='localhost',
        port=5672,
        credentials=pika.PlainCredentials('guest', 'guest')
    )
    
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        
        # Declare queue (ensure it exists)
        channel.queue_declare(queue='model-service.trading_signals', durable=True)
        
        # Publish message
        message_body = json.dumps(test_signal)
        channel.basic_publish(
            exchange='',
            routing_key='model-service.trading_signals',
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                headers={'trace_id': test_signal['trace_id']}
            )
        )
        
        print(f"✅ Signal sent successfully!")
        print(f"   Signal ID: {test_signal['signal_id']}")
        print(f"   Type: {test_signal['signal_type']}")
        print(f"   Asset: {test_signal['asset']}")
        print(f"   Amount: {test_signal['amount']} USDT")
        print(f"   Trace ID: {test_signal['trace_id']}")
        print(f"\n📋 Full signal JSON:")
        print(json.dumps(test_signal, indent=2))
        
        connection.close()
        return 0
        
    except pika.exceptions.AMQPConnectionError as e:
        print(f"❌ Failed to connect to RabbitMQ: {e}")
        print("   Make sure RabbitMQ is running: docker compose up -d rabbitmq")
        return 1
    except Exception as e:
        print(f"❌ Error sending signal: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(send_signal())

