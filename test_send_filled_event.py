#!/usr/bin/env python3
"""
Script to simulate a filled order event by publishing directly to RabbitMQ queue.

This allows testing the execution_events creation flow without waiting for actual order execution.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import aio_pika


async def send_filled_event(order_id: str, signal_id: str):
    """
    Send a filled order event to RabbitMQ queue order-manager.order_events.
    
    Args:
        order_id: Bybit order ID (e.g., from orders table)
        signal_id: Signal ID associated with the order
    """
    connection = await aio_pika.connect_robust('amqp://guest:guest@rabbitmq:5672/')
    
    async with connection:
        channel = await connection.channel()
        
        # Declare queue to ensure it exists
        queue = await channel.declare_queue('order-manager.order_events', durable=True)
        
        # Get order details from database (we'll use hardcoded values for now)
        # In production, you'd query the database for actual order details
        
        # Create enriched event matching order-manager format
        filled_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "filled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": f"test-trace-{uuid.uuid4()}",
            "order": {
                "id": str(uuid.uuid4()),  # Internal order ID (UUID)
                "order_id": order_id,  # Bybit order ID
                "signal_id": signal_id,
                "asset": "BTCUSDT",
                "side": "Buy",
                "order_type": "Market",
                "quantity": "0.001",
                "price": None,  # Market order has no price
                "status": "filled",
                "filled_quantity": "0.001",
                "average_price": "341694.20000000",  # Example execution price
                "fees": "0.00034169",  # Example fees
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "is_dry_run": False,
            },
            "execution_details": {
                "execution_latency_seconds": 0.2,
                "fill_percentage": 100.0,
                "remaining_quantity": "0.00000000",
            },
            "market_conditions": {
                "spread": 0.0015,
                "volume_24h": 1000000.0,
                "volatility": 0.02,
            },
            "signal": {
                "signal_id": signal_id,
                "strategy_id": "test-strategy",
                "price": "97000.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        
        # Serialize to JSON
        event_json = json.dumps(filled_event, default=str)
        
        # Create message
        message = aio_pika.Message(
            event_json.encode("utf-8"),
            headers={"trace_id": filled_event["trace_id"]},
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        
        # Publish to queue
        await channel.default_exchange.publish(
            message,
            routing_key='order-manager.order_events',
        )
        
        print(f"✅ Filled event sent successfully!")
        print(f"   Order ID: {order_id}")
        print(f"   Signal ID: {signal_id}")
        print(f"   Event ID: {filled_event['event_id']}")
        print(f"   Queue: order-manager.order_events")
        
        return filled_event


async def send_filled_event_from_db(signal_id: str):
    """
    Send a filled event using actual order data from database.
    
    Args:
        signal_id: Signal ID to look up order for
    """
    import asyncpg
    
    # Connect to database using environment variables
    import os
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = int(os.getenv('POSTGRES_PORT', '5432'))
    db_user = os.getenv('POSTGRES_USER', 'ytrader')
    db_password = os.getenv('POSTGRES_PASSWORD', 'ytrader')
    db_name = os.getenv('POSTGRES_DB', 'ytrader')
    
    conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )
    
    try:
        # Get order details
        order = await conn.fetchrow(
            """
            SELECT 
                id, order_id, signal_id, asset, side, order_type,
                quantity, price, status, filled_quantity, average_price,
                fees, created_at, executed_at, is_dry_run
            FROM orders
            WHERE signal_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            signal_id,
        )
        
        if not order:
            print(f"❌ No order found for signal_id: {signal_id}")
            return None
        
        if order['status'] != 'filled':
            print(f"⚠️  Order status is '{order['status']}', not 'filled'")
            print(f"   Using order data anyway for testing...")
        
        # Calculate execution latency
        execution_latency = None
        if order['executed_at'] and order['created_at']:
            latency_delta = order['executed_at'] - order['created_at']
            execution_latency = latency_delta.total_seconds()
        
        # Calculate fill percentage
        fill_percentage = None
        if order['quantity'] and order['quantity'] > 0:
            fill_percentage = float(order['filled_quantity'] / order['quantity'] * 100)
        
        # Get signal info
        signal = await conn.fetchrow(
            """
            SELECT signal_id, strategy_id, price, timestamp
            FROM trading_signals
            WHERE signal_id = $1
            LIMIT 1
            """,
            signal_id,
        )
        
        # Create enriched event
        filled_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "filled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": f"test-trace-{uuid.uuid4()}",
            "order": {
                "id": str(order['id']),
                "order_id": order['order_id'],
                "signal_id": str(order['signal_id']),
                "asset": order['asset'],
                "side": order['side'],
                "order_type": order['order_type'],
                "quantity": str(order['quantity']),
                "price": str(order['price']) if order['price'] else None,
                "status": "filled",
                "filled_quantity": str(order['filled_quantity']),
                "average_price": str(order['average_price']) if order['average_price'] else None,
                "fees": str(order['fees']) if order['fees'] else "0.0",
                "created_at": order['created_at'].isoformat() if order['created_at'] else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "executed_at": order['executed_at'].isoformat() if order['executed_at'] else datetime.now(timezone.utc).isoformat(),
                "is_dry_run": order['is_dry_run'],
            },
            "execution_details": {
                "execution_latency_seconds": execution_latency,
                "fill_percentage": fill_percentage,
                "remaining_quantity": str(order['quantity'] - order['filled_quantity']),
            },
            "market_conditions": {
                "spread": 0.0015,
                "volume_24h": 1000000.0,
                "volatility": 0.02,
            },
        }
        
        # Add signal info if available
        if signal:
            filled_event["signal"] = {
                "signal_id": str(signal['signal_id']),
                "strategy_id": signal['strategy_id'],
                "price": str(signal['price']) if signal['price'] else None,
                "timestamp": signal['timestamp'].isoformat() if signal['timestamp'] else None,
            }
        
        # Publish to RabbitMQ
        connection_rmq = await aio_pika.connect_robust('amqp://guest:guest@rabbitmq:5672/')
        
        async with connection_rmq:
            channel = await connection_rmq.channel()
            queue = await channel.declare_queue('order-manager.order_events', durable=True)
            
            event_json = json.dumps(filled_event, default=str)
            
            message = aio_pika.Message(
                event_json.encode("utf-8"),
                headers={"trace_id": filled_event["trace_id"]},
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            
            await channel.default_exchange.publish(
                message,
                routing_key='order-manager.order_events',
            )
            
            print(f"✅ Filled event sent successfully!")
            print(f"   Order ID: {order['order_id']}")
            print(f"   Signal ID: {signal_id}")
            print(f"   Event ID: {filled_event['event_id']}")
            print(f"   Average Price: {order['average_price']}")
            print(f"   Filled Quantity: {order['filled_quantity']}")
            print(f"   Queue: order-manager.order_events")
            
            return filled_event
            
    finally:
        await conn.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 test_send_filled_event.py <signal_id>")
        print("  python3 test_send_filled_event.py <order_id> <signal_id>")
        print("\nExample:")
        print("  python3 test_send_filled_event.py 849ca68f-72f1-4ced-ae2f-05d625a27b72")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Use signal_id to look up order from database
        signal_id = sys.argv[1]
        asyncio.run(send_filled_event_from_db(signal_id))
    else:
        # Use provided order_id and signal_id
        order_id = sys.argv[1]
        signal_id = sys.argv[2]
        asyncio.run(send_filled_event(order_id, signal_id))

