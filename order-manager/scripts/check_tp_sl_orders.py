#!/usr/bin/env python3
"""Script to check TP/SL orders on Bybit."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.bybit_client import get_bybit_client
from src.config.logging import get_logger

logger = get_logger(__name__)


async def check_tp_sl_orders():
    """Check TP/SL orders and positions on Bybit."""
    client = get_bybit_client()
    
    # Check position
    print("\n=== Position Info ===")
    pos_resp = await client.get(
        '/v5/position/list',
        params={'category': 'linear', 'symbol': 'BTCUSDT'},
        authenticated=True
    )
    positions = pos_resp.get('result', {}).get('list', [])
    if positions:
        for pos in positions:
            print(f"Symbol: {pos.get('symbol')}")
            print(f"Size: {pos.get('size')}")
            print(f"Entry Price: {pos.get('avgPrice')}")
            print(f"Mark Price: {pos.get('markPrice')}")
            print(f"Unrealized PnL: {pos.get('unrealisedPnl')}")
            print(f"Take Profit: {pos.get('takeProfit')}")
            print(f"Stop Loss: {pos.get('stopLoss')}")
    else:
        print("No position found")
    
    # Check active orders (including conditional orders)
    print("\n=== Active Orders ===")
    orders_resp = await client.get(
        '/v5/order/realtime',
        params={'category': 'linear', 'settleCoin': 'USDT'},
        authenticated=True
    )
    orders = orders_resp.get('result', {}).get('list', [])
    active_orders = [o for o in orders if o.get('orderStatus') in ['New', 'PartiallyFilled']]
    
    print(f"Total active orders: {len(active_orders)}")
    for order in active_orders:
        if order.get('symbol') == 'BTCUSDT':
            print(f"\nOrder ID: {order.get('orderId')}")
            print(f"Symbol: {order.get('symbol')}")
            print(f"Side: {order.get('side')}")
            print(f"Order Type: {order.get('orderType')}")
            print(f"Status: {order.get('orderStatus')}")
            print(f"Price: {order.get('price')}")
            print(f"Quantity: {order.get('qty')}")
            print(f"Trigger Price: {order.get('triggerPrice')}")
            print(f"Is Conditional: {order.get('triggerBy')}")
            print(f"Take Profit: {order.get('takeProfit')}")
            print(f"Stop Loss: {order.get('stopLoss')}")


if __name__ == "__main__":
    asyncio.run(check_tp_sl_orders())

