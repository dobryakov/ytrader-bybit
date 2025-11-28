#!/usr/bin/env python3
"""Cleanup script: cancel all orders and close all positions via Bybit API."""

import asyncio
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now import modules using absolute imports
from utils.bybit_client import get_bybit_client
from config.logging import setup_logging, get_logger
from decimal import Decimal

setup_logging()
logger = get_logger(__name__)


async def cleanup_all():
    """Cancel all open orders and close all positions."""
    client = get_bybit_client()
    
    print("=== Step 1: Cancelling all open orders ===\n")
    
    # Get open orders
    response = await client.get(
        "/v5/order/realtime",
        params={"category": "linear", "settleCoin": "USDT"},
        authenticated=True
    )
    
    if response.get("retCode") != 0:
        print(f"Error getting orders: {response.get('retMsg')}")
        return
    
    orders = response.get("result", {}).get("list", [])
    active_orders = [
        o for o in orders 
        if o.get("orderStatus") in ["New", "PartiallyFilled"]
    ]
    
    print(f"Found {len(active_orders)} open orders\n")
    
    cancelled = 0
    for order in active_orders:
        order_id = order.get("orderId")
        symbol = order.get("symbol")
        print(f"Cancelling order {order_id} for {symbol}...")
        
        resp = await client.post(
            "/v5/order/cancel",
            json_data={
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id,
            },
            authenticated=True
        )
        
        if resp.get("retCode") == 0:
            print(f"  ✓ Cancelled successfully\n")
            cancelled += 1
        else:
            print(f"  ✗ Error: {resp.get('retMsg')}\n")
        
        await asyncio.sleep(0.1)
    
    print(f"Cancelled {cancelled}/{len(active_orders)} orders\n")
    
    print("=== Step 2: Closing all positions ===\n")
    
    # Get positions
    response = await client.get(
        "/v5/position/list",
        params={"category": "linear", "settleCoin": "USDT"},
        authenticated=True
    )
    
    if response.get("retCode") != 0:
        print(f"Error getting positions: {response.get('retMsg')}")
        return
    
    positions = response.get("result", {}).get("list", [])
    open_positions = [
        p for p in positions 
        if p.get("size") and Decimal(str(p.get("size", "0"))) != Decimal("0")
    ]
    
    print(f"Found {len(open_positions)} open positions\n")
    
    closed = 0
    for pos in open_positions:
        symbol = pos.get("symbol")
        size = Decimal(str(pos.get("size", "0")))
        
        # Determine side: positive size = long (Buy), negative = short (Sell)
        side = "Sell" if size > 0 else "Buy"
        qty = str(abs(size))
        
        print(f"Closing {symbol}: size={size}, side={side}...")
        
        resp = await client.post(
            "/v5/order/create",
            json_data={
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty,
                "reduceOnly": True,
            },
            authenticated=True
        )
        
        if resp.get("retCode") == 0:
            print(f"  ✓ Closed successfully\n")
            closed += 1
        else:
            print(f"  ✗ Error: {resp.get('retMsg')}\n")
        
        await asyncio.sleep(0.2)
    
    print(f"Closed {closed}/{len(open_positions)} positions\n")
    print("=== Cleanup completed ===\n")


if __name__ == "__main__":
    try:
        asyncio.run(cleanup_all())
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
