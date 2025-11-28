#!/usr/bin/env python3
"""
Script to cancel all open orders and close all positions via Bybit API.
Run from order-manager container: python3 scripts/cleanup_bybit.py
"""

import asyncio
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.bybit_client import get_bybit_client
from config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def get_all_open_orders():
    """Get all open orders from Bybit."""
    try:
        client = get_bybit_client()
        
        response = await client.get(
            "/v5/order/realtime",
            params={"category": "linear", "settleCoin": "USDT"},
            authenticated=True
        )
        
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.error("Failed to get open orders", error=error_msg)
            return []
        
        orders = response.get("result", {}).get("list", [])
        active_orders = [
            o for o in orders 
            if o.get("orderStatus") in ["New", "PartiallyFilled"]
        ]
        
        logger.info("Retrieved open orders", count=len(active_orders))
        return active_orders
        
    except Exception as e:
        logger.error("Error getting open orders", error=str(e), exc_info=True)
        return []


async def cancel_order(order_id: str, symbol: str):
    """Cancel a single order."""
    try:
        client = get_bybit_client()
        
        response = await client.post(
            "/v5/order/cancel",
            json_data={
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id,
            },
            authenticated=True
        )
        
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.warning("Failed to cancel order", order_id=order_id, symbol=symbol, error=error_msg)
            return False
        
        logger.info("Order cancelled", order_id=order_id, symbol=symbol)
        return True
        
    except Exception as e:
        logger.error("Error cancelling order", order_id=order_id, symbol=symbol, error=str(e), exc_info=True)
        return False


async def get_all_positions():
    """Get all open positions from Bybit."""
    try:
        client = get_bybit_client()
        
        response = await client.get(
            "/v5/position/list",
            params={"category": "linear", "settleCoin": "USDT"},
            authenticated=True
        )
        
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.error("Failed to get positions", error=error_msg)
            return []
        
        positions = response.get("result", {}).get("list", [])
        open_positions = [
            p for p in positions 
            if p.get("size") and Decimal(str(p.get("size", "0"))) != Decimal("0")
        ]
        
        logger.info("Retrieved open positions", count=len(open_positions))
        return open_positions
        
    except Exception as e:
        logger.error("Error getting positions", error=str(e), exc_info=True)
        return []


async def close_position(symbol: str, size: Decimal, side: str):
    """Close a position by creating opposite order with reduceOnly."""
    try:
        client = get_bybit_client()
        
        # Determine opposite side
        opposite_side = "Sell" if side == "Buy" else "Buy"
        
        response = await client.post(
            "/v5/order/create",
            json_data={
                "category": "linear",
                "symbol": symbol,
                "side": opposite_side,
                "orderType": "Market",
                "qty": str(abs(size)),
                "reduceOnly": True,
            },
            authenticated=True
        )
        
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.warning("Failed to close position", symbol=symbol, size=str(size), side=side, error=error_msg)
            return False
        
        logger.info("Position closed", symbol=symbol, size=str(size), side=side)
        return True
        
    except Exception as e:
        logger.error("Error closing position", symbol=symbol, size=str(size), side=side, error=str(e), exc_info=True)
        return False


async def main():
    """Main cleanup function."""
    logger.info("Starting Bybit cleanup: cancel orders and close positions")
    
    try:
        # Step 1: Cancel all open orders
        logger.info("Step 1: Cancelling all open orders")
        orders = await get_all_open_orders()
        cancelled_count = 0
        for order in orders:
            order_id = order.get("orderId")
            symbol = order.get("symbol")
            if order_id and symbol:
                if await cancel_order(order_id, symbol):
                    cancelled_count += 1
                await asyncio.sleep(0.1)
        logger.info("Cancelled orders", count=cancelled_count, total=len(orders))
        
        # Step 2: Close all positions
        await asyncio.sleep(1)
        logger.info("Step 2: Closing all positions")
        positions = await get_all_positions()
        closed_count = 0
        for pos in positions:
            symbol = pos.get("symbol")
            size_str = pos.get("size", "0")
            if symbol and size_str:
                try:
                    size = Decimal(str(size_str))
                    if size != Decimal("0"):
                        side = "Buy" if size > 0 else "Sell"
                        if await close_position(symbol, abs(size), side):
                            closed_count += 1
                        await asyncio.sleep(0.2)
                except Exception as e:
                    logger.warning("Invalid position data", symbol=symbol, size=size_str, error=str(e))
        logger.info("Closed positions", count=closed_count, total=len(positions))
        
        logger.info("Cleanup completed", cancelled_orders=cancelled_count, closed_positions=closed_count)
        return 0
        
    except Exception as e:
        logger.error("Cleanup failed", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)