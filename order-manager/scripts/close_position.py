#!/usr/bin/env python3
"""
Script to close position for a specific asset via Bybit API.
Usage: python3 scripts/close_position.py ETHUSDT [--partial <size>]
"""

import asyncio
import sys
import os
import argparse
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.bybit_client import get_bybit_client
from config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def get_position(symbol: str):
    """Get position for a specific symbol from Bybit."""
    try:
        client = get_bybit_client()
        
        response = await client.get(
            "/v5/position/list",
            params={"category": "linear", "symbol": symbol},
            authenticated=True
        )
        
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.error("Failed to get position", symbol=symbol, error=error_msg)
            return None
        
        positions = response.get("result", {}).get("list", [])
        if not positions:
            logger.warning("No position found", symbol=symbol)
            return None
        
        position = positions[0]
        size_str = position.get("size", "0")
        size = Decimal(str(size_str))
        
        if size == Decimal("0"):
            logger.warning("Position size is zero", symbol=symbol)
            return None
        
        return {
            "symbol": symbol,
            "size": size,
            "side": "Buy" if size > 0 else "Sell",
            "avg_price": position.get("avgPrice", "0"),
            "mark_price": position.get("markPrice", "0"),
        }
        
    except Exception as e:
        logger.error("Error getting position", symbol=symbol, error=str(e), exc_info=True)
        return None
    finally:
        await client.close()


async def close_position(symbol: str, size: Decimal, side: str):
    """Close a position by creating opposite order with reduceOnly."""
    try:
        client = get_bybit_client()
        
        # Determine opposite side
        opposite_side = "Sell" if side == "Buy" else "Buy"
        
        logger.info(
            "Closing position",
            symbol=symbol,
            size=str(size),
            current_side=side,
            order_side=opposite_side,
        )
        
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
            logger.error(
                "Failed to close position",
                symbol=symbol,
                size=str(size),
                side=side,
                error=error_msg,
                ret_code=response.get("retCode"),
            )
            return False
        
        order_id = response.get("result", {}).get("orderId")
        logger.info(
            "Position closed successfully",
            symbol=symbol,
            size=str(size),
            side=side,
            order_id=order_id,
        )
        return True
        
    except Exception as e:
        logger.error(
            "Error closing position",
            symbol=symbol,
            size=str(size),
            side=side,
            error=str(e),
            exc_info=True,
        )
        return False
    finally:
        await client.close()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Close position for a specific asset")
    parser.add_argument("symbol", help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument(
        "--partial",
        type=float,
        help="Close only partial size (e.g., 0.45 to close 0.45 ETH)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Close entire position (default)",
    )
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    
    logger.info("Starting position close", symbol=symbol)
    
    try:
        # Get current position
        position = await get_position(symbol)
        if not position:
            logger.error("No position to close", symbol=symbol)
            return 1
        
        current_size = position["size"]
        current_side = position["side"]
        
        logger.info(
            "Current position",
            symbol=symbol,
            size=str(current_size),
            side=current_side,
            avg_price=position["avg_price"],
            mark_price=position["mark_price"],
        )
        
        # Determine size to close
        if args.partial:
            close_size = Decimal(str(args.partial))
            if close_size > abs(current_size):
                logger.warning(
                    "Partial size exceeds position size, closing entire position",
                    requested=str(close_size),
                    current=str(abs(current_size)),
                )
                close_size = abs(current_size)
        else:
            # Close all
            close_size = abs(current_size)
        
        # Close position
        success = await close_position(symbol, close_size, current_side)
        
        if success:
            logger.info("Position close completed", symbol=symbol, closed_size=str(close_size))
            return 0
        else:
            logger.error("Position close failed", symbol=symbol)
            return 1
        
    except Exception as e:
        logger.error("Position close script failed", symbol=symbol, error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

