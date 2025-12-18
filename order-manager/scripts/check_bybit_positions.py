#!/usr/bin/env python3
"""Script to check positions from Bybit API and compare with local database."""

import asyncio
import sys
import os

# Add src to path - need to add parent directory first
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from utils.bybit_client import get_bybit_client
from config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def get_bybit_positions():
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
            print(f"Error: {error_msg}")
            return []
        
        positions = response.get("result", {}).get("list", [])
        open_positions = [
            p for p in positions 
            if p.get("size") and float(p.get("size", "0")) != 0
        ]
        
        logger.info("Retrieved open positions", count=len(open_positions))
        return open_positions
        
    except Exception as e:
        logger.error("Error getting positions", error=str(e), exc_info=True)
        print(f"Error: {e}")
        return []


async def main():
    """Main function."""
    print("=" * 60)
    print("Checking positions from Bybit API...")
    print("=" * 60)
    
    positions = await get_bybit_positions()
    
    if not positions:
        print("\nNo open positions found on Bybit")
        return
    
    print(f"\nFound {len(positions)} open positions on Bybit:")
    print("-" * 60)
    for pos in positions:
        symbol = pos.get("symbol", "N/A")
        size = pos.get("size", "0")
        side = pos.get("side", "N/A")
        avg_price = pos.get("avgPrice", "N/A")
        mark_price = pos.get("markPrice", "N/A")
        unrealised_pnl = pos.get("unrealisedPnl", "0")
        realised_pnl = pos.get("cumRealisedPnl", "0")
        
        print(f"\n{symbol}:")
        print(f"  Size: {size} ({side})")
        print(f"  Avg Price: {avg_price}")
        print(f"  Mark Price: {mark_price}")
        print(f"  Unrealised PnL: {unrealised_pnl}")
        print(f"  Realised PnL: {realised_pnl}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

