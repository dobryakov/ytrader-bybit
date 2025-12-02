#!/usr/bin/env python3
"""Simple script to cancel specific orders by order_id using direct Bybit API calls."""

import asyncio
import hmac
import hashlib
import json
import os
import time
from typing import Dict, Any

import httpx


def get_bybit_config():
    """Get Bybit API configuration from environment."""
    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")
    base_url = os.environ.get("BYBIT_API_BASE_URL", "https://api-testnet.bybit.com")
    
    if not api_key or not api_secret:
        raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set")
    
    return api_key, api_secret, base_url


def generate_signature(api_key: str, api_secret: str, timestamp: int, recv_window: str, json_body: str) -> str:
    """Generate HMAC-SHA256 signature for POST requests."""
    signature_string = f"{timestamp}{api_key}{recv_window}{json_body}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        signature_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


async def cancel_order(order_id: str, symbol: str, api_key: str, api_secret: str, base_url: str) -> bool:
    """Cancel a single order on Bybit."""
    try:
        # Prepare request data
        json_body_dict = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        json_body = json.dumps(json_body_dict, separators=(",", ":"))
        
        # Generate timestamp and signature
        timestamp = int(time.time() * 1000)
        recv_window = "5000"
        signature = generate_signature(api_key, api_secret, timestamp, recv_window, json_body)
        
        # Prepare headers
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        # Make request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v5/order/cancel",
                headers=headers,
                content=json_body,
                timeout=10.0
            )
            
            result = response.json()
            
            if result.get("retCode") == 0:
                print(f"✓ Order {order_id} ({symbol}) cancelled successfully")
                return True
            else:
                error_msg = result.get("retMsg", "Unknown error")
                print(f"✗ Failed to cancel order {order_id} ({symbol}): {error_msg}")
                return False
                
    except Exception as e:
        print(f"✗ Error cancelling order {order_id} ({symbol}): {e}")
        return False


async def main():
    """Main function to cancel specific orders."""
    
    # Get configuration
    try:
        api_key, api_secret, base_url = get_bybit_config()
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return 1
    
    # Orders to cancel: (order_id, asset)
    orders_to_cancel = [
        ("7295b440-6f2f-4d11-a551-72e2202f6473", "ETHUSDT"),
        ("5b3588f7-40ca-47fd-a466-d919c7d2d85c", "BTCUSDT"),
    ]
    
    print(f"Starting cancellation of {len(orders_to_cancel)} orders...")
    print(f"Bybit API URL: {base_url}\n")
    
    cancelled_count = 0
    for order_id, asset in orders_to_cancel:
        print(f"Cancelling order {order_id} for {asset}...")
        if await cancel_order(order_id, asset, api_key, api_secret, base_url):
            cancelled_count += 1
        await asyncio.sleep(0.5)  # Small delay between cancellations
    
    print(f"\n✓ Cancelled {cancelled_count}/{len(orders_to_cancel)} orders")
    return 0 if cancelled_count == len(orders_to_cancel) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

