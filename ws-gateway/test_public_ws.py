"""Test public WebSocket connection to Bybit."""
import asyncio
import websockets

async def test_public_connection():
    """Test connection to Bybit public WebSocket endpoint."""
    url = "wss://stream-testnet.bybit.com/v5/public"
    print(f"Attempting to connect to: {url}")
    
    try:
        async with asyncio.wait_for(
            websockets.connect(url, ping_interval=None, ping_timeout=None),
            timeout=10.0
        ) as ws:
            print("✅ Successfully connected to public endpoint!")
            await asyncio.sleep(2)
            print("Connection is stable")
            return True
    except asyncio.TimeoutError:
        print("❌ Connection timeout after 10 seconds")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_public_connection())
    exit(0 if result else 1)

