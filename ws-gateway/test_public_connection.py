"""Test public WebSocket connection to Bybit."""
import asyncio
import websockets

async def test_public():
    # Test with correct URL format for unified trading API v5
    url = "wss://stream-testnet.bybit.com/v5/public/linear"
    print(f"Testing connection to: {url}")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(url, ping_interval=None, ping_timeout=None),
            timeout=10.0
        )
        print("✅ Connected successfully!")
        await asyncio.sleep(2)
        await ws.close()
        return True
    except asyncio.TimeoutError:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_public())
    exit(0 if result else 1)

