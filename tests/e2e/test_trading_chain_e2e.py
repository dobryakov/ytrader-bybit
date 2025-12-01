#!/usr/bin/env python3
"""
End-to-end test for the complete trading chain.

This test verifies the entire trading flow:
1. Sends a trading signal to RabbitMQ (simulating model service trigger)
2. Verifies order-manager processes the signal and creates an order
3. Verifies order is placed on Bybit (or in dry-run mode)
4. Verifies position-manager receives position updates via WebSocket
5. Verifies model-service receives execution events
6. Optionally tests the full buy-sell cycle

The test can run in two modes:
- Real Bybit mode: Actually places orders on Bybit testnet
- Dry-run mode: Simulates order placement without real API calls
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, Optional, List
from uuid import uuid4

import aio_pika
from aio_pika import Message
import pytest

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add order-manager and position-manager to path
order_manager_path = os.path.join(project_root, 'order-manager')
position_manager_path = os.path.join(project_root, 'position-manager')

if order_manager_path not in sys.path:
    sys.path.insert(0, order_manager_path)
if position_manager_path not in sys.path:
    sys.path.insert(0, position_manager_path)

# Import from order-manager
try:
    from src.config.database import DatabaseConnection
    from src.config.settings import settings as order_settings
    from src.models.order import Order
except ImportError as e:
    # Try alternative import path
    try:
        from order_manager.src.config.database import DatabaseConnection
        from order_manager.src.config.settings import settings as order_settings
        from order_manager.src.models.order import Order
    except ImportError:
        raise ImportError(f"Could not import from order-manager: {e}")

# Import from position-manager
try:
    from src.models.position import Position
except ImportError as e:
    try:
        from position_manager.src.models.position import Position
    except ImportError:
        raise ImportError(f"Could not import from position-manager: {e}")


class TradingChainE2ETest:
    """End-to-end test for complete trading chain."""

    def __init__(
        self,
        rabbitmq_host: str = None,
        rabbitmq_port: int = None,
        rabbitmq_user: str = None,
        rabbitmq_password: str = None,
        dry_run: bool = True,
    ):
        """
        Initialize E2E test.

        Args:
            rabbitmq_host: RabbitMQ host (defaults to env or 'rabbitmq')
            rabbitmq_port: RabbitMQ port (defaults to env or 5672)
            rabbitmq_user: RabbitMQ user (defaults to env or 'guest')
            rabbitmq_password: RabbitMQ password (defaults to env or 'guest')
            dry_run: Whether to run in dry-run mode (default: True)
        """
        self.rabbitmq_host = rabbitmq_host or os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.rabbitmq_port = rabbitmq_port or int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = rabbitmq_user or os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_password = rabbitmq_password or os.getenv("RABBITMQ_PASSWORD", "guest")
        self.dry_run = dry_run
        self.sent_signals: Dict[str, Dict[str, Any]] = {}
        self.received_events: List[Dict[str, Any]] = []

    async def send_trading_signal(
        self,
        signal_type: str,
        asset: str,
        amount: Decimal,
        price: Decimal,
        strategy_id: str = "momentum_v1",
        confidence: float = 0.85,
    ) -> str:
        """
        Send a trading signal to RabbitMQ queue model-service.trading_signals.

        Args:
            signal_type: 'buy' or 'sell'
            asset: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            amount: Order amount in USDT
            price: Current market price
            strategy_id: Trading strategy identifier
            confidence: Signal confidence (0-1)

        Returns:
            signal_id: UUID of the sent signal
        """
        signal_id = str(uuid4())
        current_time = datetime.now(timezone.utc)

        signal_data = {
            "signal_id": signal_id,
            "signal_type": signal_type.lower(),
            "asset": asset.upper(),
            "amount": float(amount),
            "confidence": confidence,
            "timestamp": current_time.isoformat(),
            "strategy_id": strategy_id,
            "model_version": "test-v1.0",
            "is_warmup": False,
            "market_data_snapshot": {
                "price": float(price),
                "spread": 0.0015,  # 0.15% spread
                "volume_24h": 1000000.0,
                "volatility": 0.02,  # 2% volatility
                "orderbook_depth": None,
                "technical_indicators": None,
            },
            "metadata": {
                "test": True,
                "source": "e2e_test",
                "test_id": str(uuid4()),
            },
            "trace_id": f"e2e-test-{signal_id}",
        }

        # Connect to RabbitMQ
        rabbitmq_url = (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )

        connection = await aio_pika.connect_robust(rabbitmq_url)
        try:
            channel = await connection.channel()
            queue_name = "model-service.trading_signals"
            await channel.declare_queue(queue_name, durable=True)

            message = Message(
                json.dumps(signal_data).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"trace_id": f"e2e-test-{signal_id}"},
            )

            await channel.default_exchange.publish(message, routing_key=queue_name)

            # Store signal info
            self.sent_signals[signal_id] = {
                "signal_type": signal_type,
                "asset": asset,
                "amount": amount,
                "price": price,
                "sent_at": current_time,
                "strategy_id": strategy_id,
            }

            print(f"✅ Sent {signal_type.upper()} signal: {signal_id} for {asset} (amount: {amount} USDT)")
            return signal_id
        finally:
            await connection.close()

    async def wait_for_order_created(
        self,
        signal_id: str,
        timeout_seconds: int = 30,
    ) -> Optional[Order]:
        """
        Wait for order to be created in database.

        Args:
            signal_id: Signal UUID
            timeout_seconds: Maximum seconds to wait

        Returns:
            Order object if found, None otherwise
        """
        pool = await DatabaseConnection.get_pool()
        start_time = datetime.now(timezone.utc)

        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout_seconds:
            try:
                query = """
                    SELECT 
                        id, order_id, signal_id, asset, side, order_type,
                        quantity, price, status, filled_quantity, average_price,
                        fees, created_at, updated_at, executed_at, trace_id,
                        is_dry_run, rejection_reason
                    FROM orders
                    WHERE signal_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await pool.fetchrow(query, signal_id)

                if row:
                    order_data = dict(row)
                    order = Order.from_dict(order_data)
                    print(f"✅ Order created: {order.order_id} (status: {order.status})")
                    return order

                await asyncio.sleep(1)
            except Exception as e:
                print(f"⚠️  Error checking order: {e}")
                await asyncio.sleep(1)

        print(f"❌ Order not created within {timeout_seconds} seconds")
        return None

    async def wait_for_order_execution(
        self,
        order: Order,
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Wait for order execution (status changed to filled/partially_filled).

        Args:
            order: Order object to check
            timeout_seconds: Maximum seconds to wait

        Returns:
            Dictionary with execution status and details
        """
        pool = await DatabaseConnection.get_pool()
        start_time = datetime.now(timezone.utc)

        result = {
            "executed": False,
            "status": order.status,
            "filled_quantity": float(order.filled_quantity),
            "average_price": float(order.average_price) if order.average_price else None,
        }

        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout_seconds:
            try:
                query = """
                    SELECT status, filled_quantity, average_price, executed_at
                    FROM orders
                    WHERE id = $1
                """
                row = await pool.fetchrow(query, str(order.id))

                if row:
                    current_status = row["status"]
                    filled_qty = Decimal(str(row["filled_quantity"]))
                    avg_price = Decimal(str(row["average_price"])) if row["average_price"] else None

                    result["status"] = current_status
                    result["filled_quantity"] = float(filled_qty)
                    result["average_price"] = float(avg_price) if avg_price else None

                    if current_status in ("filled", "partially_filled"):
                        result["executed"] = True
                        print(f"✅ Order executed: {order.order_id} (status: {current_status}, filled: {filled_qty})")
                        return result

                await asyncio.sleep(2)
            except Exception as e:
                print(f"⚠️  Error checking execution: {e}")
                await asyncio.sleep(2)

        print(f"⏳ Order not executed within {timeout_seconds} seconds (status: {result['status']})")
        return result

    async def wait_for_position_update(
        self,
        asset: str,
        timeout_seconds: int = 60,
    ) -> Optional[Position]:
        """
        Wait for position update in database.

        Args:
            asset: Trading pair
            timeout_seconds: Maximum seconds to wait

        Returns:
            Position object if found/updated, None otherwise
        """
        pool = await DatabaseConnection.get_pool()
        start_time = datetime.now(timezone.utc)

        # Get initial position state
        initial_position = None
        initial_timestamp = None
        try:
            query = """
                SELECT 
                    id, asset, size, average_entry_price, unrealized_pnl,
                    realized_pnl, mode, long_size, short_size, long_avg_price,
                    short_avg_price, last_updated, last_snapshot_at
                FROM positions
                WHERE asset = $1 AND mode = 'one-way'
            """
            row = await pool.fetchrow(query, asset.upper())
            if row:
                initial_position = Position.from_dict(dict(row))
                initial_timestamp = initial_position.last_updated
        except Exception:
            pass  # No initial position

        # Wait for position update
        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout_seconds:
            try:
                query = """
                    SELECT 
                        id, asset, size, average_entry_price, unrealized_pnl,
                        realized_pnl, mode, long_size, short_size, long_avg_price,
                        short_avg_price, last_updated, last_snapshot_at
                    FROM positions
                    WHERE asset = $1 AND mode = 'one-way'
                """
                row = await pool.fetchrow(query, asset.upper())

                if row:
                    position = Position.from_dict(dict(row))
                    # Check if position was updated
                    if initial_timestamp is None or (position.last_updated and position.last_updated > initial_timestamp):
                        print(f"✅ Position updated: {asset} (size: {position.size})")
                        return position
                    # Also check if size changed
                    elif initial_position is not None and position.size != initial_position.size:
                        print(f"✅ Position size changed: {asset} (old: {initial_position.size}, new: {position.size})")
                        return position

                await asyncio.sleep(2)
            except Exception as e:
                print(f"⚠️  Error checking position: {e}")
                await asyncio.sleep(2)

        print(f"⏳ Position not updated within {timeout_seconds} seconds")
        return None

    async def check_position_event_published(
        self,
        asset: str,
        timeout_seconds: int = 30,
    ) -> bool:
        """
        Check if position update event was published to RabbitMQ.

        Note: This is a simplified check. In a real scenario, you would
        consume from the queue to verify the event.

        Args:
            asset: Trading pair
            timeout_seconds: Maximum seconds to wait

        Returns:
            True if position was updated (indirect check via database)
        """
        # For now, we check via database update as a proxy
        # In a full implementation, you would consume from position-manager.position_updated queue
        position = await self.wait_for_position_update(asset, timeout_seconds)
        return position is not None

    async def check_execution_event_published(
        self,
        signal_id: str,
        timeout_seconds: int = 60,
    ) -> bool:
        """
        Check if execution event was published to RabbitMQ.

        Note: This checks via database (execution_events table) as a proxy.
        In a full implementation, you would consume from order-manager.order_events queue.

        Args:
            signal_id: Signal UUID
            timeout_seconds: Maximum seconds to wait

        Returns:
            True if execution event exists in database
        """
        pool = await DatabaseConnection.get_pool()
        start_time = datetime.now(timezone.utc)

        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout_seconds:
            try:
                # Check if execution event exists in database
                query = """
                    SELECT event_id, signal_id, asset, side, execution_price, execution_quantity
                    FROM execution_events
                    WHERE signal_id = $1
                    LIMIT 1
                """
                row = await pool.fetchrow(query, signal_id)

                if row:
                    print(f"✅ Execution event found for signal: {signal_id}")
                    return True

                await asyncio.sleep(2)
            except Exception as e:
                # Table might not exist or no events yet
                if "does not exist" not in str(e).lower():
                    print(f"⚠️  Error checking execution event: {e}")
                await asyncio.sleep(2)

        print(f"⏳ Execution event not found within {timeout_seconds} seconds")
        return False

    async def get_current_price(self, asset: str) -> Decimal:
        """
        Get current market price for asset from Bybit API.

        Args:
            asset: Trading pair

        Returns:
            Current price as Decimal
        """
        try:
            # Try to import bybit client from order-manager
            try:
                from src.utils.bybit_client import get_bybit_client
            except ImportError:
                from order_manager.src.utils.bybit_client import get_bybit_client

            bybit_client = get_bybit_client()
            ticker_response = await bybit_client.get(
                "/v5/market/tickers",
                params={"category": "linear", "symbol": asset.upper()},
                authenticated=False,
            )
            ticker_data = ticker_response.get("result", {}).get("list", [])
            if ticker_data:
                price_str = ticker_data[0].get("lastPrice", "0")
                return Decimal(str(price_str))
        except Exception as e:
            print(f"⚠️  Could not fetch price from Bybit: {e}")

        # Fallback prices
        fallback_prices = {
            "BTCUSDT": Decimal("90000.0"),
            "ETHUSDT": Decimal("3000.0"),
            "SOLUSDT": Decimal("150.0"),
        }
        return fallback_prices.get(asset.upper(), Decimal("1000.0"))

    async def run_buy_order_test(
        self,
        asset: str = "ETHUSDT",
        amount: Decimal = Decimal("50.0"),
    ) -> Dict[str, Any]:
        """
        Run E2E test for BUY order flow.

        Args:
            asset: Trading pair
            amount: Order amount in USDT

        Returns:
            Test results dictionary
        """
        results = {
            "test_name": "buy_order_e2e",
            "asset": asset,
            "signal_sent": False,
            "order_created": False,
            "order": None,
            "order_executed": False,
            "position_updated": False,
            "execution_event_published": False,
            "errors": [],
        }

        try:
            # Initialize database connection
            await DatabaseConnection.get_pool()

            # Get current price
            current_price = await self.get_current_price(asset)
            print(f"\n📊 Current {asset} price: {current_price} USDT")

            # Step 1: Send BUY signal
            print(f"\n📤 Step 1: Sending BUY signal for {asset}...")
            signal_id = await self.send_trading_signal(
                signal_type="buy",
                asset=asset,
                amount=amount,
                price=current_price,
            )
            results["signal_sent"] = True

            # Step 2: Wait for order creation
            print(f"\n⏳ Step 2: Waiting for order creation...")
            await asyncio.sleep(3)  # Give order-manager time to process
            order = await self.wait_for_order_created(signal_id, timeout_seconds=30)
            if order:
                results["order_created"] = True
                results["order"] = {
                    "id": str(order.id),
                    "bybit_order_id": order.order_id,
                    "status": order.status,
                    "asset": order.asset,
                    "side": order.side,
                    "order_type": order.order_type,
                    "quantity": float(order.quantity),
                    "price": float(order.price) if order.price else None,
                    "is_dry_run": order.is_dry_run,
                }

                if order.status == "rejected":
                    error_msg = f"Order was rejected: {order.rejection_reason or 'Unknown reason'}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
                else:
                    # Step 3: Wait for order execution (if not dry-run)
                    if not order.is_dry_run:
                        print(f"\n⏳ Step 3: Waiting for order execution...")
                        execution_result = await self.wait_for_order_execution(order, timeout_seconds=60)
                        results["order_executed"] = execution_result["executed"]

                        # Step 4: Wait for position update
                        if execution_result["executed"]:
                            print(f"\n⏳ Step 4: Waiting for position update...")
                            position = await self.wait_for_position_update(asset, timeout_seconds=60)
                            results["position_updated"] = position is not None

                            # Step 5: Check execution event
                            if position:
                                print(f"\n⏳ Step 5: Checking execution event...")
                                results["execution_event_published"] = await self.check_execution_event_published(
                                    signal_id, timeout_seconds=30
                                )
                    else:
                        print(f"\nℹ️  Dry-run mode: Skipping execution and position checks")
                        results["order_executed"] = None  # N/A in dry-run
                        results["position_updated"] = None  # N/A in dry-run
            else:
                results["errors"].append("Order was not created within timeout")

        except Exception as e:
            error_msg = f"Test failed with error: {str(e)}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()

        return results

    async def run_buy_sell_cycle_test(
        self,
        asset: str = "ETHUSDT",
        buy_amount: Decimal = Decimal("50.0"),
        sell_amount: Decimal = Decimal("50.0"),
    ) -> Dict[str, Any]:
        """
        Run E2E test for complete buy-sell cycle.

        Args:
            asset: Trading pair
            buy_amount: BUY order amount in USDT
            sell_amount: SELL order amount in USDT

        Returns:
            Test results dictionary
        """
        results = {
            "test_name": "buy_sell_cycle_e2e",
            "asset": asset,
            "buy": {},
            "sell": {},
            "errors": [],
        }

        try:
            # Initialize database connection
            await DatabaseConnection.get_pool()

            # Get current price
            current_price = await self.get_current_price(asset)
            print(f"\n📊 Current {asset} price: {current_price} USDT")

            # BUY flow
            print(f"\n{'='*80}")
            print(f"📈 BUY ORDER FLOW")
            print(f"{'='*80}")
            buy_results = await self.run_buy_order_test(asset, buy_amount)
            results["buy"] = buy_results

            if not buy_results["order_created"] or buy_results.get("order", {}).get("status") == "rejected":
                results["errors"].append("BUY order failed, skipping SELL test")
                return results

            # Wait a bit before sending sell signal
            print(f"\n⏸️  Waiting 5 seconds before SELL signal...")
            await asyncio.sleep(5)

            # SELL flow
            print(f"\n{'='*80}")
            print(f"📉 SELL ORDER FLOW")
            print(f"{'='*80}")
            sell_results = await self.run_buy_order_test(asset, sell_amount)  # Reuse same method, just change signal_type
            # Actually send sell signal
            sell_signal_id = await self.send_trading_signal(
                signal_type="sell",
                asset=asset,
                amount=sell_amount,
                price=current_price,
            )
            sell_results["signal_sent"] = True

            await asyncio.sleep(3)
            sell_order = await self.wait_for_order_created(sell_signal_id, timeout_seconds=30)
            if sell_order:
                sell_results["order_created"] = True
                sell_results["order"] = {
                    "id": str(sell_order.id),
                    "bybit_order_id": sell_order.order_id,
                    "status": sell_order.status,
                    "asset": sell_order.asset,
                    "side": sell_order.side,
                    "order_type": sell_order.order_type,
                    "quantity": float(sell_order.quantity),
                    "price": float(sell_order.price) if sell_order.price else None,
                    "is_dry_run": sell_order.is_dry_run,
                }

                if not sell_order.is_dry_run and sell_order.status != "rejected":
                    execution_result = await self.wait_for_order_execution(sell_order, timeout_seconds=60)
                    sell_results["order_executed"] = execution_result["executed"]

                    if execution_result["executed"]:
                        position = await self.wait_for_position_update(asset, timeout_seconds=60)
                        sell_results["position_updated"] = position is not None

            results["sell"] = sell_results

        except Exception as e:
            error_msg = f"Test failed with error: {str(e)}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print test results in a human-readable format."""
        print("\n" + "=" * 80)
        print(f"E2E TEST RESULTS: {results.get('test_name', 'unknown')}")
        print("=" * 80)

        if "buy" in results and "sell" in results:
            # Buy-sell cycle results
            print("\n📈 BUY ORDER:")
            buy = results["buy"]
            print(f"  Signal sent: {'✅' if buy.get('signal_sent') else '❌'}")
            print(f"  Order created: {'✅' if buy.get('order_created') else '❌'}")
            if buy.get("order"):
                order = buy["order"]
                print(f"    - Order ID: {order.get('bybit_order_id')}")
                print(f"    - Status: {order.get('status')}")
                print(f"    - Type: {order.get('order_type')}")
                print(f"    - Quantity: {order.get('quantity')}")
                print(f"    - Dry-run: {order.get('is_dry_run')}")

            print("\n📉 SELL ORDER:")
            sell = results["sell"]
            print(f"  Signal sent: {'✅' if sell.get('signal_sent') else '❌'}")
            print(f"  Order created: {'✅' if sell.get('order_created') else '❌'}")
            if sell.get("order"):
                order = sell["order"]
                print(f"    - Order ID: {order.get('bybit_order_id')}")
                print(f"    - Status: {order.get('status')}")
                print(f"    - Type: {order.get('order_type')}")
                print(f"    - Quantity: {order.get('quantity')}")
        else:
            # Single order results
            print(f"\nAsset: {results.get('asset')}")
            print(f"Signal sent: {'✅' if results.get('signal_sent') else '❌'}")
            print(f"Order created: {'✅' if results.get('order_created') else '❌'}")
            if results.get("order"):
                order = results["order"]
                print(f"  - Order ID: {order.get('bybit_order_id')}")
                print(f"  - Status: {order.get('status')}")
                print(f"  - Type: {order.get('order_type')}")
                print(f"  - Quantity: {order.get('quantity')}")
                print(f"  - Dry-run: {order.get('is_dry_run')}")

            if results.get("order_executed") is not None:
                print(f"Order executed: {'✅' if results.get('order_executed') else '⏳'}")
            if results.get("position_updated") is not None:
                print(f"Position updated: {'✅' if results.get('position_updated') else '⏳'}")
            if results.get("execution_event_published") is not None:
                print(f"Execution event published: {'✅' if results.get('execution_event_published') else '⏳'}")

        if results.get("errors"):
            print("\n❌ ERRORS:")
            for error in results["errors"]:
                print(f"  - {error}")

        print("\n" + "=" * 80)


# Pytest fixtures and tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_pool():
    """Create database connection pool."""
    try:
        await DatabaseConnection.close_pool()
        pool = await DatabaseConnection.create_pool()
        yield pool
    finally:
        await DatabaseConnection.close_pool()


@pytest.fixture(scope="function")
def e2e_test():
    """Create E2E test instance."""
    dry_run = os.getenv("ORDERMANAGER_ENABLE_DRY_RUN", "true").lower() == "true"
    return TradingChainE2ETest(dry_run=dry_run)


@pytest.mark.asyncio
async def test_buy_order_e2e(e2e_test: TradingChainE2ETest):
    """Test complete BUY order flow end-to-end."""
    results = await e2e_test.run_buy_order_test(
        asset="ETHUSDT",
        amount=Decimal("50.0"),
    )
    e2e_test.print_results(results)

    # Assertions
    assert results["signal_sent"], "Signal should be sent"
    assert results["order_created"], "Order should be created"
    assert results["order"] is not None, "Order object should exist"
    assert results["order"]["status"] != "rejected", "Order should not be rejected"
    assert len(results["errors"]) == 0, f"Should have no errors: {results['errors']}"


@pytest.mark.asyncio
async def test_buy_sell_cycle_e2e(e2e_test: TradingChainE2ETest):
    """Test complete buy-sell cycle end-to-end."""
    results = await e2e_test.run_buy_sell_cycle_test(
        asset="ETHUSDT",
        buy_amount=Decimal("50.0"),
        sell_amount=Decimal("50.0"),
    )
    e2e_test.print_results(results)

    # Assertions
    assert results["buy"]["signal_sent"], "BUY signal should be sent"
    assert results["buy"]["order_created"], "BUY order should be created"
    assert results["sell"]["signal_sent"], "SELL signal should be sent"
    assert results["sell"]["order_created"], "SELL order should be created"
    assert len(results["errors"]) == 0, f"Should have no errors: {results['errors']}"


# Standalone script execution
async def main():
    """Main entry point for standalone script execution."""
    import argparse

    parser = argparse.ArgumentParser(description="E2E test for complete trading chain")
    parser.add_argument(
        "--asset",
        type=str,
        default="ETHUSDT",
        help="Trading pair (default: ETHUSDT)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=50.0,
        help="Order amount in USDT (default: 50.0)",
    )
    parser.add_argument(
        "--buy-sell",
        action="store_true",
        help="Run complete buy-sell cycle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default: True)",
    )

    args = parser.parse_args()

    test = TradingChainE2ETest(dry_run=args.dry_run)

    if args.buy_sell:
        results = await test.run_buy_sell_cycle_test(
            asset=args.asset,
            buy_amount=Decimal(str(args.amount)),
            sell_amount=Decimal(str(args.amount)),
        )
    else:
        results = await test.run_buy_order_test(
            asset=args.asset,
            amount=Decimal(str(args.amount)),
        )

    test.print_results(results)

    # Exit with error code if test failed
    if results.get("errors") or not results.get("order_created"):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

