#!/usr/bin/env python3
"""
Integration test for order creation and execution.

This test:
1. Sends test trading signals (buy and sell) to RabbitMQ queue
2. Verifies orders are created in the database
3. Optionally checks order execution and position updates within a timeout period
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import Optional, Dict, Any

import aio_pika
from aio_pika import Message

# Add src parent directory to path for package imports
import os
# Try different paths depending on where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_path = os.path.join(project_root, 'src')

# Check if we're running from /app (inside container) or from project root
if os.path.exists('/app/src'):
    # Inside container: add /app to path so we can import as src.config
    if '/app' not in sys.path:
        sys.path.insert(0, '/app')
elif os.path.exists(src_path):
    # From project root: add parent of src to path
    parent_path = os.path.dirname(src_path)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
else:
    # Fallback: try to find src relative to script
    src_path_alt = os.path.join(os.path.dirname(script_dir), '..', 'src')
    if os.path.exists(src_path_alt):
        parent_path = os.path.dirname(os.path.abspath(src_path_alt))
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)

# Import as package (src.config, src.models, etc.)
from src.config.settings import settings
from src.config.database import DatabaseConnection
from src.config.logging import get_logger
from src.models.order import Order
from src.models.position import Position

logger = get_logger(__name__)


class OrderCreationTest:
    """Test class for order creation and execution."""

    def __init__(
        self,
        wait_for_execution: bool = True,
        execution_timeout_seconds: int = 60,
    ):
        """
        Initialize test.

        Args:
            wait_for_execution: Whether to wait for order execution and check positions
            execution_timeout_seconds: Maximum seconds to wait for order execution
        """
        self.wait_for_execution = wait_for_execution
        self.execution_timeout = execution_timeout_seconds
        self.sent_signals: Dict[str, Dict[str, Any]] = {}

    async def send_test_signal(
        self,
        signal_type: str,
        asset: str,
        amount: Decimal,
        price: Decimal,
    ) -> str:
        """
        Send a test trading signal to RabbitMQ.

        Args:
            signal_type: 'buy' or 'sell'
            asset: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            amount: Order amount in USDT
            price: Market price for the asset

        Returns:
            signal_id: UUID of the sent signal
        """
        signal_id = uuid4()
        current_time = datetime.utcnow()

        signal_data = {
            "signal_id": str(signal_id),
            "signal_type": signal_type.lower(),
            "asset": asset,
            "amount": str(amount),
            "confidence": str(Decimal("0.85")),
            "timestamp": current_time.isoformat() + "Z",
            "strategy_id": "test-strategy",
            "model_version": "test-v1.0",
            "is_warmup": False,
            "market_data_snapshot": {
                "price": str(price),
                "spread": str(Decimal("0.01")),
                "volume_24h": str(Decimal("1000000.0")),
                "volatility": str(Decimal("0.02")),
                "orderbook_depth": None,
                "technical_indicators": None,
            },
            "metadata": {
                "test": True,
                "source": "integration_test",
            },
            "trace_id": f"test-{signal_id}",
        }

        # Connect to RabbitMQ
        rabbitmq_url = (
            f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}"
            f"@{settings.rabbitmq_host}:{settings.rabbitmq_port}/"
        )

        connection = await aio_pika.connect_robust(rabbitmq_url)
        try:
            channel = await connection.channel()
            queue_name = "model-service.trading_signals"
            await channel.declare_queue(queue_name, durable=True)

            message = Message(
                json.dumps(signal_data).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            await channel.default_exchange.publish(message, routing_key=queue_name)

            # Store signal info for later verification
            self.sent_signals[str(signal_id)] = {
                "signal_type": signal_type,
                "asset": asset,
                "amount": amount,
                "price": price,
                "sent_at": current_time,
            }

            logger.info(
                "test_signal_sent",
                signal_id=str(signal_id),
                signal_type=signal_type,
                asset=asset,
                amount=float(amount),
                price=float(price),
            )

            return str(signal_id)
        finally:
            await connection.close()

    async def check_order_created(
        self,
        signal_id: str,
        timeout_seconds: int = 30,
    ) -> Optional[Order]:
        """
        Check if an order was created for the given signal_id.

        Args:
            signal_id: Signal UUID
            timeout_seconds: Maximum seconds to wait for order creation

        Returns:
            Order object if found, None otherwise
        """
        start_time = datetime.utcnow()
        pool = await DatabaseConnection.get_pool()

        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
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
                    # Convert to Order model
                    order = Order.from_dict(order_data)
                    logger.info(
                        "test_order_found",
                        signal_id=signal_id,
                        order_id=str(order.id),
                        bybit_order_id=order.order_id,
                        status=order.status,
                        asset=order.asset,
                        side=order.side,
                    )
                    return order

                # Wait a bit before retrying
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(
                    "test_order_check_failed",
                    signal_id=signal_id,
                    error=str(e),
                )
                await asyncio.sleep(1)

        logger.warning(
            "test_order_not_found",
            signal_id=signal_id,
            timeout=timeout_seconds,
        )
        return None

    async def check_order_execution(
        self,
        order: Order,
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Check if order was executed (status changed to filled/partially_filled).

        Args:
            order: Order object to check
            timeout_seconds: Maximum seconds to wait for execution

        Returns:
            Dictionary with execution status and details
        """
        start_time = datetime.utcnow()
        pool = await DatabaseConnection.get_pool()

        result = {
            "executed": False,
            "status": order.status,
            "filled_quantity": float(order.filled_quantity),
            "average_price": float(order.average_price) if order.average_price else None,
        }

        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
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
                        logger.info(
                            "test_order_executed",
                            order_id=str(order.id),
                            status=current_status,
                            filled_quantity=float(filled_qty),
                            average_price=float(avg_price) if avg_price else None,
                        )
                        return result

                await asyncio.sleep(2)
            except Exception as e:
                logger.error(
                    "test_order_execution_check_failed",
                    order_id=str(order.id),
                    error=str(e),
                )
                await asyncio.sleep(2)

        logger.warning(
            "test_order_not_executed",
            order_id=str(order.id),
            timeout=timeout_seconds,
            final_status=result["status"],
        )
        return result

    async def check_position_update(
        self,
        asset: str,
        timeout_seconds: int = 60,
    ) -> Optional[Position]:
        """
        Check if position was updated for the given asset.

        Args:
            asset: Trading pair (e.g., 'BTCUSDT')
            timeout_seconds: Maximum seconds to wait for position update

        Returns:
            Position object if found, None otherwise
        """
        start_time = datetime.utcnow()
        pool = await DatabaseConnection.get_pool()

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
                WHERE asset = $1
            """
            row = await pool.fetchrow(query, asset)
            if row:
                initial_position = Position.from_dict(dict(row))
                initial_timestamp = initial_position.last_updated
                logger.info(
                    "test_position_initial_state",
                    asset=asset,
                    initial_size=float(initial_position.size),
                    initial_last_updated=initial_timestamp.isoformat() if initial_timestamp else None,
                )
        except Exception as e:
            logger.debug(
                "test_position_initial_state_check_failed",
                asset=asset,
                error=str(e),
            )

        # Wait for position update
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                query = """
                    SELECT 
                        id, asset, size, average_entry_price, unrealized_pnl,
                        realized_pnl, mode, long_size, short_size, long_avg_price,
                        short_avg_price, last_updated, last_snapshot_at
                    FROM positions
                    WHERE asset = $1
                """
                row = await pool.fetchrow(query, asset)

                if row:
                    position = Position.from_dict(dict(row))
                    # Check if position was updated
                    if initial_timestamp is None or position.last_updated > initial_timestamp:
                        logger.info(
                            "test_position_updated",
                            asset=asset,
                            position_id=str(position.id),
                            size=float(position.size),
                            average_entry_price=float(position.average_entry_price) if position.average_entry_price else None,
                            last_updated=position.last_updated.isoformat(),
                        )
                        return position
                    # Also check if size changed even if timestamp is same (for immediate updates)
                    elif initial_position is not None and position.size != initial_position.size:
                        logger.info(
                            "test_position_updated_size_changed",
                            asset=asset,
                            position_id=str(position.id),
                            old_size=float(initial_position.size),
                            new_size=float(position.size),
                        )
                        return position

                await asyncio.sleep(2)
            except Exception as e:
                logger.error(
                    "test_position_check_failed",
                    asset=asset,
                    error=str(e),
                )
                await asyncio.sleep(2)

        logger.warning(
            "test_position_not_updated",
            asset=asset,
            timeout=timeout_seconds,
        )
        return None

    async def run_test(
        self,
        buy_asset: str = "ETHUSDT",
        buy_amount: Decimal = Decimal("30.0"),
        buy_price: Decimal = Decimal("3000.0"),
        sell_asset: str = "ETHUSDT",
        sell_amount: Decimal = Decimal("30.0"),  # Increased to ensure order is not rejected
        sell_price: Decimal = Decimal("3000.0"),
    ) -> Dict[str, Any]:
        """
        Run the full integration test.

        Args:
            buy_asset: Asset for buy signal
            buy_amount: Buy order amount in USDT
            buy_price: Buy order price
            sell_asset: Asset for sell signal
            sell_amount: Sell order amount in USDT
            sell_price: Sell order price

        Returns:
            Test results dictionary
        """
        results = {
            "buy_signal_sent": False,
            "buy_order_created": False,
            "buy_order": None,
            "buy_order_executed": False,
            "buy_execution_details": None,
            "buy_position_updated": False,
            "sell_signal_sent": False,
            "sell_order_created": False,
            "sell_order": None,
            "sell_order_executed": False,
            "sell_execution_details": None,
            "sell_position_updated": False,
            "errors": [],
        }

        try:
            # Initialize database connection
            await DatabaseConnection.get_pool()

            # Send BUY signal
            logger.info("test_starting_buy_signal")
            buy_signal_id = await self.send_test_signal(
                signal_type="buy",
                asset=buy_asset,
                amount=buy_amount,
                price=buy_price,
            )
            results["buy_signal_sent"] = True

            # Wait a bit for signal processing
            await asyncio.sleep(2)

            # Check if BUY order was created
            buy_order = await self.check_order_created(buy_signal_id, timeout_seconds=30)
            if buy_order:
                results["buy_order_created"] = True
                results["buy_order"] = {
                    "id": str(buy_order.id),
                    "bybit_order_id": buy_order.order_id,
                    "status": buy_order.status,
                    "asset": buy_order.asset,
                    "side": buy_order.side,
                    "order_type": buy_order.order_type,
                    "quantity": float(buy_order.quantity),
                    "price": float(buy_order.price) if buy_order.price else None,
                }
                
                # Verify order is not rejected
                if buy_order.status == "rejected":
                    error_msg = (
                        f"BUY order was rejected: {buy_order.rejection_reason or 'Unknown reason'}. "
                        f"Order ID: {buy_order.order_id}, Signal ID: {buy_signal_id}"
                    )
                    logger.error("test_buy_order_rejected", error=error_msg)
                    results["errors"].append(error_msg)

                # Optionally check execution (skip for rejected orders)
                if self.wait_for_execution and buy_order.status != "rejected":
                    execution_details = await self.check_order_execution(
                        buy_order,
                        timeout_seconds=self.execution_timeout,
                    )
                    results["buy_order_executed"] = execution_details["executed"]
                    results["buy_execution_details"] = execution_details

                    # Check position update (only for successful orders)
                    if execution_details["executed"]:
                        position = await self.check_position_update(
                            buy_asset,
                            timeout_seconds=min(30, self.execution_timeout),
                        )
                        if position:
                            results["buy_position_updated"] = True
                elif buy_order.status == "rejected":
                    # For rejected orders, skip execution check
                    results["buy_order_executed"] = False
                    results["buy_execution_details"] = {
                        "executed": False,
                        "status": "rejected",
                        "filled_quantity": 0.0,
                        "average_price": None,
                    }

            # Wait a bit before sending sell signal
            await asyncio.sleep(5)

            # Send SELL signal
            logger.info("test_starting_sell_signal")
            sell_signal_id = await self.send_test_signal(
                signal_type="sell",
                asset=sell_asset,
                amount=sell_amount,
                price=sell_price,
            )
            results["sell_signal_sent"] = True

            # Wait a bit for signal processing
            await asyncio.sleep(2)

            # Check if SELL order was created
            sell_order = await self.check_order_created(sell_signal_id, timeout_seconds=30)
            if sell_order:
                results["sell_order_created"] = True
                results["sell_order"] = {
                    "id": str(sell_order.id),
                    "bybit_order_id": sell_order.order_id,
                    "status": sell_order.status,
                    "asset": sell_order.asset,
                    "side": sell_order.side,
                    "order_type": sell_order.order_type,
                    "quantity": float(sell_order.quantity),
                    "price": float(sell_order.price) if sell_order.price else None,
                }
                
                # Verify order is not rejected
                if sell_order.status == "rejected":
                    error_msg = (
                        f"SELL order was rejected: {sell_order.rejection_reason or 'Unknown reason'}. "
                        f"Order ID: {sell_order.order_id}, Signal ID: {sell_signal_id}"
                    )
                    logger.error("test_sell_order_rejected", error=error_msg)
                    results["errors"].append(error_msg)

                # Optionally check execution (skip for rejected orders)
                if self.wait_for_execution and sell_order.status != "rejected":
                    execution_details = await self.check_order_execution(
                        sell_order,
                        timeout_seconds=self.execution_timeout,
                    )
                    results["sell_order_executed"] = execution_details["executed"]
                    results["sell_execution_details"] = execution_details

                    # Check position update (only for successful orders)
                    if execution_details["executed"]:
                        position = await self.check_position_update(
                            sell_asset,
                            timeout_seconds=min(30, self.execution_timeout),
                        )
                        if position:
                            results["sell_position_updated"] = True
                elif sell_order.status == "rejected":
                    # For rejected orders, skip execution check
                    results["sell_order_executed"] = False
                    results["sell_execution_details"] = {
                        "executed": False,
                        "status": "rejected",
                        "filled_quantity": 0.0,
                        "average_price": None,
                    }

        except Exception as e:
            error_msg = f"Test failed with error: {str(e)}"
            logger.error("test_failed", error=error_msg, exc_info=True)
            results["errors"].append(error_msg)

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print test results in a human-readable format."""
        print("\n" + "=" * 80)
        print("ORDER CREATION AND EXECUTION TEST RESULTS")
        print("=" * 80)

        # BUY order results
        print("\n📈 BUY ORDER:")
        print(f"  Signal sent: {'✅' if results['buy_signal_sent'] else '❌'}")
        print(f"  Order created: {'✅' if results['buy_order_created'] else '❌'}")
        if results["buy_order"]:
            order = results["buy_order"]
            print(f"    - Order ID: {order['id']}")
            print(f"    - Bybit Order ID: {order['bybit_order_id']}")
            print(f"    - Status: {order['status']}")
            print(f"    - Asset: {order['asset']}")
            print(f"    - Side: {order['side']}")
            print(f"    - Type: {order['order_type']}")
            print(f"    - Quantity: {order['quantity']}")
            print(f"    - Price: {order['price'] if order['price'] else 'N/A (Market)'}")
        if self.wait_for_execution:
            print(f"  Order executed: {'✅' if results['buy_order_executed'] else '⏳'}")
            if results["buy_execution_details"]:
                details = results["buy_execution_details"]
                print(f"    - Status: {details['status']}")
                print(f"    - Filled: {details['filled_quantity']}")
                print(f"    - Avg Price: {details['average_price'] or 'N/A'}")
            print(f"  Position updated: {'✅' if results['buy_position_updated'] else '⏳'}")

        # SELL order results
        print("\n📉 SELL ORDER:")
        print(f"  Signal sent: {'✅' if results['sell_signal_sent'] else '❌'}")
        print(f"  Order created: {'✅' if results['sell_order_created'] else '❌'}")
        if results["sell_order"]:
            order = results["sell_order"]
            print(f"    - Order ID: {order['id']}")
            print(f"    - Bybit Order ID: {order['bybit_order_id']}")
            print(f"    - Status: {order['status']}")
            print(f"    - Asset: {order['asset']}")
            print(f"    - Side: {order['side']}")
            print(f"    - Type: {order['order_type']}")
            print(f"    - Quantity: {order['quantity']}")
            print(f"    - Price: {order['price'] if order['price'] else 'N/A (Market)'}")
        if self.wait_for_execution:
            print(f"  Order executed: {'✅' if results['sell_order_executed'] else '⏳'}")
            if results["sell_execution_details"]:
                details = results["sell_execution_details"]
                print(f"    - Status: {details['status']}")
                print(f"    - Filled: {details['filled_quantity']}")
                print(f"    - Avg Price: {details['average_price'] or 'N/A'}")
            print(f"  Position updated: {'✅' if results['sell_position_updated'] else '⏳'}")

        # Errors
        if results["errors"]:
            print("\n❌ ERRORS:")
            for error in results["errors"]:
                print(f"  - {error}")

        # Summary
        print("\n" + "=" * 80)
        # Check if orders were rejected
        buy_rejected = results.get("buy_order") and results["buy_order"]["status"] == "rejected"
        sell_rejected = results.get("sell_order") and results["sell_order"]["status"] == "rejected"
        
        all_passed = (
            results["buy_signal_sent"]
            and results["buy_order_created"]
            and results["sell_signal_sent"]
            and results["sell_order_created"]
            and len(results["errors"]) == 0
            and not buy_rejected
            and not sell_rejected
        )
        print(f"SUMMARY: {'✅ ALL CHECKS PASSED' if all_passed else '❌ TEST FAILED'}")
        if not all_passed:
            print("\nFAILURE REASONS:")
            if buy_rejected:
                print(f"  - BUY order was rejected")
            if sell_rejected:
                print(f"  - SELL order was rejected")
            for error in results["errors"]:
                print(f"  - {error}")
        print("=" * 80 + "\n")


async def main():
    """Main test entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test order creation and execution")
    parser.add_argument(
        "--no-execution-check",
        action="store_true",
        help="Skip execution and position update checks",
    )
    parser.add_argument(
        "--execution-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for execution checks (default: 60)",
    )
    parser.add_argument(
        "--buy-asset",
        type=str,
        default="ETHUSDT",
        help="Asset for buy signal (default: ETHUSDT)",
    )
    parser.add_argument(
        "--buy-amount",
        type=float,
        default=30.0,
        help="Buy order amount in USDT (default: 30.0)",
    )
    parser.add_argument(
        "--buy-price",
        type=float,
        default=3000.0,
        help="Buy order price (default: 3000.0)",
    )
    parser.add_argument(
        "--sell-asset",
        type=str,
        default="ETHUSDT",
        help="Asset for sell signal (default: ETHUSDT)",
    )
    parser.add_argument(
        "--sell-amount",
        type=float,
        default=30.0,
        help="Sell order amount in USDT (default: 30.0)",
    )
    parser.add_argument(
        "--sell-price",
        type=float,
        default=3000.0,
        help="Sell order price (default: 3000.0)",
    )

    args = parser.parse_args()

    test = OrderCreationTest(
        wait_for_execution=not args.no_execution_check,
        execution_timeout_seconds=args.execution_timeout,
    )

    results = await test.run_test(
        buy_asset=args.buy_asset,
        buy_amount=Decimal(str(args.buy_amount)),
        buy_price=Decimal(str(args.buy_price)),
        sell_asset=args.sell_asset,
        sell_amount=Decimal(str(args.sell_amount)),
        sell_price=Decimal(str(args.sell_price)),
    )

    test.print_results(results)

    # Exit with error code if critical checks failed or orders were rejected
    if not (results["buy_order_created"] and results["sell_order_created"]):
        sys.exit(1)
    
    # Check if any orders were rejected
    if results.get("buy_order") and results["buy_order"]["status"] == "rejected":
        print("❌ TEST FAILED: BUY order was rejected")
        sys.exit(1)
    
    if results.get("sell_order") and results["sell_order"]["status"] == "rejected":
        print("❌ TEST FAILED: SELL order was rejected")
        sys.exit(1)
    
    # Check for other errors
    if results["errors"]:
        print(f"❌ TEST FAILED: {len(results['errors'])} error(s) occurred")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

