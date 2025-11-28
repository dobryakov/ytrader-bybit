#!/usr/bin/env python3
"""Script to compare balances from database (WebSocket) with actual Bybit API balances.

This script:
1. Retrieves latest balances from account_balances table (from WebSocket events)
2. Fetches actual balances from Bybit REST API
3. Compares them and identifies discrepancies
4. Investigates potential causes of discrepancies
"""

import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.config.settings import settings
from src.services.database.connection import DatabaseConnection
from src.services.database.balance_repository import BalanceRepository
from src.config.logging import setup_logging, get_logger

import httpx
import hashlib
import hmac
import time

setup_logging()
logger = get_logger(__name__)


class SimpleBybitClient:
    """Simple Bybit REST API client for balance comparison script."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client
    
    def _generate_signature(self, params: Dict[str, Any], timestamp: int, recv_window: str = "5000") -> str:
        """Generate HMAC-SHA256 signature for Bybit REST API."""
        # Sort parameters alphabetically
        sorted_params = sorted([(k, str(v)) for k, v in params.items() if v is not None])
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Create signature string: timestamp + api_key + recv_window + query_string
        signature_string = f"{timestamp}{self.api_key}{recv_window}{query_string}"
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        
        return signature
    
    async def get_wallet_balance(self) -> Dict[str, Any]:
        """Get wallet balance from Bybit API."""
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED"}
        
        timestamp = int(time.time() * 1000)
        recv_window = "5000"
        signature = self._generate_signature(params, timestamp, recv_window)
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        
        client = await self._get_client()
        response = await client.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


async def get_db_balances() -> Dict[str, Dict]:
    """Get latest balances from database for each coin.
    
    Returns:
        Dictionary mapping coin -> {wallet_balance, available_balance, frozen, received_at, event_timestamp}
    """
    logger.info("fetching_db_balances")
    
    # Get all unique coins
    query = """
        SELECT DISTINCT coin
        FROM account_balances
        ORDER BY coin
    """
    rows = await DatabaseConnection.fetch(query)
    coins = [row["coin"] for row in rows]
    
    logger.info("found_coins_in_db", coins=coins, count=len(coins))
    
    # Get latest balance for each coin
    db_balances = {}
    for coin in coins:
        balance = await BalanceRepository.get_latest_balance(coin)
        if balance:
            db_balances[coin] = {
                "wallet_balance": balance.wallet_balance,
                "available_balance": balance.available_balance,
                "frozen": balance.frozen,
                "received_at": balance.received_at,
                "event_timestamp": balance.event_timestamp,
                "trace_id": balance.trace_id,
            }
    
    return db_balances


async def get_bybit_api_balances(bybit_client: SimpleBybitClient) -> Dict[str, Dict]:
    """Get actual balances from Bybit REST API.
    
    Args:
        bybit_client: Initialized Bybit client
        
    Returns:
        Dictionary mapping coin -> {wallet_balance, available_balance, frozen}
    """
    logger.info("fetching_bybit_api_balances")
    
    try:
        response = await bybit_client.get_wallet_balance()
        
        # Parse response structure
        # Bybit API v5 returns: { "retCode": 0, "retMsg": "OK", "result": { "list": [...] } }
        if response.get("retCode") != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.error("bybit_api_error", ret_code=response.get("retCode"), ret_msg=error_msg)
            return {}
        
        result = response.get("result", {})
        account_list = result.get("list", [])
        
        if not account_list:
            logger.warning("bybit_api_no_accounts")
            return {}
        
        # Unified account should have one account in list
        account = account_list[0]
        coins = account.get("coin", [])
        
        api_balances = {}
        for coin_data in coins:
            coin = coin_data.get("coin")
            if not coin:
                continue
            
            wallet_balance = coin_data.get("walletBalance", "0")
            locked = coin_data.get("locked", "0") or "0"
            
            # availableToWithdraw can be empty string
            available_balance = coin_data.get("availableToWithdraw")
            if available_balance == "" or available_balance is None:
                # Calculate: available = walletBalance - locked
                try:
                    wallet_bal = Decimal(str(wallet_balance)) if wallet_balance else Decimal(0)
                    locked_decimal = Decimal(str(locked)) if locked else Decimal(0)
                    available_balance = wallet_bal - locked_decimal
                except Exception:
                    available_balance = Decimal(0)
            else:
                available_balance = Decimal(str(available_balance))
            
            api_balances[coin] = {
                "wallet_balance": Decimal(str(wallet_balance)) if wallet_balance else Decimal(0),
                "available_balance": available_balance,
                "frozen": Decimal(str(locked)) if locked else Decimal(0),
            }
        
        logger.info("fetched_bybit_api_balances", coins=list(api_balances.keys()), count=len(api_balances))
        return api_balances
        
    except Exception as e:
        logger.error("bybit_api_fetch_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        return {}


def compare_balances(
    db_balances: Dict[str, Dict],
    api_balances: Dict[str, Dict],
    tolerance: Decimal = Decimal("0.00000001"),  # Very small tolerance for floating point
) -> List[Dict]:
    """Compare database balances with API balances and identify discrepancies.
    
    Args:
        db_balances: Balances from database
        api_balances: Balances from Bybit API
        tolerance: Tolerance for comparison (default: 0.00000001)
        
    Returns:
        List of discrepancies with details
    """
    discrepancies = []
    
    # Get all coins from both sources
    all_coins = set(db_balances.keys()) | set(api_balances.keys())
    
    for coin in all_coins:
        db_bal = db_balances.get(coin)
        api_bal = api_balances.get(coin)
        
        if not db_bal:
            discrepancies.append({
                "coin": coin,
                "issue": "missing_in_db",
                "api_wallet_balance": str(api_bal["wallet_balance"]) if api_bal else None,
                "api_available_balance": str(api_bal["available_balance"]) if api_bal else None,
                "api_frozen": str(api_bal["frozen"]) if api_bal else None,
            })
            continue
        
        if not api_bal:
            discrepancies.append({
                "coin": coin,
                "issue": "missing_in_api",
                "db_wallet_balance": str(db_bal["wallet_balance"]),
                "db_available_balance": str(db_bal["available_balance"]),
                "db_frozen": str(db_bal["frozen"]),
                "db_received_at": db_bal["received_at"].isoformat() if db_bal.get("received_at") else None,
            })
            continue
        
        # Compare values
        wallet_diff = abs(db_bal["wallet_balance"] - api_bal["wallet_balance"])
        available_diff = abs(db_bal["available_balance"] - api_bal["available_balance"])
        frozen_diff = abs(db_bal["frozen"] - api_bal["frozen"])
        
        if wallet_diff > tolerance or available_diff > tolerance or frozen_diff > tolerance:
            discrepancies.append({
                "coin": coin,
                "issue": "mismatch",
                "db_wallet_balance": str(db_bal["wallet_balance"]),
                "api_wallet_balance": str(api_bal["wallet_balance"]),
                "wallet_diff": str(wallet_diff),
                "db_available_balance": str(db_bal["available_balance"]),
                "api_available_balance": str(api_bal["available_balance"]),
                "available_diff": str(available_diff),
                "db_frozen": str(db_bal["frozen"]),
                "api_frozen": str(api_bal["frozen"]),
                "frozen_diff": str(frozen_diff),
                "db_received_at": db_bal["received_at"].isoformat() if db_bal.get("received_at") else None,
                "db_event_timestamp": db_bal["event_timestamp"].isoformat() if db_bal.get("event_timestamp") else None,
                "age_seconds": (datetime.now() - db_bal["received_at"]).total_seconds() if db_bal.get("received_at") else None,
            })
    
    return discrepancies


async def investigate_discrepancy(coin: str, db_balance: Dict, api_balance: Dict) -> Dict:
    """Investigate potential causes of balance discrepancy.
    
    Args:
        coin: Coin symbol
        db_balance: Balance from database
        api_balance: Balance from API
        
    Returns:
        Dictionary with investigation results
    """
    investigation = {
        "coin": coin,
        "potential_causes": [],
        "recommendations": [],
    }
    
    # Check age of database record
    if db_balance.get("received_at"):
        age_seconds = (datetime.now() - db_balance["received_at"]).total_seconds()
        age_minutes = age_seconds / 60
        
        if age_minutes > 5:
            investigation["potential_causes"].append(
                f"Database record is {age_minutes:.1f} minutes old - WebSocket may have missed recent updates"
            )
            investigation["recommendations"].append(
                "Check WebSocket connection status and subscription activity"
            )
        
        if age_minutes > 60:
            investigation["potential_causes"].append(
                f"Database record is very old ({age_minutes:.1f} minutes) - likely WebSocket disconnection"
            )
            investigation["recommendations"].append(
                "Verify WebSocket connection is active and wallet subscription is working"
            )
    
    # Check for missing events
    # Query recent balance updates for this coin
    query = """
        SELECT received_at, wallet_balance, available_balance, frozen
        FROM account_balances
        WHERE coin = $1
        ORDER BY received_at DESC
        LIMIT 10
    """
    recent_updates = await DatabaseConnection.fetch(query, coin)
    
    if len(recent_updates) < 2:
        investigation["potential_causes"].append(
            "Very few balance updates in database - WebSocket may not be receiving wallet events"
        )
        investigation["recommendations"].append(
            "Check wallet subscription status and verify wallet events are being received"
        )
    else:
        # Check update frequency
        if len(recent_updates) >= 2:
            latest_time = recent_updates[0]["received_at"]
            previous_time = recent_updates[1]["received_at"]
            time_diff = (latest_time - previous_time).total_seconds()
            
            if time_diff > 300:  # More than 5 minutes between updates
                investigation["potential_causes"].append(
                    f"Long gap between balance updates ({time_diff/60:.1f} minutes) - events may be missed"
                )
                investigation["recommendations"].append(
                    "Check WebSocket message processing and event parsing logic"
                )
    
    # Check if frozen balance calculation is correct
    db_wallet = db_balance["wallet_balance"]
    db_available = db_balance["available_balance"]
    db_frozen = db_balance["frozen"]
    
    calculated_frozen = db_wallet - db_available
    if abs(calculated_frozen - db_frozen) > Decimal("0.00000001"):
        investigation["potential_causes"].append(
            f"Frozen balance calculation mismatch: wallet ({db_wallet}) - available ({db_available}) = {calculated_frozen}, but stored frozen = {db_frozen}"
        )
        investigation["recommendations"].append(
            "Review balance parsing logic in balance_service.py - check how frozen/locked is calculated"
        )
    
    # Check if API balance structure matches what we parse
    api_wallet = api_balance["wallet_balance"]
    api_available = api_balance["available_balance"]
    api_frozen = api_balance["frozen"]
    
    api_calculated_frozen = api_wallet - api_available
    if abs(api_calculated_frozen - api_frozen) > Decimal("0.00000001"):
        investigation["potential_causes"].append(
            f"API frozen balance calculation mismatch: wallet ({api_wallet}) - available ({api_available}) = {api_calculated_frozen}, but API frozen = {api_frozen}"
        )
        investigation["recommendations"].append(
            "Verify Bybit API response structure - may need to use different field for frozen balance"
        )
    
    # Check for parsing issues
    wallet_diff = abs(db_wallet - api_wallet)
    if wallet_diff > Decimal("0.01"):  # Significant difference
        investigation["potential_causes"].append(
            f"Significant wallet balance difference ({wallet_diff}) - possible parsing error or missed transaction"
        )
        investigation["recommendations"].append(
            "Check WebSocket wallet message parsing - verify all fields are correctly extracted"
        )
    
    return investigation


async def main():
    """Main function to compare balances and investigate discrepancies."""
    logger.info("starting_balance_comparison")
    
    try:
        # Initialize database connection
        await DatabaseConnection.create_pool()
        logger.info("database_connected")
        
        # Initialize Bybit client
        bybit_base_url = (
            "https://api.bybit.com" if settings.bybit_environment == "mainnet"
            else "https://api-testnet.bybit.com"
        )
        
        bybit_client = SimpleBybitClient(
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            base_url=bybit_base_url,
        )
        logger.info("bybit_client_initialized", environment=settings.bybit_environment)
        
        # Get balances from both sources
        db_balances = await get_db_balances()
        api_balances = await get_bybit_api_balances(bybit_client)
        
        # Compare balances
        discrepancies = compare_balances(db_balances, api_balances)
        
        # Print results
        print("\n" + "=" * 80)
        print("BALANCE COMPARISON RESULTS")
        print("=" * 80)
        print(f"\nDatabase balances: {len(db_balances)} coins")
        print(f"API balances: {len(api_balances)} coins")
        print(f"Discrepancies found: {len(discrepancies)}\n")
        
        if not discrepancies:
            print("✅ No discrepancies found! All balances match.")
        else:
            print("⚠️  DISCREPANCIES DETECTED:\n")
            
            for disc in discrepancies:
                print(f"\n{'=' * 80}")
                print(f"Coin: {disc['coin']}")
                print(f"Issue: {disc['issue']}")
                print("-" * 80)
                
                if disc["issue"] == "mismatch":
                    print(f"Database Wallet Balance:  {disc.get('db_wallet_balance', 'N/A')}")
                    print(f"API Wallet Balance:       {disc.get('api_wallet_balance', 'N/A')}")
                    print(f"Difference:               {disc.get('wallet_diff', 'N/A')}")
                    print()
                    print(f"Database Available:       {disc.get('db_available_balance', 'N/A')}")
                    print(f"API Available:            {disc.get('api_available_balance', 'N/A')}")
                    print(f"Difference:               {disc.get('available_diff', 'N/A')}")
                    print()
                    print(f"Database Frozen:          {disc.get('db_frozen', 'N/A')}")
                    print(f"API Frozen:                {disc.get('api_frozen', 'N/A')}")
                    print(f"Difference:               {disc.get('frozen_diff', 'N/A')}")
                    print()
                    if disc.get("db_received_at"):
                        print(f"DB Record Age:            {disc.get('age_seconds', 0):.1f} seconds")
                    
                    # Investigate this discrepancy
                    coin = disc["coin"]
                    db_bal = db_balances.get(coin)
                    api_bal = api_balances.get(coin)
                    
                    if db_bal and api_bal:
                        print("\n" + "-" * 80)
                        print("INVESTIGATION:")
                        print("-" * 80)
                        investigation = await investigate_discrepancy(coin, db_bal, api_bal)
                        
                        if investigation["potential_causes"]:
                            print("\nPotential Causes:")
                            for i, cause in enumerate(investigation["potential_causes"], 1):
                                print(f"  {i}. {cause}")
                        
                        if investigation["recommendations"]:
                            print("\nRecommendations:")
                            for i, rec in enumerate(investigation["recommendations"], 1):
                                print(f"  {i}. {rec}")
                
                elif disc["issue"] == "missing_in_db":
                    print(f"⚠️  Coin exists in API but not in database")
                    print(f"API Wallet Balance:  {disc.get('api_wallet_balance', 'N/A')}")
                    print(f"API Available:        {disc.get('api_available_balance', 'N/A')}")
                    print(f"API Frozen:           {disc.get('api_frozen', 'N/A')}")
                    print("\nPossible causes:")
                    print("  - WebSocket never received wallet event for this coin")
                    print("  - Balance parsing failed for this coin")
                    print("  - Coin was added to account after last WebSocket connection")
                
                elif disc["issue"] == "missing_in_api":
                    print(f"⚠️  Coin exists in database but not in API")
                    print(f"DB Wallet Balance:    {disc.get('db_wallet_balance', 'N/A')}")
                    print(f"DB Available:         {disc.get('db_available_balance', 'N/A')}")
                    print(f"DB Frozen:            {disc.get('db_frozen', 'N/A')}")
                    if disc.get("db_received_at"):
                        print(f"DB Record Age:        {disc.get('age_seconds', 0):.1f} seconds")
                    print("\nPossible causes:")
                    print("  - Coin was removed from account (balance became zero)")
                    print("  - API response structure changed")
                    print("  - Stale database record")
        
        print("\n" + "=" * 80)
        
        # Close connections
        await bybit_client.close()
        await DatabaseConnection.close_pool()
        logger.info("balance_comparison_complete")
        
    except Exception as e:
        logger.error("balance_comparison_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

