#!/usr/bin/env python3
"""Check balance from Bybit API directly."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'order-manager', 'src'))

from utils.bybit_client import get_bybit_client

async def check_balance():
    """Check balance from Bybit API."""
    client = get_bybit_client()
    
    try:
        response = await client.get(
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            authenticated=True,
        )
        
        print("=== Bybit API Balance Response ===")
        import json
        print(json.dumps(response, indent=2))
        
        if response.get("retCode") == 0:
            result = response.get("result", {})
            account_list = result.get("list", [])
            
            if account_list:
                account = account_list[0]
                print("\n=== Account Summary ===")
                print(f"Account Type: {account.get('accountType')}")
                print(f"Total Equity: {account.get('totalEquity')}")
                print(f"Total Wallet Balance: {account.get('totalWalletBalance')}")
                print(f"Total Available Balance: {account.get('totalAvailableBalance')}")
                print(f"Total Margin Balance: {account.get('totalMarginBalance')}")
                
                coins = account.get("coin", [])
                print(f"\n=== Coin Details ({len(coins)} coins) ===")
                for coin in coins:
                    coin_name = coin.get("coin")
                    wallet = coin.get("walletBalance", "0")
                    available = coin.get("availableToWithdraw", "") or coin.get("walletBalance", "0")
                    equity = coin.get("equity", "0")
                    
                    if float(wallet or 0) > 0 or float(available or 0) > 0:
                        print(f"{coin_name:10} | Wallet: {wallet:>15} | Available: {available:>15} | Equity: {equity:>15}")
                
                # Check USDT specifically
                usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)
                if usdt_coin:
                    print(f"\n=== USDT Balance ===")
                    print(f"Wallet Balance: {usdt_coin.get('walletBalance', '0')}")
                    print(f"Available To Withdraw: {usdt_coin.get('availableToWithdraw', 'N/A')}")
                    print(f"Equity: {usdt_coin.get('equity', '0')}")
                    print(f"USD Value: {usdt_coin.get('usdValue', '0')}")
            else:
                print("No account data found in response")
        else:
            print(f"API Error: retCode={response.get('retCode')}, retMsg={response.get('retMsg')}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_balance())

