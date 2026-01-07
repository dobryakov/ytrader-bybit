"""Metrics endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics/overview")
async def get_overview_metrics():
    """Get aggregated overview metrics."""
    trace_id = get_or_create_trace_id()
    logger.info("metrics_overview_request", trace_id=trace_id)

    try:
        # Get position metrics
        position_query = """
            SELECT 
                COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
                COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
                COUNT(*) FILTER (WHERE size != 0) as open_positions_count,
                COUNT(*) as total_positions_count
            FROM positions
        """
        position_row = await DatabaseConnection.fetchrow(position_query)

        # Get latest USDT balance
        # For unified accounts, try account-level balance first (from account_margin_balances),
        # which is more accurate when there are borrowed funds. Fall back to coin-level balance
        # (from account_balances) if account-level is not available.
        # This ensures consistency with order-manager balance checks.
        
        balance_row = None
        balance_source = None
        
        # Try account-level balance first (more accurate for unified accounts)
        account_query = """
            SELECT 
                total_available_balance,
                total_wallet_balance,
                base_currency,
                received_at
            FROM account_margin_balances
            ORDER BY received_at DESC
            LIMIT 1
        """
        account_row = await DatabaseConnection.fetchrow(account_query)
        
        if account_row is not None:
            base_currency = str(account_row["base_currency"]) or "USDT"
            
            # If base currency is USDT, use account-level balance directly
            if base_currency.upper() == "USDT":
                balance_row = {
                    "total_balance": account_row["total_wallet_balance"],
                    "total_available_balance": account_row["total_available_balance"],
                }
                balance_source = "account-level"
                logger.info(
                    "dashboard_balance_db_account_level_usdt",
                    available_balance=str(account_row["total_available_balance"]),
                    wallet_balance=str(account_row["total_wallet_balance"]),
                    balance_source=balance_source,
                    received_at=account_row["received_at"].isoformat() if account_row["received_at"] else None,
                    trace_id=trace_id,
                )
            else:
                # Base currency is not USDT, but we can still use it as it represents total available margin
                # For unified accounts, this is the actual available margin for trading
                balance_row = {
                    "total_balance": account_row["total_wallet_balance"],
                    "total_available_balance": account_row["total_available_balance"],
                }
                balance_source = "account-level"
                logger.info(
                    "dashboard_balance_db_account_level_non_usdt",
                    available_balance=str(account_row["total_available_balance"]),
                    wallet_balance=str(account_row["total_wallet_balance"]),
                    base_currency=base_currency,
                    balance_source=balance_source,
                    received_at=account_row["received_at"].isoformat() if account_row["received_at"] else None,
                    trace_id=trace_id,
                    note="Using account-level balance even though base_currency is not USDT (unified account margin)",
                )
        
        # Fallback to coin-level balance if account-level is not available
        if balance_row is None:
            coin_query = """
                SELECT 
                    available_balance,
                    wallet_balance,
                    received_at
                FROM account_balances
                WHERE coin = 'USDT'
                ORDER BY received_at DESC
                LIMIT 1
            """
            coin_row = await DatabaseConnection.fetchrow(coin_query)
            
            if coin_row is not None:
                balance_row = {
                    "total_balance": coin_row["wallet_balance"],
                    "total_available_balance": coin_row["available_balance"],
                }
                balance_source = "coin-level"
                
                # Warn if coin-level balance is negative (indicates borrowed funds)
                if coin_row["available_balance"] < 0:
                    logger.warning(
                        "dashboard_balance_db_negative_coin_level",
                        available_balance=str(coin_row["available_balance"]),
                        received_at=coin_row["received_at"].isoformat() if coin_row["received_at"] else None,
                        balance_source=balance_source,
                        trace_id=trace_id,
                        note="Coin-level balance is negative (likely due to borrowed funds). Consider using account-level balance from account_margin_balances.",
                    )
                
                logger.info(
                    "dashboard_balance_db_latest_usdt",
                    available_balance=str(coin_row["available_balance"]),
                    wallet_balance=str(coin_row["wallet_balance"]),
                    balance_source=balance_source,
                    received_at=coin_row["received_at"].isoformat() if coin_row["received_at"] else None,
                    trace_id=trace_id,
                )
            else:
                logger.warning(
                    "dashboard_balance_db_no_usdt",
                    trace_id=trace_id,
                )

        metrics = {
            "total_unrealized_pnl": str(position_row["total_unrealized_pnl"]),
            "total_realized_pnl": str(position_row["total_realized_pnl"]),
            "open_positions_count": position_row["open_positions_count"],
            "total_positions_count": position_row["total_positions_count"],
            "balance": str(balance_row["total_balance"]) if balance_row and balance_row["total_balance"] is not None else None,
            "available_balance": str(balance_row["total_available_balance"]) if balance_row and balance_row["total_available_balance"] is not None else None,
        }

        logger.info("metrics_overview_completed", trace_id=trace_id)

        return JSONResponse(status_code=200, content=metrics)

    except Exception as e:
        logger.error("metrics_overview_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.get("/metrics/portfolio")
async def get_portfolio_metrics():
    """Get portfolio metrics by asset."""
    trace_id = get_or_create_trace_id()
    logger.info("metrics_portfolio_request", trace_id=trace_id)

    try:
        query = """
            SELECT 
                asset,
                SUM(unrealized_pnl) as unrealized_pnl,
                SUM(realized_pnl) as realized_pnl,
                SUM(size * COALESCE(current_price, average_entry_price)) as exposure
            FROM positions
            WHERE size != 0
            GROUP BY asset
            ORDER BY exposure DESC
        """

        rows = await DatabaseConnection.fetch(query)

        portfolio_data = []
        for row in rows:
            portfolio_dict = {
                "asset": row["asset"],
                "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else "0",
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else "0",
                "exposure": str(row["exposure"]) if row["exposure"] else "0",
            }
            portfolio_data.append(portfolio_dict)

        logger.info("metrics_portfolio_completed", count=len(portfolio_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "portfolio": portfolio_data,
                "count": len(portfolio_data),
            },
        )

    except Exception as e:
        logger.error("metrics_portfolio_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve portfolio metrics: {str(e)}")


@router.get("/metrics/balances")
async def get_balances_by_asset():
    """Get available balances by asset/coin for trading."""
    trace_id = get_or_create_trace_id()
    logger.info("metrics_balances_request", trace_id=trace_id)

    try:
        # Get latest available balance for each coin
        # available_balance is the balance available for trading (not frozen)
        query = """
            WITH latest_balances AS (
                SELECT DISTINCT ON (coin)
                    coin,
                    available_balance,
                    wallet_balance,
                    frozen,
                    received_at
                FROM account_balances
                ORDER BY coin, received_at DESC
            )
            SELECT 
                coin,
                available_balance,
                wallet_balance,
                frozen,
                received_at
            FROM latest_balances
            WHERE available_balance > 0 OR wallet_balance > 0
            ORDER BY coin
        """

        rows = await DatabaseConnection.fetch(query)

        balances_data = []
        for row in rows:
            balance_dict = {
                "coin": row["coin"],
                "available_balance": str(row["available_balance"]),  # Available for trading
                "wallet_balance": str(row["wallet_balance"]),  # Total balance
                "frozen": str(row["frozen"]),  # Frozen balance
                "last_updated": row["received_at"].isoformat() + "Z" if row["received_at"] else None,
            }
            balances_data.append(balance_dict)

        logger.info("metrics_balances_completed", count=len(balances_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "balances": balances_data,
                "count": len(balances_data),
            },
        )

    except Exception as e:
        logger.error("metrics_balances_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve balances: {str(e)}")

