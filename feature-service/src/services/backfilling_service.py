"""
Backfilling service for fetching historical market data from Bybit REST API.

Fetches historical data when insufficient data is available for model training,
enabling immediate model training without waiting for data accumulation.
"""
import asyncio
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pathlib import Path
import pandas as pd
import structlog

from src.utils.bybit_client import BybitClient, BybitAPIError
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.config import config
from src.logging import get_logger

logger = structlog.get_logger(__name__)


class BackfillingJob:
    """Represents a backfilling job."""
    
    def __init__(self, job_id: str, symbol: str, start_date: date, end_date: date, data_types: List[str]):
        self.job_id = job_id
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_types = data_types
        self.status = "pending"  # pending, in_progress, completed, failed
        self.progress = {
            "dates_completed": 0,
            "dates_total": 0,
            "current_date": None,
        }
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.completed_dates: List[date] = []
        self.failed_dates: List[date] = []


class BackfillingService:
    """Service for backfilling historical market data from Bybit REST API."""
    
    def __init__(
        self,
        parquet_storage: ParquetStorage,
        feature_registry_loader: Optional[FeatureRegistryLoader] = None,
        bybit_client: Optional[BybitClient] = None,
    ):
        """
        Initialize backfilling service.
        
        Args:
            parquet_storage: Parquet storage for saving backfilled data
            feature_registry_loader: Optional Feature Registry loader for determining data types
            bybit_client: Optional Bybit REST API client (creates default if not provided)
        """
        self._parquet_storage = parquet_storage
        self._feature_registry_loader = feature_registry_loader
        
        if bybit_client is None:
            self._bybit_client = BybitClient(
                api_key=config.bybit_api_key,
                api_secret=config.bybit_api_secret,
                base_url=config.bybit_rest_base_url,
                rate_limit_delay_ms=config.feature_service_backfill_rate_limit_delay_ms,
            )
        else:
            self._bybit_client = bybit_client
        
        self._jobs: Dict[str, BackfillingJob] = {}
        self._max_candles_per_request = 200  # Bybit API limit
    
    async def backfill_klines(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Backfill kline (candlestick) data from Bybit REST API.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling
            end_date: End date for backfilling
            interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
            
        Returns:
            List of kline data dictionaries in internal format
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.info(
            "backfilling_klines_start",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval=interval,
        )
        
        # Convert dates to timestamps (milliseconds)
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            # Calculate end timestamp for this chunk (max 200 candles)
            # For 1-minute interval: 200 minutes = 200 * 60 * 1000 ms
            chunk_duration_ms = self._max_candles_per_request * interval * 60 * 1000
            current_end = min(current_start + chunk_duration_ms, end_timestamp)
            
            try:
                # Make API request
                # Note: Bybit API returns data starting from 'start' timestamp
                # We use 'end' to limit the range, but API may return data up to 'end' inclusive
                response = await self._bybit_client.get(
                    endpoint="/v5/market/kline",
                    params={
                        "category": config.bybit_market_category,
                        "symbol": symbol,
                        "interval": str(interval),
                        "start": current_start,
                        "end": current_end,
                        "limit": self._max_candles_per_request,
                    },
                    authenticated=False,  # Public endpoint
                )
                
                # Parse response
                if "result" not in response or "list" not in response["result"]:
                    logger.warning(
                        "bybit_api_unexpected_response",
                        symbol=symbol,
                        response_keys=list(response.keys()),
                    )
                    break
                
                klines = response["result"]["list"]
                if not klines:
                    # No more data available, stop pagination
                    break
                
                # Log pagination details for debugging
                first_timestamp = int(klines[0][0]) if klines else None
                last_timestamp_in_response = int(klines[-1][0]) if klines else None
                logger.debug(
                    "backfilling_klines_api_response",
                    symbol=symbol,
                    request_start=current_start,
                    request_end=current_end,
                    response_count=len(klines),
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp_in_response,
                    first_datetime=datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).isoformat() if first_timestamp else None,
                    last_datetime=datetime.fromtimestamp(last_timestamp_in_response / 1000, tz=timezone.utc).isoformat() if last_timestamp_in_response else None,
                )
                
                # Convert to internal format using unified normalizer
                from src.storage.data_normalizer import normalize_kline_data
                
                for kline in klines:
                    # Bybit format: [startTime, open, high, low, close, volume, turnover]
                    # Convert to dict format for normalizer
                    kline_dict = {
                        "start": int(kline[0]),
                        "open": kline[1],
                        "high": kline[2],
                        "low": kline[3],
                        "close": kline[4],
                        "volume": kline[5],
                        "symbol": symbol,
                        "interval": str(interval),  # Add interval from function parameter
                    }
                    # Normalize to unified format (adds interval, internal_timestamp, exchange_timestamp)
                    internal_kline = normalize_kline_data(
                        kline_dict,
                        source="backfilling",
                        internal_timestamp=None,  # Backfilling doesn't have internal timestamp
                        exchange_timestamp=None,  # Will use timestamp from data
                    )
                    all_klines.append(internal_kline)
                
                # Check for duplicates in current response
                timestamps_in_response = [int(k[0]) for k in klines]
                unique_timestamps = set(timestamps_in_response)
                if len(timestamps_in_response) != len(unique_timestamps):
                    duplicates_in_response = len(timestamps_in_response) - len(unique_timestamps)
                    logger.warning(
                        "backfilling_duplicates_in_api_response",
                        symbol=symbol,
                        request_start=current_start,
                        total_records=len(klines),
                        unique_records=len(unique_timestamps),
                        duplicates=duplicates_in_response,
                        message="API returned duplicate timestamps in single response",
                    )
                
                # Check for overlap with previously fetched data
                if all_klines:
                    existing_timestamps = {int(k["timestamp"].timestamp() * 1000) for k in all_klines}
                    overlapping = existing_timestamps.intersection(unique_timestamps)
                    if overlapping:
                        logger.warning(
                            "backfilling_pagination_overlap",
                            symbol=symbol,
                            request_start=current_start,
                            overlapping_count=len(overlapping),
                            overlapping_timestamps=sorted(list(overlapping))[:10],  # First 10 for logging
                            message="API returned data that overlaps with previously fetched data",
                        )
                        # Filter out overlapping data before adding to all_klines
                        # This prevents duplicates even if API returns overlapping data
                        new_klines = [k for k in klines if int(k[0]) not in existing_timestamps]
                        if len(new_klines) < len(klines):
                            logger.info(
                                "backfilling_filtered_overlapping",
                                symbol=symbol,
                                original_count=len(klines),
                                filtered_count=len(new_klines),
                                removed_count=len(klines) - len(new_klines),
                            )
                            klines = new_klines
                            # Recalculate unique timestamps after filtering
                            unique_timestamps = {int(k[0]) for k in klines}
                
                # Update current_start to last kline timestamp + interval duration
                # This ensures we don't request the same data twice
                # Use the last timestamp from filtered data (without overlaps)
                if klines:
                    last_timestamp = int(klines[-1][0])
                    next_start = last_timestamp + (interval * 60 * 1000)
                else:
                    # No new data in this response, advance by chunk duration to avoid infinite loop
                    next_start = current_end + (interval * 60 * 1000)
                
                # Safety check: if next_start doesn't advance, break to avoid infinite loop
                if next_start <= current_start:
                    logger.warning(
                        "backfilling_pagination_stuck",
                        symbol=symbol,
                        current_start=current_start,
                        next_start=next_start,
                        last_timestamp=last_timestamp,
                    )
                    break
                
                current_start = next_start
                
                # Safety check: if we've processed all requested data, break
                if current_start >= end_timestamp:
                    break
                
                logger.debug(
                    "backfilling_klines_chunk",
                    symbol=symbol,
                    chunk_size=len(klines),
                    total_klines=len(all_klines),
                )
                
            except BybitAPIError as e:
                logger.error(
                    "backfilling_klines_api_error",
                    symbol=symbol,
                    error=str(e),
                    current_start=current_start,
                )
                raise
        
        # Remove duplicates by timestamp (keep last occurrence)
        # This handles cases where API returns duplicate data
        if all_klines:
            # Convert to DataFrame for efficient deduplication
            df_temp = pd.DataFrame(all_klines)
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            initial_count = len(df_temp)
            df_temp = df_temp.drop_duplicates(subset=["timestamp"], keep="last")
            df_temp = df_temp.sort_values("timestamp")
            
            if len(df_temp) < initial_count:
                logger.warning(
                    "backfilling_klines_duplicates_removed",
                    symbol=symbol,
                    initial_count=initial_count,
                    final_count=len(df_temp),
                    duplicates_removed=initial_count - len(df_temp),
                )
            
            # Convert back to list of dicts
            all_klines = df_temp.to_dict("records")
        
        logger.info(
            "backfilling_klines_complete",
            symbol=symbol,
            total_klines=len(all_klines),
        )
        
        return all_klines
    
    async def backfill_trades(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Backfill trades data from Bybit REST API.
        
        Note: Bybit /v5/market/recent-trade endpoint returns only recent trades (limit 60 for spot).
        Historical trades may not be available through REST API. This method attempts to fetch
        available data but may be limited to recent trades only.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling
            end_date: End date for backfilling
            
        Returns:
            List of trade data dictionaries in internal format
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.info(
            "backfilling_trades_start",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        
        all_trades = []
        
        # Note: Bybit /v5/market/recent-trade endpoint doesn't support historical pagination
        # It only returns recent trades (up to 60 for spot category)
        # We'll attempt to fetch recent trades, but this may not cover the full date range
        try:
            response = await self._bybit_client.get(
                endpoint="/v5/market/recent-trade",
                params={
                    "category": config.bybit_market_category,
                    "symbol": symbol,
                    "limit": 60,  # Max 60 for spot category
                },
                authenticated=False,  # Public endpoint
            )
            
            # Parse response
            if "result" not in response or "list" not in response["result"]:
                logger.warning(
                    "bybit_api_unexpected_response_trades",
                    symbol=symbol,
                    response_keys=list(response.keys()),
                )
                return all_trades
            
            trades = response["result"]["list"]
            if not trades:
                logger.warning(
                    "backfilling_trades_no_data",
                    symbol=symbol,
                )
                return all_trades
            
            # Convert to internal format using unified normalizer
            from src.storage.data_normalizer import normalize_trade_data
            
            for trade in trades:
                trade_timestamp_ms = int(trade["time"])
                trade_datetime = datetime.fromtimestamp(trade_timestamp_ms / 1000, tz=timezone.utc)
                
                # Filter by date range
                if trade_datetime.date() < start_date or trade_datetime.date() > end_date:
                    continue
                
                # Convert to dict format for normalizer
                trade_dict = {
                    "timestamp": trade_timestamp_ms,
                    "price": trade["price"],
                    "quantity": trade["size"],
                    "side": trade["side"],  # "Buy" or "Sell"
                    "symbol": symbol,
                }
                
                # Normalize to unified format (adds internal_timestamp, exchange_timestamp)
                internal_trade = normalize_trade_data(
                    trade_dict,
                    source="backfilling",
                    internal_timestamp=None,  # Backfilling doesn't have internal timestamp
                    exchange_timestamp=None,  # Will use timestamp from data
                )
                all_trades.append(internal_trade)
            
            logger.info(
                "backfilling_trades_complete",
                symbol=symbol,
                total_trades=len(all_trades),
                note="Bybit API returns only recent trades, historical trades may not be available",
            )
            
        except BybitAPIError as e:
            logger.error(
                "backfilling_trades_api_error",
                symbol=symbol,
                error=str(e),
            )
            raise
        
        return all_trades
    
    async def backfill_orderbook_snapshots(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Backfill orderbook snapshots from Bybit REST API.
        
        Note: Bybit /v5/market/orderbook endpoint returns only current orderbook snapshot.
        Historical orderbook snapshots are not available through REST API. This method
        fetches the current snapshot, which may not match the requested date range.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling (not used - API returns current snapshot)
            end_date: End date for backfilling (not used - API returns current snapshot)
            
        Returns:
            List containing single orderbook snapshot in internal format (empty if unavailable)
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.warning(
            "backfilling_orderbook_snapshots_limited",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            note="Bybit REST API does not provide historical orderbook snapshots, only current snapshot",
        )
        
        # Bybit API only provides current snapshot, not historical
        # Return empty list as historical snapshots are not available
        logger.warning(
            "backfilling_orderbook_snapshots_not_available",
            symbol=symbol,
            message="Historical orderbook snapshots not available via REST API",
        )
        return []
    
    async def backfill_orderbook_deltas(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Backfill orderbook deltas from Bybit REST API.
        
        Note: Bybit REST API does not provide historical orderbook deltas. Orderbook deltas
        are typically reconstructed from snapshots or require WebSocket stream data.
        This method returns empty list as deltas are not available via REST API.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling (not used)
            end_date: End date for backfilling (not used)
            
        Returns:
            Empty list (orderbook deltas not available via REST API)
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.warning(
            "backfilling_orderbook_deltas_not_available",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            message="Historical orderbook deltas not available via Bybit REST API",
        )
        return []
    
    async def backfill_ticker(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Backfill ticker data from Bybit REST API.
        
        Note: Bybit /v5/market/tickers endpoint returns only current ticker data.
        Historical ticker data is not available through REST API. This method
        fetches the current ticker, which may not match the requested date range.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling (not used - API returns current ticker)
            end_date: End date for backfilling (not used - API returns current ticker)
            
        Returns:
            List containing single ticker in internal format (empty if unavailable)
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.warning(
            "backfilling_ticker_limited",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            note="Bybit REST API does not provide historical ticker data, only current ticker",
        )
        
        try:
            response = await self._bybit_client.get(
                endpoint="/v5/market/tickers",
                params={
                    "category": config.bybit_market_category,
                    "symbol": symbol,
                },
                authenticated=False,  # Public endpoint
            )
            
            # Parse response
            if "result" not in response or "list" not in response["result"]:
                logger.warning(
                    "bybit_api_unexpected_response_ticker",
                    symbol=symbol,
                    response_keys=list(response.keys()),
                )
                return []
            
            tickers = response["result"]["list"]
            if not tickers:
                return []
            
            # Find ticker for our symbol
            ticker_data = None
            for ticker in tickers:
                if ticker.get("symbol") == symbol:
                    ticker_data = ticker
                    break
            
            if not ticker_data:
                logger.warning(
                    "backfilling_ticker_symbol_not_found",
                    symbol=symbol,
                )
                return []
            
            # Convert to internal format
            # Bybit format: {"symbol", "lastPrice", "bid1Price", "ask1Price", "volume24h", "time"}
            # Internal format: timestamp, last_price, bid_price, ask_price, volume_24h, symbol
            ticker_timestamp_ms = int(ticker_data.get("time", int(datetime.now(timezone.utc).timestamp() * 1000)))
            ticker_datetime = datetime.fromtimestamp(ticker_timestamp_ms / 1000, tz=timezone.utc)
            
            internal_ticker = {
                "timestamp": ticker_datetime,
                "last_price": float(ticker_data["lastPrice"]) if ticker_data.get("lastPrice") else None,
                "bid_price": float(ticker_data["bid1Price"]) if ticker_data.get("bid1Price") else None,
                "ask_price": float(ticker_data["ask1Price"]) if ticker_data.get("ask1Price") else None,
                "volume_24h": float(ticker_data["volume24h"]) if ticker_data.get("volume24h") else None,
                "symbol": symbol,
            }
            
            logger.info(
                "backfilling_ticker_complete",
                symbol=symbol,
                note="Only current ticker available, historical ticker data not supported",
            )
            
            return [internal_ticker]
            
        except BybitAPIError as e:
            logger.error(
                "backfilling_ticker_api_error",
                symbol=symbol,
                error=str(e),
            )
            raise
    
    async def backfill_funding(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Backfill funding rate data from Bybit REST API.
        
        Note: Funding rates are only available for perpetual contracts (linear/inverse),
        not for spot trading. This method uses /v5/market/funding/history endpoint
        which supports historical data with pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT" for linear perpetual)
            start_date: Start date for backfilling
            end_date: End date for backfilling
            
        Returns:
            List of funding rate data dictionaries in internal format
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.info(
            "backfilling_funding_start",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        
        # Convert dates to timestamps (milliseconds)
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_funding = []
        current_start = start_timestamp
        max_records_per_request = 200  # Bybit API limit
        
        while current_start < end_timestamp:
            # Calculate end timestamp for this chunk
            current_end = min(current_start + (max_records_per_request * 8 * 60 * 60 * 1000), end_timestamp)
            # Funding rates occur every 8 hours, so 200 records = ~66 days
            
            try:
                response = await self._bybit_client.get(
                    endpoint="/v5/market/funding/history",
                    params={
                        "category": "linear",  # Funding rates are for perpetual contracts
                        "symbol": symbol,
                        "startTime": current_start,
                        "endTime": current_end,
                        "limit": max_records_per_request,
                    },
                    authenticated=False,  # Public endpoint
                )
                
                # Parse response
                if "result" not in response or "list" not in response["result"]:
                    logger.warning(
                        "bybit_api_unexpected_response_funding",
                        symbol=symbol,
                        response_keys=list(response.keys()),
                    )
                    break
                
                funding_list = response["result"]["list"]
                if not funding_list:
                    # No more data available, stop pagination
                    break
                
                # Convert to internal format using unified normalizer
                from src.storage.data_normalizer import normalize_funding_data
                
                for funding in funding_list:
                    # Convert to dict format for normalizer
                    funding_dict = {
                        "fundingRateTimestamp": int(funding["fundingRateTimestamp"]),
                        "fundingRate": funding["fundingRate"],
                        "symbol": symbol,
                        "nextFundingTime": funding.get("nextFundingTime"),  # May not be present
                    }
                    
                    # Normalize to unified format (adds internal_timestamp, exchange_timestamp)
                    internal_funding = normalize_funding_data(
                        funding_dict,
                        source="backfilling",
                        internal_timestamp=None,  # Backfilling doesn't have internal timestamp
                        exchange_timestamp=None,  # Will use timestamp from data
                    )
                    all_funding.append(internal_funding)
                
                # Check if we got fewer records than requested (last page)
                if len(funding_list) < max_records_per_request:
                    break
                
                # Move to next page: use last timestamp + 1ms
                last_timestamp = int(funding_list[-1]["fundingRateTimestamp"])
                current_start = last_timestamp + 1
                
                logger.debug(
                    "backfilling_funding_chunk",
                    symbol=symbol,
                    chunk_size=len(funding_list),
                    total_funding=len(all_funding),
                )
                
            except BybitAPIError as e:
                # Check if error is about spot symbol (funding rates are only for perpetuals)
                error_msg = str(e).lower()
                if "spot" in error_msg or "not found" in error_msg:
                    logger.warning(
                        "backfilling_funding_spot_not_supported",
                        symbol=symbol,
                        message="Funding rates are only available for perpetual contracts, not spot",
                    )
                    return []
                logger.error(
                    "backfilling_funding_api_error",
                    symbol=symbol,
                    error=str(e),
                    current_start=current_start,
                )
                raise
        
        # Sort by timestamp
        all_funding.sort(key=lambda x: x["timestamp"])
        
        logger.info(
            "backfilling_funding_complete",
            symbol=symbol,
            total_funding=len(all_funding),
        )
        
        return all_funding
    
    async def _delete_data_for_date(
        self, 
        symbol: str, 
        date_str: str, 
        data_types: List[str]
    ) -> None:
        """
        Delete existing data files for specific data types for a symbol and date.
        This ensures clean backfilling without partial or corrupted data.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data_types: List of data types to delete (e.g., ["klines", "trades"])
        """
        data_type_paths = {
            "klines": self._parquet_storage._get_klines_path,
            "trades": self._parquet_storage._get_trades_path,
            "funding": self._parquet_storage._get_funding_path,
            "ticker": self._parquet_storage._get_ticker_path,
            "orderbook_snapshots": self._parquet_storage._get_orderbook_snapshots_path,
            "orderbook_deltas": self._parquet_storage._get_orderbook_deltas_path,
        }
        
        deleted_count = 0
        for data_type in data_types:
            if data_type not in data_type_paths:
                logger.warning(
                    "backfilling_delete_unknown_type",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    message=f"Unknown data type for deletion: {data_type}",
                )
                continue
            
            try:
                get_path_method = data_type_paths[data_type]
                file_path = get_path_method(symbol, date_str)
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(
                        "backfilling_deleted_existing_data",
                        symbol=symbol,
                        date=date_str,
                        data_type=data_type,
                        file_path=str(file_path),
                    )
            except Exception as e:
                logger.warning(
                    "backfilling_delete_error",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    error=str(e),
                )
        
        if deleted_count > 0:
            logger.info(
                "backfilling_cleaned_existing_data",
                symbol=symbol,
                date=date_str,
                files_deleted=deleted_count,
                data_types=data_types,
            )
    
    async def _save_klines(
        self,
        symbol: str,
        date_str: str,
        klines: List[Dict[str, Any]],
    ) -> None:
        """
        Save klines to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            klines: List of kline data dictionaries
        """
        if not klines:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        # This prevents duplicate data from being saved
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "klines_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        # Delete existing file if it exists to ensure clean data
        klines_path = self._parquet_storage._get_klines_path(symbol, date_str)
        if klines_path.exists():
            logger.debug(
                "klines_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(klines_path),
            )
            klines_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_klines(symbol, date_str, df)
        
        logger.debug(
            "klines_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(klines),
        )
    
    async def _save_trades(
        self,
        symbol: str,
        date_str: str,
        trades: List[Dict[str, Any]],
    ) -> None:
        """
        Save trades to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            trades: List of trade data dictionaries
        """
        if not trades:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "trades_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        trades_path = self._parquet_storage._get_trades_path(symbol, date_str)
        if trades_path.exists():
            logger.debug(
                "trades_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(trades_path),
            )
            trades_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_trades(symbol, date_str, df)
        
        logger.debug(
            "trades_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(trades),
        )
    
    async def _save_orderbook_snapshots(
        self,
        symbol: str,
        date_str: str,
        snapshots: List[Dict[str, Any]],
    ) -> None:
        """
        Save orderbook snapshots to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            snapshots: List of orderbook snapshot data dictionaries
        """
        if not snapshots:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(snapshots)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "orderbook_snapshots_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        snapshots_path = self._parquet_storage._get_orderbook_snapshots_path(symbol, date_str)
        if snapshots_path.exists():
            logger.debug(
                "orderbook_snapshots_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(snapshots_path),
            )
            snapshots_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_orderbook_snapshots(symbol, date_str, df)
        
        logger.debug(
            "orderbook_snapshots_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(snapshots),
        )
    
    async def _save_orderbook_deltas(
        self,
        symbol: str,
        date_str: str,
        deltas: List[Dict[str, Any]],
    ) -> None:
        """
        Save orderbook deltas to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            deltas: List of orderbook delta data dictionaries
        """
        if not deltas:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(deltas)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by sequence if available, otherwise by timestamp
        initial_count = len(df)
        if "sequence" in df.columns:
            df = df.drop_duplicates(subset=["sequence"], keep="last")
        else:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "orderbook_deltas_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        deltas_path = self._parquet_storage._get_orderbook_deltas_path(symbol, date_str)
        if deltas_path.exists():
            logger.debug(
                "orderbook_deltas_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(deltas_path),
            )
            deltas_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_orderbook_deltas(symbol, date_str, df)
        
        logger.debug(
            "orderbook_deltas_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(deltas),
        )
    
    async def _save_ticker(
        self,
        symbol: str,
        date_str: str,
        tickers: List[Dict[str, Any]],
    ) -> None:
        """
        Save ticker data to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            tickers: List of ticker data dictionaries
        """
        if not tickers:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(tickers)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "ticker_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        ticker_path = self._parquet_storage._get_ticker_path(symbol, date_str)
        if ticker_path.exists():
            logger.debug(
                "ticker_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(ticker_path),
            )
            ticker_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_ticker(symbol, date_str, df)
        
        logger.debug(
            "ticker_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(tickers),
        )
    
    async def _save_funding(
        self,
        symbol: str,
        date_str: str,
        funding: List[Dict[str, Any]],
    ) -> None:
        """
        Save funding rate data to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            funding: List of funding rate data dictionaries
        """
        if not funding:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(funding)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "funding_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        funding_path = self._parquet_storage._get_funding_path(symbol, date_str)
        if funding_path.exists():
            logger.debug(
                "funding_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(funding_path),
            )
            funding_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_funding(symbol, date_str, df)
        
        logger.debug(
            "funding_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(funding),
        )
    
    async def _validate_saved_data(
        self,
        symbol: str,
        date_str: str,
        expected_count: int,
        data_type: str,
    ) -> bool:
        """
        Validate that saved data is correct and readable.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            expected_count: Expected number of records
            data_type: Data type ("klines", "trades", etc.)
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Read data back from Parquet
            if data_type == "klines":
                read_data = await self._parquet_storage.read_klines(symbol, date_str)
            elif data_type == "trades":
                read_data = await self._parquet_storage.read_trades(symbol, date_str)
            elif data_type == "orderbook_snapshots":
                read_data = await self._parquet_storage.read_orderbook_snapshots(symbol, date_str)
            elif data_type == "orderbook_deltas":
                read_data = await self._parquet_storage.read_orderbook_deltas(symbol, date_str)
            elif data_type == "ticker":
                read_data = await self._parquet_storage.read_ticker(symbol, date_str)
            elif data_type == "funding":
                read_data = await self._parquet_storage.read_funding(symbol, date_str)
            else:
                logger.warning(
                    "data_validation_unknown_type",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                )
                return True
            
            # Verify record count
            actual_count = len(read_data)
            if actual_count != expected_count:
                logger.error(
                    "data_validation_failed_count_mismatch",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    expected_count=expected_count,
                    actual_count=actual_count,
                )
                return False
            
            # Verify data structure based on data type
            if data_type == "klines":
                required_fields = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            elif data_type == "trades":
                required_fields = ["timestamp", "price", "quantity", "side", "symbol"]
            elif data_type == "orderbook_snapshots":
                required_fields = ["timestamp", "bids", "asks", "symbol"]
            elif data_type == "orderbook_deltas":
                required_fields = ["timestamp", "bids", "asks", "symbol"]  # sequence is optional
            elif data_type == "ticker":
                required_fields = ["timestamp", "last_price", "bid_price", "ask_price", "volume_24h", "symbol"]
            elif data_type == "funding":
                required_fields = ["timestamp", "funding_rate", "symbol"]
            else:
                required_fields = ["timestamp", "symbol"]  # Minimum required
            
            missing_fields = [field for field in required_fields if field not in read_data.columns]
            if missing_fields:
                logger.error(
                    "data_validation_failed_missing_fields",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    missing_fields=missing_fields,
                )
                return False
            
            # Verify data types
            if not pd.api.types.is_datetime64_any_dtype(read_data["timestamp"]):
                logger.error(
                    "data_validation_failed_timestamp_type",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                )
                return False
            
            # Verify data integrity based on data type
            if data_type == "klines":
                # Prices must be positive
                if (read_data["open"] <= 0).any() or (read_data["high"] <= 0).any() or (read_data["low"] <= 0).any() or (read_data["close"] <= 0).any():
                    logger.error(
                        "data_validation_failed_invalid_prices",
                        symbol=symbol,
                        date=date_str,
                        data_type=data_type,
                    )
                    return False
            elif data_type == "trades":
                # Price and quantity must be positive
                if (read_data["price"] <= 0).any() or (read_data["quantity"] <= 0).any():
                    logger.error(
                        "data_validation_failed_invalid_trade_values",
                        symbol=symbol,
                        date=date_str,
                        data_type=data_type,
                    )
                    return False
            elif data_type == "ticker":
                # Prices should be positive (but can be None)
                price_cols = ["last_price", "bid_price", "ask_price"]
                for col in price_cols:
                    if col in read_data.columns:
                        non_null = read_data[col].dropna()
                        if len(non_null) > 0 and (non_null <= 0).any():
                            logger.error(
                                "data_validation_failed_invalid_ticker_prices",
                                symbol=symbol,
                                date=date_str,
                                data_type=data_type,
                                column=col,
                            )
                            return False
            elif data_type == "funding":
                # Funding rate can be positive or negative, but should be finite
                if (read_data["funding_rate"].isna()).any() or not pd.api.types.is_numeric_dtype(read_data["funding_rate"]):
                    logger.error(
                        "data_validation_failed_invalid_funding_rate",
                        symbol=symbol,
                        date=date_str,
                        data_type=data_type,
                    )
                    return False
            
            logger.debug(
                "data_validation_passed",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
                record_count=actual_count,
            )
            
            return True
            
        except FileNotFoundError:
            logger.error(
                "data_validation_failed_file_not_found",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
            )
            return False
        except Exception as e:
            logger.error(
                "data_validation_failed_exception",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
                error=str(e),
                exc_info=True,
            )
            return False
    
    async def _check_data_availability(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        data_types: List[str],
    ) -> Dict[date, List[str]]:
        """
        Check which dates need backfilling.
        
        ALWAYS returns all dates as requiring backfilling to force overwrite
        existing data, even if files exist. This ensures data completeness
        and correctness by re-fetching data that might be incomplete or corrupted.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            data_types: List of data types to check
            
        Returns:
            Dict mapping date to list of data types to backfill (always all types for all dates)
        """
        # Force backfill all dates regardless of existing files
        # This ensures data completeness and correctness
        missing_data: Dict[date, List[str]] = {}
        
        current_date = start_date
        while current_date <= end_date:
            # Always mark all data types as missing to force backfill
            missing_data[current_date] = data_types.copy()
            current_date += timedelta(days=1)
        
        logger.info(
            "backfilling_force_overwrite",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            dates_count=len(missing_data),
            message="Forcing backfill for all dates to ensure data completeness",
        )
        
        return missing_data
    
    async def backfill_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        data_types: Optional[List[str]] = None,
    ) -> str:
        """
        Backfill historical data for a symbol and date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for backfilling
            end_date: End date for backfilling
            data_types: Optional list of data types to backfill (if None, uses Feature Registry)
            
        Returns:
            Job ID for tracking backfilling progress
        """
        # Determine data types to backfill
        if data_types is None:
            if self._feature_registry_loader is not None:
                try:
                    required_types = self._feature_registry_loader.get_required_data_types()
                    data_type_mapping = self._feature_registry_loader.get_data_type_mapping()
                    # Map input sources to storage types
                    data_types = []
                    supported_types = {
                        "klines",
                        "trades",
                        "orderbook_snapshots",
                        "orderbook_deltas",
                        "ticker",
                        "funding",
                    }  # Supported types for backfilling
                    
                    for input_source in required_types:
                        if input_source in data_type_mapping:
                            storage_types = data_type_mapping[input_source]
                            for storage_type in storage_types:
                                # Add all supported types from mapping
                                if storage_type in supported_types:
                                    if storage_type not in data_types:
                                        data_types.append(storage_type)
                                else:
                                    logger.warning(
                                        "backfilling_unsupported_data_type",
                                        symbol=symbol,
                                        input_source=input_source,
                                        storage_type=storage_type,
                                        message=f"Backfilling for {storage_type} is not yet implemented",
                                    )
                    
                    # If no supported types found, default to klines
                    if not data_types:
                        logger.warning(
                            "backfilling_no_supported_types",
                            symbol=symbol,
                            required_types=required_types,
                            data_type_mapping=data_type_mapping,
                            fallback="using klines",
                        )
                        data_types = ["klines"]
                    
                    logger.info(
                        "backfilling_data_types_from_registry",
                        symbol=symbol,
                        data_types=data_types,
                        required_input_sources=required_types,
                    )
                except Exception as e:
                    logger.warning(
                        "feature_registry_analysis_failed",
                        error=str(e),
                        fallback="backfilling_all_data_types",
                    )
                    data_types = ["klines"]  # Default to klines
            else:
                data_types = ["klines"]  # Default to klines
        
        # Check data availability
        missing_data = await self._check_data_availability(symbol, start_date, end_date, data_types)
        
        if not missing_data:
            logger.info(
                "backfilling_no_missing_data",
                symbol=symbol,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            # Create a completed job
            job_id = str(uuid4())
            job = BackfillingJob(job_id, symbol, start_date, end_date, data_types)
            job.status = "completed"
            job.start_time = datetime.now(timezone.utc)
            job.end_time = datetime.now(timezone.utc)
            job.progress["dates_total"] = 0
            job.progress["dates_completed"] = 0
            self._jobs[job_id] = job
            return job_id
        
        # Create job
        job_id = str(uuid4())
        job = BackfillingJob(job_id, symbol, start_date, end_date, data_types)
        job.progress["dates_total"] = len(missing_data)
        self._jobs[job_id] = job
        
        # Start backfilling in background
        asyncio.create_task(self._backfill_job_task(job, missing_data))
        
        return job_id
    
    async def _backfill_job_task(
        self,
        job: BackfillingJob,
        missing_data: Dict[date, List[str]],
    ) -> None:
        """Background task for backfilling job."""
        job.status = "in_progress"
        job.start_time = datetime.now(timezone.utc)
        
        try:
            for date_obj, missing_types in missing_data.items():
                job.progress["current_date"] = date_obj.isoformat()
                date_str = date_obj.isoformat()
                processed_types = set()  # Track which types were processed for this date
                
                logger.info(
                    "backfilling_date_start",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    date=date_str,
                    missing_types=missing_types,
                )
                
                # Delete existing data for this date before backfilling
                # This ensures clean data without partial or corrupted files
                # Only delete and lock data types that are being backfilled
                # This prevents deleting data for types that are not being backfilled
                locks = {}
                for data_type in missing_types:
                    lock = await self._parquet_storage._get_file_lock(
                        job.symbol, date_str, data_type
                    )
                    locks[data_type] = lock
                
                # Acquire all locks before deleting (prevents queue writes during deletion)
                # This ensures no queue writes can happen between deletion and backfilling
                for lock in locks.values():
                    await lock.acquire()
                
                try:
                    # Delete files while holding locks (prevents queue writes)
                    # Only delete files for data types that are being backfilled
                    await self._delete_data_for_date(job.symbol, date_str, missing_types)
                    
                    # Release locks before writing (write methods will acquire their own locks)
                    # This allows write methods to use standard locking mechanism
                    for lock in locks.values():
                        lock.release()
                    
                    for data_type in missing_types:
                        processed_types.add(data_type)
                        try:
                            date_str = date_obj.isoformat()
                            backfilled_data = None
                            save_method = None
                            get_path_method = None
                            
                            # Backfill data based on type
                            if data_type == "klines":
                                backfilled_data = await self.backfill_klines(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                    interval=config.feature_service_backfill_default_interval,
                                )
                                save_method = self._save_klines
                                get_path_method = self._parquet_storage._get_klines_path
                            elif data_type == "trades":
                                backfilled_data = await self.backfill_trades(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                )
                                save_method = self._save_trades
                                get_path_method = self._parquet_storage._get_trades_path
                            elif data_type == "orderbook_snapshots":
                                backfilled_data = await self.backfill_orderbook_snapshots(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                )
                                save_method = self._save_orderbook_snapshots
                                get_path_method = self._parquet_storage._get_orderbook_snapshots_path
                            elif data_type == "orderbook_deltas":
                                backfilled_data = await self.backfill_orderbook_deltas(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                )
                                save_method = self._save_orderbook_deltas
                                get_path_method = self._parquet_storage._get_orderbook_deltas_path
                            elif data_type == "ticker":
                                backfilled_data = await self.backfill_ticker(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                )
                                save_method = self._save_ticker
                                get_path_method = self._parquet_storage._get_ticker_path
                            elif data_type == "funding":
                                backfilled_data = await self.backfill_funding(
                                    job.symbol,
                                    date_obj,
                                    date_obj,
                                )
                                save_method = self._save_funding
                                get_path_method = self._parquet_storage._get_funding_path
                            else:
                                logger.warning(
                                    "backfilling_unsupported_data_type_in_task",
                                    symbol=job.symbol,
                                    date=date_str,
                                    data_type=data_type,
                                    message=f"Backfilling for {data_type} is not implemented",
                                )
                                job.failed_dates.append(date_obj)
                                continue  # Skip to next data_type
                            
                            # Save and validate data
                            if backfilled_data:
                                # Count unique records before saving (after deduplication in save methods)
                                df_temp = pd.DataFrame(backfilled_data)
                                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
                                unique_count = df_temp["timestamp"].nunique()
                                
                                # Save to Parquet
                                await save_method(job.symbol, date_str, backfilled_data)
                                
                                # Validate saved data
                                validation_passed = await self._validate_saved_data(
                                    job.symbol,
                                    date_str,
                                    unique_count,
                                    data_type,
                                )
                                
                                if not validation_passed:
                                    # Delete corrupted file and mark date as failed
                                    try:
                                        data_path = get_path_method(job.symbol, date_str)
                                        if data_path.exists():
                                            data_path.unlink()
                                        logger.warning(
                                            "backfilling_validation_failed_deleted",
                                            symbol=job.symbol,
                                            date=date_str,
                                            data_type=data_type,
                                        )
                                    except Exception as e:
                                        logger.error(
                                            "backfilling_validation_failed_delete_error",
                                            symbol=job.symbol,
                                            date=date_str,
                                            data_type=data_type,
                                            error=str(e),
                                        )
                                    
                                    # Remove from completed if it was there, add to failed
                                    if date_obj in job.completed_dates:
                                        job.completed_dates.remove(date_obj)
                                    if date_obj not in job.failed_dates:
                                        job.failed_dates.append(date_obj)
                                else:
                                    logger.debug(
                                        "backfilling_data_type_completed",
                                        symbol=job.symbol,
                                        date=date_str,
                                        data_type=data_type,
                                        record_count=unique_count,
                                    )
                                    # Only mark date as completed if it's not already failed for another type
                                    if date_obj not in job.failed_dates:
                                        if date_obj not in job.completed_dates:
                                            job.completed_dates.append(date_obj)
                            else:
                                # No data backfilled - this is expected for some types
                                # For funding: spot symbols don't have funding rates (only perpetuals)
                                # For trades: historical trades may not be available via REST API for old dates
                                # For orderbook_snapshots, orderbook_deltas, ticker: no historical data available via REST API
                                if data_type == "funding":
                                    # Funding rates are only for perpetual contracts, not spot
                                    # Empty result is expected for spot symbols like BTCUSDT
                                    logger.debug(
                                        "backfilling_funding_empty_expected",
                                        symbol=job.symbol,
                                        date=date_str,
                                        note="Funding rates are only available for perpetual contracts, not spot symbols",
                                    )
                                elif data_type == "trades":
                                    # Historical trades may not be available via REST API
                                    # This is not necessarily a failure - data might be available via WebSocket only
                                    logger.debug(
                                        "backfilling_trades_empty",
                                        symbol=job.symbol,
                                        date=date_str,
                                        note="Historical trades may not be available via REST API for this date",
                                    )
                                else:
                                    # For orderbook_snapshots, orderbook_deltas, ticker - no historical data available is expected
                                    logger.debug(
                                        "backfilling_no_historical_data_available",
                                        symbol=job.symbol,
                                        date=date_str,
                                        data_type=data_type,
                                        note="Historical data not available via REST API for this type",
                                    )
                        
                        except Exception as e:
                            logger.error(
                                "backfilling_date_failed",
                                symbol=job.symbol,
                                date=date_obj.isoformat(),
                                data_type=data_type,
                                error=str(e),
                                exc_info=True,
                            )
                            # Remove from completed if it was there, add to failed
                            if date_obj in job.completed_dates:
                                job.completed_dates.remove(date_obj)
                            if date_obj not in job.failed_dates:
                                job.failed_dates.append(date_obj)
                    
                finally:
                    # Release all locks after processing all data types for this date
                    # This allows queue writes to proceed after backfilling is complete
                    for lock in locks.values():
                        if lock.locked():
                            lock.release()
                
                # After processing all types for this date, check if date should be marked as completed
                # If all types were processed and date is not in failed_dates, mark as completed
                if date_obj not in job.failed_dates and len(processed_types) == len(missing_types):
                    if date_obj not in job.completed_dates:
                        job.completed_dates.append(date_obj)
                        logger.info(
                            "backfilling_date_completed",
                            job_id=job.job_id,
                            symbol=job.symbol,
                            date=date_str,
                            processed_types=list(processed_types),
                        )
                else:
                    logger.warning(
                        "backfilling_date_not_completed",
                        job_id=job.job_id,
                        symbol=job.symbol,
                        date=date_str,
                        in_failed_dates=date_obj in job.failed_dates,
                        processed_types_count=len(processed_types),
                        missing_types_count=len(missing_types),
                        processed_types=list(processed_types),
                    )
                
                # Note: dates_completed will be recalculated at the end based on actual completed_dates
            
            # Clean up: remove duplicates and ensure no overlap between completed and failed
            job.completed_dates = list(set(job.completed_dates))
            job.failed_dates = list(set(job.failed_dates))
            # Remove any dates from completed that are also in failed
            job.completed_dates = [d for d in job.completed_dates if d not in job.failed_dates]
            
            # Update dates_completed to match actual completed dates
            job.progress["dates_completed"] = len(job.completed_dates)
            
            # Determine final job status
            job.end_time = datetime.now(timezone.utc)
            
            if job.failed_dates:
                # Some dates failed - mark as failed
                job.status = "failed"
                job.error_message = f"Failed to backfill {len(job.failed_dates)} date(s): {[d.isoformat() for d in job.failed_dates]}"
                logger.warning(
                    "backfilling_job_completed_with_failures",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    total_dates=job.progress["dates_total"],
                    completed_dates=len(job.completed_dates),
                    failed_dates=len(job.failed_dates),
                    failed_date_list=[d.isoformat() for d in job.failed_dates],
                )
            else:
                # All dates completed successfully
                job.status = "completed"
                logger.info(
                    "backfilling_job_completed",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    total_dates=job.progress["dates_total"],
                    completed_dates=len(job.completed_dates),
                    failed_dates=len(job.failed_dates),
                )
        
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now(timezone.utc)
            job.error_message = str(e)
            
            logger.error(
                "backfilling_job_failed",
                job_id=job.job_id,
                symbol=job.symbol,
                error=str(e),
                exc_info=True,
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get backfilling job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary or None if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        
        return {
            "job_id": job.job_id,
            "symbol": job.symbol,
            "start_date": job.start_date.isoformat(),
            "end_date": job.end_date.isoformat(),
            "data_types": job.data_types,
            "status": job.status,
            "progress": job.progress,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "error_message": job.error_message,
            "completed_dates": [d.isoformat() for d in job.completed_dates],
            "failed_dates": [d.isoformat() for d in job.failed_dates],
        }
    
    async def close(self) -> None:
        """Close Bybit client."""
        await self._bybit_client.close()

