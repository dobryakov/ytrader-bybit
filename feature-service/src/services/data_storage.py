"""
Raw data storage service for storing market data in Parquet format.
"""
import asyncio
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional
import structlog
import pandas as pd
from collections import defaultdict

from src.storage.parquet_storage import ParquetStorage
from src.logging import get_logger

logger = get_logger(__name__)


class DataStorageService:
    """Service for storing raw market data in Parquet format."""
    
    def __init__(
        self,
        base_path: str,
        parquet_storage: ParquetStorage,
        retention_days: int = 90,
        archive_path: Optional[str] = None,
        enable_archiving: bool = False,
        cleanup_interval_seconds: int = 3600,  # 1 hour
    ):
        """
        Initialize data storage service.
        
        Args:
            base_path: Base path for raw data storage
            parquet_storage: ParquetStorage instance for file operations
            retention_days: Number of days to retain data (default: 90)
            archive_path: Path for archived data (optional)
            enable_archiving: Whether to archive expired data instead of deleting
            cleanup_interval_seconds: Interval for automatic cleanup task (seconds)
        """
        self._base_path = Path(base_path)
        self._parquet_storage = parquet_storage
        self._retention_days = retention_days
        self._archive_path = Path(archive_path) if archive_path else None
        self._enable_archiving = enable_archiving
        self._cleanup_interval = cleanup_interval_seconds
        
        # Buffers for batching writes (reduce I/O operations)
        self._buffers: Dict[str, List[Dict]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()
        self._buffer_size = 100  # Flush buffer after 100 events
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Create base directory
        self._base_path.mkdir(parents=True, exist_ok=True)
        if self._archive_path:
            self._archive_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "data_storage_service_initialized",
            base_path=str(self._base_path),
            retention_days=self._retention_days,
            archiving_enabled=self._enable_archiving,
            archive_path=str(self._archive_path) if self._archive_path else None,
        )
    
    async def start(self) -> None:
        """Start the data storage service (start cleanup task)."""
        if self._running:
            return
        
        self._running = True
        
        # Start periodic cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("data_storage_service_started")
    
    async def stop(self) -> None:
        """Stop the data storage service (flush buffers, stop cleanup task)."""
        if not self._running:
            return
        
        self._running = False
        
        # Flush all buffers
        await self._flush_all_buffers()
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("data_storage_service_stopped")
    
    async def store_market_data_event(self, event: Dict) -> None:
        """
        Store a market data event (routes to appropriate storage method).
        
        Args:
            event: Market data event dictionary
        """
        event_type = event.get("event_type")
        symbol = event.get("symbol")
        
        if not symbol:
            logger.warning(
                "event_missing_symbol",
                event_type=event_type,
                message="Skipping event without symbol",
            )
            return
        
        try:
            # Handle orderbook events from ws-gateway (event_type="orderbook")
            # Need to determine if it's a snapshot or delta
            if event_type == "orderbook":
                payload = event.get("payload", {})
                data = payload.get("data", payload) if isinstance(payload, dict) else {}
                if not isinstance(data, dict):
                    data = payload if isinstance(payload, dict) else {}
                
                # Determine type: check payload.data.type, payload.type, or infer from structure
                orderbook_type = data.get("type") or payload.get("type")
                
                # If no type field, check if it's a snapshot by structure (has bids/asks)
                if orderbook_type is None:
                    if "b" in data or "a" in data or "bids" in data or "asks" in data:
                        orderbook_type = "snapshot"
                    else:
                        orderbook_type = "delta"
                
                # Convert to normalized event format for storage
                if orderbook_type == "snapshot":
                    # Create normalized snapshot event
                    timestamp = event.get("timestamp") or event.get("exchange_timestamp")
                    if isinstance(timestamp, str):
                        from dateutil.parser import parse
                        timestamp = parse(timestamp)
                    elif timestamp is None:
                        timestamp = datetime.now(timezone.utc)
                    elif not isinstance(timestamp, datetime):
                        if isinstance(timestamp, (int, float)):
                            timestamp = datetime.fromtimestamp(
                                timestamp / 1000 if timestamp > 1e10 else timestamp,
                                tz=timezone.utc
                            )
                        else:
                            timestamp = datetime.now(timezone.utc)
                    
                    snapshot_event = {
                        **event,
                        "event_type": "orderbook_snapshot",
                        "symbol": data.get("s") or symbol,
                        "bids": data.get("b", []),
                        "asks": data.get("a", []),
                        "sequence": data.get("seq", data.get("u", 0)),
                        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                    }
                    await self.store_orderbook_snapshot(snapshot_event)
                elif orderbook_type == "delta" or orderbook_type == "update":
                    # For deltas, we need to extract individual updates
                    # Bybit sends updates in data.b (bids) and data.a (asks)
                    # Each update is [price, quantity] or [price, quantity, action]
                    bids = data.get("b", [])
                    asks = data.get("a", [])
                    sequence = data.get("seq", data.get("u", 0))
                    
                    timestamp = event.get("timestamp") or event.get("exchange_timestamp")
                    if isinstance(timestamp, str):
                        from dateutil.parser import parse
                        timestamp = parse(timestamp)
                    elif timestamp is None:
                        timestamp = datetime.now(timezone.utc)
                    elif not isinstance(timestamp, datetime):
                        if isinstance(timestamp, (int, float)):
                            timestamp = datetime.fromtimestamp(
                                timestamp / 1000 if timestamp > 1e10 else timestamp,
                                tz=timezone.utc
                            )
                        else:
                            timestamp = datetime.now(timezone.utc)
                    
                    # Store each bid update
                    for bid_update in bids:
                        if isinstance(bid_update, list) and len(bid_update) >= 2:
                            delta_event = {
                                **event,
                                "event_type": "orderbook_delta",
                                "symbol": data.get("s") or symbol,
                                "sequence": sequence,
                                "delta_type": "update",
                                "side": "bid",
                                "price": float(bid_update[0]) if isinstance(bid_update[0], str) else bid_update[0],
                                "quantity": float(bid_update[1]) if isinstance(bid_update[1], str) else bid_update[1],
                                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                            }
                            await self.store_orderbook_delta(delta_event)
                    
                    # Store each ask update
                    for ask_update in asks:
                        if isinstance(ask_update, list) and len(ask_update) >= 2:
                            delta_event = {
                                **event,
                                "event_type": "orderbook_delta",
                                "symbol": data.get("s") or symbol,
                                "sequence": sequence,
                                "delta_type": "update",
                                "side": "ask",
                                "price": float(ask_update[0]) if isinstance(ask_update[0], str) else ask_update[0],
                                "quantity": float(ask_update[1]) if isinstance(ask_update[1], str) else ask_update[1],
                                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                            }
                            await self.store_orderbook_delta(delta_event)
            elif event_type == "orderbook_snapshot":
                await self.store_orderbook_snapshot(event)
            elif event_type == "orderbook_delta":
                await self.store_orderbook_delta(event)
            elif event_type == "trade" or event_type == "trades":
                # ws-gateway may publish as "trade" or "trades"
                # Normalize trade event format (ws-gateway uses "p" for price, "v" for volume)
                payload = event.get("payload", {})
                
                # ws-gateway sends trade data directly in payload (not wrapped in "data")
                # Format: {"p": price, "v": volume, "S": side, "T": timestamp, "s": symbol, ...}
                trade_data = payload if isinstance(payload, dict) else {}
                
                # If payload is a list (shouldn't happen with current ws-gateway, but handle it)
                if isinstance(payload, list) and len(payload) > 0:
                    trade_data = payload[0] if isinstance(payload[0], dict) else {}
                
                # Extract and normalize trade fields
                # ws-gateway format: "p" = price, "v" = volume, "S" = side, "T" = timestamp (ms), "s" = symbol
                normalized_trade_event = {
                    **event,
                    "event_type": "trade",
                    "symbol": trade_data.get("s") or trade_data.get("symbol") or symbol,
                    "price": trade_data.get("p") or trade_data.get("price"),
                    "quantity": trade_data.get("v") or trade_data.get("quantity") or trade_data.get("volume"),
                    "side": trade_data.get("S") or trade_data.get("side", "Buy"),
                    "trade_time": trade_data.get("T") or trade_data.get("trade_time") or event.get("timestamp") or event.get("exchange_timestamp"),
                }
                
                # Convert price and quantity to float if they are strings
                if normalized_trade_event.get("price"):
                    try:
                        normalized_trade_event["price"] = float(normalized_trade_event["price"])
                    except (ValueError, TypeError):
                        normalized_trade_event["price"] = 0.0
                else:
                    normalized_trade_event["price"] = 0.0
                
                if normalized_trade_event.get("quantity"):
                    try:
                        normalized_trade_event["quantity"] = float(normalized_trade_event["quantity"])
                    except (ValueError, TypeError):
                        normalized_trade_event["quantity"] = 0.0
                else:
                    normalized_trade_event["quantity"] = 0.0
                
                await self.store_trade(normalized_trade_event)
            elif event_type == "kline":
                await self.store_kline(event)
            elif event_type == "ticker":
                await self.store_ticker(event)
            elif event_type == "funding_rate":
                await self.store_funding_rate(event)
            elif event_type == "execution":
                await self.store_execution_event(event)
            else:
                logger.warning(
                    "unknown_event_type",
                    event_type=event_type,
                    symbol=symbol,
                    message="Skipping unknown event type",
                )
        except Exception as e:
            logger.error(
                "event_storage_error",
                event_type=event_type,
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )
    
    async def store_market_data_events(self, events: List[Dict]) -> None:
        """
        Store multiple market data events.
        
        Args:
            events: List of market data event dictionaries
        """
        for event in events:
            await self.store_market_data_event(event)
    
    async def store_orderbook_snapshot(self, event: Dict) -> None:
        """
        Store orderbook snapshot event.
        
        Args:
            event: Orderbook snapshot event dictionary
        """
        symbol = event.get("symbol")
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "sequence": event.get("sequence", 0),
            "bids": str(event.get("bids", [])),  # Store as string for Parquet
            "asks": str(event.get("asks", [])),  # Store as string for Parquet
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_orderbook_snapshots(symbol, date_str, df)
            logger.debug(
                "orderbook_snapshot_stored",
                symbol=symbol,
                date=date_str,
                sequence=event.get("sequence"),
            )
        except Exception as e:
            logger.error(
                "orderbook_snapshot_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_orderbook_delta(self, event: Dict) -> None:
        """
        Store orderbook delta event (all types: insert, update, delete).
        
        Args:
            event: Orderbook delta event dictionary
        """
        symbol = event.get("symbol")
        if not symbol:
            logger.warning("orderbook_delta_missing_symbol", event_data=str(event))
            return
        
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "sequence": event.get("sequence", 0),
            "delta_type": event.get("delta_type", "update"),
            "side": event.get("side", "bid"),
            "price": event.get("price", 0.0),
            "quantity": event.get("quantity", 0.0),
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_orderbook_deltas(symbol, date_str, df)
            logger.debug(
                "orderbook_delta_stored",
                symbol=symbol,
                date=date_str,
                sequence=event.get("sequence"),
                delta_type=event.get("delta_type"),
            )
        except Exception as e:
            logger.error(
                "orderbook_delta_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_trade(self, event: Dict) -> None:
        """
        Store trade event.
        
        Args:
            event: Trade event dictionary
        """
        symbol = event.get("symbol")
        if not symbol:
            logger.warning("trade_missing_symbol", event_data=str(event))
            return
        
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "price": event.get("price", 0.0),
            "quantity": event.get("quantity", 0.0),
            "side": event.get("side", "Buy"),
            "trade_time": self._parse_timestamp(event.get("trade_time", timestamp_str)),
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_trades(symbol, date_str, df)
            logger.debug(
                "trade_stored",
                symbol=symbol,
                date=date_str,
                price=event.get("price"),
            )
        except Exception as e:
            logger.error(
                "trade_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_kline(self, event: Dict) -> None:
        """
        Store kline/candlestick event.
        
        Args:
            event: Kline event dictionary
        """
        symbol = event.get("symbol")
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "interval": event.get("interval", "1m"),
            "open": event.get("open", 0.0),
            "high": event.get("high", 0.0),
            "low": event.get("low", 0.0),
            "close": event.get("close", 0.0),
            "volume": event.get("volume", 0.0),
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_klines(symbol, date_str, df)
            logger.debug(
                "kline_stored",
                symbol=symbol,
                date=date_str,
                interval=event.get("interval"),
            )
        except Exception as e:
            logger.error(
                "kline_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_ticker(self, event: Dict) -> None:
        """
        Store ticker event.
        
        Args:
            event: Ticker event dictionary
        """
        symbol = event.get("symbol")
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "last_price": event.get("last_price", 0.0),
            "bid_price": event.get("bid_price", 0.0),
            "ask_price": event.get("ask_price", 0.0),
            "volume_24h": event.get("volume_24h", 0.0),
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_ticker(symbol, date_str, df)
            logger.debug(
                "ticker_stored",
                symbol=symbol,
                date=date_str,
            )
        except Exception as e:
            logger.error(
                "ticker_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_funding_rate(self, event: Dict) -> None:
        """
        Store funding rate event.
        
        Args:
            event: Funding rate event dictionary
        """
        symbol = event.get("symbol")
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        date_str = timestamp.date().strftime("%Y-%m-%d")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "funding_rate": event.get("funding_rate", 0.0),
            "next_funding_time": event.get("next_funding_time"),
            "internal_timestamp": self._parse_timestamp(event.get("internal_timestamp", timestamp_str)),
            "exchange_timestamp": self._parse_timestamp(event.get("exchange_timestamp", timestamp_str)),
        }])
        
        try:
            await self._parquet_storage.write_funding(symbol, date_str, df)
            logger.debug(
                "funding_rate_stored",
                symbol=symbol,
                date=date_str,
                funding_rate=event.get("funding_rate"),
            )
        except Exception as e:
            logger.error(
                "funding_rate_storage_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                exc_info=True,
            )
    
    async def store_execution_event(self, event: Dict) -> None:
        """
        Store execution event (optional, may use trades storage or separate).
        
        Args:
            event: Execution event dictionary
        """
        # For now, execution events can be stored similar to trades
        # or in a separate execution_events directory
        # This is an optional feature (T070, T071)
        symbol = event.get("symbol")
        timestamp_str = event.get("timestamp") or event.get("internal_timestamp")
        
        if not timestamp_str:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = self._parse_timestamp(timestamp_str)
        
        logger.debug(
            "execution_event_received",
            symbol=symbol,
            timestamp=timestamp_str,
            message="Execution event storage not yet implemented (optional feature)",
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string to datetime.
        
        Args:
            timestamp_str: Timestamp string (ISO format or Unix timestamp)
            
        Returns:
            datetime object
        """
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            try:
                # Try Unix timestamp (milliseconds or seconds)
                ts = float(timestamp_str)
                if ts > 1e10:
                    ts = ts / 1000  # Convert milliseconds to seconds
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError):
                # Fallback to current time
                logger.warning(
                    "timestamp_parse_error",
                    timestamp_str=timestamp_str,
                    message="Failed to parse timestamp, using current time",
                )
                return datetime.now(timezone.utc)
    
    def _is_expired(self, data_date: date) -> bool:
        """
        Check if a date is expired based on retention policy.
        
        Args:
            data_date: Date to check
            
        Returns:
            True if expired, False otherwise
        """
        if self._retention_days == 0:
            # With 0 retention days, all data including today is expired
            return True
        
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).date()
        return data_date < cutoff_date
    
    async def enforce_retention_policy(self) -> None:
        """Enforce data retention policy (archive or delete expired data)."""
        logger.info(
            "enforcing_retention_policy",
            retention_days=self._retention_days,
            archiving_enabled=self._enable_archiving,
        )
        
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).date()
        
        # Scan all data type directories
        data_types = ["orderbook/snapshots", "orderbook/deltas", "trades", "klines", "ticker", "funding"]
        
        expired_dates = set()
        
        for data_type in data_types:
            type_path = self._base_path / data_type
            if not type_path.exists():
                continue
            
            # Scan date directories
            for date_dir in type_path.iterdir():
                if not date_dir.is_dir():
                    continue
                
                try:
                    data_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                    if data_date < cutoff_date:
                        expired_dates.add((data_type, data_date))
                except ValueError:
                    # Skip invalid date directories
                    logger.debug(
                        "invalid_date_directory",
                        path=str(date_dir),
                        message="Skipping invalid date directory",
                    )
                    continue
        
        # Process expired dates
        for data_type, expired_date in expired_dates:
            if self._enable_archiving and self._archive_path:
                await self.archive_expired_data(data_type, expired_date)
            else:
                await self.delete_expired_data(data_type, expired_date)
        
        logger.info(
            "retention_policy_enforced",
            expired_dates_count=len(expired_dates),
        )
    
    async def archive_expired_data(self, data_type: str, expired_date: date) -> None:
        """
        Archive expired data.
        
        Args:
            data_type: Type of data (e.g., "trades", "orderbook/snapshots")
            expired_date: Date of expired data
        """
        type_path = self._base_path / data_type
        date_dir = type_path / expired_date.strftime("%Y-%m-%d")
        
        if not date_dir.exists():
            return
        
        try:
            # Create archive directory structure
            archive_type_dir = self._archive_path / data_type.replace("/", "_")
            archive_type_dir.mkdir(parents=True, exist_ok=True)
            
            # Create archive file name
            archive_file = archive_type_dir / f"{expired_date.strftime('%Y-%m-%d')}.tar.gz"
            
            # Archive the directory (simplified - in production, use proper tar.gz creation)
            # For now, we'll move the directory to archive location
            import shutil
            archive_dest = archive_type_dir / expired_date.strftime("%Y-%m-%d")
            if archive_dest.exists():
                shutil.rmtree(archive_dest)
            shutil.move(str(date_dir), str(archive_dest))
            
            logger.info(
                "data_archived",
                data_type=data_type,
                date=expired_date.strftime("%Y-%m-%d"),
                archive_path=str(archive_dest),
            )
        except Exception as e:
            logger.error(
                "archive_error",
                data_type=data_type,
                date=expired_date.strftime("%Y-%m-%d"),
                error=str(e),
                exc_info=True,
            )
    
    async def delete_expired_data(self, data_type: str, expired_date: date) -> None:
        """
        Delete expired data.
        
        Args:
            data_type: Type of data (e.g., "trades", "orderbook/snapshots")
            expired_date: Date of expired data
        """
        type_path = self._base_path / data_type
        date_dir = type_path / expired_date.strftime("%Y-%m-%d")
        
        if not date_dir.exists():
            return
        
        try:
            import shutil
            shutil.rmtree(date_dir)
            
            logger.info(
                "data_deleted",
                data_type=data_type,
                date=expired_date.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.error(
                "delete_error",
                data_type=data_type,
                date=expired_date.strftime("%Y-%m-%d"),
                error=str(e),
                exc_info=True,
            )
    
    async def recover_from_archive(self, data_type: str, target_date: date) -> bool:
        """
        Recover data from archive.
        
        Args:
            data_type: Type of data to recover
            target_date: Date of data to recover
            
        Returns:
            True if recovery successful, False otherwise
        """
        if not self._archive_path:
            logger.warning("archive_path_not_configured", message="Cannot recover from archive")
            return False
        
        archive_type_dir = self._archive_path / data_type.replace("/", "_")
        archive_dir = archive_type_dir / target_date.strftime("%Y-%m-%d")
        
        if not archive_dir.exists():
            logger.warning(
                "archive_not_found",
                data_type=data_type,
                date=target_date.strftime("%Y-%m-%d"),
            )
            return False
        
        try:
            # Restore to original location
            type_path = self._base_path / data_type
            type_path.mkdir(parents=True, exist_ok=True)
            date_dir = type_path / target_date.strftime("%Y-%m-%d")
            
            import shutil
            if date_dir.exists():
                shutil.rmtree(date_dir)
            shutil.move(str(archive_dir), str(date_dir))
            
            logger.info(
                "data_recovered_from_archive",
                data_type=data_type,
                date=target_date.strftime("%Y-%m-%d"),
            )
            return True
        except Exception as e:
            logger.error(
                "recovery_error",
                data_type=data_type,
                date=target_date.strftime("%Y-%m-%d"),
                error=str(e),
                exc_info=True,
            )
            return False
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup task loop."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                if self._running:
                    await self.enforce_retention_policy()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "cleanup_task_error",
                    error=str(e),
                    exc_info=True,
                )
    
    async def _flush_all_buffers(self) -> None:
        """Flush all buffered events to storage."""
        # This is a placeholder for future batching optimization
        # For now, events are written immediately
        pass
