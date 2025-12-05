"""
Parquet storage service for reading and writing market data.
"""
import asyncio
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime, date
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.logging import get_logger

logger = get_logger(__name__)


class ParquetStorage:
    """Service for reading and writing Parquet files."""
    
    def __init__(self, base_path: str):
        """
        Initialize Parquet storage.
        
        Args:
            base_path: Base path for Parquet file storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_orderbook_snapshots_path(self, symbol: str, date_str: str) -> Path:
        """Get path for orderbook snapshots file."""
        return self._base_path / "orderbook" / "snapshots" / date_str / f"{symbol}.parquet"
    
    def _get_orderbook_deltas_path(self, symbol: str, date_str: str) -> Path:
        """Get path for orderbook deltas file."""
        return self._base_path / "orderbook" / "deltas" / date_str / f"{symbol}.parquet"
    
    def _get_trades_path(self, symbol: str, date_str: str) -> Path:
        """Get path for trades file."""
        return self._base_path / "trades" / date_str / f"{symbol}.parquet"
    
    def _get_klines_path(self, symbol: str, date_str: str) -> Path:
        """Get path for klines file."""
        return self._base_path / "klines" / date_str / f"{symbol}.parquet"
    
    def _get_ticker_path(self, symbol: str, date_str: str) -> Path:
        """Get path for ticker file."""
        return self._base_path / "ticker" / date_str / f"{symbol}.parquet"
    
    def _get_funding_path(self, symbol: str, date_str: str) -> Path:
        """Get path for funding rate file."""
        return self._base_path / "funding" / date_str / f"{symbol}.parquet"
    
    async def write_orderbook_snapshots(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write orderbook snapshots to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with orderbook snapshot data
        """
        file_path = self._get_orderbook_snapshots_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run in thread pool to avoid blocking
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written orderbook snapshots to {file_path}")
    
    async def write_orderbook_deltas(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write orderbook deltas to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with orderbook delta data
        """
        file_path = self._get_orderbook_deltas_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written orderbook deltas to {file_path}")
    
    async def write_trades(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write trades to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with trade data
        """
        file_path = self._get_trades_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written trades to {file_path}")
    
    async def write_klines(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write klines to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with kline data
        """
        file_path = self._get_klines_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written klines to {file_path}")
    
    async def write_ticker(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write ticker data to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with ticker data
        """
        file_path = self._get_ticker_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written ticker to {file_path}")
    
    async def write_funding(
        self,
        symbol: str,
        date_str: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Write funding rate data to Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            data: DataFrame with funding rate data
        """
        file_path = self._get_funding_path(symbol, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.to_thread(self._write_parquet, file_path, data)
        logger.debug(f"Written funding rate to {file_path}")
    
    def _write_parquet(self, file_path: Path, data: pd.DataFrame) -> None:
        """Write DataFrame to Parquet file (synchronous). Appends to existing file if it exists."""
        if data.empty:
            logger.warning(f"Attempting to write empty DataFrame to {file_path}")
            return
        
        # Validate data before writing
        if data is None:
            raise ValueError(f"Cannot write None data to {file_path}")
        
        # Log data info for debugging
        logger.debug(
            f"Writing Parquet file {file_path}: {len(data)} rows, columns: {list(data.columns)}"
        )
        
        # If file exists, read existing data and append
        if file_path.exists():
            # Check file size - Parquet files should be at least 8 bytes (minimum footer size)
            file_size = file_path.stat().st_size
            if file_size < 8:
                logger.warning(
                    f"Existing file {file_path} is too small ({file_size} bytes), will overwrite"
                )
                file_path.unlink()  # Delete corrupted file
            else:
                try:
                    existing_table = pq.read_table(file_path)
                    existing_df = existing_table.to_pandas()
                    # Combine data
                    combined_df = pd.concat([existing_df, data], ignore_index=True)
                    
                    # Determine duplicate key based on available columns
                    # For orderbook deltas, use sequence if available, otherwise timestamp
                    if 'sequence' in combined_df.columns:
                        # Use sequence for orderbook deltas (more unique)
                        combined_df = combined_df.drop_duplicates(subset=['sequence'], keep='last')
                    else:
                        # Use timestamp for other data types
                        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    
                    # Sort by timestamp
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    data = combined_df
                except Exception as e:
                    logger.warning(f"Error reading existing file {file_path}, will overwrite: {e}")
                    # Delete corrupted file before overwriting
                    try:
                        file_path.unlink()
                    except Exception as delete_error:
                        logger.warning(f"Failed to delete corrupted file {file_path}: {delete_error}")
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert DataFrame to PyArrow table
        try:
            # Clean data: replace inf with NaN, then drop rows with all NaN
            data_clean = data.replace([float('inf'), float('-inf')], None)
            
            table = pa.Table.from_pandas(data_clean)
            logger.debug(
                f"Converted DataFrame to PyArrow table for {file_path}: "
                f"{len(table)} rows, {len(table.column_names)} columns"
            )
        except Exception as table_error:
            raise IOError(f"Failed to convert DataFrame to PyArrow table: {file_path}: {table_error}") from table_error
        
        # Write directly to final file (we already handled corrupted files above)
        # Using direct write is simpler and more reliable than temp file + move
        file_path_str = str(file_path)
        try:
            # Write table to Parquet file
            pq.write_table(table, file_path_str, compression='snappy')
            
            # Small delay to ensure filesystem sync
            import time
            time.sleep(0.01)
            
            logger.debug(f"Successfully wrote Parquet file {file_path_str}")
        except Exception as write_error:
            raise IOError(f"Failed to write Parquet file {file_path_str}: {write_error}") from write_error
        
        # Verify file was created and has content
        if not file_path.exists():
            raise IOError(
                f"File {file_path} was not created after write. "
                f"Table has {len(table)} rows, {len(table.column_names)} columns."
            )
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise IOError(
                f"File {file_path} is empty after write. "
                f"Table has {len(table)} rows, {len(table.column_names)} columns."
            )
    
    async def read_orderbook_snapshots(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read orderbook snapshots from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with orderbook snapshot data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_orderbook_snapshots_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def read_orderbook_deltas(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read orderbook deltas from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with orderbook delta data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_orderbook_deltas_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def read_trades(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read trades from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with trade data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_trades_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def read_klines(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read klines from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with kline data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_klines_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def read_ticker(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read ticker data from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with ticker data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_ticker_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def read_funding(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame:
        """
        Read funding rate data from Parquet file.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with funding rate data
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._get_funding_path(symbol, date_str)
        return await self._read_parquet(file_path)
    
    async def _read_parquet(self, file_path: Path) -> pd.DataFrame:
        """Read Parquet file to DataFrame (async)."""
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        # Run in thread pool to avoid blocking
        return await asyncio.to_thread(self._read_parquet_sync, file_path)
    
    def _read_parquet_sync(self, file_path: Path) -> pd.DataFrame:
        """Read Parquet file to DataFrame (synchronous)."""
        table = pq.read_table(file_path)
        return table.to_pandas()
    
    async def read_trades_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Read trades for a date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with trades for the date range
        """
        all_trades = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                trades = await self.read_trades(symbol, date_str)
                if not trades.empty:
                    all_trades.append(trades)
            except FileNotFoundError:
                logger.debug(f"No trades file for {symbol} on {date_str}")
            
            # Move to next day
            from datetime import timedelta
            current_date += timedelta(days=1)
        
        if not all_trades:
            return pd.DataFrame()
        
        return pd.concat(all_trades, ignore_index=True).sort_values("timestamp")
    
    async def read_klines_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Read klines for a date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with klines for the date range
        """
        all_klines = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                klines = await self.read_klines(symbol, date_str)
                if not klines.empty:
                    all_klines.append(klines)
            except FileNotFoundError:
                logger.debug(f"No klines file for {symbol} on {date_str}")
            
            # Move to next day
            from datetime import timedelta
            current_date += timedelta(days=1)
        
        if not all_klines:
            return pd.DataFrame()
        
        return pd.concat(all_klines, ignore_index=True).sort_values("timestamp")
    
    async def read_orderbook_snapshots_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Read orderbook snapshots for a date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with orderbook snapshots for the date range
        """
        all_snapshots = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                snapshots = await self.read_orderbook_snapshots(symbol, date_str)
                if not snapshots.empty:
                    all_snapshots.append(snapshots)
            except FileNotFoundError:
                logger.debug(f"No orderbook snapshots file for {symbol} on {date_str}")
            
            # Move to next day
            from datetime import timedelta
            current_date += timedelta(days=1)
        
        if not all_snapshots:
            return pd.DataFrame()
        
        return pd.concat(all_snapshots, ignore_index=True).sort_values("timestamp")
    
    async def read_orderbook_deltas_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Read orderbook deltas for a date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with orderbook deltas for the date range
        """
        all_deltas = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                deltas = await self.read_orderbook_deltas(symbol, date_str)
                if not deltas.empty:
                    all_deltas.append(deltas)
            except FileNotFoundError:
                logger.debug(f"No orderbook deltas file for {symbol} on {date_str}")
            
            # Move to next day
            from datetime import timedelta
            current_date += timedelta(days=1)
        
        if not all_deltas:
            return pd.DataFrame()
        
        return pd.concat(all_deltas, ignore_index=True).sort_values("timestamp")
