#!/usr/bin/env python3
"""
Script to check data quality and availability for feature computation.

Checks:
1. Data availability for last 10 days
2. Data completeness (all required types present)
3. Data validity (format, structure)
4. No duplicates
5. Suitability for dataset building
"""
# Import standard library modules first (before adding src to path)
import asyncio
import sys
from pathlib import Path

# Save original path
_original_path = sys.path[:]

# Add parent directory to path
_app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_app_dir))

# Now import src modules
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.config import Config
from src.logging import get_logger

# Restore original path to avoid conflicts
sys.path[:] = _original_path
sys.path.insert(0, str(_app_dir))

# Import other standard library modules
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Set, Optional
import pandas as pd

logger = get_logger(__name__)


class DataQualityChecker:
    """Check data quality and availability."""
    
    def __init__(self, base_path: str, feature_registry_path: str, symbol: str = "BTCUSDT"):
        self._parquet_storage = ParquetStorage(base_path=base_path)
        self._feature_registry_loader = FeatureRegistryLoader(config_path=feature_registry_path)
        self._config = self._feature_registry_loader.load()
        self._required_types = self._feature_registry_loader.get_required_data_types()
        self._data_type_mapping = self._feature_registry_loader.get_data_type_mapping()
        self._symbol = symbol
        
        # Map input sources to storage types
        self._storage_types_needed = set()
        for input_source in self._required_types:
            if input_source in self._data_type_mapping:
                self._storage_types_needed.update(self._data_type_mapping[input_source])
        
        # Categorize data types by availability constraints
        # Critical: must be available via REST API backfilling
        # These are essential for feature computation and can be backfilled
        self._critical_types = {"klines", "ticker"}
        
        # Optional for historical: may not be available via REST API for old dates
        # - orderbook_deltas/snapshots: only via WebSocket (real-time), not available via REST API
        # - trades: limited via REST API (only recent trades, max 60 for spot)
        # These are collected in real-time via WebSocket, not backfilled
        self._optional_historical_types = {"orderbook_deltas", "orderbook_snapshots", "trades"}
        
        # Conditional: depends on symbol type
        # - funding: only for perpetual contracts, not spot
        self._conditional_types = {"funding"}
        
        # Check if symbol is spot (ends with USDT, ETH, BTC, etc. but not perpetual)
        # For spot symbols, funding is not available
        self._is_spot_symbol = self._is_spot_symbol_check(symbol)
        
        logger.info(
            "data_quality_checker_initialized",
            required_input_sources=sorted(self._required_types),
            storage_types_needed=sorted(self._storage_types_needed),
            is_spot_symbol=self._is_spot_symbol,
        )
    
    def _is_spot_symbol_check(self, symbol: str) -> bool:
        """Check if symbol is a spot trading pair (not perpetual)."""
        # Spot symbols typically don't have funding rates
        # Perpetual symbols might have different naming, but for BTCUSDT, ETHUSDT - these are spot
        # This is a heuristic - in production, might need to check exchange API
        return True  # Assume spot for now, can be enhanced
    
    def _is_type_required_for_date(self, storage_type: str, check_date: date) -> tuple[bool, str]:
        """
        Check if data type is required for a specific date.
        
        Returns:
            (is_required, reason) tuple
        """
        # Critical types are always required (can be backfilled via REST API)
        if storage_type in self._critical_types:
            return (True, "critical")
        
        # Conditional types
        if storage_type == "funding":
            if self._is_spot_symbol:
                return (False, "funding_not_available_for_spot_symbols")
            return (True, "conditional")
        
        # Optional historical types - always optional, never critical
        # These are collected via WebSocket in real-time, not via REST API backfilling
        if storage_type in self._optional_historical_types:
            days_ago = (date.today() - check_date).days
            if days_ago <= 2:
                # For recent dates, data should be available if WebSocket is working
                # But it's still optional - absence doesn't block dataset building
                return (False, "optional_expected_for_recent_dates_if_websocket_active")
            else:
                return (False, "optional_for_historical_dates_websocket_only")
        
        # Unknown type - treat as required (conservative approach)
        return (True, "unknown")
    
    async def check_data_availability(
        self,
        symbol: str,
        days: int = 10
    ) -> Dict[str, any]:
        """
        Check data availability for last N days.
        
        Returns:
            Dict with availability status for each date and data type
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        
        results = {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_checked": days,
            "required_types": sorted(self._storage_types_needed),
            "dates": {},
            "summary": {
                "total_dates": 0,
                "dates_with_all_data": 0,
                "dates_with_missing_data": 0,
                "missing_data_by_type": {},
            }
        }
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            days_ago = (date.today() - current_date).days
            date_result = {
                "date": date_str,
                "days_ago": days_ago,
                "data_types": {},
                "has_all_required": True,
                "missing_critical": [],
                "missing_optional": [],
            }
            
            for storage_type in self._storage_types_needed:
                is_required, reason = self._is_type_required_for_date(storage_type, current_date)
                try:
                    # Try to read data
                    if storage_type == "klines":
                        data = await self._parquet_storage.read_klines(symbol, date_str)
                    elif storage_type == "trades":
                        data = await self._parquet_storage.read_trades(symbol, date_str)
                    elif storage_type == "orderbook_snapshots":
                        data = await self._parquet_storage.read_orderbook_snapshots(symbol, date_str)
                    elif storage_type == "orderbook_deltas":
                        data = await self._parquet_storage.read_orderbook_deltas(symbol, date_str)
                    elif storage_type == "ticker":
                        data = await self._parquet_storage.read_ticker(symbol, date_str)
                    elif storage_type == "funding":
                        data = await self._parquet_storage.read_funding(symbol, date_str)
                    else:
                        date_result["data_types"][storage_type] = {
                            "available": False,
                            "error": f"Unknown storage type: {storage_type}",
                        }
                        date_result["has_all_required"] = False
                        date_result["missing_types"].append(storage_type)
                        continue
                    
                    # Check if data exists and is not empty
                    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                        date_result["data_types"][storage_type] = {
                            "available": False,
                            "error": "Data file exists but is empty",
                            "required": is_required,
                            "reason": reason,
                        }
                        if is_required:
                            date_result["has_all_required"] = False
                            date_result["missing_critical"].append(storage_type)
                        else:
                            date_result["missing_optional"].append(storage_type)
                    else:
                        date_result["data_types"][storage_type] = {
                            "available": True,
                            "record_count": len(data) if isinstance(data, pd.DataFrame) else 0,
                            "required": is_required,
                            "reason": reason,
                        }
                
                except FileNotFoundError:
                    date_result["data_types"][storage_type] = {
                        "available": False,
                        "error": "File not found",
                        "required": is_required,
                        "reason": reason,
                    }
                    if is_required:
                        date_result["has_all_required"] = False
                        date_result["missing_critical"].append(storage_type)
                    else:
                        date_result["missing_optional"].append(storage_type)
                except Exception as e:
                    date_result["data_types"][storage_type] = {
                        "available": False,
                        "error": str(e),
                        "required": is_required,
                        "reason": reason,
                    }
                    if is_required:
                        date_result["has_all_required"] = False
                        date_result["missing_critical"].append(storage_type)
                    else:
                        date_result["missing_optional"].append(storage_type)
            
            results["dates"][date_str] = date_result
            results["summary"]["total_dates"] += 1
            
            if date_result["has_all_required"]:
                results["summary"]["dates_with_all_data"] += 1
            else:
                results["summary"]["dates_with_missing_data"] += 1
                # Only count critical missing types in summary
                for missing_type in date_result["missing_critical"]:
                    if missing_type not in results["summary"]["missing_data_by_type"]:
                        results["summary"]["missing_data_by_type"][missing_type] = 0
                    results["summary"]["missing_data_by_type"][missing_type] += 1
            
            current_date += timedelta(days=1)
        
        return results
    
    async def check_data_quality(
        self,
        symbol: str,
        date_str: str,
        data_type: str
    ) -> Dict[str, any]:
        """
        Check data quality for specific date and type.
        
        Checks:
        - Required columns present
        - Timestamp format
        - No duplicates
        - Data types correct
        """
        try:
            # Read data
            if data_type == "klines":
                data = await self._parquet_storage.read_klines(symbol, date_str)
                required_columns = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
            elif data_type == "trades":
                data = await self._parquet_storage.read_trades(symbol, date_str)
                required_columns = {"timestamp", "price", "quantity", "side", "symbol"}
            elif data_type == "orderbook_snapshots":
                data = await self._parquet_storage.read_orderbook_snapshots(symbol, date_str)
                required_columns = {"timestamp", "bids", "asks", "sequence", "symbol"}
            elif data_type == "orderbook_deltas":
                data = await self._parquet_storage.read_orderbook_deltas(symbol, date_str)
                required_columns = {"timestamp", "sequence", "symbol"}
            elif data_type == "ticker":
                data = await self._parquet_storage.read_ticker(symbol, date_str)
                required_columns = {"timestamp", "last_price", "symbol"}
            elif data_type == "funding":
                data = await self._parquet_storage.read_funding(symbol, date_str)
                required_columns = {"timestamp", "funding_rate", "symbol"}
            else:
                return {
                    "valid": False,
                    "error": f"Unknown data type: {data_type}",
                }
            
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                return {
                    "valid": False,
                    "error": "Data is empty",
                }
            
            result = {
                "valid": True,
                "record_count": len(data),
                "columns": list(data.columns),
                "missing_columns": [],
                "duplicates": 0,
                "timestamp_issues": [],
                "data_type_issues": [],
            }
            
            # Check required columns
            missing_cols = required_columns - set(data.columns)
            if missing_cols:
                result["valid"] = False
                result["missing_columns"] = list(missing_cols)
            
            # Check timestamp column
            if "timestamp" in data.columns:
                # Check if timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                    result["valid"] = False
                    result["timestamp_issues"].append("Timestamp column is not datetime type")
                
                # Check for duplicates by timestamp
                duplicates = data.duplicated(subset=["timestamp"], keep=False)
                if duplicates.any():
                    result["duplicates"] = int(duplicates.sum())
                    result["valid"] = False
                    result["timestamp_issues"].append(f"Found {result['duplicates']} duplicate timestamps")
                
                # Check timestamp range (should be within the date)
                date_obj = datetime.fromisoformat(date_str).date()
                timestamps = pd.to_datetime(data["timestamp"])
                min_ts = timestamps.min()
                max_ts = timestamps.max()
                
                if min_ts.date() < date_obj or max_ts.date() > date_obj:
                    result["timestamp_issues"].append(
                        f"Timestamps out of range: {min_ts.date()} to {max_ts.date()}, expected {date_obj}"
                    )
            else:
                result["valid"] = False
                result["timestamp_issues"].append("Timestamp column missing")
            
            # Check data types for numeric columns
            numeric_columns = ["open", "high", "low", "close", "volume", "price", "quantity", "last_price", "funding_rate"]
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        result["data_type_issues"].append(f"Column {col} is not numeric")
            
            return result
        
        except FileNotFoundError:
            return {
                "valid": False,
                "error": "File not found",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }
    
    async def check_dataset_readiness(
        self,
        symbol: str,
        days: int = 10
    ) -> Dict[str, any]:
        """
        Comprehensive check for dataset building readiness.
        
        Returns:
            Dict with overall readiness status and detailed results
        """
        logger.info("Starting comprehensive data quality check", symbol=symbol, days=days)
        
        # Update symbol if changed
        if symbol != self._symbol:
            self._symbol = symbol
            self._is_spot_symbol = self._is_spot_symbol_check(symbol)
        
        # Check availability
        availability = await self.check_data_availability(symbol, days)
        
        # Check quality for available data
        quality_results = {}
        for date_str, date_info in availability["dates"].items():
            quality_results[date_str] = {}
            for data_type in self._storage_types_needed:
                if date_info["data_types"].get(data_type, {}).get("available", False):
                    quality_results[date_str][data_type] = await self.check_data_quality(
                        symbol, date_str, data_type
                    )
        
        # Overall assessment - only check critical types
        # A date is ready if all critical types are available
        dates_with_critical_data = sum(
            1
            for date_info in availability["dates"].values()
            if len(date_info.get("missing_critical", [])) == 0
        )
        all_critical_dates_ready = dates_with_critical_data == availability["summary"]["total_dates"]
        
        all_data_valid = all(
            quality.get("valid", False)
            for date_qualities in quality_results.values()
            for quality in date_qualities.values()
        )
        
        readiness = {
            "ready_for_dataset": all_critical_dates_ready and all_data_valid,
            "availability": availability,
            "quality": quality_results,
            "recommendations": [],
            "summary": {
                "dates_with_critical_data": dates_with_critical_data,
                "dates_with_missing_critical": availability["summary"]["total_dates"] - dates_with_critical_data,
            }
        }
        
        # Generate recommendations
        if not all_critical_dates_ready:
            missing_critical_count = availability["summary"]["total_dates"] - dates_with_critical_data
            readiness["recommendations"].append(
                f"Missing critical data (klines, ticker) for {missing_critical_count} out of {availability['summary']['total_dates']} dates. "
                f"Run backfilling to fill missing data."
            )
        
        # Check optional data availability
        optional_missing = {}
        for date_info in availability["dates"].values():
            for missing_type in date_info.get("missing_optional", []):
                if missing_type not in optional_missing:
                    optional_missing[missing_type] = 0
                optional_missing[missing_type] += 1
        
        if optional_missing:
            missing_list = ", ".join([f"{k}: {v} dates" for k, v in sorted(optional_missing.items())])
            readiness["recommendations"].append(
                f"Optional data missing (acceptable for historical dates): {missing_list}. "
                f"These types are only available via WebSocket (orderbook) or have limited REST API access (trades)."
            )
        
        if self._is_spot_symbol and "funding" in self._storage_types_needed:
            readiness["recommendations"].append(
                f"Funding rates are not available for spot symbols like {symbol}. "
                f"This is expected and will not affect dataset building for spot trading."
            )
        
        if not all_data_valid:
            invalid_count = sum(
                1
                for date_qualities in quality_results.values()
                for quality in date_qualities.values()
                if not quality.get("valid", False)
            )
            readiness["recommendations"].append(
                f"Found {invalid_count} data quality issues. Review quality results for details."
            )
        
        if all_critical_dates_ready and all_data_valid:
            readiness["recommendations"].append("All critical data is ready for dataset building!")
        
        return readiness


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check data quality and availability")
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to check (default: 10)"
    )
    parser.add_argument(
        "--base-path",
        default="/data/raw",
        help="Base path for Parquet storage (default: /data/raw)"
    )
    parser.add_argument(
        "--feature-registry",
        default="/app/config/feature_registry.yaml",
        help="Path to Feature Registry config (default: /app/config/feature_registry.yaml)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    checker = DataQualityChecker(
        base_path=args.base_path,
        feature_registry_path=args.feature_registry,
        symbol=args.symbol
    )
    
    readiness = await checker.check_dataset_readiness(symbol=args.symbol, days=args.days)
    
    if args.output_format == "json":
        import json
        print(json.dumps(readiness, indent=2, default=str))
    else:
        # Text output
        print("=" * 80)
        print(f"Data Quality Check for {args.symbol}")
        print("=" * 80)
        print(f"\nPeriod: {readiness['availability']['start_date']} to {readiness['availability']['end_date']}")
        print(f"Required data types: {', '.join(sorted(checker._storage_types_needed))}")
        print(f"\nSummary:")
        print(f"  Total dates checked: {readiness['availability']['summary']['total_dates']}")
        print(f"  Dates with all critical data: {readiness['summary']['dates_with_critical_data']}")
        print(f"  Dates with missing critical data: {readiness['summary']['dates_with_missing_critical']}")
        print(f"  Dates with all data (including optional): {readiness['availability']['summary']['dates_with_all_data']}")
        
        if readiness['availability']['summary']['missing_data_by_type']:
            print(f"\nMissing critical data by type:")
            for data_type, count in sorted(readiness['availability']['summary']['missing_data_by_type'].items()):
                print(f"  {data_type}: {count} dates")
        
        # Show optional missing data
        optional_missing = {}
        for date_info in readiness['availability']['dates'].values():
            for missing_type in date_info.get("missing_optional", []):
                if missing_type not in optional_missing:
                    optional_missing[missing_type] = 0
                optional_missing[missing_type] += 1
        
        if optional_missing:
            print(f"\nMissing optional data (acceptable for historical dates):")
            for data_type, count in sorted(optional_missing.items()):
                print(f"  {data_type}: {count} dates")
        
        print(f"\nDataset Readiness: {'✓ READY' if readiness['ready_for_dataset'] else '✗ NOT READY'}")
        if readiness['ready_for_dataset']:
            print("  (All critical data types available)")
        else:
            print("  (Missing critical data types: klines, ticker)")
        
        if readiness['recommendations']:
            print(f"\nRecommendations:")
            for rec in readiness['recommendations']:
                print(f"  - {rec}")
        
        # Show quality issues
        quality_issues = []
        for date_str, date_qualities in readiness['quality'].items():
            for data_type, quality in date_qualities.items():
                if not quality.get("valid", False):
                    quality_issues.append((date_str, data_type, quality))
        
        # Show dates with missing critical data
        missing_critical_dates = []
        for date_str, date_info in readiness['availability']['dates'].items():
            if date_info.get("missing_critical"):
                missing_critical_dates.append((date_str, date_info.get("missing_critical", [])))
        
        if missing_critical_dates:
            print(f"\nDates with missing critical data:")
            for date_str, missing_types in missing_critical_dates:
                print(f"  {date_str}: missing {', '.join(missing_types)}")
        
        if quality_issues:
            print(f"\nQuality Issues:")
            for date_str, data_type, quality in quality_issues:
                print(f"  {date_str} / {data_type}:")
                if "error" in quality:
                    print(f"    Error: {quality['error']}")
                if quality.get("missing_columns"):
                    print(f"    Missing columns: {', '.join(quality['missing_columns'])}")
                if quality.get("duplicates", 0) > 0:
                    print(f"    Duplicates: {quality['duplicates']}")
                if quality.get("timestamp_issues"):
                    for issue in quality["timestamp_issues"]:
                        print(f"    {issue}")
        
        print("\n" + "=" * 80)
    
    # Exit with error code if not ready
    sys.exit(0 if readiness['ready_for_dataset'] else 1)


if __name__ == "__main__":
    asyncio.run(main())

