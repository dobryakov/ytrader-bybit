#!/usr/bin/env python3
"""Script to run backfilling directly."""
import asyncio
import sys
from pathlib import Path
from datetime import date, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.backfilling_service import BackfillingService
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.utils.bybit_client import BybitClient
from src.config import config

async def main():
    """Run backfilling for missing dates."""
    symbol = "BTCUSDT"
    start_date = date(2025, 12, 3)
    end_date = date(2025, 12, 6)
    
    print(f"Starting backfilling for {symbol} from {start_date} to {end_date}")
    
    # Initialize services
    parquet_storage = ParquetStorage(base_path=config.feature_service_raw_data_path)
    feature_registry_loader = FeatureRegistryLoader(config_path=config.feature_registry_path)
    feature_registry_loader.load()
    
    bybit_client = BybitClient(
        api_key=config.bybit_api_key,
        api_secret=config.bybit_api_secret,
        base_url=config.bybit_rest_base_url,
    )
    
    backfilling_service = BackfillingService(
        parquet_storage=parquet_storage,
        feature_registry_loader=feature_registry_loader,
        bybit_client=bybit_client,
    )
    
    # Run backfilling
    job_id = await backfilling_service.backfill_historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_types=["klines", "ticker"],  # Only critical types
    )
    
    print(f"Backfilling job started: {job_id}")
    
    # Wait for completion
    import time
    max_wait = 300  # 5 minutes
    waited = 0
    while waited < max_wait:
        status = backfilling_service.get_job_status(job_id)
        if status:
            print(f"Status: {status['status']}, Progress: {status['progress']}")
            if status['status'] in ['completed', 'failed']:
                print(f"Job {status['status']}: {status.get('error_message', 'No errors')}")
                print(f"Completed dates: {status.get('completed_dates', [])}")
                print(f"Failed dates: {status.get('failed_dates', [])}")
                break
        await asyncio.sleep(5)
        waited += 5
    
    await bybit_client.close()

if __name__ == "__main__":
    asyncio.run(main())

