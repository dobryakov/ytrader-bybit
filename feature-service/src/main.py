"""
Main application entry point for Feature Service.

Initializes FastAPI application with basic routing and health check.
"""

import asyncio
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from src.api.health import router as health_router
from src.api.features import router as features_router, set_feature_computer
from src.api.dataset import router as dataset_router, set_metadata_storage, set_dataset_builder
from src.api.backfill import router as backfill_router, set_backfilling_service
from src.api.middleware.auth import verify_api_key
from src.logging import setup_logging, get_logger
from src.config import config

# Service components
from src.mq.connection import MQConnectionManager
from src.http.client import HTTPClient
from src.services.orderbook_manager import OrderbookManager
from src.services.feature_computer import FeatureComputer
from src.services.feature_registry import FeatureRegistryLoader
from src.consumers.market_data_consumer import MarketDataConsumer
from src.publishers.feature_publisher import FeaturePublisher
from src.services.feature_scheduler import FeatureScheduler
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.services.dataset_builder import DatasetBuilder
from src.services.data_storage import DataStorageService
from src.services.backfilling_service import BackfillingService

# Setup logging
setup_logging(level=config.feature_service_log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Feature Service",
    description="Service for real-time feature computation and dataset building",
    version="0.1.0",
)

# Global service instances
mq_manager: MQConnectionManager = None
http_client: HTTPClient = None
orderbook_manager: OrderbookManager = None
feature_computer: FeatureComputer = None
feature_registry_loader: FeatureRegistryLoader = None
market_data_consumer: MarketDataConsumer = None
feature_publisher: FeaturePublisher = None
feature_scheduler: FeatureScheduler = None
metadata_storage: MetadataStorage = None
dataset_builder: DatasetBuilder = None
data_storage: DataStorageService = None
backfilling_service: BackfillingService = None

# Include routers
app.include_router(health_router)
app.include_router(backfill_router)
app.include_router(features_router)
app.include_router(dataset_router)

# Add authentication middleware to all routes except health
@app.middleware("http")
async def auth_middleware(request, call_next):
    """Authentication middleware."""
    try:
        await verify_api_key(request)
    except Exception:
        # Health endpoint is allowed without auth
        if request.url.path not in ["/health", "/", "/docs", "/openapi.json"]:
            raise
    response = await call_next(request)
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        content={
            "service": "feature-service",
            "version": "0.1.0",
            "status": "running",
        }
    )


@app.on_event("startup")
async def startup():
    """Application startup event."""
    global mq_manager, http_client, orderbook_manager, feature_computer
    global feature_registry_loader, market_data_consumer, feature_publisher, feature_scheduler
    global metadata_storage, dataset_builder, data_storage, backfilling_service
    
    logger.info("Feature Service starting up")
    
    try:
        # Initialize components
        mq_manager = MQConnectionManager()
        http_client = HTTPClient(
            base_url=config.ws_gateway_api_url,
            api_key=config.ws_gateway_api_key,
        )
        orderbook_manager = OrderbookManager()
        
        # Load Feature Registry
        feature_registry_loader = FeatureRegistryLoader(config_path=config.feature_registry_path)
        registry_config = feature_registry_loader.load()
        registry_version = registry_config.get("version", "1.0.0")
        
        # Initialize Feature Computer
        feature_computer = FeatureComputer(
            orderbook_manager=orderbook_manager,
            feature_registry_version=registry_version,
        )
        
        # Set feature computer for API
        set_feature_computer(feature_computer)
        
        # Initialize Metadata Storage
        metadata_storage = MetadataStorage()
        await metadata_storage.initialize()
        set_metadata_storage(metadata_storage)
        
        # Initialize Parquet Storage
        parquet_storage = ParquetStorage(base_path=config.feature_service_raw_data_path)
        
        # Initialize Data Storage Service (T135: Integrate raw data storage)
        data_storage = DataStorageService(
            base_path=config.feature_service_raw_data_path,
            parquet_storage=parquet_storage,
            retention_days=config.feature_service_retention_days,
        )
        await data_storage.start()
        
        # Initialize Backfilling Service (if enabled)
        if config.feature_service_backfill_enabled:
            backfilling_service = BackfillingService(
                parquet_storage=parquet_storage,
                feature_registry_loader=feature_registry_loader,
            )
            set_backfilling_service(backfilling_service)
            logger.info("Backfilling service initialized")
        else:
            backfilling_service = None
        
        # Initialize Dataset Builder
        dataset_builder = DatasetBuilder(
            metadata_storage=metadata_storage,
            parquet_storage=parquet_storage,
            dataset_storage_path=config.feature_service_dataset_storage_path,
            feature_registry_version=registry_version,
            feature_registry_loader=feature_registry_loader,
            backfilling_service=backfilling_service,
        )
        set_dataset_builder(dataset_builder)
        
        # Initialize Publisher
        feature_publisher = FeaturePublisher(mq_manager=mq_manager)
        await feature_publisher.initialize()
        
        # Get symbols from config
        symbols = []
        if config.feature_service_symbols:
            symbols = [s.strip() for s in config.feature_service_symbols.split(',') if s.strip()]
        
        # Initialize Consumer (with data storage integration)
        market_data_consumer = MarketDataConsumer(
            mq_manager=mq_manager,
            http_client=http_client,
            feature_computer=feature_computer,
            orderbook_manager=orderbook_manager,
            data_storage=data_storage,  # T135: Integrate raw data storage
            service_name=config.feature_service_service_name,
            symbols=symbols,
        )
        
        # Initialize Scheduler
        feature_scheduler = FeatureScheduler(
            feature_computer=feature_computer,
            feature_publisher=feature_publisher,
            symbols=symbols,
        )
        
        # Connect to RabbitMQ
        await mq_manager.connect()
        
        # Start consumer and scheduler
        await market_data_consumer.start()
        await feature_scheduler.start()
        
        logger.info(
            "Feature Service started",
            symbols=symbols,
            feature_registry_version=registry_version,
        )
    
    except Exception as e:
        logger.error("Failed to start Feature Service", error=str(e), exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown event."""
    global market_data_consumer, feature_scheduler, mq_manager, data_storage
    
    logger.info("Feature Service shutting down")
    
    try:
        # Stop scheduler
        if feature_scheduler:
            await feature_scheduler.stop()
        
        # Stop consumer (doesn't cancel subscriptions per T069)
        if market_data_consumer:
            await market_data_consumer.stop()
        
        # Stop data storage service
        if data_storage:
            await data_storage.stop()
        
        # Close connections
        if mq_manager:
            await mq_manager.close()
        
        # Close metadata storage
        if metadata_storage:
            await metadata_storage.close()
        
        logger.info("Feature Service shut down complete")
    
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)

