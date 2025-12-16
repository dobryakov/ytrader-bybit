"""
Main application entry point for Feature Service.

Initializes FastAPI application with basic routing and health check.
"""

import asyncio
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from src.api.health import router as health_router
from src.api.features import router as features_router, set_feature_computer
from src.api.dataset import router as dataset_router, set_metadata_storage, set_dataset_builder, set_target_registry_version_manager as set_target_registry_version_manager_dataset
from src.api.target_registry import router as target_registry_router, set_target_registry_version_manager, set_metadata_storage_for_target_registry
from src.api.backfill import router as backfill_router, set_backfilling_service
from src.api.cache import router as cache_router, set_cache_service
from src.api.feature_registry import (
    router as feature_registry_router,
    set_feature_registry_loader,
    set_feature_registry_version_manager as set_feature_registry_version_manager_api,
    set_metadata_storage_for_registry,
)
from src.api.middleware.auth import verify_api_key
from src.logging import setup_logging, get_logger
from src.config import config

# Service components
from src.mq.connection import MQConnectionManager
from src.http.client import HTTPClient
from src.services.orderbook_manager import OrderbookManager
from src.services.feature_computer import FeatureComputer
from src.services.feature_registry import FeatureRegistryLoader
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.services.target_registry_version_manager import TargetRegistryVersionManager
from src.consumers.market_data_consumer import MarketDataConsumer
from src.publishers.feature_publisher import FeaturePublisher
from src.publishers.dataset_publisher import DatasetPublisher
from src.services.feature_scheduler import FeatureScheduler
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder
from src.services.data_storage import DataStorageService
from src.services.backfilling_service import BackfillingService
from src.services.cache_service import CacheServiceFactory

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
feature_registry_version_manager: FeatureRegistryVersionManager = None
target_registry_version_manager: TargetRegistryVersionManager = None
market_data_consumer: MarketDataConsumer = None
feature_publisher: FeaturePublisher = None
feature_scheduler: FeatureScheduler = None
metadata_storage: MetadataStorage = None
dataset_builder: OptimizedDatasetBuilder = None
data_storage: DataStorageService = None
backfilling_service: BackfillingService = None

# Include routers
app.include_router(health_router)
app.include_router(backfill_router)
app.include_router(features_router)
app.include_router(dataset_router)
app.include_router(feature_registry_router)
app.include_router(target_registry_router)
app.include_router(cache_router)

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
    global feature_registry_loader, feature_registry_version_manager, target_registry_version_manager
    global market_data_consumer, feature_publisher, feature_scheduler
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
        
        # Initialize Metadata Storage (needed for Feature Registry version management)
        metadata_storage = MetadataStorage()
        await metadata_storage.initialize()
        set_metadata_storage(metadata_storage)
        
        # Initialize Feature Registry Version Manager (if database-driven mode enabled)
        registry_config = None
        registry_version = "1.0.0"
        use_db_mode = config.feature_registry_use_db
        
        if use_db_mode:
            try:
                # Initialize FeatureRegistryVersionManager
                feature_registry_version_manager = FeatureRegistryVersionManager(
                    metadata_storage=metadata_storage,
                    versions_dir=config.feature_registry_versions_dir,
                )
                
                # Try to load active version from database
                try:
                    config_data = await feature_registry_version_manager.load_active_version()
                    registry_version = config_data.get("version", "1.0.0")
                    registry_config = config_data  # Store config for later use
                    logger.info(
                        "Feature Registry loaded from database",
                        version=registry_version,
                        mode="database",
                    )
                except FileNotFoundError as e:
                    # No active version in DB - try automatic migration from legacy file
                    if config.feature_registry_auto_migrate:
                        logger.info(
                            "No active version in database, attempting automatic migration from legacy file"
                        )
                        try:
                            migrated_version = await feature_registry_version_manager.migrate_legacy_to_db()
                            config_data = await feature_registry_version_manager.load_active_version()
                            registry_config = config_data
                            registry_version = config_data.get("version", "1.0.0")
                            logger.info(
                                "Feature Registry migrated from legacy file to database",
                                version=registry_version,
                                migrated_version=migrated_version["version"],
                            )
                        except Exception as migration_error:
                            logger.warning(
                                "Automatic migration failed, falling back to file mode",
                                error=str(migration_error),
                            )
                            use_db_mode = False
                            feature_registry_version_manager = None
                    else:
                        logger.warning(
                            "No active version in database and auto_migrate disabled, falling back to file mode"
                        )
                        use_db_mode = False
                        feature_registry_version_manager = None
                
            except Exception as db_error:
                logger.warning(
                    "Failed to initialize database-driven mode, falling back to file mode",
                    error=str(db_error),
                )
                use_db_mode = False
                feature_registry_version_manager = None
        
        # Initialize Feature Registry Loader
        if use_db_mode and feature_registry_version_manager:
            # Database-driven mode
            feature_registry_loader = FeatureRegistryLoader(
                config_path=config.feature_registry_path,
                use_db=True,
                version_manager=feature_registry_version_manager,
            )
            # Config already loaded, just set it
            if registry_config:
                feature_registry_loader.set_config(registry_config)
            else:
                # Fallback: try to load from DB again (async)
                registry_config = await feature_registry_loader.load_async()
                registry_version = registry_config.get("version", "1.0.0")
        else:
            # Legacy file mode
            feature_registry_loader = FeatureRegistryLoader(
                config_path=config.feature_registry_path,
                use_db=False,
            )
            registry_config = feature_registry_loader.load()
            registry_version = registry_config.get("version", "1.0.0")
            logger.info(
                "Feature Registry loaded from file",
                version=registry_version,
                mode="file",
                path=str(config.feature_registry_path),
            )
        
        # Initialize Feature Computer
        feature_computer = FeatureComputer(
            orderbook_manager=orderbook_manager,
            feature_registry_version=registry_version,
            feature_registry_loader=feature_registry_loader,
        )
        
        # Set feature computer for API
        set_feature_computer(feature_computer)
        
        # Set Feature Registry components for API
        set_feature_registry_loader(feature_registry_loader)
        if feature_registry_version_manager:
            set_feature_registry_version_manager_api(feature_registry_version_manager)
        set_metadata_storage_for_registry(metadata_storage)
        
        # Initialize Target Registry Version Manager
        target_registry_version_manager = TargetRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=config.feature_registry_versions_dir,  # Reuse same directory
        )
        set_target_registry_version_manager(target_registry_version_manager)
        set_metadata_storage_for_target_registry(metadata_storage)
        set_target_registry_version_manager_dataset(target_registry_version_manager)
        
        # Set components for hot reload
        from src.api.feature_registry import (
            set_feature_computer_for_registry,
            set_dataset_builder_for_registry,
            set_orderbook_manager_for_registry,
        )
        set_feature_computer_for_registry(feature_computer)
        set_dataset_builder_for_registry(dataset_builder)
        set_orderbook_manager_for_registry(orderbook_manager)
        
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
        
        # Initialize Publishers
        feature_publisher = FeaturePublisher(mq_manager=mq_manager)
        await feature_publisher.initialize()
        
        dataset_publisher = DatasetPublisher(mq_manager=mq_manager)
        await dataset_publisher.initialize()
        
        # Initialize cache service
        cache_service = None
        if config.dataset_builder_cache_enabled:
            try:
                cache_service = await CacheServiceFactory.create(
                    redis_host=config.redis_host,
                    redis_port=config.redis_port,
                    redis_db=config.redis_db,
                    redis_password=config.redis_password,
                    redis_max_connections=config.redis_max_connections,
                    redis_socket_timeout=config.redis_socket_timeout,
                    redis_socket_connect_timeout=config.redis_socket_connect_timeout,
                    cache_redis_enabled=config.cache_redis_enabled,
                    cache_max_size_mb=config.cache_max_size_mb,
                    cache_max_entries=config.cache_max_entries,
                )
                logger.info("Cache service initialized", type=type(cache_service).__name__)
                # Set cache service for API endpoints
                set_cache_service(cache_service)
            except Exception as e:
                logger.warning("Failed to initialize cache service, caching disabled", error=str(e))
                cache_service = None
        
        # Initialize Optimized Dataset Builder
        dataset_builder = OptimizedDatasetBuilder(
            metadata_storage=metadata_storage,
            parquet_storage=parquet_storage,
            dataset_storage_path=config.feature_service_dataset_storage_path,
            cache_service=cache_service,
            feature_registry_loader=feature_registry_loader,
            target_registry_version_manager=target_registry_version_manager,
            dataset_publisher=dataset_publisher,
            batch_size=config.dataset_builder_batch_size,
        )
        set_dataset_builder(dataset_builder)
        
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
        
        # Warmup rolling windows with recent klines (if Bybit client available)
        if symbols and config.feature_service_backfill_enabled:
            try:
                from src.utils.bybit_client import BybitClient
                bybit_client = BybitClient(
                    api_key=config.bybit_api_key,
                    api_secret=config.bybit_api_secret,
                    base_url=config.bybit_rest_base_url,
                )
                await feature_computer.warmup_rolling_windows(
                    symbols=symbols,
                    bybit_client=bybit_client,
                )
                await bybit_client.close()
            except Exception as e:
                logger.warning(
                    "warmup_rolling_windows_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    message="Continuing without warmup - klines will accumulate from WebSocket stream",
                )
        
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

