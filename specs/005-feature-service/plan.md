# Implementation Plan: Feature Service

**Branch**: `005-feature-service` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-feature-service/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

A microservice that receives market data streams from ws-gateway via RabbitMQ queues, computes real-time features (price, orderflow, orderbook, perpetual, temporal, position features) with latency ≤ 70ms, stores raw market data in Parquet format for 90+ days, and rebuilds features from historical data for model training datasets with explicit train/validation/test splits. The service provides REST API for feature retrieval, dataset building requests, Feature Registry management, and data quality reports. Built with Python using async message queue consumers, FastAPI for REST endpoints, time-series processing libraries for feature computation, and Parquet storage for raw data persistence. Ensures feature identity between online and offline computation modes using shared Feature Registry configuration.

## Technical Context

**Language/Version**: Python 3.11+ (async/await support for message queue consumers and REST API)  
**Primary Dependencies**: 
- REST API framework: `FastAPI>=0.104.0` (async support, OpenAPI generation)
- Message queue: `aio-pika>=9.0.0` for RabbitMQ (async client for consuming market data streams)
- Database: `asyncpg>=0.29.0` (fastest async PostgreSQL driver for metadata storage)
- HTTP client: `httpx>=0.25.0` for Position Manager REST API calls (fallback)
- Configuration: `pydantic-settings>=2.0.0` for environment variable management
- Logging: `structlog>=23.2.0` for structured logging with trace IDs
- Time series processing: `pandas>=2.0.0` (standard library for time series, rolling windows, see research.md)
- Parquet storage: `pyarrow>=14.0.0` (official Apache Arrow implementation, best performance, see research.md)
- Feature computation: `pandas>=2.0.0` + `numpy>=1.24.0` + `scipy>=1.10.0` (rolling aggregations, VWAP, volatility calculations, see research.md)
- Orderbook reconstruction: `sortedcontainers>=2.4.0` (efficient sorted data structures for in-memory orderbook state, snapshot + delta reconstruction algorithm, see research.md)

**Storage**: 
- PostgreSQL (shared database for metadata: dataset metadata, Feature Registry version, data quality reports)
- Local filesystem (mounted volumes) for raw Parquet data files organized by data type and date
- In-memory state: orderbook state, rolling windows (1s, 3s, 15s, 1m), position cache with TTL ≤ 30s

**Testing**: `pytest` with `pytest-asyncio` for async tests, `pytest-mock` for mocking, derivation tests for feature identity (online vs offline comparison using identical input data, see research.md)

**Target Platform**: Linux server (Docker container)

**Project Type**: single (microservice)

**Performance Goals**: 
- Feature computation latency: ≤ 70ms from market data update to feature availability (95th percentile)
- Dataset building: complete within 2 hours for 1 month of historical data for a single symbol
- Data quality report generation: ≤ 5 seconds for any 24-hour period
- Position update processing: ≤ 100ms from receiving position event to feature vector update

**Constraints**: 
- Feature identity: 100% match between online and offline feature computation (same code and parameters)
- Data leakage prevention: features use only data before time t, targets use only data after time t
- Raw data retention: minimum 90 days before archiving/deletion
- Orderbook desynchronization handling: detect and restore within 1 second
- Position data fallback: use default values (0 for most features, current price for entry_price) when Position Manager unavailable

**Scale/Scope**: 
- Multiple symbols processed concurrently (horizontal scaling by symbol)
- Real-time feature computation at intervals: 1s, 3s, 15s, 1m
- Raw data storage: 90+ days of market data (orderbook snapshots/deltas, trades, klines, ticker, funding) in Parquet format
- Dataset building: support walk-forward validation with configurable windows and time-based splits
- Feature Registry: YAML/JSON configuration with validation for temporal boundaries and data leakage prevention

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- **Status**: COMPLIANT
- **Check**: This is a new independent microservice with single responsibility (feature computation, raw data storage, dataset building)
- **Deployment**: Will be added to `docker-compose.yml` as a separate container
- **Scalability**: Service is independently deployable, testable, and supports horizontal scaling by symbol

### II. Shared Database Strategy ✅
- **Status**: COMPLIANT
- **Check**: Uses shared PostgreSQL database for metadata (dataset metadata, Feature Registry version, data quality reports)
- **Migrations**: Database schema changes will be managed through reversible migrations
- **Migration Ownership**: PostgreSQL migrations will be located in `ws-gateway/migrations/` per constitution (ws-gateway is designated owner of all PostgreSQL migrations)
- **Justification**: Local filesystem (mounted volumes) used for raw Parquet data files is not a database - it's file storage for large time-series data, which is appropriate for this use case

### III. Inter-Service Communication ✅
- **Status**: COMPLIANT
- **Check**: 
  - Event-driven: Consumes market data from RabbitMQ queues (`ws-gateway.*` queues), publishes computed features to `features.live` queue, publishes dataset completion notifications to `features.dataset.ready` queue
  - REST API: Used for feature retrieval, dataset building requests, Feature Registry management, data quality reports
  - Logging: All inter-service communication will be logged with full request/response bodies and trace IDs

### IV. Testing Discipline ✅
- **Status**: COMPLIANT
- **Check**: 
  - Tests will run inside Docker containers (unit tests in service container, API/e2e in test containers)
  - Test containers connected to main `docker-compose.yml`
  - Automated tests will be executed after each task completion
  - Using `pytest` with `pytest-asyncio` for async testing
  - Feature identity tests (online vs offline comparison) will be included

### V. Observability & Diagnostics ✅
- **Status**: COMPLIANT
- **Check**: 
  - Structured logging for incoming market data events, feature computation, REST API requests, dataset building progress
  - Health checks for service health and data quality metrics
  - Trace IDs for request flow tracking
  - All errors logged with sufficient context (missing data, desynchronization events, data leakage violations)

### VI. Infrastructure Standards ✅
- **Status**: COMPLIANT
- **Check**: 
  - Docker Compose V2 (`docker compose`) will be used
  - No `version:` field in `docker-compose.yml`
  - Commands run inside containers
  - Non-standard ports with explicit configuration
  - Base images reused when possible

### VII. Documentation & Configuration ✅
- **Status**: COMPLIANT
- **Check**: 
  - Configuration variables in `.env` with samples in `env.example`
  - README, quickstart, and specifications will be synchronized after changes
  - Convenient examples using `curl` and shell scripts will be provided
  - Documentation for deploying from scratch (container builds, migrations, data generation) will be maintained

## Constitution Check - Post Phase 1 Re-evaluation

*Re-checked after Phase 1 design completion (data-model.md, contracts/, quickstart.md created)*

### I. Microservices Architecture ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: Data model confirms single responsibility (feature computation, raw data storage, dataset building). Project structure defined as single microservice with clear module separation.

### II. Shared Database Strategy ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: Data model confirms use of shared PostgreSQL for metadata (datasets, feature_registry_versions, data_quality_reports). Raw Parquet files stored on local filesystem (not a database). Migration ownership confirmed: all PostgreSQL migrations in `ws-gateway/migrations/`.

### III. Inter-Service Communication ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: API contracts confirm event-driven communication (consumes from `ws-gateway.*` queues, publishes to `features.live` and `features.dataset.ready`). REST API endpoints defined for feature retrieval, dataset building, Feature Registry management, data quality reports. All endpoints include trace_id for logging.

### IV. Testing Discipline ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: Research confirms feature identity testing strategy (online vs offline comparison). Project structure includes test directories (unit, integration, contract). Quickstart includes test execution examples.

### V. Observability & Diagnostics ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: API contracts include trace_id in responses. Data model includes trace_id fields. Health check endpoint defined. Data quality reporting API included.

### VI. Infrastructure Standards ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: Quickstart confirms Docker Compose V2 usage. Non-standard port (4500) configured. Volume mounts for Parquet data storage defined.

### VII. Documentation & Configuration ✅
- **Status**: COMPLIANT (unchanged)
- **Post-Phase 1 Check**: Quickstart created with curl examples and shell scripts. Data model documented. API contracts (OpenAPI) created. All documentation synchronized.

**Overall Status**: All gates remain COMPLIANT after Phase 1 design. No violations detected. Ready to proceed to Phase 2 (task breakdown).

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
feature-service/
├── src/
│   ├── api/                    # REST API endpoints (FastAPI routes)
│   │   ├── features.py         # GET /features/latest
│   │   ├── dataset.py          # POST /dataset/build, GET /dataset/list, GET /dataset/{id}
│   │   ├── feature_registry.py # GET /feature-registry, POST /feature-registry/reload
│   │   └── data_quality.py     # GET /data-quality/report
│   ├── services/
│   │   ├── feature_computer.py # Online feature computation engine
│   │   ├── offline_engine.py   # Offline feature computation for dataset building
│   │   ├── dataset_builder.py  # Dataset building with train/val/test splits
│   │   ├── orderbook_manager.py # Orderbook state management (snapshot + delta)
│   │   ├── data_storage.py     # Parquet file storage and retrieval
│   │   ├── feature_registry.py # Feature Registry configuration management
│   │   ├── position_integration.py # Position Manager integration (events + REST fallback)
│   │   └── data_quality.py     # Data quality monitoring and reporting
│   ├── models/
│   │   ├── feature_vector.py   # Feature vector data models
│   │   ├── dataset.py          # Dataset metadata models
│   │   ├── feature_registry.py # Feature Registry configuration models
│   │   └── market_data.py      # Market data event models
│   ├── consumers/
│   │   ├── market_data_consumer.py # RabbitMQ consumer for market data streams
│   │   └── position_consumer.py     # RabbitMQ consumer for position events
│   ├── publishers/
│   │   ├── feature_publisher.py    # Publish computed features to features.live queue
│   │   └── dataset_publisher.py    # Publish dataset completion to features.dataset.ready queue
│   ├── features/
│   │   ├── price_features.py       # Price/candlestick features (mid_price, spread, returns, VWAP, volatility)
│   │   ├── orderflow_features.py   # Orderflow features (signed_volume, buy/sell ratio, trade_count)
│   │   ├── orderbook_features.py   # Orderbook features (depth, imbalance, slope, churn_rate)
│   │   ├── perpetual_features.py   # Perpetual features (funding_rate, time_to_funding)
│   │   ├── temporal_features.py    # Temporal/meta features (time_of_day, rolling z-score)
│   │   └── position_features.py    # Position features (position_size, PnL, entry_price, exposure)
│   ├── storage/
│   │   ├── parquet_storage.py      # Parquet file read/write operations
│   │   └── metadata_storage.py     # PostgreSQL metadata storage
│   └── main.py                 # FastAPI application entry point
├── tests/
│   ├── unit/
│   │   ├── test_feature_computation.py
│   │   ├── test_orderbook_manager.py
│   │   ├── test_dataset_builder.py
│   │   └── test_feature_registry.py
│   ├── integration/
│   │   ├── test_feature_identity.py  # Online vs offline feature identity tests
│   │   ├── test_dataset_building.py
│   │   └── test_data_quality.py
│   └── contract/
│       └── test_api_contracts.py
├── migrations/                 # Database migrations (in ws-gateway per constitution)
├── config/
│   └── feature_registry.yaml  # Default Feature Registry configuration
├── Dockerfile
├── requirements.txt
├── README.md
└── env.example
```

**Structure Decision**: Single project structure (microservice). The service is organized into clear modules: API layer (REST endpoints), services layer (business logic), models (data structures), consumers (message queue consumers), publishers (message queue publishers), features (feature computation modules), and storage (Parquet and PostgreSQL). This structure supports separation of concerns while maintaining a single deployable unit.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
