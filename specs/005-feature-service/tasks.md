# Tasks: Feature Service

**Input**: Design documents from `/specs/005-feature-service/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: **MANDATORY** - All phases MUST include test tasks. Tests are written FIRST (TDD approach) before implementation. Tests include unit tests, integration tests, contract tests, and test data generation (mocks/stubs/fixtures).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `feature-service/src/`, `feature-service/tests/` at repository root
- Migrations: `ws-gateway/migrations/` (per constitution principle II)
- Test structure: `tests/unit/`, `tests/integration/`, `tests/contract/`, `tests/fixtures/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan in feature-service/
- [ ] T002 Initialize Python project with dependencies in feature-service/requirements.txt (include pytest, pytest-asyncio, pytest-mock)
- [ ] T003 [P] Create Dockerfile in feature-service/Dockerfile
- [ ] T004 [P] Create docker-compose.yml service configuration for feature-service
- [ ] T005 [P] Create env.example with all required environment variables in feature-service/env.example
- [ ] T006 [P] Configure linting and formatting tools (black, ruff) in feature-service/
- [ ] T007 Create README.md in feature-service/README.md
- [ ] T008 [P] Create test directory structure (tests/unit/, tests/integration/, tests/contract/, tests/fixtures/) in feature-service/
- [ ] T009 [P] Create pytest configuration file (pytest.ini or pyproject.toml) in feature-service/
- [ ] T010 [P] Create conftest.py with shared test fixtures in feature-service/tests/conftest.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T011 [P] Create test fixtures for database connection mocking in feature-service/tests/fixtures/database.py
- [ ] T012 [P] Create test fixtures for RabbitMQ connection mocking in feature-service/tests/fixtures/rabbitmq.py
- [ ] T013 [P] Create test fixtures for HTTP client mocking (ws-gateway API) in feature-service/tests/fixtures/http_client.py
- [ ] T014 [P] Create unit tests for configuration management in feature-service/tests/unit/test_config.py
- [ ] T015 [P] Create unit tests for logging setup in feature-service/tests/unit/test_logging.py
- [ ] T016 [P] Create integration tests for database connection pool in feature-service/tests/integration/test_metadata_storage.py
- [ ] T017 [P] Create integration tests for RabbitMQ connection manager in feature-service/tests/integration/test_mq_connection.py
- [ ] T018 [P] Create unit tests for HTTP client setup in feature-service/tests/unit/test_http_client.py
- [ ] T019 [P] Create unit tests for API authentication middleware in feature-service/tests/unit/test_auth_middleware.py
- [ ] T020 [P] Create contract tests for health check endpoint in feature-service/tests/contract/test_health.py
- [ ] T021 [P] Create test fixtures for market data events in feature-service/tests/fixtures/market_data.py
- [ ] T022 [P] Create unit tests for Feature Registry configuration loader in feature-service/tests/unit/test_feature_registry_loader.py

### Implementation for Foundational Phase

- [ ] T023 Create database migration for datasets table in ws-gateway/migrations/XXX_create_datasets_table.sql
- [ ] T024 Create database migration for feature_registry_versions table in ws-gateway/migrations/XXX_create_feature_registry_versions_table.sql
- [ ] T025 Create database migration for data_quality_reports table in ws-gateway/migrations/XXX_create_data_quality_reports_table.sql
- [ ] T026 [P] Create base configuration management in feature-service/src/config.py using pydantic-settings
- [ ] T027 [P] Create base logging setup with structlog in feature-service/src/logging.py
- [ ] T028 [P] Create database connection pool in feature-service/src/storage/metadata_storage.py using asyncpg
- [ ] T029 [P] Create RabbitMQ connection manager in feature-service/src/mq/connection.py using aio-pika
- [ ] T030 [P] Create HTTP client setup for ws-gateway REST API integration in feature-service/src/http/client.py using httpx
- [ ] T031 [P] Create base FastAPI application structure in feature-service/src/main.py
- [ ] T032 [P] Create API authentication middleware in feature-service/src/api/middleware/auth.py
- [ ] T033 [P] Create health check endpoint in feature-service/src/api/health.py
- [ ] T034 Create base models for market data events in feature-service/src/models/market_data.py
- [ ] T035 Create base Feature Registry configuration loader in feature-service/src/services/feature_registry.py (basic structure)
- [ ] T036 Create default Feature Registry YAML configuration in feature-service/config/feature_registry.yaml

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Получение признаков в реальном времени для торговых решений (Priority: P1) 🎯 MVP

**Goal**: Model Service получает готовые признаки в реальном времени для принятия торговых решений. Система вычисляет признаки из потока маркет-данных и предоставляет их с минимальной задержкой через API или очередь событий.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что признаки вычисляются и доступны через API в течение заданного времени задержки.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T037 [P] [US1] Create test fixtures for feature vectors in feature-service/tests/fixtures/feature_vectors.py
- [ ] T038 [P] [US1] Create test fixtures for orderbook state (snapshots and deltas) in feature-service/tests/fixtures/orderbook.py
- [ ] T039 [P] [US1] Create test fixtures for rolling windows data in feature-service/tests/fixtures/rolling_windows.py
- [ ] T040 [P] [US1] Create test fixtures for market data streams (orderbook, trades, klines, ticker, funding) in feature-service/tests/fixtures/market_data_streams.py
- [ ] T041 [P] [US1] Create unit tests for Feature Vector model in feature-service/tests/unit/test_feature_vector.py
- [ ] T042 [P] [US1] Create unit tests for Orderbook State model in feature-service/tests/unit/test_orderbook_state.py
- [ ] T043 [P] [US1] Create unit tests for Rolling Windows model in feature-service/tests/unit/test_rolling_windows.py
- [ ] T044 [P] [US1] Create unit tests for Orderbook Manager service in feature-service/tests/unit/test_orderbook_manager.py (snapshot + delta reconstruction, desynchronization)
- [ ] T045 [P] [US1] Create unit tests for price features computation in feature-service/tests/unit/test_price_features.py
- [ ] T046 [P] [US1] Create unit tests for orderflow features computation in feature-service/tests/unit/test_orderflow_features.py
- [ ] T047 [P] [US1] Create unit tests for orderbook features computation in feature-service/tests/unit/test_orderbook_features.py
- [ ] T048 [P] [US1] Create unit tests for perpetual features computation in feature-service/tests/unit/test_perpetual_features.py
- [ ] T049 [P] [US1] Create unit tests for temporal features computation in feature-service/tests/unit/test_temporal_features.py
- [ ] T050 [US1] Create unit tests for Feature Computer service in feature-service/tests/unit/test_feature_computer.py
- [ ] T051 [US1] Create integration tests for market data consumer in feature-service/tests/integration/test_market_data_consumer.py (with mocked RabbitMQ)
- [ ] T052 [US1] Create integration tests for subscription management in feature-service/tests/integration/test_subscription_management.py (with mocked ws-gateway API)
- [ ] T053 [US1] Create integration tests for feature publisher in feature-service/tests/integration/test_feature_publisher.py (with mocked RabbitMQ)
- [ ] T054 [US1] Create contract tests for GET /features/latest endpoint in feature-service/tests/contract/test_features_api.py
- [ ] T055 [US1] Create integration tests for feature computation latency (≤70ms) in feature-service/tests/integration/test_feature_latency.py
- [ ] T056 [US1] Create integration tests for ws-gateway unavailability handling in feature-service/tests/integration/test_ws_gateway_resilience.py

### Implementation for User Story 1

- [ ] T057 [P] [US1] Create Feature Vector model in feature-service/src/models/feature_vector.py
- [ ] T058 [P] [US1] Create Orderbook State model in feature-service/src/models/orderbook_state.py
- [ ] T059 [P] [US1] Create Rolling Windows model in feature-service/src/models/rolling_windows.py
- [ ] T060 [US1] Implement Orderbook Manager service in feature-service/src/services/orderbook_manager.py (snapshot + delta reconstruction)
- [ ] T061 [US1] Implement price features computation in feature-service/src/features/price_features.py (mid_price, spread, returns, VWAP, volatility)
- [ ] T062 [US1] Implement orderflow features computation in feature-service/src/features/orderflow_features.py (signed_volume, buy/sell ratio, trade_count, net_aggressor_pressure)
- [ ] T063 [US1] Implement orderbook features computation in feature-service/src/features/orderbook_features.py (depth, imbalance)
- [ ] T064 [US1] Implement perpetual features computation in feature-service/src/features/perpetual_features.py (funding_rate, time_to_funding)
- [ ] T065 [US1] Implement temporal features computation in feature-service/src/features/temporal_features.py (time_of_day with cyclic encoding)
- [ ] T066 [US1] Implement Feature Computer service in feature-service/src/services/feature_computer.py (orchestrates all feature computations)
- [ ] T067 [US1] Implement market data consumer in feature-service/src/consumers/market_data_consumer.py (consumes from ws-gateway.* queues)
- [ ] T068 [US1] Implement subscription management for WebSocket channels via ws-gateway REST API in feature-service/src/consumers/market_data_consumer.py
- [ ] T069 [US1] Implement subscription lifecycle management (create on startup, handle failures gracefully with retry, don't cancel on shutdown) in feature-service/src/consumers/market_data_consumer.py
- [ ] T070 [US1] Implement optional execution events consumer from ws-gateway.order or order-manager.order_events queues in feature-service/src/consumers/market_data_consumer.py
- [ ] T071 [US1] Implement optional subscription to order execution events via ws-gateway REST API in feature-service/src/consumers/market_data_consumer.py
- [ ] T072 [US1] Implement feature publisher in feature-service/src/publishers/feature_publisher.py (publishes to features.live queue)
- [ ] T073 [US1] Implement GET /features/latest endpoint in feature-service/src/api/features.py with 404 handling for missing symbols
- [ ] T074 [US1] Add internal timestamp and exchange timestamp to all received messages in feature-service/src/consumers/market_data_consumer.py
- [ ] T075 [US1] Implement feature computation scheduling at intervals (1s, 3s, 15s, 1m) in feature-service/src/services/feature_computer.py
- [ ] T076 [US1] Add logging for feature computation operations in feature-service/src/services/feature_computer.py
- [ ] T077 [US1] Add latency monitoring and metrics for feature computation in feature-service/src/services/feature_computer.py
- [ ] T078 [US1] Implement handling of ws-gateway unavailability (continue with last available data, log issues, update quality metrics) in feature-service/src/consumers/market_data_consumer.py
- [ ] T079 [US1] Implement latency warning when computation exceeds 70ms threshold in feature-service/src/services/feature_computer.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Features are computed in real-time and available via API and message queue.

---

## Phase 4: User Story 2 - Сборка датасета для обучения моделей с корректным разделением (Priority: P1)

**Goal**: Model Service может запросить сборку датасета обучения из исторических данных с явным разделением на train/validation/test периоды. Система пересчитывает признаки на исторических данных идентично онлайн-режиму и генерирует целевые переменные без data leakage.

**Independent Test**: Можно протестировать независимо, запросив сборку датасета за определенный период и проверив, что датасет содержит корректно разделенные данные train/val/test, признаки идентичны онлайн-версии, и targets вычислены без использования будущей информации.

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T080 [P] [US2] Create test fixtures for dataset metadata in feature-service/tests/fixtures/datasets.py
- [ ] T081 [P] [US2] Create test fixtures for historical market data (Parquet format) in feature-service/tests/fixtures/historical_data.py
- [ ] T082 [P] [US2] Create test fixtures for target variables (regression, classification, risk-adjusted) in feature-service/tests/fixtures/targets.py
- [ ] T083 [P] [US2] Create unit tests for Dataset model in feature-service/tests/unit/test_dataset.py
- [ ] T084 [P] [US2] Create unit tests for Parquet storage service in feature-service/tests/unit/test_parquet_storage.py
- [ ] T085 [P] [US2] Create unit tests for offline feature engine in feature-service/tests/unit/test_offline_engine.py
- [ ] T086 [P] [US2] Create unit tests for orderbook state reconstruction in feature-service/tests/unit/test_orderbook_reconstruction.py
- [ ] T087 [P] [US2] Create unit tests for rolling windows reconstruction in feature-service/tests/unit/test_rolling_windows_reconstruction.py
- [ ] T088 [P] [US2] Create unit tests for target variable computation in feature-service/tests/unit/test_target_computation.py
- [ ] T089 [P] [US2] Create unit tests for data leakage prevention validation in feature-service/tests/unit/test_data_leakage_prevention.py
- [ ] T090 [P] [US2] Create unit tests for time-based dataset splitting in feature-service/tests/unit/test_dataset_splitting.py
- [ ] T091 [P] [US2] Create unit tests for walk-forward validation strategy in feature-service/tests/unit/test_walk_forward.py
- [ ] T092 [P] [US2] Create unit tests for random split strategy in feature-service/tests/unit/test_random_split.py
- [ ] T093 [US2] Create integration tests for feature identity (online vs offline comparison) in feature-service/tests/integration/test_feature_identity.py
- [ ] T094 [US2] Create integration tests for dataset building workflow in feature-service/tests/integration/test_dataset_building.py
- [ ] T095 [US2] Create integration tests for batch processing of large datasets in feature-service/tests/integration/test_batch_processing.py
- [ ] T096 [US2] Create contract tests for POST /dataset/build endpoint in feature-service/tests/contract/test_dataset_api.py
- [ ] T097 [US2] Create contract tests for GET /dataset/list endpoint in feature-service/tests/contract/test_dataset_api.py
- [ ] T098 [US2] Create contract tests for GET /dataset/{dataset_id} endpoint in feature-service/tests/contract/test_dataset_api.py
- [ ] T099 [US2] Create contract tests for GET /dataset/{dataset_id}/download endpoint in feature-service/tests/contract/test_dataset_api.py
- [ ] T100 [US2] Create contract tests for POST /model/evaluate endpoint in feature-service/tests/contract/test_dataset_api.py
- [ ] T101 [US2] Create integration tests for dataset completion publisher in feature-service/tests/integration/test_dataset_publisher.py

### Implementation for User Story 2

- [ ] T102 [P] [US2] Create Dataset model in feature-service/src/models/dataset.py
- [ ] T103 [US2] Implement Parquet storage service in feature-service/src/storage/parquet_storage.py (read/write operations)
- [ ] T104 [US2] Implement offline feature engine in feature-service/src/services/offline_engine.py (rebuilds features from historical data)
- [ ] T105 [US2] Implement orderbook state reconstruction from snapshot + deltas in feature-service/src/services/offline_engine.py
- [ ] T106 [US2] Implement rolling windows reconstruction for historical data in feature-service/src/services/offline_engine.py
- [ ] T107 [US2] Implement target variable computation (regression: returns, classification: direction, risk-adjusted) in feature-service/src/services/dataset_builder.py
- [ ] T108 [US2] Implement configurable threshold for classification targets (default 0.001 = 0.1%) in feature-service/src/services/dataset_builder.py
- [ ] T109 [US2] Implement data leakage prevention validation in feature-service/src/services/dataset_builder.py
- [ ] T110 [US2] Implement time-based dataset splitting in feature-service/src/services/dataset_builder.py
- [ ] T111 [US2] Implement walk-forward validation strategy with configurable parameters in feature-service/src/services/dataset_builder.py
- [ ] T112 [US2] Implement random split strategy (for testing only, with temporal order preserved) in feature-service/src/services/dataset_builder.py
- [ ] T113 [US2] Implement Dataset Builder service in feature-service/src/services/dataset_builder.py (orchestrates dataset building)
- [ ] T114 [US2] Implement POST /dataset/build endpoint in feature-service/src/api/dataset.py with estimated_completion in response
- [ ] T115 [US2] Implement batch processing for large datasets with progress tracking in feature-service/src/services/dataset_builder.py
- [ ] T116 [US2] Implement queue management for concurrent dataset build requests in feature-service/src/services/dataset_builder.py
- [ ] T117 [US2] Implement GET /dataset/list endpoint in feature-service/src/api/dataset.py
- [ ] T118 [US2] Implement GET /dataset/{dataset_id} endpoint in feature-service/src/api/dataset.py
- [ ] T119 [US2] Implement GET /dataset/{dataset_id}/download endpoint in feature-service/src/api/dataset.py
- [ ] T120 [US2] Implement POST /model/evaluate endpoint in feature-service/src/api/dataset.py
- [ ] T121 [US2] Implement dataset completion publisher in feature-service/src/publishers/dataset_publisher.py (publishes to features.dataset.ready queue)
- [ ] T122 [US2] Implement error handling for missing historical data with available period suggestions in feature-service/src/services/dataset_builder.py
- [ ] T123 [US2] Add logging for dataset building operations in feature-service/src/services/dataset_builder.py
- [ ] T124 [US2] Implement feature identity validation (online vs offline comparison) in feature-service/src/services/offline_engine.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Datasets can be built from historical data with proper train/val/test splits.

---

## Phase 5: User Story 3 - Хранение сырых данных для последующей пересборки (Priority: P2)

**Goal**: Система хранит сырые маркет-данные в структурированном формате, обеспечивающем эффективную пересборку признаков на исторических данных. Данные хранятся минимальный период времени с возможностью архивирования.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что данные сохраняются в структурированном формате, доступны для чтения, и старые данные автоматически архивируются или удаляются после истечения срока хранения.

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T125 [P] [US3] Create test fixtures for raw market data (orderbook snapshots, deltas, trades, klines, ticker, funding, execution events) in feature-service/tests/fixtures/raw_data.py
- [ ] T126 [P] [US3] Create unit tests for raw data storage service in feature-service/tests/unit/test_data_storage.py
- [ ] T127 [P] [US3] Create unit tests for data organization by type in feature-service/tests/unit/test_data_organization.py
- [ ] T128 [P] [US3] Create unit tests for data retention policy enforcement in feature-service/tests/unit/test_data_retention.py
- [ ] T129 [P] [US3] Create unit tests for automatic archiving/deletion in feature-service/tests/unit/test_data_archiving.py
- [ ] T130 [US3] Create integration tests for raw data storage workflow in feature-service/tests/integration/test_data_storage_workflow.py
- [ ] T131 [US3] Create integration tests for data retrieval for dataset rebuilding in feature-service/tests/integration/test_data_retrieval.py

### Implementation for User Story 3

- [ ] T132 [US3] Implement raw data storage service in feature-service/src/services/data_storage.py (writes to Parquet files organized by type and date)
- [ ] T133 [US3] Implement data organization by type (orderbook snapshots, deltas, trades, klines, ticker, funding, execution events) in feature-service/src/services/data_storage.py
- [ ] T134 [US3] Implement storage of all orderbook deltas for offline reconstruction in feature-service/src/services/data_storage.py
- [ ] T135 [US3] Integrate raw data storage into market data consumer in feature-service/src/consumers/market_data_consumer.py
- [ ] T136 [US3] Implement data retention policy enforcement in feature-service/src/services/data_storage.py (90 days default, configurable)
- [ ] T137 [US3] Implement automatic archiving/deletion of expired data with archive recovery support in feature-service/src/services/data_storage.py
- [ ] T138 [US3] Add logging for data storage operations in feature-service/src/services/data_storage.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently. Raw data is stored and available for dataset rebuilding.

---

## Phase 6: User Story 4 - Мониторинг качества данных и обнаружение проблем (Priority: P2)

**Goal**: Система отслеживает качество входящих данных, обнаруживает проблемы (пропуски, аномалии, рассинхронизации) и предоставляет отчеты о качестве данных через API.

**Independent Test**: Можно протестировать независимо, отправив данные с проблемами (пропуски, аномалии) и проверив, что система их обнаруживает и предоставляет отчет через API.

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T139 [P] [US4] Create test fixtures for data quality scenarios (missing data, anomalies, sequence gaps, desynchronization) in feature-service/tests/fixtures/data_quality.py
- [ ] T140 [P] [US4] Create unit tests for Data Quality Report model in feature-service/tests/unit/test_data_quality_model.py
- [ ] T141 [P] [US4] Create unit tests for missing data detection in feature-service/tests/unit/test_missing_data_detection.py
- [ ] T142 [P] [US4] Create unit tests for anomaly detection in feature-service/tests/unit/test_anomaly_detection.py
- [ ] T143 [P] [US4] Create unit tests for sequence gap detection in feature-service/tests/unit/test_sequence_gap_detection.py
- [ ] T144 [P] [US4] Create unit tests for timestamp validation in feature-service/tests/unit/test_timestamp_validation.py
- [ ] T145 [P] [US4] Create unit tests for orderbook desynchronization detection in feature-service/tests/unit/test_desynchronization_detection.py
- [ ] T146 [US4] Create integration tests for data quality monitoring service in feature-service/tests/integration/test_data_quality_monitoring.py
- [ ] T147 [US4] Create integration tests for data quality report generation in feature-service/tests/integration/test_data_quality_reports.py
- [ ] T148 [US4] Create contract tests for GET /data-quality/report endpoint in feature-service/tests/contract/test_data_quality_api.py
- [ ] T149 [US4] Create integration tests for data quality report performance (≤5 seconds for 24-hour period) in feature-service/tests/integration/test_data_quality_performance.py

### Implementation for User Story 4

- [ ] T150 [P] [US4] Create Data Quality Report model in feature-service/src/models/data_quality.py
- [ ] T151 [US4] Implement data quality monitoring service in feature-service/src/services/data_quality.py (tracks missing rate, anomaly rate, sequence gaps)
- [ ] T152 [US4] Implement missing data detection in feature-service/src/services/data_quality.py
- [ ] T153 [US4] Implement anomaly detection (outliers, price spikes) in feature-service/src/services/data_quality.py
- [ ] T154 [US4] Implement sequence gap detection in feature-service/src/services/data_quality.py
- [ ] T155 [US4] Implement timestamp validation (internal vs exchange timestamp) in feature-service/src/services/data_quality.py
- [ ] T156 [US4] Implement orderbook desynchronization detection and recovery in feature-service/src/services/orderbook_manager.py
- [ ] T157 [US4] Implement snapshot request on desynchronization (within 1 second) in feature-service/src/services/orderbook_manager.py
- [ ] T158 [US4] Implement data quality report generation with recommendations in feature-service/src/services/data_quality.py
- [ ] T159 [US4] Implement GET /data-quality/report endpoint in feature-service/src/api/data_quality.py (≤5 seconds for 24-hour period)
- [ ] T160 [US4] Add logging for data quality issues in feature-service/src/services/data_quality.py

**Checkpoint**: At this point, User Stories 1, 2, 3, AND 4 should all work independently. Data quality is monitored and reported.

---

## Phase 7: User Story 5 - Управление конфигурацией признаков через Feature Registry (Priority: P3)

**Goal**: Система предоставляет возможность управления конфигурацией признаков через Feature Registry с проверкой на data leakage и валидацией временных границ.

**Independent Test**: Можно протестировать независимо, загрузив конфигурацию Feature Registry и проверив, что система валидирует конфигурацию, проверяет временные границы и позволяет перезагрузить конфигурацию.

### Tests for User Story 5

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T161 [P] [US5] Create test fixtures for Feature Registry configurations (valid, invalid, with data leakage) in feature-service/tests/fixtures/feature_registry.py
- [ ] T162 [P] [US5] Create unit tests for Feature Registry model in feature-service/tests/unit/test_feature_registry_model.py
- [ ] T163 [P] [US5] Create unit tests for Feature Registry configuration validation in feature-service/tests/unit/test_feature_registry_validation.py
- [ ] T164 [P] [US5] Create unit tests for Feature Registry version management in feature-service/tests/unit/test_feature_registry_versioning.py
- [ ] T165 [P] [US5] Create unit tests for Feature Registry fallback to previous valid version in feature-service/tests/unit/test_feature_registry_fallback.py
- [ ] T166 [US5] Create integration tests for Feature Registry loading and activation in feature-service/tests/integration/test_feature_registry_integration.py
- [ ] T167 [US5] Create contract tests for GET /feature-registry endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T168 [US5] Create contract tests for POST /feature-registry/reload endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169 [US5] Create contract tests for GET /feature-registry/validate endpoint in feature-service/tests/contract/test_feature_registry_api.py

### Implementation for User Story 5

- [ ] T170 [P] [US5] Create Feature Registry model in feature-service/src/models/feature_registry.py
- [ ] T171 [US5] Implement Feature Registry configuration validation in feature-service/src/services/feature_registry.py (temporal boundaries, data leakage prevention, max_lookback_days)
- [ ] T172 [US5] Implement Feature Registry version management in feature-service/src/services/feature_registry.py
- [ ] T173 [US5] Implement Feature Registry loading and activation with fallback to previous valid version on validation failure in feature-service/src/services/feature_registry.py
- [ ] T174 [US5] Integrate Feature Registry into feature computation (online and offline) in feature-service/src/services/feature_computer.py and feature-service/src/services/offline_engine.py
- [ ] T175 [US5] Implement GET /feature-registry endpoint in feature-service/src/api/feature_registry.py
- [ ] T176 [US5] Implement POST /feature-registry/reload endpoint in feature-service/src/api/feature_registry.py
- [ ] T177 [US5] Implement GET /feature-registry/validate endpoint in feature-service/src/api/feature_registry.py
- [ ] T178 [US5] Add logging for Feature Registry operations in feature-service/src/services/feature_registry.py

**Checkpoint**: At this point, all user stories should be independently functional. Feature Registry manages feature configuration with validation.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Additional Tests

- [ ] T179 [P] Create end-to-end tests for complete feature computation workflow in feature-service/tests/e2e/test_feature_computation_workflow.py
- [ ] T180 [P] Create end-to-end tests for dataset building workflow in feature-service/tests/e2e/test_dataset_building_workflow.py
- [ ] T181 [P] Create performance tests for feature computation latency in feature-service/tests/performance/test_feature_latency.py
- [ ] T182 [P] Create performance tests for dataset building performance in feature-service/tests/performance/test_dataset_building_performance.py
- [ ] T183 [P] Create load tests for concurrent API requests in feature-service/tests/performance/test_api_load.py

### Implementation

- [ ] T184 [P] Update README.md with complete usage examples in feature-service/README.md
- [ ] T185 [P] Synchronize quickstart.md with implementation in specs/005-feature-service/quickstart.md
- [ ] T186 [P] Add comprehensive error handling across all services (missing data, ws-gateway unavailability, missing symbols)
- [ ] T187 [P] Add performance monitoring and metrics collection (latency, throughput, data quality metrics)
- [ ] T188 [P] Implement request tracing with trace_id across all endpoints and message queues
- [ ] T189 Code cleanup and refactoring across all modules
- [ ] T190 [P] Add comprehensive logging for all operations (raw data events, sequence errors, missing data gaps, desynchronization events)
- [ ] T191 Run quickstart.md validation and update examples
- [ ] T192 [P] Add API documentation updates (OpenAPI spec synchronization)
- [ ] T193 Add health check improvements (database, message queue status, data quality metrics)
- [ ] T194 [P] Security hardening (API key validation, input sanitization)
- [ ] T195 [P] Add data handling strategies (interpolation, forward fill, skipping records for missing values)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
  - **Tests FIRST**: Write all foundational tests before implementation
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - **Tests FIRST**: Write all tests for a user story before implementation
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Depends on US1 for feature computation logic (offline engine uses same feature computation code)
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories (can run in parallel with US1/US2)
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 for orderbook manager and data quality monitoring
- **User Story 5 (P3)**: Can start after Foundational (Phase 2) - Depends on US1 and US2 for feature computation integration

### Within Each User Story

- **Tests FIRST** (TDD approach): Write all tests before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational test tasks marked [P] can run in parallel
- All Foundational implementation tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes:
  - US1 and US3 can start in parallel (no dependencies between them)
  - US2 can start after US1 (needs feature computation logic)
  - US4 can start after US1 (needs orderbook manager)
  - US5 can start after US1 and US2 (needs feature computation integration)
- All test fixtures within a story marked [P] can run in parallel
- All unit tests within a story marked [P] can run in parallel
- All models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members (respecting dependencies)

---

## Parallel Example: User Story 1

```bash
# Launch all test fixtures for User Story 1 together:
Task: "Create test fixtures for feature vectors in feature-service/tests/fixtures/feature_vectors.py"
Task: "Create test fixtures for orderbook state in feature-service/tests/fixtures/orderbook.py"
Task: "Create test fixtures for rolling windows data in feature-service/tests/fixtures/rolling_windows.py"
Task: "Create test fixtures for market data streams in feature-service/tests/fixtures/market_data_streams.py"

# Launch all unit tests for User Story 1 together (after fixtures):
Task: "Create unit tests for Feature Vector model in feature-service/tests/unit/test_feature_vector.py"
Task: "Create unit tests for Orderbook State model in feature-service/tests/unit/test_orderbook_state.py"
Task: "Create unit tests for Rolling Windows model in feature-service/tests/unit/test_rolling_windows.py"
Task: "Create unit tests for price features computation in feature-service/tests/unit/test_price_features.py"
Task: "Create unit tests for orderflow features computation in feature-service/tests/unit/test_orderflow_features.py"
Task: "Create unit tests for orderbook features computation in feature-service/tests/unit/test_orderbook_features.py"
Task: "Create unit tests for perpetual features computation in feature-service/tests/unit/test_perpetual_features.py"
Task: "Create unit tests for temporal features computation in feature-service/tests/unit/test_temporal_features.py"

# Launch all models for User Story 1 together (after tests):
Task: "Create Feature Vector model in feature-service/src/models/feature_vector.py"
Task: "Create Orderbook State model in feature-service/src/models/orderbook_state.py"
Task: "Create Rolling Windows model in feature-service/src/models/rolling_windows.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
   - Write all foundational tests FIRST
   - Implement foundational components
3. Complete Phase 3: User Story 1
   - Write all User Story 1 tests FIRST
   - Implement User Story 1 components
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
   - Write tests FIRST, then implement
2. Add User Story 1 → Write tests FIRST, then implement → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Write tests FIRST, then implement → Test independently → Deploy/Demo
4. Add User Story 3 → Write tests FIRST, then implement → Test independently → Deploy/Demo
5. Add User Story 4 → Write tests FIRST, then implement → Test independently → Deploy/Demo
6. Add User Story 5 → Write tests FIRST, then implement → Test independently → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
   - All developers write foundational tests in parallel
   - Then implement foundational components in parallel
2. Once Foundational is done:
   - Developer A: User Story 1 (P1) - writes tests FIRST, then implements
   - Developer B: User Story 3 (P2) - writes tests FIRST, then implements (can start in parallel with US1)
   - Developer C: User Story 2 (P1) - writes tests FIRST, then implements (starts after US1 core features complete)
3. After US1 and US2 complete:
   - Developer A: User Story 4 (P2) - writes tests FIRST, then implements
   - Developer B: User Story 5 (P3) - writes tests FIRST, then implements
   - Developer C: Polish & Cross-Cutting Concerns - writes additional tests FIRST, then implements improvements
4. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- **TDD Approach**: Write tests FIRST, ensure they FAIL, then implement to make them pass
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Database migrations MUST be located in `ws-gateway/migrations/` per constitution principle II
- All PostgreSQL migrations must be reversible
- Feature computation code must be shared between online and offline engines to ensure feature identity
- Raw data storage uses local filesystem (mounted volumes) for Parquet files
- Message queue subscriptions are managed via ws-gateway REST API on service startup
- **Test Data**: All test fixtures, mocks, and stubs must be created before writing tests
- **Test Coverage**: Aim for comprehensive coverage including edge cases, error scenarios, and performance requirements

---

## Task Summary

- **Total Tasks**: 195
- **Phase 1 (Setup)**: 10 tasks (added test structure setup)
- **Phase 2 (Foundational)**: 26 tasks (12 tests + 14 implementation)
- **Phase 3 (User Story 1)**: 43 tasks (20 tests + 23 implementation)
- **Phase 4 (User Story 2)**: 45 tasks (25 tests + 23 implementation)
- **Phase 5 (User Story 3)**: 14 tasks (7 tests + 7 implementation)
- **Phase 6 (User Story 4)**: 22 tasks (11 tests + 11 implementation)
- **Phase 7 (User Story 5)**: 18 tasks (9 tests + 9 implementation)
- **Phase 8 (Polish)**: 17 tasks (5 additional tests + 12 implementation)

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) = 79 tasks

**Test Tasks Breakdown**:

- **Test Fixtures/Mocks/Stubs**: 25 tasks
- **Unit Tests**: 45 tasks
- **Integration Tests**: 20 tasks
- **Contract Tests**: 15 tasks
- **E2E/Performance Tests**: 5 tasks
- **Total Test Tasks**: 110 tasks

**Parallel Opportunities**:

- Setup phase: 4 tasks can run in parallel
- Foundational phase: 12 test tasks can run in parallel, 9 implementation tasks can run in parallel
- User Story 1: 4 test fixtures can run in parallel, 8 unit tests can run in parallel, 3 models can run in parallel
- User Stories 1 and 3 can start in parallel after foundational phase
