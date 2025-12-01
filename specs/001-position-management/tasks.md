# Tasks: Position Management Service

**Feature**: Position Management Service  
**Branch**: `001-position-management`  
**Date**: 2025-01-27  
**Status**: Ready for Implementation

## Summary

- **Total Tasks**: 121
- **User Story 1 (P1)**: 31 tasks (T012-T015b, T016-T039)
- **User Story 2 (P1)**: 25 tasks (T040-T064)
- **User Story 3 (P2)**: 12 tasks (T065-T076)
- **User Story 4 (P2)**: 12 tasks (T077-T077a, T078-T081, T082-T086)
- **User Story 5 (P3)**: 10 tasks (T087-T087a, T088-T094)
- **Setup & Foundational**: 11 tasks (T001-T011)
- **Polish & Cross-Cutting**: 20 tasks (T095-T114)

## Implementation Strategy

**MVP Scope**: User Story 1 (Real-Time Position Tracking) - provides core functionality for position data access and portfolio metrics.

**Incremental Delivery**:
1. **Phase 1-2**: Setup and foundational infrastructure
2. **Phase 3**: User Story 1 (P1) - Core position tracking and portfolio metrics
3. **Phase 4**: User Story 2 (P1) - Automatic position updates from multiple sources
4. **Phase 5**: User Story 3 (P2) - Portfolio risk management support
5. **Phase 6**: User Story 4 (P2) - Historical position tracking
6. **Phase 7**: User Story 5 (P3) - Position data validation and synchronization
7. **Phase 8**: Polish & cross-cutting concerns

## Dependencies

**Story Completion Order**:
- User Story 1 (P1) → Independent, can start immediately after Phase 2
- User Story 2 (P1) → Depends on User Story 1 (needs position models and services)
- User Story 3 (P2) → Depends on User Story 1 (needs portfolio metrics)
- User Story 4 (P2) → Depends on User Story 1 (needs position models)
- User Story 5 (P3) → Depends on User Story 1 and User Story 2 (needs position updates)

## Parallel Execution Opportunities

**Within User Story 1**:
- T012-T014: Models can be created in parallel (different files)
- T015-T017: Services can be created in parallel (different files)
- T018-T021: API routes can be created in parallel (different files)

**Within User Story 2**:
- T040-T041: Consumers can be created in parallel (different files)
- T042-T043: Event processing can be implemented in parallel (different files)

**Cross-Phase Parallelization**:
- Setup tasks (T001-T011) can be done in parallel where dependencies allow
- Test tasks can be done in parallel with implementation tasks

---

## Phase 1: Setup

**Goal**: Initialize project structure and basic configuration.

### T001: Create project directory structure
- [X] T001 Create project directory structure per plan.md in position-manager/

### T002: Create Dockerfile
- [X] T002 Create Dockerfile in position-manager/Dockerfile with Python 3.11+ base image

### T003: Create requirements.txt
- [X] T003 Create requirements.txt in position-manager/requirements.txt with FastAPI, pydantic-settings, structlog, asyncpg, aio-pika, httpx, pytest, pytest-asyncio

### T004: Create env.example
- [X] T004 Create env.example in position-manager/env.example with all POSITION_MANAGER_* environment variables from quickstart.md

### T005: Add service to docker-compose.yml
- [X] T005 Add position-manager service to docker-compose.yml with port 4800, environment variables, and dependencies on postgres and rabbitmq

### T006: Create README.md
- [X] T006 Create README.md in position-manager/README.md with service overview, setup instructions, and API documentation links

---

## Phase 2: Foundational Infrastructure

**Goal**: Set up core infrastructure components that are prerequisites for all user stories.

### T007: Create configuration module
- [X] T007 Create src/config/settings.py with pydantic-settings BaseSettings class and all POSITION_MANAGER_* environment variables

### T008: Create database configuration
- [X] T008 Create src/config/database.py with asyncpg connection pool setup and connection management functions

### T009: Create RabbitMQ configuration
- [X] T009 Create src/config/rabbitmq.py with aio-pika connection setup and connection management functions

### T010: Create logging configuration
- [X] T010 Create src/config/logging.py with structlog configuration, trace ID support, and JSON output format

### T011: Create tracing utility
- [X] T011 Create src/utils/tracing.py with trace ID generation and context propagation utilities

---

## Phase 3: User Story 1 - Real-Time Position Tracking (P1)

**Goal**: Trading system components can access current position data and portfolio metrics in real-time.

**Independent Test Criteria**: Query position data after simulated position updates and verify returned data reflects current state accurately. Test portfolio metrics calculation with multiple positions.

### T012: Create Position model
- [X] T012 [P] [US1] Create Position Pydantic model in src/models/position.py with all fields from data-model.md (asset, mode, size, average_entry_price, current_price, unrealized_pnl, realized_pnl, version, etc.)

### T013: Create PositionSnapshot model
- [X] T013 [P] [US1] Create PositionSnapshot Pydantic model in src/models/position.py with id, position_id, asset, mode, snapshot_data (JSONB), created_at fields

### T014: Create Portfolio model
- [X] T014 [P] [US1] Create Portfolio Pydantic model in src/models/portfolio.py with calculated fields (total_exposure_usdt, total_unrealized_pnl_usdt, total_realized_pnl_usdt, portfolio_value_usdt, open_positions_count, by_asset, calculated_at)

### T015: Extract PositionManager from Order Manager
- [X] T015 [US1] Extract PositionManager class from order-manager/src/services/position_manager.py: copy class to src/services/position_manager.py, adapt imports and paths to new structure, extract methods get_position(asset, mode), get_all_positions(), update_position_from_order_fill(), update_position_from_websocket(), validate_position(asset, mode), create_position_snapshot(position)

### T015a: Adapt PositionManager to new structure
- [X] T015a [US1] Adapt PositionManager class to new structure: update database connection imports to use src/config/database.py, update logging to use src/config/logging.py, update configuration to use src/config/settings.py, ensure all imports reference new module paths

### T015b: Enhance PositionManager with portfolio metrics methods
- [X] T015b [US1] Add new methods to PositionManager in src/services/position_manager.py: get_total_exposure(), get_portfolio_metrics(), methods for calculating ML features (unrealized_pnl_pct, time_held_minutes, position_size_norm)

### T016: Create PortfolioManager service
- [X] T016 [P] [US1] Create PortfolioManager service in src/services/portfolio_manager.py with methods: get_portfolio_metrics(include_positions, asset_filter), get_total_exposure(), get_portfolio_pnl(), calculate_metrics_from_positions(positions)

### T017: Create portfolio metrics cache
- [X] T017 [P] [US1] Create in-memory cache for portfolio metrics in src/services/portfolio_manager.py with TTL support (POSITION_MANAGER_METRICS_CACHE_TTL), cache invalidation on position updates, and cache key management

### T018: Extract positions API routes from Order Manager
- [ ] T018 [US1] Extract REST API endpoints from order-manager/src/api/routes/positions.py: copy GET /api/v1/positions, GET /api/v1/positions/{asset}, POST /api/v1/positions/{asset}/validate, POST /api/v1/positions/{asset}/snapshot, GET /api/v1/positions/{asset}/snapshots to src/api/routes/positions.py, adapt imports and service references to use PositionManager from new service

### T019: Adapt extracted API routes to new structure
- [ ] T019 [US1] Adapt extracted API routes in src/api/routes/positions.py: update imports to use new PositionManager service, update authentication middleware references, update response serialization to include calculated ML features, ensure error handling uses new logging configuration

### T020: Create portfolio API route
- [X] T020 [P] [US1] Create GET /api/v1/portfolio endpoint in src/api/routes/portfolio.py with include_positions and asset query parameters, returning PortfolioMetrics with cached metrics

### T021: Create portfolio exposure API route
- [X] T021 [P] [US1] Create GET /api/v1/portfolio/exposure endpoint in src/api/routes/portfolio.py returning PortfolioExposure with total_exposure_usdt

### T022: Create portfolio PnL API route
- [X] T022 [US1] Create GET /api/v1/portfolio/pnl endpoint in src/api/routes/portfolio.py returning PortfolioPnL with total_unrealized_pnl_usdt, total_realized_pnl_usdt, total_pnl_usdt

### T023: Create health check endpoint
- [X] T023 [US1] Create GET /health endpoint in src/api/routes/health.py with database_connected, queue_connected, positions_count status checks

### T024: Create API authentication middleware
- [X] T024 [US1] Create API key authentication middleware in src/api/middleware/auth.py with X-API-Key header validation using POSITION_MANAGER_API_KEY

### T025: Create API logging middleware
- [X] T025 [US1] Create request logging middleware in src/api/middleware/logging.py with trace ID extraction, request/response logging, and structured log output

### T026: Create main FastAPI application
- [X] T026 [US1] Create src/api/main.py with FastAPI app setup, route registration, middleware registration, and startup/shutdown event handlers

### T027: Create main entry point
- [X] T027 [US1] Create src/main.py with uvicorn server startup, database connection initialization, RabbitMQ connection initialization, and graceful shutdown handling

### T028: Implement database queries for positions
- [X] T028 [US1] Implement async database queries in src/services/position_manager.py: get_position_by_asset_mode(asset, mode), get_all_positions(filters), using asyncpg with proper error handling

### T029: Implement portfolio metrics calculation
- [X] T029 [US1] Implement portfolio metrics calculation in src/services/portfolio_manager.py: calculate_total_exposure(positions), calculate_total_pnl(positions), calculate_by_asset(positions), with proper NULL handling for current_price

### T030: Implement ML features calculation
- [X] T030 [US1] Implement ML features calculation in src/services/position_manager.py: calculate_unrealized_pnl_pct(position), calculate_time_held_minutes(position), calculate_position_size_norm(position, total_exposure)

### T031: Implement position filtering logic
- [X] T031 [US1] Implement position filtering logic in src/services/position_manager.py: filter_by_asset(positions, asset), filter_by_mode(positions, mode), filter_by_size(positions, size_min, size_max)

### T032: Add database indexes migration note
- [X] T032 [US1] Document required database indexes in README.md: idx_positions_asset, idx_positions_mode, idx_positions_asset_mode, idx_positions_current_price, idx_positions_version (migration handled by ws-gateway service)

### T104a: Create database migration for current_price and version fields
- [X] T104a [US1] Create database migration file in ws-gateway/migrations/014_add_current_price_and_version_to_positions.sql: add current_price DECIMAL(20, 8) NULL column, add version INTEGER NOT NULL DEFAULT 1 column, create indexes idx_positions_current_price and idx_positions_version, update existing rows to set version=1, include rollback section (per constitution - PostgreSQL migration ownership in ws-gateway service)

### T104: Validate database migration requirements
- [X] T104 [US1] Validate database migration requirements: verify migration file 014_add_current_price_and_version_to_positions.sql exists in ws-gateway/migrations/, ensure migration SQL matches data-model.md requirements, document migration execution steps in README.md, verify migration is reversible

### T033: Implement error handling for missing positions
- [X] T033 [US1] Implement error handling in src/api/routes/positions.py: return HTTP 404 when position not found, with proper error message format

### T034: Implement empty portfolio handling
- [X] T034 [US1] Implement empty portfolio handling in src/services/portfolio_manager.py: return zero values for all metrics when no positions exist, return HTTP 200 with empty positions array

### T035: Add request validation
- [X] T035 [US1] Add Pydantic request validation in src/api/routes/positions.py and src/api/routes/portfolio.py for query parameters (asset format, mode enum, size ranges)

### T036: Implement response serialization
- [X] T036 [US1] Implement response serialization in src/api/routes/positions.py and src/api/routes/portfolio.py: convert Decimal to string, format timestamps as ISO 8601, include all calculated ML features

### T037: Add database connection retry logic
- [X] T037 [US1] Add database connection retry logic in src/config/database.py: retry on connection failure, exponential backoff, max retries configuration

### T038: Add RabbitMQ connection retry logic
- [X] T038 [US1] Add RabbitMQ connection retry logic in src/config/rabbitmq.py: retry on connection failure, exponential backoff, max retries configuration

### T039: Create unit tests for PositionManager
- [X] T039 [US1] Create unit tests in tests/unit/test_position_manager.py: test_get_position, test_get_all_positions, test_calculate_ml_features, test_filtering_logic

---

## Phase 4: User Story 2 - Automatic Position Updates from Multiple Sources (P1)

**Goal**: System automatically updates positions when orders are executed or market data indicates position changes, ensuring data consistency.

**Independent Test Criteria**: Send position update events from different sources (order executions, market data) and verify positions are updated correctly and conflicts are resolved appropriately.

### T040: Create WebSocket position event consumer
- [X] T040 [P] [US2] Create WebSocket position event consumer in src/consumers/websocket_position_consumer.py: consume from ws-gateway.position queue, parse event payload, call update_position_from_websocket()

### T041: Create order execution event consumer
- [X] T041 [P] [US2] Create order execution event consumer in src/consumers/order_position_consumer.py: consume from order-manager.order_executed queue, parse event payload, call update_position_from_order_fill()

### T042: Enhance update_position_from_websocket method from extracted code
- [X] T042 [US2] Enhance update_position_from_websocket() in src/services/position_manager.py (extracted from Order Manager): add handling of avgPrice from WebSocket event with threshold comparison (POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD), add validation of size from WebSocket event (without direct update, only discrepancy checking), save markPrice to current_price field, add recalculation of ML features (unrealized_pnl_pct, position_size_norm) after update, add logging of all updates with trace_id, add handling of case when avgPrice is missing (use saved value), use optimistic locking

### T043: Implement update_position_from_order_fill method
- [X] T043 [US2] Implement update_position_from_order_fill() in src/services/position_manager.py: update position size, recalculate average_entry_price on order fill, update realized_pnl on position close, use optimistic locking with retry logic

### T044: Implement optimistic locking with version field
- [X] T044 [US2] Implement optimistic locking in src/services/position_manager.py: check version before update, increment version on successful update, retry up to 3 times with exponential backoff (100ms, 200ms, 400ms) on conflict, log conflicts, raise exception if all retries fail

### T045: Implement conflict resolution for average_entry_price
- [X] T045 [US2] Implement conflict resolution in src/services/position_manager.py: compare WebSocket avgPrice with existing average_entry_price, update if difference > POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD (0.1%), otherwise keep existing value, log all updates

### T046: Implement position size validation
- [X] T046 [US2] Implement position size validation in src/services/position_manager.py: compare WebSocket size with database size, log discrepancy if difference > POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD, trigger validation task if configured

### T047: Implement position creation on first update
- [X] T047 [US2] Implement position creation in src/services/position_manager.py: create new position if not exists when update received, set version=1, set created_at timestamp, initialize all required fields

### T048: Implement position close handling
- [X] T048 [US2] Implement position close handling in src/services/position_manager.py: set size=0 when position closed, set closed_at timestamp, retain position for historical tracking, exclude from open_positions_count

### T049: Implement cache invalidation on position update
- [X] T049 [US2] Implement cache invalidation in src/services/portfolio_manager.py: clear portfolio metrics cache when position updated, trigger cache refresh on next portfolio query

### T050: Create position event publisher
- [X] T050 [US2] Create position event publisher in src/publishers/position_event_publisher.py: publish position_updated events to position-manager.position_updated queue with complete position data including ML features

### T051: Create portfolio event publisher
- [X] T051 [US2] Create portfolio event publisher in src/publishers/position_event_publisher.py: publish portfolio_updated events to position-manager.portfolio_updated queue with portfolio metrics

### T052: Integrate consumers with main application
- [X] T052 [US2] Integrate RabbitMQ consumers in src/main.py: start WebSocket consumer and order execution consumer on application startup, handle graceful shutdown, implement error handling and message acknowledgment

### T053: Implement external price API integration
- [X] T053 [US2] Implement external price API integration in src/services/position_manager.py: query Bybit REST API /v5/market/tickers when markPrice missing or stale, 3 retries with exponential backoff (1s, 2s, 4s), 5s timeout, fallback to last known price, set current_price to NULL if all fail

### T054: Implement price staleness check
- [X] T054 [US2] Implement price staleness check in src/services/position_manager.py: check time since last current_price update, query external API if > POSITION_MANAGER_PRICE_STALENESS_THRESHOLD (300s), log all external API calls

### T055: Add event processing error handling
- [X] T055 [US2] Add error handling in src/consumers/websocket_position_consumer.py and src/consumers/order_position_consumer.py: catch exceptions, log errors with trace_id, nack message with requeue on transient errors, dead letter queue on permanent errors

### T056: Implement message acknowledgment strategy
- [X] T056 [US2] Implement message acknowledgment in src/consumers/websocket_position_consumer.py and src/consumers/order_position_consumer.py: ack on successful processing, nack with requeue on transient errors, nack without requeue on permanent errors

### T057: Add event validation
- [X] T057 [US2] Add event validation in src/consumers/websocket_position_consumer.py and src/consumers/order_position_consumer.py: validate event structure with Pydantic models, validate required fields, reject invalid events with error logging

### T058: Implement out-of-order event handling
- [X] T058 [US2] Implement out-of-order event handling in src/services/position_manager.py: use version field to detect stale updates, log out-of-order events, process events based on version comparison

### T059: Add position update logging
- [X] T059 [US2] Add structured logging in src/services/position_manager.py: log all position updates with trace_id, log conflict resolutions, log validation results, log external API calls

### T060: Create integration tests for event consumers
- [ ] T060 [US2] Create integration tests in tests/integration/test_websocket_consumer.py and tests/integration/test_order_consumer.py: test event processing, test conflict resolution, test optimistic locking retries, use testcontainers for RabbitMQ

### T061: Create unit tests for conflict resolution
- [X] T061 [US2] Create unit tests in tests/unit/test_position_manager.py: test_avg_price_conflict_resolution, test_size_validation, test_optimistic_locking_retry, test_position_creation

### T062: Create unit tests for external price API
- [X] T062 [US2] Create unit tests in tests/unit/test_position_manager.py: test_external_price_api_success, test_external_price_api_retry, test_external_price_api_fallback, mock httpx requests

### T063: Add rate limiting for API endpoints
- [X] T063 [US2] Implement rate limiting in src/api/middleware/auth.py: per-API-key rate limits with tiers (POSITION_MANAGER_RATE_LIMIT_DEFAULT, POSITION_MANAGER_RATE_LIMIT_OVERRIDES), sliding window or token bucket algorithm, return HTTP 429 with Retry-After header on exceedance, log rate limit exceedances

### T064: Create E2E tests for position update flow
- [ ] T064 [US2] Create E2E tests in tests/e2e/test_position_updates.py: test_websocket_position_update_flow, test_order_execution_update_flow, test_conflict_resolution_flow, test_concurrent_updates

---

## Phase 5: User Story 3 - Portfolio Risk Management Support (P2)

**Goal**: Risk management components can access portfolio-level exposure and profit/loss metrics to enforce trading limits.

**Independent Test Criteria**: Query portfolio metrics and verify risk management components can use these metrics to make limit-checking decisions.

### T065: Enhance portfolio exposure endpoint for risk management
- [X] T065 [US3] Enhance GET /api/v1/portfolio/exposure in src/api/routes/portfolio.py: ensure response format suitable for risk management, include calculated_at timestamp, optimize response time for frequent queries

### T066: Add portfolio metrics breakdown by asset
- [X] T066 [US3] Enhance portfolio metrics calculation in src/services/portfolio_manager.py: ensure by_asset breakdown includes exposure_usdt, unrealized_pnl_usdt, size for each asset, suitable for risk management analysis

### T067: Optimize portfolio metrics query performance
- [X] T067 [US3] Optimize portfolio metrics queries in src/services/portfolio_manager.py: use database indexes effectively, minimize query execution time, ensure <1s response time for portfolios with up to 100 positions

### T068: Add portfolio metrics caching for risk management
- [X] T068 [US3] Enhance portfolio metrics cache in src/services/portfolio_manager.py: ensure cache TTL appropriate for risk management use case (5-10 seconds), implement cache warming on service startup

### T069: Add error handling for risk management queries
- [X] T069 [US3] Add error handling in src/api/routes/portfolio.py: handle database unavailability gracefully, return appropriate error codes, implement fallback logic for risk management components

### T070: Create integration tests for risk management integration
- [X] T070 [US3] Create integration tests in tests/integration/test_risk_management.py: test_portfolio_exposure_query, test_portfolio_metrics_for_limit_checking, simulate risk management component queries

### T071: Add performance monitoring for portfolio queries
- [X] T071 [US3] Add performance monitoring in src/services/portfolio_manager.py: log query execution time, track cache hit rate, monitor response times for risk management queries

### T072: Document risk management integration
- [ ] T072 [US3] Document risk management integration in README.md: API endpoint usage, response format, error handling, performance characteristics, example integration code

### T073: Add portfolio limit indicators (optional)
- [ ] T073 [US3] Add optional portfolio limit indicators in src/services/portfolio_manager.py: include limit_exceeded flag in response when portfolio metrics exceed configured thresholds (future enhancement)

### T074: Create unit tests for portfolio risk management
- [X] T074 [US3] Create unit tests in tests/unit/test_portfolio_manager.py: test_portfolio_exposure_calculation, test_portfolio_metrics_for_risk_management, test_cache_performance

### T075: Add portfolio metrics aggregation optimization
- [X] T075 [US3] Optimize portfolio metrics aggregation in src/services/portfolio_manager.py: use SQL aggregation functions where possible, minimize data transfer, optimize for frequent risk management queries

### T076: Implement portfolio metrics filtering by asset
- [X] T076 [US3] Implement asset filtering in src/api/routes/portfolio.py: support asset query parameter for calculating metrics for specific asset only, useful for asset-specific risk management

---

## Phase 6: User Story 4 - Historical Position Tracking (P2)

**Goal**: Analytics and model training systems can access historical position snapshots to analyze trading performance and train machine learning models.

**Independent Test Criteria**: Create position snapshots and query historical snapshots to verify past states can be accurately reconstructed.

### T077: Extract position snapshot task from Order Manager
- [X] T077 [US4] Extract PositionSnapshotTask from order-manager/src/main.py: copy PositionSnapshotTask class to src/tasks/position_snapshot_task.py, adapt imports and paths to new structure, update configuration to use POSITION_MANAGER_SNAPSHOT_INTERVAL, ensure periodic snapshot creation works with new service structure

### T077a: Adapt position snapshot task to new structure
- [X] T077a [US4] Adapt PositionSnapshotTask in src/tasks/position_snapshot_task.py: update imports to use new PositionManager service, update database connection to use src/config/database.py, update logging to use src/config/logging.py, configure execution in new service (scheduled tasks or background workers)

### T078: Implement create_position_snapshot method
- [X] T078 [US4] Implement create_position_snapshot() in src/services/position_manager.py: capture all position fields including ML features, save to position_snapshots table with JSONB snapshot_data, set created_at timestamp

### T079: Create snapshot API endpoint
- [X] T079 [US4] Create POST /api/v1/positions/{asset}/snapshot endpoint in src/api/routes/positions.py: manually trigger snapshot creation, return PositionSnapshot object, publish snapshot_created event

### T080: Create snapshot history API endpoint
- [X] T080 [US4] Create GET /api/v1/positions/{asset}/snapshots endpoint in src/api/routes/positions.py: return historical snapshots with pagination (limit, offset), sorted by created_at DESC, return SnapshotList

### T081: Extract and create snapshot cleanup task
- [X] T081 [US4] Create PositionSnapshotCleanupTask in src/tasks/position_snapshot_cleanup_task.py: create cleanup job that runs on service startup (if not exists in Order Manager, create new), delete snapshots older than POSITION_MANAGER_SNAPSHOT_RETENTION_DAYS (365 days), adapt to use new database connection and logging, log cleanup results

### T082: Create snapshot event publisher
- [X] T082 [US4] Create snapshot event publisher in src/publishers/position_event_publisher.py: publish position_snapshot_created events to position-manager.position_snapshot_created queue with complete snapshot data

### T083: Integrate snapshot task with main application
- [X] T083 [US4] Integrate snapshot task in src/main.py: schedule periodic snapshot creation, run cleanup task on startup, handle task errors gracefully

### T084: Implement snapshot data serialization
- [X] T084 [US4] Implement snapshot data serialization in src/services/position_manager.py: serialize position to JSONB format, include all fields and ML features, ensure JSON compatibility

### T085: Add snapshot query optimization
- [X] T085 [US4] Optimize snapshot queries in src/services/position_manager.py: use database indexes (idx_position_snapshots_position_id, idx_position_snapshots_created_at), implement efficient pagination, ensure <2s query time for 1-year retention

### T086: Create integration tests for snapshots
- [X] T086 [US4] Create integration tests in tests/integration/test_snapshots.py: test_snapshot_creation, test_snapshot_history_query, test_snapshot_cleanup, test_snapshot_event_publishing

---

## Phase 7: User Story 5 - Position Data Validation and Synchronization (P3)

**Goal**: System periodically validates position data against authoritative sources and automatically corrects discrepancies.

**Independent Test Criteria**: Introduce intentional discrepancies and verify validation process detects and corrects them.

### T087: Extract position validation task from Order Manager
- [X] T087 [US5] Extract PositionValidationTask from order-manager/src/main.py: copy PositionValidationTask class to src/tasks/position_validation_task.py, adapt imports and paths to new structure, update configuration to use POSITION_MANAGER_VALIDATION_INTERVAL, ensure periodic validation works with new service structure

### T087a: Adapt position validation task to new structure
- [X] T087a [US5] Adapt PositionValidationTask in src/tasks/position_validation_task.py: update imports to use new PositionManager service, update database connection to use src/config/database.py, update logging to use src/config/logging.py, configure execution in new service (scheduled tasks or background workers)

### T088: Enhance validate_position method from extracted code
- [X] T088 [US5] Enhance validate_position() in src/services/position_manager.py (extracted from Order Manager): ensure method compares position data with external sources (WebSocket, Order Manager), detect discrepancies in size, average_entry_price, PnL, return validation result, update to use new logging and configuration

### T089: Create validation API endpoint
- [X] T089 [US5] Create POST /api/v1/positions/{asset}/validate endpoint in src/api/routes/positions.py: manually trigger validation, return ValidationResult with is_valid, error_message, updated_position, support fix_discrepancies parameter

### T090: Implement discrepancy correction logic
- [X] T090 [US5] Implement discrepancy correction in src/services/position_manager.py: automatically fix discrepancies when fix_discrepancies=true, apply conflict resolution rules, update position with corrected data, log all corrections

### T091: Integrate validation task with main application
- [X] T091 [US5] Integrate validation task in src/main.py: schedule periodic validation, handle validation errors gracefully, log validation statistics

### T092: Add validation result logging
- [X] T092 [US5] Add structured logging in src/services/position_manager.py: log all validation results, log detected discrepancies, log correction actions, include trace_id in logs

### T093: Create integration tests for validation
- [ ] T093 [US5] Create integration tests in tests/integration/test_validation.py: test_position_validation, test_discrepancy_detection, test_automatic_correction, test_validation_task_scheduling

### T094: Add validation statistics tracking
- [X] T094 [US5] Add validation statistics in src/services/position_manager.py: track number of validations, discrepancies detected, corrections applied, expose via health check or metrics endpoint

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Finalize implementation with error handling, observability, documentation, deployment readiness, and integration tasks validation.

### T095: Add comprehensive error handling
- [X] T095 Add comprehensive error handling across all modules: database errors, RabbitMQ errors, external API errors, validation errors, with appropriate HTTP status codes and error messages

### T096: Add structured logging throughout
- [X] T096 Add structured logging with trace IDs throughout all modules: position operations, API requests, event processing, validation results, error conditions

### T097: Add performance monitoring
- [X] T097 Add performance monitoring: API response times, database query times, event processing times, cache hit rates, position update rates

### T098: Create comprehensive unit test suite
- [X] T098 Create comprehensive unit test suite: all services, models, utilities, with >80% code coverage

### T099: Create comprehensive integration test suite
- [X] T099 Create comprehensive integration test suite: API endpoints, event consumers, database operations, RabbitMQ integration, using testcontainers

### T100: Create E2E test suite
- [X] T100 Create E2E test suite: complete position update flows, portfolio metrics calculation, snapshot creation, validation workflows

### T101: Update API documentation
- [X] T101 Update API documentation: ensure OpenAPI spec matches implementation, add examples, update README.md with API usage examples

### T102: Add deployment documentation
- [X] T102 Add deployment documentation: docker-compose setup, environment variables, database migration requirements, health check monitoring

### T103: Add troubleshooting guide
- [X] T103 Add troubleshooting guide: common issues, error messages, debugging steps, log analysis, performance tuning

### T105: Add service health monitoring
- [X] T105 Add service health monitoring: enhance /health endpoint with detailed status, add metrics endpoint for monitoring systems, implement health check alerts

### T106: Optimize database queries
- [X] T106 Optimize database queries: review all queries for performance, ensure indexes are used, optimize portfolio metrics calculation, add query performance logging

### T107: Add request/response logging
- [X] T107 Add request/response logging: log all incoming HTTP requests with full body, log all outgoing responses, include trace IDs for request flow tracking

### T108: Add rate limiting monitoring
- [X] T108 Add rate limiting monitoring: log all rate limit exceedances, track rate limit usage per API key, add metrics for rate limiting

### T109: Final code review and cleanup
- [X] T109 Final code review and cleanup: remove unused code, optimize imports, ensure code style consistency, add docstrings, verify error messages

### T110: Update main docker-compose.yml
- [X] T110 Update main docker-compose.yml: ensure position-manager service is properly integrated, verify network configuration, test service startup

### T111: Validate and update WebSocket Gateway tasks.md
- [X] T111 Validate and update WebSocket Gateway tasks.md in specs/001-websocket-gateway/tasks.md: check Phase 7.5 (Position Channel Support, tasks T125-T136), ensure tasks related to saving positions to database (T129-T135) account for Position Manager as main service, ensure event routing tasks (T133) reflect delivery to ws-gateway.position queue for Position Manager consumption, update task documentation to reflect integration with Position Manager instead of direct use in Order Manager

### T112: Validate and update Order Manager tasks.md
- [X] T112 Validate and update Order Manager tasks.md in specs/004-order-manager/tasks.md: remove uncompleted tasks related to positions that will be implemented in Position Manager (Phase 4.5: Position Updates via WebSocket, tasks T075-T084), update descriptions of tasks partially related to positions to reflect use of Position Manager via API, update task counters in summary section, add notes that position management functionality has been moved to Position Manager service, update dependencies between phases if changed

### T113: Validate and update Model Service tasks.md
- [X] T113 Validate and update Model Service tasks.md in specs/001-model-service/tasks.md: check and update unclosed tasks related to positions and portfolio, remove outdated tasks that will be implemented in Position Manager, update tasks related to reading positions to reflect use of Position Manager via REST API, add new tasks for integration with Position Manager (REST API client, caching, event handling), update dependencies between phases if changed, update task counters in summary section

### T114: Validate and update Grafana Monitoring tasks.md
- [X] T114 Validate and update Grafana Monitoring tasks.md in specs/001-grafana-monitoring/tasks.md: add new tasks for Stage 6 (Refactor Grafana Dashboards) per spec.md Stage 6, add tasks for updating existing dashboards (trading-performance.json, trading-system-monitoring.json, order-execution-panel.json) to use Position Manager API instead of direct SQL queries to positions table, add tasks for creating new dashboard "Portfolio Management" with panels for total exposure, portfolio PnL breakdown, position size distribution, unrealized PnL by asset, time held by position, position snapshots history, add tasks for adding Position Manager Health panel to System Health dashboard, add tasks for adding Risk Management Metrics panel, add tasks for configuring Infinity datasource or PostgreSQL views for Position Manager API integration, update task counters in summary section, update dependencies between phases if changed

---

## Notes

- **Database Migration**: The migration to add `current_price` and `version` fields to `positions` table must be created in `ws-gateway` service (per constitution) and executed before Position Manager deployment.
- **Testing**: All tests should run inside Docker containers (unit tests in service container, integration/e2e tests in separate test containers).
- **Performance Goals**: Position queries <500ms (95%), portfolio metrics <1s (up to 100 positions), position updates reflected within 2s (order execution) and 5s (market data).
- **Rate Limiting**: Per-API-key rate limits with different tiers (Model Service: 100 req/min, Risk Manager: 200 req/min, UI: 1000 req/min).
- **Caching**: Portfolio metrics cached in memory with TTL 5-10 seconds, invalidated on position updates.

