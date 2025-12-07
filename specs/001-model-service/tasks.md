# Tasks: Model Service - Trading Decision and ML Training Microservice

**Input**: Design documents from `/specs/001-model-service/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are OPTIONAL - only include them if explicitly requested in the feature specification. This task list does NOT include test tasks by default.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `model-service/src/`, `model-service/tests/` at repository root
- Model files stored in `model-service/models/` (Docker volume)
- Database migrations in `ws-gateway/migrations/` (per constitution)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in model-service/
- [X] T002 Initialize Python 3.11+ project with requirements.txt (scikit-learn>=1.3.0, xgboost>=2.0.0, pandas>=2.0.0, joblib>=1.3.0, aio-pika>=9.0.0, asyncpg>=0.29.0, fastapi>=0.104.0, uvicorn[standard]>=0.24.0, pydantic-settings>=2.0.0, structlog>=23.2.0, pytest, pytest-asyncio, pytest-mock)
- [X] T003 [P] Create Dockerfile for model-service container in model-service/Dockerfile
- [X] T004 [P] Add model-service to docker-compose.yml with PostgreSQL and RabbitMQ dependencies
- [X] T005 [P] Configure environment variables in env.example (MODEL_SERVICE_PORT, MODEL_SERVICE_API_KEY, MODEL_STORAGE_PATH, training config, signal generation config, warm-up config)
- [X] T006 [P] Create model storage directory structure and Docker volume mount configuration
- [X] T007 [P] Setup project structure: model-service/src/models/, model-service/src/services/, model-service/src/api/, model-service/src/consumers/, model-service/src/publishers/, model-service/src/database/, model-service/src/config/, model-service/tests/unit/, model-service/tests/integration/, model-service/tests/e2e/
- [X] T008 [P] Configure linting and formatting tools (black, flake8, mypy) in model-service/
- [X] T009 Create README.md in model-service/ with setup instructions

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T010 Create database migration scripts in ws-gateway/migrations/003_create_model_versions_table.sql (model_versions table with indexes and constraints)
- [X] T011 Create database migration scripts in ws-gateway/migrations/004_create_model_quality_metrics_table.sql (model_quality_metrics table with indexes and constraints)
- [X] T012 [P] Implement configuration management in model-service/src/config/settings.py using pydantic-settings
- [X] T013 [P] Implement structured logging setup in model-service/src/config/logging.py using structlog with trace IDs
- [X] T014 [P] Create database connection pool in model-service/src/database/connection.py using asyncpg
- [X] T015 [P] Create database repository base class in model-service/src/database/base.py
- [X] T016 [P] Create RabbitMQ connection manager in model-service/src/config/rabbitmq.py using aio-pika
- [X] T017 [P] Implement error handling and exception classes in model-service/src/config/exceptions.py
- [X] T018 Create main application entry point in model-service/src/main.py with FastAPI app initialization
- [X] T019 [P] Implement health check endpoint in model-service/src/api/health.py (/health endpoint with database, message queue, and model storage checks)
- [X] T020 [P] Setup API routing structure in model-service/src/api/router.py with API key authentication middleware
- [X] T021 Create model storage utilities in model-service/src/services/storage.py (file system operations for model files)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Warm-up Mode Signal Generation (Priority: P1) 🎯 MVP

**Goal**: The system generates trading signals using simple heuristics or controlled random generation when no trained model exists, allowing trading to begin immediately and accumulate initial data for learning.

**Independent Test**: Can be fully tested by configuring the system in warm-up mode and verifying that it generates trading signals at the configured frequency with appropriate risk controls, publishes them to the message queue, and logs all activities. The system delivers immediate trading capability without any model dependencies.

### Implementation for User Story 1

- [X] T022 [P] [US1] Create TradingSignal data model in model-service/src/models/signal.py (signal_type, asset, amount, confidence, timestamp, strategy_id, model_version, is_warmup, market_data_snapshot with price, spread, volume_24h, volatility, optional orderbook_depth and technical_indicators, metadata, trace_id)
- [X] T022a [P] [US1] Implement market data subscription service in model-service/src/services/market_data_subscriber.py (subscribe to ws-gateway channels via REST API POST /api/v1/subscriptions for ticker, orderbook, kline channels, manage subscriptions lifecycle, handle subscription errors and retries)
- [X] T022b [P] [US1] Implement market data consumer in model-service/src/consumers/market_data_consumer.py (consume events from RabbitMQ queues ws-gateway.ticker, ws-gateway.orderbook, ws-gateway.kline, parse and cache latest values in memory for fast access, extract price, spread, volume_24h, volatility from events)
- [X] T023 [US1] Implement warm-up signal generation service in model-service/src/services/warmup_signal_generator.py (heuristics or controlled random generation with configurable frequency and risk parameters, MUST retrieve and include market data snapshot at signal generation time: price, spread, volume_24h, volatility from cached market data consumer, implement fallback to default values or skip signal generation with logging if data unavailable)
- [X] T024 [US1] Implement rate limiting service in model-service/src/services/rate_limiter.py (configurable rate limit with burst allowance for signal generation)
- [X] T025 [US1] Implement signal validation service in model-service/src/services/signal_validator.py (required fields, value ranges, format compliance)
- [X] T026 [US1] Implement trading signal publisher in model-service/src/publishers/signal_publisher.py (publish to RabbitMQ queue model-service.trading_signals with JSON format)
- [X] T027 [US1] Implement warm-up mode orchestrator in model-service/src/services/warmup_orchestrator.py (coordinates signal generation, validation, rate limiting, and publishing)
- [X] T028 [US1] Integrate warm-up mode into main application in model-service/src/main.py (start warm-up signal generation loop on service startup when no trained model exists)
- [X] T029 [US1] Add structured logging for warm-up mode operations in model-service/src/services/warmup_signal_generator.py (signal generation, rate limiting events, publishing status)
- [X] T030 [US1] Add configuration for warm-up mode parameters in model-service/src/config/settings.py (WARMUP_MODE_ENABLED, WARMUP_SIGNAL_FREQUENCY, minimum order parameters, randomness level)
- [X] T030a [US1] Add configuration for open orders check behavior in model-service/src/config/settings.py (SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true/false to enable/disable skipping signal generation when open order exists, SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY=true/false to check only opposite direction orders or all orders, default: skip if any open order exists)
- [X] T030b [P] [US1] Create AccountBalance database repository in model-service/src/database/repositories/account_balance_repo.py (read latest available balance for a coin from account_balances table using coin and ORDER BY received_at DESC LIMIT 1, handle missing balance data gracefully, support querying multiple coins)
- [X] T030c [US1] Implement balance-aware signal amount calculator in model-service/src/services/balance_calculator.py (extract base/quote currency from trading pair, determine required currency for order type buy/sell, retrieve available balance from account_balance_repo, calculate maximum affordable amount based on balance with safety margin, adapt signal amount to fit available balance if needed, return adapted amount or None if insufficient balance)
- [X] T030d [US1] Extend warm-up signal generator to use balance calculator in model-service/src/services/warmup_signal_generator.py (before generating signal amount, check available balance using balance_calculator, adapt signal amount to available balance if calculated amount exceeds balance, skip signal generation if balance is insufficient, log balance checks and amount adaptations with structured logging)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. The system can generate and publish warm-up trading signals without any trained models.

---

## Phase 4: User Story 2 - Model Training from Execution Feedback (Priority: P2)

**Goal**: The system trains or retrains ML models in real-time using feedback from order execution events, analyzing actual trade performance and market data to improve decision-making capabilities.

**Independent Test**: Can be fully tested by providing the system with order execution events and market data, verifying that it processes these into training datasets, performs model training or retraining operations, and tracks training progress and model versions. The system delivers continuous learning capability that improves over time.

### Implementation for User Story 2

- [X] T031 [P] [US2] Create OrderExecutionEvent data model in model-service/src/models/execution_event.py (event_id, order_id, signal_id, strategy_id, asset, side, execution_price, execution_quantity, execution_fees, executed_at, signal_price, signal_timestamp, market_conditions, performance, trace_id)
- [X] T032 [P] [US2] Create TrainingDataset data model in model-service/src/models/training_dataset.py (dataset_id, strategy_id, features DataFrame, labels Series, metadata dict)
- [X] T033 [P] [US2] Create ModelVersion database repository in model-service/src/database/repositories/model_version_repo.py (CRUD operations for model_versions table)
- [X] T034 [P] [US2] Create ModelQualityMetrics database repository in model-service/src/database/repositories/quality_metrics_repo.py (CRUD operations for model_quality_metrics table)
- [X] T035 [US2] Implement order execution event consumer in model-service/src/consumers/execution_event_consumer.py (subscribe to RabbitMQ queue order-manager.order_events, parse and validate events, handle corrupted/invalid events with logging and graceful continuation)
- [X] T036 [US2] Document order execution event validation rules in model-service/docs/validation-rules.md (specify required fields, value ranges, format constraints, corruption detection criteria, error handling procedures per FR-025)
- [X] T037 [US2] Implement feature engineering service in model-service/src/services/feature_engineer.py (process execution events and market data into features: MUST use market_data_snapshot from trading signals for features describing market state at decision time, use execution event market_conditions only for performance metrics, generate price features, volume features, technical indicators, market context, execution features)
- [X] T037a [US2] Extend feature engineering service to include open orders features in model-service/src/services/feature_engineer.py (add order_position_state parameter to engineer_features method, include open_orders_count, pending_buy_orders, pending_sell_orders features in training dataset, ensure these features are available at both training and inference time for model consistency)
- [X] T038 [US2] Implement label generation service in model-service/src/services/label_generator.py (extract labels from execution events: binary classification, multi-class, regression targets)
- [X] T039 [US2] Implement training dataset builder in model-service/src/services/dataset_builder.py (aggregate execution events, match execution events with corresponding trading signals by signal_id, use signal market_data_snapshot for feature engineering, apply feature engineering, generate labels from execution event performance, validate dataset quality)
- [X] T039a [US2] Extend training dataset builder to include order position state in model-service/src/services/dataset_builder.py (retrieve order_position_state from database for each execution event timestamp, pass order_position_state to feature_engineer.engineer_features to include open orders features in training data, ensure historical order state is reconstructed accurately for training dataset)
- [X] T040 [US2] Implement ML model trainer service in model-service/src/services/model_trainer.py (train XGBoost and scikit-learn models, support batch retraining, handle model serialization using joblib for scikit-learn and XGBoost native JSON format)
- [X] T041 [US2] Implement model quality evaluator in model-service/src/services/quality_evaluator.py (calculate accuracy, precision, recall, f1_score, sharpe_ratio, profit_factor, and other metrics)
- [X] T042 [US2] Implement model version manager in model-service/src/services/model_version_manager.py (create model versions, store model files in /models/v{version}/, update database metadata, handle version activation)
- [X] T043 [US2] Implement retraining trigger service in model-service/src/services/retraining_trigger.py (detect scheduled periodic retraining, data accumulation thresholds, quality degradation detection)
- [X] T044 [US2] Implement training orchestration service in model-service/src/services/training_orchestrator.py (coordinate dataset building, model training, quality evaluation, version management, handle training cancellation and restart on new triggers)
- [X] T045 [US2] Integrate training pipeline into main application in model-service/src/main.py (start execution event consumer, trigger retraining based on configured schedules and thresholds)
- [X] T046 [US2] Add structured logging for training operations in model-service/src/services/training_orchestrator.py (training start/completion, dataset size, quality metrics, version creation, cancellation events)
- [X] T047 [US2] Add configuration for training parameters in model-service/src/config/settings.py (MODEL_TRAINING_MIN_DATASET_SIZE, MODEL_TRAINING_MAX_DURATION_SECONDS, MODEL_RETRAINING_SCHEDULE, MODEL_QUALITY_THRESHOLD_ACCURACY)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. The system can train models from execution feedback and manage model versions.

---

## Phase 5: User Story 3 - Intelligent Signal Generation from Trained Model (Priority: P3)

**Goal**: The system generates high-level trading signals using trained ML models, making strategic decisions based on current market state and learned patterns from historical execution data.

**Independent Test**: Can be fully tested by providing the system with a trained model and current market state, verifying that it generates trading signals with appropriate confidence scores, publishes them to the message queue, and logs the decision-making process. The system delivers intelligent trading recommendations based on learned patterns.

### Implementation for User Story 3

- [X] T048 [P] [US3] Create OrderPositionState data model in model-service/src/models/position_state.py (read-only model for current orders and positions from shared database)
- [X] T049 [US3] Implement order/position state reader in model-service/src/database/repositories/position_state_repo.py (read current open orders and positions from shared PostgreSQL database tables)
- [X] T050 [US3] Implement model loader service in model-service/src/services/model_loader.py (load trained models from file system, validate model files, cache active models)
- [X] T051 [US3] Implement model inference service in model-service/src/services/model_inference.py (prepare features from order/position state and market data, run model prediction, generate confidence scores, MUST capture market data snapshot at inference time for inclusion in generated signals)
- [X] T052 [US3] Implement intelligent signal generator in model-service/src/services/intelligent_signal_generator.py (use model inference to generate trading signals with confidence scores, apply quality thresholds)
- [X] T052a [US3] Add open orders check in intelligent signal generator in model-service/src/services/intelligent_signal_generator.py (before generating signal, check if there are existing open orders with status 'pending' or 'partially_filled' for the same asset and strategy_id, skip signal generation if open order exists, log reason for skipping with structured logging including asset, strategy_id, existing_order_id, order_status)
- [X] T052b [US3] Extend intelligent signal generator to use balance calculator in model-service/src/services/intelligent_signal_generator.py (after model inference generates signal amount, check available balance using balance_calculator service, adapt signal amount to available balance if model-suggested amount exceeds balance, skip signal generation if balance is insufficient, log balance checks and amount adaptations with structured logging including original_model_amount and adapted_amount)
- [X] T052c [US3] [Risk Management] Implement take profit rule in intelligent signal generator in model-service/src/services/intelligent_signal_generator.py (before model inference, query Position Manager REST API GET /api/v1/positions/{asset} to read unrealized_pnl_pct, if unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT config, force generate SELL signal to close position with confidence=1.0, amount=abs(position.size), reason="take_profit_triggered", log take profit trigger with structured logging including asset, unrealized_pnl_pct, take_profit_threshold)
- [X] T052d [US3] [Risk Management] Implement position size limit check in intelligent signal generator in model-service/src/services/intelligent_signal_generator.py (before generating BUY signal, query Position Manager REST API GET /api/v1/positions/{asset} to read position_size_norm, if position_size_norm > MODEL_SERVICE_MAX_POSITION_SIZE_RATIO config, skip BUY signal generation, record skip with signal_skip_metrics.record_skip with reason="position_size_limit", log skip with structured logging including asset, position_size_norm, max_ratio_threshold)
- [X] T052e [US3] [Risk Management] Add configuration for risk management rules in model-service/src/config/settings.py (MODEL_SERVICE_TAKE_PROFIT_PCT=3.0 for take profit threshold in percentage, MODEL_SERVICE_MAX_POSITION_SIZE_RATIO=0.8 for maximum position size ratio relative to total exposure; both rules depend on Position Manager as the single source of truth for positions)
- [X] T053 [US3] Implement mode transition service in model-service/src/services/mode_transition.py (automatically transition from warm-up mode to model-based generation when model quality reaches configured threshold)
- [X] T054 [US3] Integrate intelligent signal generation into main application in model-service/src/main.py (replace warm-up mode with model-based generation when active model is available)
- [X] T055 [US3] Add structured logging for model-based signal generation in model-service/src/services/intelligent_signal_generator.py (signal generation, model inference results, confidence scores, mode transitions)
- [X] T056 [US3] Update signal publisher to include model_version and is_warmup fields in model-service/src/publishers/signal_publisher.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently. The system can generate signals using trained models and automatically transition from warm-up to model-based mode.

---

## Phase 6: User Story 4 - Model Quality Tracking and Versioning (Priority: P4)

**Goal**: The system tracks model quality metrics, maintains version history, and provides observability into model performance and system behavior for monitoring and debugging.

**Independent Test**: Can be fully tested by generating multiple model versions, executing trades, and verifying that quality metrics are calculated, version history is maintained, all operations are logged appropriately, and monitoring data is available. The system delivers full observability into model performance and system health.

### Implementation for User Story 4

- [X] T057 [P] [US4] Implement model versions list API endpoint in model-service/src/api/models.py (GET /api/v1/models with filtering by strategy_id, is_active, pagination)
- [X] T058 [P] [US4] Implement model version details API endpoint in model-service/src/api/models.py (GET /api/v1/models/{version} with full details and quality metrics)
- [X] T059 [P] [US4] Implement model activation API endpoint in model-service/src/api/models.py (POST /api/v1/models/{version} to activate a model version, deactivate previous active model)
- [X] T060 [P] [US4] Implement model quality metrics API endpoint in model-service/src/api/metrics.py (GET /api/v1/models/{version}/metrics with filtering by metric_type)
- [X] T061 [P] [US4] Implement training status API endpoint in model-service/src/api/training.py (GET /api/v1/training/status with current training info, last training, next scheduled training)
- [X] T062 [P] [US4] Implement manual training trigger API endpoint in model-service/src/api/training.py (POST /api/v1/training/trigger to manually trigger training for a strategy)
- [X] T063 [US4] Implement model version history service in model-service/src/services/version_history.py (query and manage version history, support rollback operations)
- [X] T064 [US4] Implement quality monitoring service in model-service/src/services/quality_monitor.py (periodically evaluate model quality, detect degradation, trigger alerts)
- [X] T065 [US4] Add comprehensive structured logging for all operations in model-service/src/services/ (signal generation, model training, mode transitions, quality evaluation, version management)
- [X] T066 [US4] Implement model rollback functionality in model-service/src/services/model_version_manager.py (switch to previous model version, update is_active flags)
- [X] T067 [US4] Add monitoring and observability endpoints in model-service/src/api/monitoring.py (model performance metrics, system health details, active models count)
- [X] T068 [US4] Implement model cleanup policy in model-service/src/services/model_cleanup.py (keep last N versions, archive old versions, manage disk space)
- [X] T083 [P] [US4] Create database migration script in ws-gateway/migrations/009_create_execution_events_table.sql (execution_events table with columns: id, signal_id, strategy_id, asset, side, execution_price, execution_quantity, execution_fees, executed_at, signal_price, signal_timestamp, performance JSONB, created_at, indexes on executed_at, signal_id, strategy_id for time-series queries)
- [X] T084 [US4] Extend execution event consumer in model-service/src/consumers/execution_event_consumer.py to persist execution events to PostgreSQL database (save to execution_events table after validation, handle database errors gracefully, continue processing on persistence failures)
- [X] T085 [US4] Extend quality monitoring service in model-service/src/services/quality_monitor.py to periodically evaluate and store metrics (evaluate model quality every hour based on recent execution events, calculate metrics: win_rate, sharpe_ratio, profit_factor, total_pnl, save to model_quality_metrics table with evaluated_at timestamp, support configurable evaluation interval)
- [X] T086 [P] [US4] Implement time-series metrics API endpoint in model-service/src/api/metrics.py (GET /api/v1/models/{version}/metrics/time-series with parameters: granularity=hour|day|week, start_time, end_time, metric_names, return time-series data for charting, support aggregation by specified granularity)
- [X] T087 [P] [US4] Implement strategy performance time-series API endpoint in model-service/src/api/monitoring.py (GET /api/v1/strategies/{strategy_id}/performance/time-series with parameters: granularity=hour|day|week, start_time, end_time, return metrics: success_rate, total_pnl, avg_pnl, total_orders, successful_orders aggregated by time granularity from execution_events table)

**Checkpoint**: All user stories should now be independently functional with full observability and version management capabilities.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T069 [P] Update README.md in model-service/ with complete setup, usage, and API documentation (fix queue name: order-manager.order_events instead of order-manager.execution_events)
- [X] T070 [P] Synchronize quickstart.md in specs/001-model-service/ with actual implementation (fix queue name: order-manager.order_events instead of order-manager.execution_events)
- [X] T088 [P] Fix queue name references in documentation: update data-model.md in specs/001-model-service/ to use correct queue name order-manager.order_events instead of order-manager.execution_events
- [X] T089 [P] [US1] Create database migration script in ws-gateway/migrations/010_create_trading_signals_table.sql (trading_signals table with columns: id UUID PRIMARY KEY, signal_id UUID NOT NULL UNIQUE, strategy_id VARCHAR(100) NOT NULL, asset VARCHAR(50) NOT NULL, side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')), price DECIMAL(20, 8) NOT NULL, confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1), timestamp TIMESTAMP NOT NULL, model_version VARCHAR(50), is_warmup BOOLEAN NOT NULL DEFAULT false, market_data_snapshot JSONB, metadata JSONB, trace_id VARCHAR(100), created_at TIMESTAMP NOT NULL DEFAULT NOW(), indexes on timestamp DESC, signal_id, strategy_id, asset for Grafana dashboard queries)
- [X] T090 [US1] Extend signal publisher in model-service/src/publishers/signal_publisher.py to persist trading signals to PostgreSQL database (save to trading_signals table after successful RabbitMQ publish, handle database errors gracefully with logging and continue processing, ensure signals are persisted for Grafana monitoring dashboard visibility)
- [X] T071 [P] Add error handling and retry logic for RabbitMQ operations in model-service/src/publishers/ and model-service/src/consumers/
- [X] T072 [P] Add error handling and retry logic for database operations in model-service/src/database/
- [X] T073 [P] Implement graceful shutdown handling in model-service/src/main.py (close connections, finish in-progress operations)
- [X] T074 [P] Add request/response logging middleware in model-service/src/api/middleware.py (log all API requests and responses with trace IDs)
- [X] T075 [P] Add comprehensive error responses with proper HTTP status codes in model-service/src/api/
- [X] T076 [P] Implement model file health checks in model-service/src/services/storage.py (verify file existence, permissions, disk space)
- [X] T077 [P] Add configuration validation on startup in model-service/src/config/settings.py (validate all required settings, check file paths, verify database connectivity)
- [X] T078 [P] Update docker-compose.yml to ensure proper service dependencies and health checks
- [X] T079 [P] Add trace ID propagation across all services (consumers, publishers, API endpoints)
- [ ] T080 Run quickstart.md validation to ensure all examples work correctly
- [ ] T081 Code cleanup and refactoring (remove unused code, improve code organization)
- [ ] T082 Performance optimization (connection pooling, model caching, efficient data processing)
- [X] T091 [P] Add metrics for signal generation skipping in model-service/src/services/intelligent_signal_generator.py (track count of signals skipped due to open orders, log metrics with asset, strategy_id, reason, expose via monitoring endpoint for observability)
- [X] T092 [P] Update documentation in model-service/README.md to describe open orders check behavior and configuration options (explain how signal generation prevents duplicate orders, document configuration parameters SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER and SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY, provide examples of behavior)
- [X] T093 [P] [Grafana] Standardize health check endpoint response format for Grafana monitoring: Update HealthResponse in model-service/src/api/health.py to include flat fields database_connected (boolean from checks.database.connected) and queue_connected (boolean from checks.message_queue.connected) in addition to existing checks object. This enables Grafana System Health dashboard panel to extract dependency status directly without nested JSON parsing. Alternatively, update health endpoint to return both formats (flat fields for backward compatibility and checks object for detailed status). Required for Grafana dashboard User Story 5 compliance.
- [X] T094 [P] [Optimization] Implement Position Manager event consumer for cache invalidation in model-service/src/consumers/position_update_consumer.py (subscribe to RabbitMQ queue position-manager.position_updated, parse position update events, invalidate local cache for affected assets, handle event parsing errors gracefully, log cache invalidation events with structured logging including asset, trace_id)
- [X] T095 [P] [Optimization] Implement in-memory cache for position data in model-service/src/services/position_cache.py (cache position data from Position Manager REST API with TTL, support cache invalidation by asset, provide get/set/invalidate methods, handle cache expiration, support configurable cache size limits, thread-safe operations for concurrent access)
- [X] T096 [Optimization] Integrate position cache with PositionManagerClient in model-service/src/services/position_manager_client.py (check cache before making REST API request, update cache after successful API response, invalidate cache on position update events, fallback to REST API if cache miss or expired, log cache hits/misses for monitoring)
- [X] T097 [Optimization] Add configuration for position cache in model-service/src/config/settings.py (POSITION_CACHE_ENABLED=true/false to enable/disable caching, POSITION_CACHE_TTL_SECONDS=30 for cache time-to-live, POSITION_CACHE_MAX_SIZE=1000 for maximum cached positions, default: caching enabled with 30s TTL for optimization while REST API remains primary source of truth)
- [ ] T117 [P] [Enhancement] Implement automatic asset selection based on account balances in model-service/src/services/asset_selector.py (query account_balances table to identify currencies with non-zero available_balance, extract base currencies from trading pairs, generate list of trading pairs for currencies with balance, support configurable quote currency list USDT/USDC/BUSD, filter out pairs with insufficient balance below minimum threshold, cache asset list with TTL to reduce database queries, handle missing balance data gracefully with fallback to configured asset list)
- [ ] T118 [Enhancement] Integrate asset selector into intelligent orchestrator in model-service/src/services/intelligent_orchestrator.py (replace hardcoded assets list with automatic asset selection when enabled, use asset_selector.get_trading_assets_with_balance() instead of hardcoded ["BTCUSDT", "ETHUSDT"], add configuration flag AUTO_SELECT_ASSETS=true/false to enable/disable automatic selection, fallback to configured asset list if automatic selection fails or returns empty list, log asset selection results with structured logging)
- [ ] T119 [Enhancement] Integrate asset selector into warmup orchestrator in model-service/src/services/warmup_orchestrator.py (replace hardcoded assets list with automatic asset selection when enabled, use same asset_selector service as intelligent orchestrator, respect AUTO_SELECT_ASSETS configuration flag, fallback to configured asset list if automatic selection fails)
- [ ] T120 [Enhancement] Add configuration for automatic asset selection in model-service/src/config/settings.py (AUTO_SELECT_ASSETS=true/false to enable/disable automatic asset selection based on balances, AUTO_SELECT_MIN_BALANCE_USDT=10.0 for minimum balance threshold in USDT equivalent, AUTO_SELECT_QUOTE_CURRENCIES=USDT,USDC,BUSD for supported quote currencies, AUTO_SELECT_CACHE_TTL_SECONDS=300 for asset list cache TTL, AUTO_SELECT_FALLBACK_ASSETS=BTCUSDT,ETHUSDT for fallback when automatic selection fails, default: automatic selection disabled for backward compatibility)
- [ ] T121 [Enhancement] Update main.py to use asset selector in model-service/src/main.py (replace hardcoded assets list with asset_selector when AUTO_SELECT_ASSETS is enabled, initialize asset_selector service on startup, handle asset selection errors gracefully with fallback to configured list)
- [ ] T122 [Enhancement] Add structured logging for asset selection in model-service/src/services/asset_selector.py (log selected assets with balance information, log cache hits/misses, log fallback to configured list, include trace_id for request flow tracking)
- [X] T141 [Enhancement] Investigate and fix balance-aware signal amount calculation discrepancies in model-service/src/services/intelligent_signal_generator.py and model-service/src/services/balance_calculator.py (analyze why signals with adapted amounts are still rejected by order-manager due to insufficient balance, compare balance data sources: Model Service reads from account_balances table vs Order Manager reads from exchange API, investigate data freshness: check TTL of balance data in account_balances table, verify balance_calculator.calculate_affordable_amount() logic: ensure safety margin (0.95) accounts for fees and slippage, add logging to compare adapted_amount from Model Service with balance check results in Order Manager, investigate race conditions: balance may change between signal generation and order creation, add balance data freshness check: warn if balance data is older than threshold (e.g., 60 seconds), improve balance adaptation logic: consider exchange fees, slippage, and minimum order size requirements, add configuration BALANCE_ADAPTATION_SAFETY_MARGIN=0.90 for configurable safety margin (default 0.95), add configuration BALANCE_DATA_MAX_AGE_SECONDS=60 for maximum acceptable age of balance data, add metrics for balance adaptation: track adapted_amount vs requested_amount, track rejection rate due to insufficient balance, add integration test to verify balance adaptation works correctly with realistic balance scenarios, document balance adaptation behavior in model-service/README.md)
- [X] T142 [Enhancement] Add market data cache freshness check in model-service/src/consumers/market_data_consumer.py (extend MarketDataCache.get_market_data() method to accept optional max_age_seconds parameter, check if last_updated timestamp is older than max_age_seconds threshold, log warning with symbol and age_seconds if data is stale, return None if data exceeds max age to force signal generation to skip, add configuration MARKET_DATA_MAX_AGE_SECONDS=60 for maximum acceptable age of cached market data, default: 60 seconds to prevent using stale data for signal generation, ensure backward compatibility: if max_age_seconds not provided, return data without age check for existing behavior)
- [X] T143 [Enhancement] Add signal generation delay monitoring in model-service/src/publishers/signal_publisher.py (log timestamp when signal is created and published, calculate and log time delta between signal creation and publication, add structured logging fields: signal_creation_time, signal_publication_time, processing_delay_seconds, add metrics for monitoring: track average processing delay, maximum delay, delay distribution, add configuration SIGNAL_PROCESSING_DELAY_ALERT_THRESHOLD_SECONDS=300 for alert threshold (default: 5 minutes), log alert if processing delay exceeds threshold, expose metrics via monitoring API endpoint for observability)
- [X] T144 [Enhancement] Add configuration for market data cache freshness in model-service/src/config/settings.py (MARKET_DATA_MAX_AGE_SECONDS=60 for maximum acceptable age of cached market data in seconds, MARKET_DATA_STALE_WARNING_THRESHOLD_SECONDS=30 for warning threshold before max age, default: 60 seconds max age, 30 seconds warning threshold, document configuration options in model-service/README.md)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete
- **Feature Service Integration (Phase 10)**: Depends on Feature Service being deployed and providing features.live queue, REST API endpoints (/features/latest, /dataset/build, /dataset/{id}/download), and features.dataset.ready queue. Can be implemented after Phase 3-6 are complete. **CRITICAL**: Feature Service must be available before Phase 10 can be completed.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories - **MVP SCOPE**
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on US1, but US3 depends on US2
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) and US2 (needs trained models) - Depends on US2 for model availability
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Depends on US2 for model versions and quality metrics, but can be implemented in parallel with US3

### Within Each User Story

- Models/data structures before services
- Services before endpoints/consumers/publishers
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T003-T009)
- All Foundational tasks marked [P] can run in parallel (T012-T017, T019-T021)
- Once Foundational phase completes, User Stories 1, 2, and 4 can start in parallel (if team capacity allows)
- User Story 3 must wait for User Story 2 to complete (needs trained models)
- All API endpoints in US4 marked [P] can run in parallel (T057-T062, T086-T087)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members (except US3 which depends on US2)

### Task Dependencies for Time-Series Reporting (T083-T087)

- **T083** (execution_events table): Can be done independently, must complete before T084
- **T084** (persist execution events): Depends on T083 (table exists) and T035 (execution event consumer exists)
- **T085** (periodic quality evaluation): Depends on T064 (quality monitor exists) and T084 (execution events in DB)
- **T086** (time-series metrics API): Depends on T060 (base metrics endpoint) and T085 (periodic metrics available)
- **T087** (time-series performance API): Depends on T084 (execution events in DB)

### Task Dependencies for Trading Signals Persistence (T089-T090)

- **T089** (trading_signals table): Can be done independently, must complete before T090
- **T090** (persist trading signals): Depends on T089 (table exists) and T026 (signal publisher exists). Extends T026 to add database persistence for Grafana monitoring dashboard visibility

### Task Dependencies for Open Orders Check and Duplicate Prevention (T030a, T037a, T039a, T052a, T091-T092)

- **T030a** (configuration): Can be done independently, must complete before T052a (needs configuration values)
- **T037a** (feature engineering): Depends on T037 (feature_engineer exists), must complete before T039a (dataset builder needs extended feature engineer)
- **T039a** (dataset builder): Depends on T037a (extended feature engineer) and T039 (dataset_builder exists), must complete before model training can use open orders features
- **T052a** (open orders check): Depends on T030a (configuration), T049 (position_state_repo exists), and T052 (intelligent_signal_generator exists). Extends T052 to add duplicate prevention logic
- **T091** (metrics): Depends on T052a (open orders check exists) to track skipped signals
- **T092** (documentation): Can be done independently, but should complete after T030a and T052a to document implemented behavior

### Task Dependencies for Risk Management Rules (T052c-T052e)

- **T052e** (risk management configuration): Can be done independently, must complete before T052c and T052d (needs configuration values)
- **T052c** (take profit rule): Depends on T052e (configuration), T052 (intelligent_signal_generator exists), and Position Manager service availability (REST API endpoint GET /api/v1/positions/{asset}). Extends T052 to add take profit logic before model inference, using Position Manager as the sole source of truth for unrealized_pnl_pct
- **T052d** (position size limit check): Depends on T052e (configuration), T052 (intelligent_signal_generator exists), and Position Manager service availability (REST API endpoint GET /api/v1/positions/{asset}). Extends T052 to add position size limit check before BUY signal generation, using Position Manager as the sole source of truth for position_size_norm
- **Note**: T052c and T052d can be implemented in parallel, both require Position Manager service to be deployed and accessible

### Task Dependencies for Balance-Aware Signal Generation (T030b, T030c, T030d, T052b)

- **T030b** (account balance repository): Can be done independently, must complete before T030c (balance calculator needs repository to read balances from database)
- **T030c** (balance calculator): Depends on T030b (account_balance_repo exists), must complete before T030d and T052b (signal generators need balance calculator to adapt amounts)
- **T030d** (warm-up balance integration): Depends on T030c (balance_calculator exists) and T023 (warmup_signal_generator exists). Extends T023 to add balance-aware signal amount calculation
- **T052b** (intelligent balance integration): Depends on T030c (balance_calculator exists) and T052 (intelligent_signal_generator exists). Extends T052 to add balance-aware signal amount calculation

### Task Dependencies for Position Cache Optimization (T094-T097)

- **T095** (position cache service): Can be done independently, must complete before T096 (PositionManagerClient needs cache service to use)
- **T097** (cache configuration): Can be done independently, must complete before T095 and T096 (cache service needs configuration values)
- **T096** (cache integration with client): Depends on T095 (position_cache exists) and T097 (configuration exists), must complete after T052c and T052d (risk management rules use PositionManagerClient)
- **T094** (position update consumer): Depends on T095 (position_cache exists for invalidation), must complete after T096 (cache integration is working). Consumer invalidates cache when position updates are received from Position Manager via RabbitMQ events (`position-manager.position_updated` queue)

### Task Dependencies for Feature Service Integration (T145-T172)

**Note**: This integration implements training on market data only (no execution_events). Model Service learns from market movements, not from own trading results.

- **T145** (Feature Service HTTP client): Can be done independently, must complete before T149, T152 (services need client to request features and datasets)
- **T146** (Feature Vector model): Can be done independently, must complete before T147, T148 (consumer and inference need model)
- **T147** (feature consumer): Depends on T146 (Feature Vector model exists), must complete before T150 (main app integration)
- **T148** (ModelInference update): Depends on T146 (Feature Vector model exists), must complete before T149 (IntelligentSignalGenerator uses updated inference)
- **T149** (IntelligentSignalGenerator update): Depends on T145 (HTTP client exists), T148 (updated inference exists), must complete before T150 (main app integration)
- **T150** (main app integration): Depends on T147 (feature consumer exists), T149 (updated signal generator exists), T145 (HTTP client exists)
- **T151** (dataset ready consumer): Can be done independently, must complete before T152 (training orchestrator needs consumer)
- **T152** (TrainingOrchestrator update): Depends on T145 (HTTP client exists), T151 (dataset consumer exists), T158 (retraining trigger update), T171 (time-based config), T172 (period calculation), must complete after T044 (training orchestrator exists). Removes execution_events dependency from training pipeline.
- **T153** (remove FeatureEngineer): Can be done independently, but should complete after T148, T149, T152 (ensure all feature engineering code is replaced)
- **T154** (remove market data subscriber): Can be done independently, but should complete after T149 (signal generator no longer needs market data subscription)
- **T155** (remove market data consumer): Can be done independently, but should complete after T147 (feature consumer replaces market data consumer)
- **T156** (Feature Service configuration): Can be done independently, must complete before T145, T147, T149, T152 (services need configuration values)
- **T157** (feature alignment validation): Depends on T148 (ModelInference update exists), can be done in parallel with T149
- **T158** (retraining trigger update): Can be done independently, must complete before T152 (TrainingOrchestrator needs updated triggers). Removes execution_events accumulation threshold.
- **T159** (dataset building configuration): Can be done independently, must complete before T152 (services need configuration values)
- **T160** (REST API fallback): Depends on T145 (HTTP client exists), T149 (signal generator exists), can be done in parallel with T161
- **T161** (structured logging): Can be done independently, but should complete after T145-T160 (to log all integration operations)
- **T164** (remove execution event consumer from training): Can be done independently, but should complete after T152, T165, T168 (training pipeline no longer uses execution_events)
- **T165** (remove buffer persistence): Can be done independently, but should complete after T152, T168 (training pipeline no longer uses buffer)
- **T166** (remove dataset builder): Can be done independently, but should complete after T152 (Feature Service replaces dataset builder)
- **T167** (remove label generator): Can be done independently, but should complete after T152 (Feature Service generates labels/targets)
- **T168** (remove execution events buffer): Depends on T152 (TrainingOrchestrator updated), must complete after T152 to remove buffer from orchestrator
- **T169** (update quality evaluator): Can be done independently, but should complete after T170 (remove execution_events dependency from quality monitoring)
- **T170** (update quality monitor): Can be done independently, removes execution_events dependency from quality monitoring
- **T171** (time-based retraining config): Can be done independently, must complete before T152, T172 (services need configuration values)
- **T172** (dataset period calculation): Depends on T171 (time-based config exists), must complete before T152 (TrainingOrchestrator needs period calculation)
- **T162, T163** (documentation updates): Can be done independently, but should complete after T145-T172 (to document implemented integration including market-data-only training approach)

### Task Dependencies for Test Split Evaluation (T173-T180)

**Note**: These tasks implement proper test set evaluation for final quality assessment and model activation, aligning with docs/feature-service.md workflow.

- **T173** (test split evaluation): Depends on T152 (TrainingOrchestrator updated), T174 (save_quality_metrics supports dataset_split), must complete after T152 to add test set evaluation to training pipeline
- **T174** (save_quality_metrics update): Can be done independently, must complete before T173 (test split evaluation needs updated method to save metrics with dataset_split metadata)
- **T175** (quality metrics repository filtering): Can be done independently, must complete before T177, T178 (mode transition and retraining trigger need filtering by dataset_split)
- **T176** (update SC-005): Can be done independently, but should complete after T173 to align specification with implementation
- **T177** (mode transition update): Depends on T175 (repository filtering exists), T173 (test split evaluation exists), must complete after T173 to use test set metrics for mode transitions
- **T178** (retraining trigger update): Depends on T175 (repository filtering exists), T173 (test split evaluation exists), must complete after T173 to use test set metrics for quality degradation detection
- **T179** (README documentation): Can be done independently, but should complete after T173-T178 (to document implemented test set evaluation workflow)
- **T180** (quickstart documentation): Can be done independently, but should complete after T173-T178 (to synchronize quickstart with test set evaluation)
- **T181** (warning logging): Depends on T173 (test split evaluation exists), can be done in parallel with T179, T180 (logging enhancement for operational awareness)
- **Note**: T174 and T175 can be done in parallel, T177 and T178 can be done in parallel after T175 and T173 are complete, T181 can be done in parallel with T179, T180

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create TradingSignal data model in model-service/src/models/signal.py"
Task: "Create OrderExecutionEvent data model in model-service/src/models/execution_event.py" (if starting US2 in parallel)

# Launch all foundational services together:
Task: "Implement warm-up signal generation service in model-service/src/services/warmup_signal_generator.py"
Task: "Implement rate limiting service in model-service/src/services/rate_limiter.py"
Task: "Implement signal validation service in model-service/src/services/signal_validator.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch all data models and repositories together:
Task: "Create OrderExecutionEvent data model in model-service/src/models/execution_event.py"
Task: "Create TrainingDataset data model in model-service/src/models/training_dataset.py"
Task: "Create ModelVersion database repository in model-service/src/database/repositories/model_version_repo.py"
Task: "Create ModelQualityMetrics database repository in model-service/src/database/repositories/quality_metrics_repo.py"

# Launch all service components together (after models):
Task: "Implement feature engineering service in model-service/src/services/feature_engineer.py"
Task: "Implement label generation service in model-service/src/services/label_generator.py"
Task: "Implement ML model trainer service in model-service/src/services/model_trainer.py"
Task: "Implement model quality evaluator in model-service/src/services/quality_evaluator.py"
```

---

## Parallel Example: User Story 4

```bash
# Launch all API endpoints together:
Task: "Implement model versions list API endpoint in model-service/src/api/models.py"
Task: "Implement model version details API endpoint in model-service/src/api/models.py"
Task: "Implement model activation API endpoint in model-service/src/api/models.py"
Task: "Implement model quality metrics API endpoint in model-service/src/api/metrics.py"
Task: "Implement training status API endpoint in model-service/src/api/training.py"
Task: "Implement manual training trigger API endpoint in model-service/src/api/training.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Warm-up Mode Signal Generation)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Verify warm-up signals are generated at configured frequency
   - Verify signals are published to RabbitMQ
   - Verify rate limiting works correctly
   - Verify all operations are logged
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo (Model Training)
4. Add User Story 3 → Test independently → Deploy/Demo (Intelligent Signals)
5. Add User Story 4 → Test independently → Deploy/Demo (Quality Tracking)
6. Add User Story 5 → Test independently → Deploy/Demo (Position-Based Exit Strategy)
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Warm-up Mode) - **MVP**
   - Developer B: User Story 2 (Model Training) - can start in parallel
   - Developer C: User Story 4 (Quality Tracking) - can start in parallel with US2
3. After US2 completes:
   - Developer B: User Story 3 (Intelligent Signals) - depends on US2
4. After US3 completes:
   - Developer A: User Story 5 (Position-Based Exit Strategy) - can start in parallel with US4
5. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Model files stored in `/models/v{version}/` directory structure
- Database migrations must be in `ws-gateway/migrations/` per constitution
- All API endpoints require API key authentication (X-API-Key header)
- All operations must include structured logging with trace IDs
- Rate limiting must be enforced on signal generation with burst allowance
- Training operations must support cancellation and restart on new triggers

---

## Task Summary

- **Total Tasks**: 173 tasks (added T052c, T052d, T052e for risk management rules, T094-T097 for position cache optimization, T098-T116 for position-based exit strategy, T123-T140 for training orchestrator improvements, T141 for balance adaptation investigation and fixes, T142-T144 for market data cache freshness and signal delay monitoring, T145-T163 for Feature Service integration, T173-T181 for test split evaluation and related updates)
- **Phase 1 (Setup)**: 9 tasks
- **Phase 2 (Foundational)**: 12 tasks
- **Phase 3 (User Story 1 - MVP)**: 15 tasks (added T030a, T030b, T030c, T030d)
- **Phase 4 (User Story 2)**: 19 tasks (added T037a, T039a)
- **Phase 5 (User Story 3)**: 14 tasks (added T052a, T052b, T052c, T052d, T052e for risk management)
- **Phase 6 (User Story 4)**: 17 tasks
- **Phase 7 (Polish)**: 35 tasks (added T091, T092, T093, T094-T097 for position cache optimization, T117-T122 for automatic asset selection, T141 for balance adaptation investigation and fixes, T142-T144 for market data cache freshness and signal delay monitoring)
- **Phase 8 (User Story 5)**: 19 tasks (position-based exit strategy)
- **Phase 9 (Training Orchestrator Improvements)**: 18 tasks (T123-T140 for persistent buffer, training queue, and additional improvements)
- **Phase 10 (Feature Service Integration)**: 20 tasks (T145-T163 for integrating with Feature Service, removing feature engineering code, using ready features for inference and training)
- **Phase 11 (Test Split Evaluation)**: 9 tasks (T173-T181 for implementing test set evaluation, updating quality metrics handling, updating mode transition and retraining triggers, updating documentation and specifications, adding warning logging for empty test splits)

### Task Count per User Story

- **User Story 1 (P1 - MVP)**: 14 tasks (added T030b, T030c, T030d)
- **User Story 2 (P2)**: 17 tasks
- **User Story 3 (P3)**: 13 tasks (added T052b, T052c, T052d, T052e for risk management rules)
- **User Story 4 (P4)**: 17 tasks
- **User Story 5 (P3)**: 19 tasks (position-based exit strategy)

### Parallel Opportunities Identified

- **Setup Phase**: 6 parallel tasks (T003-T008)
- **Foundational Phase**: 9 parallel tasks (T012-T017, T019-T021)
- **User Story 1**: 3 parallel tasks (T022, T022a, T022b)
- **User Story 2**: 4 parallel tasks (T031-T034), plus T036 can run in parallel with T037+
- **User Story 4**: 8 parallel tasks (T057-T062, T083, T086-T087)
- **User Story 5**: 6 parallel tasks (T098, T099, T101-T104, T105)

### Independent Test Criteria

- **User Story 1**: Configure system in warm-up mode, verify signal generation, publishing, and logging without any trained models
- **User Story 2**: Provide execution events and market data, verify training pipeline, model versioning, and quality tracking
- **User Story 3**: Provide trained model and market state, verify intelligent signal generation and mode transitions
- **User Story 4**: Generate multiple model versions, verify quality metrics, version history, and API endpoints
- **User Story 5**: Simulate position updates with various PnL values and holding times, verify exit signals are generated when rules trigger, rate limiting prevents excessive signals

## Phase 8: User Story 5 - Position-Based Exit Strategy (Priority: P3)

**Goal**: The system reacts to position updates in real-time and generates exit signals (SELL) based on configurable exit rules such as take profit, stop loss, trailing stop, and time-based exits, enabling proactive risk management and profit protection.

**Independent Test**: Can be fully tested by simulating position updates with various unrealized PnL values, holding times, and market conditions, verifying that exit signals are generated when rules are triggered, rate limiting prevents excessive signal generation, and all exit decisions are logged with traceability.

### Implementation for User Story 5

- [X] T098 [P] [US5] Create ExitDecision data model in model-service/src/models/exit_decision.py (should_exit: bool, exit_reason: str, exit_amount: float, priority: int, rule_triggered: str, metadata: dict)
- [X] T099 [P] [US5] Create PositionState data model for tracking in model-service/src/models/position_state_tracker.py (asset: str, entry_price: float, entry_time: datetime, peak_price: float, highest_unrealized_pnl: float, last_exit_signal_time: Optional[datetime])
- [X] T100 [US5] Create ExitStrategyEvaluator service in model-service/src/services/exit_strategy_evaluator.py (evaluates all active exit rules, applies rules in priority order, returns ExitDecision)
- [X] T101 [P] [US5] Create TakeProfitRule in model-service/src/services/exit_rules/take_profit_rule.py (percentage-based threshold, absolute value threshold optional, partial exit support, extends base ExitRule class)
- [X] T102 [P] [US5] Create StopLossRule in model-service/src/services/exit_rules/stop_loss_rule.py (percentage-based stop loss, absolute value stop loss, extends base ExitRule class)
- [X] T103 [P] [US5] Create TrailingStopRule in model-service/src/services/exit_rules/trailing_stop_rule.py (activation threshold, trailing distance, lock-in mechanism, requires position state tracking)
- [X] T104 [P] [US5] Create TimeBasedExitRule in model-service/src/services/exit_rules/time_based_exit_rule.py (maximum holding time, time-based profit targets, decay function support)
- [X] T105 [P] [US5] Create base ExitRule abstract class in model-service/src/services/exit_rules/base.py (defines interface: evaluate(position_data, position_state) -> Optional[ExitDecision], enabled flag, priority)
- [X] T106 [US5] Create PositionStateTracker service in model-service/src/services/position_state_tracker.py (tracks entry price/time, peak price, highest PnL, last exit signal time, persists to Redis or database)
- [X] T107 [US5] Create PositionBasedSignalGenerator service in model-service/src/services/position_based_signal_generator.py (evaluates position exit on position updates, generates SELL signals when exit rules trigger, integrates with signal publisher)
- [X] T108 [US5] Create ExitSignalRateLimiter service in model-service/src/services/exit_signal_rate_limiter.py (per-asset rate limiting, cooldown period after exit signal, maximum signals per time window, prevents excessive signal generation)
- [X] T109 [US5] Extend PositionUpdateConsumer in model-service/src/consumers/position_update_consumer.py (extract position data from events, trigger exit strategy evaluation via PositionBasedSignalGenerator, handle evaluation errors gracefully)
- [X] T110 [US5] Add exit strategy configuration to model-service/src/config/settings.py (EXIT_STRATEGY_ENABLED, EXIT_STRATEGY_RATE_LIMIT, TAKE_PROFIT_ENABLED, TAKE_PROFIT_PARTIAL_EXIT, TAKE_PROFIT_PARTIAL_AMOUNT_PCT, STOP_LOSS_ENABLED, STOP_LOSS_THRESHOLD_PCT, TRAILING_STOP_ENABLED, TRAILING_STOP_ACTIVATION_PCT, TRAILING_STOP_DISTANCE_PCT, TIME_BASED_EXIT_ENABLED, TIME_BASED_EXIT_MAX_HOURS, TIME_BASED_EXIT_PROFIT_TARGET_PCT)
Note: Take profit threshold uses MODEL_SERVICE_TAKE_PROFIT_PCT (unified with intelligent signal generator)
- [X] T111 [US5] Integrate PositionBasedSignalGenerator into main application in model-service/src/main.py (initialize on startup, connect to PositionUpdateConsumer, handle graceful shutdown)
- [X] T112 [US5] Add structured logging for exit strategy evaluation in model-service/src/services/exit_strategy_evaluator.py (log all rule evaluations, exit decisions, rate limiting events, position state updates)
- [X] T113 [US5] Add metrics for exit strategy operations in model-service/src/services/position_based_signal_generator.py (exit signals generated, rules triggered, rate limiting events, evaluation latency)
- [X] T114 [US5] Implement debouncing for rapid position updates in model-service/src/services/position_based_signal_generator.py (prevent excessive evaluation when multiple updates arrive quickly for same asset, configurable debounce window)
- [X] T115 [US5] Add fallback to periodic evaluation mode in model-service/src/services/position_based_signal_generator.py (when event-driven processing unavailable, fall back to periodic position checks, log degradation)
- [X] T116 [US5] Add position update event validation in model-service/src/consumers/position_update_consumer.py (validate required fields: asset, unrealized_pnl, position_size, timestamp, log validation failures with full context, skip invalid events gracefully)

**Checkpoint**: At this point, User Story 5 should be fully functional and testable independently. The system can react to position updates in real-time and generate exit signals based on configurable rules.

---

### Suggested MVP Scope

- **MVP**: Phase 1 (Setup) + Phase 2 (Foundational) + Phase 3 (User Story 1 - Warm-up Mode Signal Generation)
- This delivers immediate trading capability without requiring historical data or pre-trained models
- Enables data collection for future model training
- Total MVP tasks: 32 tasks (9 + 12 + 11)

---

## Phase 9: Training Orchestrator Improvements (Priority: P2)

**Goal**: Improve training orchestrator reliability and efficiency by implementing persistent buffer storage and training queue management to prevent data loss on service restart and avoid wasteful training cancellations.

**Context**: Current implementation has two critical issues:
1. **In-memory buffer**: Events stored in memory are lost on service restart, requiring re-accumulation of events up to `MODEL_TRAINING_MIN_DATASET_SIZE`
2. **Training cancellation**: New events arriving during training cause current training to be cancelled and restarted, wasting computational resources and potentially preventing training from ever completing

**Independent Test**: Can be fully tested by simulating service restarts and verifying buffer recovery, and by generating events during training to verify queue behavior without cancellation.

### Implementation for Training Orchestrator Improvements

#### Part 1: Persistent Buffer Storage

- [X] T123 [P] [US2] Create database migration script in ws-gateway/migrations/011_add_used_for_training_to_execution_events.sql (add columns: used_for_training BOOLEAN DEFAULT FALSE, training_id UUID NULL, add index on used_for_training and strategy_id for efficient queries, add foreign key constraint on training_id referencing model_versions.id)
- [X] T124 [US2] Extend ExecutionEventRepository in model-service/src/database/repositories/execution_event_repo.py (add methods: mark_as_used_for_training(event_ids, training_id), get_unused_events(strategy_id, limit), get_unused_events_count(strategy_id), support filtering by strategy_id and date range)
- [X] T125 [US2] Implement buffer persistence service in model-service/src/services/buffer_persistence.py (save buffer state to database by marking events as used_for_training=FALSE, restore buffer from database on startup by loading unused events, handle buffer recovery errors gracefully with logging, support incremental buffer updates)
- [X] T126 [US2] Integrate buffer persistence into TrainingOrchestrator in model-service/src/services/training_orchestrator.py (on add_execution_event: mark event as unused in DB if not already persisted, on training start: mark events as used_for_training=TRUE with training_id, on service startup: restore buffer from database by loading unused events, handle persistence errors gracefully without blocking training)
- [X] T127 [US2] Add buffer recovery on startup in model-service/src/main.py (call training_orchestrator.restore_buffer_from_database() after service initialization, log buffer recovery results with event count, handle recovery failures gracefully with fallback to empty buffer)
- [X] T128 [US2] Add configuration for buffer persistence in model-service/src/config/settings.py (BUFFER_PERSISTENCE_ENABLED=true/false to enable/disable persistent buffer, BUFFER_RECOVERY_ON_STARTUP=true/false to enable/disable automatic buffer recovery, BUFFER_MAX_RECOVERY_EVENTS=10000 for maximum events to recover on startup, default: persistence enabled for reliability)

#### Part 2: Training Queue Management

- [X] T129 [US2] Implement training queue data structure in model-service/src/services/training_orchestrator.py (add _training_queue: List[Tuple[str, List[OrderExecutionEvent], Optional[datetime]]] for (strategy_id, events, priority_timestamp), support FIFO queue with optional priority-based ordering for critical training)
- [X] T130 [US2] Modify check_and_trigger_training in model-service/src/services/training_orchestrator.py (if training in progress: add events to queue instead of cancelling, if queue not empty after training completes: automatically start next training, log queue operations with queue size and strategy_id)
- [X] T131 [US2] Implement training completion handler in model-service/src/services/training_orchestrator.py (on training completion: check if queue has pending training, if queue not empty: pop next training from queue and start, log queue processing with remaining queue size)
- [X] T132 [US2] Add training queue metrics in model-service/src/services/training_orchestrator.py (track queue size, average wait time, maximum queue depth, expose via training status API endpoint)
- [X] T133 [US2] Extend training status API in model-service/src/api/training.py (add queue_size, queue_wait_time_seconds, next_queued_training_strategy_id fields to training status response)
- [X] T134 [US2] Add configuration for training queue in model-service/src/config/settings.py (TRAINING_QUEUE_ENABLED=true/false to enable/disable queue (default: enabled), TRAINING_QUEUE_MAX_SIZE=10 for maximum queue depth, TRAINING_FORCE_CANCEL_ON_CRITICAL=false to allow critical training to cancel current training, default: queue enabled, no forced cancellation)

#### Part 3: Additional Improvements

- [X] T135 [US2] Implement graceful shutdown for training orchestrator in model-service/src/services/training_orchestrator.py (on shutdown signal: save current buffer state to database, wait for current training to complete (with timeout), save queue state to database, log shutdown operations with buffer size and queue size)
- [X] T136 [US2] Implement parallel training support in model-service/src/services/training_orchestrator.py (support multiple concurrent training tasks for different strategy_id values, track training tasks per strategy_id in _training_tasks: Dict[str, asyncio.Task], prevent duplicate training for same strategy_id, add configuration MAX_PARALLEL_TRAINING=3 for maximum concurrent training tasks)
- [X] T137 [US2] Implement training prioritization in model-service/src/services/training_orchestrator.py (add priority field to training queue items, support priority levels: CRITICAL (quality degradation), HIGH (scheduled retraining), NORMAL (data accumulation), CRITICAL priority can cancel current training if TRAINING_FORCE_CANCEL_ON_CRITICAL=true, log priority-based decisions)
- [X] T138 [US2] Implement batch event processing in model-service/src/services/training_orchestrator.py (group events by time window before adding to buffer, reduce database write frequency with batch updates, add configuration BATCH_BUFFER_UPDATE_INTERVAL_SECONDS=10 for batch update interval, handle batch processing errors gracefully)
- [X] T139 [US2] Add monitoring metrics for training orchestrator in model-service/src/services/training_orchestrator.py (track metrics: buffer_size, queue_size, training_duration, training_success_rate, cancelled_trainings_count, buffer_recovery_count, expose via monitoring API endpoint)
- [X] T140 [US2] Add structured logging for training orchestrator improvements in model-service/src/services/training_orchestrator.py (log buffer persistence operations, queue operations, training completion and queue processing, buffer recovery on startup, graceful shutdown operations, include trace_id for request flow tracking)

**Checkpoint**: At this point, training orchestrator should be resilient to service restarts and efficiently handle concurrent training requests without wasteful cancellations.

### Task Dependencies for Training Orchestrator Improvements Phase (T123-T140)

- **T123** (database migration): Can be done independently, must complete before T124 (repository needs columns)
- **T124** (repository extension): Depends on T123 (columns exist), must complete before T125 (buffer persistence needs repository)
- **T125** (buffer persistence service): Depends on T124 (repository methods exist), must complete before T126 (orchestrator needs persistence service)
- **T126** (orchestrator integration): Depends on T125 (buffer persistence exists) and T044 (training orchestrator exists), must complete before T127 (startup integration)
- **T127** (startup integration): Depends on T126 (orchestrator integration complete)
- **T128** (configuration): Can be done independently, must complete before T125, T126, T127 (services need configuration values)
- **T129** (queue data structure): Can be done independently, must complete before T130 (queue management needs structure)
- **T130** (queue management): Depends on T129 (queue structure exists) and T044 (training orchestrator exists), must complete before T131 (completion handler needs queue)
- **T131** (completion handler): Depends on T130 (queue management exists)
- **T132** (queue metrics): Depends on T130 (queue management exists), can be done in parallel with T133
- **T133** (API extension): Depends on T132 (queue metrics exist) and T061 (training status API exists)
- **T134** (queue configuration): Can be done independently, must complete before T130, T131 (queue management needs configuration)
- **T135** (graceful shutdown): Depends on T126 (buffer persistence integration) and T130 (queue management), can be done in parallel with T136-T140
- **T136** (parallel training): Depends on T044 (training orchestrator exists), can be done in parallel with T137-T140
- **T137** (prioritization): Depends on T129 (queue structure exists) and T130 (queue management), can be done in parallel with T138-T140
- **T138** (batch processing): Depends on T126 (buffer persistence integration), can be done in parallel with T139, T140
- **T139** (monitoring metrics): Depends on T126 (buffer persistence) and T130 (queue management), can be done in parallel with T140
- **T140** (logging): Can be done independently, but should complete after T126, T130, T135 (to log all improvements)

### Parallel Opportunities

- **T123, T128, T129, T134**: Can run in parallel (independent setup tasks)
- **T132, T133**: Can run in parallel (metrics and API)
- **T135, T136, T137, T138, T139, T140**: Can run in parallel (additional improvements)

---

## Phase 10: Feature Service Integration (Priority: P1)

**Goal**: Integrate Model Service with Feature Service to receive ready feature vectors instead of computing features internally. Remove feature engineering code from Model Service and use Feature Service for both inference and training datasets. **Training pipeline uses only market data from Feature Service (not execution_events)** - model learns from market movements (price predictions), not from own trading results.

**Independent Test**: Can be fully tested by verifying that Model Service receives features from Feature Service (via queue or REST API), performs inference on ready features, and trains models on datasets provided by Feature Service. All feature engineering code should be removed from Model Service. Training is triggered by scheduled/time-based retraining, not by execution_events accumulation.

### Implementation for Feature Service Integration

**Architecture Change**: Training pipeline now uses only market data from Feature Service, not execution_events. Model learns from market movements (price predictions based on future returns), not from own trading results. This simplifies architecture by removing execution_events buffer, buffer persistence, dataset builder, label generator, and execution_events consumer from training pipeline. Training is triggered by scheduled/time-based retraining instead of execution_events accumulation.

- [X] T145 [P] [US3] Create Feature Service HTTP client in model-service/src/services/feature_service_client.py (httpx client for REST API calls to Feature Service, methods: get_latest_features(symbol) -> FeatureVector, build_dataset(request) -> dataset_id, get_dataset(dataset_id) -> Dataset, download_dataset(dataset_id, split) -> file_path, handle API errors gracefully with retry logic, support API key authentication via X-API-Key header, use Dataset and DatasetBuildRequest models)
- [X] T146 [P] [US3] Create Feature Vector data model in model-service/src/models/feature_vector.py (timestamp, symbol, features: Dict[str, float], feature_registry_version, trace_id, matches Feature Service FeatureVector structure)
- [X] T147 [US3] Create feature consumer in model-service/src/consumers/feature_consumer.py (subscribe to RabbitMQ queue features.live, parse FeatureVector messages, cache latest features per symbol with TTL, handle message parsing errors gracefully, log feature reception with trace_id)
- [X] T148 [US3] Update ModelInference service in model-service/src/services/model_inference.py (remove feature calculation logic from prepare_features method, replace with feature alignment: receive FeatureVector from Feature Service, extract features dict, align feature names with model's expected features from feature_names_in_ or get_booster().feature_names, handle missing features with default values or error, handle extra features by ignoring them, preserve feature order matching model training)
- [X] T149 [US3] Update IntelligentSignalGenerator in model-service/src/services/intelligent_signal_generator.py (replace market data subscription with Feature Service integration: request features via feature_service_client.get_latest_features(symbol) or get from feature_consumer cache, pass ready FeatureVector to ModelInference, remove market data snapshot collection logic, keep business rules and risk management unchanged)
- [X] T150 [US3] Integrate feature consumer into main application in model-service/src/main.py (start feature consumer on service startup, initialize feature service client, connect feature consumer to intelligent signal generator)
- [X] T151 [US2] Create dataset request consumer in model-service/src/consumers/dataset_ready_consumer.py (subscribe to RabbitMQ queue features.dataset.ready, parse dataset completion notifications, trigger model training when dataset is ready, handle message parsing errors gracefully)
- [X] T152 [US2] Update TrainingOrchestrator in model-service/src/services/training_orchestrator.py (refactor to use Feature Service for dataset building: remove execution_events_buffer and dataset building logic, implement scheduled/time-based retraining triggers instead of event accumulation, request dataset build via feature_service_client.build_dataset() with explicit train/val/test periods based on time ranges, wait for dataset.ready notification from dataset_ready_consumer, download dataset via feature_service_client.download_dataset(dataset_id, split), load dataset from downloaded Parquet file, use ready features and targets from Feature Service dataset, remove all execution_events dependency from training pipeline)
- [X] T153 [US2] Remove FeatureEngineer class in model-service/src/services/feature_engineer.py (delete file, all feature engineering moved to Feature Service)
- [X] T154 [US1] Remove market data subscription service in model-service/src/services/market_data_subscriber.py (delete file, market data now comes from Feature Service via features)
- [X] T155 [US1] Remove market data consumer in model-service/src/consumers/market_data_consumer.py (delete file or remove ws-gateway subscription logic, market data now comes from Feature Service via features)
- [X] T156 [P] [US3] Add configuration for Feature Service integration in model-service/src/config/settings.py (FEATURE_SERVICE_HOST, FEATURE_SERVICE_PORT, FEATURE_SERVICE_API_KEY, FEATURE_SERVICE_USE_QUEUE=true/false to enable/disable queue subscription vs REST API polling, FEATURE_SERVICE_FEATURE_CACHE_TTL_SECONDS=30 for feature cache TTL, default: use queue for real-time features)
- [X] T157 [US3] Add feature alignment validation in model-service/src/services/model_inference.py (validate that all required features from model are present in FeatureVector, log warnings for missing features, log errors if critical features are missing and fail inference, support feature name mapping if Feature Service uses different names)
- [X] T158 [US2] Update retraining trigger logic in model-service/src/services/retraining_trigger.py (remove execution_events accumulation threshold check, implement time-based retraining: check if N days passed since last training, add scheduled retraining support with configurable intervals, remove dependency on execution_events buffer size, use only scheduled/time-based triggers for model retraining)
- [X] T159 [P] [US2] Add configuration for dataset building in model-service/src/config/settings.py (FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS=3600 for maximum wait time, FEATURE_SERVICE_DATASET_POLL_INTERVAL_SECONDS=60 for status polling interval, FEATURE_SERVICE_DATASET_STORAGE_PATH for downloaded dataset files)
- [X] T160 [US3] Add fallback to REST API if queue unavailable in model-service/src/services/intelligent_signal_generator.py (if feature_consumer cache is empty or stale, fallback to feature_service_client.get_latest_features(symbol), log fallback usage, handle API errors gracefully)
- [X] T161 [US3] Add structured logging for Feature Service integration in model-service/src/services/ (log feature requests, feature reception, feature alignment operations, dataset build requests, dataset downloads, include trace_id for request flow tracking)
- [X] T164 [US2] Remove execution event consumer from training pipeline in model-service/src/consumers/execution_event_consumer.py (disable or remove consumer for training purposes, keep only if needed for quality monitoring/other purposes, update main.py to stop starting execution event consumer for training)
- [X] T165 [US2] Remove buffer persistence service from training pipeline in model-service/src/services/buffer_persistence.py (delete file or disable for training, remove execution_events buffer persistence logic, remove restore_buffer_from_database() calls from TrainingOrchestrator)
- [X] T166 [US2] Remove dataset builder service in model-service/src/services/dataset_builder.py (delete file, dataset building now handled by Feature Service, all feature engineering and label generation moved to Feature Service)
- [X] T167 [US2] Remove label generator service in model-service/src/services/label_generator.py (delete file, label/target generation now handled by Feature Service)
- [X] T168 [US2] Remove execution events buffer from TrainingOrchestrator in model-service/src/services/training_orchestrator.py (remove _execution_events_buffer, remove _signal_market_data, remove add_execution_event() method, remove buffer persistence integration, remove execution_events dependency from training logic)
- [X] T169 [US2] Update quality evaluator in model-service/src/services/quality_evaluator.py (remove calculate_trading_metrics() method that uses execution_events, or make it optional for backtesting only, use only ML metrics: accuracy, precision, recall, F1 for classification, MSE, MAE, R2 for regression)
- [X] T170 [US2] Update quality monitor in model-service/src/services/quality_monitor.py (remove periodic evaluation based on execution_events, use only ML metrics from validation/test sets, optionally get trading metrics from other services for monitoring but not for training triggers)
- [X] T171 [US2] Add time-based retraining configuration in model-service/src/config/settings.py (MODEL_RETRAINING_INTERVAL_DAYS=7 for time-based retraining interval, MODEL_RETRAINING_TRAIN_PERIOD_DAYS=30 for training period length, MODEL_RETRAINING_VALIDATION_PERIOD_DAYS=7 for validation period length, MODEL_RETRAINING_TEST_PERIOD_DAYS=1 for test period length, document new retraining strategy)
- [X] T172 [US2] Add dataset period calculation logic in model-service/src/services/training_orchestrator.py (implement method to calculate train/val/test periods based on current time and configuration, calculate periods relative to current time for rolling window approach, support configurable period lengths from settings)
- [X] T162 [P] Update documentation in model-service/README.md (document Feature Service integration, remove references to feature engineering and execution_events for training, document new time-based/scheduled retraining approach, update API examples to show Feature Service usage, document configuration options)
- [X] T163 [P] Update quickstart.md in specs/001-model-service/ (synchronize with Feature Service integration, update examples to show Feature Service workflow, document training workflow without execution_events)

**Checkpoint**: At this point, Model Service should be fully integrated with Feature Service. All feature engineering code is removed, and Model Service receives ready features for both inference and training. Training pipeline uses only market data from Feature Service, not execution_events. Model training is triggered by scheduled/time-based retraining, not by execution_events accumulation.

**Summary of Changes for Market-Data-Only Training**:
- Training no longer depends on execution_events buffer accumulation
- Removed: execution_events buffer, buffer persistence, dataset builder, label generator, execution_events consumer (for training)
- Added: time-based/scheduled retraining triggers, period calculation for dataset requests
- Model learns from market movements (price predictions), not from own trading results
- Quality evaluation uses only ML metrics (accuracy, precision, recall, F1, MSE, MAE, R2)
- Simpler architecture with fewer dependencies between services

---

## Phase 11: Test Split Evaluation (Priority: P2)

**Goal**: Implement proper test set evaluation for final quality assessment and model activation, aligning with docs/feature-service.md workflow which specifies that Model Service should evaluate final quality on test set (out-of-sample) and activate model only if test set metrics exceed threshold.

**Context**: Current implementation uses only validation split for quality evaluation and model activation, which is a discrepancy with documentation. Test set should be used for final out-of-sample evaluation to prevent overfitting and ensure model generalization.

**Independent Test**: Can be fully tested by verifying that test split is downloaded, model is evaluated on test set, test set metrics are saved separately with dataset_split metadata, model activation uses test set metrics (not validation), and fallback to validation metrics works if test split is unavailable.

### Implementation for Test Split Evaluation

- [ ] T173 [US2] Implement test split evaluation in model-service/src/services/training_orchestrator.py (add download of test split via feature_service_client.download_dataset(dataset_id, split="test") after validation evaluation, load test dataset from Parquet file, extract features and labels from test split using same target_column logic as train/validation, evaluate model quality on test set using quality_evaluator.evaluate(), save test set metrics separately with metadata={"dataset_split": "test"} to distinguish from validation metrics in database, use test set metrics (not validation) for final quality assessment and model activation decision, implement fallback to validation set metrics if test split is unavailable (with warning log), add structured logging for test split download, loading, evaluation, and activation decision, align with docs/feature-service.md workflow which specifies that Model Service should evaluate final quality on test set (out-of-sample) and activate model only if test set metrics exceed threshold, currently only validation split is used for evaluation which is a discrepancy with documentation)
- [ ] T174 [US2] Update save_quality_metrics to support dataset split metadata in model-service/src/services/model_version_manager.py (extend save_quality_metrics method to accept optional dataset_split parameter, include dataset_split in metadata when saving metrics: metadata={"dataset_split": dataset_split} if dataset_split provided, ensure backward compatibility: if dataset_split not provided, metadata remains unchanged, update method signature and documentation to reflect new parameter)
- [ ] T175 [US2] Update quality metrics repository to support filtering by dataset split in model-service/src/database/repositories/quality_metrics_repo.py (add optional dataset_split parameter to get_by_model_version and get_latest_by_model_version methods, filter metrics by metadata->>'dataset_split' = dataset_split when parameter provided, support querying validation vs test metrics separately, ensure backward compatibility: if dataset_split not provided, return all metrics as before)
- [ ] T176 [US4] Update SC-005 success criteria in specs/001-model-service/spec.md (change "validation set" to "test set" in SC-005: "System transitions from warm-up mode to model-based generation when model quality metrics exceed 75% classification accuracy threshold (percentage of correct buy/sell predictions on test set)", align with docs/feature-service.md workflow and T173 implementation which uses test set for final quality assessment)
- [ ] T177 [US2] Update mode transition service to use test set metrics in model-service/src/services/mode_transition.py (modify _get_model_quality method to prefer test set metrics over validation metrics when available, query quality_metrics_repo.get_latest_by_model_version with dataset_split="test" first, fallback to validation metrics if test set metrics not available, log which dataset split metrics are used for mode transition decision, ensure backward compatibility with existing models that may only have validation metrics)
- [ ] T178 [US2] Update retraining trigger to use test set metrics for quality degradation detection in model-service/src/services/retraining_trigger.py (modify _check_quality_degradation method to prefer test set metrics over validation metrics when checking quality thresholds, query quality_metrics_repo.get_latest_by_model_version with dataset_split="test" first, fallback to validation metrics if test set metrics not available, log which dataset split metrics are used for degradation detection, ensure backward compatibility with existing models)
- [ ] T179 [P] Update documentation in model-service/README.md to document test set evaluation workflow (add section explaining train/validation/test split usage, document that test set is used for final quality assessment and model activation, explain fallback to validation set if test split unavailable, update training workflow documentation to match docs/feature-service.md, add examples showing test set metrics in API responses)
- [ ] T180 [P] Update quickstart.md in specs/001-model-service/ to reflect test set evaluation (synchronize training workflow with test set evaluation, update examples to show test set metrics, document that model activation uses test set metrics, align with T173 implementation and docs/feature-service.md workflow)
- [ ] T181 [US2] Add warning logging for empty or unavailable test split in model-service/src/services/training_orchestrator.py (when test split download fails or test split is empty, log structured warning with dataset_id, symbol, reason (download_failed, empty_split, file_not_found), record warning in metrics for monitoring, document fallback behavior to validation metrics in logs, ensure warning is visible in monitoring dashboards for operational awareness)

**Checkpoint**: At this point, Model Service should properly evaluate models on test set for final quality assessment, save test set metrics separately from validation metrics, use test set metrics for model activation decisions, and have all related services (mode transition, retraining trigger) updated to use test set metrics when available. Documentation and specifications should be aligned with implementation.

**Summary of Changes for Test Split Evaluation**:
- Test set is now used for final quality assessment and model activation (not validation set)
- Test set metrics are saved separately with dataset_split="test" metadata
- Validation metrics are saved with dataset_split="validation" metadata
- Mode transition and retraining trigger prefer test set metrics over validation metrics
- Fallback to validation metrics if test split is unavailable (with warning)
- SC-005 specification updated to reference test set instead of validation set
- Documentation updated to reflect test set evaluation workflow
