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

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

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

- **Total Tasks**: 129 tasks (added T052c, T052d, T052e for risk management rules, T094-T097 for position cache optimization, T098-T116 for position-based exit strategy)
- **Phase 1 (Setup)**: 9 tasks
- **Phase 2 (Foundational)**: 12 tasks
- **Phase 3 (User Story 1 - MVP)**: 15 tasks (added T030a, T030b, T030c, T030d)
- **Phase 4 (User Story 2)**: 19 tasks (added T037a, T039a)
- **Phase 5 (User Story 3)**: 14 tasks (added T052a, T052b, T052c, T052d, T052e for risk management)
- **Phase 6 (User Story 4)**: 17 tasks
- **Phase 7 (Polish)**: 30 tasks (added T091, T092, T093, T094-T097 for position cache optimization, T117-T122 for automatic asset selection)
- **Phase 8 (User Story 5)**: 19 tasks (position-based exit strategy)

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
- [X] T110 [US5] Add exit strategy configuration to model-service/src/config/settings.py (EXIT_STRATEGY_ENABLED, EXIT_STRATEGY_RATE_LIMIT, TAKE_PROFIT_ENABLED, TAKE_PROFIT_THRESHOLD_PCT, TAKE_PROFIT_PARTIAL_EXIT, TAKE_PROFIT_PARTIAL_AMOUNT_PCT, STOP_LOSS_ENABLED, STOP_LOSS_THRESHOLD_PCT, TRAILING_STOP_ENABLED, TRAILING_STOP_ACTIVATION_PCT, TRAILING_STOP_DISTANCE_PCT, TIME_BASED_EXIT_ENABLED, TIME_BASED_EXIT_MAX_HOURS, TIME_BASED_EXIT_PROFIT_TARGET_PCT)
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
