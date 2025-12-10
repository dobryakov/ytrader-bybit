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

- [X] T001 Create project structure per implementation plan in feature-service/
- [X] T002 Initialize Python project with dependencies in feature-service/requirements.txt (include pytest, pytest-asyncio, pytest-mock)
- [X] T003 [P] Create Dockerfile in feature-service/Dockerfile
- [X] T004 [P] Create docker-compose.yml service configuration for feature-service
- [X] T005 [P] Create env.example with all required environment variables in feature-service/env.example
- [X] T006 [P] Configure linting and formatting tools (black, ruff) in feature-service/
- [X] T007 Create README.md in feature-service/README.md
- [X] T008 [P] Create test directory structure (tests/unit/, tests/integration/, tests/contract/, tests/fixtures/) in feature-service/
- [X] T009 [P] Create pytest configuration file (pytest.ini or pyproject.toml) in feature-service/
- [X] T010 [P] Create conftest.py with shared test fixtures in feature-service/tests/conftest.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T011 [P] Create test fixtures for database connection mocking in feature-service/tests/fixtures/database.py
- [X] T012 [P] Create test fixtures for RabbitMQ connection mocking in feature-service/tests/fixtures/rabbitmq.py
- [X] T013 [P] Create test fixtures for HTTP client mocking (ws-gateway API) in feature-service/tests/fixtures/http_client.py
- [X] T014 [P] Create unit tests for configuration management in feature-service/tests/unit/test_config.py
- [X] T015 [P] Create unit tests for logging setup in feature-service/tests/unit/test_logging.py
- [X] T016 [P] Create integration tests for database connection pool in feature-service/tests/integration/test_metadata_storage.py
- [X] T017 [P] Create integration tests for RabbitMQ connection manager in feature-service/tests/integration/test_mq_connection.py
- [X] T018 [P] Create unit tests for HTTP client setup in feature-service/tests/unit/test_http_client.py
- [X] T019 [P] Create unit tests for API authentication middleware in feature-service/tests/unit/test_auth_middleware.py
- [X] T020 [P] Create contract tests for health check endpoint in feature-service/tests/contract/test_health.py
- [X] T021 [P] Create test fixtures for market data events in feature-service/tests/fixtures/market_data.py
- [X] T022 [P] Create unit tests for Feature Registry configuration loader in feature-service/tests/unit/test_feature_registry_loader.py

### Implementation for Foundational Phase

- [X] T023 Create database migration for datasets table in ws-gateway/migrations/018_create_datasets_table.sql
- [X] T023a Apply database migration for datasets table: Applied via `docker compose exec postgres psql -U ytrader -d ytrader -f /tmp/018_create_datasets_table.sql`
- [X] T024 Create database migration for feature_registry_versions table in ws-gateway/migrations/019_create_feature_registry_versions_table.sql (includes all fields: version VARCHAR(50) PRIMARY KEY, config JSONB NOT NULL, is_active BOOLEAN NOT NULL DEFAULT false, validated_at TIMESTAMP, validation_errors TEXT[], loaded_at TIMESTAMP, created_at TIMESTAMP NOT NULL DEFAULT NOW(), created_by VARCHAR(100), activated_by VARCHAR(100), rollback_from VARCHAR(50), previous_version VARCHAR(50), schema_version VARCHAR(50), migration_script TEXT, compatibility_warnings TEXT[], breaking_changes TEXT[], activation_reason TEXT, indexes on is_active, created_at DESC, previous_version for version management and rollback queries). This table supports full version management capabilities from the start.
- [X] T024a Apply database migration for feature_registry_versions table: Applied via `docker compose exec postgres psql -U ytrader -d ytrader -f /tmp/019_create_feature_registry_versions_table.sql`
- [X] T025 Create database migration for data_quality_reports table in ws-gateway/migrations/020_create_data_quality_reports_table.sql
- [X] T025a Apply database migration for data_quality_reports table: Applied via `docker compose exec postgres psql -U ytrader -d ytrader -f /tmp/020_create_data_quality_reports_table.sql`
- [X] T026 [P] Create base configuration management in feature-service/src/config.py using pydantic-settings
- [X] T027 [P] Create base logging setup with structlog in feature-service/src/logging.py
- [X] T028 [P] Create database connection pool in feature-service/src/storage/metadata_storage.py using asyncpg
- [X] T029 [P] Create RabbitMQ connection manager in feature-service/src/mq/connection.py using aio-pika
- [X] T030 [P] Create HTTP client setup for ws-gateway REST API integration in feature-service/src/http/client.py using httpx
- [X] T031 [P] Create base FastAPI application structure in feature-service/src/main.py
- [X] T032 [P] Create API authentication middleware in feature-service/src/api/middleware/auth.py
- [X] T033 [P] Create health check endpoint in feature-service/src/api/health.py
- [X] T034 Create base models for market data events in feature-service/src/models/market_data.py
- [X] T035 Create base Feature Registry configuration loader in feature-service/src/services/feature_registry.py (basic structure)
- [X] T036 Create default Feature Registry YAML configuration in feature-service/config/feature_registry.yaml

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Получение признаков в реальном времени для торговых решений (Priority: P1) 🎯 MVP

**Goal**: Model Service получает готовые признаки в реальном времени для принятия торговых решений. Система вычисляет признаки из потока маркет-данных и предоставляет их с минимальной задержкой через API или очередь событий.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что признаки вычисляются и доступны через API в течение заданного времени задержки.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T037 [P] [US1] Create test fixtures for feature vectors in feature-service/tests/fixtures/feature_vectors.py
- [X] T038 [P] [US1] Create test fixtures for orderbook state (snapshots and deltas) in feature-service/tests/fixtures/orderbook.py
- [X] T039 [P] [US1] Create test fixtures for rolling windows data in feature-service/tests/fixtures/rolling_windows.py
- [X] T040 [P] [US1] Create test fixtures for market data streams (orderbook, trades, klines, ticker, funding) in feature-service/tests/fixtures/market_data_streams.py
- [X] T041 [P] [US1] Create unit tests for Feature Vector model in feature-service/tests/unit/test_feature_vector.py
- [X] T042 [P] [US1] Create unit tests for Orderbook State model in feature-service/tests/unit/test_orderbook_state.py
- [X] T043 [P] [US1] Create unit tests for Rolling Windows model in feature-service/tests/unit/test_rolling_windows.py
- [X] T044 [P] [US1] Create unit tests for Orderbook Manager service in feature-service/tests/unit/test_orderbook_manager.py (snapshot + delta reconstruction, desynchronization)
- [X] T045 [P] [US1] Create unit tests for price features computation in feature-service/tests/unit/test_price_features.py
- [ ] T288 [P] [US1] Add unit tests for volatility_n_candles feature in feature-service/tests/unit/test_price_features.py (test volatility computation with exactly 20 candles, test with more than 20 candles (should use last 20), test with less than 20 candles (should return None), test with insufficient data for std calculation (should return None), test with different candle intervals, test edge cases: empty klines, zero prices, verify volatility calculation matches expected std of returns)
- [ ] T289 [P] [US1] Add unit tests for returns_n_candles feature in feature-service/tests/unit/test_price_features.py (test returns computation with exactly 3 candles, test with more than 3 candles (should use first and last of last 3), test with less than 3 candles (should return None), test with zero close price (should return None), test with different candle intervals, test edge cases: empty klines, verify returns calculation matches expected (close_last - close_first) / close_first)
- [ ] T290 [P] [US1] Add unit tests for volume_zscore feature in feature-service/tests/unit/test_price_features.py (test z-score computation with sufficient historical data, test with zero std (should return None), test with insufficient historical data (should return None), test with different historical window sizes (50, 100, 200 candles), test edge cases: all volumes equal (std=0), verify z-score calculation matches expected (current_volume - mean) / std, test with normal distribution of volumes, test with outlier volumes)
- [ ] T291 [P] [US1] Add unit tests for technical indicators (RSI, EMA, MACD, Bollinger Bands) in feature-service/tests/unit/test_technical_indicators.py (test RSI(14) computation with sufficient price history, test RSI with insufficient data (should return None), test RSI edge cases: all prices equal, price going up/down only, verify RSI values in valid range [0, 100], test EMA(20) computation with exponential smoothing, test EMA with insufficient data (should return None), verify EMA calculation matches expected exponential moving average formula, test MACD computation (EMA(12) - EMA(26)), test MACD signal line (EMA(9) of MACD), test MACD histogram, verify MACD calculation matches expected values, test Bollinger Bands computation (upper, lower, width), test with different window sizes (20, 50), verify bands calculation: upper = SMA + 2*std, lower = SMA - 2*std, test edge cases: insufficient data, all prices equal)
- [ ] T292 [P] [US1] Add unit tests for orderbook slope feature in feature-service/tests/unit/test_orderbook_features.py (test slope_bid computation using linear regression on price levels vs volumes, test slope_ask computation, test with insufficient orderbook levels (should return None), test with single level (should return None), verify slope calculation: negative slope indicates normal orderbook (volumes decrease with distance from best price), test edge cases: empty orderbook, orderbook with only best bid/ask, test with different number of levels (5, 10, 20))
- [ ] T293 [P] [US1] Add unit tests for orderbook churn rate feature in feature-service/tests/unit/test_orderbook_features.py (test churn rate computation by counting best price changes over time window, test with insufficient historical data (should return None), test with no price changes (should return 0.0), verify churn rate calculation: number_of_changes / time_window_seconds, test with different time windows (1s, 3s, 15s, 1m), test edge cases: empty orderbook history, single snapshot)
- [ ] T294 [P] [US1] Add unit tests for Rate of Change (ROC) feature in feature-service/tests/unit/test_price_features.py (test ROC computation for different periods (5m, 15m, 1h), test ROC with insufficient price history (should return None), test ROC calculation: (current_price - price_N_periods_ago) / price_N_periods_ago, verify ROC matches expected rate of change formula, test edge cases: zero historical price, insufficient data, test with different candle intervals)
- [ ] T295 [P] [US1] Add unit tests for relative volume feature in feature-service/tests/unit/test_price_features.py (test relative volume computation: current_volume / average_volume_24h, test with insufficient historical data for average (should return None), test with zero average volume (should return None), verify relative volume calculation matches expected formula, test with different comparison windows (1h, 24h, 7d), test edge cases: all volumes zero, insufficient historical data, verify relative volume > 1.0 indicates above-average volume, relative volume < 1.0 indicates below-average volume)
- [X] T046 [P] [US1] Create unit tests for orderflow features computation in feature-service/tests/unit/test_orderflow_features.py
- [X] T047 [P] [US1] Create unit tests for orderbook features computation in feature-service/tests/unit/test_orderbook_features.py
- [X] T048 [P] [US1] Create unit tests for perpetual features computation in feature-service/tests/unit/test_perpetual_features.py
- [X] T049 [P] [US1] Create unit tests for temporal features computation in feature-service/tests/unit/test_temporal_features.py
- [X] T050 [US1] Create unit tests for Feature Computer service in feature-service/tests/unit/test_feature_computer.py
- [X] T051 [US1] Create integration tests for market data consumer in feature-service/tests/integration/test_market_data_consumer.py (with mocked RabbitMQ)
- [ ] T052 [US1] Create integration tests for subscription management in feature-service/tests/integration/test_subscription_management.py (with mocked ws-gateway API)
- [X] T053 [US1] Create integration tests for feature publisher in feature-service/tests/integration/test_feature_publisher.py (with mocked RabbitMQ)
- [X] T054 [US1] Create contract tests for GET /features/latest endpoint in feature-service/tests/contract/test_features_api.py
- [ ] T055 [US1] Create integration tests for feature computation latency (≤70ms) in feature-service/tests/integration/test_feature_latency.py
- [ ] T056 [US1] Create integration tests for ws-gateway unavailability handling in feature-service/tests/integration/test_ws_gateway_resilience.py

### Implementation for User Story 1

- [X] T057 [P] [US1] Create Feature Vector model in feature-service/src/models/feature_vector.py
- [X] T058 [P] [US1] Create Orderbook State model in feature-service/src/models/orderbook_state.py
- [X] T059 [P] [US1] Create Rolling Windows model in feature-service/src/models/rolling_windows.py
- [X] T060 [US1] Implement Orderbook Manager service in feature-service/src/services/orderbook_manager.py (snapshot + delta reconstruction)
- [X] T061 [US1] Implement price features computation in feature-service/src/features/price_features.py (mid_price, spread, returns, VWAP, volatility)
- [X] T062 [US1] Implement orderflow features computation in feature-service/src/features/orderflow_features.py (signed_volume, buy/sell ratio, trade_count, net_aggressor_pressure)
- [X] T063 [US1] Implement orderbook features computation in feature-service/src/features/orderbook_features.py (depth, imbalance)
- [X] T064 [US1] Implement perpetual features computation in feature-service/src/features/perpetual_features.py (funding_rate, time_to_funding)
- [X] T065 [US1] Implement temporal features computation in feature-service/src/features/temporal_features.py (time_of_day with cyclic encoding)
- [X] T066 [US1] Implement Feature Computer service in feature-service/src/services/feature_computer.py (orchestrates all feature computations)
- [X] T067 [US1] Implement market data consumer in feature-service/src/consumers/market_data_consumer.py (consumes from ws-gateway.* queues)
- [X] T068 [US1] Implement subscription management for WebSocket channels via ws-gateway REST API in feature-service/src/consumers/market_data_consumer.py
- [X] T069 [US1] Implement subscription lifecycle management (create on startup, handle failures gracefully with retry, don't cancel on shutdown) in feature-service/src/consumers/market_data_consumer.py
- [ ] T070 [US1] Implement optional execution events consumer from ws-gateway.order or order-manager.order_events queues in feature-service/src/consumers/market_data_consumer.py
- [ ] T071 [US1] Implement optional subscription to order execution events via ws-gateway REST API in feature-service/src/consumers/market_data_consumer.py
- [X] T072 [US1] Implement feature publisher in feature-service/src/publishers/feature_publisher.py (publishes to features.live queue)
- [X] T073 [US1] Implement GET /features/latest endpoint in feature-service/src/api/features.py with 404 handling for missing symbols
- [X] T074 [US1] Add internal timestamp and exchange timestamp to all received messages in feature-service/src/consumers/market_data_consumer.py
- [X] T075 [US1] Implement feature computation scheduling at intervals (1s, 3s, 15s, 1m) in feature-service/src/services/feature_scheduler.py
- [X] T076 [US1] Add logging for feature computation operations in feature-service/src/services/feature_computer.py
- [X] T077 [US1] Add latency monitoring and metrics for feature computation in feature-service/src/services/feature_computer.py
- [X] T078 [US1] Implement handling of ws-gateway unavailability (continue with last available data, log issues, update quality metrics) in feature-service/src/consumers/market_data_consumer.py
- [X] T079 [US1] Implement latency warning when computation exceeds 70ms threshold in feature-service/src/services/feature_computer.py
- [ ] T285 [P] [US1] Add volatility feature for last N candles in feature-service/src/features/price_features.py (implement compute_volatility_n_candles(rolling_windows, n_candles=20, candle_interval="1m") function: get last N candles using get_klines_for_window(), sort by timestamp, take last N candles, compute returns between consecutive candles, return std(returns) as volatility, handle edge cases: return None if less than N candles available, return None if insufficient data for std calculation, add feature "volatility_20_candles" to compute_all_price_features(), add unit tests in feature-service/tests/unit/test_price_features.py for volatility_n_candles computation with various candle counts and edge cases)
- [ ] T286 [P] [US1] Add return feature for last N candles in feature-service/src/features/price_features.py (implement compute_returns_n_candles(rolling_windows, n_candles=3, candle_interval="1m") function: get last N candles using get_klines_for_window(), sort by timestamp, take first and last close prices, compute return as (close_last - close_first) / close_first, handle edge cases: return None if less than N candles available, return None if close_first is zero, add feature "returns_3_candles" to compute_all_price_features(), add unit tests in feature-service/tests/unit/test_price_features.py for returns_n_candles computation with various candle counts and edge cases)
- [ ] T287 [P] [US1] Add volume z-score feature in feature-service/src/features/price_features.py (implement compute_volume_zscore(rolling_windows, current_volume, historical_window_candles=100, candle_interval="1m") function: get historical volumes from last historical_window_candles candles using get_klines_for_window(), compute mean and std of historical volumes, compute z-score as (current_volume - mean) / std, handle edge cases: return None if std is zero, return None if insufficient historical data, add feature "volume_zscore" to compute_all_price_features() using current volume from last candle or volume_1m, add unit tests in feature-service/tests/unit/test_price_features.py for volume_zscore computation with various historical window sizes and edge cases including zero std, insufficient data, and normal distribution scenarios)
- [ ] T296 [P] [US1] Create technical indicators computation module in feature-service/src/features/technical_indicators.py (implement RSI(14) calculation: compute price changes, calculate average gain/loss over 14 periods, RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss, implement EMA(20) calculation: EMA = price * multiplier + EMA_prev * (1 - multiplier) where multiplier = 2 / (period + 1), implement EMA(12) and EMA(26) for MACD, implement MACD calculation: MACD_line = EMA(12) - EMA(26), MACD_signal = EMA(9) of MACD_line, MACD_histogram = MACD_line - MACD_signal, implement Bollinger Bands: SMA(20), upper_band = SMA + 2*std, lower_band = SMA - 2*std, bandwidth = (upper - lower) / SMA, handle edge cases: return None if insufficient price history, return None if all prices are equal (for RSI), use pandas_ta library or implement custom calculation, add function compute_all_technical_indicators(rolling_windows, candle_interval="1m") that computes all indicators from klines)
- [ ] T297 [P] [US1] Integrate technical indicators into price features in feature-service/src/features/price_features.py (add RSI_14, EMA_20, MACD_line, MACD_signal, MACD_histogram, BB_upper, BB_lower, BB_width to compute_all_price_features(), import compute_all_technical_indicators from technical_indicators module, compute technical indicators from klines using rolling_windows.get_klines_for_window(), handle None values gracefully, ensure indicators are computed from historical klines only (no lookahead))
- [ ] T298 [P] [US1] Add orderbook slope feature in feature-service/src/features/orderbook_features.py (implement compute_slope_bid(orderbook, top_n=10) function: extract top N bid price levels and corresponding volumes, perform linear regression: volume = a + b * price_distance_from_best, where price_distance = (best_bid - level_price) / best_bid, return slope coefficient b (negative indicates normal orderbook), implement compute_slope_ask(orderbook, top_n=10) similarly for ask side, handle edge cases: return None if less than 3 levels available, return None if orderbook is None, add features "slope_bid" and "slope_ask" to compute_all_orderbook_features(), use numpy.polyfit or scipy.stats.linregress for linear regression)
- [ ] T299 [P] [US1] Add orderbook churn rate feature in feature-service/src/features/orderbook_features.py (implement compute_orderbook_churn_rate(orderbook_manager, symbol, window_seconds=60) function: track best bid/ask price changes over time window, count number of times best bid or best ask changed, churn_rate = number_of_changes / window_seconds, requires maintaining history of best price changes in OrderbookManager or separate tracking, for online: track price changes in memory, for offline: reconstruct from orderbook snapshots/deltas, handle edge cases: return None if insufficient history, return 0.0 if no changes, add feature "orderbook_churn_rate" to compute_all_orderbook_features(), requires extending OrderbookManager to track best price change history)
- [ ] T300 [P] [US1] Add Rate of Change (ROC) feature in feature-service/src/features/price_features.py (implement compute_roc(rolling_windows, period_minutes=5, candle_interval="1m") function: get current price and price N minutes ago from klines, compute ROC = (current_price - price_N_ago) / price_N_ago, handle edge cases: return None if insufficient price history, return None if price_N_ago is zero, add features "roc_5m", "roc_15m", "roc_1h" to compute_all_price_features() with different periods, use klines from rolling_windows.get_klines_for_window() to get historical prices)
- [ ] T301 [P] [US1] Add relative volume feature in feature-service/src/features/price_features.py (implement compute_relative_volume(rolling_windows, current_volume, comparison_window_hours=24, candle_interval="1m") function: get historical volumes from last N hours using get_klines_for_window(), compute average volume over comparison window, relative_volume = current_volume / average_volume, handle edge cases: return None if insufficient historical data, return None if average_volume is zero, add features "relative_volume_1h", "relative_volume_24h" to compute_all_price_features() with different comparison windows, use volume from klines or trades depending on available data)
- [ ] T302 [P] [US1] Update Feature Registry YAML configuration in feature-service/config/feature_registry.yaml (add feature entries for: volatility_20_candles with input_sources: ["kline"], lookback_window: "20m", max_lookback_days: 1; returns_3_candles with input_sources: ["kline"], lookback_window: "3m", max_lookback_days: 1; volume_zscore with input_sources: ["kline", "trades"], lookback_window: "100m" (for 100 candles historical window), max_lookback_days: 1; RSI_14 with input_sources: ["kline"], lookback_window: "14m", max_lookback_days: 1; EMA_20 with input_sources: ["kline"], lookback_window: "20m", max_lookback_days: 1; MACD_line, MACD_signal, MACD_histogram with input_sources: ["kline"], lookback_window: "26m" (for EMA(26)), max_lookback_days: 1; BB_upper, BB_lower, BB_width with input_sources: ["kline"], lookback_window: "20m", max_lookback_days: 1; slope_bid, slope_ask with input_sources: ["orderbook"], lookback_window: "0s", max_lookback_days: 0; orderbook_churn_rate with input_sources: ["orderbook"], lookback_window: "60s" (for 1 minute window), max_lookback_days: 1 (for offline reconstruction); roc_5m, roc_15m, roc_1h with input_sources: ["kline"], lookback_window: "5m"/"15m"/"1h", max_lookback_days: 1; relative_volume_1h, relative_volume_24h with input_sources: ["kline", "trades"], lookback_window: "1h"/"24h", max_lookback_days: 1-2; increment version to "1.1.0" for Feature Registry update)
- [ ] T303 [US1] Extend OrderbookManager to track best price change history in feature-service/src/services/orderbook_manager.py (add _best_price_history: Dict[str, List[Tuple[datetime, float, float]]] to track (timestamp, best_bid, best_ask) changes per symbol, update history when orderbook state changes and best bid/ask prices change, maintain history for configurable time window (default 5 minutes), implement get_best_price_changes(symbol, window_seconds) method to retrieve price changes within window, handle history cleanup: remove entries older than retention window, add configuration BEST_PRICE_HISTORY_RETENTION_SECONDS=300 for history retention, ensure thread-safe access for concurrent updates)
- [ ] T304 [P] [US1] Add pandas_ta library dependency in feature-service/requirements.txt (add pandas_ta>=0.3.14b for technical indicators calculation: RSI, EMA, MACD, Bollinger Bands, or implement custom calculation functions if library is not available, document why pandas_ta is chosen: comprehensive technical indicators, well-tested, efficient implementation, alternative: ta-lib but requires C library installation which complicates Docker deployment)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Features are computed in real-time and available via API and message queue.

---

## Phase 4: User Story 2 - Сборка датасета для обучения моделей с корректным разделением (Priority: P1)

**Goal**: Model Service может запросить сборку датасета обучения из исторических данных с явным разделением на train/validation/test периоды. Система пересчитывает признаки на исторических данных идентично онлайн-режиму и генерирует целевые переменные без data leakage.

**Independent Test**: Можно протестировать независимо, запросив сборку датасета за определенный период и проверив, что датасет содержит корректно разделенные данные train/val/test, признаки идентичны онлайн-версии, и targets вычислены без использования будущей информации.

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T080 [P] [US2] Create test fixtures for dataset metadata in feature-service/tests/fixtures/datasets.py
- [X] T081 [P] [US2] Create test fixtures for historical market data (Parquet format) in feature-service/tests/fixtures/historical_data.py
- [X] T082 [P] [US2] Create test fixtures for target variables (regression, classification, risk-adjusted) in feature-service/tests/fixtures/targets.py
- [X] T083 [P] [US2] Create unit tests for Dataset model in feature-service/tests/unit/test_dataset.py
- [X] T084 [P] [US2] Create unit tests for Parquet storage service in feature-service/tests/unit/test_parquet_storage.py
- [X] T085 [P] [US2] Create unit tests for offline feature engine in feature-service/tests/unit/test_offline_engine.py
- [X] T086 [P] [US2] Create unit tests for orderbook state reconstruction in feature-service/tests/unit/test_orderbook_reconstruction.py
- [X] T087 [P] [US2] Create unit tests for rolling windows reconstruction in feature-service/tests/unit/test_rolling_windows_reconstruction.py
- [X] T088 [P] [US2] Create unit tests for target variable computation in feature-service/tests/unit/test_target_computation.py
- [X] T089 [P] [US2] Create unit tests for data leakage prevention validation in feature-service/tests/unit/test_data_leakage_prevention.py
- [X] T090 [P] [US2] Create unit tests for time-based dataset splitting in feature-service/tests/unit/test_dataset_splitting.py
- [X] T091 [P] [US2] Create unit tests for walk-forward validation strategy in feature-service/tests/unit/test_walk_forward.py
- [X] T092 [P] [US2] Create unit tests for random split strategy in feature-service/tests/unit/test_random_split.py
- [X] T093 [US2] Create integration tests for feature identity (online vs offline comparison) in feature-service/tests/integration/test_feature_identity.py
- [X] T094 [US2] Create integration tests for dataset building workflow in feature-service/tests/integration/test_dataset_building.py
- [X] T095 [US2] Create integration tests for batch processing of large datasets in feature-service/tests/integration/test_batch_processing.py
- [X] T096 [US2] Create contract tests for POST /dataset/build endpoint in feature-service/tests/contract/test_dataset_api.py
- [X] T097 [US2] Create contract tests for GET /dataset/list endpoint in feature-service/tests/contract/test_dataset_api.py
- [X] T098 [US2] Create contract tests for GET /dataset/{dataset_id} endpoint in feature-service/tests/contract/test_dataset_api.py
- [X] T099 [US2] Create contract tests for GET /dataset/{dataset_id}/download endpoint in feature-service/tests/contract/test_dataset_api.py
- [X] T100 [US2] Create contract tests for POST /model/evaluate endpoint in feature-service/tests/contract/test_dataset_api.py
- [X] T101 [US2] Create integration tests for dataset completion publisher in feature-service/tests/integration/test_dataset_publisher.py

### Implementation for User Story 2

- [X] T102 [P] [US2] Create Dataset model in feature-service/src/models/dataset.py
- [X] T103 [US2] Implement Parquet storage service in feature-service/src/storage/parquet_storage.py (read/write operations)
- [X] T104 [US2] Implement offline feature engine in feature-service/src/services/offline_engine.py (rebuilds features from historical data)
- [X] T105 [US2] Implement orderbook state reconstruction from snapshot + deltas in feature-service/src/services/offline_engine.py
- [X] T106 [US2] Implement rolling windows reconstruction for historical data in feature-service/src/services/offline_engine.py
- [X] T107 [US2] Implement target variable computation (regression: returns, classification: direction, risk-adjusted) in feature-service/src/services/dataset_builder.py
- [X] T108 [US2] Implement configurable threshold for classification targets (default 0.001 = 0.1%) in feature-service/src/services/dataset_builder.py
- [X] T109 [US2] Implement data leakage prevention validation in feature-service/src/services/dataset_builder.py
- [X] T110 [US2] Implement time-based dataset splitting in feature-service/src/services/dataset_builder.py
- [X] T111 [US2] Implement walk-forward validation strategy with configurable parameters in feature-service/src/services/dataset_builder.py
- [ ] T111a [US2] Enhance walk-forward validation to generate all folds in feature-service/src/services/dataset_builder.py (modify _split_walk_forward() to generate multiple folds instead of only first fold, iterate through all possible folds based on step_days, train_window_days, validation_window_days, start_date, end_date, return all folds as separate train/validation/test splits or combine train windows into single dataset, add fold metadata to track which fold each split belongs to, handle edge cases: insufficient data for folds, overlapping windows, ensure chronological order of folds)
- [ ] T111b [US2] Add unit tests for multiple fold generation in feature-service/tests/unit/test_walk_forward.py (test that all folds are generated correctly, test fold count calculation based on step_days and date range, test that folds do not overlap incorrectly, test edge cases: insufficient data, single fold, maximum folds, verify fold metadata is correct)
- [ ] T111c [US2] Add integration tests for walk-forward with multiple folds in feature-service/tests/integration/test_dataset_building.py (test end-to-end dataset building with walk-forward strategy generating multiple folds, verify all folds are saved correctly, verify fold metadata is stored in dataset record, test that model-service can process multiple folds)
- [X] T112 [US2] Implement random split strategy (for testing only, with temporal order preserved) in feature-service/src/services/dataset_builder.py
- [X] T113 [US2] Implement Dataset Builder service in feature-service/src/services/dataset_builder.py (orchestrates dataset building)
- [X] T114 [US2] Implement POST /dataset/build endpoint in feature-service/src/api/dataset.py with estimated_completion in response
- [X] T115 [US2] Implement batch processing for large datasets with progress tracking in feature-service/src/services/dataset_builder.py
- [X] T116 [US2] Implement queue management for concurrent dataset build requests in feature-service/src/services/dataset_builder.py
- [X] T117 [US2] Implement GET /dataset/list endpoint in feature-service/src/api/dataset.py
- [X] T118 [US2] Implement GET /dataset/{dataset_id} endpoint in feature-service/src/api/dataset.py
- [X] T119 [US2] Implement GET /dataset/{dataset_id}/download endpoint in feature-service/src/api/dataset.py
- [X] T120 [US2] Implement POST /model/evaluate endpoint in feature-service/src/api/dataset.py
- [X] T121 [US2] Implement dataset completion publisher in feature-service/src/publishers/dataset_publisher.py (publishes to features.dataset.ready queue)
- [X] T122 [US2] Implement error handling for missing historical data with available period suggestions in feature-service/src/services/dataset_builder.py
- [X] T123 [US2] Add logging for dataset building operations in feature-service/src/services/dataset_builder.py
- [X] T124 [US2] Implement feature identity validation (online vs offline comparison) in feature-service/src/services/offline_engine.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Datasets can be built from historical data with proper train/val/test splits.

---

## Phase 5: User Story 3 - Хранение сырых данных для последующей пересборки (Priority: P2)

**Goal**: Система хранит сырые маркет-данные в структурированном формате, обеспечивающем эффективную пересборку признаков на исторических данных. Данные хранятся минимальный период времени с возможностью архивирования.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что данные сохраняются в структурированном формате, доступны для чтения, и старые данные автоматически архивируются или удаляются после истечения срока хранения.

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T125 [P] [US3] Create test fixtures for raw market data (orderbook snapshots, deltas, trades, klines, ticker, funding, execution events) in feature-service/tests/fixtures/raw_data.py
- [X] T126 [P] [US3] Create unit tests for raw data storage service in feature-service/tests/unit/test_data_storage.py
- [X] T127 [P] [US3] Create unit tests for data organization by type in feature-service/tests/unit/test_data_organization.py
- [X] T128 [P] [US3] Create unit tests for data retention policy enforcement in feature-service/tests/unit/test_data_retention.py
- [X] T129 [P] [US3] Create unit tests for automatic archiving/deletion in feature-service/tests/unit/test_data_archiving.py
- [X] T130 [US3] Create integration tests for raw data storage workflow in feature-service/tests/integration/test_data_storage_workflow.py
- [X] T131 [US3] Create integration tests for data retrieval for dataset rebuilding in feature-service/tests/integration/test_data_retrieval.py

### Implementation for User Story 3

- [X] T132 [US3] Implement raw data storage service in feature-service/src/services/data_storage.py (writes to Parquet files organized by type and date)
- [X] T133 [US3] Implement data organization by type (orderbook snapshots, deltas, trades, klines, ticker, funding, execution events) in feature-service/src/services/data_storage.py
- [X] T134 [US3] Implement storage of all orderbook deltas for offline reconstruction in feature-service/src/services/data_storage.py
- [X] T135 [US3] Integrate raw data storage into market data consumer in feature-service/src/consumers/market_data_consumer.py
- [X] T136 [US3] Implement data retention policy enforcement in feature-service/src/services/data_storage.py (90 days default, configurable)
- [X] T137 [US3] Implement automatic archiving/deletion of expired data with archive recovery support in feature-service/src/services/data_storage.py
- [X] T138 [US3] Add logging for data storage operations in feature-service/src/services/data_storage.py

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
- [ ] T164 [P] [US5] Create unit tests for Feature Registry version management in feature-service/tests/unit/test_feature_registry_versioning.py (version storage, retrieval, activation, usage tracking, deletion prevention)
- [ ] T165 [P] [US5] Create unit tests for Feature Registry automatic fallback/rollback in feature-service/tests/unit/test_feature_registry_rollback.py (automatic rollback on validation errors, migration errors, runtime errors)
- [ ] T166a [P] [US5] Create unit tests for backward compatibility checking in feature-service/tests/unit/test_feature_registry_compatibility.py (detect removed features, changed names, changed logic, populate warnings/breaking_changes)
- [ ] T166b [P] [US5] Create unit tests for automatic schema migration in feature-service/tests/unit/test_feature_registry_migration.py (automatic migration execution, migration script handling, rollback on migration failure)
- [ ] T166 [US5] Create integration tests for Feature Registry loading and activation in feature-service/tests/integration/test_feature_registry_integration.py
- [ ] T167 [US5] Create contract tests for GET /feature-registry endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T168 [US5] Create contract tests for POST /feature-registry/reload endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169 [US5] Create contract tests for GET /feature-registry/validate endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169a [US5] Create contract tests for GET /feature-registry/versions endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169b [US5] Create contract tests for GET /feature-registry/versions/{version} endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169c [US5] Create contract tests for POST /feature-registry/versions/{version}/activate endpoint in feature-service/tests/contract/test_feature_registry_api.py (test activation, automatic rollback on failure, breaking changes acknowledgment)
- [ ] T169d [US5] Create contract tests for POST /feature-registry/rollback endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169e [US5] Create contract tests for GET /feature-registry/versions/{version}/usage endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [ ] T169f [US5] Create contract tests for DELETE /feature-registry/versions/{version} endpoint in feature-service/tests/contract/test_feature_registry_api.py (test deletion prevention when version is in use)

### Implementation for User Story 5

- [ ] T170 [P] [US5] Create Feature Registry model in feature-service/src/models/feature_registry.py
- [ ] T171 [US5] Implement Feature Registry configuration validation in feature-service/src/services/feature_registry.py (temporal boundaries, data leakage prevention, max_lookback_days)
- [ ] T172 [US5] Implement Feature Registry version management in feature-service/src/services/feature_registry.py (version storage, retrieval, activation tracking, usage tracking, deletion prevention if in use)
- [ ] T173 [US5] Implement Feature Registry loading and activation with automatic fallback to previous valid version on validation failure in feature-service/src/services/feature_registry.py (automatic rollback on validation errors, migration errors, runtime errors during initial feature computation test)
- [ ] T174 [US5] Integrate Feature Registry into feature computation (online and offline) in feature-service/src/services/feature_computer.py and feature-service/src/services/offline_engine.py
- [ ] T175 [US5] Implement GET /feature-registry endpoint in feature-service/src/api/feature_registry.py
- [ ] T176 [US5] Implement POST /feature-registry/reload endpoint in feature-service/src/api/feature_registry.py
- [ ] T177 [US5] Implement GET /feature-registry/validate endpoint in feature-service/src/api/feature_registry.py
- [ ] T178 [US5] Add logging for Feature Registry operations in feature-service/src/services/feature_registry.py
- [ ] T208 [US5] Implement GET /feature-registry/versions endpoint in feature-service/src/api/feature_registry.py (FR-058: list all versions with metadata)
- [ ] T209 [US5] Implement GET /feature-registry/versions/{version} endpoint in feature-service/src/api/feature_registry.py (FR-059: get specific version)
- [ ] T210 [US5] Implement POST /feature-registry/versions/{version}/activate endpoint in feature-service/src/api/feature_registry.py (FR-060: activate version with automatic rollback on failure, acknowledge breaking changes parameter)
- [ ] T211 [US5] Implement POST /feature-registry/rollback endpoint in feature-service/src/api/feature_registry.py (FR-061: automatic rollback to previous version)
- [ ] T212 [US5] Implement GET /feature-registry/versions/{version}/usage endpoint in feature-service/src/api/feature_registry.py (FR-062: check version usage)
- [ ] T213 [US5] Implement DELETE /feature-registry/versions/{version} endpoint in feature-service/src/api/feature_registry.py (FR-063: delete version only if not in use)
- [ ] T214 [US5] Implement backward compatibility checking in feature-service/src/services/feature_registry.py (FR-065: detect breaking changes - removed features, changed names, changed logic, populate compatibility_warnings and breaking_changes fields)
- [ ] T215 [US5] Implement automatic schema migration in feature-service/src/services/feature_registry.py (FR-064: automatic migration when activating new version with changed feature definitions, apply migration_script if provided)
- [ ] T216 [US5] Implement version usage tracking in feature-service/src/services/feature_registry.py (FR-067: track version usage in datasets and feature computations, prevent deletion of in-use versions)
- [ ] T217 [US5] Implement audit trail for version changes in feature-service/src/services/feature_registry.py (FR-068: track who activated/rolled back version, when, and reason - populate created_by, activated_by, activation_reason fields)
- [ ] T218 [US5] Skip (fields already included in T024 migration) - Database migration for versioning fields is included in T024. This task serves as placeholder to ensure all versioning functionality is implemented in services layer.

### Implementation for Feature Registry Version Management via Database (Files as Source of Truth)

**Goal**: Реализовать управление версиями Feature Registry через БД, где YAML файлы являются единственным источником истины для конфигурации, а БД хранит только метаданные и указатель на активную версию. Это позволяет версионировать конфигурации через Git, упрощает редактирование и исключает рассинхронизацию между БД и файлами.

**Architecture**:
- YAML файлы хранятся в `config/versions/feature_registry_v{version}.yaml`
- БД хранит только метаданные: `version`, `file_path`, `is_active`, метаданные активации
- При старте сервис загружает активную версию из БД, затем читает конфиг из файла
- API endpoints позволяют создавать, активировать версии и выполнять hot reload без рестарта

**Tests for Version Management via Database**:

- [ ] T305 [P] [US5] Create unit tests for FeatureRegistryVersionManager in feature-service/tests/unit/test_feature_registry_version_manager.py (test load_active_version: load from DB, read file, validate version match, test create_version: save file, save metadata to DB, test activate_version: update is_active flags, validate config from file, test get_version_file_path: construct correct path, test can_delete_version: check dataset usage, test sync_db_to_files: create files for all DB versions, test sync_files_to_db: create DB records for all files, test migrate_legacy_to_db: migrate existing feature_registry.yaml, test file_not_found_fallback: handle missing files gracefully, test version_mismatch_warning: warn if file version != DB version)

- [ ] T306 [P] [US5] Create unit tests for MetadataStorage feature registry version methods in feature-service/tests/unit/test_metadata_storage_feature_registry.py (test get_active_feature_registry_version: query with is_active=true, test create_feature_registry_version: insert with file_path and metadata, test activate_feature_registry_version: update is_active flags atomically, test list_feature_registry_versions: return all versions ordered by created_at, test get_feature_registry_version: get specific version by version string, test check_version_usage: count datasets using version, test rollback_version: update previous_version and is_active flags)

- [ ] T307 [P] [US5] Create integration tests for Feature Registry version management via DB in feature-service/tests/integration/test_feature_registry_version_db.py (test full lifecycle: create version, activate, load on startup, rollback, test concurrent activation prevention, test file deletion handling, test version activation with hot reload, test startup with missing active version fallback, test startup with missing file fallback)

- [ ] T308 [US5] Create contract tests for POST /feature-registry/versions endpoint in feature-service/tests/contract/test_feature_registry_api.py (test create version: save file, save to DB, return metadata, test create version with invalid config: validation error, test create version with duplicate version: conflict error)

- [ ] T309 [US5] Create contract tests for POST /feature-registry/versions/{version}/activate with hot reload in feature-service/tests/contract/test_feature_registry_api.py (test activation updates components without restart, test activation failure triggers rollback, test breaking changes require acknowledgment, test hot reload updates FeatureComputer and DatasetBuilder)

- [ ] T310 [US5] Create contract tests for POST /feature-registry/versions/{version}/sync-file endpoint in feature-service/tests/contract/test_feature_registry_api.py (test sync file to DB when file changed manually, test sync validates config before updating DB)

**Implementation for Version Management via Database**:

- [ ] T311 [US5] Create database migration to modify feature_registry_versions table in ws-gateway/migrations/021_modify_feature_registry_versions_for_file_based_storage.sql (ALTER TABLE to: remove config JSONB column (if exists), add file_path VARCHAR(500) NOT NULL column, add index on file_path for quick lookups, add migration script to backfill file_path from existing config if migrating from old schema, migration must be reversible: restore config JSONB and remove file_path if rolling back). Note: If T024 already has config JSONB, this migration removes it and adds file_path. Keep all other fields (is_active, validated_at, etc.) unchanged.

- [ ] T311a [US5] Apply database migration: `docker compose exec postgres psql -U ytrader -d ytrader -f /tmp/021_modify_feature_registry_versions_for_file_based_storage.sql`

- [ ] T312 [P] [US5] Create FeatureRegistryVersionManager service in feature-service/src/services/feature_registry_version_manager.py (implement load_active_version: query DB for active version, load file from file_path, validate version match between file and DB, return config dict, implement create_version: save config to file in versions/ directory, create DB record with file_path and metadata, validate config before saving, implement activate_version: validate config from file, atomically update is_active flags in DB (deactivate old, activate new), update loaded_at timestamp, return activated version metadata, implement get_version_file_path: construct path from versions_dir and version string, implement can_delete_version: check if version is used in any datasets via MetadataStorage, return boolean, implement sync_db_to_files: iterate all DB versions, ensure files exist on disk, create missing files from DB config if config JSONB still exists (migration helper), implement sync_files_to_db: scan versions/ directory, create DB records for files not in DB, extract version from filename, implement migrate_legacy_to_db: load existing feature_registry.yaml, extract version, save to versions/ directory, create DB record with is_active=true if no active version exists, add comprehensive error handling and structured logging for all operations)

- [ ] T313 [P] [US5] Extend MetadataStorage with feature registry version methods in feature-service/src/storage/metadata_storage.py (implement get_active_feature_registry_version: SELECT * FROM feature_registry_versions WHERE is_active = true LIMIT 1, return Dict or None, implement create_feature_registry_version: INSERT with version, file_path, is_active, created_at, created_by, validated_at (if validated), return created record, implement activate_feature_registry_version: BEGIN transaction, UPDATE feature_registry_versions SET is_active = false WHERE is_active = true, UPDATE feature_registry_versions SET is_active = true, loaded_at = NOW(), activated_by = $activated_by, activation_reason = $reason, previous_version = current_active_version WHERE version = $version, COMMIT transaction, return activated record, implement list_feature_registry_versions: SELECT * FROM feature_registry_versions ORDER BY created_at DESC, return List[Dict], implement get_feature_registry_version: SELECT * FROM feature_registry_versions WHERE version = $version, return Dict or None, implement check_version_usage: SELECT COUNT(*) FROM datasets WHERE feature_registry_version = $version, return integer count, add proper error handling and structured logging)

- [ ] T314 [US5] Update FeatureRegistryLoader to support database-driven mode in feature-service/src/services/feature_registry.py (add use_db: bool parameter to __init__, add version_manager: Optional[FeatureRegistryVersionManager] parameter, implement load_active_from_db: call version_manager.load_active_version() if use_db=True, fallback to file_mode if DB load fails, implement set_config: allow setting config manually (for hot reload), keep existing load() method for file_mode (backward compatibility), add validate_version_match: check file version matches DB version (warning if mismatch), update get_required_data_types and get_data_type_mapping to work with both modes)

- [ ] T315 [US5] Update startup process to load active version from database in feature-service/src/main.py (modify startup function: initialize MetadataStorage, initialize FeatureRegistryVersionManager with versions_dir from config, try to load_active_version() from DB, if success: use db_mode, if failure: fallback to legacy file_mode with automatic migration, update FeatureRegistryLoader initialization to use db_mode, pass version_manager to FeatureRegistryLoader, update DatasetBuilder to use version from loaded config, add comprehensive error handling and logging for startup sequence, ensure backward compatibility with existing deployments)

- [ ] T316 [P] [US5] Add configuration variables for version management in feature-service/src/config/__init__.py (add feature_registry_versions_dir: str = Field(default="/app/config/versions", env="FEATURE_REGISTRY_VERSIONS_DIR"), add feature_registry_use_db: bool = Field(default=True, env="FEATURE_REGISTRY_USE_DB"), add feature_registry_auto_migrate: bool = Field(default=True, env="FEATURE_REGISTRY_AUTO_MIGRATE"), update env.example with new variables)

- [ ] T317 [US5] Implement POST /feature-registry/versions endpoint in feature-service/src/api/feature_registry.py (accept FeatureRegistryVersionCreateRequest with config dict and version string, call version_manager.create_version(), validate config before saving, return created version metadata with file_path, handle validation errors and duplicate version errors, add authentication and authorization checks)

- [ ] T318 [US5] Update POST /feature-registry/versions/{version}/activate to support hot reload in feature-service/src/api/feature_registry.py (modify existing activate endpoint: call version_manager.activate_version(), after successful DB update: reload registry in memory without restart, update global feature_computer with new version, update dataset_builder._feature_registry_version, update feature_registry_loader config, return activation response with hot_reload: true, handle rollback on activation failure, add comprehensive error handling and logging)

- [ ] T319 [US5] Implement POST /feature-registry/versions/{version}/sync-file endpoint in feature-service/src/api/feature_registry.py (load config from file on disk, validate config, update DB metadata (validated_at, validation_errors if invalid), return sync result with validation status, use case: file edited manually, need to sync metadata in DB, handle file not found error)

- [ ] T320 [US5] Implement hot reload mechanism for Feature Registry in feature-service/src/main.py (create reload_registry_in_memory function: accept new config dict, recreate FeatureComputer with new version, update DatasetBuilder version (for new datasets only), update FeatureRegistryLoader config, update global variables atomically, add lock to prevent concurrent reloads, add comprehensive error handling and rollback on reload failure, integrate into activate endpoint via version_manager callback)

- [ ] T321 [US5] Create migration script for legacy feature_registry.yaml in feature-service/src/scripts/migrate_legacy_registry.py (load existing feature_registry.yaml from config.feature_registry_path, extract version from config, create versions/ directory if not exists, save file as versions/feature_registry_{version}.yaml, create DB record with is_active=true if no active version exists, validate config before migration, add logging and error handling, make script executable and add to Dockerfile if needed)

- [ ] T322 [US5] Update docker-compose.yml to mount versions directory as volume in docker-compose.yml (add volume mount for feature-service: ./feature-service/config/versions:/app/config/versions, ensure directory exists on host, document in README.md)

- [ ] T323 [US5] Update documentation in feature-service/README.md (document version management via database, explain files as source of truth approach, document API endpoints for version management, document environment variables (FEATURE_REGISTRY_VERSIONS_DIR, FEATURE_REGISTRY_USE_DB, FEATURE_REGISTRY_AUTO_MIGRATE), document hot reload capabilities, document migration from legacy mode, add examples of creating and activating versions via API)

**Checkpoint**: At this point, all user stories should be independently functional. Feature Registry manages feature configuration with validation. Version management is handled via database with files as source of truth, enabling Git versioning and hot reload without service restart.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Additional Tests

- [ ] T179 [P] Create end-to-end tests for complete feature computation workflow in feature-service/tests/e2e/test_feature_computation_workflow.py
- [ ] T180 [P] Create end-to-end tests for dataset building workflow in feature-service/tests/e2e/test_dataset_building_workflow.py
- [ ] T181 [P] Create performance tests for feature computation latency in feature-service/tests/performance/test_feature_latency.py
- [ ] T182 [P] Create performance tests for dataset building performance in feature-service/tests/performance/test_dataset_building_performance.py
- [ ] T183 [P] Create load tests for concurrent API requests in feature-service/tests/performance/test_api_load.py

### Statistics API Endpoints

**Purpose**: Provide HTTP API endpoints for service statistics and monitoring, equivalent to check_feature_service_data.py script functionality

#### Tests for Statistics API

- [ ] T219 [P] Create test fixtures for statistics data (storage, processing, rolling windows, subscriptions) in feature-service/tests/fixtures/statistics.py
- [ ] T220 [P] Create unit tests for Storage Statistics model in feature-service/tests/unit/test_storage_statistics.py
- [ ] T221 [P] Create unit tests for Processing Statistics model in feature-service/tests/unit/test_processing_statistics.py
- [ ] T222 [P] Create unit tests for Rolling Windows Statistics model in feature-service/tests/unit/test_rolling_windows_statistics.py
- [ ] T223 [P] Create unit tests for Statistics Service in feature-service/tests/unit/test_statistics_service.py
- [ ] T224 [US1] Create integration tests for GET /stats/storage endpoint in feature-service/tests/integration/test_stats_storage_api.py
- [ ] T225 [US1] Create integration tests for GET /stats/processing endpoint in feature-service/tests/integration/test_stats_processing_api.py
- [ ] T226 [US1] Create integration tests for GET /stats/rolling-windows endpoint in feature-service/tests/integration/test_stats_rolling_windows_api.py
- [ ] T227 [US1] Create integration tests for GET /stats/subscriptions endpoint in feature-service/tests/integration/test_stats_subscriptions_api.py
- [ ] T228 [US1] Create integration tests for GET /stats/service endpoint in feature-service/tests/integration/test_stats_service_api.py
- [ ] T229 [US1] Create contract tests for GET /stats/storage endpoint in feature-service/tests/contract/test_stats_api.py
- [ ] T230 [US1] Create contract tests for GET /stats/processing endpoint in feature-service/tests/contract/test_stats_api.py
- [ ] T231 [US1] Create contract tests for GET /stats/rolling-windows endpoint in feature-service/tests/contract/test_stats_api.py
- [ ] T232 [US1] Create contract tests for GET /stats/subscriptions endpoint in feature-service/tests/contract/test_stats_api.py
- [ ] T233 [US1] Create contract tests for GET /stats/service endpoint in feature-service/tests/contract/test_stats_api.py
- [ ] T234 [US1] Create contract tests for GET /stats endpoint (aggregated statistics) in feature-service/tests/contract/test_stats_api.py

#### Implementation for Statistics API

- [ ] T235 [P] [US1] Create Storage Statistics model in feature-service/src/models/storage_statistics.py (raw_data_storage: size_bytes, file_count; dataset_storage: size_bytes, file_count)
- [ ] T236 [P] [US1] Create Processing Statistics model in feature-service/src/models/processing_statistics.py (features_computed_1h, features_computed_24h, events_processed_1h, events_processed_24h, computed_at timestamp)
- [ ] T237 [P] [US1] Create Rolling Windows Statistics model in feature-service/src/models/rolling_windows_statistics.py (symbol, intervals: dict with counts per interval "1s", "3s", "15s", "1m", total_trades_count, total_klines_count, memory_usage_bytes_estimate, last_update timestamp)
- [ ] T238 [P] [US1] Create Subscription Statistics model in feature-service/src/models/subscription_statistics.py (queue_name, messages_in_queue, active_consumers, subscription_status)
- [ ] T239 [P] [US1] Create Service Statistics model in feature-service/src/models/service_statistics.py (tracked_symbols: list, service_status, last_health_check)
- [ ] T240 [P] [US1] Create Aggregated Statistics model in feature-service/src/models/statistics.py (storage, processing, rolling_windows: list per symbol, subscriptions: list, service, collected_at timestamp)
- [ ] T241 [US1] Implement Statistics Service in feature-service/src/services/statistics_service.py (collects all statistics: storage stats from filesystem, processing stats from logs/metrics, rolling windows stats from feature_computer, subscription stats from RabbitMQ connection, service stats from config and health)
- [ ] T242 [US1] Implement storage statistics collection (raw data and dataset storage sizes, file counts) in feature-service/src/services/statistics_service.py
- [ ] T243 [US1] Implement processing statistics collection (features computed count, events processed count from logs or metrics) in feature-service/src/services/statistics_service.py
- [ ] T244 [US1] Implement rolling windows statistics collection (per symbol: trade/klines counts per interval, memory usage estimate) in feature-service/src/services/statistics_service.py
- [ ] T245 [US1] Implement subscription statistics collection (queue status from RabbitMQ connection manager) in feature-service/src/services/statistics_service.py
- [ ] T246 [US1] Implement service statistics collection (tracked symbols from config, service status) in feature-service/src/services/statistics_service.py
- [ ] T247 [US1] Implement GET /stats/storage endpoint in feature-service/src/api/statistics.py (returns Storage Statistics with raw_data and dataset storage info)
- [ ] T248 [US1] Implement GET /stats/processing endpoint in feature-service/src/api/statistics.py (returns Processing Statistics with computation and event processing counts for 1h and 24h periods)
- [ ] T249 [US1] Implement GET /stats/rolling-windows endpoint in feature-service/src/api/statistics.py (returns list of Rolling Windows Statistics, one per tracked symbol, with optional symbol query parameter to filter)
- [ ] T250 [US1] Implement GET /stats/subscriptions endpoint in feature-service/src/api/statistics.py (returns list of Subscription Statistics for all ws-gateway.* queues)
- [ ] T251 [US1] Implement GET /stats/service endpoint in feature-service/src/api/statistics.py (returns Service Statistics with tracked symbols and service status)
- [ ] T252 [US1] Implement GET /stats endpoint (aggregated statistics) in feature-service/src/api/statistics.py (returns Aggregated Statistics combining all statistics in single response)
- [ ] T253 [US1] Add error handling for statistics collection failures (graceful degradation, return partial data with error flags) in feature-service/src/services/statistics_service.py
- [ ] T254 [US1] Add logging for statistics API requests in feature-service/src/api/statistics.py

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

### Grafana Observability Dashboard

- [ ] T196 [P] [Grafana] Create database migration for feature computation metrics table in ws-gateway/migrations/XXX_create_feature_computation_metrics_table.sql (feature_computation_metrics table with columns: id UUID PRIMARY KEY, symbol VARCHAR(20) NOT NULL, computation_timestamp TIMESTAMP NOT NULL, latency_ms DECIMAL(10,3) NOT NULL, feature_registry_version VARCHAR(50) NOT NULL, computation_interval VARCHAR(10) NOT NULL CHECK (computation_interval IN ('1s', '3s', '15s', '1m')), features_count INTEGER NOT NULL, error_count INTEGER DEFAULT 0, trace_id VARCHAR(100), created_at TIMESTAMP NOT NULL DEFAULT NOW(), indexes on computation_timestamp DESC, symbol, feature_registry_version for Grafana dashboard queries and time-series visualization)
- [ ] T196a [P] [Grafana] Apply database migration for feature_computation_metrics table in ws-gateway container: `docker compose run --rm ws-gateway psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/XXX_create_feature_computation_metrics_table.sql`
- [ ] T197 [P] [Grafana] Create database migration for data quality metrics table in ws-gateway/migrations/XXX_create_data_quality_metrics_table.sql (data_quality_metrics table with columns: id UUID PRIMARY KEY, symbol VARCHAR(20) NOT NULL, metric_timestamp TIMESTAMP NOT NULL, missing_rate DECIMAL(5,4) NOT NULL CHECK (missing_rate >= 0 AND missing_rate <= 1), anomaly_rate DECIMAL(5,4) NOT NULL CHECK (anomaly_rate >= 0 AND anomaly_rate <= 1), sequence_gaps_count INTEGER DEFAULT 0, desynchronization_events_count INTEGER DEFAULT 0, data_completeness_rate DECIMAL(5,4) NOT NULL CHECK (data_completeness_rate >= 0 AND data_completeness_rate <= 1), trace_id VARCHAR(100), created_at TIMESTAMP NOT NULL DEFAULT NOW(), indexes on metric_timestamp DESC, symbol for Grafana dashboard queries and time-series visualization)
- [ ] T197a [P] [Grafana] Apply database migration for data_quality_metrics table in ws-gateway container: `docker compose run --rm ws-gateway psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/XXX_create_data_quality_metrics_table.sql`
- [ ] T198 [P] [Grafana] Create database migration for dataset building metrics table in ws-gateway/migrations/XXX_create_dataset_building_metrics_table.sql (dataset_building_metrics table with columns: id UUID PRIMARY KEY, dataset_id UUID NOT NULL, symbol VARCHAR(20) NOT NULL, build_started_at TIMESTAMP NOT NULL, build_completed_at TIMESTAMP, build_status VARCHAR(20) NOT NULL CHECK (build_status IN ('building', 'ready', 'failed')), build_duration_seconds INTEGER, records_count INTEGER, train_records INTEGER, validation_records INTEGER, test_records INTEGER, feature_registry_version VARCHAR(50) NOT NULL, split_strategy VARCHAR(50), trace_id VARCHAR(100), created_at TIMESTAMP NOT NULL DEFAULT NOW(), indexes on build_started_at DESC, dataset_id, symbol, build_status for Grafana dashboard queries and performance tracking)
- [ ] T198a [P] [Grafana] Apply database migration for dataset_building_metrics table in ws-gateway container: `docker compose run --rm ws-gateway psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/XXX_create_dataset_building_metrics_table.sql`
- [ ] T199 [P] [Grafana] Create database migration for API endpoint metrics table in ws-gateway/migrations/XXX_create_api_endpoint_metrics_table.sql (api_endpoint_metrics table with columns: id UUID PRIMARY KEY, endpoint VARCHAR(200) NOT NULL, method VARCHAR(10) NOT NULL, response_time_ms DECIMAL(10,3) NOT NULL, status_code INTEGER NOT NULL, symbol VARCHAR(20), request_timestamp TIMESTAMP NOT NULL, trace_id VARCHAR(100), created_at TIMESTAMP NOT NULL DEFAULT NOW(), indexes on request_timestamp DESC, endpoint, method, status_code for Grafana dashboard queries and API performance monitoring)
- [ ] T199a [P] [Grafana] Apply database migration for api_endpoint_metrics table in ws-gateway container: `docker compose run --rm ws-gateway psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/XXX_create_api_endpoint_metrics_table.sql`
- [ ] T200 [Grafana] Implement feature computation metrics persistence in feature-service/src/services/feature_computer.py (log metrics to feature_computation_metrics table after each computation: symbol, computation_timestamp, latency_ms, feature_registry_version, computation_interval, features_count, error_count if any errors occurred, trace_id, handle database errors gracefully with logging and continue processing)
- [ ] T201 [Grafana] Implement data quality metrics persistence in feature-service/src/services/data_quality.py (log metrics to data_quality_metrics table periodically (every 1 minute or configurable interval): symbol, metric_timestamp, missing_rate, anomaly_rate, sequence_gaps_count, desynchronization_events_count, data_completeness_rate, trace_id, handle database errors gracefully)
- [ ] T202 [Grafana] Implement dataset building metrics persistence in feature-service/src/services/dataset_builder.py (log metrics to dataset_building_metrics table when dataset building starts, updates progress, and completes: dataset_id, symbol, build_started_at, build_completed_at, build_status, build_duration_seconds when completed, records_count, train_records, validation_records, test_records, feature_registry_version, split_strategy, trace_id, handle database errors gracefully)
- [ ] T203 [Grafana] Implement API endpoint metrics persistence in feature-service/src/api/middleware/metrics.py (log metrics to api_endpoint_metrics table for all API requests: endpoint, method, response_time_ms, status_code, symbol if applicable, request_timestamp, trace_id, handle database errors gracefully, use async logging to avoid blocking request processing)
- [ ] T204 [P] [Grafana] Extend health check endpoint for Grafana monitoring in feature-service/src/api/health.py (add flat fields to HealthResponse: database_connected boolean, queue_connected boolean, feature_registry_loaded boolean, latest_feature_computation_timestamp, average_latency_ms_last_5min, data_quality_ok boolean, active_dataset_builds_count, last_dataset_build_duration_seconds, in addition to existing checks object for backward compatibility, enable Grafana System Health dashboard panel to extract dependency status directly without nested JSON parsing)
- [ ] T255 [US2] Add validation and warning for empty test split in feature-service/src/services/dataset_builder.py (in _write_dataset_splits method, check if test split DataFrame is empty before skipping file creation, if test split is empty, log structured warning with dataset_id, symbol, test_period_start, test_period_end, reason (no_data_in_period, data_gap, insufficient_historical_data), update dataset metadata with warning flag or empty_test_split flag, ensure warning is visible in dataset status/metadata for Model Service to handle gracefully, document that empty test split may occur if historical data is insufficient for test period, add test_records=0 to metadata to indicate empty split)

**Note**: Grafana dashboard creation tasks (T205-T207) are located in `specs/001-grafana-monitoring/tasks.md` as they belong to the Grafana monitoring service. See Phase 12: Feature Service Observability Dashboard tasks in that file.

---

## Phase 9: Historical Data Backfilling (Priority: P1)

**Goal**: Enable Feature Service to fetch historical market data from Bybit REST API when insufficient data is available for model training. This allows immediate model training without waiting for data accumulation through WebSocket streams.

**Context**: Feature Service currently receives data only through WebSocket streams via ws-gateway, which means it takes time to accumulate sufficient historical data (e.g., 38 days for training). Bybit REST API v5 provides `/v5/market/kline` endpoint that allows fetching up to 200 historical candles per request, enabling backfilling of missing historical data.

**Independent Test**: Can be fully tested by requesting backfilling for a specific period, verifying that historical data is fetched from Bybit REST API, saved to Parquet storage in the same format as WebSocket data, and then used by Dataset Builder for training dataset creation.

### Implementation for Historical Data Backfilling

#### Part 1: Feature Registry Data Type Analysis

- [X] T256 [US2] Add method to determine required data types from Feature Registry in feature-service/src/services/feature_registry.py (add get_required_data_types() method to FeatureRegistryLoader class, extract unique input_sources from all features in registry, return set of required data types: ["orderbook", "kline", "trades", "ticker", "funding"], handle empty registry gracefully, cache result for performance, add unit tests for this method)
- [X] T256a [US2] Add data type mapping utility in feature-service/src/services/feature_registry.py (add get_data_type_mapping() method that maps Feature Registry input_sources to actual data storage types: "orderbook" → ["orderbook_snapshots", "orderbook_deltas"], "kline" → ["klines"], "trades" → ["trades"], "ticker" → ["ticker"], "funding" → ["funding"], return dict mapping input_source to list of storage types, used by DatasetBuilder and BackfillingService to determine which data files to load/backfill)

#### Part 2: Bybit REST API Client

- [X] T257 [P] [US2] Create Bybit REST API client in feature-service/src/utils/bybit_client.py (async HTTP client for Bybit REST API v5 with HMAC-SHA256 authentication, support for public endpoints (no auth required for market data), support for authenticated endpoints (optional for future use), retry logic with exponential backoff for rate limits, timeout handling, error handling with BybitAPIError exceptions, base URL configuration for mainnet/testnet, similar to order-manager/src/utils/bybit_client.py but adapted for Feature Service needs)
- [X] T258 [P] [US2] Add Bybit API configuration to feature-service/src/config/__init__.py (BYBIT_API_KEY optional for authenticated endpoints, BYBIT_API_SECRET optional for authenticated endpoints, BYBIT_ENVIRONMENT=mainnet|testnet default mainnet, BYBIT_REST_BASE_URL derived from environment, BYBIT_RATE_LIMIT_DELAY_MS=100 for delay between requests to respect rate limits, document that API keys are optional for public market data endpoints)
- [X] T259 [P] [US2] Add Bybit API configuration to env.example (BYBIT_API_KEY= optional, BYBIT_API_SECRET= optional, BYBIT_ENVIRONMENT=mainnet, BYBIT_RATE_LIMIT_DELAY_MS=100, document that keys are optional for public market data backfilling)

#### Part 3: Dataset Builder Data Type Optimization

- [X] T260 [US2] Update DatasetBuilder to use Feature Registry for determining required data types in feature-service/src/services/dataset_builder.py (in _read_historical_data method, use FeatureRegistryLoader.get_required_data_types() to determine which data types to load, use get_data_type_mapping() to map input_sources to storage types, only load data types that are actually needed by features, maintain backward compatibility: if Feature Registry not available, load all data types as before, add logging to indicate which data types are being loaded and why, optimize performance by skipping unnecessary data loading)
- [X] T260a [US2] Update DatasetBuilder constructor to accept FeatureRegistryLoader in feature-service/src/services/dataset_builder.py (add optional feature_registry_loader parameter to __init__, if provided, use it to determine required data types, if not provided, fall back to loading all data types, update main.py to pass FeatureRegistryLoader instance to DatasetBuilder)

#### Part 4: Backfilling Service

- [X] T261 [US2] Create backfilling service in feature-service/src/services/backfilling_service.py (BackfillingService class that fetches historical data from Bybit REST API, methods: backfill_klines(symbol, start_date, end_date, interval) -> List[Dict], backfill_trades(symbol, start_date, end_date) -> List[Dict] optional, backfill_orderbook_snapshots(symbol, start_date, end_date) -> List[Dict] optional, handle pagination for klines (200 candles per request), split large periods into chunks, respect rate limits with configurable delays, convert Bybit API response format to internal format matching WebSocket data structure, handle errors gracefully with retry logic, log backfilling progress with structured logging, use FeatureRegistryLoader to determine which data types to backfill based on required features)
- [X] T262 [US2] Implement kline backfilling in feature-service/src/services/backfilling_service.py (use Bybit REST API GET /v5/market/kline endpoint, parameters: category=spot, symbol, interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W), start timestamp in milliseconds, end timestamp in milliseconds, limit=200 max per request, handle pagination by splitting period into chunks of 200 candles, convert response format: [startTime, open, high, low, close, volume, turnover] to internal format with timestamp, open, high, low, close, volume fields matching WebSocket kline events, save to Parquet using existing ParquetStorage.write_klines method)
- [X] T263 [US2] Integrate backfilling with data storage in feature-service/src/services/backfilling_service.py (use existing ParquetStorage service to save backfilled data, use existing DataStorageService for file organization, ensure backfilled data is saved in same format and structure as WebSocket data: /data/raw/klines/{date}/{symbol}.parquet, ensure backfilled data can be read by DatasetBuilder using existing read_klines_range method, handle date-based file organization matching existing structure)
- [X] T263a [US2] Add data validation after backfilling save in feature-service/src/services/backfilling_service.py (after saving each chunk/date, immediately read data back from Parquet file to verify it was saved correctly, verify record count matches downloaded data (compare downloaded records count with read records count), verify data structure: all required fields present (timestamp, open, high, low, close, volume for klines), verify data types are correct (timestamp is datetime, prices are float, volume is float), verify data integrity: no corrupted records, timestamps are in valid range, prices are positive, if validation fails: mark date as failed, log validation error with details (date, symbol, data_type, error_reason, expected_count, actual_count), delete corrupted file if validation fails, retry backfilling for failed dates, track validation failures in job metadata for monitoring and debugging)
- [X] T264 [US2] Add data availability check before backfilling in feature-service/src/services/backfilling_service.py (check existing Parquet files to determine which dates need backfilling, skip dates that already have data, only backfill missing periods, log which dates are being backfilled vs skipped, optimize to avoid redundant API calls, use FeatureRegistryLoader to determine which data types actually need backfilling based on required features)

#### Part 5: API Endpoints

- [X] T265 [P] [US2] Create backfilling API endpoints in feature-service/src/api/backfill.py (POST /backfill/historical endpoint: accepts symbol, start_date, end_date, data_types (optional, if not provided, use Feature Registry to determine required types), returns backfill_job_id, GET /backfill/status/{job_id} endpoint: returns backfill status, progress, completed dates, failed dates, POST /backfill/auto endpoint: automatically backfill missing data for a symbol up to configured maximum days, use Feature Registry to determine which data types to backfill, handle API key authentication via verify_api_key middleware, return structured responses with job status)
- [X] T266 [US2] Implement backfilling job tracking in feature-service/src/services/backfilling_service.py (track backfilling jobs with job_id, status (pending, in_progress, completed, failed), progress (dates_completed, dates_total, current_date), start_time, end_time, error_message if failed, store job metadata in memory or database for status queries, support cancellation of in-progress jobs, track which data types are being backfilled)

#### Part 6: Automatic Backfilling Integration

- [X] T267 [US2] Integrate automatic backfilling into dataset builder in feature-service/src/services/dataset_builder.py (in _check_data_availability method, if insufficient data detected and FEATURE_SERVICE_BACKFILL_AUTO=true, automatically trigger backfilling for missing periods, use FeatureRegistryLoader to determine which data types need backfilling based on required features, wait for backfilling to complete before proceeding with dataset build, log automatic backfilling trigger with dataset_id, symbol, missing_period, required_data_types, handle backfilling failures gracefully with fallback to available data, add configuration check for FEATURE_SERVICE_BACKFILL_AUTO)
- [X] T268 [US2] Add backfilling configuration to feature-service/src/config/__init__.py (FEATURE_SERVICE_BACKFILL_ENABLED=true/false to enable/disable backfilling feature, FEATURE_SERVICE_BACKFILL_AUTO=true/false to enable/disable automatic backfilling when data insufficient, FEATURE_SERVICE_BACKFILL_MAX_DAYS=90 for maximum days to backfill in one operation, FEATURE_SERVICE_BACKFILL_RATE_LIMIT_DELAY_MS=100 for delay between API requests, FEATURE_SERVICE_BACKFILL_DEFAULT_INTERVAL=1 for default kline interval (1 minute), document configuration options)
- [X] T269 [US2] Add backfilling configuration to env.example (FEATURE_SERVICE_BACKFILL_ENABLED=true, FEATURE_SERVICE_BACKFILL_AUTO=true, FEATURE_SERVICE_BACKFILL_MAX_DAYS=40, FEATURE_SERVICE_BACKFILL_RATE_LIMIT_DELAY_MS=100, FEATURE_SERVICE_BACKFILL_DEFAULT_INTERVAL=1, document that backfilling uses Bybit REST API public endpoints and does not require API keys for market data)

#### Part 7: Error Handling and Logging

- [X] T270 [US2] Add comprehensive error handling for backfilling in feature-service/src/services/backfilling_service.py (handle Bybit API rate limits (429 errors) with exponential backoff, handle network timeouts with retry logic, handle invalid date ranges with clear error messages, handle missing symbols with 404 errors, log all errors with structured logging including symbol, date_range, data_types, error_type, retry_count, continue processing other dates if one date fails, track failed dates for retry or manual intervention)
- [X] T271 [US2] Add structured logging for backfilling operations in feature-service/src/services/backfilling_service.py (log backfilling start with symbol, date_range, data_types (from Feature Registry), log progress for each date/chunk processed, log completion with total_dates, successful_dates, failed_dates, duration_seconds, log rate limit delays and retries, include trace_id for request flow tracking, log data volume saved (bytes, records), log which data types were backfilled and why)

#### Part 8: Testing

- [X] T272 [P] [US2] Create unit tests for Feature Registry data type analysis in feature-service/tests/unit/test_feature_registry.py (test get_required_data_types() method with various Feature Registry configurations, test get_data_type_mapping() method, test with empty registry, test with registry containing all data types, test with registry containing subset of data types, verify correct mapping of input_sources to storage types)
- [X] T273 [P] [US2] Create unit tests for DatasetBuilder data type optimization in feature-service/tests/unit/test_dataset_builder.py (test _read_historical_data only loads required data types when Feature Registry provided, test fallback to loading all data types when Feature Registry not provided, test data type mapping logic, use mocks for FeatureRegistryLoader and ParquetStorage)
- [X] T274 [P] [US2] Create unit tests for Bybit REST API client in feature-service/tests/unit/test_bybit_client.py (test HMAC-SHA256 signature generation, test public endpoint requests (no auth), test authenticated endpoint requests (with API keys), test retry logic for 429 errors, test timeout handling, test error parsing from Bybit API responses, use mocks for HTTP requests)
- [X] T275 [P] [US2] Create unit tests for backfilling service in feature-service/tests/unit/test_backfilling_service.py (test kline backfilling with pagination, test date range splitting into chunks, test data format conversion from Bybit API to internal format, test data availability check logic, test error handling and retry logic, test Feature Registry integration for determining data types to backfill, test data validation after save: verify validation passes for correct data, verify validation fails for corrupted data, verify failed dates are retried, verify corrupted files are deleted on validation failure, use mocks for Bybit API client, ParquetStorage, and FeatureRegistryLoader)
- [X] T276 [P] [US2] Create integration tests for backfilling in feature-service/tests/integration/test_backfilling_integration.py (test end-to-end backfilling: fetch from Bybit API, save to Parquet, verify data validation passes after save, verify data can be read by DatasetBuilder, test automatic backfilling trigger in dataset builder with Feature Registry, test backfilling API endpoints, test that only required data types are backfilled based on Feature Registry, test validation failure handling: simulate corrupted save, verify file is deleted and date is marked as failed, verify retry mechanism for failed dates, use test Bybit API or mocks, verify data format matches WebSocket data format)
- [X] T277 [P] [US2] Create contract tests for backfilling API in feature-service/tests/contract/test_backfill_api.py (test POST /backfill/historical endpoint contract, test GET /backfill/status/{job_id} endpoint contract, test error responses, test authentication requirements, test automatic data type determination from Feature Registry, verify response schemas match API documentation)

**Checkpoint**: At this point, Feature Service should be able to fetch historical data from Bybit REST API when insufficient data is available, automatically backfill missing data when dataset building is requested, and provide manual backfilling via API endpoints. This enables immediate model training without waiting for data accumulation.

**Summary of Changes for Historical Data Backfilling**:
- Feature Registry analysis: methods to determine required data types from Feature Registry
- Dataset Builder optimization: only loads data types required by features (reduces memory and I/O)
- Bybit REST API client added for fetching historical market data
- Backfilling service implements kline data fetching with pagination
- Backfilling service uses Feature Registry to determine which data types to backfill
- Backfilled data saved in same format as WebSocket data (Parquet)
- Data validation after save: immediate verification that saved data is correct and readable, prevents wasted API calls by catching save failures early, validates record count, data structure, data types, and data integrity
- Automatic backfilling integrated into dataset builder with Feature Registry awareness
- Manual backfilling available via REST API endpoints with automatic data type determination
- Comprehensive error handling and logging for backfilling operations
- Configuration for enabling/disabling backfilling and auto-backfilling

#### Part 9: Extended Data Type Support for Backfilling

**Context**: Currently, backfilling service only supports klines data type. Feature Registry requires additional data types (trades, orderbook_snapshots, orderbook_deltas, ticker, funding) for feature computation. To enable complete backfilling based on Feature Registry requirements, we need to implement backfilling methods for all supported data types.

**Note**: This is a future enhancement. Current implementation supports klines backfilling, which is sufficient for basic use cases. Other data types will be added as needed.

- [X] T278 [US2] Implement trades backfilling in feature-service/src/services/backfilling_service.py (add backfill_trades(symbol, start_date, end_date) -> List[Dict] method, use Bybit REST API GET /v5/market/recent-trade or /v5/market/public-trading-history endpoint, handle pagination with cursor-based or timestamp-based pagination, convert Bybit API response format to internal format matching WebSocket trade events: timestamp, price, quantity, side, symbol, save to Parquet using ParquetStorage.write_trades method, add data validation after save, handle rate limits and errors, add unit and integration tests)
- [X] T279 [US2] Implement orderbook snapshots backfilling in feature-service/src/services/backfilling_service.py (add backfill_orderbook_snapshots(symbol, start_date, end_date) -> List[Dict] method, use Bybit REST API GET /v5/market/orderbook endpoint with appropriate parameters, handle pagination if available, convert Bybit API response format to internal format matching WebSocket orderbook_snapshot events: timestamp, bids, asks, symbol, save to Parquet using ParquetStorage.write_orderbook_snapshots method, add data validation after save, handle rate limits and errors, add unit and integration tests)
- [X] T280 [US2] Implement orderbook deltas backfilling in feature-service/src/services/backfilling_service.py (add backfill_orderbook_deltas(symbol, start_date, end_date) -> List[Dict] method, note: Bybit REST API may not provide historical orderbook deltas, may need to reconstruct from snapshots or use alternative approach, if available, use appropriate Bybit REST API endpoint, convert to internal format matching WebSocket orderbook_delta events: timestamp, bids, asks, symbol, sequence, save to Parquet using ParquetStorage.write_orderbook_deltas method, add data validation after save, handle rate limits and errors, add unit and integration tests)
- [X] T281 [US2] Implement ticker backfilling in feature-service/src/services/backfilling_service.py (add backfill_ticker(symbol, start_date, end_date) -> List[Dict] method, use Bybit REST API GET /v5/market/tickers endpoint, handle pagination if needed, convert Bybit API response format to internal format matching WebSocket ticker events: timestamp, last_price, bid_price, ask_price, volume_24h, symbol, save to Parquet using ParquetStorage.write_ticker method, add data validation after save, handle rate limits and errors, add unit and integration tests)
- [X] T282 [US2] Implement funding rate backfilling in feature-service/src/services/backfilling_service.py (add backfill_funding(symbol, start_date, end_date) -> List[Dict] method, use Bybit REST API GET /v5/market/funding/history endpoint, handle pagination, convert Bybit API response format to internal format matching WebSocket funding_rate events: timestamp, funding_rate, symbol, save to Parquet using ParquetStorage.write_funding method, add data validation after save, handle rate limits and errors, add unit and integration tests)
- [X] T283 [US2] Update backfilling service to support all data types in feature-service/src/services/backfilling_service.py (update _backfill_job_task to handle all data types: trades, orderbook_snapshots, orderbook_deltas, ticker, funding, update _check_data_availability to check all data types, update _save_klines pattern to create generic _save_data method or separate methods for each type, update _validate_saved_data to validate all data types, ensure all data types are properly integrated with Feature Registry data type mapping, update supported_types set in backfill_historical to include all implemented types, add comprehensive logging for all data types)
- [X] T284 [US2] Add tests for extended data type backfilling in feature-service/tests/unit/test_backfilling_service.py and feature-service/tests/integration/test_backfilling_integration.py (add unit tests for each new backfilling method: backfill_trades, backfill_orderbook_snapshots, backfill_orderbook_deltas, backfill_ticker, backfill_funding, test pagination for each type, test data format conversion, test error handling, add integration tests for end-to-end backfilling with all data types, test that Feature Registry correctly determines all required types, verify all data types are saved and validated correctly)

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
- **Grafana Observability Dashboard (Phase 8)**: Depends on US1 (feature computation), US2 (dataset building), US4 (data quality monitoring) for metrics collection, can run in parallel with other polish tasks once underlying services are implemented

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Depends on US1 for feature computation logic (offline engine uses same feature computation code)
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories (can run in parallel with US1/US2)
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 for orderbook manager and data quality monitoring
- **User Story 5 (P3)**: Can start after Foundational (Phase 2) - Depends on US1 and US2 for feature computation integration
- **Historical Data Backfilling (Phase 9)**: Depends on US2 (dataset building) and US3 (raw data storage) - requires ParquetStorage and DataStorageService infrastructure, enables immediate model training by fetching historical data from Bybit REST API when insufficient data is available

### Task Dependencies for Historical Data Backfilling (T256-T277)

**Note**: These tasks implement backfilling of historical market data from Bybit REST API to enable immediate model training without waiting for data accumulation.

- **T256** (Bybit REST API client): Can be done independently, must complete before T259, T260 (backfilling service needs client to fetch data from Bybit API)
- **T257** (Bybit API configuration): Can be done independently, must complete before T256 (client needs configuration values)
- **T258** (env.example configuration): Can be done independently, can be done in parallel with T257
- **T260** (DatasetBuilder data type optimization): Depends on T256, T256a (Feature Registry analysis exists), must complete after US2 (dataset builder exists), can be done in parallel with T261-T264
- **T260a** (DatasetBuilder constructor update): Depends on T260, must complete after main.py integration
- **T261** (backfilling service): Depends on T256, T256a (Feature Registry analysis exists), T257 (Bybit client exists), T263 (data storage integration), must complete before T262, T264 (kline backfilling and data availability check need service structure)
- **T262** (kline backfilling): Depends on T261 (backfilling service exists), T263 (data storage integration), implements core backfilling logic
- **T263** (data storage integration): Depends on US3 (ParquetStorage and DataStorageService exist), must complete before T261, T262 (backfilling needs storage services)
- **T263a** (data validation after save): Depends on T263 (data storage integration exists), must complete after T262 (kline backfilling exists), validates that saved data is correct and readable, prevents wasted API calls by catching save failures immediately
- **T264** (data availability check): Depends on T261 (backfilling service exists), T256, T256a (Feature Registry analysis), can be done in parallel with T262, T263a
- **T265** (backfilling API endpoints): Depends on T261 (backfilling service exists), T266 (job tracking exists), T256, T256a (Feature Registry analysis), must complete before T267 (automatic integration may use API)
- **T266** (job tracking): Can be done independently, must complete before T265 (API endpoints need job tracking)
- **T267** (automatic integration): Depends on T261 (backfilling service exists), T264 (data availability check exists), T268 (configuration exists), T256, T256a (Feature Registry analysis), must complete after US2 (dataset builder exists)
- **T268** (backfilling configuration): Can be done independently, must complete before T267 (automatic integration needs configuration)
- **T269** (env.example backfilling config): Can be done independently, can be done in parallel with T268
- **T270** (error handling): Depends on T261 (backfilling service exists), can be done in parallel with T271
- **T271** (structured logging): Depends on T261 (backfilling service exists), can be done in parallel with T270
- **T272** (unit tests for Feature Registry): Depends on T256, T256a (Feature Registry analysis exists), can be done in parallel with T273-T277
- **T273** (unit tests for DatasetBuilder): Depends on T260 (DatasetBuilder optimization exists), can be done in parallel with T272, T274-T277
- **T274** (unit tests for client): Depends on T257 (Bybit client exists), can be done in parallel with T272, T273, T275-T277
- **T275** (unit tests for service): Depends on T261 (backfilling service exists), can be done in parallel with T272-T274, T276, T277
- **T276** (integration tests): Depends on T261 (backfilling service exists), T265 (API endpoints exist), T260 (DatasetBuilder optimization), can be done in parallel with T272-T275, T277
- **T277** (contract tests): Depends on T265 (API endpoints exist), can be done in parallel with T272-T276
- **Note**: T256, T256a, T257, T258, T259 can be done in parallel (setup tasks), T272-T277 can be done in parallel (test tasks), T270 and T271 can be done in parallel (enhancement tasks)

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
- Grafana observability tasks (T196-T207) can run in parallel after corresponding services are implemented: T196-T199 (database migrations) can run in parallel, T200-T203 (metrics persistence) can run in parallel after services are ready, T204-T207 (dashboard creation) can run in parallel after metrics tables are created
- Historical Data Backfilling tasks (T256-T277, T263a) can run in parallel after US2 and US3 are complete: T256-T259 (setup tasks including Feature Registry analysis) can run in parallel, T260-T260a (DatasetBuilder optimization) can run after T256-T256a, T261-T264, T263a (core backfilling with validation) can run sequentially after T256-T256a, T257, T263, T265-T266 (API and tracking) can run in parallel, T267-T269 (integration and config) can run in parallel, T270-T271 (enhancements) can run in parallel, T272-T277 (tests) can run in parallel

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

- **Total Tasks**: 343 (added 9 Grafana observability tasks T196-T204 + 4 migration application tasks T196a-T199a + 17 Feature Registry versioning tasks T208-T218, T166a-T166b, T169a-T169f: version management, backward compatibility, automatic migration, rollback, audit trail + 36 Statistics API tasks T219-T254: HTTP API endpoints for service statistics and monitoring + 1 dataset validation task T255: empty test split validation and warning + 23 Historical Data Backfilling tasks T256-T277, T263a: Feature Registry data type analysis, DatasetBuilder optimization, Bybit REST API client, backfilling service with data validation, API endpoints, automatic integration, error handling, testing + 3 walk-forward enhancement tasks T111a-T111c: multiple folds generation, tests + 3 new price feature tasks T285-T287: volatility for N candles, returns for N candles, volume z-score + 3 unit test tasks T288-T290 for new price features + 15 advanced feature tasks T291-T304: technical indicators (RSI, EMA, MACD, Bollinger Bands) with pandas_ta dependency, orderbook slope, orderbook churn rate with OrderbookManager extension, Rate of Change (ROC), relative volume, Feature Registry YAML updates + 36 Feature Registry Version Management via Database tasks T305-T323: database schema modification (remove config JSONB, add file_path), FeatureRegistryVersionManager service with file-based source of truth, MetadataStorage extensions, FeatureRegistryLoader db_mode support, startup process updates, API endpoints for version creation and hot reload activation, migration script, docker-compose volume mounting, documentation). Dashboard creation tasks (3 tasks) are in `specs/001-grafana-monitoring/tasks.md`
- **Phase 1 (Setup)**: 10 tasks (added test structure setup)
- **Phase 2 (Foundational)**: 29 tasks (12 tests + 17 implementation: 3 migrations + 3 migration application tasks)
- **Phase 3 (User Story 1)**: 64 tasks (29 tests + 35 implementation, added T285-T287 for new price features: volatility for N candles, returns for N candles, volume z-score, added T288-T290 for unit tests for new price features, added T291-T295 for unit tests for advanced features: technical indicators, orderbook slope/churn rate, ROC, relative volume, added T296-T304 for implementation of advanced features: technical indicators module, orderbook slope/churn rate, ROC, relative volume, Feature Registry updates, OrderbookManager extension, pandas_ta dependency)
- **Phase 4 (User Story 2)**: 48 tasks (25 tests + 26 implementation, added T111a-T111c for walk-forward multiple folds enhancement)
- **Phase 5 (User Story 3)**: 14 tasks (7 tests + 7 implementation)
- **Phase 6 (User Story 4)**: 22 tasks (11 tests + 11 implementation)
- **Phase 7 (User Story 5)**: 71 tasks (22 tests + 49 implementation, including original version management T161-T218 + 19 new tasks for database-driven version management T305-T323: 6 test tasks for FeatureRegistryVersionManager and DB integration, 13 implementation tasks for file-based source of truth architecture with hot reload, database schema modification, service layer updates, API endpoints, migration script, docker-compose configuration, documentation)
- **Phase 8 (Polish)**: 61 tasks (5 additional tests + 12 implementation + 7 Grafana observability tasks: T196-T199 database migrations + T196a-T199a migration application, T200-T203 metrics persistence, T204 health check extension + 36 Statistics API tasks T219-T254: 16 tests + 20 implementation + 1 dataset validation task T255: empty test split validation). Grafana dashboard creation tasks (T205-T207) are in `specs/001-grafana-monitoring/tasks.md`
- **Phase 9 (Historical Data Backfilling)**: 23 tasks (T256-T277, T263a: Feature Registry data type analysis and mapping, DatasetBuilder optimization to load only required data types, Bybit REST API client, backfilling service with Feature Registry awareness and data validation after save, API endpoints for manual and automatic backfilling, integration with dataset builder, comprehensive error handling and logging, unit/integration/contract tests)

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) = 103 tasks (10 + 29 + 64)

**Test Tasks Breakdown**:

- **Test Fixtures/Mocks/Stubs**: 26 tasks (added 1 Statistics API fixture: T219)
- **Unit Tests**: 60 tasks (added 4 Statistics API unit tests: T220-T223, added 3 price feature unit tests: T288-T290, added 5 advanced feature unit tests: T291-T295 for technical indicators, orderbook slope/churn rate, ROC, relative volume, added 3 Feature Registry version management unit tests: T305-T307 for FeatureRegistryVersionManager, MetadataStorage extensions, DB integration)
- **Integration Tests**: 26 tasks (added 5 Statistics API integration tests: T224-T228, added 1 Feature Registry version management integration test: T307)
- **Contract Tests**: 24 tasks (added 6 Statistics API contract tests: T229-T234, added 3 Feature Registry version management contract tests: T308-T310 for version creation, activation with hot reload, file sync)
- **E2E/Performance Tests**: 5 tasks
- **Total Test Tasks**: 141 tasks (added 16 Statistics API test tasks + 6 Feature Registry version management test tasks)

**Parallel Opportunities**:

- Setup phase: 4 tasks can run in parallel
- Foundational phase: 12 test tasks can run in parallel, 9 implementation tasks can run in parallel
- User Story 1: 4 test fixtures can run in parallel, 8 unit tests can run in parallel, 3 models can run in parallel
- User Stories 1 and 3 can start in parallel after foundational phase
