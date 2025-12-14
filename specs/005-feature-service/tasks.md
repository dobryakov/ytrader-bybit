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

### Minimal Feature Set (7 features) - Based on docs/feature-registry-todo.md

**Purpose**: Implement minimal feature set optimized for 5-15 minute prediction horizon with threshold 0.001-0.002 (0.1-0.2%). All features computed from historical klines and funding rates available via backfilling.

**Minimal features:**
1. `returns_5m` - 5-minute returns
2. `volatility_5m` - 5-minute volatility
3. `rsi_14` - RSI(14) indicator
4. `ema_21` - EMA(21) indicator
5. `price_ema21_ratio` - Price to EMA21 ratio
6. `volume_ratio_20` - Volume to 20-period MA ratio
7. `funding_rate` - Current funding rate (already implemented)

### Tests for Minimal Feature Set

- [X] T336 [P] [US1] Add unit tests for returns_5m feature in feature-service/tests/unit/test_price_features.py (test returns computation for 5-minute window: get close prices 5 minutes apart, compute return as (close[t] - close[t-5m]) / close[t-5m], test with insufficient data (should return None), test with zero historical price (should return None), test edge cases: empty klines, verify returns calculation matches expected formula, test with different candle intervals (1m, 3m))
- [X] T337 [P] [US1] Add unit tests for volatility_5m feature in feature-service/tests/unit/test_price_features.py (test volatility computation for 5-minute window: get klines for 5-minute period, compute returns between consecutive candles, return std(returns) as volatility, test with insufficient data (should return None), test with less than 2 candles (should return None), test edge cases: empty klines, zero prices, verify volatility calculation matches expected std of returns)
- [X] T338 [P] [US1] Add unit tests for ema_21 feature in feature-service/tests/unit/test_technical_indicators.py (test EMA(21) computation with sufficient price history (at least 21 candles), test EMA with insufficient data (should return None), verify EMA calculation matches expected exponential moving average formula: EMA = price * multiplier + EMA_prev * (1 - multiplier) where multiplier = 2 / (21 + 1), test EMA smoothing: recent prices should have more weight than older prices, test edge cases: all prices equal, single price value)
- [X] T339 [P] [US1] Add unit tests for price_ema21_ratio feature in feature-service/tests/unit/test_price_features.py (test price_ema21_ratio computation: compute as close / ema_21, test with insufficient data for EMA (should return None), test with zero EMA (should return None), verify ratio calculation: ratio > 1.0 indicates price above EMA (uptrend), ratio < 1.0 indicates price below EMA (downtrend), test edge cases: zero price, insufficient EMA data)
- [X] T340 [P] [US1] Add unit tests for volume_ratio_20 feature in feature-service/tests/unit/test_price_features.py (test volume_ratio_20 computation: compute as volume / volume_ma_20 where volume_ma_20 is 20-period moving average of volume, test with insufficient historical data for MA (should return None), test with zero average volume (should return None), verify ratio calculation: ratio > 1.0 indicates above-average volume, ratio < 1.0 indicates below-average volume, test edge cases: all volumes zero, insufficient historical data, verify MA calculation matches expected simple moving average)

### Implementation for Minimal Feature Set

- [X] T341 [P] [US1] Add returns_5m feature computation in feature-service/src/features/price_features.py (implement compute_returns_5m(rolling_windows, current_price) function: get close price 5 minutes ago from klines using get_klines_for_window("5m"), compute return as (current_price - price_5m_ago) / price_5m_ago, handle edge cases: return None if insufficient data, return None if price_5m_ago is zero, add feature "returns_5m" to compute_all_price_features(), ensure current_price is from latest kline close or provided parameter)
- [X] T342 [P] [US1] Add volatility_5m feature computation in feature-service/src/features/price_features.py (implement compute_volatility_5m(rolling_windows) function: get klines for 5-minute window using get_klines_for_window("5m"), compute returns between consecutive close prices, return std(returns) as volatility, handle edge cases: return None if less than 2 candles available, return None if insufficient data for std calculation, add feature "volatility_5m" to compute_all_price_features(), reuse existing compute_volatility function with window_seconds=300)
- [X] T343 [P] [US1] Add ema_21 feature computation in feature-service/src/features/technical_indicators.py (implement compute_ema_21(rolling_windows, candle_interval="1m") function: get close prices from klines using get_klines_for_window() with sufficient history (at least 21 candles), compute EMA(21) using exponential smoothing: EMA = price * multiplier + EMA_prev * (1 - multiplier) where multiplier = 2 / (21 + 1), handle edge cases: return None if insufficient price history (< 21 candles), return None if all prices are equal, add function compute_ema(rolling_windows, period=21, candle_interval="1m") for reusability, add "ema_21" to compute_all_technical_indicators() output)
- [X] T344 [P] [US1] Add price_ema21_ratio feature computation in feature-service/src/features/price_features.py (implement compute_price_ema21_ratio(rolling_windows, current_price) function: compute EMA(21) using technical_indicators.compute_ema_21(), compute ratio as current_price / ema_21, handle edge cases: return None if EMA is None (insufficient data), return None if EMA is zero, add feature "price_ema21_ratio" to compute_all_price_features(), ensure current_price is from latest kline close or provided parameter, normalize ratio for better model stability)
- [X] T345 [P] [US1] Add volume_ratio_20 feature computation in feature-service/src/features/price_features.py (implement compute_volume_ratio_20(rolling_windows, current_volume, candle_interval="1m") function: get historical volumes from last 20 candles using get_klines_for_window() with sufficient history, compute volume_ma_20 as simple moving average of volumes over 20 periods, compute ratio as current_volume / volume_ma_20, handle edge cases: return None if insufficient historical data (< 20 candles), return None if volume_ma_20 is zero, add feature "volume_ratio_20" to compute_all_price_features(), use current volume from latest kline or provided parameter)
- [X] T346 [P] [US1] Create Feature Registry version 1.2.0 for minimal feature set via API POST /feature-registry/versions (script created: feature-service/scripts/create_minimal_feature_set_version.py) (load current active version config via GET /feature-registry, prepare updated config dict: add feature entries for minimal set: returns_5m with input_sources: ["kline"], lookback_window: "5m", lookahead_forbidden: true, max_lookback_days: 1; volatility_5m with input_sources: ["kline"], lookback_window: "5m", lookahead_forbidden: true, max_lookback_days: 1; rsi_14 with input_sources: ["kline"], lookback_window: "14m", lookahead_forbidden: true, max_lookback_days: 1; ema_21 with input_sources: ["kline"], lookback_window: "21m", lookahead_forbidden: true, max_lookback_days: 1; price_ema21_ratio with input_sources: ["kline"], lookback_window: "21m", lookahead_forbidden: true, max_lookback_days: 1; volume_ratio_20 with input_sources: ["kline"], lookback_window: "20m", lookahead_forbidden: true, max_lookback_days: 1; ensure funding_rate entry exists with input_sources: ["funding"], lookback_window: "0s", lookahead_forbidden: true, max_lookback_days: 0; remove features requiring unavailable backfill data: vwap_3s, vwap_15s, vwap_1m, volume_3s, volume_15s, volume_1m, signed_volume_3s, signed_volume_15s, signed_volume_1m, buy_sell_volume_ratio, trade_count_3s, net_aggressor_pressure, mid_price, spread_abs, spread_rel, depth_bid_top5, depth_bid_top10, depth_ask_top5, depth_ask_top10, depth_imbalance_top5, returns_1s, returns_3s; set version to "1.2.0", add comment in config: "Minimal feature set optimized for 5-15 minute prediction horizon, threshold 0.001-0.002, all features from backfillable sources (klines + funding). Removed features requiring historical trades/orderbook/ticker data unavailable via REST API backfilling", send POST request to /feature-registry/versions with FeatureRegistryVersionCreateRequest containing version="1.2.0" and updated config dict, verify response contains created version record with file_path and metadata, verify version file created in versions/ directory, verify DB record created in feature_registry_versions table with is_active=false)
- [X] T347 [US1] Activate Feature Registry version 1.2.0 via API POST /feature-registry/versions/1.2.0/activate (script created: feature-service/scripts/create_minimal_feature_set_version.py) (send POST request to /feature-registry/versions/1.2.0/activate with FeatureRegistryVersionActivateRequest: acknowledge_breaking_changes=true (required because removed features are breaking changes), activation_reason="Minimal feature set: removed features requiring unavailable backfill data, added 7 features optimized for 5-15 minute prediction horizon", activated_by="system" or user identifier, verify response contains activated version record with compatibility_warnings and breaking_changes fields populated, verify breaking changes list includes removed features, verify hot reload executed: feature_computer updated with new config, feature_registry_loader config updated, dataset_builder._feature_registry_version updated, verify previous active version is_active set to false in DB, verify new version is_active set to true in DB, verify service continues operating with new feature set without restart, test feature computation with new minimal feature set: verify returns_5m, volatility_5m, rsi_14, ema_21, price_ema21_ratio, volume_ratio_20, funding_rate are computed correctly, verify removed features are no longer computed, document activation in logs with trace_id for audit trail)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Features are computed in real-time and available via API and message queue. Minimal feature set (7 features) optimized for 5-15 minute prediction horizon is implemented and ready for model training.

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
- [ ] T336 [P] [US2] Add timestamp continuity validation for datasets in feature-service/src/services/dataset_builder.py (implement _validate_timestamp_continuity method: analyze timestamps in features_df after computation, compute time intervals between consecutive timestamps using pd.Series.diff(), detect gaps exceeding expected interval (configurable threshold, default: 5x expected interval for trades/klines frequency), identify missing periods: calculate expected timestamps based on data source frequency (e.g., 1-minute klines, variable trade frequency), detect systematic gaps (e.g., missing overnight hours, missing weekends for spot markets), compute coverage metrics: actual_coverage_ratio = (actual_timestamps / expected_timestamps) * 100%, gap_count = number of gaps exceeding threshold, largest_gap_seconds = maximum gap duration, gap_locations = list of (gap_start, gap_end, gap_duration_seconds) tuples, add structured logging: log continuity check results with dataset_id, total_timestamps, expected_timestamps, actual_coverage_ratio, gap_count, largest_gap_seconds, gap_locations (first 10 gaps for performance), add warnings for coverage_ratio < 80% (configurable), add warnings for gaps > 1 hour (configurable), add errors for critical gaps (e.g., missing entire trading sessions) if dataset_fail_on_large_gaps enabled, return continuity_stats dict with all metrics for metadata storage, integrate into _build_dataset_task after features computation and before targets computation, handle edge cases: single timestamp (no gaps), empty timestamps, non-monotonic timestamps (already sorted, but verify), ensure efficient computation: use vectorized pandas operations, avoid loops over large datasets)
- [ ] T337 [P] [US2] Add configuration variables for timestamp continuity validation in feature-service/src/config/__init__.py (add dataset_max_timestamp_gap_ratio: float = Field(default=5.0, env="DATASET_MAX_TIMESTAMP_GAP_RATIO", description="Maximum allowed gap ratio (gap / expected_interval). Gaps exceeding this threshold will be logged as warnings. Default: 5.0 (e.g., if expected interval is 60 seconds, gaps > 300 seconds = 5 minutes will be warned)"), add dataset_expected_timestamp_interval_seconds: int = Field(default=60, env="DATASET_EXPECTED_TIMESTAMP_INTERVAL_SECONDS", description="Expected interval between timestamps in seconds for continuity checking. Default: 60 (1 minute for klines-based features). For trade-based features, use variable interval detection from actual data."), add dataset_min_coverage_ratio: float = Field(default=0.8, env="DATASET_MIN_COVERAGE_RATIO", description="Minimum required coverage ratio (actual_timestamps / expected_timestamps) for dataset. Below this ratio, warnings will be logged. Default: 0.8 (80% coverage)."), add dataset_critical_gap_threshold_seconds: int = Field(default=3600, env="DATASET_CRITICAL_GAP_THRESHOLD_SECONDS", description="Critical gap threshold in seconds. Gaps exceeding this duration will trigger errors if dataset_fail_on_large_gaps is enabled. Default: 3600 (1 hour)."), add dataset_fail_on_large_gaps: bool = Field(default=False, env="DATASET_FAIL_ON_LARGE_GAPS", description="Fail dataset build if critical gaps (exceeding DATASET_CRITICAL_GAP_THRESHOLD_SECONDS) are detected. Default: false (only logs warnings)."), add dataset_detect_variable_intervals: bool = Field(default=True, env="DATASET_DETECT_VARIABLE_INTERVALS", description="Auto-detect expected interval from actual data (for trade-based features with variable frequency). If true, uses median interval from actual timestamps. If false, uses DATASET_EXPECTED_TIMESTAMP_INTERVAL_SECONDS. Default: true."), update env.example with new variables and documentation)
- [ ] T338 [P] [US2] Add timestamp continuity statistics to dataset metadata in feature-service/src/services/dataset_builder.py (extend dataset metadata with continuity_stats: store continuity check results in dataset metadata JSONB field, include fields: actual_coverage_ratio, expected_timestamps_count, actual_timestamps_count, gap_count, largest_gap_seconds, largest_gap_start, largest_gap_end, gap_locations (first 10 gaps), coverage_by_period (optional: breakdown by hour/day if applicable), warnings_count, critical_gaps_count, enable Model Service to access continuity metrics via dataset metadata API, add continuity_stats to dataset record in MetadataStorage.update_dataset() call, ensure backward compatibility: existing datasets without continuity_stats should still work)
- [ ] T339 [P] [US2] Create unit tests for timestamp continuity validation in feature-service/tests/unit/test_dataset_timestamp_continuity.py (test continuous timestamps: no gaps, validation passes, test small gaps: gaps within threshold, warnings logged but validation passes, test large gaps: gaps exceeding threshold, warnings logged with gap details, test critical gaps: gaps exceeding critical threshold, errors if fail_on_large_gaps enabled, test variable interval detection: auto-detect median interval from actual data, test coverage ratio calculation: verify coverage_ratio = (actual / expected) * 100%, test gap detection: verify all gaps > threshold are detected and logged, test gap_locations: verify gap start/end timestamps and durations are correct, test edge cases: single timestamp, empty timestamps, non-monotonic timestamps (after sorting), test performance: verify efficient computation for large datasets (100k+ timestamps), test integration: verify continuity_stats added to dataset metadata, test configuration: verify different threshold values work correctly)
- [ ] T340 [US2] Create integration tests for timestamp continuity in dataset building workflow in feature-service/tests/integration/test_dataset_timestamp_continuity.py (test end-to-end dataset building with timestamp gaps: build dataset with intentional gaps in source data, verify continuity check detects gaps, verify warnings are logged, verify continuity_stats are stored in dataset metadata, test dataset building with continuous timestamps: verify no warnings for continuous data, test dataset building with critical gaps: verify errors if fail_on_large_gaps enabled, verify dataset marked as FAILED with appropriate error message, test continuity_stats retrieval: verify Model Service can retrieve continuity metrics from dataset metadata, test performance: verify continuity check doesn't significantly impact dataset build time)
- [ ] T341 [P] [US2] Add documentation for timestamp continuity validation in feature-service/README.md (document timestamp continuity validation process, explain expected_interval concept and variable interval detection, document coverage_ratio metric and its interpretation, document gap detection and threshold configuration, document continuity_stats metadata structure, explain when gaps occur (market closures, data collection issues, missing backfill), provide examples of continuity_stats output, explain impact of gaps on model training (temporal dependencies, feature computation, target calculation), recommend threshold values for different use cases (high-frequency trading vs daily predictions), document fail_on_large_gaps usage for strict quality requirements)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Datasets can be built from historical data with proper train/val/test splits. Timestamp continuity validation ensures datasets have adequate temporal coverage and detects missing periods that could harm model training.

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

- [ ] T334 [P] [Enhancement] Add dataset quality validation before dataset building completion in feature-service/src/services/dataset_builder.py (implement quality checks on computed features DataFrame: check for missing values per feature, check for infinite values, check for constant features (zero variance), check for feature correlation analysis (highly correlated features), check for data distribution anomalies (extreme outliers), add structured logging: log quality check results with feature names, missing value counts, constant feature names, correlation matrix for highly correlated features (threshold > 0.95), add quality metrics to dataset metadata: store quality_report in dataset record with checks summary, add configuration DATASET_QUALITY_CHECKS_ENABLED=true to enable/disable quality checks (default: true), handle quality issues: log warnings for minor issues (missing values < 5%), raise errors for critical issues (data leakage detected), add quality report to dataset metadata JSONB field for Model Service to use, document in feature-service/README.md dataset quality validation process)

- [ ] T335 [P] [Enhancement] Add target variable quality validation in feature-service/src/services/dataset_builder.py (implement target quality checks: check for class distribution imbalance (log distribution percentages), check for single-class targets (all samples same class), check for missing target values, check target range validation (for regression: check for extreme outliers), check target consistency (target should match target_config type), add structured logging: log target distribution, target statistics (min, max, mean, std for regression), class counts and percentages (for classification), add warnings for severe class imbalance (>80% in one class), add error handling: prevent dataset completion if critical target issues detected (e.g., all targets missing), add target quality metrics to dataset metadata, document in feature-service/README.md target quality validation and common issues)

**Checkpoint**: At this point, User Stories 1, 2, 3, AND 4 should all work independently. Data quality is monitored and reported. Dataset quality validation ensures high-quality datasets for model training.

---

## Phase 7: User Story 5 - Управление конфигурацией признаков через Feature Registry (Priority: P3)

**Goal**: Система предоставляет возможность управления конфигурацией признаков через Feature Registry с проверкой на data leakage и валидацией временных границ. Управление версиями реализовано через БД, где YAML файлы являются единственным источником истины для конфигурации, а БД хранит только метаданные и указатель на активную версию. Это позволяет версионировать конфигурации через Git, упрощает редактирование и исключает рассинхронизацию между БД и файлами.

**Architecture**:
- YAML файлы хранятся в `config/versions/feature_registry_v{version}.yaml`
- БД хранит только метаданные: `version`, `file_path`, `is_active`, метаданные активации
- При старте сервис загружает активную версию из БД, затем читает конфиг из файла
- API endpoints позволяют создавать, активировать версии и выполнять hot reload без рестарта

**Independent Test**: Можно протестировать независимо, загрузив конфигурацию Feature Registry и проверив, что система валидирует конфигурацию, проверяет временные границы и позволяет перезагрузить конфигурацию через hot reload без рестарта сервиса.

### Tests for User Story 5

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

**Base Tests**:
- [X] T161 [P] [US5] Create test fixtures for Feature Registry configurations (valid, invalid, with data leakage) in feature-service/tests/fixtures/feature_registry.py
- [X] T162 [P] [US5] Create unit tests for Feature Registry model in feature-service/tests/unit/test_feature_registry_model.py
- [X] T163 [P] [US5] Create unit tests for Feature Registry configuration validation in feature-service/tests/unit/test_feature_registry_validation.py (temporal boundaries, data leakage prevention, max_lookback_days)

**Version Management Tests**:
- [X] T305 [P] [US5] Create unit tests for FeatureRegistryVersionManager in feature-service/tests/unit/test_feature_registry_version_manager.py (test load_active_version: load from DB, read file, validate version match, test create_version: save file, save metadata to DB, test activate_version: update is_active flags, validate config from file, test get_version_file_path: construct correct path, test can_delete_version: check dataset usage, test migrate_legacy_to_db: migrate existing feature_registry.yaml, test file_not_found_fallback: handle missing files gracefully, test version_mismatch_warning: warn if file version != DB version)

- [X] T306 [P] [US5] Create unit tests for MetadataStorage feature registry version methods in feature-service/tests/unit/test_metadata_storage_feature_registry.py (test get_active_feature_registry_version: query with is_active=true, test create_feature_registry_version: insert with file_path and metadata, test activate_feature_registry_version: update is_active flags atomically, test list_feature_registry_versions: return all versions ordered by created_at, test get_feature_registry_version: get specific version by version string, test check_version_usage: count datasets using version, test rollback_version: update previous_version and is_active flags)

- [X] T165 [P] [US5] Create unit tests for Feature Registry automatic fallback/rollback in feature-service/tests/unit/test_feature_registry_rollback.py (automatic rollback on validation errors, migration errors, runtime errors during initial feature computation test)

- [X] T166a [P] [US5] Create unit tests for backward compatibility checking in feature-service/tests/unit/test_feature_registry_compatibility.py (detect removed features, changed names, changed logic, populate warnings/breaking_changes)

- [X] T166b [P] [US5] Create unit tests for automatic schema migration in feature-service/tests/unit/test_feature_registry_migration.py (automatic migration execution, migration script handling, rollback on migration failure)

**Integration Tests**:
- [X] T307 [P] [US5] Create integration tests for Feature Registry version management via DB in feature-service/tests/integration/test_feature_registry_version_db.py (test full lifecycle: create version, activate, load on startup, rollback, test concurrent activation prevention, test file deletion handling, test version activation with hot reload, test startup with missing active version fallback, test startup with missing file fallback)

- [X] T166 [US5] Create integration tests for Feature Registry loading and activation in feature-service/tests/integration/test_feature_registry_integration.py

**Contract Tests**:
- [X] T167 [US5] Create contract tests for GET /feature-registry endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T168 [US5] Create contract tests for POST /feature-registry/reload endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169 [US5] Create contract tests for GET /feature-registry/validate endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169a [US5] Create contract tests for GET /feature-registry/versions endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169b [US5] Create contract tests for GET /feature-registry/versions/{version} endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169c [US5] Create contract tests for POST /feature-registry/versions/{version}/activate endpoint in feature-service/tests/contract/test_feature_registry_api.py (test activation, automatic rollback on failure, breaking changes acknowledgment)
- [X] T169d [US5] Create contract tests for POST /feature-registry/rollback endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169e [US5] Create contract tests for GET /feature-registry/versions/{version}/usage endpoint in feature-service/tests/contract/test_feature_registry_api.py
- [X] T169f [US5] Create contract tests for DELETE /feature-registry/versions/{version} endpoint in feature-service/tests/contract/test_feature_registry_api.py (test deletion prevention when version is in use)
- [X] T308 [US5] Create contract tests for POST /feature-registry/versions endpoint in feature-service/tests/contract/test_feature_registry_api.py (test create version: save file, save to DB, return metadata, test create version with invalid config: validation error, test create version with duplicate version: conflict error)
- [X] T309 [US5] Create contract tests for POST /feature-registry/versions/{version}/activate with hot reload in feature-service/tests/contract/test_feature_registry_api.py (test activation updates components without restart, test activation failure triggers rollback, test breaking changes require acknowledgment, test hot reload updates FeatureComputer and DatasetBuilder)
- [X] T310 [US5] Create contract tests for POST /feature-registry/versions/{version}/sync-file endpoint in feature-service/tests/contract/test_feature_registry_api.py (test sync file to DB when file changed manually, test sync validates config before updating DB)

### Implementation for User Story 5

**Base Implementation**:
- [X] T170 [P] [US5] Create Feature Registry model in feature-service/src/models/feature_registry.py
- [X] T171 [US5] Implement Feature Registry configuration validation in feature-service/src/services/feature_registry.py (temporal boundaries, data leakage prevention, max_lookback_days)
- [X] T174 [US5] Integrate Feature Registry into feature computation (online and offline) in feature-service/src/services/feature_computer.py and feature-service/src/services/offline_engine.py
- [X] T178 [US5] Add logging for Feature Registry operations in feature-service/src/services/feature_registry.py (use structlog.dev.ConsoleRenderer() for structured logging)

**Database Migration**:
- [X] T311 [US5] Create database migration to modify feature_registry_versions table in ws-gateway/migrations/021_modify_feature_registry_versions_for_file_based_storage.sql (ALTER TABLE to: remove config JSONB column (if exists), add file_path VARCHAR(500) NOT NULL column, add index on file_path for quick lookups, add migration script to backfill file_path from existing config if migrating from old schema, migration must be reversible: restore config JSONB and remove file_path if rolling back). Note: If T024 already has config JSONB, this migration removes it and adds file_path. Keep all other fields (is_active, validated_at, validation_errors, loaded_at, created_at, created_by, activated_by, rollback_from, previous_version, schema_version, migration_script, compatibility_warnings, breaking_changes, activation_reason) unchanged.

- [X] T311a [US5] Apply database migration: `docker compose exec postgres psql -U ytrader -d ytrader -f /tmp/021_modify_feature_registry_versions_for_file_based_storage.sql`

**Version Management Implementation**:
- [X] T312 [P] [US5] Create FeatureRegistryVersionManager service in feature-service/src/services/feature_registry_version_manager.py (implement load_active_version: query DB for active version, load file from file_path, validate version match between file and DB, return config dict, implement create_version: save config to file in versions/ directory, create DB record with file_path and metadata, validate config before saving, implement activate_version: validate config from file, atomically update is_active flags in DB (deactivate old, activate new), update loaded_at timestamp, return activated version metadata, implement get_version_file_path: construct path from versions_dir and version string, implement can_delete_version: check if version is used in any datasets via MetadataStorage, return boolean, implement sync_db_to_files: iterate all DB versions, ensure files exist on disk, create missing files from DB config if config JSONB still exists (migration helper), implement sync_files_to_db: scan versions/ directory, create DB records for files not in DB, extract version from filename, implement migrate_legacy_to_db: load existing feature_registry.yaml, extract version, save to versions/ directory, create DB record with is_active=true if no active version exists, add comprehensive error handling and structured logging for all operations)

- [X] T313 [P] [US5] Extend MetadataStorage with feature registry version methods in feature-service/src/storage/metadata_storage.py (implement get_active_feature_registry_version: SELECT * FROM feature_registry_versions WHERE is_active = true LIMIT 1, return Dict or None, implement create_feature_registry_version: INSERT with version, file_path, is_active, created_at, created_by, validated_at (if validated), return created record, implement activate_feature_registry_version: BEGIN transaction, UPDATE feature_registry_versions SET is_active = false WHERE is_active = true, UPDATE feature_registry_versions SET is_active = true, loaded_at = NOW(), activated_by = $activated_by, activation_reason = $reason, previous_version = current_active_version WHERE version = $version, COMMIT transaction, return activated record, implement list_feature_registry_versions: SELECT * FROM feature_registry_versions ORDER BY created_at DESC, return List[Dict], implement get_feature_registry_version: SELECT * FROM feature_registry_versions WHERE version = $version, return Dict or None, implement check_version_usage: SELECT COUNT(*) FROM datasets WHERE feature_registry_version = $version, return integer count, add proper error handling and structured logging)

- [X] T314 [US5] Update FeatureRegistryLoader to support database-driven mode in feature-service/src/services/feature_registry.py (add use_db: bool parameter to __init__, add version_manager: Optional[FeatureRegistryVersionManager] parameter, implement load_active_from_db: call version_manager.load_active_version() if use_db=True, fallback to file_mode if DB load fails, implement set_config: allow setting config manually (for hot reload), keep existing load() method for file_mode (backward compatibility), add validate_version_match: check file version matches DB version (warning if mismatch), update get_required_data_types and get_data_type_mapping to work with both modes)

- [X] T315 [US5] Update startup process to load active version from database in feature-service/src/main.py (modify startup function: initialize MetadataStorage, initialize FeatureRegistryVersionManager with versions_dir from config, try to load_active_version() from DB, if success: use db_mode, if failure: fallback to legacy file_mode with automatic migration, update FeatureRegistryLoader initialization to use db_mode, pass version_manager to FeatureRegistryLoader, update DatasetBuilder to use version from loaded config, add comprehensive error handling and logging for startup sequence, ensure backward compatibility with existing deployments)

- [X] T214 [US5] Implement backward compatibility checking in feature-service/src/services/feature_registry_version_manager.py (FR-065: detect breaking changes - removed features, changed names, changed logic, populate compatibility_warnings and breaking_changes fields)

- [X] T215 [US5] Implement automatic schema migration in feature-service/src/services/feature_registry_version_manager.py (FR-064: automatic migration when activating new version with changed feature definitions, apply migration_script if provided)

- [ ] T216 [US5] Implement version usage tracking in feature-service/src/services/feature_registry_version_manager.py (FR-067: track version usage in datasets and feature computations, prevent deletion of in-use versions)

- [ ] T217 [US5] Implement audit trail for version changes in feature-service/src/services/feature_registry_version_manager.py (FR-068: track who activated/rolled back version, when, and reason - populate created_by, activated_by, activation_reason fields)

**Configuration**:
- [X] T316 [P] [US5] Add configuration variables for version management in feature-service/src/config/__init__.py (add feature_registry_versions_dir: str = Field(default="/app/config/versions", env="FEATURE_REGISTRY_VERSIONS_DIR"), add feature_registry_use_db: bool = Field(default=True, env="FEATURE_REGISTRY_USE_DB"), add feature_registry_auto_migrate: bool = Field(default=True, env="FEATURE_REGISTRY_AUTO_MIGRATE"), update env.example with new variables)

**API Endpoints**:
- [X] T175 [US5] Implement GET /feature-registry endpoint in feature-service/src/api/feature_registry.py
- [X] T176 [US5] Implement POST /feature-registry/reload endpoint in feature-service/src/api/feature_registry.py
- [X] T177 [US5] Implement GET /feature-registry/validate endpoint in feature-service/src/api/feature_registry.py
- [X] T208 [US5] Implement GET /feature-registry/versions endpoint in feature-service/src/api/feature_registry.py (FR-058: list all versions with metadata)
- [X] T209 [US5] Implement GET /feature-registry/versions/{version} endpoint in feature-service/src/api/feature_registry.py (FR-059: get specific version)
- [X] T210 [US5] Implement POST /feature-registry/versions/{version}/activate endpoint in feature-service/src/api/feature_registry.py (FR-060: activate version with automatic rollback on failure, acknowledge breaking changes parameter)
- [X] T211 [US5] Implement POST /feature-registry/rollback endpoint in feature-service/src/api/feature_registry.py (FR-061: automatic rollback to previous version)
- [X] T212 [US5] Implement GET /feature-registry/versions/{version}/usage endpoint in feature-service/src/api/feature_registry.py (FR-062: check version usage)
- [X] T213 [US5] Implement DELETE /feature-registry/versions/{version} endpoint in feature-service/src/api/feature_registry.py (FR-063: delete version only if not in use)
- [X] T317 [US5] Implement POST /feature-registry/versions endpoint in feature-service/src/api/feature_registry.py (accept FeatureRegistryVersionCreateRequest with config dict and version string, call version_manager.create_version(), validate config before saving, return created version metadata with file_path, handle validation errors and duplicate version errors, add authentication and authorization checks)
- [X] T318 [US5] Update POST /feature-registry/versions/{version}/activate to support hot reload in feature-service/src/api/feature_registry.py (modify existing activate endpoint: call version_manager.activate_version(), after successful DB update: reload registry in memory without restart, update global feature_computer with new version, update dataset_builder._feature_registry_version, update feature_registry_loader config, return activation response with hot_reload: true, handle rollback on activation failure, add comprehensive error handling and logging)
- [X] T319 [US5] Implement POST /feature-registry/versions/{version}/sync-file endpoint in feature-service/src/api/feature_registry.py (load config from file on disk, validate config, update DB metadata (validated_at, validation_errors if invalid), return sync result with validation status, use case: file edited manually, need to sync metadata in DB, handle file not found error)

**Hot Reload**:
- [X] T320 [US5] Implement hot reload mechanism for Feature Registry in feature-service/src/services/feature_registry_hot_reload.py (create reload_registry_in_memory function: accept new config dict, recreate FeatureComputer with new version, update DatasetBuilder version (for new datasets only), update FeatureRegistryLoader config, update global variables atomically, add lock to prevent concurrent reloads, add comprehensive error handling and rollback on reload failure, integrate into activate endpoint)

**Migration and Infrastructure**:
- [X] T321 [US5] Implement automatic migration for legacy feature_registry.yaml in feature-service/src/main.py startup (automatic migration on startup if FEATURE_REGISTRY_AUTO_MIGRATE=true and no active version exists in DB, load existing feature_registry.yaml from config.feature_registry_path, extract version from config, create versions/ directory if not exists, save file as versions/feature_registry_{version}.yaml, create DB record with is_active=true, validate config before migration, add logging and error handling)

- [X] T322 [US5] Update docker-compose.yml to mount versions directory as volume in docker-compose.yml (add volume mount for feature-service: ./feature-service/config/versions:/app/config/versions, ensure directory exists on host, document in README.md)

- [X] T323 [US5] Update documentation in feature-service/README.md (document version management via database, explain files as source of truth approach, document API endpoints for version management, document environment variables (FEATURE_REGISTRY_VERSIONS_DIR, FEATURE_REGISTRY_USE_DB, FEATURE_REGISTRY_AUTO_MIGRATE), document hot reload capabilities, document migration from legacy mode, add examples of creating and activating versions via API)

**Checkpoint**: At this point, all user stories should be independently functional. Feature Registry manages feature configuration with validation. Version management is handled via database with files as source of truth, enabling Git versioning and hot reload without service restart.

---

## Phase 7.5: Dataset Building Performance Optimization (Critical) (Priority: P1) 🚀

**Goal**: Ускорить построение датасетов с ~2 часов для 60 дней до ~5-15 минут через критичные software оптимизации. Текущая проблема: последовательное вычисление признаков и полная реконструкция orderbook/rolling windows для каждого timestamp приводит к чрезмерному времени обработки.

**Expected Performance Improvement**: 10-20x ускорение (с ~2 часов до ~5-15 минут для 60 дней данных) после реализации всех критичных оптимизаций.

**Independent Test**: Можно протестировать независимо, построив датасет на тестовых данных и измерив время выполнения до и после оптимизаций.

### Tests for Dataset Building Performance Optimization

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T324 [P] [US2] Create unit tests for incremental orderbook reconstruction in feature-service/tests/unit/test_offline_engine_incremental_orderbook.py (test incremental orderbook update: start with snapshot, apply only new deltas between timestamps, test snapshot refresh logic: reload snapshot periodically (e.g., every hour), test orderbook state persistence between timestamps, test correctness: incremental reconstruction matches full reconstruction, test edge cases: no deltas between timestamps, snapshot missing, deltas missing, test performance: verify that incremental update is faster than full reconstruction)

- [X] T325 [P] [US2] Create unit tests for incremental rolling windows update in feature-service/tests/unit/test_offline_engine_incremental_rolling_windows.py (test incremental rolling windows update: reuse RollingWindows object, add only new trades/klines, test automatic trimming: old data removed by RollingWindows.trim_old_data(), test correctness: incremental reconstruction matches full reconstruction for all window sizes (1s, 3s, 15s, 1m), test state persistence between timestamps, test edge cases: no new data between timestamps, empty windows, test performance: verify that incremental update is faster than full reconstruction)

- [ ] T326 [P] [US2] Create unit tests for parallel feature computation in feature-service/tests/unit/test_dataset_builder_parallel.py (test parallel feature computation using asyncio.gather (primary method): compute features for multiple timestamps concurrently with semaphore limiting concurrency, verify asyncio.gather works correctly with async feature computation, test result ordering: ensure results match sequential processing order, test error handling: failed computations don't break batch (return_exceptions=True), test worker pool size configuration: respect max_workers parameter and semaphore limits, test memory usage: verify that asyncio.gather doesn't significantly increase memory usage, test performance: verify that asyncio.gather is faster than sequential for large datasets (target: 3-4x with 4 workers), test multiprocessing mode (optional): test ProcessPoolExecutor when worker_pool_mode="multiprocessing", verify multiprocessing correctly handles data serialization, test fallback: verify graceful degradation if parallel processing fails)

- [X] T327 [US2] Create integration tests for dataset building performance improvements in feature-service/tests/integration/test_dataset_building_performance.py (test full dataset build with incremental orderbook: build 7-day dataset, measure time, verify correctness, test full dataset build with incremental rolling windows: build 7-day dataset, measure time, verify correctness, test full dataset build with parallel computation: build 7-day dataset, measure time, verify correctness, test combined optimizations: all three optimizations together, measure total speedup (target: 10-20x for 60 days), verify feature correctness: compare features computed with optimized vs original implementation, test memory efficiency: verify that optimizations don't significantly increase memory usage)

### Implementation for Dataset Building Performance Optimization

- [X] T328 [P] [US2] Implement incremental orderbook reconstruction in feature-service/src/services/offline_engine.py (modify _reconstruct_orderbook_state to support incremental updates: add optional previous_orderbook_state and last_timestamp parameters, if previous_orderbook_state provided: load snapshot only if needed (e.g., every hour or when missing), apply only deltas between last_timestamp and current timestamp, reuse existing OrderbookManager instance, if previous_orderbook_state not provided: use existing full reconstruction logic (backward compatibility), add snapshot refresh logic: reload snapshot periodically to prevent delta accumulation errors, add method get_orderbook_snapshot_refresh_interval: return recommended refresh interval (e.g., 3600 seconds for 1 hour), add comprehensive logging for incremental vs full reconstruction paths, ensure feature correctness: incremental reconstruction produces identical results to full reconstruction)

- [X] T329 [P] [US2] Implement incremental rolling windows update in feature-service/src/services/offline_engine.py (modify _reconstruct_rolling_windows to support incremental updates: add optional previous_rolling_windows and last_timestamp parameters, if previous_rolling_windows provided: reuse existing RollingWindows object, add only new trades/klines between last_timestamp and current timestamp using add_trade() and add_kline() methods, call trim_old_data() to remove outdated data, if previous_rolling_windows not provided: use existing full reconstruction logic (backward compatibility), ensure RollingWindows methods handle incremental updates correctly (already implemented, but verify), add comprehensive logging for incremental vs full reconstruction paths, ensure feature correctness: incremental reconstruction produces identical results to full reconstruction for all window sizes)

- [X] T330 [P] [US2] Update DatasetBuilder to use incremental orderbook and rolling windows in feature-service/src/services/dataset_builder.py (modify _compute_features_batch: maintain orderbook_state and rolling_windows objects between timestamps, initialize orderbook_state and rolling_windows as None before loop, for each timestamp: if orderbook_state/rolling_windows exist: pass them to compute_features_at_timestamp for incremental update, if orderbook_state/rolling_windows don't exist or snapshot refresh needed: create new objects (full reconstruction), after feature computation: update orderbook_state and rolling_windows references, add periodic snapshot refresh: check if snapshot refresh interval elapsed, reload snapshot if needed, add progress logging: log when incremental vs full reconstruction is used, add configuration parameter dataset_builder_orderbook_snapshot_refresh_interval: default 3600 seconds (1 hour), ensure backward compatibility: if incremental update fails, fallback to full reconstruction) **NOTE: snapshot_refresh_interval is currently hardcoded to 3600 seconds. See T333 for moving to configuration.**

- [X] T330a [P] [US2] Optimize klines processing in _reconstruct_rolling_windows to avoid redundant operations in feature-service/src/services/offline_engine.py (CRITICAL PERFORMANCE ISSUE: current implementation adds ALL klines to DataFrame for each timestamp via loop + concat + sort_values, causing O(n²) complexity where n grows with each timestamp processed, PROPOSED SOLUTION: optimize _reconstruct_rolling_windows to efficiently handle klines: pre-filter klines once per timestamp batch instead of iterating all klines for each timestamp, use vectorized pandas operations instead of row-by-row loop, cache klines DataFrame with timestamp index for O(1) lookup, avoid repeated concat and sort_values operations by maintaining sorted DataFrame and appending only new rows, use DataFrame.loc with boolean indexing instead of iterating rows, implement incremental kline addition: track last_processed_kline_index and only process new klines since last timestamp, add benchmark test: measure time before/after optimization (target: 5-10x speedup for datasets with 1000+ timestamps), add logging: track klines processing time per timestamp vs batch processing time, ensure correctness: verify all klines are correctly added and sorted, maintain backward compatibility: same output format and behavior, document performance impact: explain why current implementation is slow (O(n²) complexity) and how optimization reduces it to O(n log n) or better)

- [ ] T330b [P] [US2] Optimize incremental rolling windows updates to use vectorized pandas operations in feature-service/src/services/offline_engine.py (CURRENT ISSUE: _reconstruct_rolling_windows_incremental uses iterrows() loop and calls add_trade()/add_kline() for each new trade/kline, which internally calls pd.concat() for each addition, causing performance bottleneck when multiple trades/klines need to be added between timestamps, PROPOSED SOLUTION: optimize _reconstruct_rolling_windows_incremental to use vectorized pandas operations: batch process all new trades/klines at once instead of iterrows() loop, create DataFrame for all new trades/klines using vectorized operations (pd.DataFrame constructor with dict comprehension or direct column assignment), use single pd.concat() call to append all new data to existing windows instead of multiple concat calls, optimize RollingWindows.add_trade() and add_kline() methods: accept DataFrame or list of trades/klines for batch addition, or create separate batch_add_trades() and batch_add_klines() methods, use vectorized boolean indexing for filtering by window sizes instead of per-trade/kline filtering, ensure backward compatibility: existing add_trade()/add_kline() methods should still work for single additions, add comprehensive logging for batch vs single addition paths, add benchmark test: measure time before/after optimization for dataset with many trades/klines between timestamps, verify correctness: batch addition produces identical results to sequential addition)

- [ ] T331 [P] [US2] Implement parallel feature computation in feature-service/src/services/dataset_builder.py (PRIMARY METHOD - asyncio.gather: modify _compute_features_batch to use asyncio.gather with semaphore for concurrency control, create async wrapper function that respects semaphore limit, use asyncio.gather(*tasks, return_exceptions=True) to handle all timestamps concurrently, preserve result ordering by mapping results back to original timestamps, add configuration parameter dataset_builder_parallel_workers: int (default: 4, maximum: 8 for safety), limit max_workers to prevent excessive concurrency: max_workers = min(config.parallel_workers, 8), use asyncio.Semaphore(max_workers) to control concurrent async tasks, OPTIONAL METHOD - multiprocessing: add worker_pool_mode configuration: "asyncio" (default, recommended for Docker/VDS) or "multiprocessing" (optional, requires proper CPU limits in Docker), implement ProcessPoolExecutor path only when worker_pool_mode="multiprocessing", use get_optimal_workers() helper: read Docker cgroup limits from /sys/fs/cgroup/cpu/, fallback to config value if cgroup unavailable, chunk timestamps into batches for multiprocessing, handle data serialization correctly (pickle-compatible objects), add error handling: failed computations return None via return_exceptions=True, collect successful results, log failures with context (timestamp, error), add progress tracking: update progress thread-safely for asyncio.gather (simple counter with lock), for multiprocessing: use shared memory or queue for progress updates, add memory monitoring: log peak memory usage during parallel processing, add feature correctness validation: compare sample results from parallel vs sequential (optional, for debugging), add comprehensive logging: log workers used, actual concurrency achieved, speedup metrics, processing time per batch, ensure backward compatibility: if parallel processing disabled, use sequential processing, add Docker/VDS compatibility notes in code comments: asyncio.gather works reliably in Docker with any CPU limits, multiprocessing requires proper CPU limits configuration (minimum 4 cores recommended))

- [ ] T332 [P] [US2] Add performance metrics and monitoring to DatasetBuilder in feature-service/src/services/dataset_builder.py (add timing measurements: track time spent on data loading, feature computation (per timestamp and total), target computation, dataset splitting, file writing, add memory measurements: track peak memory usage during dataset build, track memory per phase (data loading, feature computation, etc.), add performance logging: log timing metrics after dataset build completion, log speedup metrics when optimizations enabled, log memory usage statistics, add metrics to dataset metadata: store build_duration_seconds, peak_memory_mb, optimization_mode (incremental/parallel/full) in dataset record, add configuration parameter dataset_builder_enable_performance_metrics: bool, default True, add structured logging with performance fields: duration_ms, memory_mb, records_per_second, optimization_mode)

- [ ] T333 [P] [US2] Add configuration variables for performance optimizations in feature-service/src/config/__init__.py (add dataset_builder_parallel_workers: int = Field(default=4, env="DATASET_BUILDER_PARALLEL_WORKERS", description="Number of parallel workers for feature computation using asyncio.gather (default: 4, maximum recommended: 8, works reliably in Docker/VDS with any CPU limits)"), add dataset_builder_worker_pool_mode: str = Field(default="asyncio", env="DATASET_BUILDER_WORKER_POOL_MODE", description="Worker pool mode: 'asyncio' (default, recommended for Docker/VDS - uses asyncio.gather with semaphore) or 'multiprocessing' (optional - requires proper CPU limits in Docker, minimum 4 cores recommended)"), add dataset_builder_orderbook_snapshot_refresh_interval: int = Field(default=3600, env="DATASET_BUILDER_ORDERBOOK_SNAPSHOT_REFRESH_INTERVAL", description="Orderbook snapshot refresh interval in seconds (default: 3600 = 1 hour)"), add dataset_builder_enable_incremental_updates: bool = Field(default=True, env="DATASET_BUILDER_ENABLE_INCREMENTAL_UPDATES", description="Enable incremental orderbook and rolling windows updates (default: true)"), add dataset_builder_enable_parallel_computation: bool = Field(default=True, env="DATASET_BUILDER_ENABLE_PARALLEL_COMPUTATION", description="Enable parallel feature computation using asyncio.gather (default: true, works reliably in Docker)"), add dataset_builder_enable_performance_metrics: bool = Field(default=True, env="DATASET_BUILDER_ENABLE_PERFORMANCE_METRICS", description="Enable performance metrics collection (default: true)"), update env.example with new variables and add comments: recommended values for VDS, Docker CPU limits requirements, asyncio.gather vs multiprocessing trade-offs) **NOTE: In T330, snapshot_refresh_interval is currently hardcoded to 3600 seconds in dataset_builder.py. When implementing T333, replace the hardcoded value with config.dataset_builder_orderbook_snapshot_refresh_interval.**

**Checkpoint**: Dataset building performance should be significantly improved (target: 10-20x speedup). For 60 days of data, build time should decrease from ~2 hours to ~5-15 minutes. All optimizations maintain feature correctness (incremental/parallel computation produces identical results to sequential computation).

**Implementation Notes for Docker/VDS**:
- **asyncio.gather (recommended)**: Works reliably in Docker containers with any CPU limits. Uses semaphore for concurrency control, suitable for mixed I/O+CPU workloads (feature computation with Parquet I/O and pandas operations). Default mode, no special Docker configuration required.
- **multiprocessing (optional)**: Requires proper CPU limits in docker-compose.yml (minimum 4 cores recommended). More complex setup, higher memory usage, but can provide better performance for purely CPU-bound operations. Use only if asyncio.gather performance is insufficient.
- **Docker CPU limits**: Current configuration has `cpus: '0.5'` which is insufficient. Recommended: increase to `cpus: '4'` or `cpus: '8'` for optimal performance with asyncio.gather (works with any limit, but more CPUs = better parallelism).
- **Memory requirements**: With incremental updates and parallel processing, minimum 16GB RAM recommended for 60 days of data. Monitor memory usage during initial dataset builds.

---

## Phase 7.6: Dataset Building Caching Strategy (Priority: P1) 🚀

**Goal**: Ускорить повторные сборки датасетов через кэширование исторических данных и вычисленных фичей. Ожидаемое ускорение: 10-30x для повторных запусков с теми же параметрами (символ, период, версия Feature Registry). Устранить повторное чтение Parquet файлов и пересчёт фичей при повторных сборках для тех же периодов.

**Expected Performance Improvement**: 10-30x ускорение для повторных сборок (с ~5-15 минут до ~30 секунд - 2 минуты для 60 дней данных) при полном попадании в кэш. Частичное кэширование (например, только исторические данные) даёт 5-10x ускорение.

**Independent Test**: Задействовать Redis в тестах и убедиться что минимальный датасет собирается корректно.

**Cache Strategy Overview**:
1. **Кэширование исторических данных** (приоритет 1): Кэшировать DataFrame'ы исторических данных в Redis (приоритетный вариант) или в памяти (fallback) для повторного использования при повторных сборках тех же периодов.
2. **Кэширование вычисленных фичей** (приоритет 2): Кэшировать FeatureVector для каждого timestamp в Redis (приоритетный вариант) или в памяти (fallback) при повторных сборках с теми же параметрами.
3. **Redis как приоритетный вариант**: Redis используется как основной кэш (default), обеспечивает persistence и распределённое кэширование между перезапусками контейнера feature-service. Кэш в памяти используется только как fallback при недоступности Redis.

### Tests for Dataset Building Caching Strategy

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T348 [P] [US2] Create unit tests for historical data caching in feature-service/tests/unit/test_dataset_builder_cache.py (test cache hit: same symbol, start_date, end_date, feature_registry_version should return cached data, test cache miss: different parameters should read from Parquet, test Redis as primary cache: verify Redis is used when available, test fallback to memory cache: verify automatic fallback when Redis unavailable, test Redis reconnection: verify automatic switch back to Redis when connection restored, test cache invalidation on Feature Registry version change: verify cache is invalidated when registry version differs (works for both Redis and memory), test cache invalidation on data file modification: verify cache is invalidated when Parquet file mtime changes (works for both Redis and memory), test cache invalidation on period change: verify cache is invalidated when start_date or end_date changes (works for both Redis and memory), test cache TTL: verify cache entries expire after TTL (works for both Redis and memory), test memory fallback cache size limits: verify LRU eviction when fallback cache exceeds max_size, test Redis memory limits: verify Redis maxmemory-policy works correctly, test cache key generation: verify cache keys include all relevant parameters (symbol, start_date, end_date, feature_registry_version, data_hash), test cache statistics: verify hit_rate, miss_count, eviction_count tracking (works for both Redis and memory), test concurrent cache access: verify thread-safe cache operations for parallel dataset builds (works for both Redis and memory), test cache serialization: verify DataFrame serialization/deserialization works correctly (pickle or Parquet format), test partial cache hit: verify cache can be used partially when only subset of requested period is cached)

- [X] T349 [P] [US2] Create unit tests for computed features caching in feature-service/tests/unit/test_dataset_builder_features_cache.py (test cache hit: same symbol, timestamp, feature_registry_version, data_hash should return cached FeatureVector, test cache miss: different parameters should compute features, test Redis as primary cache: verify Redis is used when available for features caching, test fallback to memory cache: verify automatic fallback when Redis unavailable, test Redis reconnection: verify automatic switch back to Redis when connection restored, test cache invalidation on Feature Registry version change: verify cache is invalidated when registry version differs (works for both Redis and memory), test cache invalidation on data modification: verify cache is invalidated when data_hash changes (indicates historical data was updated, works for both Redis and memory), test cache key generation: verify cache keys include symbol, timestamp, feature_registry_version, data_hash, test batch cache retrieval: verify multiple FeatureVectors can be retrieved from cache efficiently (works for both Redis and memory, Redis mget optimization), test cache TTL: verify cache entries expire after TTL (works for both Redis and memory), test memory fallback cache size limits: verify LRU eviction when fallback cache exceeds max_size, test Redis memory limits: verify Redis maxmemory-policy works correctly, test cache statistics: verify hit_rate, miss_count, eviction_count tracking (works for both Redis and memory), test concurrent cache access: verify thread-safe cache operations (works for both Redis and memory), test cache serialization: verify FeatureVector serialization/deserialization works correctly (JSON or pickle), test partial cache hit: verify that partial cache hits (e.g., 80% of timestamps cached) still provide performance benefit, test cache compression: verify that cached data can be compressed to reduce memory/Redis usage (optional))

- [X] T350 [P] [US2] Create unit tests for cache invalidation logic in feature-service/tests/unit/test_cache_invalidation.py (test Feature Registry version change invalidation: verify all cache entries are invalidated when registry version changes, test Parquet file modification detection: verify cache detects file mtime changes and invalidates relevant entries, test data hash computation: verify data_hash is computed correctly from historical data DataFrames (MD5 or SHA256 hash of sorted DataFrame content), test cache key tagging: verify cache entries are tagged with feature_registry_version for efficient bulk invalidation, test partial invalidation: verify only affected cache entries are invalidated (e.g., only entries for specific symbol or date range), test invalidation propagation: verify cache invalidation works correctly with Redis pub/sub (if Redis cluster mode enabled), test invalidation on backfill: verify cache is invalidated when new historical data is backfilled, test invalidation on dataset build completion: verify cache is not invalidated unnecessarily after dataset build (only invalidate on data change), test manual cache invalidation API: verify manual cache invalidation endpoint works correctly, test cache warming: verify cache can be pre-populated (warmed) for frequently used periods)

- [X] T351 [US2] Create integration tests for dataset building with caching in feature-service/tests/integration/test_dataset_building_with_cache.py (test full dataset build with Redis cache enabled: build dataset with Redis available, measure time, verify cache is populated in Redis, test repeat dataset build with Redis cache: build same dataset again, measure time, verify 10-30x speedup, verify features are identical, verify cache reads from Redis, test Redis cache persistence: build dataset, restart feature-service container (keep Redis running), rebuild dataset, verify Redis cache persists and is reused, verify 10-30x speedup on second build, test fallback to memory cache: start dataset build with Redis unavailable, verify automatic fallback to memory cache, verify build completes successfully, verify cache statistics indicate memory cache usage, test Redis reconnection: build dataset with Redis unavailable (uses memory), start Redis, build dataset again, verify automatic switch to Redis, verify cache is used from Redis, test cache invalidation scenario: build dataset, invalidate cache, rebuild dataset, verify cache is repopulated, test partial cache hit: build dataset for period A, build dataset for overlapping period B (e.g., period B includes period A), verify cache is reused for period A portion (works for both Redis and memory), test cache with different Feature Registry versions: build dataset with v1.0.0, switch to v1.1.0, build dataset, verify cache from v1.0.0 is not used (works for both Redis and memory), test cache with data updates: build dataset, update Parquet files (simulate backfill), rebuild dataset, verify cache is invalidated and new data is used (works for both Redis and memory), test Redis cache memory usage: verify Redis cache doesn't exceed maxmemory limits, verify LRU eviction works in Redis, test memory fallback cache usage: verify fallback cache doesn't exceed configured memory limits, test cache performance metrics: verify cache hit_rate, miss_count, eviction_count are tracked and logged for both Redis and memory cache)

### Implementation for Dataset Building Caching Strategy

- [X] T352 [P] [US2] Create cache service abstraction in feature-service/src/services/cache_service.py (create CacheService abstract base class with methods: get(key: str) -> Optional[Any], set(key: str, value: Any, ttl: Optional[int] = None) -> bool, delete(key: str) -> bool, exists(key: str) -> bool, clear(pattern: Optional[str] = None) -> int, get_statistics() -> Dict[str, Any], create RedisCacheService implementation (PRIMARY/PRIORITY): uses redis.asyncio (redis[hiredis] dependency), connection pooling, TTL support, statistics tracking, persistence across container restarts, distributed cache support, create InMemoryCacheService implementation (FALLBACK): uses dict with LRU eviction, thread-safe (asyncio.Lock), TTL support via background task, size limits with LRU eviction, statistics tracking (hit_count, miss_count, eviction_count, current_size_bytes, max_size_bytes), used only when Redis is unavailable, create CacheServiceFactory with Redis-first strategy: attempt to connect to Redis on initialization, if Redis available: return RedisCacheService (default/primary), if Redis unavailable: log warning and return InMemoryCacheService (fallback), implement automatic Redis reconnection: periodically attempt to reconnect to Redis, switch back to RedisCacheService when connection restored, handle Redis connection errors gracefully: catch connection errors, automatically fallback to InMemoryCacheService, log fallback event with context, add configuration parameters: redis_host (default: "redis"), redis_port (default: 6379), redis_db (default: 0), redis_password, redis_max_connections, redis_socket_timeout, redis_socket_connect_timeout, cache_ttl_seconds, cache_max_size_mb (for memory fallback cache), cache_max_entries (for memory fallback cache), cache_redis_enabled (default: True, enable Redis as primary cache), add comprehensive logging for cache operations: hit/miss, invalidation, eviction, errors, Redis connection status, fallback events)

- [X] T353 [P] [US2] Implement historical data caching in feature-service/src/services/dataset_builder.py (modify _read_historical_data to use cache: compute cache key from symbol, start_date, end_date, feature_registry_version, check cache for existing data, if cache hit: verify cache validity (check data_hash or file mtime), return cached data, if cache miss: read from Parquet, compute data_hash (MD5 or SHA256 of sorted DataFrame content), store in cache with TTL, return data, implement cache key generation: format: "historical_data:{symbol}:{start_date}:{end_date}:{registry_version}:{data_hash}", include data_hash in key to detect data changes, implement cache validity checking: compute current data_hash or check Parquet file mtime, compare with cached data_hash or cached mtime, invalidate cache if data changed, implement cache invalidation: detect Feature Registry version changes (compare current version with cached version), detect Parquet file modifications (check mtime or compute new data_hash), invalidate affected cache entries (use pattern matching: "historical_data:{symbol}:*"), implement partial cache hit handling: if requested period overlaps with cached period, reuse cached portion, read only missing portion from Parquet, merge cached and newly read data, update cache with merged data, add cache statistics logging: log cache hit/miss rates, cache size, eviction count, add configuration parameters: dataset_builder_cache_historical_data_enabled: bool (default: True), dataset_builder_cache_historical_data_ttl_seconds: int (default: 86400 = 24 hours), dataset_builder_cache_historical_data_max_size_mb: int (default: 1024 = 1GB for memory cache))

- [X] T354 [P] [US2] Implement computed features caching in feature-service/src/services/dataset_builder.py (modify _compute_features_batch to use cache: for each timestamp, compute cache key from symbol, timestamp, feature_registry_version, data_hash, check cache for existing FeatureVector, if cache hit: verify cache validity (check data_hash matches current), add to features_list, if cache miss: compute features via offline_engine, store in cache with TTL, add to features_list, implement cache key generation: format: "features:{symbol}:{timestamp}:{registry_version}:{data_hash}", include data_hash to detect data changes, implement batch cache retrieval: retrieve multiple FeatureVectors from cache in single operation (mget for Redis, dict lookup for memory), implement batch cache storage: store multiple FeatureVectors in cache efficiently (mset for Redis, dict update for memory), implement cache invalidation: detect Feature Registry version changes, detect data changes (data_hash mismatch), invalidate affected cache entries (use pattern matching: "features:{symbol}:*" or "features:{symbol}:{date_range}"), implement partial cache hit optimization: if 80%+ of timestamps are cached, use cached values, compute only missing 20%, significantly speed up dataset build, add cache statistics logging: log cache hit/miss rates per dataset build, partial cache hit percentage, add configuration parameters: dataset_builder_cache_features_enabled: bool (default: True), dataset_builder_cache_features_ttl_seconds: int (default: 604800 = 7 days, features are more stable than raw data), dataset_builder_cache_features_max_size_mb: int (default: 2048 = 2GB for memory cache, features are smaller than raw data but more numerous))

- [X] T355 [P] [US2] Implement cache invalidation logic in feature-service/src/services/cache_invalidation.py (create CacheInvalidationService class: manages cache invalidation triggers and logic, implement Feature Registry version change detection: monitor Feature Registry version changes, invalidate all cache entries with old registry version (use pattern: "*:{old_version}:*"), implement Parquet file modification detection: check file mtime before and after dataset build, if mtime changed: invalidate cache entries for affected symbol and date range, implement data hash computation: compute hash from DataFrame content (MD5 or SHA256), compare with cached hash, invalidate if different, implement cache key tagging: tag cache entries with metadata (feature_registry_version, symbol, date_range) for efficient bulk invalidation, implement partial invalidation: invalidate only affected entries (e.g., only entries for specific symbol when that symbol's data changes), implement invalidation on backfill completion: when backfill service completes, invalidate cache for backfilled period and symbol, implement manual invalidation API: provide endpoint to manually invalidate cache (all, by pattern, by symbol, by date range), implement cache warming: pre-populate cache for frequently used periods (optional, for production optimization), add comprehensive logging: log all invalidation events with context (reason, affected entries count, pattern used), add Redis pub/sub support for distributed invalidation: if Redis cluster mode, publish invalidation events to Redis channel, other instances subscribe and invalidate local cache (optional, for multi-instance deployments))

- [X] T356 [P] [US2] Add Redis support and configuration as PRIMARY cache in feature-service/ (add redis[hiredis] dependency to requirements.txt: redis[hiredis]>=5.0.0 (hiredis for performance), REQUIRED dependency as Redis is primary cache, add Redis configuration to config.py: redis_host: str = Field(default="redis", env="REDIS_HOST", description="Redis hostname (default: 'redis' for Docker service name)"), redis_port: int = Field(default=6379, env="REDIS_PORT"), redis_db: int = Field(default=0, env="REDIS_DB"), redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD"), redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS"), redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT"), redis_socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT"), cache_redis_enabled: bool = Field(default=True, env="CACHE_REDIS_ENABLED", description="Enable Redis as primary cache (default: true). If false, will use memory cache only."), update env.example with Redis configuration variables: mark Redis as REQUIRED/PRIMARY cache, add comments explaining that Redis is default and memory cache is fallback, add Redis connection health check: verify Redis is reachable on startup, if Redis unavailable: log warning and automatically fallback to memory cache, if Redis available: log success message, add Redis service to docker-compose.yml: redis service with official Redis image, expose port 6379, add volume for Redis persistence: /data/redis:/data (REQUIRED for persistence), add environment variables: REDIS_HOST=redis, REDIS_PORT=6379, configure Redis memory limits: maxmemory 2gb (adjustable via env), maxmemory-policy allkeys-lru (for automatic eviction), add Redis health check: healthcheck with redis-cli ping, make Redis service dependency: feature-service depends_on redis with condition: service_healthy, add documentation: explain that Redis is PRIMARY cache with persistence and distribution, memory cache is FALLBACK only, recommend Redis for all production deployments)

- [X] T357 [P] [US2] Add cache configuration variables in feature-service/src/config/__init__.py (add cache_redis_enabled: bool = Field(default=True, env="CACHE_REDIS_ENABLED", description="Enable Redis as primary cache (default: true). System will automatically fallback to memory cache if Redis unavailable."), add cache_ttl_historical_data_seconds: int = Field(default=86400, env="CACHE_TTL_HISTORICAL_DATA_SECONDS", description="TTL for historical data cache entries in seconds (default: 86400 = 24 hours). Applies to both Redis and memory fallback cache."), add cache_ttl_features_seconds: int = Field(default=604800, env="CACHE_TTL_FEATURES_SECONDS", description="TTL for computed features cache entries in seconds (default: 604800 = 7 days). Applies to both Redis and memory fallback cache."), add cache_max_size_mb: int = Field(default=1024, env="CACHE_MAX_SIZE_MB", description="Maximum cache size in MB for memory fallback cache only (default: 1024 = 1GB). Redis uses maxmemory setting from Redis configuration."), add cache_max_entries: int = Field(default=10000, env="CACHE_MAX_ENTRIES", description="Maximum number of cache entries for memory fallback cache only (default: 10000). Redis uses maxmemory-policy from Redis configuration."), add dataset_builder_cache_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_ENABLED", description="Enable caching for dataset building (default: true). Uses Redis if available, falls back to memory cache."), add dataset_builder_cache_historical_data_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_HISTORICAL_DATA_ENABLED", description="Enable historical data caching (default: true). Uses Redis if available, falls back to memory cache."), add dataset_builder_cache_features_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_FEATURES_ENABLED", description="Enable computed features caching (default: true). Uses Redis if available, falls back to memory cache."), add cache_invalidation_on_registry_change: bool = Field(default=True, env="CACHE_INVALIDATION_ON_REGISTRY_CHANGE", description="Automatically invalidate cache when Feature Registry version changes (default: true). Works for both Redis and memory cache."), add cache_invalidation_on_data_change: bool = Field(default=True, env="CACHE_INVALIDATION_ON_DATA_CHANGE", description="Automatically invalidate cache when historical data files are modified (default: true). Works for both Redis and memory cache."), update env.example with all new cache configuration variables and detailed comments: Redis is PRIMARY cache (recommended for all deployments), memory cache is FALLBACK only, Redis setup is REQUIRED for production, memory cache limitations (lost on restart, single instance only))

- [X] T358 [P] [US2] Add cache statistics and monitoring in feature-service/src/services/dataset_builder.py (add cache statistics tracking: hit_count, miss_count, hit_rate, eviction_count, current_size_bytes, current_entries_count, add cache statistics to dataset metadata: store cache_hit_rate, cache_misses_count, cache_hits_count in dataset record, add cache statistics logging: log cache statistics after each dataset build (hit_rate, misses, evictions), log cache statistics periodically (every N dataset builds), add cache performance metrics: track time saved by cache hits, estimate build time without cache, log cache performance impact, add cache health monitoring: monitor cache size growth, eviction rate, hit rate trends, log warnings if cache hit rate is low (<50% for repeated builds), log warnings if cache size exceeds 90% of max_size, add cache statistics API endpoint: GET /cache/statistics returns cache statistics (hit_rate, size, eviction_count, etc.), add cache management API endpoint: POST /cache/invalidate with query parameters (all, pattern, symbol, date_range) for manual cache invalidation, add cache warming endpoint: POST /cache/warm with parameters (symbol, start_date, end_date) to pre-populate cache (optional), add Grafana metrics integration: export cache statistics to metrics table for Grafana dashboard (if Phase 8 Grafana tasks completed))

- [X] T359 [P] [US2] Add cache serialization and compression in feature-service/src/services/cache_service.py (implement DataFrame serialization for historical data cache: use Parquet format for efficient serialization (smaller size, faster than pickle), store as bytes in cache, implement FeatureVector serialization for features cache: use JSON for human-readable format, or msgpack for compact binary format, handle NaN/Inf values correctly (convert to null in JSON), implement optional compression: use gzip compression for cache values to reduce memory/Redis usage (compress before storing, decompress after retrieval), add configuration parameter cache_compression_enabled: bool (default: False, enable if memory/Redis space is limited), add compression level configuration: cache_compression_level: int (default: 6, balance between compression ratio and CPU usage), implement cache value size limits: reject cache entries larger than max_value_size_mb (prevent single large entry from consuming entire cache), add cache serialization error handling: handle serialization/deserialization errors gracefully, log errors, fallback to cache miss, add cache compression benchmarking: measure compression ratio and CPU overhead, log compression statistics, add cache format versioning: include version in serialized cache values, handle format migrations gracefully (invalidate old format entries))

**Checkpoint**: Dataset building caching should significantly speed up repeated builds (target: 10-30x speedup for full cache hits). For 60 days of data, repeated build time should decrease from ~5-15 minutes to ~30 seconds - 2 minutes. Cache should automatically invalidate on Feature Registry version changes and historical data modifications. Cache statistics should be tracked and logged for monitoring.

**Implementation Notes**:
- **Redis Cache (PRIMARY/default)**: Distributed, persistent, suitable for all deployments. Cache persists across container restarts. Uses separate Redis container. RECOMMENDED for all production deployments. Automatic fallback to memory cache if Redis unavailable.
- **Memory Cache (FALLBACK only)**: Simple, fast, used automatically when Redis is unavailable. Cache is lost on container restart. Single-instance only. Used as temporary fallback, not recommended for production. System automatically switches back to Redis when connection restored.
- **Cache Invalidation**: Critical for correctness. Must invalidate on: Feature Registry version changes (affects feature computation), Parquet file modifications (indicates new/updated historical data), manual invalidation requests. Use pattern matching for efficient bulk invalidation.
- **Cache Key Design**: Include all relevant parameters in cache key: symbol, date range, Feature Registry version, data hash. Data hash ensures cache invalidation when data changes even if dates are same.
- **Partial Cache Hits**: Important optimization. If 80%+ of requested data is cached, reuse cached portion and compute only missing 20%. Still provides significant speedup.
- **Cache TTL**: Historical data cache TTL should be shorter (24 hours) as data may be updated via backfill. Features cache TTL can be longer (7 days) as features are more stable.
- **Cache Size Management**: Monitor cache size and eviction rate. LRU eviction ensures most recently used entries are retained. Configure max_size based on available memory/Redis memory.
- **Graceful Degradation**: If Redis unavailable, fallback to memory cache. If cache service fails, fallback to no cache (direct Parquet reads). Never fail dataset build due to cache errors.

---

## Phase 7.7: OfflineEngine and Feature Registry v1.3.0 Testing (Priority: P1) 🔍

**Goal**: Обеспечить полное покрытие тестами для OfflineEngine с Feature Registry v1.3.0. Обнаружена критическая проблема: OfflineEngine не вызывал `compute_all_candle_patterns_3m`, что приводило к отсутствию фичей после фильтрации по Feature Registry v1.3.0 (только свечные паттерны). Тесты не отловили эту проблему, так как все тесты для OfflineEngine были заглушками (placeholder assertions), а тесты для Feature Registry v1.3.0 отсутствовали.

**Problem Analysis**:
1. Тесты для OfflineEngine (`test_offline_engine.py`) - все закомментированы, только placeholder assertions, не проверяют реальную функциональность
2. Нет тестов для Feature Registry v1.3.0 - тесты только для версии 1.2.0 в `test_dataset_builder_feature_registry_filtering.py`
3. Нет проверки, что OfflineEngine вызывает все функции вычисления фичей (включая `compute_all_candle_patterns_3m`)
4. Нет тестов на идентичность фичей: FeatureComputer и OfflineEngine должны вычислять одинаковые фичи

**Expected Outcome**: Все тесты должны проваливаться до исправления проблемы, затем проходить после исправления. Полное покрытие тестами для OfflineEngine и Feature Registry v1.3.0 интеграции.

### Tests for OfflineEngine and Feature Registry v1.3.0

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T360 [P] [US2] Create unit tests for OfflineEngine feature computation in feature-service/tests/unit/test_offline_engine_features.py (uncomment and implement test_offline_engine_compute_features_from_historical_data: verify OfflineEngine computes features from historical data, verify all feature groups are computed: price_features, orderflow_features, orderbook_features, perpetual_features, temporal_features, candle_pattern_features, verify that compute_all_candle_patterns_3m is called by checking that candle pattern features are present in computed features, verify feature names match expected patterns (candle_*, pattern_*), test with sufficient klines data (at least 3 minutes for 3-minute lookback window), verify features are not None and have valid values, use real OfflineEngine instance, not mocks)

- [ ] T361 [P] [US2] Create unit tests for OfflineEngine and FeatureComputer feature identity in feature-service/tests/unit/test_offline_engine_identity.py (test that OfflineEngine and FeatureComputer compute identical features for same data: create same historical data for both engines, compute features with OfflineEngine.compute_features_at_timestamp, compute features with FeatureComputer.compute_features, compare feature names: should be identical, compare feature values: should be identical (within floating point precision), verify all feature groups are present in both: price, orderflow, orderbook, perpetual, temporal, candle_patterns, verify that candle pattern features are present in both engines, test with Feature Registry v1.3.0: both engines should filter to same feature set, both engines should compute candle pattern features)

- [ ] T362 [P] [US2] Create unit tests for OfflineEngine with Feature Registry v1.3.0 filtering in feature-service/tests/unit/test_offline_engine_v1_3_0.py (create Feature Registry v1.3.0 mock configuration with all 79 candle pattern features (candle_*, pattern_*), initialize OfflineEngine with Feature Registry v1.3.0 loader, compute features with OfflineEngine, verify that only candle pattern features are present after filtering, verify that all 79 expected features are present, verify that old features (mid_price, spread_abs, returns_1s, volatility_5m, ema_21, rsi_14) are NOT present after filtering, verify that compute_all_candle_patterns_3m was called (check for candle pattern features in computed features), verify that no features are lost during filtering (all computed candle pattern features should pass filter), test that filtering works correctly: features not in registry are removed, features in registry are kept)

- [ ] T363 [P] [US2] Create unit tests for DatasetBuilder with Feature Registry v1.3.0 in feature-service/tests/unit/test_dataset_builder_v1_3_0.py (test dataset building with Feature Registry v1.3.0: build dataset with v1.3.0, verify dataset contains only candle pattern features, verify all 79 expected features are present in dataset, verify dataset does not contain old features (mid_price, spread_abs, returns_1s, volatility_5m, ema_21, rsi_14), verify features DataFrame is not empty after filtering, verify features have valid values (not all NaN), test feature validation passes: no "No feature columns found in DataFrame" error, test that OfflineEngine is used correctly with Feature Registry v1.3.0, verify that candle pattern features are computed correctly with 3-minute lookback window)

- [ ] T364 [US2] Create integration tests for full dataset building workflow with Feature Registry v1.3.0 in feature-service/tests/integration/test_dataset_building_v1_3_0.py (test complete dataset building with Feature Registry v1.3.0: create dataset with v1.3.0, verify dataset build succeeds, verify dataset status is "ready" (not "failed"), verify dataset contains expected number of records, verify dataset features match Feature Registry v1.3.0: only candle pattern features, all 79 features present, no old features, verify features have valid values: not all NaN, reasonable value ranges, verify dataset can be used for training (model service can load it), test that feature computation includes candle patterns: verify compute_all_candle_patterns_3m was called during dataset build (check logs or metrics), test error scenarios: verify that if compute_all_candle_patterns_3m is not called, test should fail with "No feature columns found", verify that dataset build fails gracefully if insufficient klines data (< 3 minutes), test with different symbols: BTCUSDT, ETHUSDT, verify consistent behavior)

- [ ] T365 [P] [US2] Create integration tests for FeatureComputer and OfflineEngine feature identity with Feature Registry v1.3.0 in feature-service/tests/integration/test_feature_identity_v1_3_0.py (test that FeatureComputer and OfflineEngine produce identical features with Feature Registry v1.3.0: setup FeatureComputer with Feature Registry v1.3.0, setup OfflineEngine with Feature Registry v1.3.0, provide same historical data to both engines, compute features with both engines, verify feature names are identical after filtering, verify feature values are identical (within floating point precision), verify both engines compute candle pattern features, verify both engines filter to same feature set (only candle patterns), verify no features are missing in either engine, test with multiple timestamps: verify identity holds across different timestamps, test with different symbols: verify identity holds across different symbols, document that feature identity is critical for ensuring online and offline features match)

### Implementation for OfflineEngine and Feature Registry v1.3.0 Testing

- [ ] T366 [P] [US2] Update test fixtures for Feature Registry v1.3.0 in feature-service/tests/fixtures/feature_registry.py (add fixture feature_registry_v1_3_0_config: returns Feature Registry v1.3.0 configuration with all 79 candle pattern features, include all feature definitions from feature_registry_v1.3.0.yaml, ensure fixture matches actual registry file structure, add fixture mock_feature_registry_loader_v1_3_0: returns mock FeatureRegistryLoader with v1.3.0 configuration, add fixture sample_klines_for_candle_patterns: returns at least 3 minutes of klines data for testing 3-minute lookback window, ensure klines have sufficient variation (green/red candles, different body/shadow sizes) to trigger all pattern features)

- [ ] T367 [P] [US2] Update OfflineEngine tests to use real implementation in feature-service/tests/unit/test_offline_engine.py (uncomment and implement all placeholder tests: test_offline_engine_compute_features_from_historical_data, test_offline_engine_feature_identity, test_offline_engine_handles_missing_data, test_offline_engine_timestamp_ordering, replace placeholder assertions with real test logic, verify OfflineEngine computes all feature groups including candle patterns, verify features are not None, verify features have expected structure, use real OfflineEngine instance, use real historical data fixtures)

- [ ] T368 [P] [US2] Add test coverage verification for OfflineEngine in feature-service/tests/conftest.py or pytest configuration (add pytest-cov configuration to ensure test coverage for OfflineEngine, require minimum 90% coverage for OfflineEngine.compute_features_at_timestamp method, require coverage for all feature computation functions (price, orderflow, orderbook, perpetual, temporal, candle_patterns), add coverage reporting to CI/CD pipeline, verify that compute_all_candle_patterns_3m call is covered by tests, add test coverage badge or reporting to README)

- [ ] T369 [P] [US2] Document testing strategy for Feature Registry versions in feature-service/tests/README.md or docs/feature-service.md (document that each new Feature Registry version requires: unit tests for feature computation with that version, integration tests for dataset building with that version, feature identity tests comparing FeatureComputer and OfflineEngine, update test fixtures when new registry versions are added, document test execution order: run Feature Registry version tests after OfflineEngine tests, document expected test failures before implementation (TDD approach), document how to add tests for new Feature Registry versions)

**Checkpoint**: OfflineEngine and Feature Registry v1.3.0 should have comprehensive test coverage. All tests should verify that candle pattern features are computed and present in datasets. Feature identity tests should ensure FeatureComputer and OfflineEngine produce identical features. Tests should fail before bug fixes and pass after fixes.

**Implementation Notes**:
- **Test Coverage**: All new tests should follow TDD approach: write tests FIRST, verify they FAIL before implementation, then implement to make them pass
- **Feature Identity**: Critical that FeatureComputer and OfflineEngine compute identical features. This ensures that models trained on offline datasets will work correctly with online feature computation.
- **Feature Registry Version Testing**: Each new Feature Registry version should have dedicated tests. This prevents regression issues like missing feature computation functions.
- **Integration Testing**: Full dataset building workflow tests are essential to catch integration issues between OfflineEngine, DatasetBuilder, and Feature Registry.
- **Test Fixtures**: Comprehensive test fixtures for Feature Registry v1.3.0 and klines data are needed to enable proper testing.

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

- **Total Tasks**: 377 (added 9 Grafana observability tasks T196-T204 + 4 migration application tasks T196a-T199a + 17 Feature Registry versioning tasks T208-T218, T166a-T166b, T169a-T169f: version management, backward compatibility, automatic migration, rollback, audit trail + 36 Statistics API tasks T219-T254: HTTP API endpoints for service statistics and monitoring + 1 dataset validation task T255: empty test split validation and warning + 23 Historical Data Backfilling tasks T256-T277, T263a: Feature Registry data type analysis, DatasetBuilder optimization, Bybit REST API client, backfilling service with data validation, API endpoints, automatic integration, error handling, testing + 3 walk-forward enhancement tasks T111a-T111c: multiple folds generation, tests + 3 new price feature tasks T285-T287: volatility for N candles, returns for N candles, volume z-score + 3 unit test tasks T288-T290 for new price features + 15 advanced feature tasks T291-T304: technical indicators (RSI, EMA, MACD, Bollinger Bands) with pandas_ta dependency, orderbook slope, orderbook churn rate with OrderbookManager extension, Rate of Change (ROC), relative volume, Feature Registry YAML updates + 36 Feature Registry Version Management via Database tasks T305-T323: database schema modification (remove config JSONB, add file_path), FeatureRegistryVersionManager service with file-based source of truth, MetadataStorage extensions, FeatureRegistryLoader db_mode support, startup process updates, API endpoints for version creation and hot reload activation, migration script, docker-compose volume mounting, documentation + 11 Dataset Building Performance Optimization tasks T324-T333, T330b: incremental orderbook reconstruction, incremental rolling windows update, vectorized pandas operations for incremental updates, parallel feature computation, performance metrics, configuration variables. Expected: 10-20x speedup, from ~2 hours to ~5-15 minutes for 60 days + 2 Dataset Quality Validation tasks T334-T335: feature quality checks, target quality validation + 12 Dataset Building Caching Strategy tasks T348-T359: historical data caching, computed features caching, Redis support, cache invalidation logic, cache statistics and monitoring, cache serialization and compression. Expected: 10-30x speedup for repeated builds with full cache hits, from ~5-15 minutes to ~30 seconds - 2 minutes for 60 days + 9 OfflineEngine and Feature Registry v1.3.0 Testing tasks T360-T369: comprehensive test coverage for OfflineEngine with Feature Registry v1.3.0, feature identity tests, integration tests for dataset building with v1.3.0, test fixtures and documentation). Dashboard creation tasks (3 tasks) are in `specs/001-grafana-monitoring/tasks.md`
- **Phase 1 (Setup)**: 10 tasks (added test structure setup)
- **Phase 2 (Foundational)**: 29 tasks (12 tests + 17 implementation: 3 migrations + 3 migration application tasks)
- **Phase 3 (User Story 1)**: 72 tasks (29 tests + 35 implementation, added T285-T287 for new price features: volatility for N candles, returns for N candles, volume z-score, added T288-T290 for unit tests for new price features, added T291-T295 for unit tests for advanced features: technical indicators, orderbook slope/churn rate, ROC, relative volume, added T296-T304 for implementation of advanced features: technical indicators module, orderbook slope/churn rate, ROC, relative volume, Feature Registry updates, OrderbookManager extension, pandas_ta dependency, added T336-T340 for unit tests for minimal feature set: returns_5m, volatility_5m, ema_21, price_ema21_ratio, volume_ratio_20, added T341-T347 for implementation of minimal feature set: returns_5m, volatility_5m, ema_21, price_ema21_ratio, volume_ratio_20 computation, Feature Registry YAML updates, removal of features requiring unavailable backfill data)
- **Phase 4 (User Story 2)**: 48 tasks (25 tests + 26 implementation, added T111a-T111c for walk-forward multiple folds enhancement)
- **Phase 5 (User Story 3)**: 14 tasks (7 tests + 7 implementation)
- **Phase 6 (User Story 4)**: 24 tasks (11 tests + 13 implementation, added T334-T335 for dataset quality validation: feature quality checks, target quality validation)
- **Phase 7 (User Story 5)**: 71 tasks (22 tests + 49 implementation, including original version management T161-T218 + 19 new tasks for database-driven version management T305-T323: 6 test tasks for FeatureRegistryVersionManager and DB integration, 13 implementation tasks for file-based source of truth architecture with hot reload, database schema modification, service layer updates, API endpoints, migration script, docker-compose configuration, documentation)
- **Phase 7.5 (Dataset Building Performance Optimization)**: 11 tasks (4 tests + 7 implementation: incremental orderbook reconstruction, incremental rolling windows update, vectorized pandas operations for incremental updates, parallel feature computation, performance metrics, configuration variables. Expected: 10-20x speedup, from ~2 hours to ~5-15 minutes for 60 days)
- **Phase 7.6 (Dataset Building Caching Strategy)**: 12 tasks (4 tests + 8 implementation: historical data caching, computed features caching, Redis support in separate container, cache invalidation logic with Feature Registry version changes and data modifications, cache statistics and monitoring, cache serialization and compression. Expected: 10-30x speedup for repeated builds with full cache hits, from ~5-15 minutes to ~30 seconds - 2 minutes for 60 days)
- **Phase 7.7 (OfflineEngine and Feature Registry v1.3.0 Testing)**: 9 tasks (6 tests + 3 implementation: comprehensive test coverage for OfflineEngine with Feature Registry v1.3.0, feature identity tests between FeatureComputer and OfflineEngine, integration tests for dataset building with v1.3.0, test fixtures for v1.3.0, test coverage verification, documentation. Addresses critical bug: OfflineEngine not calling compute_all_candle_patterns_3m)
- **Phase 8 (Polish)**: 61 tasks (5 additional tests + 12 implementation + 7 Grafana observability tasks: T196-T199 database migrations + T196a-T199a migration application, T200-T203 metrics persistence, T204 health check extension + 36 Statistics API tasks T219-T254: 16 tests + 20 implementation + 1 dataset validation task T255: empty test split validation). Grafana dashboard creation tasks (T205-T207) are in `specs/001-grafana-monitoring/tasks.md`
- **Phase 9 (Historical Data Backfilling)**: 23 tasks (T256-T277, T263a: Feature Registry data type analysis and mapping, DatasetBuilder optimization to load only required data types, Bybit REST API client, backfilling service with Feature Registry awareness and data validation after save, API endpoints for manual and automatic backfilling, integration with dataset builder, comprehensive error handling and logging, unit/integration/contract tests)

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) = 103 tasks (10 + 29 + 64)

**Test Tasks Breakdown**:

- **Test Fixtures/Mocks/Stubs**: 26 tasks (added 1 Statistics API fixture: T219)
- **Unit Tests**: 78 tasks (added 4 Statistics API unit tests: T220-T223, added 3 price feature unit tests: T288-T290, added 5 advanced feature unit tests: T291-T295 for technical indicators, orderbook slope/churn rate, ROC, relative volume, added 3 Feature Registry version management unit tests: T305-T307 for FeatureRegistryVersionManager, MetadataStorage extensions, DB integration, added 3 Dataset Building Performance Optimization unit tests: T324-T326 for incremental orderbook, incremental rolling windows, parallel computation, added 5 minimal feature set unit tests: T336-T340 for returns_5m, volatility_5m, ema_21, price_ema21_ratio, volume_ratio_20, added 4 Dataset Building Caching Strategy unit tests: T348-T350 for historical data caching, computed features caching, cache invalidation logic, added 6 OfflineEngine and Feature Registry v1.3.0 Testing unit tests: T360-T363 for OfflineEngine feature computation, feature identity, Feature Registry v1.3.0 filtering, DatasetBuilder with v1.3.0)
- **Integration Tests**: 30 tasks (added 5 Statistics API integration tests: T224-T228, added 1 Feature Registry version management integration test: T307, added 1 Dataset Building Performance Optimization integration test: T327 for full dataset build performance, added 1 Dataset Building Caching Strategy integration test: T351 for dataset building with caching, added 2 OfflineEngine and Feature Registry v1.3.0 Testing integration tests: T364-T365 for full dataset building workflow with v1.3.0, feature identity with v1.3.0)
- **Contract Tests**: 24 tasks (added 6 Statistics API contract tests: T229-T234, added 3 Feature Registry version management contract tests: T308-T310 for version creation, activation with hot reload, file sync)
- **E2E/Performance Tests**: 5 tasks
- **Total Test Tasks**: 171 tasks (added 16 Statistics API test tasks + 6 Feature Registry version management test tasks + 4 Dataset Building Performance Optimization test tasks + 5 minimal feature set test tasks + 12 Dataset Building Caching Strategy test tasks + 9 OfflineEngine and Feature Registry v1.3.0 Testing test tasks)

**Parallel Opportunities**:

- Setup phase: 4 tasks can run in parallel
- Foundational phase: 12 test tasks can run in parallel, 9 implementation tasks can run in parallel
- User Story 1: 4 test fixtures can run in parallel, 8 unit tests can run in parallel, 3 models can run in parallel
- User Stories 1 and 3 can start in parallel after foundational phase
