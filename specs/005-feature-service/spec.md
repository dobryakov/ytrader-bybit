# Feature Specification: Feature Service

**Feature Branch**: `005-feature-service`  
**Created**: 2025-01-27  
**Status**: Draft  
**Input**: User description: "Feature Service - сервис для приёма маркет-данных, вычисления признаков в реальном времени и пересборки исторических признаков для обучения моделей"

## Clarifications

### Session 2025-01-27

- Q: Message queue naming convention for market data from ws-gateway → A: Use `ws-gateway.*` prefix to match existing pattern (e.g., `ws-gateway.orderbook`, `ws-gateway.trades`)
- Q: Storage technology for raw Parquet data files → A: Local filesystem in container (mounted volumes)
- Q: API authentication method for REST endpoints → A: API Key authentication (API key in header or query parameter)
- Q: Minimum data retention period for raw data → A: 90 days (3 months) before archiving/deletion

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Получение признаков в реальном времени для торговых решений (Priority: P1)

Model Service должен получать готовые признаки в реальном времени для принятия торговых решений. Система должна вычислять признаки из потока маркет-данных и предоставлять их с минимальной задержкой через API или очередь событий.

**Why this priority**: Это основная функция Feature Service - без возможности получать признаки в реальном времени, система не может использоваться для принятия торговых решений в production.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что признаки вычисляются и доступны через API в течение заданного времени задержки.

**Acceptance Scenarios**:

1. **Given** система получает поток маркет-данных (стакан, сделки, свечи), **When** Model Service запрашивает последние признаки для символа, **Then** система возвращает актуальный вектор признаков с задержкой не более 70 мс
2. **Given** система получает обновления маркет-данных, **When** признаки вычисляются автоматически, **Then** обновленные признаки публикуются в очередь событий для подписчиков
3. **Given** система получает маркет-данные, **When** запрашиваются признаки, **Then** система возвращает полный вектор признаков со всеми вычисленными признаками

---

### User Story 2 - Сборка датасета для обучения моделей с корректным разделением (Priority: P1)

Model Service должен иметь возможность запросить сборку датасета обучения из исторических данных с явным разделением на train/validation/test периоды. Система должна пересчитать признаки на исторических данных идентично онлайн-режиму и сгенерировать целевые переменные без data leakage.

**Why this priority**: Без возможности обучения моделей на исторических данных, система не может улучшать торговые стратегии. Корректное разделение данных критично для валидной оценки качества моделей.

**Independent Test**: Можно протестировать независимо, запросив сборку датасета за определенный период и проверив, что датасет содержит корректно разделенные данные train/val/test, признаки идентичны онлайн-версии, и targets вычислены без использования будущей информации.

**Acceptance Scenarios**:

1. **Given** система хранит исторические маркет-данные за период, **When** Model Service запрашивает сборку датасета с указанием периодов train/validation/test, **Then** система создает датасет с корректным разделением и уведомляет о готовности через очередь событий
2. **Given** система собирает датасет, **When** вычисляются признаки и targets, **Then** система проверяет отсутствие data leakage (признаки используют только данные до момента t, targets - только данные после t)
3. **Given** система собрала датасет, **When** Model Service запрашивает скачивание датасета, **Then** система предоставляет датасет в требуемом формате с метаданными о разделении и версии Feature Registry

---

### User Story 3 - Хранение сырых данных для последующей пересборки (Priority: P2)

Система должна хранить сырые маркет-данные в структурированном формате, обеспечивающем эффективную пересборку признаков на исторических данных. Данные должны храниться минимальный период времени с возможностью архивирования.

**Why this priority**: Хранение сырых данных необходимо для пересборки признаков при изменении Feature Registry и для обучения на исторических периодах, но не критично для базовой работы онлайн-режима.

**Independent Test**: Можно протестировать независимо, отправив маркет-данные в систему и проверив, что данные сохраняются в структурированном формате, доступны для чтения, и старые данные автоматически архивируются или удаляются после истечения срока хранения.

**Acceptance Scenarios**:

1. **Given** система получает поток маркет-данных, **When** данные обрабатываются, **Then** сырые данные сохраняются в структурированном формате с временными метками
2. **Given** система хранит данные дольше минимального периода хранения, **When** истекает срок хранения, **Then** старые данные автоматически архивируются или удаляются согласно политике хранения
3. **Given** система хранит исторические данные, **When** запрашивается пересборка признаков за период, **Then** система использует сохраненные сырые данные для пересчета признаков

---

### User Story 4 - Мониторинг качества данных и обнаружение проблем (Priority: P2)

Система должна отслеживать качество входящих данных, обнаруживать проблемы (пропуски, аномалии, рассинхронизации) и предоставлять отчеты о качестве данных через API.

**Why this priority**: Качество признаков зависит от качества исходных данных. Обнаружение проблем позволяет предотвратить использование некорректных данных для торговых решений.

**Independent Test**: Можно протестировать независимо, отправив данные с проблемами (пропуски, аномалии) и проверив, что система их обнаруживает и предоставляет отчет через API.

**Acceptance Scenarios**:

1. **Given** система получает маркет-данные, **When** обнаруживаются пропуски или аномалии, **Then** система логирует проблемы и обновляет метрики качества данных
2. **Given** система отслеживает качество данных, **When** запрашивается отчет о качестве данных за период, **Then** система возвращает отчет с метриками (missing rate, anomaly rate, sequence gaps) и рекомендациями
3. **Given** система обнаруживает рассинхронизацию в orderbook, **When** проблема выявлена, **Then** система автоматически запрашивает snapshot для восстановления корректного состояния

---

### User Story 5 - Управление конфигурацией признаков через Feature Registry (Priority: P3)

Система должна предоставлять возможность управления конфигурацией признаков через Feature Registry с проверкой на data leakage и валидацией временных границ.

**Why this priority**: Feature Registry позволяет централизованно управлять признаками и обеспечивает идентичность онлайн/оффлайн вычислений, но это вспомогательная функция по сравнению с основной функциональностью.

**Independent Test**: Можно протестировать независимо, загрузив конфигурацию Feature Registry и проверив, что система валидирует конфигурацию, проверяет временные границы и позволяет перезагрузить конфигурацию.

**Acceptance Scenarios**:

1. **Given** администратор загружает новую версию Feature Registry, **When** конфигурация проверяется, **Then** система валидирует временные границы признаков и отсутствие data leakage
2. **Given** Feature Registry обновлен, **When** запрашивается перезагрузка конфигурации, **Then** система применяет новую конфигурацию для последующих вычислений признаков
3. **Given** система использует Feature Registry, **When** запрашивается информация о текущей версии, **Then** система возвращает версию и метаданные о признаках

---

### Edge Cases

#### Данные и источники

- **Недоступность ws-gateway**: Когда ws-gateway недоступен или отправляет неполные данные, система должна продолжать работать с последними доступными данными, логировать проблемы, и обновлять метрики качества данных. При восстановлении соединения система должна синхронизировать пропущенные данные если возможно. *(См. FR-019, FR-020)*
- **Отсутствие данных для символа**: Когда запрашиваются признаки для символов, по которым нет данных, система должна возвращать ошибку 404 с описанием проблемы или использовать последние доступные значения с пометкой о устаревании данных. *(См. FR-005)*
- **Опциональная очередь execution events**: Когда очередь ws-gateway.order недоступна (опциональная очередь для execution events), система должна продолжать работу и логировать предупреждение. *(См. FR-001.2)*

#### Качество данных и восстановление

- **Рассинхронизация orderbook**: При обнаружении пропущенных delta-событий система должна автоматически запросить snapshot для восстановления корректного состояния orderbook, затем продолжить обработку с обновленного состояния. *(См. FR-001.3, FR-021)*

#### Конфигурация и валидация

- **Data leakage в Feature Registry**: Если Feature Registry содержит конфигурацию с потенциальным data leakage, система должна отклонять загрузку конфигурации, возвращать детальное описание проблемы, и продолжать использовать предыдущую валидную версию. *(См. FR-022, FR-045)*
- **Невалидная конфигурация Feature Registry**: Когда Feature Registry содержит невалидную конфигурацию (например, lookback_window больше max_lookback_days), система должна отклонять загрузку, возвращать ошибку валидации с указанием конкретных проблем, и продолжать использовать текущую валидную конфигурацию. *(См. FR-040, FR-045)*

#### Ограничения ресурсов и производительность

- **Превышение минимального периода хранения**: При превышении минимального периода хранения данных (90 дней) система должна автоматически архивировать или удалять старые данные согласно политике хранения, сохраняя возможность восстановления из архива при необходимости. *(См. FR-002.3)*
- **Очень большой объем данных при сборке датасета**: При обработке очень большого объема данных при сборке датасета система должна обрабатывать данные батчами, предоставлять прогресс через API, и поддерживать возможность приостановки/возобновления процесса. *(См. FR-015, FR-033)*
- **Превышение допустимой задержки**: Когда задержка вычисления признаков превышает допустимую (70 мс), система должна логировать предупреждение, обновлять метрики производительности, и продолжать работу с пометкой о сниженной производительности. *(См. FR-003, SC-001)*
- **Одновременные запросы на сборку датасетов**: При одновременных запросах на сборку нескольких датасетов система должна обрабатывать их последовательно или параллельно (в зависимости от доступных ресурсов), с приоритизацией и управлением очередью запросов. *(См. FR-015)*

#### Исторические данные

- **Отсутствие исторических данных**: Когда запрашивается сборка датасета за период, для которого отсутствуют исторические данные, система должна возвращать ошибку с указанием доступных периодов и предложением альтернативных дат. *(См. FR-009, FR-015)*

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST receive raw market data streams (orderbook, trades, klines, ticker, funding) from internal message queues
- **FR-001.1**: System MUST receive data from queues: ws-gateway.orderbook (snapshots and deltas), ws-gateway.trades, ws-gateway.kline, ws-gateway.ticker, ws-gateway.funding
- **FR-001.2**: System MUST optionally receive execution events from ws-gateway.order queue or trading.executions queue for own order executions
- **FR-001.3**: System MUST handle orderbook sequence/order correctly: build snapshot + delta, handle desynchronization by requesting snapshot, store all deltas for offline reconstruction
- **FR-001.4**: System MUST add internal timestamp and exchange timestamp to all received messages
- **FR-002**: System MUST store raw market data in structured format for at least 90 days before archiving/deletion
- **FR-002.1**: System MUST organize stored raw data by data type: orderbook snapshots, orderbook deltas, trades, klines, ticker, funding rate, execution events
- **FR-002.2**: System MUST store all orderbook deltas for offline orderbook reconstruction
- **FR-002.3**: System MUST support configurable retention period (default 90 days) with automatic archiving or deletion after expiration
- **FR-003**: System MUST compute online features in real-time from market data streams with latency ≤ 70 ms
- **FR-003.1**: System MUST compute features at intervals: every 1s, 3s, 15s, and 1m
- **FR-003.2**: System MUST compute basic price/candlestick features: mid_price, spread_abs, spread_rel, returns (1s, 3s, 1m), VWAP (3s, 15s, 1m), volume (3s, 15s, 1m), volatility (1m, 5m)
- **FR-003.3**: System MUST compute orderflow features: signed_volume (3s, 15s, 1m), buy/sell volume ratio, trade_count (3s), net_aggressor_pressure
- **FR-003.4**: System MUST compute orderbook features: depth (bid/ask top5/top10), depth_imbalance (top5)
- **FR-003.5**: System MUST compute perpetual features: funding_rate, time_to_funding
- **FR-003.6**: System MUST compute temporal/meta features: time_of_day (cyclic encoding using sin/cos components: sin(2π * hour / 24), cos(2π * hour / 24))
- **FR-004**: System MUST publish computed features to message queue for Model Service subscribers
- **FR-005**: System MUST provide REST API endpoint to retrieve latest features for a symbol with latency ≤ 70 ms
- **FR-007**: System MUST support rebuilding features from historical data in offline/batch mode
- **FR-008**: System MUST guarantee feature identity between online and offline computation modes (same calculation code and parameters)
- **FR-009**: System MUST support dataset building with explicit train/validation/test period splits
- **FR-010**: System MUST generate regression targets (future returns) for specified prediction horizons (1m, 5m, 15m, 1h)
- **FR-011**: System MUST generate classification targets (direction: up/down/flat) based on return thresholds
- **FR-011.1**: System MUST use configurable threshold (default 0.001 = 0.1%) for classification: up if return > threshold, down if return < -threshold, flat if |return| ≤ threshold
- **FR-011.2**: System MUST generate risk-adjusted targets (e.g., sharpe_return) as optional target type, using volatility calculated from past data (not future data)
- **FR-012**: System MUST prevent data leakage by validating that features use only data before time t and targets use only data after time t
- **FR-013**: System MUST support time-based dataset splitting into train/validation/test periods
- **FR-014**: System MUST support walk-forward validation strategy with configurable window sizes and steps
- **FR-014.1**: System MUST support configurable walk-forward parameters: train_window_days, validation_window_days, step_days, start_date, end_date, min_train_samples
- **FR-014.2**: System MUST generate sequential folds where each fold has train on [T0, T1) and validation on [T1, T2), with last fold optionally used as test set
- **FR-014.3**: System MUST support random split strategy (for testing only, not for production use cases) with temporal order preserved within each split
- **FR-015**: System MUST provide REST API for dataset building requests with period specifications
- **FR-016**: System MUST notify about dataset completion through message queue
- **FR-017**: System MUST provide REST API to list available datasets with metadata
- **FR-018**: System MUST provide REST API to download datasets or dataset splits (train/validation/test)
- **FR-019**: System MUST track data quality metrics (missing rate, anomaly rate, sequence gaps)
- **FR-020**: System MUST provide REST API for data quality reports over specified periods
- **FR-021**: System MUST detect and handle orderbook desynchronization by requesting snapshots
- **FR-022**: System MUST validate Feature Registry configuration for data leakage prevention
- **FR-023**: System MUST provide REST API to retrieve and reload Feature Registry configuration
- **FR-024**: System MUST authenticate all API requests (except health checks) using API Key

### Key Entities

- **Raw Market Data**: Исторические сырые данные маркет-данных (orderbook snapshots/deltas, trades, klines, ticker, funding) хранятся в структурированном формате для последующей пересборки признаков
- **Feature Vector**: Вектор вычисленных признаков для символа в конкретный момент времени, включающий ценовые признаки, orderflow признаки, orderbook признаки и мета-признаки
- **Feature Registry**: Конфигурация, определяющая какие признаки вычислять, их источники данных, временные окна, параметры расчета и правила предотвращения data leakage
- **Dataset**: Структурированный набор данных для обучения моделей, содержащий признаковые векторы и целевые переменные, разделенный на train/validation/test периоды
- **Target Variable**: Целевая переменная для обучения модели, вычисленная на основе будущих данных (returns, direction, risk-adjusted returns) для заданного горизонта предсказания

## API Endpoints

### Online Features API

- **FR-029**: System MUST provide `GET /features/latest?symbol=BTCUSDT` endpoint returning latest online features with structure: `{timestamp, symbol, features: {...}}` with latency ≤ 50-70 ms
- **FR-030**: System MUST publish computed features to message queue `features.live` for real-time streaming to subscribers

### Dataset Building API

- **FR-031**: System MUST provide `POST /dataset/build` endpoint accepting time-based split configuration with periods (train, validation, test), target_config (type, horizon, threshold), feature_registry_version, data_leakage_check flag, and output_format
- **FR-032**: System MUST provide `POST /dataset/build` endpoint accepting walk-forward validation configuration with walk_forward_config (train_window_days, validation_window_days, step_days, start_date, end_date, min_train_samples)
- **FR-033**: System MUST return dataset building response with dataset_id, status (building/ready), estimated_completion, and splits information (records count, periods)
- **FR-034**: System MUST provide `GET /dataset/list` endpoint with query parameters (status, symbol) returning list of available datasets with metadata about splits
- **FR-035**: System MUST provide `GET /dataset/{dataset_id}` endpoint returning dataset metadata: periods, record counts, feature registry version, target config
- **FR-036**: System MUST provide `GET /dataset/{dataset_id}/download?split=train` endpoint supporting split parameter (train, validation, test, or all) for downloading dataset splits
- **FR-037**: System MUST publish dataset completion notifications to message queue `features.dataset.ready` with message structure: `{dataset_id, status, splits_info, download_urls}`

### Model Evaluation API

- **FR-038**: System MUST provide `POST /model/evaluate` endpoint accepting dataset_id, split (test/validation), model_version, and metrics_to_compute array, returning prepared dataset for model evaluation with metadata about metrics

### Feature Registry API

- **FR-039**: System MUST provide `GET /feature-registry` endpoint returning current Feature Registry version and configuration
- **FR-040**: System MUST provide `POST /feature-registry/reload` endpoint for reloading feature configuration
- **FR-041**: System MUST provide `GET /feature-registry/validate` endpoint for validating Feature Registry configuration for data leakage

### Data Quality API

- **FR-042**: System MUST provide `GET /data-quality/report?symbol=BTCUSDT&from=...&to=...` endpoint returning data quality report with: missing rate, anomaly detection results, sequence gaps, desynchronization events

## Feature Registry Configuration

- **FR-043**: System MUST maintain Feature Registry configuration (YAML/JSON format) describing: feature name, input sources (trades/orderbook/kline), lookback window (3s, 15s, 1m, etc.), normalization parameters, calculation order
- **FR-044**: System MUST require each feature in Feature Registry to explicitly specify: lookback_window (time window into past, e.g., "3s", "1m"), lookahead_forbidden: true flag, max_lookback_days for validation, data_sources list with timestamps
- **FR-045**: System MUST validate Feature Registry at load time: check temporal boundaries, verify no future data usage in features, validate data source availability

## Offline Feature Engine Requirements

- **FR-046**: System MUST support restoring orderbook state from snapshot + all deltas for any historical period
- **FR-047**: System MUST support restoring all rolling windows (1s, 3s, 15s, 1m) for historical data
- **FR-048**: System MUST compute features from historical data identically to online mode using same calculation code and Feature Registry
- **FR-050**: System MUST export datasets in structured format with metadata about train/validation/test splits, feature registry version, and target configuration

## Workflows

### Model Training Workflow

- **FR-051**: System MUST support workflow where Model Service requests dataset building via `POST /dataset/build` with explicit train/validation/test periods
- **FR-052**: System MUST validate dataset build request for data leakage prevention before starting build process
- **FR-053**: System MUST notify Model Service through `features.dataset.ready` queue when dataset building is complete
- **FR-054**: System MUST provide workflow where Model Service can download dataset, train on train set, validate on validation set, evaluate on test set (out-of-sample), and activate model only if test set metrics exceed threshold

### Inference Workflow

- **FR-055**: System MUST support workflow where Model Service subscribes to `features.live` queue or polls `GET /features/latest` API for ready feature vectors
- **FR-056**: System MUST ensure Model Service receives complete feature vector including all computed features (price, orderflow, orderbook, perpetual, temporal features)
- **FR-057**: System MUST support workflow where Model Service performs inference on received feature vector and generates trading signal based on prediction

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System delivers computed features to Model Service with latency ≤ 70 ms from receiving market data update (measured at 95th percentile)
- **SC-002**: System successfully rebuilds features from historical data with 100% identity to online features (verified by automated tests comparing online vs offline feature values)
- **SC-003**: System prevents data leakage in 100% of dataset builds (validated by automated checks ensuring features use only past data and targets use only future data)
- **SC-004**: System maintains raw market data availability for dataset rebuilding for minimum 90 days with ≥ 99.9% data completeness rate
- **SC-005**: System completes dataset building requests within 2 hours for 1 month of historical data for a single symbol
- **SC-006**: System detects and handles orderbook desynchronization within 1 second of detection, restoring correct orderbook state
- **SC-007**: System provides data quality reports within 5 seconds of API request for any 24-hour period
- **SC-009**: System successfully builds datasets with explicit train/validation/test splits where test set contains data from period never seen during training (100% temporal separation)

## Assumptions

- Market data is provided by ws-gateway service through internal message queues with `ws-gateway.*` naming convention
- Raw data storage uses local filesystem with mounted volumes (Parquet format) - storage capacity is sufficient for 90+ days of data
- Model Service is refactored to accept ready feature vectors and not compute features independently
- API authentication uses API Key method (key in header or query parameter)
- System operates in containerized environment with horizontal scaling capability by symbol
- System supports horizontal scaling by symbols for distributed feature computation
- System is resilient to ws-gateway failures and can recover from snapshots
- System maintains comprehensive logging: raw data events, sequence errors, missing data gaps, desynchronization events
- System validates incoming data: missing values detection, anomaly detection (outliers, price spikes), sequence gap detection in orderbook, timestamp validation (internal timestamp vs exchange timestamp)
- System supports strategies for handling problematic data: interpolation, forward fill, skipping records for missing values; requesting snapshots and rebuilding for orderbook desynchronization
- System maintains automated tests for feature identity between online and offline modes (derivation tests)
