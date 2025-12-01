# Feature Specification: Model Service - Trading Decision and ML Training Microservice

**Feature Branch**: `001-model-service`  
**Created**: 2025-11-26  
**Status**: Draft  
**Input**: User description: "Микросервис обучения и принятия торговых решений (Model Service)"

## Clarifications

### Session 2025-11-26

- Q: What message format should be used for order execution events and trading signals in the message queue? → A: Plain JSON without schema versioning
- Q: How should trained ML models be stored and persisted (file system, database, object storage)? → A: File system with database metadata
- Q: How should the system handle retraining conflicts when multiple triggers occur simultaneously (queue, cancel/restart, concurrent, ignore)? → A: Cancel current training and restart with new data
- Q: What authentication and authorization approach should be used for service integrations? → A: API key authentication for service-to-service REST API calls (infrastructure services like RabbitMQ and PostgreSQL use their standard username/password authentication)
- Q: Should rate limiting/throttling be implemented for signal generation to prevent resource exhaustion? → A: Configurable rate limit with burst allowance

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Warm-up Mode Signal Generation (Priority: P1)

The system generates trading signals using simple heuristics or controlled random generation when no trained model exists, allowing trading to begin immediately and accumulate initial data for learning.

**Why this priority**: This is the foundation that enables the system to start operating from day one without requiring historical data or pre-trained models. Without this, the system cannot begin trading or collecting data needed for model training.

**Independent Test**: Can be fully tested by configuring the system in warm-up mode and verifying that it generates trading signals at the configured frequency with appropriate risk controls, publishes them to the message queue, and logs all activities. The system delivers immediate trading capability without any model dependencies.

**Acceptance Scenarios**:

1. **Given** the system is in warm-up mode with no trained model available, **When** the system is started, **Then** it begins generating trading signals using configured heuristics or random generation
2. **Given** the system is generating warm-up signals, **When** a signal is generated, **Then** it is published to the message queue with all required fields (signal type, asset, amount, confidence, timestamp, strategy_id)
3. **Given** the system is in warm-up mode, **When** order execution events are received, **Then** they are accepted and aggregated for future training purposes
4. **Given** warm-up mode is configured with specific parameters (frequency, randomness level, minimum order parameters), **When** signals are generated, **Then** they adhere to these configured constraints
5. **Given** the system is in warm-up mode, **When** transitions between modes occur, **Then** all transitions are logged with timestamps and parameter values

---

### User Story 2 - Model Training from Execution Feedback (Priority: P2)

The system trains or retrains ML models in real-time using feedback from order execution events, analyzing actual trade performance and market data to improve decision-making capabilities.

**Why this priority**: This enables the system to learn and improve from actual trading outcomes, transforming from heuristic-based to data-driven decision making. This is essential for transitioning from warm-up mode to intelligent trading.

**Independent Test**: Can be fully tested by providing the system with order execution events and market data, verifying that it processes these into training datasets, performs model training or retraining operations, and tracks training progress and model versions. The system delivers continuous learning capability that improves over time.

**Acceptance Scenarios**:

1. **Given** the system has received order execution events, **When** sufficient data has been accumulated, **Then** the system processes this data into training datasets
2. **Given** training datasets are available, **When** model training is triggered (by schedule, data threshold, or quality degradation), **Then** the system performs training or retraining operations and produces a new model version
3. **Given** a model training operation completes, **When** the new model is evaluated, **Then** its quality metrics are calculated and recorded
4. **Given** the system is continuously receiving execution events, **When** new data arrives, **Then** it is incorporated into the training pipeline for ongoing model improvement (online learning or batch accumulation)
5. **Given** multiple model versions exist, **When** model quality is assessed, **Then** the system can compare versions and identify the best-performing model
6. **Given** the system is configured for periodic retraining, **When** the scheduled time arrives, **Then** the system triggers model retraining using accumulated data since the last training
7. **Given** model quality metrics degrade below configured thresholds, **When** quality monitoring detects the degradation, **Then** the system automatically triggers model retraining
8. **Given** the system has accumulated a configured minimum number of new execution events, **When** the data threshold is reached, **Then** the system triggers model retraining
9. **Given** a model training operation is in progress, **When** a new retraining trigger occurs (scheduled, data threshold, or quality degradation), **Then** the system cancels the current training operation and restarts with new data

---

### User Story 3 - Intelligent Signal Generation from Trained Model (Priority: P3)

The system generates high-level trading signals using trained ML models, making strategic decisions based on current market state and learned patterns from historical execution data.

**Why this priority**: This represents the core value proposition - intelligent, data-driven trading decisions. However, it depends on having a trained model (from P2) and can initially be replaced by warm-up mode (P1), making it lower priority for initial deployment.

**Independent Test**: Can be fully tested by providing the system with a trained model and current market state, verifying that it generates trading signals with appropriate confidence scores, publishes them to the message queue, and logs the decision-making process. The system delivers intelligent trading recommendations based on learned patterns.

**Acceptance Scenarios**:

1. **Given** a trained model is available and current order/position state is retrieved, **When** the system evaluates market conditions, **Then** it generates trading signals with confidence scores
2. **Given** a trading signal is generated by the model, **When** it is published, **Then** it includes all required fields (signal type, asset, amount, confidence, timestamp, strategy_id) and meets quality thresholds
3. **Given** the system has both warm-up and trained model capabilities, **When** model quality reaches the configured threshold, **Then** the system automatically transitions from warm-up mode to model-based signal generation
4. **Given** multiple trading strategies are configured, **When** signals are generated, **Then** they are tagged with the appropriate strategy identifier
5. **Given** model-based signals are being generated, **When** execution feedback is received, **Then** it is used to evaluate and improve model performance

---

### User Story 4 - Model Quality Tracking and Versioning (Priority: P4)

The system tracks model quality metrics, maintains version history, and provides observability into model performance and system behavior for monitoring and debugging.

**Why this priority**: Essential for production operations and continuous improvement, but the system can function with basic logging initially. This enables operational excellence and long-term optimization.

**Independent Test**: Can be fully tested by generating multiple model versions, executing trades, and verifying that quality metrics are calculated, version history is maintained, all operations are logged appropriately, and monitoring data is available. The system delivers full observability into model performance and system health.

**Acceptance Scenarios**:

1. **Given** a model has been trained, **When** quality metrics are calculated, **Then** they are stored with the model version and timestamp
2. **Given** multiple model versions exist, **When** version history is queried, **Then** the system returns all versions with their quality metrics and training metadata
3. **Given** the system is operating, **When** any significant event occurs (signal generation, model training, mode transitions), **Then** it is logged with appropriate detail and traceability
4. **Given** model performance degrades below thresholds, **When** quality is monitored, **Then** alerts or notifications are generated to indicate the issue
5. **Given** the system needs to rollback to a previous model version, **When** a version is selected, **Then** the system can switch to that version and continue operations

---

### User Story 5 - Position-Based Exit Strategy (Priority: P3)

The system reacts to position updates in real-time and generates exit signals (SELL) based on configurable exit rules such as take profit, stop loss, trailing stop, and time-based exits, enabling proactive risk management and profit protection.

**Why this priority**: This enables reactive position management that responds immediately to position changes, protecting profits and limiting losses. While the system can function with periodic signal generation, real-time exit strategy evaluation significantly improves risk management and capital efficiency. This complements the existing take profit rule by adding comprehensive exit strategy framework.

**Independent Test**: Can be fully tested by simulating position updates with various unrealized PnL values, holding times, and market conditions, verifying that exit signals are generated when rules are triggered, rate limiting prevents excessive signal generation, and all exit decisions are logged with traceability. The system delivers reactive position management that protects capital and locks in profits.

**Acceptance Scenarios**:

1. **Given** a position update event is received, **When** the position has unrealized profit exceeding take profit threshold, **Then** the system generates a SELL signal to close the position
2. **Given** a position update event is received, **When** the position has unrealized loss exceeding stop loss threshold, **Then** the system generates a SELL signal to limit losses
3. **Given** a position has reached profit activation threshold, **When** price moves against the position, **Then** the system generates a SELL signal based on trailing stop distance
4. **Given** a position has been held for maximum configured time, **When** the position meets profit target, **Then** the system generates a SELL signal to close the position
5. **Given** multiple exit rules are configured, **When** a position update is received, **Then** the system evaluates all applicable rules and generates exit signal if any rule triggers
6. **Given** an exit signal was recently generated for a position, **When** another position update is received, **Then** the system respects rate limiting and cooldown period to prevent excessive signal generation
7. **Given** exit strategy evaluation fails due to missing data or errors, **When** the error occurs, **Then** the system logs the error with full context and continues processing other positions
8. **Given** position state tracking is required for trailing stop, **When** position updates are received, **Then** the system tracks peak price and highest unrealized PnL for trailing stop evaluation
9. **Given** partial exit is configured for take profit, **When** first profit threshold is reached, **Then** the system generates SELL signal for partial position closure (e.g., 50% at 3% profit)
10. **Given** the system is processing position updates, **When** multiple rapid updates occur for the same position, **Then** the system applies debouncing to prevent excessive evaluation

---

### Edge Cases

- What happens when no order execution events are received for an extended period (affecting retraining schedules)?
- How does the system handle retraining conflicts (e.g., scheduled retraining triggered while another training is in progress)? **RESOLVED**: System cancels current training operation and restarts with new data when a new retraining trigger occurs during an in-progress training operation.
- How does the system handle corrupted or invalid execution event data? **RESOLVED**: System MUST validate all incoming execution events, log validation failures with full event context and trace IDs, discard invalid events, and continue processing valid events. Invalid events are logged for debugging and monitoring purposes.
- What happens when the database connection is lost while retrieving order/position state?
- How does the system handle message queue failures when publishing signals?
- What happens when model training fails or produces invalid results?
- How does the system handle transitions between warm-up and model-based modes when model quality is borderline?
- What happens when multiple conflicting signals are generated for the same asset?
- How does the system handle database schema changes or missing required data?
- What happens when warm-up mode parameters are misconfigured (e.g., frequency too high, amounts too large)? **RESOLVED**: System enforces configurable rate limits with burst allowance; signals exceeding the rate limit are throttled to prevent resource exhaustion.
- How does the system handle partial order execution events or incomplete market data?
- What happens when position update events are received but position data is incomplete or missing required fields? **RESOLVED**: System MUST validate position update events, log validation failures with full event context and trace IDs, skip exit strategy evaluation for invalid events, and continue processing valid events.
- How does the system handle race conditions when multiple position updates arrive rapidly for the same asset? **RESOLVED**: System applies debouncing and per-asset rate limiting to prevent excessive exit signal generation, with configurable cooldown periods.
- What happens when exit strategy evaluation fails due to Position Manager API unavailability? **RESOLVED**: System falls back to periodic evaluation mode, logs the degradation, and continues operating with reduced reactivity.
- How does the system handle position state tracking persistence across service restarts? **RESOLVED**: System stores position state (peak price, entry time, highest PnL) in Redis or database for persistence and recovery.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST subscribe to message queue for enriched order execution events from the order manager microservice (format: plain JSON without schema versioning)
- **FR-002**: System MUST retrieve current state of open orders and positions from shared PostgreSQL database
- **FR-003**: System MUST analyze order execution events and associated market data to form training datasets. For feature engineering, the system MUST use market data snapshot from trading signals (captured at signal generation time) as the primary source for features describing market state at decision time. Execution event market_conditions (at execution time) are used for performance evaluation and slippage analysis, but not for feature engineering. When matching execution events with signals, the system MUST use the signal's market_data_snapshot for feature construction.
- **FR-004**: System MUST train or retrain ML models using feedback from order executions, supporting both online learning (continuous incremental updates) and periodic batch retraining
- **FR-019**: System MUST support configurable retraining triggers: scheduled periodic retraining, data accumulation thresholds, and quality degradation detection
- **FR-020**: System MUST accumulate execution events and market data for training, processing them either incrementally (online learning) or in batches (periodic retraining)
- **FR-021**: System MUST automatically trigger model retraining when model quality metrics fall below configured degradation thresholds
- **FR-022**: System MUST support configurable retraining schedules (e.g., daily, weekly) for periodic batch retraining operations
- **FR-023**: System MUST cancel any in-progress training operation and restart with new data when a new retraining trigger occurs during an active training operation
- **FR-005**: System MUST generate high-level trading signals (buy/sell, asset, amount, confidence, timestamp, strategy_id) based on model predictions. Signals MUST include market data snapshot at generation time (price, spread, volume_24h, volatility, and optionally orderbook depth and technical indicators) to enable accurate feature engineering during model training when matched with execution events.
- **FR-006**: System MUST publish generated trading signals to message queue for order manager microservice (format: plain JSON without schema versioning)
- **FR-007**: System MUST operate in warm-up mode when no trained model is available, using heuristics or controlled random generation
- **FR-008**: System MUST generate warm-up signals with configurable frequency and risk parameters. Warm-up signals MUST include market data snapshot at generation time (price, spread, volume_24h, volatility) to enable accurate feature engineering during model training. Market data MUST be retrieved from available sources (shared database, message queues, or ws-gateway subscriptions) at the time of signal generation.
- **FR-024**: System MUST enforce configurable rate limiting on signal generation with burst allowance to prevent resource exhaustion and message queue overload
- **FR-009**: System MUST accept and aggregate order execution events during warm-up mode for future training
- **FR-010**: System MUST automatically transition from warm-up mode to model-based generation when model quality reaches configured threshold
- **FR-011**: System MUST provide configuration for warm-up mode parameters (duration, randomness level, minimum order parameters)
- **FR-012**: System MUST track model quality metrics and maintain version history (models stored as files on file system, metadata in PostgreSQL database)
- **FR-013**: System MUST log all significant operations (signal generation, model training, mode transitions) with appropriate detail
- **FR-014**: System MUST provide monitoring capabilities for model performance and system health
- **FR-015**: System MUST handle integration with other microservices (order manager, ws-gateway) through defined interfaces with API key authentication for REST API calls. Integration with ws-gateway is primarily through shared PostgreSQL database (for order/position state) and RabbitMQ message queues (for event consumption). Direct REST API calls to ws-gateway are not required for core functionality.
- **FR-026**: System MAY subscribe to Bybit WebSocket data channels (trades, ticker, orderbook, kline, etc.) via REST API calls to ws-gateway service when additional real-time market data is required for feature engineering or signal generation. After subscription, events are consumed from RabbitMQ queues (ws-gateway.{event_type}). This is an optional capability that enhances model training and signal generation with real-time market data beyond what is included in order execution events.
- **FR-016**: System MUST validate trading signals before publishing (required fields, value ranges, format compliance)
- **FR-017**: System MUST handle errors gracefully (queue failures, database unavailability, model training failures) with appropriate fallbacks
- **FR-025**: System MUST validate all incoming order execution events, log validation failures with full event context and trace IDs, discard invalid or corrupted events, and continue processing valid events without interruption
- **FR-018**: System MUST support multiple trading strategies with distinct strategy identifiers
- **FR-027**: System MUST react to position update events in real-time and evaluate exit strategies when positions change
- **FR-028**: System MUST support configurable exit rules including take profit, stop loss, trailing stop, and time-based exits
- **FR-029**: System MUST generate SELL signals when exit rules are triggered, with configurable partial exit support
- **FR-030**: System MUST enforce rate limiting and cooldown periods for exit signal generation to prevent excessive signal generation
- **FR-031**: System MUST track position state (peak price, entry time, highest unrealized PnL) for trailing stop and time-based exit evaluation
- **FR-032**: System MUST handle position update event validation errors gracefully, logging failures and continuing processing
- **FR-033**: System MUST support fallback to periodic evaluation mode when event-driven processing is unavailable

### Key Entities *(include if feature involves data)*

- **Trading Signal**: Represents a high-level trading decision containing signal type (buy/sell), asset identifier, amount in currency, confidence score (0-1), timestamp, and strategy identifier. Published to message queue for execution. Signals MUST include market data snapshot at signal generation time (price, spread, volume_24h, volatility, and optionally orderbook depth and technical indicators) to enable accurate feature engineering during model training. This snapshot captures market state at decision time, which is essential for learning correct patterns from execution feedback.

- **Order Execution Event**: Enriched event from order manager containing details about executed trades, including execution price, quantity, fees, timestamp, and associated market conditions. Used for training and performance evaluation.

- **Model Version**: Represents a trained ML model instance with version identifier, training timestamp, quality metrics, configuration parameters, and performance history. Models are stored as files on the file system (e.g., `/models/v{version}.pkl`), while metadata (version ID, timestamps, quality metrics, configuration, file path) is stored in PostgreSQL database. Supports versioning and rollback capabilities.

- **Order/Position State**: Current snapshot of open orders and positions retrieved from shared database, including order status, position sizes, entry prices, and unrealized P&L. Used as input for signal generation.

- **Training Dataset**: Aggregated collection of order execution events and associated market data organized for model training, including features, labels, and metadata about data quality and coverage.

- **Model Quality Metrics**: Quantitative measures of model performance including prediction accuracy, confidence calibration, risk-adjusted returns, and other domain-specific metrics. Used for model selection and transition decisions.

- **Position State**: Tracks position lifecycle data including entry price, entry time, peak price, highest unrealized PnL, and last exit signal timestamp. Used for exit strategy evaluation, particularly for trailing stop and time-based exits. Stored in Redis or database for persistence.

- **Exit Decision**: Result of exit strategy evaluation containing should_exit flag, exit_reason, exit_amount (partial or full), and priority. Used to determine if and how to generate exit signals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System generates trading signals within 5 seconds (p95 latency) of receiving required input data (order state, market data, or warm-up trigger)
- **SC-002**: System successfully publishes 99.5% of generated signals to message queue without errors
- **SC-003**: System processes and incorporates order execution events into training pipeline within 10 seconds of receipt
- **SC-004**: System completes model training operations within 30 minutes for datasets up to 1 million records
- **SC-011**: System triggers periodic retraining according to configured schedule with accuracy within 5 minutes of scheduled time
- **SC-012**: System detects model quality degradation and triggers retraining within 1 minute of threshold breach
- **SC-005**: System transitions from warm-up mode to model-based generation when model quality metrics exceed 75% classification accuracy threshold (percentage of correct buy/sell predictions on validation set)
- **SC-006**: System maintains model version history with complete metadata for at least 100 previous versions
- **SC-007**: System logs all critical operations (signal generation, training, mode transitions) with 100% coverage and traceability
- **SC-008**: System handles message queue and database connection failures with automatic retry and graceful degradation, maintaining 99% uptime
- **SC-009**: Warm-up mode generates signals at configurable frequency (default: 1 signal per minute) with risk parameters preventing excessive exposure, subject to configurable rate limits with burst allowance
- **SC-010**: System supports concurrent operation of multiple trading strategies (minimum 5) without interference or performance degradation
- **SC-013**: System evaluates exit strategies and generates exit signals within 2 seconds (p95 latency) of receiving position update events
- **SC-014**: System successfully generates exit signals for 99% of triggered exit rules without errors
- **SC-015**: System enforces rate limiting to prevent more than 1 exit signal per position per minute (configurable)
- **SC-016**: System tracks position state for trailing stop with accuracy within 0.1% of actual peak price

## Assumptions

- Order manager microservice exists and publishes enriched execution events to a message queue (using plain JSON format without schema versioning)
- Shared relational database contains tables for open orders and positions accessible by this microservice
- Message queue infrastructure is available and configured for event streaming
- Market data required for analysis is available through integration with other microservices or data sources. Order execution events include market conditions at execution time. If additional real-time market data is needed (e.g., for feature engineering or signal generation), the system MAY subscribe to Bybit WebSocket channels via ws-gateway REST API (POST /api/v1/subscriptions) and consume events from RabbitMQ queues (ws-gateway.{event_type})
- Model training can be performed in real-time or near-real-time without blocking signal generation
- System has sufficient computational resources for ML model training operations
- Integration interfaces with other microservices are defined and stable
- Configuration management system is available for warm-up mode, retraining schedules, and other operational parameters
- System supports both online learning (incremental updates) and periodic batch retraining approaches
- Logging and monitoring infrastructure is available for observability requirements
- Service-to-service REST API calls use API key authentication; infrastructure services (RabbitMQ, PostgreSQL) use standard username/password authentication

## Dependencies

- Order Manager microservice (for execution events and signal consumption)
- Shared relational database (for order/position state)
- Message queue infrastructure (for event streaming)
- Market data sources (for training and signal generation context). Optional: ws-gateway service for subscribing to Bybit WebSocket channels when additional real-time market data is required beyond order execution events
- Logging and monitoring infrastructure (for observability)
- Configuration management system (for operational parameters)
