# Feature Specification: Order Manager Microservice

**Feature Branch**: `004-order-manager`  
**Created**: 2025-11-27  
**Status**: Draft  
**Input**: User description: "Order Manager microservice for managing Bybit orders and executing trading signals"

## Clarifications

### Session 2025-11-27

- Q: How should REST API endpoints be secured? → A: API Key Authentication (X-API-Key header), similar to WebSocket Gateway service
- Q: How should duplicate signals with the same signal_id be handled? → A: Allow duplicates if previous processing failed (retry mechanism)
- Q: How should the system handle Bybit API rate limits? → A: Implement exponential backoff with configurable retry limits
- Q: What should be the FIFO ordering scope for signal processing? → A: Per symbol (asset) FIFO
- Q: How should positions be stored and retrieved? → A: Hybrid: store current position with periodic snapshots, compute from orders for validation

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Execute Trading Signals (Priority: P1)

As a trading system, I need to receive high-level trading signals from the model service and execute them as actual orders on Bybit exchange, ensuring proper order creation, modification, and cancellation based on current balance and open positions.

**Why this priority**: This is the core functionality - without signal execution, the system cannot trade. This delivers immediate value by enabling automated trading.

**Independent Test**: Can be fully tested by sending a trading signal to the service and verifying that an appropriate order is created on Bybit and stored in the database with correct status.

**Acceptance Scenarios**:

1. **Given** a valid buy signal for 1000 USDT worth of BTCUSDT is received, **When** the service processes the signal, **Then** the service determines appropriate order type, price, and quantity based on signal parameters and market conditions, creates the order on Bybit, stores it in the database with status "pending", and publishes an order execution event to the queue
2. **Given** a sell signal is received, **When** the service processes it, **Then** the service checks available balance and open positions, determines order attributes (type, price, quantity), creates the order if sufficient funds exist, and publishes the result
3. **Given** a signal is received with insufficient balance, **When** the service processes it, **Then** the order is rejected with appropriate error logging and an event is published indicating the rejection reason
4. **Given** a signal is received while an existing order for the same asset is open, **When** the service processes it, **Then** the service either modifies the existing order or creates a new one based on signal parameters and current state
5. **Given** a trading signal with market data snapshot is received, **When** the service processes it, **Then** the service uses market data (price, spread, orderbook depth) to determine optimal order parameters (limit price, order type selection) before creating the order
6. **Given** a signal requires quantity calculation from amount, **When** the service processes it, **Then** the service converts quote currency amount to base currency quantity using appropriate price source, applies tick size/lot size precision, rounds according to defined rules, and validates minimum quantity requirements
7. **Given** a large signal amount is received, **When** the service processes it, **Then** the service determines whether to split into multiple orders based on defined criteria, creates appropriate number of orders, and stores signal-to-order relationships
8. **Given** multiple signals for the same asset with same direction are received, **When** the service processes them, **Then** the service applies scale-in logic to either accumulate into existing orders or create new orders according to defined position building rules
9. **Given** a signal is received for an asset with existing position, **When** the service processes it, **Then** the service considers current position size, average price, and PnL according to position management rules before creating the order
10. **Given** signals arrive out of chronological order, **When** the service processes them, **Then** the service applies defined ordering rules (FIFO, timestamp-based, or arrival order) to ensure correct processing sequence
11. **Given** two signals for the same asset arrive simultaneously, **When** the service processes them, **Then** the service applies conflict resolution strategy (queue, merge, prioritize, or reject) according to defined rules
12. **Given** a signal would exceed configured risk limits, **When** the service processes it, **Then** the service rejects or adjusts the order to comply with risk limits (max exposure, max order size) and logs the violation
13. **Given** a new signal arrives for an asset with existing pending orders, **When** the service processes it, **Then** the service applies cancellation strategy to determine which existing orders to cancel before creating new orders
14. **Given** a sell signal arrives while buy orders are pending, **When** the service processes it, **Then** the service applies rules for cancelling opposite orders according to defined cancellation strategy
15. **Given** a market order signal arrives while limit orders are pending, **When** the service processes it, **Then** the service handles pending limit orders according to defined strategy (cancel, keep, or modify)
16. **Given** service restarts after downtime, **When** it initializes, **Then** the service performs reconciliation by querying active orders, detecting missed WebSocket events, and synchronizing order states with Bybit
17. **Given** service is in dry-run mode, **When** a signal is received, **Then** the service processes the signal, validates it, simulates order creation, but does not send orders to Bybit and logs all simulated operations
18. **Given** a position has unrealized loss exceeding threshold, **When** the service evaluates it, **Then** the service applies stop-loss mechanism according to defined rules (automatic stop-order placement or notification to model service)

---

### User Story 2 - Maintain Order State Accuracy (Priority: P1)

As a trading system, I need to keep order status information synchronized with Bybit exchange, updating order states when they change (filled, partially filled, cancelled, rejected) so that the system always has accurate information about order execution.

**Why this priority**: Accurate order state is critical for risk management and decision-making. Without this, the system cannot make informed trading decisions.

**Independent Test**: Can be fully tested by subscribing to order execution events from WebSocket gateway and verifying that order states in the database are updated correctly when events are received.

**Acceptance Scenarios**:

1. **Given** an order is placed on Bybit, **When** an order execution event is received from WebSocket gateway indicating the order was filled, **Then** the order status in the database is updated to "filled" with execution details
2. **Given** an order is partially filled, **When** a partial fill event is received, **Then** the order status is updated to "partially_filled" with the executed quantity recorded
3. **Given** an order is cancelled on Bybit, **When** a cancellation event is received, **Then** the order status is updated to "cancelled" with timestamp
4. **Given** order state synchronization is needed, **When** a manual synchronization request is triggered, **Then** the service queries Bybit REST API for current order states and updates the database accordingly

---

### User Story 3 - Publish Order Events (Priority: P2)

As other microservices (especially the model service), I need to receive enriched order execution events so I can track order outcomes, learn from execution results, and adjust trading strategies accordingly.

**Why this priority**: Event publishing enables feedback loops for model learning and system observability, but the system can function without it initially.

**Independent Test**: Can be fully tested by verifying that when an order state changes, an enriched event is published to the appropriate queue with all relevant order and execution details.

**Acceptance Scenarios**:

1. **Given** an order is executed, **When** the order state is updated, **Then** an enriched order event is published to the queue containing order details, execution price, quantity, fees, and market conditions at execution time
2. **Given** an order is rejected, **When** the rejection occurs, **Then** an event is published with rejection reason and context for debugging and learning
3. **Given** an order is modified, **When** the modification completes, **Then** an event is published with before/after state comparison

---

### User Story 4 - Safety and Risk Protection (Priority: P1)

As a trading system, I need protection against incorrect or risky trading actions, ensuring that signals are validated before execution and that orders cannot exceed available balance or violate configured risk limits.

**Why this priority**: Safety mechanisms prevent financial losses from bugs or incorrect signals. This is critical for production trading systems.

**Independent Test**: Can be fully tested by sending invalid or risky signals and verifying that they are rejected with appropriate safety checks, without creating orders on the exchange.

**Implementation Notes**:
- Balance check before order creation is **optional** and can be disabled via `ORDERMANAGER_ENABLE_BALANCE_CHECK` configuration (default: `true`).
- When balance check is disabled, Bybit API will still reject orders with insufficient balance, but early rejection and detailed logging are lost.
- Disabling balance check reduces API calls and improves performance, but trades off early validation benefits.

**Acceptance Scenarios**:

1. **Given** a signal requests an order amount exceeding available balance, **When** the service processes it, **Then**:
   - If `ORDERMANAGER_ENABLE_BALANCE_CHECK=true`: the order is rejected with a clear error message and no order is created on Bybit (early rejection)
   - If `ORDERMANAGER_ENABLE_BALANCE_CHECK=false`: the order is sent to Bybit, which will reject it with insufficient balance error (late rejection)
2. **Given** a signal has invalid parameters (negative amount, invalid asset), **When** the service processes it, **Then** the signal is rejected with validation error before any exchange API call
3. **Given** a signal would violate configured risk limits, **When** the service processes it, **Then** the order is rejected or adjusted to comply with limits
4. **Given** multiple conflicting signals arrive simultaneously, **When** the service processes them, **Then** the service handles conflicts appropriately (e.g., queueing, prioritizing, or rejecting duplicates)

---

### Edge Cases

- What happens when Bybit API is temporarily unavailable during order creation?
- How does system handle network timeouts when communicating with Bybit?
- What happens when a signal arrives for an asset that is no longer tradeable on Bybit?
- How does system handle duplicate signals with the same signal_id?
- What happens when order execution events arrive out of order or are delayed?
- How does system handle partial order fills over multiple events?
- What happens when balance changes between signal receipt and order execution?
- How does system handle WebSocket disconnection during active order monitoring?
- What happens when database write fails after successful Bybit order creation?
- How does system handle signals with confidence below a configured threshold?
- How does system handle quantity calculation when market price changes between signal receipt and order creation?
- What happens when multiple signals for the same asset arrive simultaneously with conflicting directions?
- How does system handle order splitting when one signal must create multiple orders?
- What happens when calculated quantity violates Bybit minimum order size requirements?
- How does system handle signals that arrive out of chronological order (earlier timestamp arrives later)?
- What happens when risk limits are exceeded during signal processing?
- How does system handle position state when switching between one-way and hedge-mode?
- What happens when retry attempts are exhausted for a failed order creation?
- How does system handle signals that would create orders exceeding maximum position size?
- What happens when service restarts and finds orders in database that may have been filled during downtime?
- How does system handle position calculation when order history has gaps or inconsistencies?
- What happens when dry-run mode is enabled but service receives real WebSocket events from previous live orders?
- How does system handle stop-loss orders when position is partially closed by other means?
- What happens when manual resynchronization is triggered while orders are being actively processed?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST store orders in the shared PostgreSQL database with current status, execution details, and timestamps
- **FR-002**: System MUST receive trading signals from RabbitMQ queue `model-service.trading_signals` in the specified JSON format
- **FR-003**: System MUST validate incoming trading signals for required fields (signal_id, signal_type, asset, amount), data types, and value ranges (amount > 0, confidence 0.0-1.0)
- **FR-004**: System MUST check available balance and existing open orders before creating new orders to prevent over-trading
- **FR-005**: System MUST make decisions about order creation, modification, or cancellation based on signal parameters and current system state (balance, open orders)
- **FR-005.1**: System MUST implement architectural and/or logical solution for determining order type, composition, and attributes (order type: limit/market, price, quantity, time in force, etc.) based on received trading signal parameters, market conditions, and system state
- **FR-005.2**: System MUST implement order type selection mechanism (market vs limit) with defined rules for when to use each type, how to determine limit price, how to select time-in-force, and whether to use reduce_only/post_only flags
- **FR-005.3**: System MUST implement quantity calculation mechanism that converts quote currency amount from signals to base currency quantity with defined precision (tick size/lot size), rounding rules, and handling of minimum quantity violations
- **FR-005.4**: System MUST implement signal-to-order relationship logic that defines when one signal creates multiple orders, how to store N signals → M orders relationships, rules for incremental position building (scale-in), and handling of partial repeated signals (scale-in/scale-out)
- **FR-005.5**: System MUST implement position management rules that account for current position size, average price, PnL, and Bybit hedging mode (one-way vs hedge-mode) when processing signals and creating orders
- **FR-005.6**: System MUST implement signal processing order and priority mechanism that handles out-of-order signals, simultaneous signals for the same asset, and defines FIFO ordering rules. FIFO scope MUST be per symbol (asset): signals for the same asset processed in order, different assets can process in parallel
- **FR-005.7**: System MUST provide configurable risk limits (max exposure, max order size) and retry parameters with defined configuration format, validation, and enforcement mechanisms. Retry strategy MUST use exponential backoff with configurable retry limits for Bybit API calls
- **FR-005.8**: System MUST implement order cancellation strategy that defines when to cancel existing orders, rules for cancelling on opposite signals, handling of limit orders when market signals arrive, and automatic cancellation of stale orders
- **FR-005.9**: System MUST provide REST API endpoints for querying order lists, viewing position state, health/live/readiness probes, and manual resynchronization. All endpoints (except health probes) MUST require API key authentication via X-API-Key header
- **FR-005.10**: System MUST implement reconciliation and recovery procedures that define which orders are considered active, startup state synchronization, and handling of missed WebSocket events during downtime
- **FR-005.11**: System MUST implement or define responsibility for stop-loss/take-profit mechanism, including rules for automatic stop-order placement and handling of positions with significant unrealized PnL
- **FR-006**: System MUST create orders on Bybit exchange via REST API with appropriate order parameters (symbol, side, quantity, order type)
- **FR-007**: System MUST modify existing orders on Bybit via REST API when signal indicates order changes are needed
- **FR-008**: System MUST cancel orders on Bybit via REST API when signals indicate cancellation is needed
- **FR-009**: System MUST process responses from Bybit REST API and handle both success and error cases appropriately
- **FR-010**: System MUST subscribe to order execution event topics from WebSocket gateway to receive real-time order status updates
- **FR-011**: System MUST update order status in database when order execution events are received from WebSocket gateway
- **FR-012**: System MUST log all order operations (creation, modification, cancellation, status updates) with sufficient detail for debugging and auditing
- **FR-013**: System MUST publish enriched order events to RabbitMQ queue for other microservices (including model service) when order states change
- **FR-014**: System MUST enrich published events with additional context (execution price, fees, market conditions, timing) when available
- **FR-015**: System MUST provide safety mechanisms to prevent execution of incorrect or risky trading actions, including balance validation (optional, configurable via `ORDERMANAGER_ENABLE_BALANCE_CHECK`), parameter validation, and risk limit checks. Balance validation can be disabled for performance optimization, as Bybit API will reject orders with insufficient balance regardless
- **FR-016**: System MUST provide tools for manual order state synchronization by querying Bybit REST API directly to refresh order statuses
- **FR-017**: System MUST handle errors gracefully, including API failures, network issues, and invalid responses, with appropriate retry logic and error reporting. For Bybit API rate limits (429 errors), system MUST implement exponential backoff with configurable retry limits
- **FR-018**: System MUST maintain trace_id throughout request processing for request flow tracking across microservices
- **FR-019**: System MUST handle duplicate signals (same signal_id) by tracking signal processing state: reject duplicates if previous processing succeeded, but allow retry if previous processing failed (retry mechanism)
- **FR-020**: System MUST support both warm-up mode signals (is_warmup=true) and model-generated signals (is_warmup=false) with appropriate handling for each type
- **FR-021**: System MUST provide REST API endpoints for querying orders with filtering, pagination, and sorting capabilities. Endpoints MUST require API key authentication via X-API-Key header
- **FR-022**: System MUST provide REST API endpoint for retrieving current position state with all relevant position information. Endpoint MUST require API key authentication via X-API-Key header
- **FR-023**: System MUST implement health, liveness, and readiness probe endpoints that report service and dependency status. Health probe endpoints MAY be unauthenticated for monitoring purposes
- **FR-024**: System MUST implement reconciliation procedures on service startup to recover order state and detect missed events
- **FR-025**: System MUST implement or coordinate stop-loss/take-profit mechanism with defined responsibility boundaries and automatic placement rules
- **FR-026**: System MUST implement hybrid position storage strategy: store current position state in database with periodic snapshots, compute positions from orders for validation, and maintain data consistency procedures. The implementation MUST include: (1) defined data model for position storage and snapshots (see data-model.md), (2) configurable snapshot frequency via `ORDERMANAGER_POSITION_SNAPSHOT_INTERVAL`, (3) position computation algorithm for validation, and (4) discrepancy handling procedures when computed position differs from stored state
- **FR-027**: System MUST support dry-run mode that processes signals without sending orders to Bybit and provides comprehensive logging of simulated operations. Dry-run mode MUST: (1) accept and validate all trading signals normally, (2) perform full order processing logic (type selection, quantity calculation, risk checks), (3) simulate order creation/modification/cancellation without API calls to Bybit, (4) log all simulated operations with sufficient detail for validation, and (5) be configurable via `ORDERMANAGER_DRY_RUN` environment variable

### Key Entities *(include if feature involves data)*

- **Order**: Represents a trading order placed on Bybit exchange. Key attributes: order_id (Bybit), signal_id (from trading signal), asset/symbol, side (buy/sell), quantity, price, order_type, status (pending, filled, partially_filled, cancelled, rejected), execution details (filled quantity, average price, fees), timestamps (created, updated, executed), trace_id for request tracking. Relationship: one order can be linked to one or more signals (for scale-in/scale-out scenarios)
- **Trading Signal**: High-level trading instruction received from model service. Key attributes: signal_id (UUID), signal_type (buy/sell), asset, amount (in quote currency), confidence, timestamp, strategy_id, model_version, is_warmup flag, market_data_snapshot, metadata, trace_id. Relationship: one signal can result in one or more orders; multiple signals can result in one order (for position accumulation)
- **Signal-Order Relationship**: Represents the mapping between trading signals and orders. Key attributes: signal_id, order_id, relationship_type (one-to-one, one-to-many, many-to-one), execution_sequence, allocation_amount/quantity. Purpose: track which orders were created from which signals and enable position building across multiple signals
- **Order Execution Event**: Real-time event received from WebSocket gateway about order status changes. Key attributes: order_id, event_type (filled, partially_filled, cancelled, rejected), execution details, timestamp. Relationship: multiple events can relate to one order as it progresses through states
- **Position**: Represents current trading position for an asset. Key attributes: asset/symbol, size (positive for long, negative for short), average_entry_price, unrealized_pnl, realized_pnl, mode (one-way or hedge-mode with separate long/short positions). Relationship: position is affected by orders and affects order creation decisions

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System processes and executes valid trading signals within 2 seconds of signal receipt, from queue consumption to order creation on exchange
- **SC-002**: System maintains order state accuracy with 99% synchronization rate between database and Bybit exchange (verified through periodic reconciliation)
- **SC-003**: System successfully executes 95% of valid trading signals without errors (excluding signals rejected due to insufficient balance or risk limits, which are expected rejections)
- **SC-004**: System publishes order execution events within 1 second of order state changes, enabling timely feedback to model service
- **SC-005**: System prevents 100% of orders that would exceed available balance from being created on the exchange. When `ORDERMANAGER_ENABLE_BALANCE_CHECK=true`, this is achieved through early rejection before API call. When disabled, Bybit API rejects orders with insufficient balance, achieving the same protection through late rejection
- **SC-006**: System handles WebSocket disconnections and reconnections without losing order state updates, resubscribing to order topics within 5 seconds of reconnection
- **SC-007**: System provides manual synchronization capability that can refresh order states for all active orders within 30 seconds
- **SC-008**: System logs all order operations with sufficient detail such that 100% of order lifecycle events can be traced and audited

## Assumptions

- Bybit REST API and WebSocket gateway are available and accessible
- Shared PostgreSQL database is available and contains necessary tables for order storage
- RabbitMQ queues are configured and accessible for receiving signals and publishing events
- WebSocket gateway service is operational and can provide order execution event subscriptions
- Model service sends signals in the specified JSON format with all required fields
- Balance information is available either from database or from WebSocket gateway balance events
- Order state synchronization via REST API is available as a fallback when WebSocket events are delayed or missed
- Risk limits and safety thresholds can be configured via environment variables or configuration files
- The system operates in a single trading account context (not multi-account)
- Position data (size, average price, PnL) is available from Bybit API or can be calculated from order history
- Bybit hedging mode (one-way or hedge-mode) is configured and known to the system
- Signal timestamps are reliable and can be used for ordering decisions
- REST API endpoints can be implemented for querying order and position state
- Dry-run mode can be toggled via configuration without code changes
- Position data can be retrieved from Bybit API or computed from order history when needed

## Dependencies

- **WebSocket Gateway Service**: Provides order execution event subscriptions via WebSocket topics
- **Model Service**: Sends trading signals via RabbitMQ queue `model-service.trading_signals`
- **Shared PostgreSQL Database**: Stores order information and current state
- **RabbitMQ Message Broker**: Provides queues for signal consumption and event publishing
- **Bybit Exchange API**: REST API for order creation, modification, cancellation, and state queries; WebSocket for real-time events (via gateway)

## Design Decisions

**Status**: The following architectural and logical decisions have been resolved and documented in `research.md`. Implementation should reference the decisions documented there. For details on each decision, see the corresponding section in `research.md`:

- **Decision 1 (Order Type Selection)**: See `research.md` section "2. Order Type Selection Strategy"
- **Decision 2 (Quantity Calculation)**: See `research.md` section "3. Quantity Calculation from Amount"
- **Decision 3 (Signal-to-Order Relationships)**: See `research.md` section "4. Signal-to-Order Relationship Logic"
- **Decision 4 (Position Management)**: See `research.md` section "5. Position Management Rules"
- **Decision 5 (Signal Processing Order)**: See `research.md` section "6. Signal Processing Order and Priority"
- **Decision 6 (Configuration Parameters)**: See `research.md` section "7. Configuration Parameters"
- **Decision 7 (Order Cancellation)**: See `research.md` section "8. Order Cancellation Strategy"
- **Decision 8 (Query API)**: See `research.md` section "9. Query API and State Inspection"
- **Decision 9 (Reconciliation)**: See `research.md` section "10. Reconciliation and Recovery After Restart"
- **Decision 10 (Stop-Loss/Take-Profit)**: See `research.md` section "11. Stop-Loss / Take-Profit Mechanism"
- **Decision 11 (Position Storage)**: See `research.md` section "12. Position History Storage"
- **Decision 12 (Dry-Run Mode)**: See `research.md` section "13. Dry-Run Mode"

The following sections document the original decision questions for reference:

### 1. Order Type Selection Mechanism (Market vs Limit)

The service MUST create orders with the "correct order type", but the selection rules are not defined. The following decisions must be made:

- **When to use LIMIT vs MARKET orders?**
  - Criteria for selecting order type based on signal parameters (confidence, urgency, market conditions)
  - Rules for using market data snapshot (spread, volatility, orderbook depth) in order type decision
  - Handling of warm-up signals vs model-generated signals in order type selection
  - Risk considerations (slippage tolerance, execution certainty) affecting order type choice

- **How to determine limit order price?**
  - Strategy for setting limit price: use current market price from signal snapshot, apply spread offset, or use other pricing logic
  - Rules for buy vs sell limit price placement (above/below market)
  - Handling of stale market data in price determination
  - Consideration of orderbook depth and liquidity in price selection

- **How to select time-in-force (TIF)?**
  - Rules for choosing TIF options (GTC, IOC, FOK) based on signal characteristics
  - Relationship between order type and time-in-force selection
  - Handling of time-sensitive signals requiring immediate execution

- **Can reduce_only and post_only flags be used?**
  - Criteria for using `reduce_only` flag (position reduction scenarios)
  - Criteria for using `post_only` flag (maker fee optimization)
  - Rules for combining these flags with order type and signal parameters

### 2. Quantity Calculation Mechanism (Amount to Qty Conversion)

Trading signals contain amount in quote currency (USDT), but conversion rules are not specified. The following decisions must be made:

- **How to convert quote currency amount to base currency quantity?**
  - Price source for conversion: use signal snapshot price, current market price, or limit order price
  - Handling of price changes between signal receipt and order creation
  - Consideration of order type (market vs limit) in quantity calculation

- **What precision to use (tick size / lot size)?**
  - Rules for determining tick size and lot size from Bybit exchange specifications
  - Handling of different precision requirements for different trading pairs
  - Dynamic precision lookup based on asset symbol

- **How to round quantities?**
  - Rounding strategy: round up, round down, or round to nearest
  - Rules for rounding based on order type and signal parameters
  - Handling of edge cases (very small amounts, precision boundaries)

- **What to do if minimum quantity is not met?**
  - Behavior when calculated quantity is below Bybit minimum order size
  - Options: reject signal, adjust to minimum, or accumulate with future signals
  - Error handling and notification for quantity violations

### 3. Signal-to-Order Relationship Logic

The specification states "one signal can result in one or more orders", but the rules are not defined. The following decisions must be made:

- **When should one signal create multiple orders?**
  - Criteria for order splitting: large amounts, risk limits, market impact considerations
  - Rules for determining number of orders and size distribution
  - Time-based splitting (immediate vs staggered execution)
  - Handling of partial fills in multi-order scenarios

- **How to store relationship between N signals → M orders?**
  - Data model for tracking signal-to-order relationships (one-to-one, one-to-many, many-to-one)
  - Storage of relationship metadata (which orders belong to which signal, execution sequence)
  - Query patterns for retrieving orders by signal or signals by order

- **Can position be built incrementally (scale-in)?**
  - Rules for accumulating position across multiple signals for the same asset
  - Handling of conflicting signals (buy and sell for same asset)
  - Position tracking and state management for incremental builds
  - Limits on position size and accumulation rules

- **What to do with partial repeated signals (scale-in / scale-out)?**
  - Logic for handling multiple signals for the same asset with same direction
  - Rules for combining or separating signals into orders
  - Handling of scale-out scenarios (partial position reduction)
  - Conflict resolution when signals arrive simultaneously or in quick succession

### 4. Position Management Rules

Order Manager must account for position state when processing signals, but the rules are not defined. The following decisions must be made:

- **How to consider current position size?**
  - Rules for checking existing position before creating new orders
  - Handling of position size when processing buy/sell signals
  - Logic for position reduction vs position increase scenarios
  - Integration with position data from Bybit API or database

- **How to use average price information?**
  - Whether to consider average entry price when making order decisions
  - Rules for calculating new average price after order execution
  - Impact of average price on order type selection (e.g., limit price placement)
  - Handling of average price updates from partial fills

- **How to account for PnL (Profit and Loss)?**
  - Whether to consider current PnL when processing signals
  - Rules for using PnL in order decision-making (e.g., stop-loss, take-profit logic)
  - Handling of unrealized vs realized PnL in decision logic
  - Integration with PnL data from Bybit or calculated internally

- **How to handle Bybit hedging mode (one-way vs hedge-mode)?**
  - Rules for working in one-way mode (single position per asset)
  - Rules for working in hedge-mode (separate long/short positions per asset)
  - Decision on which mode to support or how to handle both
  - Position tracking and management differences between modes
  - Impact of hedging mode on order creation and position calculations

### 5. Signal Processing Order and Priority

Signals may arrive out of order or simultaneously, but processing rules are not defined. The following decisions must be made:

- **What to do if signals arrive in reverse order?**
  - Handling of signals with earlier timestamps arriving after later ones
  - Whether to process signals in timestamp order or arrival order
  - Rules for reordering or queueing out-of-order signals
  - Impact on position state and order creation when processing out-of-order

- **What to do if two signals for the same asset arrive simultaneously?**
  - Conflict resolution strategy: queue, merge, prioritize, or reject
  - Rules for determining which signal to process first
  - Handling of conflicting directions (buy vs sell) in simultaneous signals
  - Logic for combining compatible simultaneous signals (same direction)

- **Is FIFO ordering needed?**
  - Whether to enforce FIFO (First In First Out) processing
  - FIFO scope: per symbol (clarified - signals for same asset processed in order, different assets can process in parallel)
  - Rules for maintaining signal processing order
  - Impact of FIFO on concurrent signal processing and performance
  - Handling of signal dependencies and ordering requirements

- **Signal priority and queuing mechanisms:**
  - Rules for signal priority assignment (based on confidence, strategy_id, signal_type)
  - Queue management for signals waiting to be processed
  - Timeout handling for queued signals
  - Dead letter queue handling for signals that cannot be processed

### 6. Configuration Parameters

System requires configurable parameters for risk management and reliability, but specific values and structure are not defined. The following decisions must be made:

- **Risk limits configuration:**
  - Maximum exposure limits (total position size, per-asset limits)
  - Maximum order size limits (absolute and relative to balance)
  - Position size limits (maximum long/short position per asset)
  - Daily/weekly trading volume limits
  - Configuration format and storage (environment variables, config files, database)
  - Dynamic vs static risk limit configuration
  - Validation and enforcement of risk limits

- **Retry parameters:**
  - Retry strategy: exponential backoff with configurable retry limits (clarified)
  - Retry rules for different error types (network errors, rate limits, API errors)
  - Timeout configurations for API calls
  - Retry queue management and persistence
  - Maximum retry attempts and exponential backoff parameters (base delay, max delay, multiplier)
  - Handling of permanently failed operations

- **Other configuration parameters:**
  - Order execution timeouts
  - Signal processing timeouts
  - WebSocket reconnection parameters
  - Database connection pool settings
  - Logging and monitoring configuration
  - Feature flags for enabling/disabling specific behaviors

### 7. Order Cancellation Strategy

The specification mentions "modify or create new" orders, but cancellation rules are not defined. The following decisions must be made:

- **When to cancel existing orders?**
  - Criteria for cancelling existing orders when new signals arrive
  - Rules for determining which orders to cancel (by asset, by direction, by age, by status)
  - Handling of orders in different states (pending, partially filled) during cancellation
  - Time-based cancellation rules (e.g., cancel orders older than X minutes)

- **Should orders be cancelled on opposite signals?**
  - Rules for cancelling buy orders when sell signal arrives (and vice versa)
  - Handling of conflicting signals for the same asset
  - Logic for position reversal scenarios
  - Whether to cancel all opposite orders or only specific ones

- **What to do with pending limit orders when market signal arrives?**
  - Strategy for handling limit orders when market order signal is received
  - Whether to cancel limit orders before placing market order
  - Rules for combining limit and market orders for the same asset
  - Handling of partial fills in limit orders when market signal arrives

- **Should stale orders be cancelled automatically?**
  - Definition of "stale order" (time-based, status-based, or event-based)
  - Automatic cancellation mechanism and triggers
  - Configuration for stale order timeout periods
  - Handling of stale orders that are partially filled
  - Notification and logging for automatic cancellations

### 8. Query API and State Inspection

Order Manager typically should provide query capabilities, but specific endpoints and functionality are not defined. The following decisions must be made:

- **REST API for order list retrieval:**
  - Endpoint structure and URL patterns for querying orders
  - Filtering capabilities (by asset, status, date range, signal_id)
  - Pagination and sorting options
  - Response format and data structure
  - Authentication: API key authentication via X-API-Key header (clarified)

- **Endpoint for position state viewing:**
  - Endpoint for retrieving current position information
  - Data included in position response (size, average price, PnL, unrealized/realized)
  - Support for multiple positions (hedge-mode) vs single position (one-way mode)
  - Real-time vs cached position data

- **Health/live/readiness probes:**
  - Health check endpoint implementation
  - Liveness probe criteria (service is running and responsive)
  - Readiness probe criteria (service can accept requests, dependencies available)
  - Status indicators for dependencies (database, RabbitMQ, Bybit API, WebSocket gateway)
  - Response format and status codes

- **API for manual resynchronization:**
  - Endpoint for triggering manual order state synchronization
  - Scope of resynchronization (all orders, specific asset, specific order)
  - Response format indicating synchronization results
  - Rate limiting and access control for manual sync operations

### 9. Reconciliation and Recovery After Restart

Service may miss events during downtime, but recovery procedures are not defined. The following decisions must be made:

- **Which orders are considered "active" and require recovery?**
  - Definition of active order states (pending, partially filled, etc.)
  - Rules for identifying orders that need state verification
  - Handling of orders in different states during recovery
  - Time-based criteria for order recovery (e.g., orders created in last X hours)

- **Should service query state of all open orders on startup?**
  - Startup recovery procedure for order state synchronization
  - Scope of initial state query (all orders, recent orders, active orders only)
  - Performance considerations for large order history
  - Handling of startup synchronization failures

- **What to do if service was down and missed WebSocket events?**
  - Strategy for detecting missed events (time gaps, event sequence gaps)
  - Recovery mechanism for missed order execution events
  - Reconciliation with Bybit API to verify actual order states
  - Handling of events that occurred during downtime
  - Rules for updating order states based on missed events

- **Recovery procedures:**
  - Order of operations during service startup
  - Dependency checks before starting recovery
  - Error handling during recovery process
  - Logging and monitoring of recovery operations
  - Timeout and retry logic for recovery operations

### 10. Stop-Loss / Take-Profit Mechanism

The specification mentions PnL accounting and risk limits, but stop-loss/take-profit logic is not defined. The following decisions must be made:

- **Should Order Manager place stop-orders automatically?**
  - Decision on responsibility: Order Manager vs Model Service
  - Rules for automatic stop-loss order placement
  - Rules for automatic take-profit order placement
  - Configuration for stop-loss/take-profit levels (absolute, percentage, or dynamic)

- **What to do if position is open and market moves significantly?**
  - Handling of positions with unrealized losses exceeding thresholds
  - Automatic position closure rules based on PnL
  - Emergency position closure mechanisms
  - Integration with risk limit enforcement

- **Stop-order management:**
  - Lifecycle of stop-loss/take-profit orders (creation, modification, cancellation)
  - Handling of stop-orders when position changes
  - Rules for updating stop-orders based on position PnL
  - Integration with position management and order creation logic

- **Responsibility boundaries:**
  - Clear definition of what Order Manager handles vs what Model Service handles
  - Communication protocol for stop-loss/take-profit decisions
  - Handling of conflicting signals and stop-orders

### 11. Position History Storage

Position is defined as an entity, but storage strategy is not specified. The following decisions must be made:

- **Position storage strategy (clarified):**
  - Hybrid approach: store current position state in database with periodic snapshots, compute positions from orders for validation
  - Data model for position storage (current state, historical snapshots)
  - Storage of position history (periodic snapshots, event-based updates)
  - Relationship between stored positions and orders

- **Position computation for validation:**
  - Algorithm for computing position from order history (used for validation against stored state)
  - Performance considerations for position calculation
  - Validation frequency and triggers
  - Handling of order history gaps in position calculation

- **Position snapshot strategy:**
  - Frequency of position snapshots (periodic, event-based) - to be determined
  - Data included in position snapshots (size, price, PnL, timestamp)
  - Retention policy for position history
  - Query patterns for position history retrieval

- **Data consistency:**
  - Ensuring consistency between stored positions and computed positions (validation mechanism)
  - Handling of discrepancies between stored and computed positions
  - Reconciliation procedures for position data

### 12. Dry-Run Mode

Dry-run mode is important for safe deployment of new systems, but implementation is not defined. The following decisions must be made:

- **Dry-run mode implementation:**
  - Mechanism for enabling/disabling dry-run mode (configuration, feature flag)
  - Behavior in dry-run mode: accept signals but do not send orders to Bybit
  - Logging of what would have been executed in dry-run mode
  - Format and detail level of dry-run logs

- **Dry-run validation:**
  - Whether to perform full validation and processing in dry-run mode
  - Simulation of order creation, modification, cancellation logic
  - Simulation of position updates and state changes
  - Error simulation and handling in dry-run mode

- **Dry-run reporting:**
  - Format for reporting dry-run execution results
  - Metrics and statistics for dry-run operations
  - Comparison tools for dry-run vs live execution
  - Integration with monitoring and alerting systems

- **Transition from dry-run to live:**
  - Procedures for switching from dry-run to live mode
  - Validation checks before enabling live mode
  - Handling of state differences between dry-run and live environments

## Out of Scope

- Model training and signal generation (handled by model service)
- Market data aggregation and WebSocket management (handled by WebSocket gateway)
- User interface for manual order management
- Multi-account trading support
- Advanced order types beyond basic limit and market orders (unless required by Bybit API)
- Historical order analysis and reporting dashboards
- Order strategy optimization (focused on execution, not strategy)
