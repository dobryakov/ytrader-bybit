# Research: Order Manager Technology and Design Decisions

**Feature**: Order Manager Microservice  
**Date**: 2025-01-27  
**Purpose**: Resolve technical clarification items and document key design decisions from implementation plan

## Research Questions

### 1. Bybit REST API Client Approach

**Question**: Should we use a Bybit Python SDK (e.g., `pybit`) or implement direct REST API calls using `httpx`?

**Decision**: Direct REST API calls using `httpx` with custom authentication wrapper

**Rationale**:
- Consistent with existing architecture: `ws-gateway` uses direct WebSocket connections with custom auth logic rather than SDK
- Full control over request/response handling, retry logic, and error handling (critical for order execution)
- Better observability: can log all API interactions with full request/response bodies per constitution requirement
- No dependency on potentially outdated SDK versions or SDK-specific abstractions
- Easier to implement custom retry logic with exponential backoff for rate limits (429 errors)
- Better integration with async architecture using `httpx` async client
- Can implement request signing directly (similar to WebSocket auth pattern already in codebase)

**Alternatives Considered**:
- `pybit` (official Bybit Python SDK): Evaluated but adds abstraction layer that may limit control over retry logic, error handling, and observability requirements
- `requests` library: Sync-only, incompatible with async architecture

**Implementation Notes**:
- Use `httpx>=0.25.0` with async client for Bybit REST API calls
- Implement custom authentication wrapper that generates HMAC-SHA256 signatures for authenticated endpoints
- Follow Bybit REST API v5 documentation for endpoint structure and request formats
- Implement request signing per Bybit API authentication requirements (similar to WebSocket auth pattern in `ws-gateway/src/services/websocket/auth.py`)
- Support both testnet and mainnet environments via configuration

---

### 2. Order Type Selection Strategy (Market vs Limit)

**Question**: When should the service use MARKET vs LIMIT orders, and how to determine limit price?

**Decision**: Configurable rules-based approach with initial defaults

**Rationale**:
- Initial implementation uses conservative defaults to minimize slippage risk
- Rules can be refined based on production experience and market conditions
- Provides flexibility for future strategy optimization without code changes

**Decision Details**:
- **Market Orders**: Use when signal confidence > 0.9 OR signal has explicit urgency flag OR spread is < 0.1%
- **Limit Orders**: Default choice for most signals to control execution price
- **Limit Price Calculation**: Use signal snapshot price with offset:
  - Buy orders: limit price = snapshot_price - (spread * 0.5) [slightly below market]
  - Sell orders: limit price = snapshot_price + (spread * 0.5) [slightly above market]
- **Time-in-Force**: GTC (Good Till Cancel) for limit orders, IOC (Immediate or Cancel) for market orders when immediate execution needed
- **post_only flag**: Set to true for limit orders to ensure maker fees
- **reduce_only flag**: Set to true for sell orders when position exists and order would reduce position size

**Alternatives Considered**:
- Always use limit orders: Too conservative, may miss execution opportunities
- Always use market orders: Higher fees and slippage risk
- Complex ML-based selection: Too complex for initial implementation, can be added later

**Implementation Notes**:
- Configuration parameters for thresholds (confidence threshold, spread threshold) in `.env`
- Rules implemented in `order_type_selector.py` service module
- Can be extended with more sophisticated logic based on market conditions (volatility, orderbook depth)

---

### 3. Quantity Calculation from Amount

**Question**: How to convert quote currency amount (USDT) to base currency quantity with proper precision?

**Decision**: Use signal snapshot price for conversion, fetch tick size/lot size from Bybit API, round down to nearest valid quantity

**Rationale**:
- Signal snapshot price is most relevant as it represents market conditions when decision was made
- Rounding down (floor) prevents over-commitment of capital
- Dynamic precision lookup ensures compliance with exchange requirements

**Decision Details**:
- **Price Source**: Use `market_data_snapshot.price` from trading signal
- **Conversion Formula**: `quantity = amount / price`
- **Precision Handling**:
  - Fetch tick size and lot size from Bybit API (symbol info endpoint) or cache from initial sync
  - Round quantity down to nearest valid tick size
  - Apply lot size rounding (round down to nearest lot size multiple)
- **Minimum Quantity Check**: Reject signal if calculated quantity < Bybit minimum order size for symbol
- **Edge Cases**:
  - If price changed significantly (> 5% from snapshot), log warning but proceed with calculation
  - If snapshot price unavailable, fetch current market price from Bybit ticker API as fallback

**Alternatives Considered**:
- Use current market price: May differ from signal decision price, introduces execution price uncertainty
- Round up: Could exceed available balance
- Round to nearest: Could slightly exceed or fall short, less predictable

**Implementation Notes**:
- Cache symbol info (tick size, lot size, min quantity) to avoid repeated API calls
- Implement in `quantity_calculator.py` service module
- Log all quantity calculations with before/after values for auditing

---

### 4. Signal-to-Order Relationship Logic

**Question**: When should one signal create multiple orders, and how to track relationships?

**Decision**: Initial implementation: one signal → one order (1:1). Support for 1:N relationships via configuration flags.

**Rationale**:
- Simplifies initial implementation and reduces complexity
- Most trading signals will map 1:1 to orders
- 1:N relationships (order splitting) can be added as enhancement after core functionality is proven

**Decision Details**:
- **Default Behavior**: One signal creates one order (1:1 mapping)
- **Order Splitting**: Enable via configuration flag `ENABLE_ORDER_SPLITTING=true`
  - When enabled, split orders when amount exceeds `MAX_SINGLE_ORDER_SIZE` configuration
  - Split into N orders of approximately equal size (respecting min order size)
- **Data Model**: 
  - `signal_order_relationships` table stores mappings (signal_id, order_id, allocation_amount)
  - Supports tracking of 1:1, 1:N, and future N:1 (scale-in) relationships
- **Scale-in Logic**: Future enhancement - multiple signals for same asset/direction can accumulate into position
- **Query Patterns**: Index on signal_id for fast lookup of orders by signal

**Alternatives Considered**:
- Always split large orders: Too complex for initial implementation, unnecessary for most cases
- No relationship tracking: Required for audit trail and position building features

**Implementation Notes**:
- Start with simple 1:1 mapping, add splitting logic incrementally
- Design data model to support future enhancements (N:1 scale-in, partial allocations)
- Configuration parameters: `ENABLE_ORDER_SPLITTING`, `MAX_SINGLE_ORDER_SIZE`

---

### 5. Position Management Rules

**Question**: How should the service account for current position state when processing signals?

**Decision**: Hybrid approach - use stored position state for decisions, validate against computed position from orders

**Rationale**:
- Stored position enables fast decision-making without computing from order history
- Validation ensures data consistency and detects discrepancies
- Aligns with specified hybrid storage strategy

**Decision Details**:
- **Position State Source**: Query stored position from database (current position table)
- **Position Validation**: Periodically compute position from order history and compare with stored state
- **Position Usage in Order Decisions**:
  - Check position size before creating orders (prevent over-exposure)
  - For sell signals: check available position size (cannot sell more than owned)
  - For buy signals: consider existing position when determining order size (scale-in logic)
- **Bybit Hedging Mode**: Support both one-way and hedge-mode
  - One-way: single position per asset (can be long or short, not both)
  - Hedge-mode: separate long and short positions per asset (track both separately)
- **Average Price**: Use stored average entry price for PnL calculations
- **PnL Consideration**: Log PnL but do not use in order creation logic initially (future enhancement for stop-loss)

**Alternatives Considered**:
- Always compute from orders: Too slow for real-time order decisions
- Always use stored state: Risk of inconsistency if updates fail

**Implementation Notes**:
- Implement position query in `position_manager.py`
- Position validation runs on service startup and periodically (configurable interval)
- Store position updates when orders are filled (via WebSocket events)

---

### 6. Signal Processing Order and Priority

**Question**: How to handle out-of-order signals and simultaneous signals for same asset?

**Decision**: Per-symbol FIFO queue with async processing for different assets

**Rationale**:
- Per-symbol FIFO ensures consistent order execution for same asset (clarified requirement)
- Parallel processing for different assets maximizes throughput
- Simple queue-based approach is reliable and debuggable

**Decision Details**:
- **FIFO Scope**: Per symbol (asset) - signals for same asset processed sequentially, different assets process in parallel
- **Queue Implementation**: In-memory asyncio queue per symbol
- **Out-of-Order Handling**: Process signals in queue order (arrival order), not timestamp order
  - Rationale: Timestamp may be unreliable, arrival order is deterministic
  - Log warning if timestamp significantly differs from arrival time
- **Simultaneous Signals**: Queue handles naturally - second signal waits in queue
- **Conflict Resolution**: 
  - If two signals for same asset arrive simultaneously: both queued, processed sequentially
  - If signals have opposite directions: process in order, cancel opposite orders if needed (see cancellation strategy)
- **Priority**: No priority initially - FIFO order. Can add priority field later if needed

**Alternatives Considered**:
- Global FIFO queue: Too restrictive, blocks processing of unrelated assets
- Timestamp-based ordering: Timestamps may be unreliable or inaccurate
- Priority queues: Unnecessary complexity for initial implementation

**Implementation Notes**:
- Implement signal queue manager in `signal_processor.py`
- Use asyncio queues for thread-safe signal processing
- Monitor queue depth for observability (alert if queue grows too large)

---

### 7. Configuration Parameters Structure

**Question**: What configuration parameters are needed and how should they be structured?

**Decision**: Environment variables in `.env` with pydantic-settings for validation, organized by functional area

**Rationale**:
- Consistent with existing services (ws-gateway, model-service) using `.env` and pydantic-settings
- Type-safe configuration with validation
- Easy to override for different environments (testnet, mainnet, dry-run)

**Configuration Categories**:
1. **Service Configuration**: Port, API keys, logging level, service name
2. **Bybit API Configuration**: API key, secret, environment (testnet/mainnet)
3. **Database Configuration**: PostgreSQL connection (shared with other services)
4. **RabbitMQ Configuration**: Connection details (shared with other services)
5. **Order Execution Configuration**:
   - `ENABLE_DRY_RUN`: Enable dry-run mode (boolean)
   - `MAX_SINGLE_ORDER_SIZE`: Maximum order size for single order (float, USDT)
   - `ENABLE_ORDER_SPLITTING`: Enable order splitting for large amounts (boolean)
   - `ORDER_EXECUTION_TIMEOUT`: Timeout for order creation API calls (seconds)
6. **Risk Limits Configuration**:
   - `MAX_POSITION_SIZE`: Maximum position size per asset (float, base currency)
   - `MAX_EXPOSURE`: Maximum total exposure across all positions (float, USDT)
   - `MAX_ORDER_SIZE_RATIO`: Maximum order size as ratio of available balance (float, 0.0-1.0)
7. **Retry Configuration**:
   - `BYBIT_API_RETRY_MAX_ATTEMPTS`: Maximum retry attempts (integer, default 3)
   - `BYBIT_API_RETRY_BASE_DELAY`: Base delay for exponential backoff (seconds, default 1.0)
   - `BYBIT_API_RETRY_MAX_DELAY`: Maximum delay between retries (seconds, default 30.0)
   - `BYBIT_API_RETRY_MULTIPLIER`: Exponential backoff multiplier (float, default 2.0)
8. **Order Type Selection Configuration**:
   - `MARKET_ORDER_CONFIDENCE_THRESHOLD`: Confidence threshold for market orders (float, 0.0-1.0, default 0.9)
   - `MARKET_ORDER_SPREAD_THRESHOLD`: Spread threshold for market orders (percentage, default 0.1)
   - `LIMIT_ORDER_PRICE_OFFSET_RATIO`: Price offset ratio for limit orders (float, default 0.5)
9. **Position Management Configuration**:
   - `POSITION_SNAPSHOT_INTERVAL`: Interval for position snapshots (seconds, default 300)
   - `POSITION_VALIDATION_INTERVAL`: Interval for position validation (seconds, default 3600)
10. **WebSocket Configuration**:
    - `WS_GATEWAY_HOST`: WebSocket gateway service hostname
    - `WS_GATEWAY_PORT`: WebSocket gateway service port
    - `WS_GATEWAY_API_KEY`: API key for WebSocket gateway subscription management

**Implementation Notes**:
- Use `pydantic-settings>=2.0.0` for configuration management
- Validate all configuration values on startup
- Provide sensible defaults for optional parameters
- Document all parameters in `env.example`

---

### 8. Order Cancellation Strategy

**Question**: When should existing orders be cancelled, and what rules apply?

**Decision**: Cancel existing orders when new signal for same asset arrives, with configurable behavior

**Rationale**:
- Prevents accumulation of stale orders
- Allows rapid response to new signals
- Configurable behavior provides flexibility for different trading strategies

**Decision Details**:
- **Default Behavior**: When new signal arrives for asset with existing pending orders:
  - Cancel all pending orders for same asset (regardless of direction)
  - Create new order based on signal
- **Configuration Options**:
  - `CANCEL_OPPOSITE_ORDERS_ONLY`: If true, only cancel orders with opposite direction (buy vs sell)
  - `CANCEL_STALE_ORDER_TIMEOUT`: Automatically cancel orders older than X seconds (default: 3600 seconds)
- **Limit Order Cancellation**: When market order signal arrives, cancel all pending limit orders for same asset
- **Partial Fill Handling**: Do not cancel partially filled orders unless explicitly configured
- **Cancellation Logging**: Log all cancellations with reason (new signal, stale, opposite direction)

**Alternatives Considered**:
- Never cancel automatically: Too conservative, leads to order accumulation
- Always cancel on new signal: May be too aggressive for some strategies

**Implementation Notes**:
- Implement cancellation logic in `signal_processor.py` before creating new orders
- Use Bybit REST API cancel endpoint with proper error handling
- Track cancellation reasons for analytics and debugging

---

### 9. Reconciliation and Recovery Procedures

**Question**: How should the service recover order state after restart or missed WebSocket events?

**Decision**: Query all active orders on startup, validate against database, sync discrepancies

**Rationale**:
- Ensures service starts with accurate order state
- Detects and corrects missed events from downtime
- Provides safety mechanism for state consistency

**Decision Details**:
- **Startup Reconciliation**:
  - Query Bybit REST API for all active orders (status: New, PartiallyFilled)
  - Compare with database orders (status: pending, partially_filled)
  - Update database with actual Bybit state
  - Log discrepancies for investigation
- **Active Order Definition**: Orders with status New, PartiallyFilled (not Filled, Cancelled, Rejected)
- **Missed Event Detection**: Compare order states - if database shows pending but Bybit shows filled, update database
- **Recovery Scope**: All active orders (not just recent) to ensure complete state
- **Performance**: Query in batches if order count is large
- **Error Handling**: If reconciliation fails, log error but allow service to start (manual sync available via API)

**Alternatives Considered**:
- Query only recent orders: May miss older active orders
- No startup reconciliation: Risk of stale state

**Implementation Notes**:
- Implement reconciliation in `order_state_sync.py`
- Run on service startup after database connection established
- Provide manual sync endpoint for on-demand reconciliation
- Log reconciliation results for monitoring

---

### 10. Stop-Loss / Take-Profit Mechanism

**Question**: Should Order Manager automatically place stop-loss/take-profit orders?

**Decision**: Initial implementation: Log PnL and positions with unrealized loss, but do not automatically place stop-orders. Defer to Model Service for stop-loss decisions.

**Rationale**:
- Stop-loss logic is strategy-related and should be handled by Model Service
- Order Manager focuses on execution, not strategy decisions
- Simplifies initial implementation
- Model Service can send stop-loss signals via trading signal queue if needed

**Decision Details**:
- **Position Monitoring**: Track unrealized PnL for all positions
- **Logging**: Log warning when unrealized loss exceeds threshold (configurable, e.g., 10% of position value)
- **Notification**: Publish event to queue with position PnL information for Model Service
- **Future Enhancement**: Can add automatic stop-order placement if needed, but requires clear requirements

**Alternatives Considered**:
- Automatic stop-order placement: Adds complexity, blurs responsibility boundaries
- No PnL tracking: Required for risk management and reporting

**Implementation Notes**:
- Implement PnL calculation in `position_manager.py`
- Publish position PnL events to queue for Model Service consumption
- Configuration: `UNREALIZED_LOSS_WARNING_THRESHOLD` (percentage)

---

### 11. Position Storage Strategy

**Question**: How should positions be stored and validated?

**Decision**: Hybrid approach - store current position state with periodic snapshots, compute from orders for validation

**Rationale**:
- Aligns with specification requirement (FR-005.12, FR-026)
- Fast queries for order decisions (stored state)
- Data integrity through validation (computed state)

**Decision Details**:
- **Current Position Table**: Store current position state per asset
  - Columns: asset, size (positive=long, negative=short), average_entry_price, unrealized_pnl, last_updated
  - Updated when orders are filled (via WebSocket events)
- **Position Snapshot Table**: Periodic snapshots for historical tracking
  - Snapshot interval: configurable (default 5 minutes)
  - Store same fields as current position plus snapshot timestamp
- **Position Computation**: Compute position from order history for validation
  - Algorithm: Sum filled quantities by side (buy = +quantity, sell = -quantity)
  - Average price: Weighted average of filled order prices
  - Run validation periodically (default: hourly) and on startup
- **Discrepancy Handling**: If computed position differs from stored position, log error and update stored position
- **Bybit Hedge Mode**: Support both one-way and hedge-mode
  - One-way: single row per asset (size can be positive or negative)
  - Hedge-mode: separate rows for long and short positions per asset

**Alternatives Considered**:
- Always compute from orders: Too slow for real-time queries
- Only stored state: Risk of inconsistency

**Implementation Notes**:
- Position updates in `position_manager.py`
- Snapshot job runs periodically (background task)
- Validation runs on schedule and can be triggered manually

---

### 12. Dry-Run Mode Implementation

**Question**: How should dry-run mode work?

**Decision**: Process all logic normally but skip Bybit API calls, log simulated operations

**Rationale**:
- Enables testing of order logic without risking real funds
- Comprehensive logging allows verification of behavior
- Simple implementation via configuration flag

**Decision Details**:
- **Configuration**: `ENABLE_DRY_RUN=true` in `.env`
- **Behavior**:
  - Accept and validate trading signals normally
  - Process all business logic (order type selection, quantity calculation, risk checks)
  - Skip actual Bybit REST API calls (order creation, modification, cancellation)
  - Log all operations with `[DRY-RUN]` prefix
  - Store simulated orders in database with special status `dry_run`
- **WebSocket Events**: In dry-run mode, do not subscribe to real order events (would receive events from previous live orders)
- **Validation**: Perform full validation as if real orders were being created
- **Logging Format**: Structured logs with dry-run flag for easy filtering

**Alternatives Considered**:
- Separate dry-run database: Unnecessary complexity, can filter by status
- Skip all processing: Would not test business logic

**Implementation Notes**:
- Check dry-run flag in `order_executor.py` before making Bybit API calls
- Store dry-run orders in same database with status flag
- Log all simulated operations with full detail

---

## Summary of Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | Async/await support, ecosystem maturity |
| REST API Framework | FastAPI | 0.104+ | Async support, OpenAPI generation |
| HTTP Client | httpx | 0.25+ | Async HTTP client for Bybit REST API |
| Message Queue Client | aio-pika | 9.0+ | Async RabbitMQ client |
| Database Driver | asyncpg | 0.29+ | Fastest async PostgreSQL driver |
| Configuration | pydantic-settings | 2.0+ | Type-safe configuration management |
| Logging | structlog | 23.2+ | Structured logging with trace IDs |
| Testing | pytest + pytest-asyncio | Latest | Async test support |

## Dependencies Summary

**Core Dependencies**:
- `fastapi>=0.104.0` - REST API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `httpx>=0.25.0` - Async HTTP client for Bybit REST API
- `aio-pika>=9.0.0` - RabbitMQ async client
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `pydantic>=2.0.0` - Data validation
- `pydantic-settings>=2.0.0` - Configuration management
- `structlog>=23.2.0` - Structured logging

**Additional Dependencies**:
- `cryptography>=41.0.0` - For HMAC-SHA256 signature generation (Bybit API auth)

