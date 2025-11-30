# Feature Specification: Position Management Service

**Use as a reference** the source document `docs/position-manager.md` for all unclarified questions.

**Feature Branch**: `001-position-management`  
**Created**: 2025-01-15  
**Status**: Draft  
**Input**: User description: "Create Position Manager Service for centralized portfolio position management"

## Clarifications

### Session 2025-01-15

- Q: What is the position identity/uniqueness rule? → A: Composite key (asset, mode) - one position per asset per trading mode
- Q: What is the historical snapshot data retention period? → A: 1 year retention
- Q: How should conflicts between WebSocket avgPrice and Order Manager calculated average_entry_price be resolved? → A: Use WebSocket avgPrice if present and difference exceeds threshold (e.g., 0.1%), otherwise keep existing value
- Q: What happens when position size becomes zero (closed position)? → A: Keep position with size=0, mark as closed (add closed_at timestamp)
- Q: What should portfolio metrics return when no positions exist? → A: Return zero values for all metrics (exposure=0, pnl=0, count=0), empty positions array, valid HTTP 200 response
- Q: How should concurrent updates to the same position be handled? → A: Optimistic locking with version field (check version before update, retry on conflict)
- Q: What is the source of current_price for portfolio calculations? → A: Use markPrice from WebSocket position events (store latest markPrice per asset)
- Q: How to handle missing or stale markPrice? → A: Query external price API when markPrice missing/stale
- Q: What rate limiting strategy should be used for REST API? → A: Per-API-Key rate limits with different tiers (e.g., 100 req/min for Model Service, 1000 req/min for UI)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Real-Time Position Tracking (Priority: P1)

Trading system components need to access current position data and portfolio metrics in real-time to make informed trading decisions, manage risk, and generate trading signals.

**Why this priority**: This is the core functionality - without real-time position data, the trading system cannot operate safely. Risk management and signal generation depend on accurate, up-to-date position information.

**Independent Test**: Can be fully tested by querying position data after simulated position updates and verifying that the returned data reflects the current state accurately. Delivers immediate value by providing a single source of truth for position information.

**Acceptance Scenarios**:

1. **Given** a position exists for an asset, **When** a trading component requests current position data, **Then** the system returns accurate position details including size, entry price, and profit/loss metrics
2. **Given** multiple positions exist across different assets, **When** a component requests portfolio-level metrics, **Then** the system returns aggregated metrics including total exposure, total profit/loss, and position counts
3. **Given** a position is updated from an order execution, **When** a component queries the position immediately after, **Then** the position data reflects the latest changes
4. **Given** a position is updated from a market data event, **When** a component queries the position, **Then** the profit/loss metrics reflect current market prices

---

### User Story 2 - Automatic Position Updates from Multiple Sources (Priority: P1)

The system must automatically update positions when orders are executed or when market data indicates position changes, ensuring data consistency across different information sources.

**Why this priority**: Manual position updates are error-prone and cannot scale. The system must automatically stay synchronized with order executions and market data to maintain accuracy.

**Independent Test**: Can be fully tested by sending position update events from different sources (order executions, market data) and verifying that positions are updated correctly and conflicts are resolved appropriately. Delivers value by eliminating manual data entry and reducing errors.

**Acceptance Scenarios**:

1. **Given** an order is executed, **When** the execution event is received, **Then** the corresponding position is updated with new size and recalculated average entry price
2. **Given** market data indicates a position change, **When** the market data event is received, **Then** the position profit/loss metrics are updated to reflect current market conditions
3. **Given** conflicting position data arrives from different sources, **When** both updates are processed, **Then** the system resolves conflicts using defined priority rules (WebSocket avgPrice used if difference exceeds threshold, Order Manager for size, WebSocket for PnL) and maintains data consistency
4. **Given** a position update fails validation, **When** the discrepancy is detected, **Then** the system logs the issue and optionally triggers corrective action

---

### User Story 3 - Portfolio Risk Management Support (Priority: P2)

Risk management components need portfolio-level exposure and profit/loss metrics to enforce trading limits and prevent excessive risk exposure.

**Why this priority**: Risk management is critical for protecting capital. Portfolio-level metrics enable enforcement of limits that individual position checks cannot provide.

**Independent Test**: Can be fully tested by querying portfolio metrics and verifying that risk management components can use these metrics to make limit-checking decisions. Delivers value by enabling portfolio-wide risk controls.

**Acceptance Scenarios**:

1. **Given** multiple positions exist, **When** a risk management component requests total portfolio exposure, **Then** the system returns the sum of all position exposures in a standardized currency
2. **Given** a new order is being evaluated, **When** risk management checks if adding this order would exceed portfolio limits, **Then** the system provides current portfolio metrics to support the decision
3. **Given** portfolio exposure exceeds configured limits, **When** risk management queries portfolio metrics, **Then** the system includes indicators that limits are exceeded

---

### User Story 4 - Historical Position Tracking (Priority: P2)

Analytics and model training systems need historical position snapshots to analyze trading performance, train machine learning models, and reconstruct past portfolio states.

**Why this priority**: Historical data enables learning and improvement. Without position history, the system cannot analyze what worked, train better models, or audit trading decisions.

**Independent Test**: Can be fully tested by creating position snapshots and then querying historical snapshots to verify that past states can be accurately reconstructed. Delivers value by enabling performance analysis and model training.

**Acceptance Scenarios**:

1. **Given** positions exist, **When** a snapshot is created, **Then** the system captures the complete state of all positions at that moment
2. **Given** historical snapshots exist, **When** an analytics system requests position state for a specific time, **Then** the system returns the snapshot data that represents positions at that time
3. **Given** snapshots are created periodically, **When** a system queries snapshot history, **Then** the system returns snapshots in chronological order with accurate timestamps

---

### User Story 5 - Position Data Validation and Synchronization (Priority: P3)

The system must periodically validate position data against authoritative sources and automatically correct discrepancies to maintain data integrity.

**Why this priority**: Data integrity is essential for reliable trading decisions. Automatic validation and correction reduce manual intervention and prevent errors from accumulating.

**Independent Test**: Can be fully tested by introducing intentional discrepancies and verifying that the validation process detects and corrects them. Delivers value by maintaining data quality with minimal manual effort.

**Acceptance Scenarios**:

1. **Given** a position exists in the system, **When** validation runs and detects a discrepancy with an external source, **Then** the system logs the discrepancy and optionally corrects it automatically
2. **Given** validation is configured to run periodically, **When** the scheduled time arrives, **Then** the system validates all positions and reports any issues found
3. **Given** a position has significant discrepancies, **When** validation detects this, **Then** the system flags the position for manual review or takes corrective action based on configuration

---

### Edge Cases

- What happens when position updates arrive out of order from different sources?
- How does the system handle position data when the external data source is temporarily unavailable?
- What happens when a position update contains invalid or missing required fields?
- How does the system handle simultaneous updates to the same position from multiple sources?
- What happens when portfolio metrics are requested but no positions exist? → System returns HTTP 200 with zero values for all metrics (exposure=0, pnl=0, count=0) and empty positions array
- How does the system handle positions that are closed (size becomes zero)? → Position is retained with size=0 and marked with closed_at timestamp; position remains queryable for historical purposes
- What happens when position validation detects discrepancies that cannot be automatically resolved?
- How does the system handle requests for position data that doesn't exist?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide real-time access to current position data for any asset and trading mode combination (position identity is composite key: asset + mode)
- **FR-002**: System MUST automatically update positions when order execution events are received
- **FR-003**: System MUST automatically update position profit/loss metrics when market data events are received
- **FR-004**: System MUST calculate and provide portfolio-level metrics including total exposure, total profit/loss, and position counts
- **FR-005**: System MUST resolve conflicts when position data from different sources conflicts, using defined priority rules: WebSocket avgPrice is used if present and difference from existing value exceeds configured threshold (default 0.1%), otherwise existing value is retained; Order Manager is source of truth for position size; WebSocket is source of truth for PnL metrics
- **FR-006**: System MUST support querying positions by asset, trading mode, and size filters
- **FR-007**: System MUST provide portfolio metrics aggregated across all positions or filtered by specific criteria
- **FR-008**: System MUST create periodic snapshots of all positions for historical tracking
- **FR-009**: System MUST allow querying historical position snapshots by time range
- **FR-010**: System MUST validate position data against authoritative sources and detect discrepancies
- **FR-011**: System MUST automatically correct position discrepancies when validation rules allow
- **FR-012**: System MUST provide position data in a format suitable for risk management limit checking
- **FR-013**: System MUST provide position data in a format suitable for machine learning model training, including pre-calculated ML features (unrealized_pnl_pct, time_held_minutes, position_size_norm)
- **FR-014**: System MUST maintain data consistency when processing concurrent position updates using optimistic locking with version field (check version before update, retry on conflict)
- **FR-021**: System MUST provide data for risk management rules in Model Service (Take Profit and Position Size Limit rules)
- **FR-015**: System MUST log all position updates and validation results for audit purposes
- **FR-016**: System MUST handle position updates that arrive out of chronological order
- **FR-017**: System MUST provide health status indicating system availability and data freshness
- **FR-018**: System MUST support querying individual position details including calculated metrics like profit percentage and position size relative to portfolio
- **FR-019**: System MUST update position average entry price when authoritative market data provides this information
- **FR-020**: System MUST validate position size against market data without directly overwriting size from non-authoritative sources

### Key Entities *(include if feature involves data)*

- **Position**: Represents a trading position (open or closed) for a specific asset and trading mode combination. Position identity is defined by composite key (asset, mode), meaning one position exists per asset per trading mode. Key attributes include asset identifier, trading mode, position size (positive for long, negative for short, zero when closed), average entry price, current unrealized profit/loss, realized profit/loss, last update timestamp, and closed_at timestamp (when position is closed). Closed positions are retained with size=0 for historical tracking.

  **ML Features** (calculated and included in API responses):
  - `unrealized_pnl_pct`: Percentage of unrealized PnL. Formula: `(unrealized_pnl / (abs(size) * average_entry_price)) * 100` (if entry price exists), or `(unrealized_pnl / total_exposure) * 100` (relative to portfolio exposure)
  - `time_held_minutes`: Time position held in minutes. Formula: `(current_timestamp - last_updated) / 60` (or time since position creation)
  - `position_size_norm`: Normalized position size. Formula: `abs(size * current_price) / total_exposure` (relative to portfolio exposure), or `abs(size * current_price) / balance` (relative to balance)
  
  Relationships: belongs to portfolio, has historical snapshots.

- **Portfolio**: Represents the aggregate of all positions. Key attributes include total exposure (sum of all position values), total unrealized profit/loss, total realized profit/loss, number of open positions, and distribution metrics by asset and direction. Relationships: aggregates multiple positions.

- **Position Snapshot**: Represents a historical record of position state at a specific point in time. Key attributes include complete position data at snapshot time, snapshot timestamp, and position identifier. Relationships: belongs to a position, used for historical reconstruction.

- **Portfolio Metrics**: Represents calculated aggregate metrics for the portfolio. Key attributes include exposure totals, profit/loss totals, position counts, and distribution breakdowns. Relationships: derived from current positions.

  **Price Source for Calculations**: Use `markPrice` from WebSocket position events (field `markPrice` in event payload). When updating position from WebSocket event, save `markPrice` to position's `current_price` field for use in portfolio metrics calculations.

  **Handling Missing or Stale Prices**:
  - If `markPrice` missing in WebSocket event: query current price via external API (Bybit REST API) and save to `current_price`
  - If `current_price` is stale (time since last update > `POSITION_MANAGER_PRICE_STALENESS_THRESHOLD`): query current price via external API before calculating portfolio metrics
  - Log all cases of using external API for price retrieval (for monitoring)

  **Portfolio Metrics Formulas**:
  - **Total Exposure (USDT)**: `SUM(ABS(position.size) * current_price)` for all positions, where `current_price` is latest `markPrice` from WebSocket event
  - **Total Unrealized PnL (USDT)**: `SUM(position.unrealized_pnl)` for all positions
  - **Total Realized PnL (USDT)**: `SUM(position.realized_pnl)` for all positions
  - **Portfolio Value (USDT)**: `SUM(position.size * current_price)` for all positions
  - **Open Positions Count**: Count of positions with `size != 0`

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Position data queries return results in under 500 milliseconds for 95% of requests
- **SC-002**: Portfolio metrics calculations complete in under 1 second for portfolios with up to 100 positions
- **SC-003**: Position updates from order executions are reflected in queries within 2 seconds of event receipt
- **SC-004**: Position updates from market data are reflected in queries within 5 seconds of event receipt
- **SC-005**: System maintains data consistency with zero data loss when processing up to 1000 concurrent position updates per minute
- **SC-006**: Position validation detects 100% of introduced discrepancies within the configured validation interval
- **SC-007**: Historical snapshots can be queried and retrieved for any time point within the 1-year retention period in under 2 seconds
- **SC-008**: Portfolio metrics are available for query 99.9% of the time during normal operations
- **SC-009**: Position data conflicts are resolved automatically without manual intervention in 95% of cases
- **SC-010**: System supports querying positions and portfolio metrics for portfolios containing up to 500 distinct assets simultaneously

## Assumptions

- Position updates will arrive from multiple sources (order execution system and market data feed) and may occasionally conflict
- Portfolio metrics need to be calculated in real-time but can tolerate slight delays (seconds) for non-critical queries
- Historical snapshots are created periodically (e.g., hourly) rather than on every position change
- Position validation runs on a scheduled interval rather than continuously
- The system will handle positions for multiple trading modes (one-way, hedge) with potentially different data structures
- Position size can be positive (long), negative (short), or zero (closed); closed positions are retained with size=0 and closed_at timestamp for historical tracking
- Profit/loss calculations require current market prices which may be provided in update events
- The system must support filtering and aggregation queries for analytics and reporting purposes
- Historical snapshots are retained for 1 year, after which they are automatically deleted or archived

## Dependencies

- Order execution system must publish order execution events in a standardized format
- Market data system must publish position update events in a standardized format
- Risk management components require portfolio metrics for limit enforcement
- Analytics and model training systems require historical position data
- Monitoring systems require position and portfolio data for dashboards and alerts

## Technical Stack & Architecture

### Technology Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: PostgreSQL (shared database, tables `positions`, `position_snapshots`)
- **Message Queue**: RabbitMQ (consuming position update events)
- **Logging**: structlog with trace IDs
- **Configuration**: pydantic-settings
- **Containerization**: Docker, docker-compose

### Service Configuration

- **REST API Port**: 4800 (following pattern: ws-gateway: 4400, order-manager: 4600)
- **Service Name**: `position-manager`
- **Environment Variable Prefix**: `POSITION_MANAGER_*`

### Project Structure

```text
position-manager/
├── Dockerfile
├── docker-compose.yml (add to main)
├── requirements.txt
├── README.md
├── env.example
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── settings.py
│   │   ├── database.py
│   │   ├── rabbitmq.py
│   │   └── logging.py
│   ├── models/
│   │   ├── position.py
│   │   ├── portfolio.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── position_manager.py
│   │   ├── portfolio_manager.py
│   │   ├── position_event_consumer.py
│   │   └── position_sync.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── positions.py
│   │   │   ├── portfolio.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── logging.py
│   ├── consumers/
│   │   ├── order_position_consumer.py
│   │   └── websocket_position_consumer.py
│   ├── publishers/
│   │   └── position_event_publisher.py
│   ├── tasks/
│   │   ├── position_snapshot_task.py
│   │   └── position_validation_task.py
│   └── utils/
│       └── tracing.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### Database Schema

**Existing Tables** (shared database):
- `positions` - current positions (may need additional fields: `current_price` for latest markPrice, `version` for optimistic locking)
- `position_snapshots` - position snapshots

**Optional Table** (for metrics caching):
```sql
CREATE TABLE portfolio_metrics_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    total_exposure_usdt DECIMAL(20, 8) NOT NULL,
    total_unrealized_pnl_usdt DECIMAL(20, 8) NOT NULL,
    total_realized_pnl_usdt DECIMAL(20, 8) NOT NULL,
    portfolio_value_usdt DECIMAL(20, 8) NOT NULL,
    open_positions_count INTEGER NOT NULL,
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_portfolio_metrics_expires_at ON portfolio_metrics_cache(expires_at);
```

**Required Indexes**:
- Index on `positions.asset`
- Index on `positions.mode`
- Composite index on `positions(asset, mode)` for position identity lookups
- Index on `positions.current_price` for portfolio calculations (if `current_price` field is added)
- Index on `positions.version` for optimistic locking (if `version` field is added)

**Position Table Additional Fields** (may need to be added to existing table):
- `current_price`: Latest `markPrice` from WebSocket events, used for portfolio metrics calculations. Updated when position is updated from WebSocket event.
- `version`: Version field for optimistic locking (increment on each update, check before update to handle concurrent modifications). Initial value: 1, increment on each successful update.

## Configuration

### Environment Variables

```bash
# Position Manager Service Configuration
POSITION_MANAGER_PORT=4800
POSITION_MANAGER_API_KEY=<api_key>
POSITION_MANAGER_LOG_LEVEL=INFO
POSITION_MANAGER_SERVICE_NAME=position-manager

# Database (uses shared database)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=<password>

# RabbitMQ (uses shared RabbitMQ)
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Position Management
POSITION_MANAGER_SNAPSHOT_INTERVAL=3600  # seconds
POSITION_MANAGER_VALIDATION_INTERVAL=1800  # seconds
POSITION_MANAGER_METRICS_CACHE_TTL=10  # seconds

# Position Update Strategy
POSITION_MANAGER_USE_WS_AVG_PRICE=true  # Use avgPrice from WebSocket events for updating average_entry_price
POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001  # Threshold for updating average_entry_price (0.1% = 0.001)
POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD=0.0001  # Threshold for triggering position validation on size discrepancy
POSITION_MANAGER_PRICE_STALENESS_THRESHOLD=300  # Seconds - threshold for considering current_price stale and querying external API

# Integration
ORDER_MANAGER_URL=http://order-manager:4600
WS_GATEWAY_URL=http://ws-gateway:4400
```

## REST API Specification

### Authentication

All endpoints (except `/health`) require API Key authentication:
- Header: `X-API-Key: <api_key>`
- Configuration: `POSITION_MANAGER_API_KEY`

### Endpoints for Individual Positions

#### GET /api/v1/positions

Get list of all positions with optional filtering.

**Query Parameters**:
- `asset` (optional): Filter by trading pair (e.g., BTCUSDT)
- `mode` (optional): Filter by trading mode (one-way, hedge)
- `size_min` (optional): Minimum position size
- `size_max` (optional): Maximum position size

**Response**:
```json
{
  "positions": [
    {
      "id": "uuid",
      "asset": "BTCUSDT",
      "size": "1.5",
      "average_entry_price": "50000.00",
      "unrealized_pnl": "150.00",
      "realized_pnl": "50.00",
      "mode": "one-way",
      "long_size": null,
      "short_size": null,
      "unrealized_pnl_pct": "0.30",
      "time_held_minutes": 120,
      "position_size_norm": "0.15",
      "last_updated": "2025-01-15T10:00:00Z",
      "last_snapshot_at": "2025-01-15T09:00:00Z"
    }
  ],
  "count": 1
}
```

**Note**: Fields `unrealized_pnl_pct`, `time_held_minutes`, `position_size_norm` are pre-calculated ML features ready for use in ML models.

#### GET /api/v1/positions/{asset}

Get position for a specific asset.

**Path Parameters**:
- `asset`: Trading pair (e.g., BTCUSDT)

**Query Parameters**:
- `mode` (optional, default: "one-way"): Trading mode

**Response**: Position object (same format as element in list above)

#### POST /api/v1/positions/{asset}/validate

Manually trigger position validation.

**Path Parameters**:
- `asset`: Trading pair

**Query Parameters**:
- `mode` (optional, default: "one-way"): Trading mode
- `fix_discrepancies` (optional, default: true): Automatically fix discrepancies

**Response**:
```json
{
  "is_valid": true,
  "error_message": null,
  "updated_position": null
}
```

#### POST /api/v1/positions/{asset}/snapshot

Manually create position snapshot.

**Path Parameters**:
- `asset`: Trading pair

**Query Parameters**:
- `mode` (optional, default: "one-way"): Trading mode

**Response**: Position snapshot object

#### GET /api/v1/positions/{asset}/snapshots

Get position snapshot history.

**Query Parameters**:
- `limit` (optional, default: 100): Maximum number of snapshots
- `offset` (optional, default: 0): Offset for pagination

**Response**: List of position snapshots

### Endpoints for Portfolio

#### GET /api/v1/portfolio

Get aggregated portfolio metrics.

**Query Parameters**:
- `include_positions` (optional, default: false): Include list of positions in response
- `asset` (optional): Calculate metrics only for specified asset

**Response**:
```json
{
  "total_exposure_usdt": "10000.00",
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "portfolio_value_usdt": "10200.00",
  "open_positions_count": 3,
  "long_positions_count": 2,
  "short_positions_count": 1,
  "net_exposure_usdt": "5000.00",
  "by_asset": {
    "BTCUSDT": {
      "exposure_usdt": "7500.00",
      "unrealized_pnl_usdt": "100.00",
      "size": "1.5"
    },
    "ETHUSDT": {
      "exposure_usdt": "2500.00",
      "unrealized_pnl_usdt": "50.00",
      "size": "10.0"
    }
  },
  "positions": [...],
  "calculated_at": "2025-01-15T10:00:00Z"
}
```

#### GET /api/v1/portfolio/exposure

Get only total portfolio exposure.

**Response**:
```json
{
  "total_exposure_usdt": "10000.00",
  "calculated_at": "2025-01-15T10:00:00Z"
}
```

#### GET /api/v1/portfolio/pnl

Get only PnL metrics for portfolio.

**Response**:
```json
{
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "total_pnl_usdt": "200.00",
  "calculated_at": "2025-01-15T10:00:00Z"
}
```

### Health Check

#### GET /health

Service health check.

**Response**:
```json
{
  "status": "healthy",
  "service": "position-manager",
  "database_connected": true,
  "queue_connected": true,
  "positions_count": 5,
  "timestamp": "2025-01-15T10:00:00Z"
}
```

## Event-Driven Integration

### Consuming Events from RabbitMQ

#### Order Manager Events

**Queue**: `order-manager.order_executed` (or new queue `order-manager.position_updated`)

**Event Format**:
```json
{
  "event_type": "position_updated_from_order",
  "order_id": "uuid",
  "asset": "BTCUSDT",
  "side": "Buy",
  "filled_quantity": "0.1",
  "execution_price": "50000.00",
  "mode": "one-way",
  "trace_id": "uuid"
}
```

**Processing**:
- Call `update_position_from_order_fill()`
- Update position in database
- Invalidate portfolio metrics cache
- Optionally publish position update event

#### WebSocket Gateway Events

**Queue**: `ws-gateway.position`

**Event Format** (standard WS Gateway format):
```json
{
  "event_type": "position",
  "channel": "position",
  "data": {
    "symbol": "BTCUSDT",
    "size": "1.5",
    "side": "Buy",
    "avgPrice": "50000.00",
    "unrealisedPnl": "150.00",
    "realisedPnl": "50.00",
    "mode": "one-way",
    "leverage": "10",
    "markPrice": "50100.00"
  },
  "trace_id": "uuid"
}
```

**Processing**:
- Parse position data from payload
- Extract fields: `unrealisedPnl`, `realisedPnl`, `avgPrice` (if present), `size` (for validation)
- Call `update_position_from_websocket()` with full data
- Update position in database:
  - Always update `unrealized_pnl`, `realized_pnl`
  - If `avgPrice` present in event:
    - Compare with saved `average_entry_price`
    - If difference > threshold (`POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD`), update `average_entry_price`
    - Log all updates with trace_id
  - Use `size` from event for validation (do not update directly, only check discrepancies)
- Invalidate portfolio metrics cache
- Recalculate ML features (`unrealized_pnl_pct`, `position_size_norm`) with updated `average_entry_price`
- Optionally publish position update event

**Error Handling**:
- If `avgPrice` missing in event, use saved `average_entry_price` value
- If `size` discrepancy is critical, trigger position validation
- Log all discrepancies for monitoring

### Publishing Events to RabbitMQ

#### Position Updated Event

**Queue**: `position-manager.position_updated`

**Event Format**:
```json
{
  "event_type": "position_updated",
  "position_id": "uuid",
  "asset": "BTCUSDT",
  "size": "1.5",
  "unrealized_pnl": "150.00",
  "realized_pnl": "50.00",
  "mode": "one-way",
  "unrealized_pnl_pct": "0.30",
  "time_held_minutes": 120,
  "position_size_norm": "0.15",
  "update_source": "order_execution" | "websocket",
  "trace_id": "uuid",
  "timestamp": "2025-01-15T10:00:00Z"
}
```

**Subscribers**:
- Model Service (for updating state during signal generation)
- Risk Manager (for limit checking)
- UI / Monitoring (for real-time display)

#### Portfolio Updated Event

**Queue**: `position-manager.portfolio_updated`

**Event Format**:
```json
{
  "event_type": "portfolio_updated",
  "total_exposure_usdt": "10000.00",
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "open_positions_count": 3,
  "trace_id": "uuid",
  "timestamp": "2025-01-15T10:00:00Z"
}
```

**Subscribers**:
- Risk Manager (for portfolio-level limit checking)
- UI / Monitoring (for metrics display)
- Alerting Service (for alerts on limit exceedance)

#### Position Snapshot Created Event

**Queue**: `position-manager.position_snapshot_created`

**Event Format**:
```json
{
  "event_type": "position_snapshot_created",
  "snapshot_id": "uuid",
  "position_id": "uuid",
  "asset": "BTCUSDT",
  "size": "1.5",
  "average_entry_price": "50000.00",
  "unrealized_pnl": "150.00",
  "realized_pnl": "50.00",
  "mode": "one-way",
  "long_size": null,
  "short_size": null,
  "unrealized_pnl_pct": "0.30",
  "time_held_minutes": 120,
  "position_size_norm": "0.15",
  "snapshot_timestamp": "2025-01-15T10:00:00Z",
  "trace_id": "uuid",
  "timestamp": "2025-01-15T10:00:00Z"
}
```

**Subscribers**:
- Model Service (for historical reconstruction of position state during model training)
- UI / Monitoring (for historical analysis and visualization)
- Analytics Service (for reporting and analytics)

**Note**: Snapshots are created periodically (configurable via `POSITION_MANAGER_SNAPSHOT_INTERVAL`). Each snapshot contains complete position state at creation time, critical for accurate historical reconstruction during model training.

## Integration Details

### Order Manager Integration

**Integration Method**: RabbitMQ events (recommended) or REST API

**Changes in Order Manager**:
- Remove `PositionManager` class
- Remove REST API endpoints for positions
- Remove background tasks for positions
- Replace `PositionManager` calls with REST API calls to Position Manager
- Publish position update events to RabbitMQ

**Integration Options**:
- **Option A (Recommended)**: Order Manager publishes events to RabbitMQ, Position Manager processes them via consumers. Benefits: asynchrony, fault tolerance, scalability, loose coupling.
- **Option B**: Order Manager calls Position Manager via REST API. Use only for synchronous operations requiring immediate response (e.g., checking position before creating order).

### Risk Manager Integration

**Integration Method**: REST API

**Changes in Risk Manager**:
- Add call to `GET /api/v1/portfolio/exposure` to get `total_exposure`
- Use obtained value in `check_max_exposure()`
- Add checks for other portfolio-level limits
- Add error handling for Position Manager unavailability (fallback logic)
- Add logging and alerts on limit exceedance
- Optimize performance (cache metrics, batch requests when checking multiple signals)

**Location**: Risk Manager remains in Order Manager (risk checks execute synchronously before order creation)

### Model Service Integration

**Integration Method**: REST API (primary) + RabbitMQ events (optional for caching)

**Changes in Model Service**:
- **Primary Method**: REST API requests to Position Manager during signal generation
  - Replace `PositionStateRepository._get_open_positions()` with REST API request `GET /api/v1/positions`
  - Replace local metric calculations (`get_total_exposure()`, `get_unrealized_pnl()`) with REST API request `GET /api/v1/portfolio`
  - Use synchronous REST requests for getting current data during signal generation
  - **Use pre-calculated ML features from Position Manager**: `unrealized_pnl_pct`, `time_held_minutes`, `position_size_norm` - these features are already calculated in Position Manager and included in REST API responses, no need to calculate locally in Model Service
  - **Implement risk management rules** (see Risk Management Rules section):
    - Take Profit: forced SELL when `unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT` (default 3.0%)
    - Position Size Limit: skip BUY when `position_size_norm > MODEL_SERVICE_MAX_POSITION_SIZE_RATIO` (default 0.8)
    - Use data from Position Manager via REST API to check these rules before model signal generation
- **RabbitMQ Events for Caching** (optional):
  - Subscribe to position update events (`position-manager.position_updated`) for invalidating local cache
  - Cache portfolio metrics in memory in Model Service
  - Reduce number of REST requests during frequent signal generation
  - **Important**: Cache is used for optimization, but REST API remains primary source of current data
- **RabbitMQ Events for Historical Reconstruction** (required for model training):
  - Subscribe to position snapshot creation events (`position-manager.position_snapshot_created`)
  - Save snapshots to local storage (database or file system) for subsequent use during training
  - Use snapshots to restore position state at the time of each execution event when building training dataset
  - Implement snapshot search by timestamp for accurate historical state reconstruction
  - Integrate snapshots into `DatasetBuilder` to pass historical `OrderPositionState` to feature engineering
- Add error handling for Position Manager unavailability (fallback to reading from database or using cache)
- Optimize performance (cache TTL, batch requests when generating multiple signals)

### WebSocket Gateway Integration

**Integration Method**: RabbitMQ events

**Changes**: None (WebSocket Gateway already publishes events to queue)

## Risk Management Rules for Model Service

Position Manager provides data for implementing risk management rules in Model Service:

### 1. Take Profit (Forced SELL on Profit Achievement)

- **Data**: `unrealized_pnl_pct` from Position Manager (via REST API or events)
- **Rule**: If `unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT`, Model Service must forcibly generate SELL signal to close position
- **Implementation**: In `IntelligentSignalGenerator.generate_signal()` before model signal generation, check `unrealized_pnl_pct` from Position Manager
- **Configuration**: `MODEL_SERVICE_TAKE_PROFIT_PCT` (default: 3.0)
- **Benefits**: Automatic profit locking, protection from reversals, disciplined risk management

### 2. Position Size Limit (Skip BUY on Large Position Size)

- **Data**: `position_size_norm` from Position Manager (via REST API or events)
- **Rule**: If `position_size_norm > MODEL_SERVICE_MAX_POSITION_SIZE_RATIO`, Model Service must skip BUY signal generation
- **Implementation**: In `IntelligentSignalGenerator.generate_signal()` before BUY signal generation, check `position_size_norm` from Position Manager
- **Configuration**: `MODEL_SERVICE_MAX_POSITION_SIZE_RATIO` (default: 0.8)
- **Benefits**: Protection from over-exposure, portfolio diversification, concentration risk management

**Note**: These rules complement existing Risk Manager checks in Order Manager but are applied at signal generation stage, allowing to avoid creating unnecessary signals and reduce system load.

## Migration Plan

### Stage 1: Create New Service

1. Create project structure `position-manager/`
2. Configure Docker, docker-compose
3. Set up basic infrastructure (logging, database, RabbitMQ)
4. Create data models (Position, PositionSnapshot, PortfolioMetrics)
5. **Validate WebSocket Gateway tasks**: Check and update if necessary uncompleted tasks in `specs/001-websocket-gateway/tasks.md` (Phase 7.5: Position Channel Support, tasks T125-T136) to correlate with new Position Manager service architecture

### Stage 2: Extract Functionality from Order Manager

#### 2.1. Extract PositionManager

**Source**: `order-manager/src/services/position_manager.py`

**Actions**:
1. Copy `PositionManager` class to new service
2. Adapt to new structure (change imports, paths)
3. Add new methods for portfolio metrics
4. **Update `update_position_from_websocket()` method**:
   - Add handling of `avgPrice` from WebSocket event
   - Implement logic to compare with saved `average_entry_price`
   - Add update of `average_entry_price` when difference exceeds threshold (`POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD`)
   - Add validation of `size` from WebSocket event (without direct update, only discrepancy checking)
   - Add recalculation of ML features (`unrealized_pnl_pct`, `position_size_norm`) after update
   - Add logging of all updates with trace_id
   - Add handling of case when `avgPrice` is missing in event (use saved value)
5. Update Order Manager to use Position Manager via event stream (RabbitMQ)

#### 2.2. Extract REST API Endpoints

**Source**: `order-manager/src/api/routes/positions.py`

**Actions**:
1. Copy endpoints to new service
2. Add new endpoints for portfolio
3. Remove endpoints from Order Manager
4. Update API documentation

#### 2.3. Extract Background Tasks

**Source**: `order-manager/src/main.py` (classes `PositionSnapshotTask`, `PositionValidationTask`)

**Actions**:
1. Copy tasks to new service
2. Adapt to new structure
3. Remove from Order Manager
4. Configure execution in new service

#### 2.4. Clean Up Order Manager tasks.md

**Source**: `specs/004-order-manager/tasks.md`

**Actions**:
1. Remove uncompleted tasks related to positions that will be implemented in Position Manager
2. Update task descriptions that are partially related to positions to reflect use of Position Manager via API
3. Update task counters in summary section
4. Add notes that position management functionality has been moved to Position Manager service

### Stage 3: Integrate with Risk Manager

**Actions**:
1. Add REST API endpoint in Position Manager for getting `total_exposure` (`GET /api/v1/portfolio/exposure`)
2. Add method in Risk Manager to get `total_exposure` from Position Manager via REST API
3. Update `check_max_exposure()` in Risk Manager to use data from Position Manager
4. Implement `max_exposure` check at portfolio level before order creation
5. Add error handling for Position Manager unavailability (fallback logic)
6. Add logging and alerts on limit exceedance
7. Optimize performance (cache metrics, batch requests when checking multiple signals)

### Stage 4: Integrate with Model Service

**Actions**:
1. **Validate Model Service tasks.md**: Check and update unclosed tasks in `specs/001-model-service/tasks.md` related to positions and portfolio
2. Add REST API client in Model Service for calling Position Manager
3. Replace `PositionStateRepository._get_open_positions()` with REST API request `GET /api/v1/positions`
4. Replace local metric calculations with REST API requests
5. Implement caching of portfolio metrics in memory (optional)
6. Subscribe to position update events from RabbitMQ for cache invalidation (optional)
7. Subscribe to position snapshot creation events for historical reconstruction
8. Integrate snapshots into model training process
9. Implement risk management rules (Take Profit, Position Size Limit)
10. Add error handling for Position Manager unavailability
11. Optimize performance

### Stage 5: Testing and Validation

1. Unit tests for all Position Manager methods
2. Integration tests for REST API
3. E2E tests for complete position update flow
4. Performance tests for portfolio metrics
5. Migration validation (compare results before/after)

### Stage 6: Refactor Grafana Dashboards

After deploying Position Manager, update Grafana dashboards to use data from new service instead of direct queries to `positions` table in database.

#### 6.1. Dashboard "Trading Performance" (`trading-performance.json`)

**Current State**: Dashboard uses direct SQL queries to `positions` table for PnL metrics.

**Tasks**:

1. **Panel "Total PnL"** (ID: 1):
   - **Current Query**: `SELECT COALESCE(SUM((e.performance->>'realized_pnl')::DECIMAL), 0) + COALESCE((SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0), 0) as total_pnl FROM execution_events e`
   - **New Approach**: Replace `(SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0)` with REST API request to Position Manager: `GET /api/v1/portfolio/pnl`. Use Infinity datasource for HTTP requests to Position Manager API, or create PostgreSQL function/view that reads from Position Manager via HTTP (if supported)
   - **Alternative**: Use PostgreSQL datasource with query to `portfolio_metrics_view` (if created), which syncs with Position Manager

2. **Panel "Unrealized PnL"** (ID: 3):
   - **Current Query**: `SELECT COALESCE(SUM(unrealized_pnl), 0) as unrealized_pnl FROM positions WHERE size != 0`
   - **New Approach**: Replace with REST API request: `GET /api/v1/portfolio/pnl` → use `total_unrealized_pnl` field. Use Infinity datasource for HTTP requests
   - **Alternative**: Use PostgreSQL view synced with Position Manager

3. **Panel "Cumulative PnL Over Time"** (ID: 9):
   - **Current Query**: Includes `(SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0) as current_unrealized_pnl`
   - **New Approach**: Replace subquery with REST API request to Position Manager, or use historical data from position snapshots (`position_snapshots`) for building unrealized PnL time series
   - **Recommendation**: Use position snapshots for historical chart, as they contain complete change history

4. **Panel "PnL by Asset"** (ID: 13):
   - **Current Query**: Aggregates data from `execution_events` by `asset`
   - **New Approach**: Add panel with data from Position Manager: `GET /api/v1/portfolio?include_positions=true`. Use `by_asset` field from API response to display PnL by assets. Keep existing panel for realized PnL from execution_events, add new one for unrealized PnL from Position Manager

**File to Update**: `grafana/dashboards/trading-performance.json`

#### 6.2. Dashboard "Trading System Monitoring" (`trading-system-monitoring.json`)

**Current State**: Panel "Order Execution" uses JOIN with `positions` table for position data.

**Tasks**:

1. **Panel "Order Execution"** (ID: 2):
   - **Current Query**: Uses `LEFT JOIN LATERAL (SELECT unrealized_pnl, size, average_entry_price FROM positions WHERE asset = o.asset) p`
   - **New Approach**:
     - **Option 1 (Recommended)**: Use PostgreSQL view `positions_view`, which syncs with Position Manager via triggers or periodic updates
     - **Option 2**: Split panel into two parts: main table with order data (from `orders` and `execution_events`), additional table with position data from Position Manager via REST API (Infinity datasource)
     - **Option 3**: Use PostgreSQL function that makes HTTP request to Position Manager API (if PostgreSQL supports HTTP extensions)

**File to Update**: `grafana/dashboards/trading-system-monitoring.json`

#### 6.3. Dashboard "Order Execution Panel" (`order-execution-panel.json`)

**Current State**: Uses JOIN with `positions` table for PnL calculation.

**Tasks**:

1. **Panel "Order Execution Panel"**:
   - **Current Query**: Uses `LEFT JOIN LATERAL (SELECT unrealized_pnl, size, average_entry_price, current_price FROM positions WHERE asset = o.asset) p`
   - **New Approach**: Similar to panel in "Trading System Monitoring", use `positions_view` view, or split into two panels: orders + positions from Position Manager API

**File to Update**: `grafana/dashboards/order-execution-panel.json`

#### 6.4. New Dashboards and Panels

**Tasks**:

1. **Create new dashboard "Portfolio Management"**:
   - Panel "Total Exposure" with data from `GET /api/v1/portfolio/exposure`
   - Panel "Portfolio PnL Breakdown" with asset breakdown from `GET /api/v1/portfolio?include_positions=true`
   - Panel "Position Size Distribution" using `position_size_norm` from Position Manager
   - Panel "Unrealized PnL by Asset" using `unrealized_pnl_pct` from Position Manager
   - Panel "Time Held by Position" using `time_held_minutes` from Position Manager
   - Panel "Position Snapshots History" with data from `GET /api/v1/positions/{asset}/snapshots`

2. **Add panel "Position Manager Health"** to "System Health" dashboard:
   - Position Manager health check status: `GET /health`
   - Performance metrics: API response time, number of position updates
   - Validation statistics: number of discrepancies, successful synchronizations

3. **Add panel "Risk Management Metrics"**:
   - Display `total_exposure` for Risk Manager limit checking
   - Visualize `position_size_norm` for each asset (for Position Size Limit rule)
   - Visualize `unrealized_pnl_pct` for each asset (for Take Profit rule)

**Files to Create**: `grafana/dashboards/portfolio-management.json` (new dashboard)

#### 6.5. Technical Implementation Details

**Approach 1: PostgreSQL Views (Recommended)**

Create views in database that sync with Position Manager:

```sql
-- View for portfolio metrics
CREATE VIEW portfolio_metrics_view AS
SELECT 
    'total_exposure' as metric_name,
    (SELECT total_exposure::text FROM http_get('http://position-manager:4800/api/v1/portfolio/exposure')::json) as metric_value
UNION ALL
SELECT 
    'total_unrealized_pnl' as metric_name,
    (SELECT total_unrealized_pnl::text FROM http_get('http://position-manager:4800/api/v1/portfolio/pnl')::json) as metric_value;

-- View for positions (synced via triggers or periodic updates)
CREATE VIEW positions_view AS
SELECT * FROM positions;  -- If positions table is synced with Position Manager
```

**Approach 2: Infinity Datasource for REST API**

Use Grafana Infinity datasource for direct HTTP requests to Position Manager API:
- Configure Infinity datasource with URL `http://position-manager:4800`
- Use JSON parser for parsing API responses
- Create variables for filtering by assets, strategies, etc.

**Approach 3: Hybrid Approach**
- For real-time: use Infinity datasource for direct requests to Position Manager API
- For historical data: use PostgreSQL with data from position snapshots (`position_snapshots`)

#### 6.6. Execution Order

1. **Preparation**: Create PostgreSQL views or configure Infinity datasource for Position Manager, test Position Manager API accessibility from Grafana container, create backup of existing dashboards
2. **Update Existing Dashboards**: Update `trading-performance.json`, `trading-system-monitoring.json`, `order-execution-panel.json`
3. **Create New Dashboards**: Create `portfolio-management.json` with new panels, add panels to existing dashboards
4. **Testing**: Verify data display correctness, compare results with data from old `positions` table, check query performance
5. **Documentation**: Update dashboard documentation, add description of new panels and data sources

#### 6.7. Dependencies

- Position Manager must be deployed and accessible
- Position Manager REST API must be tested
- Grafana must have access to Position Manager (network, port 4800)
- If using Infinity datasource, plugin must be installed in Grafana

### Stage 7: Documentation and Deployment

1. Update README with deployment instructions
2. Update API documentation
3. Update docker-compose.yml
4. Update env.example
5. Deploy to test environment
6. Monitoring and debugging
7. Update Grafana dashboard documentation

## Security & Error Handling

### API Authentication

- API Key authentication for all REST API endpoints
- API Key validation on each request
- Logging of all requests with trace IDs
- Rate limiting: Per-API-Key rate limits with different tiers (e.g., 100 req/min for Model Service, 1000 req/min for UI)

### Data Validation

- Validation of all input data (Pydantic models)
- Type and range value checks
- String parameter sanitization

### Error Handling

- Graceful degradation when dependencies are unavailable
- Retry logic for external calls
- Detailed error logging

## Observability

### Metrics

- Number of positions
- Number of position updates per second
- REST API endpoint response time
- Metrics cache size
- Number of validation errors

### Logging

- Structured logging with trace IDs
- Logging of all position operations
- Logging of discrepancies during validation
- Logging of limit exceedance

### Health Checks

- Database connection check
- RabbitMQ connection check
- Dependent service availability check

## Performance & Scalability

### Query Optimization

- Use indexes on `positions.asset`, `positions.mode`
- Cache aggregated metrics in memory (optional Redis)
- Batch processing of position updates
- Asynchronous database queries

### Scaling

- Horizontal scaling through multiple instances
- Shared state via database and RabbitMQ
- Stateless REST API endpoints
- Metrics caching to reduce database load

## Risks & Mitigation

### Risks

1. **Functionality Gap During Migration**
   - Mitigation: Gradual migration, parallel operation of old and new service

2. **Performance During Metrics Calculation**
   - Mitigation: Caching, query optimization, database indexes

3. **Data Consistency Between Services**
   - Mitigation: Shared database, transactions, validation

4. **Integration Complexity with Existing Services**
   - Mitigation: Gradual migration, API backward compatibility

### Rollback Plan

- Ability to rollback to old architecture
- Preserve old code until full validation
- Gradual traffic switching
