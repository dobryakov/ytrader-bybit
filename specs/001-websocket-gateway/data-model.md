# Data Model: WebSocket Gateway

**Feature**: WebSocket Gateway for Bybit Data Aggregation and Routing  
**Date**: 2025-11-25  
**Database**: PostgreSQL (shared database)

## Overview

This document defines the data models for the WebSocket Gateway service. All entities are persisted in the shared PostgreSQL database unless otherwise noted.

## Core Entities

### 1. Subscription

Represents an active subscription to a specific Bybit WebSocket data channel. Subscriptions are persisted to enable automatic resubscription after reconnection.

**Table**: `subscriptions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique subscription identifier |
| `channel_type` | VARCHAR(50) | NOT NULL | Type of channel (e.g., 'trades', 'ticker', 'orderbook', 'order', 'balance') |
| `symbol` | VARCHAR(20) | NULL | Trading pair symbol (e.g., 'BTCUSDT'), NULL for non-symbol-specific channels |
| `topic` | VARCHAR(200) | NOT NULL | Full topic string as required by Bybit API (e.g., 'trade.BTCUSDT') |
| `requesting_service` | VARCHAR(100) | NOT NULL | Name/identifier of the microservice that requested this subscription |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT true | Whether subscription is currently active |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When subscription was created |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last update timestamp |
| `last_event_at` | TIMESTAMP | NULL | Timestamp of last event received for this subscription |

**Indexes**:
- `idx_subscriptions_topic` on `topic` (for fast lookup during resubscription)
- `idx_subscriptions_active` on `is_active` (for filtering active subscriptions)
- `idx_subscriptions_service` on `requesting_service` (for service-specific queries)

**Validation Rules**:
- `channel_type` must be one of: 'trades', 'ticker', 'orderbook', 'order', 'balance', 'kline', 'liquidation'
- `topic` must match Bybit WebSocket topic format
- `symbol` is required for symbol-specific channels (trades, ticker, orderbook, order)
- `symbol` must be NULL for non-symbol-specific channels (e.g., balance)

**State Transitions**:
- `is_active: false` → `is_active: true`: Subscription reactivated (via REST API)
- `is_active: true` → `is_active: false`: Subscription cancelled (via REST API or service shutdown)
- On reconnection: All subscriptions with `is_active=true` are automatically resubscribed

**Relationships**:
- Multiple subscriptions can share the same `topic` (when multiple services request the same channel)
- One subscription can have many events (Event entity references topic)

---

### 2. Event

Represents a data message received from Bybit WebSocket. Events are temporarily stored in memory/queues and not persisted to database (except for specific event types like balances).

**In-Memory Structure** (not a database table):

```python
{
    "event_id": "uuid",           # Unique identifier for this event
    "event_type": "string",        # Type: 'trade', 'ticker', 'orderbook', 'order', 'balance', etc.
    "topic": "string",             # Topic this event belongs to (e.g., 'trade.BTCUSDT')
    "timestamp": "datetime",       # Event timestamp from exchange
    "received_at": "datetime",     # When gateway received the event
    "payload": "dict",             # Structured event data (varies by event_type)
    "trace_id": "string"           # Trace ID for request flow tracking
}
```

**Event Types and Payloads**:

- **Trade Event** (`event_type: 'trade'`):
  ```json
  {
    "symbol": "BTCUSDT",
    "price": "50000.00",
    "quantity": "0.1",
    "side": "Buy",
    "trade_time": 1234567890000
  }
  ```

- **Ticker Event** (`event_type: 'ticker'`):
  ```json
  {
    "symbol": "BTCUSDT",
    "last_price": "50000.00",
    "volume_24h": "1000.5",
    "turnover_24h": "50000000.00"
  }
  ```

- **Order Book Event** (`event_type: 'orderbook'`):
  ```json
  {
    "symbol": "BTCUSDT",
    "bids": [["50000.00", "1.5"], ...],
    "asks": [["50001.00", "2.0"], ...],
    "update_id": 12345
  }
  ```

- **Order Status Event** (`event_type: 'order'`):
  ```json
  {
    "order_id": "abc123",
    "symbol": "BTCUSDT",
    "side": "Buy",
    "order_type": "Limit",
    "price": "50000.00",
    "qty": "0.1",
    "status": "Filled",
    "executed_qty": "0.1"
  }
  ```

- **Balance Event** (`event_type: 'balance'`):
  ```json
  {
    "coin": "USDT",
    "wallet_balance": "10000.00",
    "available_balance": "9500.00",
    "frozen": "500.00"
  }
  ```

**Processing Flow**:
1. Event received from Bybit WebSocket
2. Event parsed and validated
3. Event placed in appropriate RabbitMQ queue (by event_type)
4. If `event_type == 'balance'`, also persisted to `account_balances` table
5. Event delivered to subscribers via queues

---

### 3. Account Balance

Represents account balance information that requires immediate persistence to PostgreSQL (per requirement FR-013).

**Table**: `account_balances`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique record identifier |
| `coin` | VARCHAR(10) | NOT NULL | Coin symbol (e.g., 'USDT', 'BTC') |
| `wallet_balance` | DECIMAL(20, 8) | NOT NULL | Total wallet balance |
| `available_balance` | DECIMAL(20, 8) | NOT NULL | Available balance (not frozen) |
| `frozen` | DECIMAL(20, 8) | NOT NULL, DEFAULT 0 | Frozen balance |
| `event_timestamp` | TIMESTAMP | NOT NULL | Timestamp from exchange event |
| `received_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When gateway received the event |
| `trace_id` | VARCHAR(100) | NULL | Trace ID for request flow tracking |

**Indexes**:
- `idx_account_balances_coin` on `coin` (for coin-specific queries)
- `idx_account_balances_received_at` on `received_at` (for time-based queries)
- `idx_account_balances_coin_received_at` on `(coin, received_at DESC)` (for latest balance per coin)

**Validation Rules**:
- `wallet_balance` = `available_balance` + `frozen`
- `coin` must be a valid cryptocurrency symbol
- All balance values must be non-negative

**Update Strategy**:
- On balance event receipt: Insert new record (append-only for audit trail)
- For latest balance queries: Use `ORDER BY received_at DESC LIMIT 1` per coin
- Alternative: Update existing record if same coin and recent timestamp (within 1 second) - requires business logic decision

**Relationships**:
- One coin can have many balance records (time series)
- No foreign key relationships (standalone entity)

---

### 4. WebSocket Connection State

Represents the state of the WebSocket connection to Bybit. This is primarily an in-memory entity but may be logged/persisted for monitoring.

**In-Memory Structure** (not a database table):

```python
{
    "connection_id": "uuid",           # Unique connection identifier
    "environment": "string",           # 'mainnet' or 'testnet'
    "status": "string",                 # 'connected', 'disconnected', 'connecting', 'reconnecting'
    "connected_at": "datetime",         # When connection was established
    "last_heartbeat_at": "datetime",    # Last successful heartbeat
    "reconnect_count": "int",           # Number of reconnection attempts
    "last_error": "string",             # Last error message (if any)
    "subscriptions_active": "int"       # Count of active subscriptions
}
```

**State Transitions**:
- `disconnected` → `connecting`: Connection attempt initiated
- `connecting` → `connected`: Connection established and authenticated
- `connected` → `disconnected`: Connection lost or closed
- `disconnected` → `reconnecting`: Automatic reconnection initiated (within 30 seconds)
- `reconnecting` → `connected`: Reconnection successful

**Persistence**:
- Connection state may be logged to database for monitoring (optional `connection_logs` table)
- Not required for core functionality (in-memory state sufficient)

---

## Database Schema

### Migration Script Structure

```sql
-- migrations/001_create_subscriptions_table.sql
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    topic VARCHAR(200) NOT NULL,
    requesting_service VARCHAR(100) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_event_at TIMESTAMP,
    
    CONSTRAINT chk_channel_type CHECK (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'balance', 'kline', 'liquidation')),
    CONSTRAINT chk_symbol_required CHECK (
        (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NOT NULL) OR
        (channel_type NOT IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NULL)
    )
);

CREATE INDEX idx_subscriptions_topic ON subscriptions(topic);
CREATE INDEX idx_subscriptions_active ON subscriptions(is_active);
CREATE INDEX idx_subscriptions_service ON subscriptions(requesting_service);

-- migrations/002_create_account_balances_table.sql
CREATE TABLE IF NOT EXISTS account_balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    coin VARCHAR(10) NOT NULL,
    wallet_balance DECIMAL(20, 8) NOT NULL,
    available_balance DECIMAL(20, 8) NOT NULL,
    frozen DECIMAL(20, 8) NOT NULL DEFAULT 0,
    event_timestamp TIMESTAMP NOT NULL,
    received_at TIMESTAMP NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(100),
    
    CONSTRAINT chk_balance_sum CHECK (wallet_balance = available_balance + frozen),
    CONSTRAINT chk_non_negative CHECK (wallet_balance >= 0 AND available_balance >= 0 AND frozen >= 0)
);

CREATE INDEX idx_account_balances_coin ON account_balances(coin);
CREATE INDEX idx_account_balances_received_at ON account_balances(received_at);
CREATE INDEX idx_account_balances_coin_received_at ON account_balances(coin, received_at DESC);
```

---

## Data Flow

### Subscription Lifecycle

1. **Creation**: REST API request → Subscription created in database → WebSocket subscription sent to Bybit
2. **Active**: Events received → Processed and queued → `last_event_at` updated periodically
3. **Cancellation**: REST API request → `is_active=false` → Unsubscribe from Bybit (if no other services need it)
4. **Reconnection**: On disconnect → Query all `is_active=true` subscriptions → Resubscribe to Bybit

### Event Processing Flow

1. **Receive**: Event arrives from Bybit WebSocket
2. **Parse**: Extract event_id, event_type, topic, timestamp, payload
3. **Route**: Determine target queue based on event_type
4. **Queue**: Publish to RabbitMQ queue `ws-gateway.{event_type}`
5. **Persist** (if balance): Insert into `account_balances` table
6. **Log**: Log event receipt with trace_id

### Balance Persistence Flow

1. **Detect**: Event with `event_type='balance'` received
2. **Extract**: Parse balance data from payload
3. **Validate**: Check balance constraints (non-negative, sum consistency)
4. **Persist**: Insert new record into `account_balances` table
5. **Error Handling**: If write fails, log error and continue (per FR-017)

---

## Validation and Constraints

### Subscription Validation

- Topic format must match Bybit WebSocket API requirements
- Channel type must be supported by Bybit
- Symbol must be valid trading pair for symbol-specific channels
- Requesting service identifier must be provided

### Balance Validation

- All balance values must be non-negative decimals
- `wallet_balance` must equal `available_balance + frozen`
- Coin symbol must be valid cryptocurrency code
- Timestamps must be valid and not in the future

---

## Performance Considerations

### Indexing Strategy

- Subscriptions: Indexed by `topic` for fast resubscription lookups
- Subscriptions: Indexed by `is_active` for filtering active subscriptions
- Balances: Indexed by `coin` and `received_at` for latest balance queries

### Query Patterns

- **Resubscription**: `SELECT * FROM subscriptions WHERE is_active = true ORDER BY topic`
- **Latest Balance**: `SELECT * FROM account_balances WHERE coin = $1 ORDER BY received_at DESC LIMIT 1`
- **Service Subscriptions**: `SELECT * FROM subscriptions WHERE requesting_service = $1 AND is_active = true`

### Scalability

- Subscription table: Expected to have <1000 active subscriptions (manageable)
- Balance table: Append-only, may grow large over time - consider partitioning by `received_at` if needed
- Use connection pooling for concurrent database operations
- Prepared statements for frequently executed queries

---

## Future Considerations

### Potential Enhancements

- **Subscription History**: Track subscription changes over time (audit trail)
- **Event Persistence**: Optionally persist all events for replay/debugging (separate table)
- **Connection Logs**: Persist connection state changes for monitoring
- **Metrics**: Track event counts, latency, error rates per subscription

### Migration Notes

- All migrations must be reversible (per constitution requirement)
- Use `IF EXISTS` and `IF NOT EXISTS` for idempotent migrations
- Test migrations on testnet database before production

**PostgreSQL Migration Ownership**: Per constitution principle II (Shared Database Strategy), the `ws-gateway` service is the single source of truth for all PostgreSQL migrations. All PostgreSQL schema changes (including those for other services in the project) MUST be located in `ws-gateway/migrations/`. Other database types (e.g., vector databases for ML models) may maintain their own migrations within their respective service containers.

