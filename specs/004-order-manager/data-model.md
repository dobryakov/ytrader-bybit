# Data Model: Order Manager Microservice

**Feature**: Order Manager Microservice  
**Date**: 2025-01-27  
**Database**: PostgreSQL (shared database, migrations managed in ws-gateway service per constitution)

## Overview

This document defines the data models for the Order Manager microservice. All entities are persisted in the shared PostgreSQL database. Database migrations MUST be located in the `ws-gateway` service per constitution requirement (PostgreSQL migration ownership).

## Core Entities

### 1. Order

Represents a trading order placed on Bybit exchange. Orders are created from trading signals and updated based on execution events from WebSocket gateway.

**Table**: `orders`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique order identifier (internal) |
| `order_id` | VARCHAR(100) | NOT NULL, UNIQUE | Bybit order ID (returned by exchange) |
| `signal_id` | UUID | NOT NULL | Trading signal identifier that created this order |
| `asset` | VARCHAR(20) | NOT NULL | Trading pair symbol (e.g., 'BTCUSDT') |
| `side` | VARCHAR(10) | NOT NULL | Order side: 'Buy' or 'Sell' |
| `order_type` | VARCHAR(20) | NOT NULL | Order type: 'Market', 'Limit' |
| `quantity` | DECIMAL(20, 8) | NOT NULL | Order quantity in base currency |
| `price` | DECIMAL(20, 8) | NULL | Limit order price (NULL for market orders) |
| `status` | VARCHAR(50) | NOT NULL, DEFAULT 'pending' | Order status: 'pending', 'partially_filled', 'filled', 'cancelled', 'rejected', 'dry_run' |
| `filled_quantity` | DECIMAL(20, 8) | NOT NULL, DEFAULT 0 | Quantity that has been filled |
| `average_price` | DECIMAL(20, 8) | NULL | Average execution price (if partially or fully filled) |
| `fees` | DECIMAL(20, 8) | NULL | Total fees paid for this order |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When order was created |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last update timestamp |
| `executed_at` | TIMESTAMP | NULL | When order was fully executed (filled) |
| `trace_id` | VARCHAR(100) | NULL | Trace ID for request flow tracking |
| `is_dry_run` | BOOLEAN | NOT NULL, DEFAULT false | Whether order was created in dry-run mode |

**Indexes**:
- `idx_orders_order_id` on `order_id` (for Bybit order ID lookups)
- `idx_orders_signal_id` on `signal_id` (for signal-to-order relationships)
- `idx_orders_asset` on `asset` (for asset-specific queries)
- `idx_orders_status` on `status` (for filtering active orders)
- `idx_orders_created_at` on `created_at` (for time-based queries)
- `idx_orders_asset_status` on `(asset, status)` (for active order queries per asset)

**Validation Rules**:
- `side` must be 'Buy' or 'SELL' (see **Data Format Conventions** section below)
- `order_type` must be 'Market' or 'Limit'
- `status` must be one of: 'pending', 'partially_filled', 'filled', 'cancelled', 'rejected', 'dry_run'
- `price` must be NULL for market orders, NOT NULL for limit orders
- `quantity` must be > 0
- `filled_quantity` must be <= `quantity`
- `asset` must match Bybit trading pair format (uppercase, e.g., 'BTCUSDT')

**State Transitions**:
- `pending` → `partially_filled`: Order partially executed
- `pending` → `filled`: Order fully executed
- `pending` → `cancelled`: Order cancelled (by user or system)
- `pending` → `rejected`: Order rejected by exchange
- `partially_filled` → `filled`: Remaining quantity executed
- `partially_filled` → `cancelled`: Partially filled order cancelled
- Any status → `updated_at` updated: Order state updated (status change, fill update)

**Relationships**:
- Many orders can be linked to one signal (via `signal_id`) - for 1:N signal-to-order relationships
- One order can reference one signal (direct relationship)
- Orders link to positions via `asset` and `side`

---

### 2. Trading Signal

Represents a high-level trading instruction received from model service. Signals are consumed from RabbitMQ queue and may not be persisted (in-memory processing). For audit trail and relationship tracking, signal metadata may be stored.

**In-Memory Structure** (consumed from RabbitMQ queue, not stored in database by default):

```python
{
    "signal_id": "uuid",                    # Unique signal identifier
    "signal_type": "buy" | "sell",          # Trading signal type
    "asset": "string",                      # Trading pair (e.g., "BTCUSDT")
    "amount": "float",                      # Amount in quote currency (USDT)
    "confidence": "float",                  # Confidence score (0.0-1.0)
    "timestamp": "datetime",                # Signal generation timestamp
    "strategy_id": "string",                # Trading strategy identifier
    "model_version": "string | null",       # Model version (null for warm-up)
    "is_warmup": "boolean",                 # Whether warm-up signal
    "market_data_snapshot": {               # Market data at signal time
        "price": "float",
        "spread": "float",
        "volume_24h": "float",
        "volatility": "float",
        "orderbook_depth": "dict | null",
        "technical_indicators": "dict | null"
    },
    "metadata": "dict | null",              # Additional metadata
    "trace_id": "string | null"             # Trace ID for request tracking
}
```

**Optional Persistence Table**: `trading_signals` (for audit trail, if needed)

If signals are persisted for audit purposes:

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `signal_id` | UUID | PRIMARY KEY | Unique signal identifier |
| `signal_type` | VARCHAR(10) | NOT NULL | 'buy' or 'sell' |
| `asset` | VARCHAR(20) | NOT NULL | Trading pair |
| `amount` | DECIMAL(20, 8) | NOT NULL | Amount in quote currency |
| `confidence` | DECIMAL(5, 4) | NOT NULL | Confidence (0.0-1.0) |
| `timestamp` | TIMESTAMP | NOT NULL | Signal generation time |
| `strategy_id` | VARCHAR(100) | NOT NULL | Strategy identifier |
| `model_version` | VARCHAR(100) | NULL | Model version |
| `is_warmup` | BOOLEAN | NOT NULL | Warm-up flag |
| `market_data_snapshot` | JSONB | NULL | Market data snapshot |
| `metadata` | JSONB | NULL | Additional metadata |
| `trace_id` | VARCHAR(100) | NULL | Trace ID |
| `received_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When Order Manager received signal |
| `processed_at` | TIMESTAMP | NULL | When signal was processed |

**Note**: Signal persistence is optional for audit trail. Initial implementation processes signals in-memory only, with optional persistence added if audit requirements emerge.

---

### 3. Signal-Order Relationship

Represents the mapping between trading signals and orders. Tracks which orders were created from which signals, enabling relationship analysis and position building across multiple signals.

**Table**: `signal_order_relationships`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique relationship identifier |
| `signal_id` | UUID | NOT NULL | Trading signal identifier |
| `order_id` | UUID | NOT NULL | Order identifier (references orders.id) |
| `relationship_type` | VARCHAR(20) | NOT NULL | Relationship type: 'one_to_one', 'one_to_many', 'many_to_one' |
| `execution_sequence` | INTEGER | NULL | Sequence number for ordering (for 1:N relationships) |
| `allocation_amount` | DECIMAL(20, 8) | NULL | Amount allocated from signal to this order (for partial allocations) |
| `allocation_quantity` | DECIMAL(20, 8) | NULL | Quantity allocated from signal to this order |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When relationship was created |

**Indexes**:
- `idx_signal_order_signal_id` on `signal_id` (for finding orders by signal)
- `idx_signal_order_order_id` on `order_id` (for finding signals by order)
- `idx_signal_order_signal_order` on `(signal_id, order_id)` (unique constraint for relationship)

**Unique Constraint**:
- `UNIQUE(signal_id, order_id)` - Prevent duplicate relationships

**Validation Rules**:
- `relationship_type` must be one of: 'one_to_one', 'one_to_many', 'many_to_one'
- `execution_sequence` must be >= 1 if not NULL
- `allocation_amount` and `allocation_quantity` must be > 0 if not NULL

**Relationships**:
- Many relationships can reference one signal (1:N signal-to-order)
- One relationship references one order
- Enables tracking of complex signal-to-order mappings (splitting, accumulation)

---

### 4. Order Execution Event

Real-time event received from WebSocket gateway about order status changes. Events are consumed from RabbitMQ queue and used to update order state in database. Events are not persisted (ephemeral).

**In-Memory Structure** (consumed from RabbitMQ queue `ws-gateway.order_status`):

```python
{
    "event_id": "uuid",                     # Unique event identifier
    "event_type": "string",                 # Event type: 'filled', 'partially_filled', 'cancelled', 'rejected'
    "order_id": "string",                   # Bybit order ID
    "symbol": "string",                     # Trading pair
    "side": "string",                       # 'Buy' or 'Sell'
    "order_type": "string",                 # Order type
    "status": "string",                     # Current order status
    "executed_qty": "float",                # Executed quantity
    "avg_price": "float",                   # Average execution price
    "cum_exec_qty": "float",                # Cumulative executed quantity
    "cum_exec_value": "float",              # Cumulative executed value
    "timestamp": "datetime",                # Event timestamp from exchange
    "received_at": "datetime",              # When gateway received event
    "trace_id": "string | null"             # Trace ID
}
```

**Processing Flow**:
1. Event received from RabbitMQ queue `ws-gateway.order_status`
2. Event parsed and validated
3. Order record updated in database using `order_id` (Bybit order ID)
4. Position updated if order was filled
5. Enriched order event published to `order-manager.order_events` queue

---

### 5. Position

Represents current trading position for an asset. Position state is stored and updated based on order executions. Supports both one-way and hedge-mode trading.

**Table**: `positions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique position identifier |
| `asset` | VARCHAR(20) | NOT NULL | Trading pair symbol (e.g., 'BTCUSDT') |
| `size` | DECIMAL(20, 8) | NOT NULL | Position size (positive = long, negative = short, zero = no position) |
| `average_entry_price` | DECIMAL(20, 8) | NULL | Average entry price for position |
| `unrealized_pnl` | DECIMAL(20, 8) | NULL | Unrealized profit and loss |
| `realized_pnl` | DECIMAL(20, 8) | NULL | Realized profit and loss (cumulative) |
| `mode` | VARCHAR(20) | NOT NULL, DEFAULT 'one-way' | Trading mode: 'one-way' or 'hedge' |
| `long_size` | DECIMAL(20, 8) | NULL | Long position size (for hedge-mode, separate from short) |
| `short_size` | DECIMAL(20, 8) | NULL | Short position size (for hedge-mode, separate from long) |
| `long_avg_price` | DECIMAL(20, 8) | NULL | Average entry price for long position (hedge-mode) |
| `short_avg_price` | DECIMAL(20, 8) | NULL | Average entry price for short position (hedge-mode) |
| `last_updated` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last update timestamp |
| `last_snapshot_at` | TIMESTAMP | NULL | Last snapshot timestamp |

**Indexes**:
- `idx_positions_asset` on `asset` (for asset-specific queries)
- `idx_positions_asset_mode` on `(asset, mode)` (unique constraint)
- `idx_positions_last_updated` on `last_updated` (for recent updates)

**Unique Constraint**:
- `UNIQUE(asset, mode)` - One position record per asset per mode

**Validation Rules**:
- `mode` must be 'one-way' or 'hedge'
- In one-way mode: `size` represents net position (positive = long, negative = short)
- In hedge-mode: `long_size` and `short_size` track separate positions (both can be non-zero)
- `size` must match mode: in one-way mode, use `size`; in hedge-mode, `size` may be 0 if long and short positions offset

**State Management**:
- Position updated when orders are filled (via WebSocket events)
- Position computed from order history for validation
- Position snapshots created periodically (configurable interval)

**Relationships**:
- One position per asset (in one-way mode) or per asset+mode (in hedge-mode)
- Position affected by orders with matching `asset`
- Position size calculated from filled orders

---

### 6. Position Snapshot

Periodic snapshots of position state for historical tracking and validation.

**Table**: `position_snapshots`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique snapshot identifier |
| `position_id` | UUID | NOT NULL | Reference to positions.id |
| `asset` | VARCHAR(20) | NOT NULL | Trading pair |
| `size` | DECIMAL(20, 8) | NOT NULL | Position size at snapshot time |
| `average_entry_price` | DECIMAL(20, 8) | NULL | Average entry price at snapshot time |
| `unrealized_pnl` | DECIMAL(20, 8) | NULL | Unrealized PnL at snapshot time |
| `realized_pnl` | DECIMAL(20, 8) | NULL | Realized PnL at snapshot time |
| `mode` | VARCHAR(20) | NOT NULL | Trading mode |
| `long_size` | DECIMAL(20, 8) | NULL | Long size (hedge-mode) |
| `short_size` | DECIMAL(20, 8) | NULL | Short size (hedge-mode) |
| `snapshot_timestamp` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When snapshot was created |

**Indexes**:
- `idx_position_snapshots_position_id` on `position_id` (for position history)
- `idx_position_snapshots_asset_timestamp` on `(asset, snapshot_timestamp DESC)` (for time-series queries)
- `idx_position_snapshots_timestamp` on `snapshot_timestamp` (for cleanup queries)

**Retention Policy**:
- Snapshots retained for configurable period (default: 30 days)
- Older snapshots can be archived or deleted

---

## Data Relationships

```
trading_signals (in-memory, from RabbitMQ)
    │
    ├──> signal_order_relationships (N:1)
    │         │
    │         └──> orders (1:1)
    │                   │
    │                   └──> order_execution_events (from RabbitMQ)
    │                               │
    │                               └──> positions (updates via asset)
    │
    └──> [optional: trading_signals table for audit]

positions
    └──> position_snapshots (1:N, periodic)
```

## Database Migration Notes

**IMPORTANT**: All PostgreSQL migrations MUST be located in the `ws-gateway/migrations/` directory per constitution requirement. Migration files should follow the naming convention: `XXX_create_order_manager_tables.sql` where XXX is the next sequential number.

Migration files to create:
1. `XXX_create_orders_table.sql` - Create orders table and indexes
2. `XXX_create_signal_order_relationships_table.sql` - Create signal-order relationships table
3. `XXX_create_positions_table.sql` - Create positions table and indexes
4. `XXX_create_position_snapshots_table.sql` - Create position snapshots table

Migrations should be:
- Reversible whenever possible
- Include indexes and constraints
- Follow existing migration patterns in ws-gateway service

---

## Data Format Conventions

### Order Side Format

**CRITICAL**: The `side` field has different formats depending on context. This is important to prevent API errors and database constraint violations.

#### Bybit API Format
- **Format**: `"Buy"` or `"Sell"` (capitalize first letter only)
- **Used in**: All API requests to Bybit (`/v5/order/create`, etc.)
- **Examples**: 
  ```python
  params = {"side": "Buy"}  # ✅ Correct for Bybit API
  params = {"side": "Sell"}  # ✅ Correct for Bybit API
  params = {"side": "SELL"}  # ❌ Wrong - will cause "Side invalid (code: 10001)"
  ```

#### Database Format
- **Format**: `"Buy"` or `"SELL"` (uppercase for SELL)
- **Used in**: All database inserts/updates in the `orders` table
- **Database constraint**: `CHECK (side IN ('Buy', 'SELL'))`
- **Examples**:
  ```python
  side_db = "Buy"   # ✅ Correct for database
  side_db = "SELL"  # ✅ Correct for database
  side_db = "Sell"  # ❌ Wrong - violates constraint "chk_side"
  ```

#### Implementation Pattern

When creating orders, use different variables for API and database:

```python
# Side for Bybit API: "Buy" or "Sell" (capitalize first letter only)
side_api = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

# Side for database: "Buy" or "SELL" (uppercase for SELL per constraint)
side_db = "Buy" if signal.signal_type.lower() == "buy" else "SELL"

# Use side_api for Bybit API requests
params = {"side": side_api, ...}

# Use side_db when saving to database
query = "INSERT INTO orders (side, ...) VALUES ($1, ...)"
await pool.execute(query, side_db, ...)
```

**Why this matters**:
- Bybit API expects `"Sell"` (capitalize first letter), not `"SELL"` (all uppercase)
- Database constraint requires `"SELL"` (all uppercase), not `"Sell"` (capitalize first letter)
- Using wrong format causes:
  - Bybit API: `"Side invalid (code: 10001)"` error
  - Database: `CHECK constraint "chk_side" violation` error

**File locations where this is used**:
- `order-manager/src/services/order_executor.py`: `_prepare_bybit_order_params()` method
- `order-manager/src/services/order_executor.py`: `_save_order_to_database()` method
- `order-manager/src/services/order_executor.py`: `_save_rejected_order()` method
- `order-manager/src/services/signal_processor.py`: `_save_rejected_order()` method

**References**:
- Bybit API Documentation: `/v5/order/create` endpoint requires `side` as `"Buy"` or `"Sell"`
- Database migration: See constraint definition in orders table migration

