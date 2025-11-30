# Data Model: Position Management Service

**Date**: 2025-01-27  
**Feature**: Position Management Service  
**Branch**: `001-position-management`

## Overview

This document defines the data models for the Position Management Service, including entities, relationships, validation rules, and state transitions.

## Entities

### Position

Represents a trading position (open or closed) for a specific asset and trading mode combination.

**Identity**: Composite key `(asset, mode)` - one position exists per asset per trading mode.

**Database Table**: `positions` (shared PostgreSQL database)

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, NOT NULL | Unique position identifier |
| `asset` | VARCHAR(20) | NOT NULL, INDEXED | Trading pair (e.g., "BTCUSDT") |
| `mode` | VARCHAR(20) | NOT NULL, INDEXED | Trading mode ("one-way", "hedge") |
| `size` | DECIMAL(20, 8) | NOT NULL | Position size (positive=long, negative=short, zero=closed) |
| `average_entry_price` | DECIMAL(20, 8) | NULL | Average entry price for position |
| `current_price` | DECIMAL(20, 8) | NULL, INDEXED | Latest markPrice from WebSocket events |
| `unrealized_pnl` | DECIMAL(20, 8) | NOT NULL, DEFAULT 0 | Current unrealized profit/loss |
| `realized_pnl` | DECIMAL(20, 8) | NOT NULL, DEFAULT 0 | Cumulative realized profit/loss |
| `long_size` | DECIMAL(20, 8) | NULL | Long position size (for hedge mode) |
| `short_size` | DECIMAL(20, 8) | NULL | Short position size (for hedge mode) |
| `version` | INTEGER | NOT NULL, DEFAULT 1, INDEXED | Version for optimistic locking |
| `last_updated` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last update timestamp |
| `closed_at` | TIMESTAMP | NULL | Timestamp when position was closed (size=0) |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Position creation timestamp |

**Indexes**:
- `idx_positions_asset` on `asset`
- `idx_positions_mode` on `mode`
- `idx_positions_asset_mode` on `(asset, mode)` - composite index for position identity lookups
- `idx_positions_current_price` on `current_price` - for portfolio calculations
- `idx_positions_version` on `version` - for optimistic locking

**Validation Rules**:
1. `asset` must be valid trading pair format (e.g., "BTCUSDT", "ETHUSDT")
2. `mode` must be one of: "one-way", "hedge"
3. `size` can be positive (long), negative (short), or zero (closed)
4. For "one-way" mode: `long_size` and `short_size` must be NULL
5. For "hedge" mode: `long_size` and `short_size` may be set
6. `average_entry_price` must be positive if `size != 0`
7. `version` must increment on each update
8. `closed_at` must be set when `size = 0`

**State Transitions**:
- **Created**: New position created with `size != 0`, `version = 1`
- **Updated**: Position updated (size, prices, PnL), `version` incremented
- **Closed**: Position size becomes 0, `closed_at` set, position retained for historical tracking

**ML Features** (calculated, not stored):
- `unrealized_pnl_pct`: `(unrealized_pnl / (abs(size) * average_entry_price)) * 100` (if entry price exists)
- `time_held_minutes`: `(current_timestamp - last_updated) / 60`
- `position_size_norm`: `abs(size * current_price) / total_exposure` (relative to portfolio)

**Relationships**:
- Has many `PositionSnapshot` records (one-to-many)
- Belongs to `Portfolio` (many-to-one, aggregated)

### Position Snapshot

Represents a historical record of position state at a specific point in time.

**Database Table**: `position_snapshots` (shared PostgreSQL database)

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, NOT NULL | Unique snapshot identifier |
| `position_id` | UUID | NOT NULL, FOREIGN KEY | Reference to position |
| `asset` | VARCHAR(20) | NOT NULL | Trading pair (denormalized for query performance) |
| `mode` | VARCHAR(20) | NOT NULL | Trading mode (denormalized) |
| `snapshot_data` | JSONB | NOT NULL | Complete position state at snapshot time |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEXED | Snapshot creation timestamp |

**Indexes**:
- `idx_position_snapshots_position_id` on `position_id`
- `idx_position_snapshots_created_at` on `created_at` - for retention cleanup queries
- `idx_position_snapshots_asset` on `asset` - for asset-based queries

**Validation Rules**:
1. `snapshot_data` must contain all position fields at creation time
2. `snapshot_data` must be valid JSON
3. `created_at` must be set to current timestamp

**Retention Policy**:
- Snapshots older than 1 year (`POSITION_MANAGER_SNAPSHOT_RETENTION_DAYS=365`) are automatically deleted
- Cleanup job runs on service startup

**Relationships**:
- Belongs to `Position` (many-to-one)

**snapshot_data JSON Structure**:
```json
{
  "id": "uuid",
  "asset": "BTCUSDT",
  "mode": "one-way",
  "size": "1.5",
  "average_entry_price": "50000.00",
  "current_price": "50100.00",
  "unrealized_pnl": "150.00",
  "realized_pnl": "50.00",
  "long_size": null,
  "short_size": null,
  "unrealized_pnl_pct": "0.30",
  "time_held_minutes": 120,
  "position_size_norm": "0.15",
  "last_updated": "2025-01-15T10:00:00Z",
  "closed_at": null
}
```

### Portfolio

Represents the aggregate of all positions. Not stored as a separate entity - calculated on-demand from positions.

**Fields** (calculated, not stored):

| Field | Type | Description |
|-------|------|-------------|
| `total_exposure_usdt` | DECIMAL(20, 8) | Sum of `ABS(size) * current_price` for all positions where `current_price IS NOT NULL` |
| `total_unrealized_pnl_usdt` | DECIMAL(20, 8) | Sum of `unrealized_pnl` for all positions |
| `total_realized_pnl_usdt` | DECIMAL(20, 8) | Sum of `realized_pnl` for all positions |
| `portfolio_value_usdt` | DECIMAL(20, 8) | Sum of `size * current_price` for all positions where `current_price IS NOT NULL` |
| `open_positions_count` | INTEGER | Count of positions where `size != 0` |
| `long_positions_count` | INTEGER | Count of positions where `size > 0` |
| `short_positions_count` | INTEGER | Count of positions where `size < 0` |
| `net_exposure_usdt` | DECIMAL(20, 8) | `total_exposure_usdt` for long positions minus short positions |
| `by_asset` | JSON | Breakdown by asset (exposure, PnL, size per asset) |
| `calculated_at` | TIMESTAMP | Timestamp when metrics were calculated |

**Calculation Rules**:
1. Positions with `current_price IS NULL` are excluded from exposure calculations
2. Positions with `size = 0` are excluded from `open_positions_count`
3. `by_asset` aggregates metrics per asset

**Caching**:
- Portfolio metrics cached in memory with TTL: 5-10 seconds (configurable via `POSITION_MANAGER_METRICS_CACHE_TTL`)
- Cache invalidated on any position update
- Optional: save to `portfolio_metrics_cache` table for historical analysis (future enhancement)

### Portfolio Metrics Cache (Optional)

Optional table for persisting portfolio metrics for historical analysis.

**Database Table**: `portfolio_metrics_cache` (optional, may be created later)

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, NOT NULL | Unique cache entry identifier |
| `total_exposure_usdt` | DECIMAL(20, 8) | NOT NULL | Total exposure at calculation time |
| `total_unrealized_pnl_usdt` | DECIMAL(20, 8) | NOT NULL | Total unrealized PnL |
| `total_realized_pnl_usdt` | DECIMAL(20, 8) | NOT NULL | Total realized PnL |
| `portfolio_value_usdt` | DECIMAL(20, 8) | NOT NULL | Portfolio value |
| `open_positions_count` | INTEGER | NOT NULL | Number of open positions |
| `calculated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Calculation timestamp |
| `expires_at` | TIMESTAMP | NOT NULL, INDEXED | Cache expiration timestamp |

**Indexes**:
- `idx_portfolio_metrics_expires_at` on `expires_at` - for cleanup queries

**Note**: This table is optional and may be created later for historical portfolio analysis. In-memory caching is sufficient for MVP.

## Relationships Summary

```
Position (1) ──< (many) PositionSnapshot
Position (many) ──> (1) Portfolio (aggregated, not stored)
```

## Data Validation

### Position Validation Rules

1. **Asset Format**: Must match pattern `^[A-Z]{3,10}USDT$` (e.g., "BTCUSDT", "ETHUSDT")
2. **Mode Values**: Must be one of: "one-way", "hedge"
3. **Size Constraints**: 
   - Can be positive (long), negative (short), or zero (closed)
   - For "one-way" mode: `size` represents net position
   - For "hedge" mode: `long_size` and `short_size` may be set separately
4. **Price Constraints**:
   - `average_entry_price` must be > 0 if `size != 0`
   - `current_price` can be NULL (excluded from exposure calculations)
5. **Version Constraints**:
   - `version` must be >= 1
   - `version` must increment on each successful update
6. **Timestamp Constraints**:
   - `created_at` <= `last_updated`
   - `closed_at` must be set when `size = 0`
   - `closed_at` >= `created_at` if set

### Conflict Resolution Rules

1. **Average Entry Price**:
   - If WebSocket `avgPrice` present and difference from existing > threshold (0.1%): use WebSocket `avgPrice`
   - Otherwise: keep existing `average_entry_price`
   - Threshold: `POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001` (0.1%)

2. **Position Size**:
   - Order Manager is source of truth for position size
   - WebSocket `size` used for validation/discrepancy detection only

3. **PnL Metrics**:
   - WebSocket is source of truth for `unrealized_pnl` and `realized_pnl`

4. **Current Price**:
   - Use `markPrice` from WebSocket events
   - If missing or stale: query external API (Bybit REST API)
   - Fallback to last known price if API fails

### Optimistic Locking

1. **Version Check**: Before update, check `version` matches expected value
2. **Retry Strategy**: Up to 3 retries with exponential backoff (100ms, 200ms, 400ms)
3. **Conflict Handling**: Log conflicts, raise exception if all retries fail
4. **Configuration**: 
   - `POSITION_MANAGER_OPTIMISTIC_LOCK_RETRIES=3`
   - `POSITION_MANAGER_OPTIMISTIC_LOCK_BACKOFF_BASE=100` (milliseconds)

## State Machine

### Position Lifecycle

```
[Created] ──> [Open] ──> [Updated] ──> [Closed]
                │            │
                └────────────┘
```

**States**:
- **Created**: Position created with `size != 0`, `version = 1`
- **Open**: Position has `size != 0`, actively traded
- **Updated**: Position modified (size, prices, PnL), `version` incremented
- **Closed**: Position has `size = 0`, `closed_at` set, retained for history

## Migration Requirements

### Required Migration (in ws-gateway service)

**Migration**: Add `current_price` and `version` fields to `positions` table

**Location**: `ws-gateway/migrations/` (per constitution - PostgreSQL migration ownership)

**SQL**:
```sql
-- Migration: add_current_price_and_version_to_positions
ALTER TABLE positions 
ADD COLUMN IF NOT EXISTS current_price DECIMAL(20, 8) NULL,
ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_positions_current_price ON positions(current_price);
CREATE INDEX IF NOT EXISTS idx_positions_version ON positions(version);

-- Update existing rows: set version = 1 for all existing positions
UPDATE positions SET version = 1 WHERE version IS NULL;
```

**Timing**: Must be executed before Position Manager service deployment

## Data Access Patterns

### Position Queries

1. **Get Position by Asset and Mode**:
   ```sql
   SELECT * FROM positions 
   WHERE asset = :asset AND mode = :mode
   ```

2. **Get All Open Positions**:
   ```sql
   SELECT * FROM positions 
   WHERE size != 0
   ORDER BY asset, mode
   ```

3. **Get Positions by Asset**:
   ```sql
   SELECT * FROM positions 
   WHERE asset = :asset
   ```

### Portfolio Metrics Queries

1. **Calculate Total Exposure**:
   ```sql
   SELECT COALESCE(SUM(ABS(size) * current_price), 0) as total_exposure
   FROM positions
   WHERE current_price IS NOT NULL
   ```

2. **Calculate Total PnL**:
   ```sql
   SELECT 
     COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
     COALESCE(SUM(realized_pnl), 0) as total_realized_pnl
   FROM positions
   ```

3. **Get Open Positions Count**:
   ```sql
   SELECT COUNT(*) as open_positions_count
   FROM positions
   WHERE size != 0
   ```

### Snapshot Queries

1. **Get Snapshots for Position**:
   ```sql
   SELECT * FROM position_snapshots
   WHERE position_id = :position_id
   ORDER BY created_at DESC
   LIMIT :limit OFFSET :offset
   ```

2. **Cleanup Old Snapshots**:
   ```sql
   DELETE FROM position_snapshots
   WHERE created_at < NOW() - INTERVAL '365 days'
   ```

## Summary

The data model consists of:
- **Position**: Core entity with composite key (asset, mode), supports optimistic locking via version field
- **Position Snapshot**: Historical records with complete position state, 1-year retention
- **Portfolio**: Aggregated metrics calculated on-demand, cached in memory
- **Portfolio Metrics Cache**: Optional table for historical analysis

All entities use shared PostgreSQL database. Migrations managed by `ws-gateway` service per constitution.

