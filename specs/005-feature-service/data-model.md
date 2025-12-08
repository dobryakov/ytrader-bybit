# Data Model: Feature Service

**Feature**: Feature Service for Real-Time Feature Computation and Dataset Building  
**Date**: 2025-01-27  
**Database**: PostgreSQL (shared database for metadata), Local filesystem (Parquet files for raw data)

## Overview

This document defines the data models for the Feature Service. Metadata entities are persisted in the shared PostgreSQL database. Raw market data is stored in Parquet format on local filesystem. In-memory state includes orderbook state and rolling windows.

## Core Entities

### 1. Feature Vector

Represents a computed feature vector for a symbol at a specific timestamp. Feature vectors are computed in real-time and published to message queues or served via REST API.

**In-Memory Structure** (not a database table):

```python
{
    "timestamp": "datetime",           # Timestamp when features were computed
    "symbol": "string",                # Trading pair symbol (e.g., "BTCUSDT")
    "features": {
        # Price features
        "mid_price": float,
        "spread_abs": float,
        "spread_rel": float,
        "returns_1s": float,
        "returns_3s": float,
        "returns_1m": float,
        "vwap_3s": float,
        "vwap_15s": float,
        "vwap_1m": float,
        "volume_3s": float,
        "volume_15s": float,
        "volume_1m": float,
        "volatility_1m": float,
        "volatility_5m": float,
        
        # Orderflow features
        "signed_volume_3s": float,
        "signed_volume_15s": float,
        "signed_volume_1m": float,
        "buy_sell_volume_ratio": float,
        "trade_count_3s": int,
        "net_aggressor_pressure": float,
        
        # Orderbook features
        "depth_bid_top5": float,
        "depth_bid_top10": float,
        "depth_ask_top5": float,
        "depth_ask_top10": float,
        "depth_imbalance_top5": float,
        
        # Perpetual features
        "funding_rate": float,
        "time_to_funding": float,
        
        # Temporal/meta features
        "time_of_day_sin": float,      # Cyclic encoding: sin(2π * hour / 24)
        "time_of_day_cos": float       # Cyclic encoding: cos(2π * hour / 24)
    },
    "feature_registry_version": "string",  # Version of Feature Registry used
    "trace_id": "string"                  # Trace ID for request flow tracking
}
```

**Computation Intervals**: Features are computed at intervals: 1s, 3s, 15s, 1m (as specified in FR-003.1)

**State Transitions**:
- New market data arrives → Features recomputed → Feature vector updated → Published to `features.live` queue

**Relationships**:
- One symbol can have many feature vectors over time (time series)
- Feature vectors reference Feature Registry version for reproducibility

---

### 2. Dataset

Represents a structured dataset for model training, containing feature vectors and target variables, split into train/validation/test periods.

**Table**: `datasets` (PostgreSQL)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique dataset identifier |
| `symbol` | VARCHAR(20) | NOT NULL | Trading pair symbol (e.g., "BTCUSDT") |
| `status` | VARCHAR(20) | NOT NULL, DEFAULT 'building' | Status: 'building', 'ready', 'failed' |
| `split_strategy` | VARCHAR(50) | NOT NULL | Split strategy: 'time_based' or 'walk_forward' |
| `train_period_start` | TIMESTAMP | NULL | Train period start (for time_based) |
| `train_period_end` | TIMESTAMP | NULL | Train period end (for time_based) |
| `validation_period_start` | TIMESTAMP | NULL | Validation period start (for time_based) |
| `validation_period_end` | TIMESTAMP | NULL | Validation period end (for time_based) |
| `test_period_start` | TIMESTAMP | NULL | Test period start (for time_based) |
| `test_period_end` | TIMESTAMP | NULL | Test period end (for time_based) |
| `walk_forward_config` | JSONB | NULL | Walk-forward configuration (for walk_forward strategy) |
| `target_config` | JSONB | NOT NULL | Target configuration: type, horizon, threshold |
| `feature_registry_version` | VARCHAR(50) | NOT NULL | Feature Registry version used |
| `train_records` | INTEGER | DEFAULT 0 | Number of records in train split |
| `validation_records` | INTEGER | DEFAULT 0 | Number of records in validation split |
| `test_records` | INTEGER | DEFAULT 0 | Number of records in test split |
| `output_format` | VARCHAR(20) | NOT NULL, DEFAULT 'parquet' | Output format: 'parquet', 'csv', 'hdf5' |
| `storage_path` | VARCHAR(500) | NULL | Path to dataset files on filesystem |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When dataset build was requested |
| `completed_at` | TIMESTAMP | NULL | When dataset building completed |
| `estimated_completion` | TIMESTAMP | NULL | Estimated completion time (updated during build) |
| `error_message` | TEXT | NULL | Error message if status is 'failed' |

**Indexes**:
- `idx_datasets_status` on `status` (for filtering by status)
- `idx_datasets_symbol` on `symbol` (for symbol-specific queries)
- `idx_datasets_created_at` on `created_at` (for time-based queries)

**Validation Rules**:
- `status` must be one of: 'building', 'ready', 'failed'
- `split_strategy` must be one of: 'time_based', 'walk_forward'
- For `time_based`: train/validation/test periods must be specified and non-overlapping
- For `walk_forward`: `walk_forward_config` must be valid JSON with required fields
- `target_config` must contain: `type` (regression/classification), `horizon` (integer, prediction horizon in seconds), `threshold` (for classification)
- Periods must be in chronological order: train_start < train_end < validation_start < validation_end < test_start < test_end

**State Transitions**:
- `building` → `ready`: Dataset building completed successfully
- `building` → `failed`: Dataset building failed (error_message populated)
- `ready`: Dataset available for download

**Relationships**:
- One symbol can have many datasets (different time periods, configurations)
- Dataset references Feature Registry version for reproducibility

**Storage**:
- Dataset files stored on local filesystem in Parquet format (or specified format)
- Path structure: `datasets/{dataset_id}/{split}.parquet` (e.g., `datasets/{id}/train.parquet`)

---

### 3. Feature Registry

Represents the configuration that defines which features to compute, their data sources, lookback windows, and data leakage prevention rules.

**Table**: `feature_registry_versions` (PostgreSQL)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `version` | VARCHAR(50) | PRIMARY KEY | Feature Registry version identifier |
| `config` | JSONB | NOT NULL | Feature Registry configuration (YAML converted to JSON) |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT false | Whether this version is currently active |
| `validated_at` | TIMESTAMP | NULL | When configuration was validated |
| `validation_errors` | TEXT[] | NULL | Array of validation error messages (if any) |
| `loaded_at` | TIMESTAMP | NULL | When configuration was loaded into service |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When version was created |
| `created_by` | VARCHAR(100) | NULL | User/service identifier who created this version (for audit trail) |
| `activated_by` | VARCHAR(100) | NULL | User/service identifier who activated this version (for audit trail) |
| `rollback_from` | VARCHAR(50) | NULL | Version identifier that was rolled back from (if this version is result of rollback) |
| `previous_version` | VARCHAR(50) | NULL | Previous active version before this one (for rollback capability) |
| `schema_version` | VARCHAR(50) | NULL | Schema version for compatibility checking (optional, for tracking breaking changes) |
| `migration_script` | TEXT | NULL | Optional migration script or instructions for automatic schema migration |
| `compatibility_warnings` | TEXT[] | NULL | Array of backward compatibility warnings (e.g., removed features, changed names) |
| `breaking_changes` | TEXT[] | NULL | Array of breaking changes detected during compatibility check |
| `activation_reason` | TEXT | NULL | Optional reason/message for version activation (for audit trail) |

**Indexes**:
- `idx_feature_registry_active` on `is_active` (for finding active version)
- `idx_feature_registry_created_at` on `created_at` (for version history queries)
- `idx_feature_registry_previous_version` on `previous_version` (for rollback queries)

**Validation Rules**:
- `config` must be valid JSONB containing feature definitions
- Each feature in config must specify:
  - `name`: Feature name
  - `input_sources`: List of data sources (trades/orderbook/kline)
  - `lookback_window`: Time window into past (e.g., "3s", "1m")
  - `lookahead_forbidden`: Boolean flag (must be true)
  - `max_lookback_days`: Maximum lookback in days for validation
  - `data_sources`: List with timestamp requirements
- Configuration must pass data leakage validation (no future data usage)
- Only one version can have `is_active=true` at a time

**State Transitions**:
- New version created → `is_active=false`, `validated_at=NULL`, `previous_version` set to current active version
- Validation succeeds → `validated_at` set, `validation_errors=NULL`, `compatibility_warnings` and `breaking_changes` populated if detected
- Validation fails → `validation_errors` populated
- Version activated → Previous active version's `is_active` set to false, new version's `is_active` set to true, `loaded_at` set, `activated_by` set, automatic migration executed if schema changes detected
- Activation fails → Automatic rollback: previous version's `is_active` restored to true, failed version's `rollback_from` set to previous version, validation/migration errors logged
- Rollback triggered → Current active version's `is_active` set to false, previous version's `is_active` set to true, `rollback_from` set on current version, `activated_by` set on previous version

**Configuration Structure** (YAML/JSON):

```yaml
version: "1.0.0"
features:
  - name: "mid_price"
    input_sources: ["orderbook"]
    lookback_window: "0s"
    lookahead_forbidden: true
    max_lookback_days: 0
    data_sources:
      - source: "orderbook"
        timestamp_required: true
  - name: "returns_1m"
    input_sources: ["kline"]
    lookback_window: "1m"
    lookahead_forbidden: true
    max_lookback_days: 1
    data_sources:
      - source: "kline"
        timestamp_required: true
  # ... more features
```

---

### 4. Data Quality Report

Represents a data quality report for a symbol over a specified time period, tracking missing data, anomalies, sequence gaps, and desynchronization events.

**Table**: `data_quality_reports` (PostgreSQL)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique report identifier |
| `symbol` | VARCHAR(20) | NOT NULL | Trading pair symbol |
| `period_start` | TIMESTAMP | NOT NULL | Report period start |
| `period_end` | TIMESTAMP | NOT NULL | Report period end |
| `missing_rate` | DECIMAL(5, 4) | NOT NULL | Missing data rate (0.0 to 1.0) |
| `anomaly_rate` | DECIMAL(5, 4) | NOT NULL | Anomaly detection rate (0.0 to 1.0) |
| `sequence_gaps` | INTEGER | NOT NULL, DEFAULT 0 | Number of sequence gaps detected |
| `desynchronization_events` | INTEGER | NOT NULL, DEFAULT 0 | Number of orderbook desynchronization events |
| `anomaly_details` | JSONB | NULL | Detailed anomaly information |
| `sequence_gap_details` | JSONB | NULL | Detailed sequence gap information |
| `recommendations` | TEXT[] | NULL | Recommendations for data quality improvement |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When report was generated |

**Indexes**:
- `idx_data_quality_symbol` on `symbol` (for symbol-specific queries)
- `idx_data_quality_period` on `(period_start, period_end)` (for time-based queries)
- `idx_data_quality_created_at` on `created_at` (for recent reports)

**Validation Rules**:
- `period_start` < `period_end`
- `missing_rate` and `anomaly_rate` must be between 0.0 and 1.0
- All counts must be non-negative integers

**Relationships**:
- One symbol can have many data quality reports (different time periods)

---

### 5. Raw Market Data

Represents raw market data stored in Parquet format on local filesystem. Not stored in database, but metadata may be tracked.

**File Structure** (Parquet files on filesystem):

```
data/
├── orderbook/
│   ├── snapshots/
│   │   └── 2025-01-27/
│   │       └── BTCUSDT.parquet
│   └── deltas/
│       └── 2025-01-27/
│           └── BTCUSDT.parquet
├── trades/
│   └── 2025-01-27/
│       └── BTCUSDT.parquet
├── klines/
│   └── 2025-01-27/
│       └── BTCUSDT.parquet
├── ticker/
│   └── 2025-01-27/
│       └── BTCUSDT.parquet
└── funding/
    └── 2025-01-27/
        └── BTCUSDT.parquet
```

**Parquet Schema Examples**:

**Orderbook Snapshot**:
- `timestamp`: Timestamp
- `symbol`: String
- `sequence`: Integer
- `bids`: Array of [price, quantity] pairs
- `asks`: Array of [price, quantity] pairs
- `internal_timestamp`: Timestamp (when received)
- `exchange_timestamp`: Timestamp (from exchange)

**Orderbook Delta**:
- `timestamp`: Timestamp
- `symbol`: String
- `sequence`: Integer
- `delta_type`: String (insert/update/delete)
- `side`: String (bid/ask)
- `price`: Decimal
- `quantity`: Decimal
- `internal_timestamp`: Timestamp
- `exchange_timestamp`: Timestamp

**Trade**:
- `timestamp`: Timestamp
- `symbol`: String
- `price`: Decimal
- `quantity`: Decimal
- `side`: String (Buy/Sell)
- `trade_time`: Timestamp (from exchange)
- `internal_timestamp`: Timestamp

**Retention**: Raw data retained for minimum 90 days before archiving/deletion (per FR-002.3)

---

### 6. Orderbook State (In-Memory)

Represents the current orderbook state for a symbol, maintained in memory for real-time processing.

**In-Memory Structure**:

```python
{
    "symbol": "string",
    "sequence": int,                    # Current sequence number
    "timestamp": "datetime",            # Last update timestamp
    "bids": SortedDict,                 # Sorted by price (descending): {price: quantity}
    "asks": SortedDict,                 # Sorted by price (ascending): {price: quantity}
    "last_snapshot_at": "datetime",     # When last snapshot was applied
    "delta_count": int                 # Number of deltas applied since snapshot
}
```

**State Transitions**:
- Snapshot received → Initialize orderbook state from snapshot
- Delta received → Apply delta to orderbook state (insert/update/delete)
- Sequence gap detected → Request snapshot → Rebuild state from snapshot + deltas

**Storage**: Not persisted to database (in-memory only). For offline reconstruction, read snapshot + deltas from Parquet files.

---

### 7. Rolling Windows (In-Memory)

Represents rolling window state for time-based feature computations (1s, 3s, 15s, 1m).

**In-Memory Structure**:

```python
{
    "symbol": "string",
    "windows": {
        "1s": pandas.DataFrame,    # Rolling window for 1-second features
        "3s": pandas.DataFrame,    # Rolling window for 3-second features
        "15s": pandas.DataFrame,   # Rolling window for 15-second features
        "1m": pandas.DataFrame     # Rolling window for 1-minute features
    },
    "last_update": "datetime"
}
```

**Data Structure**: Each window is a pandas DataFrame with columns:
- `timestamp`: Timestamp
- `price`: Price (for price-based features)
- `volume`: Volume (for volume-based features)
- `side`: Side (Buy/Sell for orderflow features)
- Other fields as needed for feature computation

**State Transitions**:
- New market data arrives → Add to appropriate rolling windows → Remove old data outside window
- Feature computation → Use rolling window data to compute features

**Storage**: Not persisted to database (in-memory only). For offline mode, reconstruct windows from historical Parquet data.

---


## Database Schema

### Migration Script Structure

**Note**: Per constitution principle II (Shared Database Strategy), PostgreSQL migrations for Feature Service MUST be located in `ws-gateway/migrations/` (ws-gateway is the designated owner of all PostgreSQL migrations).

```sql
-- migrations/XXX_create_datasets_table.sql (in ws-gateway/migrations/)
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'building',
    split_strategy VARCHAR(50) NOT NULL,
    train_period_start TIMESTAMP,
    train_period_end TIMESTAMP,
    validation_period_start TIMESTAMP,
    validation_period_end TIMESTAMP,
    test_period_start TIMESTAMP,
    test_period_end TIMESTAMP,
    walk_forward_config JSONB,
    target_config JSONB NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    train_records INTEGER DEFAULT 0,
    validation_records INTEGER DEFAULT 0,
    test_records INTEGER DEFAULT 0,
    output_format VARCHAR(20) NOT NULL DEFAULT 'parquet',
    storage_path VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    estimated_completion TIMESTAMP,
    error_message TEXT,
    
    CONSTRAINT chk_status CHECK (status IN ('building', 'ready', 'failed')),
    CONSTRAINT chk_split_strategy CHECK (split_strategy IN ('time_based', 'walk_forward')),
    CONSTRAINT chk_periods_order CHECK (
        train_period_start IS NULL OR train_period_end IS NULL OR
        train_period_start < train_period_end
    )
);

CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_symbol ON datasets(symbol);
CREATE INDEX idx_datasets_created_at ON datasets(created_at);

-- migrations/XXX_create_feature_registry_versions_table.sql
CREATE TABLE IF NOT EXISTS feature_registry_versions (
    version VARCHAR(50) PRIMARY KEY,
    config JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT false,
    validated_at TIMESTAMP,
    validation_errors TEXT[],
    loaded_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_registry_active ON feature_registry_versions(is_active);

-- migrations/XXX_create_data_quality_reports_table.sql
CREATE TABLE IF NOT EXISTS data_quality_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    missing_rate DECIMAL(5, 4) NOT NULL,
    anomaly_rate DECIMAL(5, 4) NOT NULL,
    sequence_gaps INTEGER NOT NULL DEFAULT 0,
    desynchronization_events INTEGER NOT NULL DEFAULT 0,
    anomaly_details JSONB,
    sequence_gap_details JSONB,
    recommendations TEXT[],
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_period CHECK (period_start < period_end),
    CONSTRAINT chk_rates CHECK (missing_rate >= 0 AND missing_rate <= 1 AND anomaly_rate >= 0 AND anomaly_rate <= 1)
);

CREATE INDEX idx_data_quality_symbol ON data_quality_reports(symbol);
CREATE INDEX idx_data_quality_period ON data_quality_reports(period_start, period_end);
CREATE INDEX idx_data_quality_created_at ON data_quality_reports(created_at);
```

---

## Data Flow

### Feature Computation Flow

1. **Receive Market Data**: Consumer receives market data from RabbitMQ queues (`ws-gateway.*`)
2. **Update State**: Update orderbook state and rolling windows
3. **Compute Features**: Compute features using Feature Registry configuration
4. **Publish**: Publish feature vector to `features.live` queue
5. **Store Raw Data**: Write raw market data to Parquet files (async, non-blocking)

### Dataset Building Flow

1. **Request**: REST API receives dataset build request
2. **Validate**: Validate request (periods, Feature Registry version, data leakage check)
3. **Create Record**: Insert dataset record with status='building'
4. **Read Historical Data**: Read raw market data from Parquet files for specified periods
5. **Reconstruct State**: Reconstruct orderbook state and rolling windows for each timestamp
6. **Compute Features**: Compute features identically to online mode
7. **Compute Targets**: Compute target variables (returns, direction) for specified horizons
8. **Split Data**: Split into train/validation/test periods
9. **Write Dataset**: Write dataset splits to Parquet files
10. **Update Record**: Update dataset record with status='ready', record counts, storage_path
11. **Notify**: Publish completion notification to `features.dataset.ready` queue

### Feature Registry Management Flow

1. **Load Configuration**: Load Feature Registry configuration (YAML/JSON)
2. **Validate**: Validate configuration (temporal boundaries, data leakage prevention)
3. **Store Version**: Insert or update version in `feature_registry_versions` table
4. **Activate**: Set `is_active=true` for new version, `is_active=false` for previous version
5. **Reload**: Service reloads active configuration for feature computation

---

## Validation and Constraints

### Feature Vector Validation

- All feature values must be valid numbers (float or int)
- Timestamp must be valid and not in the future
- Symbol must be valid trading pair
- Feature Registry version must exist

### Dataset Validation

- Periods must be non-overlapping and in chronological order
- Target configuration must be valid (type, horizon, threshold)
- Feature Registry version must exist and be validated
- Storage path must be writable

### Feature Registry Validation

- Configuration must be valid YAML/JSON
- All features must specify required fields (lookback_window, lookahead_forbidden, etc.)
- No data leakage (features use only past data)
- Data sources must be available

---

## Performance Considerations

### Indexing Strategy

- Datasets: Indexed by `status` and `symbol` for fast queries
- Feature Registry: Indexed by `is_active` for finding active version
- Data Quality Reports: Indexed by `symbol` and time period for fast lookups

### Query Patterns

- **Latest Features**: In-memory lookup (no database query)
- **Dataset List**: `SELECT * FROM datasets WHERE status = $1 AND symbol = $2 ORDER BY created_at DESC`
- **Active Feature Registry**: `SELECT * FROM feature_registry_versions WHERE is_active = true LIMIT 1`
- **Data Quality Report**: `SELECT * FROM data_quality_reports WHERE symbol = $1 AND period_start >= $2 AND period_end <= $3 ORDER BY created_at DESC LIMIT 1`

### Scalability

- Feature vectors: In-memory only (not persisted), published to queues
- Datasets: Can grow large - use Parquet format for efficient storage and querying
- Raw data: Partitioned by date and symbol for efficient access
- Horizontal scaling: Service can scale by symbol (each instance handles subset of symbols)

---

## Future Considerations

### Potential Enhancements

- **Feature Vector History**: Optionally persist feature vectors for analysis (separate table or Parquet files)
- **Dataset Versioning**: Track dataset versions and changes over time
- **Feature Registry History**: Track all Feature Registry versions and changes
- **Data Quality Metrics**: Real-time data quality monitoring and alerting
- **Raw Data Compression**: Compress old Parquet files for storage efficiency

### Migration Notes

- All migrations must be reversible (per constitution requirement)
- Use `IF EXISTS` and `IF NOT EXISTS` for idempotent migrations
- Test migrations on testnet database before production
- **PostgreSQL Migration Ownership**: All PostgreSQL migrations MUST be located in `ws-gateway/migrations/` per constitution principle II

