# Data Model: Model Service

**Feature**: Model Service - Trading Decision and ML Training Microservice  
**Date**: 2025-01-27  
**Database**: PostgreSQL (shared database)

## Overview

This document defines the data models for the Model Service. The service stores model metadata and quality metrics in the shared PostgreSQL database, while model files are stored on the file system. Trading signals and order execution events are exchanged via RabbitMQ message queues (not persisted by this service).

## Core Entities

### 1. Model Version

Represents a trained ML model instance with version identifier, training metadata, quality metrics, and file system path. Models are stored as files on the file system, while metadata is persisted in PostgreSQL for querying, versioning, and rollback capabilities.

**Table**: `model_versions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique model version identifier |
| `version` | VARCHAR(50) | NOT NULL, UNIQUE | Human-readable version identifier (e.g., 'v1', 'v2.1') |
| `file_path` | VARCHAR(500) | NOT NULL | File system path to model file (e.g., '/models/v1/model.json') |
| `model_type` | VARCHAR(50) | NOT NULL | Model type (e.g., 'xgboost', 'random_forest', 'logistic_regression') |
| `strategy_id` | VARCHAR(100) | NULL | Trading strategy identifier (NULL for general models, supports multiple strategies) |
| `symbol` | VARCHAR(20) | NULL | Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT') - NULL for universal models that work across all symbols |
| `trained_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When model training completed |
| `training_duration_seconds` | INTEGER | NULL | Training duration in seconds |
| `training_dataset_size` | INTEGER | NULL | Number of records in training dataset |
| `training_config` | JSONB | NULL | Training configuration parameters (hyperparameters, feature set, etc.) |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT false | Whether this version is currently active (only one active per (strategy_id, symbol) combination) |
| `is_warmup_mode` | BOOLEAN | NOT NULL, DEFAULT false | Whether system is in warm-up mode (no trained model) |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Record creation timestamp |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last update timestamp |

**Indexes**:
- `idx_model_versions_version` on `version` (for version lookup)
- `idx_model_versions_strategy_id` on `strategy_id` (for strategy-specific queries)
- `idx_model_versions_strategy_symbol` on `(strategy_id, symbol)` WHERE `symbol IS NOT NULL` (for symbol-specific model lookup)
- `idx_model_versions_active` on `(strategy_id, symbol, is_active)` WHERE `is_active = true` (for active model lookup per strategy and symbol)
- `idx_model_versions_trained_at` on `trained_at DESC` (for version history queries)

**Validation Rules**:
- `model_type` must be one of: 'xgboost', 'random_forest', 'logistic_regression', 'sgd_classifier'
- `file_path` must be a valid file system path within `/models/` directory
- `version` must match pattern: `^v\d+(\.\d+)?$` (e.g., 'v1', 'v2.1')
- Only one model version can have `is_active=true` per `(strategy_id, symbol)` combination
  - For models with `symbol`: one active model per `(strategy_id, symbol)` pair
  - For universal models (`symbol IS NULL`): one active model per `strategy_id`
- `training_config` must be valid JSON

**State Transitions**:
- New model trained → `is_active=false`, `trained_at` set, `symbol` set from training dataset
- Model activated → Previous active model for same `(strategy_id, symbol)` `is_active=false`, new model `is_active=true`
- Model rollback → Previous version `is_active=true`, current version `is_active=false` (for same `(strategy_id, symbol)`)
- Warm-up mode → `is_warmup_mode=true`, all models `is_active=false` for strategy (can be symbol-specific or universal)

**Relationships**:
- One model version has many quality metrics (see Model Quality Metrics)
- Multiple model versions can exist per `strategy_id` (version history)

---

### 2. Model Quality Metrics

Represents quantitative measures of model performance used for model selection, transition decisions, and quality degradation detection.

**Table**: `model_quality_metrics`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique metric record identifier |
| `model_version_id` | UUID | NOT NULL, FOREIGN KEY REFERENCES model_versions(id) ON DELETE CASCADE | Associated model version |
| `metric_name` | VARCHAR(100) | NOT NULL | Metric name (e.g., 'accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'profit_factor') |
| `metric_value` | DECIMAL(20, 8) | NOT NULL | Metric value |
| `metric_type` | VARCHAR(50) | NOT NULL | Metric type: 'classification', 'regression', 'trading_performance' |
| `evaluated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When metric was calculated |
| `evaluation_dataset_size` | INTEGER | NULL | Number of records in evaluation dataset |
| `metadata` | JSONB | NULL | Additional metric metadata (confidence intervals, thresholds, etc.) |

**Indexes**:
- `idx_model_quality_metrics_model_version_id` on `model_version_id` (for model-specific queries)
- `idx_model_quality_metrics_metric_name` on `metric_name` (for metric-specific queries)
- `idx_model_quality_metrics_evaluated_at` on `evaluated_at DESC` (for time-based queries)

**Validation Rules**:
- `metric_type` must be one of: 'classification', 'regression', 'trading_performance'
- `metric_value` must be within reasonable bounds (e.g., accuracy 0-1, sharpe_ratio -10 to 10)
- `metric_name` must be a recognized metric (validation in application layer)

**Common Metrics**:
- **Classification**: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`
- **Trading Performance**: `sharpe_ratio`, `profit_factor`, `max_drawdown`, `win_rate`, `average_return`
- **Regression**: `mse`, `mae`, `r2_score`

**Relationships**:
- Many quality metrics belong to one model version
- Metrics are calculated after training and periodically re-evaluated

---

### 3. Training Dataset

Represents an aggregated collection of order execution events and associated market data organized for model training. This is a transient entity (not persisted to database) used during training operations.

**In-Memory Structure** (not a database table):

```python
{
    "dataset_id": "uuid",              # Unique identifier for this dataset
    "strategy_id": "string",           # Trading strategy identifier
    "features": "DataFrame",           # pandas DataFrame with feature columns
    "labels": "Series",                # pandas Series with target labels (buy/sell signals)
    "metadata": {
        "record_count": 1000000,       # Number of records
        "date_range": {
            "start": "2025-01-01",
            "end": "2025-01-27"
        },
        "feature_names": ["price", "volume", "rsi", ...],
        "data_quality_score": 0.95,   # Data quality assessment
        "coverage": {                   # Market coverage
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "time_periods": "all"
        }
    },
    "created_at": "datetime",          # When dataset was created
    "source_events": ["uuid", ...]     # Order execution event IDs (if tracked)
}
```

**Processing Flow**:
1. Order execution events consumed from RabbitMQ
2. Events aggregated with market data (from shared database or message queue)
3. Features engineered (technical indicators, price patterns, etc.)
4. Labels extracted (profitability, signal direction, etc.)
5. Dataset validated and quality assessed
6. Dataset used for model training
7. Dataset discarded after training (not persisted)

**Feature Engineering**:
- Price features: current price, price change, volatility
- Volume features: volume, volume change, volume ratio
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages
- Market context: order book depth, spread, liquidity
- Execution features: execution price vs signal price, slippage, fees

**Label Generation**:
- Binary classification: profitable trade (1) vs unprofitable trade (0)
- Multi-class: buy signal, sell signal, hold
- Regression: expected return, risk-adjusted return

---

### 4. Trading Signal

Represents a high-level trading decision generated by the model or warm-up heuristics. Published to RabbitMQ message queue for order manager consumption (not persisted by this service).

**Message Queue Structure** (RabbitMQ):

```json
{
    "signal_id": "uuid",               # Unique signal identifier
    "signal_type": "buy" | "sell",     # Trading signal type
    "asset": "BTCUSDT",                # Asset identifier (trading pair)
    "amount": 1000.50,                 # Amount in quote currency (USDT)
    "confidence": 0.85,                # Confidence score (0-1)
    "timestamp": "2025-01-27T10:00:00Z", # Signal generation timestamp
    "strategy_id": "momentum_v1",      # Trading strategy identifier
    "model_version": "v2.1",           # Model version used (NULL for warm-up mode)
    "is_warmup": false,                # Whether signal generated in warm-up mode
    "market_data_snapshot": {          # Market data at signal generation time (REQUIRED for training)
        "price": 50000.00,             # Current market price
        "spread": 1.00,                # Bid-ask spread
        "volume_24h": 1000000.00,      # 24-hour trading volume
        "volatility": 0.02,            # Current volatility measure
        "orderbook_depth": {            # Order book depth (if available)
            "bid_depth": 50.5,
            "ask_depth": 48.2
        },
        "technical_indicators": {      # Technical indicators at signal time (if calculated)
            "rsi": 65.5,
            "macd": 12.3,
            "moving_average_20": 49800.00
        }
    },
    "metadata": {                       # Additional signal metadata
        "reasoning": "High RSI + volume spike",
        "risk_score": 0.3,
        "expected_return": 0.02
    },
    "trace_id": "string"                # Trace ID for request flow tracking
}
```

**Validation Rules**:
- `signal_type` must be 'buy' or 'sell'
- `asset` must be a valid trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
- `amount` must be positive and within configured limits
- `confidence` must be between 0 and 1
- `timestamp` must be current or recent (not future, not too old)
- `strategy_id` must be a configured strategy identifier
- `market_data_snapshot` MUST be included and contain at minimum: `price`, `spread`, `volume_24h`, `volatility` (required for accurate feature engineering during model training)
- `market_data_snapshot.price` must be positive
- `market_data_snapshot.volatility` must be non-negative

**Queue**: `model-service.trading_signals`

**Processing Flow**:
1. Signal generated by model or warm-up heuristics
2. Signal validated (required fields, value ranges, format compliance)
3. Rate limiting checked (with burst allowance)
4. Signal published to RabbitMQ queue
5. Order manager consumes and executes signal

---

### 5. Order Execution Event

Represents an enriched event from the order manager microservice containing details about executed trades. Consumed from RabbitMQ message queue for training purposes (not persisted by this service).

**Message Queue Structure** (RabbitMQ, consumed from order manager):

```json
{
    "event_id": "uuid",                # Unique event identifier
    "order_id": "string",              # Order identifier from order manager
    "signal_id": "uuid",               # Original trading signal identifier
    "strategy_id": "string",           # Trading strategy identifier
    "asset": "BTCUSDT",                # Trading pair
    "side": "buy" | "sell",            # Order side
    "execution_price": 50000.00,       # Actual execution price
    "execution_quantity": 0.1,         # Executed quantity
    "execution_fees": 5.00,            # Total fees paid
    "executed_at": "2025-01-27T10:00:05Z", # Execution timestamp
    "signal_price": 50010.00,          # Original signal price (for slippage calculation)
    "signal_timestamp": "2025-01-27T10:00:00Z", # Original signal timestamp
    "market_conditions": {             # Market data at execution time
        "spread": 1.00,
        "volume_24h": 1000000.00,
        "volatility": 0.02
    },
    "performance": {                   # Trade performance metrics
        "slippage": -10.00,            # Price difference (execution - signal)
        "slippage_percent": -0.02,     # Slippage as percentage
        "realized_pnl": 50.00,         # Realized profit/loss (if closed)
        "return_percent": 0.01         # Return percentage
    },
    "trace_id": "string"               # Trace ID for request flow tracking
}
```

**Queue**: `order-manager.order_events` (consumed by model service)

**Processing Flow**:
1. Order execution event consumed from RabbitMQ
2. Event validated and parsed
3. Event aggregated with market data for training dataset
4. Event used for model training or accumulated for batch retraining
5. Event used for model quality evaluation (if trade closed)

---

### 6. Order/Position State

Represents current snapshot of open orders and positions retrieved from shared PostgreSQL database. Used as input for signal generation (read-only, not owned by this service).

**Database Tables** (shared database, owned by order manager service):

The model service reads from tables managed by the order manager service. The exact schema is defined by the order manager, but typically includes:

- **Open Orders**: Current pending orders (order_id, symbol, side, price, quantity, status, created_at)
- **Positions**: Current open positions (position_id, symbol, side, size, entry_price, unrealized_pnl, opened_at)

**Read Operations**:
- Retrieve current open orders for a strategy
- Retrieve current positions for a strategy
- Calculate aggregate position metrics (total exposure, unrealized P&L)

**Usage in Signal Generation**:
- Input features: current position size, entry price, unrealized P&L
- Risk management: prevent over-exposure, respect position limits
- Strategy context: adjust signals based on existing positions

---

## Database Schema

### Migration Script Structure

**Note**: All PostgreSQL migrations must be located in `ws-gateway/migrations/` per constitution (ws-gateway is the designated owner of all PostgreSQL migrations for the shared database).

```sql
-- ws-gateway/migrations/003_create_model_versions_table.sql
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(50) NOT NULL UNIQUE,
    file_path VARCHAR(500) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(100),
    symbol VARCHAR(20),
    trained_at TIMESTAMP NOT NULL DEFAULT NOW(),
    training_duration_seconds INTEGER,
    training_dataset_size INTEGER,
    training_config JSONB,
    is_active BOOLEAN NOT NULL DEFAULT false,
    is_warmup_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_model_type CHECK (model_type IN ('xgboost', 'random_forest', 'logistic_regression', 'sgd_classifier')),
    CONSTRAINT chk_version_format CHECK (version ~ '^v\d+(\.\d+)?$'),
    CONSTRAINT chk_file_path CHECK (file_path LIKE '/models/%')
);

CREATE INDEX idx_model_versions_version ON model_versions(version);
CREATE INDEX idx_model_versions_strategy_id ON model_versions(strategy_id);
CREATE INDEX idx_model_versions_strategy_symbol ON model_versions(strategy_id, symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_model_versions_active ON model_versions(strategy_id, symbol, is_active) WHERE is_active = true;
CREATE INDEX idx_model_versions_trained_at ON model_versions(trained_at DESC);

-- Unique constraint: only one active model per (strategy_id, symbol) combination
-- For models with symbol: one active model per strategy_id + symbol
-- For universal models (symbol IS NULL): one active model per strategy_id
CREATE UNIQUE INDEX idx_model_versions_unique_active ON model_versions(strategy_id, symbol, is_active) WHERE is_active = true AND symbol IS NOT NULL;
CREATE UNIQUE INDEX idx_model_versions_unique_active_no_symbol ON model_versions(strategy_id, is_active) WHERE is_active = true AND symbol IS NULL;

-- ws-gateway/migrations/004_create_model_quality_metrics_table.sql
CREATE TABLE IF NOT EXISTS model_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    evaluated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluation_dataset_size INTEGER,
    metadata JSONB,
    
    CONSTRAINT chk_metric_type CHECK (metric_type IN ('classification', 'regression', 'trading_performance')),
    CONSTRAINT chk_metric_value_bounds CHECK (
        (metric_type = 'classification' AND metric_value >= 0 AND metric_value <= 1) OR
        (metric_type IN ('regression', 'trading_performance'))
    )
);

CREATE INDEX idx_model_quality_metrics_model_version_id ON model_quality_metrics(model_version_id);
CREATE INDEX idx_model_quality_metrics_metric_name ON model_quality_metrics(metric_name);
CREATE INDEX idx_model_quality_metrics_evaluated_at ON model_quality_metrics(evaluated_at DESC);
```

## Relationships Summary

```
model_versions (1) ──< (many) model_quality_metrics
model_versions (many) ──> (1) strategy_id (logical grouping)
order_execution_events (consumed from queue) ──> training_dataset (transient)
training_dataset (transient) ──> model_versions (training input)
model_versions ──> trading_signals (generation output)
trading_signals ──> (published to queue) ──> order_manager
order/position_state (read from shared DB) ──> trading_signals (generation input)
```

## Data Flow

1. **Training Flow**:
   - Order execution events (RabbitMQ) → Training dataset (in-memory) → Model training → Model version (file system + database)

2. **Signal Generation Flow**:
   - Order/position state (shared DB) + Market data → Model inference → Trading signal → RabbitMQ queue

3. **Quality Evaluation Flow**:
   - Model version → Evaluation dataset → Quality metrics → Database

4. **Version Management Flow**:
   - New model trained → Quality metrics calculated → Model activation → Previous model deactivated

