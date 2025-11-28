# Data Model: Grafana Monitoring Dashboard

**Feature**: Grafana Monitoring Dashboard  
**Date**: 2025-01-27  
**Data Sources**: PostgreSQL (shared database), RabbitMQ Management API, Service REST APIs

## Overview

This document defines the data models and data sources for the Grafana monitoring dashboard. The dashboard does not create or modify any data; it reads from existing data sources (PostgreSQL tables, RabbitMQ queues, and REST API endpoints) to provide monitoring visualizations.

## Data Sources

### 1. PostgreSQL Database (Read-Only)

Grafana connects to the shared PostgreSQL database using a read-only user account to query existing tables. No schema modifications are required.

#### Tables Accessed

| Table Name | Purpose | Key Fields |
|-----------|---------|-----------|
| `execution_events` | Order execution events with performance metrics | `id`, `signal_id`, `strategy_id`, `asset`, `side`, `execution_price`, `execution_quantity`, `execution_fees`, `executed_at`, `signal_price`, `signal_timestamp`, `performance` |
| `orders` | Order records with status and execution details | `id`, `order_id`, `signal_id`, `asset`, `side`, `order_type`, `quantity`, `price`, `status`, `filled_quantity`, `average_price`, `fees`, `created_at`, `updated_at`, `executed_at`, `is_dry_run` |
| `model_versions` | Model version information and training status | `id`, `version`, `file_path`, `model_type`, `strategy_id`, `trained_at`, `training_duration_seconds`, `training_dataset_size`, `is_active`, `is_warmup_mode`, `created_at`, `updated_at` |
| `model_quality_metrics` | Model quality and performance metrics | `id`, `model_version_id`, `metric_name`, `metric_value`, `metric_type`, `evaluated_at`, `evaluation_dataset_size`, `metadata` |
| `subscriptions` | WebSocket subscription records | `id`, `channel_type`, `symbol`, `topic`, `requesting_service`, `is_active`, `created_at`, `updated_at`, `last_event_at` |

#### Query Patterns

**Recent Trading Signals** (via execution_events):
- Query execution_events joined with signals (if signal table exists) or use execution_events as proxy
- Fields: signal_id, asset, side, execution_price (proxy for signal price), executed_at, strategy_id
- Limit: Last 100 records or last 24 hours

**Recent Orders**:
- Query orders table with execution_events for closure information
- Fields: order_id, signal_id, asset, side, execution_price, execution_quantity, execution_fees, executed_at, status
- Limit: Last 100 records or last 24 hours

**Model State**:
- Query model_versions for active models (is_active=true)
- Fields: version, strategy_id, is_active, is_warmup_mode, trained_at
- Join with model_quality_metrics for quality metrics

**Model Quality Metrics**:
- Query model_quality_metrics for latest metrics per active model
- Aggregate metrics: win rate, accuracy, total PnL
- Calculate from execution_events: win rate = successful orders / total orders

**Event History**:
- Aggregate from multiple tables:
  - Trading signals: execution_events (signal creation events)
  - Orders: orders table (created_at, executed_at, updated_at)
  - Model training: model_versions (trained_at for training completion)
  - Subscriptions: subscriptions table (created_at, updated_at for subscription events)

---

### 2. RabbitMQ Management API (HTTP/JSON)

Grafana queries RabbitMQ Management API to monitor queue health and metrics.

#### Endpoints Used

| Endpoint | Purpose | Response Fields |
|----------|---------|----------------|
| `/api/queues` | List all queues with metrics | `name`, `messages`, `messages_ready`, `messages_unacknowledged`, `message_stats` (publish_rate, ack_rate), `consumers` |
| `/api/queues/{vhost}/{queue}` | Detailed queue information | Same as above plus detailed statistics |

#### Metrics Extracted

- **Queue Length**: `messages` - Total messages in queue
- **Message Publishing Rate**: `message_stats.publish_details.rate` - Messages per second being published
- **Message Consumption Rate**: `message_stats.ack_details.rate` - Messages per second being consumed
- **Consumer Count**: `consumers` - Number of active consumers
- **Queue Lag Detection**: High `messages` count with low consumption rate indicates lag

#### Queue Monitoring

All queues in RabbitMQ are monitored, including:
- `model-service.trading_signals` - Trading signals queue
- System queues (if present): `amq.*`
- Other service queues (if any)

---

### 3. Service REST APIs (HTTP/JSON)

Grafana queries service health endpoints and statistics APIs via HTTP data source.

#### ws-gateway Service

**Endpoint**: `GET http://ws-gateway:4400/health`

**Response Fields**:
```json
{
  "status": "healthy" | "unhealthy",
  "service": "ws-gateway",
  "websocket_connected": boolean,
  "database_connected": boolean,
  "queue_connected": boolean,
  "active_subscriptions": integer
}
```

**Additional Endpoint** (if available):
- `GET /api/v1/subscriptions` - List active subscriptions (requires API key authentication)

#### model-service Service

**Endpoint**: `GET http://model-service:4500/health`

**Response Fields** (expected):
```json
{
  "status": "healthy" | "unhealthy",
  "service": "model-service",
  "database_connected": boolean,
  "queue_connected": boolean,
  "active_model_version": string,
  "training_status": "idle" | "training"
}
```

**Additional Endpoint** (if available):
- `GET /api/v1/models/statistics` - Model statistics and quality metrics (requires API key)

#### order-manager Service

**Endpoint**: `GET http://order-manager:4600/health`

**Response Fields** (expected):
```json
{
  "status": "healthy" | "unhealthy",
  "service": "order-manager",
  "database_connected": boolean,
  "queue_connected": boolean
}
```

---

## Dashboard Data Entities

### 1. Trading Signal (Read-Only View)

**Source**: PostgreSQL `execution_events` table (proxy) or RabbitMQ queue metrics

**Fields Displayed**:
- `signal_id` (UUID) - Signal identifier
- `asset` (VARCHAR) - Trading pair (e.g., "BTCUSDT")
- `side` (VARCHAR) - Buy or Sell
- `price` (DECIMAL) - Signal price (from execution_events.execution_price as proxy)
- `confidence` (DECIMAL) - Confidence score (if available in performance JSONB)
- `timestamp` (TIMESTAMP) - Signal timestamp (from executed_at)
- `strategy_id` (VARCHAR) - Strategy identifier

**Query**:
```sql
SELECT 
    signal_id,
    asset,
    side,
    execution_price as price,
    executed_at as timestamp,
    strategy_id,
    performance->>'confidence' as confidence
FROM execution_events
WHERE executed_at >= NOW() - INTERVAL '24 hours'
ORDER BY executed_at DESC
LIMIT 100;
```

---

### 2. Order Execution Event (Read-Only View)

**Source**: PostgreSQL `execution_events` and `orders` tables

**Fields Displayed**:
- `order_id` (VARCHAR) - Order identifier from exchange
- `signal_id` (UUID) - Related signal identifier
- `asset` (VARCHAR) - Trading pair
- `side` (VARCHAR) - Buy or Sell
- `execution_price` (DECIMAL) - Price at which order was executed
- `execution_quantity` (DECIMAL) - Quantity executed
- `execution_fees` (DECIMAL) - Fees paid
- `executed_at` (TIMESTAMP) - Execution timestamp
- `closure_status` (VARCHAR) - Status: filled, cancelled, rejected (from orders.status)

**Query**:
```sql
SELECT 
    e.id,
    o.order_id,
    e.signal_id,
    e.asset,
    e.side,
    e.execution_price,
    e.execution_quantity,
    e.execution_fees,
    e.executed_at,
    o.status as closure_status
FROM execution_events e
LEFT JOIN orders o ON e.signal_id = o.signal_id
WHERE e.executed_at >= NOW() - INTERVAL '24 hours'
ORDER BY e.executed_at DESC
LIMIT 100;
```

---

### 3. Model State (Read-Only View)

**Source**: PostgreSQL `model_versions` table

**Fields Displayed**:
- `active_model_version` (VARCHAR) - Version string (e.g., "v1.0")
- `strategy_id` (VARCHAR) - Strategy identifier
- `training_status` (VARCHAR) - "idle" or "training" (derived from model_versions.is_active and timestamps)
- `warmup_mode_status` (BOOLEAN) - Whether warm-up mode is active (from is_warmup_mode)
- `current_strategy_ids` (ARRAY) - List of active strategy IDs

**Query**:
```sql
SELECT 
    version as active_model_version,
    strategy_id,
    is_warmup_mode as warmup_mode_status,
    trained_at,
    CASE 
        WHEN is_active = true THEN 'active'
        ELSE 'inactive'
    END as training_status
FROM model_versions
WHERE is_active = true
ORDER BY trained_at DESC;
```

---

### 4. Model Quality Metrics (Read-Only View)

**Source**: PostgreSQL `model_quality_metrics` and `execution_events` tables

**Fields Displayed**:
- `win_rate` (DECIMAL) - Percentage of successful orders (calculated)
- `total_orders_count` (INTEGER) - Total orders executed
- `successful_orders_count` (INTEGER) - Orders with positive PnL
- `total_pnl` (DECIMAL) - Total profit and loss (calculated from execution_events.performance)

**Queries**:

**Quality Metrics from Database**:
```sql
SELECT 
    metric_name,
    metric_value,
    metric_type,
    evaluated_at
FROM model_quality_metrics
WHERE model_version_id IN (
    SELECT id FROM model_versions WHERE is_active = true
)
ORDER BY evaluated_at DESC;
```

**Win Rate Calculation**:
```sql
SELECT 
    COUNT(*) as total_orders_count,
    SUM(CASE WHEN (performance->>'realized_pnl')::DECIMAL > 0 THEN 1 ELSE 0 END) as successful_orders_count,
    ROUND(
        SUM(CASE WHEN (performance->>'realized_pnl')::DECIMAL > 0 THEN 1 ELSE 0 END)::DECIMAL / 
        NULLIF(COUNT(*), 0) * 100, 
        2
    ) as win_rate_percentage,
    SUM((performance->>'realized_pnl')::DECIMAL) as total_pnl
FROM execution_events
WHERE executed_at >= NOW() - INTERVAL '7 days'
GROUP BY strategy_id;
```

---

### 5. Queue Metrics (Read-Only View)

**Source**: RabbitMQ Management API

**Fields Displayed**:
- `queue_name` (STRING) - Queue identifier
- `queue_length` (INTEGER) - Current message count
- `message_publish_rate` (DECIMAL) - Messages per second being published
- `message_consume_rate` (DECIMAL) - Messages per second being consumed
- `consumer_count` (INTEGER) - Number of active consumers
- `lag_detected` (BOOLEAN) - Whether queue is backing up (queue_length > 1000 OR consume_rate < 10% of publish_rate)

**HTTP Query**:
- Endpoint: `GET http://rabbitmq:15672/api/queues`
- Authentication: Basic auth (RabbitMQ credentials from .env)
- Response: JSON array of queue objects

**Grafana Transformation**:
- Extract queue metrics from JSON response
- Calculate lag: `messages > 1000 OR (ack_rate / publish_rate < 0.1)`

---

### 6. Service Health Status (Read-Only View)

**Source**: Service REST API health endpoints

**Fields Displayed**:
- `service_name` (STRING) - Service identifier (ws-gateway, model-service, order-manager, postgres, rabbitmq)
- `overall_status` (STRING) - "healthy" or "unhealthy"
- `component_statuses` (JSON) - Database, queue, WebSocket connection status
- `error_information` (STRING) - Error messages if unhealthy

**HTTP Queries**:
- ws-gateway: `GET http://ws-gateway:4400/health`
- model-service: `GET http://model-service:4500/health`
- order-manager: `GET http://order-manager:4600/health`
- postgres: Docker health check status or connection test
- rabbitmq: `GET http://rabbitmq:15672/api/overview` (Management API)

---

### 7. WebSocket Connection State (Read-Only View)

**Source**: ws-gateway service health endpoint

**Fields Displayed**:
- `connection_status` (STRING) - "connected", "disconnected", "connecting", "reconnecting"
- `environment` (STRING) - "mainnet" or "testnet"
- `connection_duration` (INTEGER) - Time since connected (seconds)
- `last_heartbeat_timestamp` (TIMESTAMP) - Last heartbeat time (if available)
- `reconnection_count` (INTEGER) - Number of reconnection attempts
- `last_error_message` (STRING) - Last error if any
- `active_subscriptions_count` (INTEGER) - Number of active subscriptions

**HTTP Query**:
- Endpoint: `GET http://ws-gateway:4400/health`
- Response includes WebSocket connection state in JSON

---

### 8. Event History (Read-Only View)

**Source**: Aggregated from multiple PostgreSQL tables

**Fields Displayed**:
- `event_type` (STRING) - Type: "trading_signal_received", "order_created", "order_executed", "order_closed", "model_training_started", "model_training_completed", "subscription_created", "subscription_cancelled", "websocket_connected", "websocket_disconnected"
- `event_timestamp` (TIMESTAMP) - When event occurred
- `event_id` (UUID/STRING) - Event identifier
- `related_entity_ids` (JSON) - Signal ID, order ID, model version ID, subscription ID
- `event_details` (JSON) - Asset, side, price, status, channel type, symbol, requesting service
- `service_name` (STRING) - Source service

**Query** (Unified Event History):
```sql
-- Trading signal events (from execution_events)
SELECT 
    'trading_signal_received' as event_type,
    executed_at as event_timestamp,
    id::text as event_id,
    jsonb_build_object('signal_id', signal_id) as related_entity_ids,
    jsonb_build_object('asset', asset, 'side', side, 'price', execution_price, 'strategy_id', strategy_id) as event_details,
    'model-service' as service_name
FROM execution_events
WHERE executed_at >= NOW() - INTERVAL '24 hours'

UNION ALL

-- Order events (from orders table)
SELECT 
    CASE 
        WHEN status = 'filled' THEN 'order_executed'
        WHEN status IN ('cancelled', 'rejected') THEN 'order_closed'
        ELSE 'order_created'
    END as event_type,
    COALESCE(executed_at, created_at) as event_timestamp,
    id::text as event_id,
    jsonb_build_object('order_id', order_id, 'signal_id', signal_id) as related_entity_ids,
    jsonb_build_object('asset', asset, 'side', side, 'status', status) as event_details,
    'order-manager' as service_name
FROM orders
WHERE created_at >= NOW() - INTERVAL '24 hours'

UNION ALL

-- Model training events (from model_versions)
SELECT 
    CASE 
        WHEN is_active = true AND trained_at >= NOW() - INTERVAL '24 hours' THEN 'model_training_completed'
        ELSE NULL
    END as event_type,
    trained_at as event_timestamp,
    id::text as event_id,
    jsonb_build_object('model_version_id', id, 'version', version) as related_entity_ids,
    jsonb_build_object('strategy_id', strategy_id, 'model_type', model_type) as event_details,
    'model-service' as service_name
FROM model_versions
WHERE trained_at >= NOW() - INTERVAL '24 hours' AND is_active = true

UNION ALL

-- Subscription events (from subscriptions)
SELECT 
    CASE 
        WHEN is_active = true THEN 'subscription_created'
        ELSE 'subscription_cancelled'
    END as event_type,
    COALESCE(updated_at, created_at) as event_timestamp,
    id::text as event_id,
    jsonb_build_object('subscription_id', id) as related_entity_ids,
    jsonb_build_object('channel_type', channel_type, 'symbol', symbol, 'requesting_service', requesting_service) as event_details,
    'ws-gateway' as service_name
FROM subscriptions
WHERE created_at >= NOW() - INTERVAL '24 hours' OR updated_at >= NOW() - INTERVAL '24 hours'

ORDER BY event_timestamp DESC
LIMIT 200;
```

---

## Data Access Patterns

### Read-Only Access Requirements

**PostgreSQL User Permissions**:
```sql
-- Create read-only user for Grafana
CREATE USER grafana_monitor WITH PASSWORD 'password_from_env';
GRANT CONNECT ON DATABASE ytrader TO grafana_monitor;
GRANT USAGE ON SCHEMA public TO grafana_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_monitor;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO grafana_monitor;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_monitor;
```

**RabbitMQ Access**:
- Management API access using existing RabbitMQ credentials
- Read-only access to queue metrics (no queue modification)

**REST API Access**:
- Health endpoints: No authentication required (public)
- Statistics endpoints: API key authentication via headers

---

## Validation Rules

### Data Source Connection Validation

- PostgreSQL: Verify connection with `SELECT 1` query
- RabbitMQ Management API: Verify `/api/overview` endpoint returns 200
- Service REST APIs: Verify `/health` endpoints return 200

### Dashboard Query Validation

- All queries must be read-only (SELECT only, no INSERT/UPDATE/DELETE)
- Queries should include time range filters (last 24 hours, last 100 records)
- Handle NULL values gracefully in aggregations

---

## Performance Considerations

### Query Optimization

- Use indexed columns for filtering (`executed_at`, `created_at`, `is_active`)
- Limit result sets (LIMIT 100-200)
- Use time-range filters to reduce data scanned
- Aggregate metrics at query time rather than scanning all records

### Caching Strategy

- Grafana dashboard auto-refresh: 60 seconds (configurable)
- PostgreSQL query results cached by Grafana (default cache TTL)
- HTTP data source results cached (configurable per data source)

### Resource Usage

- Dashboard queries should complete within 3 seconds
- Limit concurrent dashboard users to prevent database overload
- Use Grafana's query timeout settings (default: 30 seconds)

---

## Error Handling

### Data Source Failures

- **PostgreSQL unavailable**: Dashboard shows connection error, cached data displayed if available
- **RabbitMQ Management API unavailable**: Queue metrics panel shows "unavailable" status
- **Service REST API unavailable**: Health status shows "unknown" or "unavailable"

### Query Failures

- Invalid SQL queries: Grafana displays query error message
- Missing tables/columns: Dashboard panel shows error, other panels continue to function
- Timeout errors: Query timeout message displayed, suggest increasing time range or limiting results

### Graceful Degradation

- Dashboard continues to function if one data source fails
- Connection status indicators show which data sources are available
- Cached data displayed when fresh data unavailable

