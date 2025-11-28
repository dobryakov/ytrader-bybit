# Service REST API Endpoint Specifications

**Feature**: Grafana Monitoring Dashboard  
**Date**: 2025-01-27

## Overview

This document specifies the REST API endpoints used by Grafana to monitor service health and retrieve statistics. These endpoints are provided by existing services (ws-gateway, model-service, order-manager) and are queried via Grafana's HTTP data source.

---

## ws-gateway Service

### Health Check Endpoint

**Endpoint**: `GET /health`

**Base URL**: `http://ws-gateway:4400`

**Authentication**: None required

**Response**:
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

**Response Fields**:
- `status` (string): Overall service health status
- `service` (string): Service identifier
- `websocket_connected` (boolean): Whether WebSocket connection to Bybit is active
- `database_connected` (boolean): Whether PostgreSQL connection is active
- `queue_connected` (boolean): Whether RabbitMQ connection is active
- `active_subscriptions` (integer): Number of active WebSocket subscriptions

**Usage in Grafana**:
- Query endpoint via HTTP data source
- Extract `status` field for overall health indicator
- Extract `websocket_connected`, `database_connected`, `queue_connected` for component health
- Extract `active_subscriptions` for subscription count metric

**Example Query**:
```json
{
  "url": "http://ws-gateway:4400/health",
  "method": "GET",
  "headers": {}
}
```

---

### WebSocket Connection State

**Endpoint**: `GET /health` (includes WebSocket state)

**Base URL**: `http://ws-gateway:4400`

**Authentication**: None required

**Response** (includes WebSocket connection details):
```json
{
  "status": "healthy",
  "service": "ws-gateway",
  "websocket_connected": true,
  "websocket_state": {
    "status": "connected" | "disconnected" | "connecting" | "reconnecting",
    "environment": "testnet" | "mainnet",
    "connected_at": "2025-01-27T10:00:00Z",
    "last_heartbeat": "2025-01-27T10:05:00Z",
    "reconnection_count": 0,
    "last_error": null
  },
  "database_connected": true,
  "queue_connected": true,
  "active_subscriptions": 5
}
```

**Note**: If `/health` endpoint doesn't include detailed WebSocket state, additional endpoint may be required:
- `GET /api/v1/websocket/status` (if available)

**Fields Used for Monitoring**:
- `websocket_state.status` - Connection status
- `websocket_state.environment` - Mainnet or testnet
- `websocket_state.connected_at` - Connection timestamp (for duration calculation)
- `websocket_state.last_heartbeat` - Last heartbeat timestamp
- `websocket_state.reconnection_count` - Number of reconnections
- `websocket_state.last_error` - Last error message (if any)

---

## model-service Service

### Health Check Endpoint

**Endpoint**: `GET /health`

**Base URL**: `http://model-service:4500`

**Authentication**: None required (public health endpoint)

**Expected Response**:
```json
{
  "status": "healthy" | "unhealthy",
  "service": "model-service",
  "database_connected": boolean,
  "queue_connected": boolean,
  "active_model_version": "v1.0" | null,
  "training_status": "idle" | "training"
}
```

**Response Fields**:
- `status` (string): Overall service health status
- `service` (string): Service identifier
- `database_connected` (boolean): Whether PostgreSQL connection is active
- `queue_connected` (boolean): Whether RabbitMQ connection is active
- `active_model_version` (string, nullable): Current active model version (if available)
- `training_status` (string): Training status ("idle" or "training")

**Usage in Grafana**:
- Query endpoint via HTTP data source
- Extract `status` for health indicator
- Extract `active_model_version` and `training_status` for model state panel

---

### Model Statistics Endpoint (if available)

**Endpoint**: `GET /api/v1/models/statistics`

**Base URL**: `http://model-service:4500`

**Authentication**: API key via `X-API-Key` header

**Expected Response** (if endpoint exists):
```json
{
  "active_model": {
    "version": "v1.0",
    "strategy_id": "momentum_v1",
    "trained_at": "2025-01-27T08:00:00Z",
    "is_warmup_mode": false
  },
  "quality_metrics": {
    "win_rate": 0.65,
    "total_orders": 100,
    "successful_orders": 65,
    "total_pnl": 1500.50
  },
  "training_status": "idle"
}
```

**Usage in Grafana**:
- Query endpoint with API key authentication
- Extract quality metrics for dashboard panels
- Display model state and performance metrics

**Note**: If this endpoint doesn't exist, Grafana will query PostgreSQL directly for model statistics (see `data-model.md`).

---

## order-manager Service

### Health Check Endpoint

**Endpoint**: `GET /health`

**Base URL**: `http://order-manager:4600`

**Authentication**: None required (public health endpoint)

**Expected Response**:
```json
{
  "status": "healthy" | "unhealthy",
  "service": "order-manager",
  "database_connected": boolean,
  "queue_connected": boolean
}
```

**Response Fields**:
- `status` (string): Overall service health status
- `service` (string): Service identifier
- `database_connected` (boolean): Whether PostgreSQL connection is active
- `queue_connected` (boolean): Whether RabbitMQ connection is active

**Usage in Grafana**:
- Query endpoint via HTTP data source
- Extract `status` for health indicator
- Monitor database and queue connectivity

---

## RabbitMQ Management API

### Overview Endpoint (Health Check)

**Endpoint**: `GET /api/overview`

**Base URL**: `http://rabbitmq:15672`

**Authentication**: Basic Auth (RabbitMQ credentials)

**Response**:
```json
{
  "management_version": "3.13.0",
  "rates_mode": "basic",
  "rabbitmq_version": "3.13.0",
  "cluster_name": "rabbit@rabbitmq",
  "erlang_version": "26.2",
  "node": "rabbit@rabbitmq"
}
```

**Usage**: RabbitMQ service health check

---

### Queues Endpoint

**Endpoint**: `GET /api/queues`

**Base URL**: `http://rabbitmq:15672`

**Authentication**: Basic Auth (RabbitMQ credentials)

**Response**:
```json
[
  {
    "name": "model-service.trading_signals",
    "vhost": "/",
    "messages": 10,
    "messages_ready": 10,
    "messages_unacknowledged": 0,
    "consumers": 1,
    "message_stats": {
      "publish_details": {
        "rate": 5.2
      },
      "ack_details": {
        "rate": 5.1
      }
    }
  }
]
```

**Response Fields**:
- `name` (string): Queue name
- `messages` (integer): Total messages in queue
- `messages_ready` (integer): Messages ready for delivery
- `messages_unacknowledged` (integer): Messages delivered but not acknowledged
- `consumers` (integer): Number of active consumers
- `message_stats.publish_details.rate` (number): Messages per second being published
- `message_stats.ack_details.rate` (number): Messages per second being consumed/acknowledged

**Usage in Grafana**:
- Query endpoint with Basic Auth
- Parse JSON array response
- Extract queue metrics for each queue
- Calculate lag: `messages > 1000 OR (ack_rate / publish_rate < 0.1)`

---

## Error Responses

### Service Unavailable

**HTTP Status**: 503 Service Unavailable

**Response**:
```json
{
  "error": "Service unavailable",
  "message": "Service is currently down or not responding"
}
```

**Grafana Handling**:
- Dashboard panel shows "unavailable" status
- Health status indicator shows "unhealthy"

### Authentication Failure

**HTTP Status**: 401 Unauthorized

**Response**:
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key or credentials"
}
```

**Grafana Handling**:
- Check API key configuration
- Display authentication error in panel

### Invalid Request

**HTTP Status**: 400 Bad Request

**Response**:
```json
{
  "error": "Bad Request",
  "message": "Invalid request parameters"
}
```

**Grafana Handling**:
- Verify endpoint URL and parameters
- Display error message in panel

---

## Rate Limiting

### Service Health Endpoints

- No rate limiting expected (health endpoints are lightweight)
- Grafana queries every 60 seconds (configurable)

### Statistics Endpoints

- Rate limiting may apply if endpoints exist
- Respect `Retry-After` header if 429 Too Many Requests returned
- Increase query interval if rate limits encountered

---

## Timeout Settings

**Grafana HTTP Data Source Timeouts**:
- Query timeout: 10 seconds (health endpoints)
- Query timeout: 30 seconds (statistics endpoints, if available)
- Overall dashboard refresh: 60 seconds

**Recommendation**: Services should respond to health endpoints within 1 second for optimal dashboard performance.

