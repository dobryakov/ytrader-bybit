# API Contracts: Grafana Monitoring Dashboard

**Feature**: Grafana Monitoring Dashboard  
**Date**: 2025-01-27

## Overview

This directory contains data source and API specifications for the Grafana monitoring dashboard. Grafana connects to multiple data sources (PostgreSQL, RabbitMQ Management API, and service REST APIs) to display monitoring dashboards.

## Files

- `datasources.yml` - Grafana data source provisioning configuration
- `api-endpoints.md` - Service REST API endpoint specifications
- `README.md` - This file

## Data Sources

### 1. PostgreSQL Database

**Type**: PostgreSQL data source (built-in plugin)

**Connection Details**:
- Host: `${POSTGRES_HOST}` (default: `postgres`)
- Port: `${POSTGRES_PORT}` (default: `5432`)
- Database: `${POSTGRES_DB}` (default: `ytrader`)
- User: `${GRAFANA_POSTGRES_USER}` (read-only user)
- Password: `${GRAFANA_POSTGRES_PASSWORD}`

**Tables Accessed** (read-only):
- `execution_events` - Order execution events
- `orders` - Order records
- `model_versions` - Model version information
- `model_quality_metrics` - Model quality metrics
- `subscriptions` - WebSocket subscription records

**Query Examples**:
See `data-model.md` for detailed SQL query patterns.

---

### 2. RabbitMQ Management API

**Type**: HTTP/JSON data source

**Connection Details**:
- URL: `http://rabbitmq:15672`
- Authentication: Basic Auth
- Username: `${RABBITMQ_USER}`
- Password: `${RABBITMQ_PASSWORD}`

**Endpoints Used**:
- `GET /api/queues` - List all queues with metrics
- `GET /api/queues/{vhost}/{queue}` - Detailed queue information
- `GET /api/overview` - RabbitMQ cluster overview (for health check)

**Response Format**: JSON

**Example Response** (`/api/queues`):
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

**Grafana Configuration**:
- Use HTTP/JSON data source or JSON API plugin
- Parse JSON response using JSON path expressions
- Transform data for visualization (table, time series, stat panels)

---

### 3. Service REST API Endpoints

See `api-endpoints.md` for detailed endpoint specifications.

**Services**:
- `ws-gateway` (port 4400)
- `model-service` (port 4500)
- `order-manager` (port 4600)

**Authentication**:
- Health endpoints: No authentication required
- Statistics endpoints: API key authentication via `X-API-Key` header (if required)

---

## Data Source Provisioning

Grafana uses provisioning configuration files to automatically configure data sources on container startup. Configuration file location:

```
grafana/provisioning/datasources/datasources.yml
```

**Environment Variable Substitution**:
- Grafana provisioning files support environment variable substitution
- Variables are replaced at container startup from `.env` file
- Format: `${VARIABLE_NAME}` or `${VARIABLE_NAME:-default_value}`

**Note**: Standard Grafana provisioning may require variable substitution to be handled via configuration management script or using Grafana's environment variable features.

---

## Dashboard Provisioning

Dashboards are provisioned via JSON files in:
```
grafana/provisioning/dashboards/dashboards.yml
grafana/dashboards/*.json
```

Dashboard JSON files define:
- Panel configurations
- Data source references
- Query definitions
- Visualization types

---

## Authentication

### Grafana UI Access

**Method**: Basic Authentication (username/password)

**Configuration**:
- Username: `${GRAFANA_ADMIN_USER}` (default: `admin`)
- Password: `${GRAFANA_ADMIN_PASSWORD}` (default: `admin`)

**Environment Variables**:
- `GF_SECURITY_ADMIN_USER` - Grafana admin username
- `GF_SECURITY_ADMIN_PASSWORD` - Grafana admin password

### Data Source Authentication

**PostgreSQL**: Username/password authentication (read-only user)

**RabbitMQ Management API**: Basic Auth (RabbitMQ credentials from `.env`)

**Service REST APIs**: 
- Health endpoints: No authentication
- Statistics endpoints: API key via `X-API-Key` header (if required)

---

## Error Handling

### Data Source Connection Failures

- **PostgreSQL unavailable**: Dashboard panel shows connection error, cached data displayed if available
- **RabbitMQ Management API unavailable**: Queue metrics panel shows "unavailable" status
- **Service REST API unavailable**: Health status shows "unknown"

### Query Failures

- Invalid SQL queries: Panel displays query error message
- Missing tables/columns: Panel shows error, other panels continue to function
- Timeout errors: Query timeout message displayed

### Graceful Degradation

- Dashboard continues to function if one data source fails
- Connection status indicators show which data sources are available
- Cached data displayed when fresh data unavailable

---

## Security Considerations

### Read-Only Access

- PostgreSQL user has SELECT-only permissions
- No INSERT, UPDATE, DELETE, or DDL permissions
- Cannot modify database schema or data

### Network Isolation

- Grafana container connects to other services via Docker network
- No external network access required (internal Docker network only)
- External access to Grafana UI only (port 4700)

### Credential Management

- All credentials stored in `.env` file (not committed to git)
- `.env.example` provides template without sensitive values
- Credentials passed to containers via environment variables

---

## Performance Considerations

### Query Timeouts

- PostgreSQL queries: 30 seconds default
- HTTP API queries: 10 seconds default
- Dashboard auto-refresh: 60 seconds (configurable)

### Connection Pooling

- PostgreSQL: max 100 connections, idle timeout 4 hours
- HTTP data sources: connection reuse enabled

### Caching

- Grafana caches query results (default TTL)
- Dashboard refresh interval: 60 seconds (configurable)
- Reduce refresh frequency for heavy queries

---

## Monitoring the Monitor

### Grafana Health Check

**Endpoint**: `GET http://grafana:4700/api/health`

**Response**:
```json
{
  "commit": "abc123",
  "database": "ok",
  "version": "10.4.0"
}
```

### Container Health Check

Docker Compose health check:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

