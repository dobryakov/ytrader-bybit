# API Contracts: Order Manager

**Feature**: Order Manager Microservice  
**Date**: 2025-01-27

## Overview

This directory contains API contract specifications for the Order Manager REST API.

## Files

- `openapi.yaml` - OpenAPI 3.1.0 specification for the REST API
- `README.md` - This file

## API Endpoints

### Health Check

- `GET /health` - Service health status
- `GET /live` - Liveness probe (service is running)
- `GET /ready` - Readiness probe (service can accept requests, dependencies available)

### Order Management

- `GET /api/v1/orders` - List orders with filtering, pagination, and sorting
- `GET /api/v1/orders/{order_id}` - Get order details by Bybit order ID

### Position Management

- `GET /api/v1/positions` - List all positions
- `GET /api/v1/positions/{asset}` - Get position details for specific asset

### Synchronization

- `POST /api/v1/sync` - Trigger manual order state synchronization with Bybit

## Authentication

All endpoints (except `/health`, `/live`, `/ready`) require API key authentication via:
- Header: `X-API-Key: <api_key>`

## Usage Examples

### List Orders

```bash
curl -X GET "http://localhost:4600/api/v1/orders?asset=BTCUSDT&status=filled&page=1&page_size=20" \
  -H "X-API-Key: your-api-key"
```

### Get Order by ID

```bash
curl -X GET "http://localhost:4600/api/v1/orders/abc123-def456-ghi789" \
  -H "X-API-Key: your-api-key"
```

### Get Position for Asset

```bash
curl -X GET "http://localhost:4600/api/v1/positions/BTCUSDT" \
  -H "X-API-Key: your-api-key"
```

### Trigger Manual Synchronization

```bash
curl -X POST "http://localhost:4600/api/v1/sync" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "active"
  }'
```

### Check Service Health

```bash
curl -X GET "http://localhost:4600/health"
```

### Check Service Readiness

```bash
curl -X GET "http://localhost:4600/ready"
```

## Response Formats

### Order Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "order_id": "abc123-def456-ghi789",
  "signal_id": "550e8400-e29b-41d4-a716-446655440001",
  "asset": "BTCUSDT",
  "side": "Buy",
  "order_type": "Limit",
  "quantity": "0.1",
  "price": "50000.00",
  "status": "filled",
  "filled_quantity": "0.1",
  "average_price": "50000.50",
  "fees": "2.50",
  "created_at": "2025-01-27T10:00:00Z",
  "updated_at": "2025-01-27T10:00:05Z",
  "executed_at": "2025-01-27T10:00:05Z",
  "trace_id": "trace-12345",
  "is_dry_run": false
}
```

### Position Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "asset": "BTCUSDT",
  "size": "0.5",
  "average_entry_price": "50000.00",
  "unrealized_pnl": "250.00",
  "realized_pnl": "100.00",
  "mode": "one-way",
  "last_updated": "2025-01-27T10:00:00Z"
}
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "error": "Order not found",
  "code": "ORDER_NOT_FOUND",
  "details": {
    "order_id": "abc123"
  }
}
```

## Pagination

List endpoints support pagination:

```json
{
  "orders": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 150,
    "total_pages": 8
  }
}
```

