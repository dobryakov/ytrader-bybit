# API Contracts: WebSocket Gateway

**Feature**: WebSocket Gateway for Bybit Data Aggregation and Routing  
**Date**: 2025-11-25

## Overview

This directory contains API contract specifications for the WebSocket Gateway REST API.

## Files

- `openapi.yaml` - OpenAPI 3.1.0 specification for the REST API
- `README.md` - This file

## API Endpoints

### Health Check

- `GET /health` - Service health status

### Subscription Management

- `POST /api/v1/subscriptions` - Create a new subscription
- `GET /api/v1/subscriptions` - List subscriptions (with optional filters)
- `GET /api/v1/subscriptions/{subscription_id}` - Get subscription details
- `DELETE /api/v1/subscriptions/{subscription_id}` - Cancel a subscription
- `DELETE /api/v1/subscriptions/by-service/{service_name}` - Cancel all subscriptions for a service

## Authentication

All endpoints (except `/health`) require API key authentication via:
- Header: `X-API-Key: <api_key>`
- Query parameter: `?api_key=<api_key>` (alternative)

## Channel Types

Supported channel types:
- `trades` - Trade execution data (requires `symbol`)
- `ticker` - 24h ticker data (requires `symbol`)
- `orderbook` - Order book updates (requires `symbol`)
- `order` - Order status updates (requires `symbol`)
- `balance` - Account balance updates (no `symbol` required)
- `kline` - K-line/candlestick data (requires `symbol`)
- `liquidation` - Liquidation events (no `symbol` required)

## Usage Examples

### Subscribe to Trades

```bash
curl -X POST http://localhost:8081/api/v1/subscriptions \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "trades",
    "symbol": "BTCUSDT",
    "requesting_service": "order-manager"
  }'
```

### Subscribe to Balance Updates

```bash
curl -X POST http://localhost:8081/api/v1/subscriptions \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "balance",
    "requesting_service": "order-manager"
  }'
```

### List Active Subscriptions

```bash
curl -X GET "http://localhost:8081/api/v1/subscriptions?is_active=true" \
  -H "X-API-Key: your-api-key"
```

### Cancel a Subscription

```bash
curl -X DELETE http://localhost:8081/api/v1/subscriptions/{subscription_id} \
  -H "X-API-Key: your-api-key"
```

### Health Check

```bash
curl http://localhost:8081/health
```

## Response Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid API key)
- `404` - Not Found
- `409` - Conflict (subscription already exists)
- `500` - Internal Server Error
- `503` - Service Unavailable (unhealthy)

## Event Delivery

Events are delivered via RabbitMQ queues, not through the REST API. The REST API is only for subscription management.

Queue naming: `ws-gateway.{event_type}` (e.g., `ws-gateway.trades`, `ws-gateway.balance`)

Subscribers consume events from these queues. See the main specification for event structure and delivery guarantees.

## Validation Rules

1. **Channel Type**: Must be one of the supported types
2. **Symbol**: Required for symbol-specific channels (trades, ticker, orderbook, order, kline). Must be NULL for non-symbol-specific channels (balance, liquidation)
3. **Topic**: If provided, must match Bybit WebSocket topic format. If not provided, will be auto-generated from channel_type and symbol
4. **Requesting Service**: Must be a non-empty string identifier

## Error Codes

- `INVALID_CHANNEL_TYPE` - Channel type not supported
- `MISSING_SYMBOL` - Symbol required but not provided
- `INVALID_SYMBOL` - Symbol format invalid
- `INVALID_TOPIC` - Topic format invalid
- `SUBSCRIPTION_EXISTS` - Subscription already exists for this service
- `AUTHENTICATION_FAILED` - Invalid or missing API key
- `WEBSOCKET_DISCONNECTED` - Cannot create subscription (WebSocket not connected)
- `INTERNAL_ERROR` - Unexpected server error

