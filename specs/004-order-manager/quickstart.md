# Quickstart: Order Manager Microservice

**Feature**: Order Manager Microservice  
**Date**: 2025-01-27

## Overview

This guide provides step-by-step instructions to set up and run the Order Manager microservice locally for development and testing.

## Prerequisites

- Docker and Docker Compose V2 installed
- Bybit API credentials (API key and secret) - use testnet for development
- Shared PostgreSQL database available (managed by ws-gateway service)
- RabbitMQ message broker available
- WebSocket Gateway service operational (for order execution event subscriptions)

## Quick Setup

### 1. Clone and Navigate

```bash
cd /home/ubuntu/ytrader
git checkout 004-order-manager
```

### 2. Configure Environment

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` and set the following variables (add to existing configuration):

```bash
# Order Manager Service Configuration
ORDER_MANAGER_PORT=4600
ORDER_MANAGER_API_KEY=your-order-manager-api-key  # For REST API authentication
ORDER_MANAGER_LOG_LEVEL=INFO
ORDER_MANAGER_SERVICE_NAME=order-manager

# Bybit API Configuration (shared with other services)
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_api_secret
BYBIT_ENVIRONMENT=testnet  # or 'mainnet' for production

# Database Configuration (shared PostgreSQL)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=your_password

# RabbitMQ Configuration (shared message broker)
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# WebSocket Gateway Configuration (for event subscriptions)
WS_GATEWAY_HOST=ws-gateway
WS_GATEWAY_PORT=4400
WS_GATEWAY_API_KEY=your-gateway-api-key

# Order Execution Configuration
ORDERMANAGER_ENABLE_DRY_RUN=false  # Set to 'true' for testing without real orders
ORDERMANAGER_MAX_SINGLE_ORDER_SIZE=10000.0  # Maximum order size in USDT
ORDERMANAGER_ENABLE_ORDER_SPLITTING=false  # Enable order splitting for large amounts
ORDERMANAGER_ORDER_EXECUTION_TIMEOUT=30  # Timeout in seconds

# Risk Limits Configuration
ORDERMANAGER_MAX_POSITION_SIZE=1.0  # Maximum position size per asset (base currency)
ORDERMANAGER_MAX_EXPOSURE=50000.0  # Maximum total exposure across all positions (USDT)
ORDERMANAGER_MAX_ORDER_SIZE_RATIO=0.1  # Maximum order size as ratio of available balance

# Retry Configuration
ORDERMANAGER_BYBIT_API_RETRY_MAX_ATTEMPTS=3
ORDERMANAGER_BYBIT_API_RETRY_BASE_DELAY=1.0  # seconds
ORDERMANAGER_BYBIT_API_RETRY_MAX_DELAY=30.0  # seconds
ORDERMANAGER_BYBIT_API_RETRY_MULTIPLIER=2.0

# Order Type Selection Configuration
ORDERMANAGER_MARKET_ORDER_CONFIDENCE_THRESHOLD=0.9
ORDERMANAGER_MARKET_ORDER_SPREAD_THRESHOLD=0.1  # percentage
ORDERMANAGER_LIMIT_ORDER_PRICE_OFFSET_RATIO=0.5

# Position Management Configuration
ORDERMANAGER_POSITION_SNAPSHOT_INTERVAL=300  # seconds (5 minutes)
ORDERMANAGER_POSITION_VALIDATION_INTERVAL=3600  # seconds (1 hour)
```

### 3. Start Dependencies

Ensure all dependencies are running:

```bash
docker compose up -d postgres rabbitmq ws-gateway
```

Wait for services to be ready:

```bash
docker compose logs -f postgres rabbitmq ws-gateway
# Press Ctrl+C when services are ready
```

### 4. Run Database Migrations

**IMPORTANT**: Per constitution principle II (Shared Database Strategy), PostgreSQL migrations MUST be located in the `ws-gateway` service. Migrations for Order Manager tables (orders, positions, signal_order_relationships, position_snapshots) should be added to `ws-gateway/migrations/` directory.

Create migration files in `ws-gateway/migrations/`:
- `XXX_create_orders_table.sql`
- `XXX_create_signal_order_relationships_table.sql`
- `XXX_create_positions_table.sql`
- `XXX_create_position_snapshots_table.sql`

Run migrations:

```bash
docker compose exec ws-gateway python -m migrations.run
```

Or if migrations are run automatically on startup, they will execute when ws-gateway starts.

### 5. Start Order Manager Service

```bash
docker compose up -d order-manager
```

### 6. Verify Service Health

Check service health:

```bash
curl http://localhost:4600/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:00:00Z"
}
```

Check service readiness (includes dependency status):

```bash
curl http://localhost:4600/ready
```

Expected response:

```json
{
  "status": "ready",
  "timestamp": "2025-01-27T10:00:00Z",
  "dependencies": {
    "database": "available",
    "rabbitmq": "available",
    "bybit_api": "available",
    "ws_gateway": "available"
  }
}
```

### 7. Subscribe to Order Execution Events

Order Manager needs to subscribe to order execution events from WebSocket Gateway:

```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "X-API-Key: your-gateway-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "order",
    "requesting_service": "order-manager"
  }'
```

Alternatively, the Order Manager service can automatically subscribe to required channels on startup.

### 8. Test Order Manager API

List orders (should be empty initially):

```bash
curl -X GET "http://localhost:4600/api/v1/orders" \
  -H "X-API-Key: your-order-manager-api-key"
```

List positions (should be empty initially):

```bash
curl -X GET "http://localhost:4600/api/v1/positions" \
  -H "X-API-Key: your-order-manager-api-key"
```

## Testing with Trading Signals

### 1. Enable Dry-Run Mode (Recommended for Testing)

Set in `.env`:

```bash
ORDERMANAGER_ENABLE_DRY_RUN=true
```

Restart service:

```bash
docker compose restart order-manager
```

### 2. Send Test Trading Signal

Send a test trading signal to RabbitMQ queue `model-service.trading_signals`:

```bash
# Using RabbitMQ management UI or CLI tool
docker compose exec rabbitmq rabbitmqadmin publish routing_key=model-service.trading_signals payload='
{
  "signal_id": "test-signal-001",
  "signal_type": "buy",
  "asset": "BTCUSDT",
  "amount": 1000.0,
  "confidence": 0.85,
  "timestamp": "2025-01-27T10:00:00Z",
  "strategy_id": "test-strategy",
  "model_version": null,
  "is_warmup": true,
  "market_data_snapshot": {
    "price": 50000.0,
    "spread": 1.5,
    "volume_24h": 1000000.0,
    "volatility": 0.02,
    "orderbook_depth": {
      "bid_depth": 100.0,
      "ask_depth": 120.0
    },
    "technical_indicators": null
  },
  "metadata": {
    "reasoning": "Test signal",
    "risk_score": 0.3
  },
  "trace_id": "test-trace-001"
}'
```

### 3. Verify Order Creation

Check Order Manager logs:

```bash
docker compose logs -f order-manager
```

You should see logs indicating:
- Signal received from queue
- Signal validated
- Order type selected
- Quantity calculated
- Order created (or simulated in dry-run mode)

Query orders:

```bash
curl -X GET "http://localhost:4600/api/v1/orders?asset=BTCUSDT" \
  -H "X-API-Key: your-order-manager-api-key"
```

In dry-run mode, orders will have `status: "dry_run"` and `is_dry_run: true`.

## Manual Order State Synchronization

Trigger manual synchronization with Bybit to refresh order states:

```bash
curl -X POST "http://localhost:4600/api/v1/sync" \
  -H "X-API-Key: your-order-manager-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "active"
  }'
```

Response:

```json
{
  "status": "completed",
  "synced_orders": 5,
  "updated_orders": 2,
  "errors": [],
  "timestamp": "2025-01-27T10:00:00Z"
}
```

## Monitoring and Logs

View service logs:

```bash
docker compose logs -f order-manager
```

Filter logs by trace ID:

```bash
docker compose logs order-manager | grep "trace-12345"
```

Monitor RabbitMQ queues:

```bash
# Check queue message count
docker compose exec rabbitmq rabbitmqctl list_queues name messages

# Purge test messages
docker compose exec rabbitmq rabbitmqadmin purge queue name=model-service.trading_signals
```

## Troubleshooting

### Service Won't Start

Check logs:

```bash
docker compose logs order-manager
```

Common issues:
- Database connection failed: Check PostgreSQL is running and credentials are correct
- RabbitMQ connection failed: Check RabbitMQ is running and credentials are correct
- Bybit API credentials invalid: Verify API key and secret in `.env`

### Orders Not Being Created

Check:
1. Trading signals are being published to `model-service.trading_signals` queue
2. Order Manager is consuming from the queue (check logs)
3. Dry-run mode is disabled if you want real orders
4. Bybit API credentials are valid
5. Sufficient balance for orders

### Order State Not Updating

Check:
1. WebSocket Gateway is operational and subscribed to order events
2. Order Manager has subscribed to order execution events
3. WebSocket connection is active (check WebSocket Gateway logs)
4. Manual sync works (try `/api/v1/sync` endpoint)

### Database Connection Issues

Verify PostgreSQL is accessible:

```bash
docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT 1;"
```

Check migration status:

```bash
docker compose exec postgres psql -U ytrader -d ytrader -c "\dt"
```

## Production Deployment

Before deploying to production:

1. **Set Environment to Mainnet**:
   ```bash
   BYBIT_ENVIRONMENT=mainnet
   ```

2. **Use Strong API Keys**:
   - Generate secure API keys for Order Manager REST API
   - Use secure Bybit API credentials

3. **Disable Dry-Run Mode**:
   ```bash
   ORDERMANAGER_ENABLE_DRY_RUN=false
   ```

4. **Configure Risk Limits**:
   - Set appropriate `ORDERMANAGER_MAX_POSITION_SIZE` and `ORDERMANAGER_MAX_EXPOSURE` limits
   - Adjust `ORDERMANAGER_MAX_ORDER_SIZE_RATIO` based on account size

5. **Set Up Monitoring**:
   - Monitor service health endpoints
   - Set up alerts for order execution failures
   - Monitor queue depths and processing rates

6. **Enable Logging**:
   - Set appropriate log level (INFO for production)
   - Ensure logs are collected and stored

7. **Test Thoroughly**:
   - Test in testnet environment first
   - Verify order execution flow end-to-end
   - Test error handling and recovery scenarios

## Next Steps

- Review [data-model.md](./data-model.md) for database schema details
- Review [research.md](./research.md) for design decisions
- Review [contracts/openapi.yaml](./contracts/openapi.yaml) for API documentation
- Implement the service following the architecture defined in [plan.md](./plan.md)

