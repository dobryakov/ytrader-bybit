# Quickstart: Position Management Service

**Date**: 2025-01-27  
**Feature**: Position Management Service  
**Branch**: `001-position-management`

## Overview

This guide provides quick setup and usage instructions for the Position Management Service. The service provides centralized portfolio position management with REST API access, portfolio metrics calculation, and position lifecycle management.

## Prerequisites

- Docker and Docker Compose V2 installed
- Access to shared PostgreSQL database (configured in main `docker-compose.yml`)
- Access to shared RabbitMQ (configured in main `docker-compose.yml`)
- API key for authentication (configured in `.env`)

## Quick Setup

### 1. Environment Configuration

Copy environment variables to `.env` (if not already present):

```bash
# Position Manager Service Configuration
POSITION_MANAGER_PORT=4800
POSITION_MANAGER_API_KEY=your-api-key-here
POSITION_MANAGER_LOG_LEVEL=INFO
POSITION_MANAGER_SERVICE_NAME=position-manager

# Database (shared PostgreSQL)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=your-password

# RabbitMQ (shared)
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Position Management
POSITION_MANAGER_SNAPSHOT_INTERVAL=3600  # 1 hour
POSITION_MANAGER_SNAPSHOT_RETENTION_DAYS=365
POSITION_MANAGER_VALIDATION_INTERVAL=1800  # 30 minutes
POSITION_MANAGER_METRICS_CACHE_TTL=10  # seconds

# Position Update Strategy
POSITION_MANAGER_USE_WS_AVG_PRICE=true
POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001  # 0.1%
POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD=0.0001
POSITION_MANAGER_PRICE_STALENESS_THRESHOLD=300  # 5 minutes
POSITION_MANAGER_PRICE_API_TIMEOUT=5  # seconds
POSITION_MANAGER_PRICE_API_RETRIES=3
POSITION_MANAGER_OPTIMISTIC_LOCK_RETRIES=3
POSITION_MANAGER_OPTIMISTIC_LOCK_BACKOFF_BASE=100  # milliseconds

# Rate Limiting
POSITION_MANAGER_RATE_LIMIT_ENABLED=true
POSITION_MANAGER_RATE_LIMIT_DEFAULT=100  # requests/minute
POSITION_MANAGER_RATE_LIMIT_OVERRIDES=model-service-key:100,risk-manager-key:200,ui-key:1000
```

### 2. Database Migration

**Important**: Before deploying Position Manager, run the migration to add `current_price` and `version` fields to the `positions` table.

The migration must be created in the `ws-gateway` service (per constitution - PostgreSQL migration ownership):

```bash
# Migration location: ws-gateway/migrations/
# Migration name: add_current_price_and_version_to_positions

# SQL:
ALTER TABLE positions 
ADD COLUMN IF NOT EXISTS current_price DECIMAL(20, 8) NULL,
ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_positions_current_price ON positions(current_price);
CREATE INDEX IF NOT EXISTS idx_positions_version ON positions(version);

UPDATE positions SET version = 1 WHERE version IS NULL;
```

Run migration:
```bash
# From ws-gateway service
docker compose exec ws-gateway python -m alembic upgrade head
```

### 3. Build and Start Service

```bash
# Build the service
docker compose build position-manager

# Start the service
docker compose up -d position-manager

# Check logs
docker compose logs -f position-manager
```

### 4. Verify Health

```bash
# Health check (no authentication required)
curl http://localhost:4800/health

# Expected response:
# {
#   "status": "healthy",
#   "service": "position-manager",
#   "database_connected": true,
#   "queue_connected": true,
#   "positions_count": 0,
#   "timestamp": "2025-01-27T10:00:00Z"
# }
```

## Basic Usage

### Get All Positions

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/positions
```

**Response**:
```json
{
  "positions": [
    {
      "id": "uuid",
      "asset": "BTCUSDT",
      "size": "1.5",
      "average_entry_price": "50000.00",
      "current_price": "50100.00",
      "unrealized_pnl": "150.00",
      "realized_pnl": "50.00",
      "mode": "one-way",
      "unrealized_pnl_pct": "0.30",
      "time_held_minutes": 120,
      "position_size_norm": "0.15",
      "last_updated": "2025-01-27T10:00:00Z"
    }
  ],
  "count": 1
}
```

### Get Position by Asset

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/positions/BTCUSDT?mode=one-way
```

### Get Portfolio Metrics

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/portfolio
```

**Response**:
```json
{
  "total_exposure_usdt": "10000.00",
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "portfolio_value_usdt": "10200.00",
  "open_positions_count": 3,
  "long_positions_count": 2,
  "short_positions_count": 1,
  "net_exposure_usdt": "5000.00",
  "by_asset": {
    "BTCUSDT": {
      "exposure_usdt": "7500.00",
      "unrealized_pnl_usdt": "100.00",
      "size": "1.5"
    }
  },
  "calculated_at": "2025-01-27T10:00:00Z"
}
```

### Get Portfolio Exposure Only

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/portfolio/exposure
```

### Get Portfolio PnL Only

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/portfolio/pnl
```

### Validate Position

```bash
curl -X POST -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/positions/BTCUSDT/validate?fix_discrepancies=true"
```

### Create Position Snapshot

```bash
curl -X POST -H "X-API-Key: your-api-key-here" \
  http://localhost:4800/api/v1/positions/BTCUSDT/snapshot
```

### Get Position Snapshots

```bash
curl -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/positions/BTCUSDT/snapshots?limit=10&offset=0"
```

## Filtering and Query Parameters

### Filter Positions by Asset

```bash
curl -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/positions?asset=BTCUSDT"
```

### Filter Positions by Size

```bash
# Minimum size
curl -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/positions?size_min=1.0"

# Maximum size
curl -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/positions?size_max=5.0"
```

### Include Positions in Portfolio Response

```bash
curl -H "X-API-Key: your-api-key-here" \
  "http://localhost:4800/api/v1/portfolio?include_positions=true"
```

## Event-Driven Integration

### Consuming Position Updates

Position Manager consumes events from:
- `ws-gateway.position` - WebSocket position events
- `order-manager.order_executed` - Order execution events

### Publishing Position Events

Position Manager publishes events to:
- `position-manager.position_updated` - Position update events
- `position-manager.portfolio_updated` - Portfolio metrics update events
- `position-manager.position_snapshot_created` - Snapshot creation events

Example: Subscribe to position updates in another service:

```python
import aio_pika

async def consume_position_updates():
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    channel = await connection.channel()
    queue = await channel.declare_queue("position-manager.position_updated")
    
    async for message in queue:
        event = json.loads(message.body)
        # Process position update event
        await message.ack()
```

## Testing

### Run Unit Tests

```bash
docker compose exec position-manager pytest tests/unit/ -v
```

### Run Integration Tests

```bash
docker compose exec test-container pytest tests/integration/ -v
```

### Run E2E Tests

```bash
docker compose exec test-container pytest tests/e2e/ -v
```

### Run All Tests

```bash
docker compose exec test-container pytest tests/ -v
```

## Monitoring

### Check Service Logs

```bash
# Follow logs
docker compose logs -f position-manager

# Last 100 lines
docker compose logs --tail 100 position-manager

# Filter by trace ID
docker compose logs position-manager | grep "trace_id=abc123"
```

### Health Check Monitoring

```bash
# Continuous health monitoring
watch -n 5 'curl -s http://localhost:4800/health | jq'
```

### Performance Metrics

Monitor:
- API response times (target: <500ms for 95% of requests)
- Portfolio metrics calculation time (target: <1s)
- Position update processing time (target: <2s for order execution, <5s for market data)
- Cache hit rate for portfolio metrics
- Rate limit exceedances

## Troubleshooting

### Service Won't Start

1. Check database connection:
   ```bash
   docker compose exec position-manager python -c "from src.config.database import check_connection; check_connection()"
   ```

2. Check RabbitMQ connection:
   ```bash
   docker compose exec position-manager python -c "from src.config.rabbitmq import check_connection; check_connection()"
   ```

3. Check logs:
   ```bash
   docker compose logs position-manager
   ```

### API Returns 401 Unauthorized

- Verify API key is correct in request header: `X-API-Key: your-api-key`
- Verify API key matches `POSITION_MANAGER_API_KEY` in `.env`

### API Returns 429 Too Many Requests

- Rate limit exceeded for your API key
- Check `POSITION_MANAGER_RATE_LIMIT_OVERRIDES` for your API key limit
- Wait for rate limit window to reset (check `Retry-After` header)

### Position Updates Not Reflecting

1. Check RabbitMQ consumers are running:
   ```bash
   docker compose logs position-manager | grep "consumer"
   ```

2. Check event queues:
   ```bash
   # Check ws-gateway.position queue
   docker compose exec rabbitmq rabbitmqctl list_queues | grep position
   ```

3. Check database for position updates:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT * FROM positions WHERE asset='BTCUSDT';"
   ```

### Portfolio Metrics Seem Stale

- Portfolio metrics are cached (TTL: 10 seconds by default)
- Cache is invalidated on position updates
- Check `POSITION_MANAGER_METRICS_CACHE_TTL` configuration
- Wait for cache to expire or trigger a position update to invalidate cache

## Next Steps

1. **Integration**: Integrate with Order Manager and Model Service (see spec.md for details)
2. **Monitoring**: Set up Grafana dashboards for position metrics
3. **Testing**: Add comprehensive test coverage
4. **Documentation**: Update main README with Position Manager details

## Additional Resources

- **API Documentation**: See `contracts/openapi.yaml` for full API specification
- **Data Model**: See `data-model.md` for entity definitions
- **Feature Specification**: See `spec.md` for complete requirements
- **Implementation Plan**: See `plan.md` for technical details

## Support

For issues or questions:
1. Check service logs: `docker compose logs position-manager`
2. Check health endpoint: `curl http://localhost:4800/health`
3. Review feature specification: `specs/001-position-management/spec.md`

