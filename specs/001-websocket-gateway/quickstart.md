# Quickstart: WebSocket Gateway

**Feature**: WebSocket Gateway for Bybit Data Aggregation and Routing  
**Date**: 2025-11-25

## Overview

This guide provides step-by-step instructions to set up and run the WebSocket Gateway service locally for development and testing.

## Prerequisites

- Docker and Docker Compose V2 installed
- Bybit API credentials (API key and secret) - use testnet for development
- Access to PostgreSQL database (or use Docker container)
- Access to RabbitMQ (or use Docker container)

## Quick Setup

### 1. Clone and Navigate

```bash
cd /home/ubuntu/ytrader
git checkout 001-websocket-gateway
```

### 2. Configure Environment

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` and set the following variables:

```bash
# Bybit API Configuration
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_api_secret
BYBIT_ENVIRONMENT=testnet  # or 'mainnet' for production

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=your_password

# RabbitMQ Configuration
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# WebSocket Gateway Configuration
WS_GATEWAY_PORT=4400
WS_GATEWAY_API_KEY=your-gateway-api-key  # For REST API authentication
WS_GATEWAY_LOG_LEVEL=INFO

# Service Configuration
WS_GATEWAY_SERVICE_NAME=ws-gateway
```

### 3. Start Dependencies

Start PostgreSQL and RabbitMQ using Docker Compose:

```bash
docker compose up -d postgres rabbitmq
```

Wait for services to be ready:

```bash
docker compose logs -f postgres rabbitmq
# Press Ctrl+C when services are ready
```

### 4. Run Database Migrations

**Note**: Per constitution principle II (Shared Database Strategy), the `ws-gateway` service is the single source of truth for all PostgreSQL migrations. All PostgreSQL schema changes (including those for other services) are located in `ws-gateway/migrations/`.

```bash
docker compose run --rm ws-gateway python -m migrations.run
```

Or if migrations are run automatically on startup, skip this step.

### 5. Start WebSocket Gateway

```bash
docker compose up -d ws-gateway
```

### 6. Verify Service Health

```bash
curl http://localhost:4400/health
```

Expected response:

```json
{
  "status": "healthy",
  "websocket_connected": true,
  "active_subscriptions": 0,
  "last_heartbeat_at": "2025-11-25T10:00:00Z",
  "version": "1.0.0"
}
```

## Basic Usage

### Subscribe to Trades

Subscribe to BTCUSDT trades:

```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "X-API-Key: your-gateway-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "trades",
    "symbol": "BTCUSDT",
    "requesting_service": "test-service"
  }'
```

Response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "channel_type": "trades",
  "symbol": "BTCUSDT",
  "topic": "trade.BTCUSDT",
  "requesting_service": "test-service",
  "is_active": true,
  "created_at": "2025-11-25T10:00:00Z",
  "updated_at": "2025-11-25T10:00:00Z",
  "last_event_at": null
}
```

### Subscribe to Balance Updates

```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "X-API-Key: your-gateway-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "balance",
    "requesting_service": "test-service"
  }'
```

### List Subscriptions

```bash
curl -X GET "http://localhost:4400/api/v1/subscriptions?is_active=true" \
  -H "X-API-Key: your-gateway-api-key"
```

### Query Latest Balances and Margin Summary

Retrieve latest balance per coin and account-level margin view:

```bash
curl -X GET "http://localhost:4400/api/v1/balances?limit=100" \
  -H "X-API-Key: your-gateway-api-key"
```

Example use cases:

- Local CLI inspecting current wallet and margin state
- Order Manager / Position Manager validating margin before placing orders

### Query Balance History

Retrieve historical balance records for analytics/debugging:

```bash
curl -X GET "http://localhost:4400/api/v1/balances/history?coin=USDT&from=2025-11-25T00:00:00Z&to=2025-11-26T00:00:00Z&limit=100" \
  -H "X-API-Key: your-gateway-api-key"
```

This returns rows from `account_balances` filtered by coin and time range, which can
be consumed by local tools without direct database access.

### Consume Events from RabbitMQ

Events are delivered to RabbitMQ queues. To consume events:

```bash
# Install RabbitMQ client tools (if not already installed)
docker compose exec rabbitmq rabbitmqadmin list queues

# Consume events from trades queue (example using Python)
python -c "
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672))
channel = connection.channel()
channel.queue_declare(queue='ws-gateway.trades')

def callback(ch, method, properties, body):
    print(f'Received: {body.decode()}')

channel.basic_consume(queue='ws-gateway.trades', on_message_callback=callback, auto_ack=True)
print('Waiting for messages...')
channel.start_consuming()
"
```

### Cancel a Subscription

```bash
curl -X DELETE http://localhost:4400/api/v1/subscriptions/{subscription_id} \
  -H "X-API-Key: your-gateway-api-key"
```

## Development Workflow

### Running Tests

Unit tests (run in service container):

```bash
docker compose run --rm ws-gateway pytest tests/unit -v
```

Integration tests (run in test container):

```bash
docker compose run --rm test-ws-gateway pytest tests/integration -v
```

End-to-end tests:

```bash
docker compose run --rm test-ws-gateway pytest tests/e2e -v
```

### Viewing Logs

```bash
# All logs
docker compose logs -f ws-gateway

# Last 100 lines
docker compose logs --tail 100 ws-gateway

# Filter by trace ID
docker compose logs ws-gateway | grep "trace_id=abc123"
```

### Database Access

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U ytrader -d ytrader

# View subscriptions
SELECT * FROM subscriptions WHERE is_active = true;

# View latest balances
SELECT * FROM account_balances ORDER BY received_at DESC LIMIT 10;
```

### RabbitMQ Management

Access RabbitMQ management UI:

```bash
# Open in browser
open http://localhost:15672
# Default credentials: guest/guest
```

Or use CLI:

```bash
docker compose exec rabbitmq rabbitmqadmin list queues
docker compose exec rabbitmq rabbitmqadmin list exchanges
```

## Troubleshooting

### WebSocket Not Connecting

1. Check Bybit API credentials:
   ```bash
   docker compose logs ws-gateway | grep -i "auth\|connection"
   ```

2. Verify environment (testnet vs mainnet):
   ```bash
   echo $BYBIT_ENVIRONMENT
   ```

3. Check network connectivity:
   ```bash
   docker compose exec ws-gateway ping -c 3 stream-testnet.bybit.com
   ```

### Events Not Appearing in Queues

1. Verify subscription is active:
   ```bash
   curl -X GET "http://localhost:4400/api/v1/subscriptions?is_active=true" \
     -H "X-API-Key: your-gateway-api-key"
   ```

2. Check RabbitMQ connection:
   ```bash
   docker compose logs ws-gateway | grep -i "rabbitmq\|queue"
   ```

3. Verify queue exists:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name messages
   ```

### Database Connection Issues

1. Check PostgreSQL is running:
   ```bash
   docker compose ps postgres
   ```

2. Verify connection string:
   ```bash
   docker compose logs ws-gateway | grep -i "postgres\|database"
   ```

3. Test connection:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT 1;"
   ```

### High Latency or Missing Events

1. Check service health:
   ```bash
   curl http://localhost:4400/health
   ```

2. Monitor logs for errors:
   ```bash
   docker compose logs --tail 100 -f ws-gateway
   ```

3. Check queue backlog:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name messages consumers
   ```

## Production Deployment

### Environment Configuration

1. Use strong API keys and secrets
2. Set `BYBIT_ENVIRONMENT=mainnet` for production
3. Configure proper logging levels (`WS_GATEWAY_LOG_LEVEL=WARNING`)
4. Use secure database credentials
5. Enable SSL/TLS for RabbitMQ and PostgreSQL connections

### Monitoring

- Monitor WebSocket connection uptime (target: 99.5%)
- Track event processing latency (target: <100ms)
- Monitor queue depths and consumer lag
- Set up alerts for connection failures and high error rates

### Scaling

- The service maintains a single WebSocket connection to Bybit
- Multiple instances can run (each with its own connection) if needed
- RabbitMQ queues support multiple consumers for fan-out delivery
- Database connection pooling handles concurrent operations

## Next Steps

- Read the [full specification](./spec.md) for detailed requirements
- Review the [data model](./data-model.md) for database schema
- Check [API contracts](./contracts/) for complete API documentation
- See [implementation plan](./plan.md) for architecture details

## Support

For issues or questions:
1. Check service logs: `docker compose logs ws-gateway`
2. Review health endpoint: `curl http://localhost:4400/health`
3. Consult the specification and documentation in this directory

