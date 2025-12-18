# WebSocket Gateway Service

A microservice that establishes and maintains a single authenticated WebSocket connection to Bybit exchange, subscribes to multiple data channels (trades, tickers, order books, order statuses, balances), and routes events to subscriber services via RabbitMQ queues.

## Features

- **Dual WebSocket Connections**: Separate public (`/v5/public`) and private (`/v5/private`) connections to Bybit for optimal scalability and separation of concerns
  - Public connection: No authentication required for public data channels (trades, tickers, orderbook, kline, liquidation)
  - Private connection: Authenticated connection for private data channels (wallet, order, position)
  - Independent reconnection: Each connection type maintains its own reconnection logic
- **WebSocket Connection**: Automatic reconnection and circuit breaker pattern for both connection types
- **Subscription Management**: Subscribe to multiple data channels (trades, tickers, order books, order statuses, balances) with automatic endpoint selection based on channel type
- **Event Routing**: Route events to RabbitMQ queues organized by event class with retention limits
- **REST API**: Dynamic subscription management via REST API with API key authentication
- **Data Persistence**: Persist critical data (balances, subscriptions) to PostgreSQL with graceful error handling
- **Structured Logging**: Comprehensive logging with trace IDs for request flow tracking and observability
- **Edge Case Handling**: Circuit breaker, malformed message handling, timeout management, queue capacity monitoring
- **Monitoring**: Queue backlog monitoring, retention monitoring, and consumption rate tracking

## Prerequisites

- Docker and Docker Compose V2
- Python 3.11+ (for local development)
- PostgreSQL 15+
- RabbitMQ 3+

## Quick Start

### Using Docker Compose

1. Copy environment variables:
   ```bash
   cp env.example .env
   ```

2. Update `.env` with your Bybit API credentials and database passwords

3. Start services:
   ```bash
   docker compose up -d
   ```

4. Check service health:
   ```bash
   curl http://localhost:4400/health
   ```

### Local Development

1. Create virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables (see `env.example`)

4. Run migrations:
   ```bash
   # Apply migrations to shared PostgreSQL using ws-gateway container and psql
   # Example for the first migrations:
   docker compose exec ws-gateway sh -c \
     'cd /app/migrations && PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -v ON_ERROR_STOP=1 -f 001_create_subscriptions_table.sql'

   docker compose exec ws-gateway sh -c \
     'cd /app/migrations && PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -v ON_ERROR_STOP=1 -f 002_create_account_balances_table.sql'

   # Аналогично можно применять любые последующие миграции, например:
   docker compose exec ws-gateway sh -c \
     'cd /app/migrations && PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -v ON_ERROR_STOP=1 -f 026_create_bybit_fee_rates_table.sql'
   ```

5. Run the service:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 4400 --reload
   ```

## Configuration

All configuration is done via environment variables. See `env.example` for available options.

### Key Configuration Variables

- `BYBIT_API_KEY`: Bybit API key for WebSocket authentication
- `BYBIT_API_SECRET`: Bybit API secret
- `BYBIT_ENVIRONMENT`: `testnet` or `mainnet`
- `POSTGRES_HOST`: PostgreSQL hostname
- `RABBITMQ_HOST`: RabbitMQ hostname
- `WS_GATEWAY_PORT`: REST API port (default: 4400)
- `WS_GATEWAY_API_KEY`: API key for REST API authentication

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status including WebSocket connection status.

### Subscription Management

All subscription endpoints require API key authentication via `X-API-Key` header.

- `POST /api/v1/subscriptions` - Create a new subscription (automatically uses correct endpoint)
- `GET /api/v1/subscriptions` - List subscriptions (with optional filters)
- `GET /api/v1/subscriptions/{subscription_id}` - Get subscription details
- `DELETE /api/v1/subscriptions/{subscription_id}` - Cancel a subscription
- `DELETE /api/v1/subscriptions/by-service/{service_name}` - Cancel all subscriptions for a service

### Balance & Margin API

The balance API exposes latest and historical balances stored in PostgreSQL so that
local tools and services can inspect account state without direct database access.

- `GET /api/v1/balances` - Latest balance per coin + latest account-level margin view
- `GET /api/v1/balances/history` - Historical balance records with time range filters
- `POST /api/v1/balances/sync` - Reserved for future manual sync from Bybit REST API (currently returns 501)

#### Subscription Examples

**Subscribe to Public Channel (Ticker)**:
```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "ticker",
    "symbol": "BTCUSDT",
    "requesting_service": "model-service"
  }'
```
This automatically uses the public endpoint (no API credentials needed for Bybit).

**Subscribe to Private Channel (Balance)**:
```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "balance",
    "requesting_service": "order-manager"
  }'
```
This automatically uses the private endpoint (requires Bybit API credentials).

### Query Latest Balances and Margin Summary

```bash
curl -X GET "http://localhost:4400/api/v1/balances?limit=100" \
  -H "X-API-Key: your-api-key"
```

Example response:

```json
{
  "balances": [
    {
      "coin": "USDT",
      "wallet_balance": "10000.0",
      "available_balance": "9500.0",
      "frozen": "500.0",
      "event_timestamp": "2025-11-25T10:00:00Z",
      "received_at": "2025-11-25T10:00:01Z"
    }
  ],
  "margin_balance": {
    "account_type": "UNIFIED",
    "total_equity": "10000.0",
    "total_wallet_balance": "10000.0",
    "total_margin_balance": "8000.0",
    "total_available_balance": "2000.0",
    "total_initial_margin": "5000.0",
    "total_maintenance_margin": "1000.0",
    "total_order_im": "500.0",
    "base_currency": "USDT",
    "event_timestamp": "2025-11-25T10:00:00Z",
    "received_at": "2025-11-25T10:00:01Z"
  },
  "total": 1
}
```

### Query Balance History for Analytics/Debugging

```bash
curl -X GET "http://localhost:4400/api/v1/balances/history?coin=USDT&from=2025-11-25T00:00:00Z&to=2025-11-26T00:00:00Z&limit=100" \
  -H "X-API-Key: your-api-key"
```

This returns individual balance records from `account_balances` for the given time
range and coin, suitable for local analytics or debugging tools.

See `specs/001-websocket-gateway/contracts/openapi.yaml` for detailed API documentation.

## Development

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Running Tests

```bash
# Unit tests (run in service container)
docker compose exec ws-gateway pytest tests/unit/

# Integration tests (run in test container)
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Architecture

### Dual Connection Architecture

The service supports dual WebSocket connections to Bybit:

1. **Public Connection** (`/v5/public`):
   - No authentication required
   - Used for public data channels: `trades`, `ticker`, `orderbook`, `kline`, `liquidation`
   - Automatically selected when subscribing to public channels

2. **Private Connection** (`/v5/private`):
   - Requires API key authentication
   - Used for private data channels: `balance` (wallet), `order`, `position`
   - Automatically selected when subscribing to private channels

**Benefits**:
- Better scalability: Public data doesn't require API credentials
- Separation of concerns: Public and private data handled independently
- Independent reconnection: Each connection type reconnects independently
- Automatic selection: The system automatically chooses the correct endpoint based on channel type

### Technical Stack

- **Language**: Python 3.11+
- **WebSocket Client**: `websockets` library with automatic reconnection and circuit breaker
- **Connection Management**: `ConnectionManager` class manages dual connections with lazy initialization
- **REST API**: FastAPI with API key authentication middleware
- **Message Queue**: RabbitMQ (aio-pika) with queue retention and backlog monitoring
- **Database**: PostgreSQL (asyncpg) with connection pooling
- **Logging**: structlog with trace ID propagation
- **Error Handling**: Comprehensive error handling with context-rich logging
- **Monitoring**: Queue capacity alerts, backlog monitoring, and consumption rate tracking

## Monitoring & Observability

### Health Check

The service provides a health check endpoint that reports:
- WebSocket connection status
- Database connection status
- RabbitMQ connection status
- Active subscription count

```bash
curl http://localhost:4400/health
```

### Logging

All logs include:
- **Trace IDs**: Unique identifiers for request flow tracking across async operations
- **Structured format**: JSON logs in DEBUG mode, readable console format otherwise
- **Context information**: Connection IDs, event types, queue names, etc.

### Queue Monitoring

The service monitors:
- **Queue capacity**: Alerts when queues approach 80% or 95% of retention limits
- **Queue backlog**: Warns when backlog exceeds thresholds (1000 messages warning, 10K critical)
- **Consumption rates**: Tracks publish vs consumption rates to detect slow subscribers

### Edge Cases Handled

- **Circuit breaker**: Prevents excessive reconnection attempts when exchange API is unavailable
- **Malformed messages**: Validates and gracefully handles invalid message formats
- **Timeout handling**: Connection timeouts prevent indefinite hangs
- **Authentication failures**: Credential validation and error recovery
- **Conflicting subscriptions**: Multiple services can safely subscribe to same topic
- **Queue capacity limits**: Monitoring and alerting when storage reaches capacity
- **Slow subscribers**: Backlog monitoring and consumption rate alerts

## Project Structure

```
ws-gateway/
├── src/
│   ├── models/          # Data models (Subscription, Event, AccountBalance, WebSocketState)
│   ├── services/        # Business logic
│   │   ├── websocket/   # WebSocket connection, auth, heartbeat, reconnection
│   │   ├── queue/       # RabbitMQ operations (publisher, retention, monitoring)
│   │   ├── database/    # PostgreSQL operations (connection, repositories)
│   │   └── subscription/ # Subscription management service
│   ├── api/             # FastAPI REST endpoints (v1, health, middleware)
│   ├── config/          # Configuration and logging setup
│   ├── utils/           # Utility functions (tracing)
│   ├── exceptions.py    # Custom exception classes
│   └── main.py          # Application entry point
├── tests/               # Test suites (unit, integration, e2e)
├── migrations/          # Database migrations (PostgreSQL)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Security

- **API Key Authentication**: All REST API endpoints require `X-API-Key` header
- **Credential Validation**: API credentials are validated before connection attempts
- **Input Sanitization**: All user inputs are validated before processing
- **Error Handling**: Sensitive information is not exposed in error messages

## Troubleshooting

### WebSocket Not Connecting

1. Check Bybit API credentials in `.env`
2. Verify environment (testnet vs mainnet)
3. Check logs for authentication errors
4. Verify network connectivity to Bybit endpoints

### Events Not Appearing in Queues

1. Verify subscription is active via REST API
2. Check RabbitMQ connection status
3. Review queue capacity alerts in logs
4. Monitor queue backlog for slow consumers

### High Latency

1. Check service health endpoint
2. Monitor queue backlog and consumption rates
3. Review logs for errors or warnings
4. Check database connection pool status

## License

See repository root for license information.

