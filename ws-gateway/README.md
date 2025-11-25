# WebSocket Gateway Service

A microservice that establishes and maintains a single authenticated WebSocket connection to Bybit exchange, subscribes to multiple data channels (trades, tickers, order books, order statuses, balances), and routes events to subscriber services via RabbitMQ queues.

## Features

- **WebSocket Connection**: Single authenticated connection to Bybit (mainnet or testnet) with automatic reconnection
- **Subscription Management**: Subscribe to multiple data channels (trades, tickers, order books, order statuses, balances)
- **Event Routing**: Route events to RabbitMQ queues organized by event class
- **REST API**: Dynamic subscription management via REST API with API key authentication
- **Data Persistence**: Persist critical data (balances, subscriptions) to PostgreSQL
- **Structured Logging**: Comprehensive logging with trace IDs for observability

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
   # Apply migrations to database
   psql -h localhost -U ytrader -d ytrader -f migrations/001_create_subscriptions_table.sql
   psql -h localhost -U ytrader -d ytrader -f migrations/002_create_account_balances_table.sql
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

- `POST /api/v1/subscriptions` - Create a new subscription
- `GET /api/v1/subscriptions` - List subscriptions (with optional filters)
- `GET /api/v1/subscriptions/{subscription_id}` - Get subscription details
- `DELETE /api/v1/subscriptions/{subscription_id}` - Cancel a subscription
- `DELETE /api/v1/subscriptions/by-service/{service_name}` - Cancel all subscriptions for a service

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

- **Language**: Python 3.11+
- **WebSocket Client**: `websockets` library
- **REST API**: FastAPI
- **Message Queue**: RabbitMQ (aio-pika)
- **Database**: PostgreSQL (asyncpg)
- **Logging**: structlog

## Project Structure

```
ws-gateway/
├── src/
│   ├── models/          # Data models
│   ├── services/        # Business logic
│   │   ├── websocket/   # WebSocket connection management
│   │   ├── queue/       # RabbitMQ operations
│   │   ├── database/     # PostgreSQL operations
│   │   └── subscription/ # Subscription management
│   ├── api/             # FastAPI REST endpoints
│   ├── config/          # Configuration management
│   └── main.py          # Application entry point
├── tests/               # Test suites
├── migrations/          # Database migrations
├── Dockerfile
├── requirements.txt
└── README.md
```

## License

See repository root for license information.

