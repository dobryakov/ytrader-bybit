# Order Manager Microservice

A microservice that receives high-level trading signals from the model service via RabbitMQ, executes them as orders on Bybit exchange, maintains accurate order state through WebSocket event subscriptions, and publishes enriched order execution events.

## Features

- **Signal Processing**: Receive and process trading signals from model service via RabbitMQ
- **Order Execution**: Execute orders on Bybit exchange with configurable order type selection (market vs limit)
- **Order State Management**: Maintain accurate order state through WebSocket event subscriptions
- **Position Queries**: Query positions via REST API (delegates to Position Manager service)
- **Risk Management**: Enforce risk limits (max exposure, max order size, position size limits) using Position Manager service as the source of truth for position-based risk checks
- **Event Publishing**: Publish enriched order execution events to RabbitMQ for other microservices
- **Dry-Run Mode**: Test order logic without executing real orders on exchange
- **Observability**: Structured logging with trace IDs for request flow tracking

## Technology Stack

- **Language**: Python 3.11+
- **REST API Framework**: FastAPI>=0.104.0
- **HTTP Client**: httpx>=0.25.0 (for Bybit REST API)
- **Message Queue**: aio-pika>=9.0.0 (RabbitMQ)
- **Database**: asyncpg>=0.29.0 (PostgreSQL)
- **Configuration**: pydantic-settings>=2.0.0
- **Logging**: structlog>=23.2.0

## Project Structure

```
order-manager/
├── src/
│   ├── models/              # Data models (Order, Position, Trading Signal, etc.)
│   ├── services/            # Core business logic (signal processing, order execution, risk management)
│   ├── api/                 # REST API endpoints
│   │   ├── routes/          # API route handlers
│   │   └── middleware/      # API middleware (authentication, logging)
│   ├── consumers/           # RabbitMQ message consumers (trading signals)
│   ├── publishers/          # RabbitMQ message publishers (order events)
│   ├── config/              # Configuration management
│   ├── utils/               # Utility functions (tracing, Bybit client)
│   └── main.py              # Application entry point
├── tests/
│   ├── unit/                # Unit tests for services, utilities
│   ├── integration/         # Integration tests for database, message queue
│   └── contract/            # Contract tests for API endpoints
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Docker and Docker Compose V2
- Python 3.11+ (for local development)
- PostgreSQL 15+ (via Docker)
- RabbitMQ 3+ (via Docker)
- **Position Manager service** (required for position queries and risk checks - Order Manager delegates position data to Position Manager as the single source of truth)

### Environment Variables

Copy `env.example` to `.env` and configure the following variables:

```bash
# Order Manager Service Configuration
ORDERMANAGER_PORT=4600
ORDERMANAGER_API_KEY=your-api-key-here
ORDERMANAGER_LOG_LEVEL=INFO
ORDERMANAGER_SERVICE_NAME=order-manager

# WebSocket Gateway Configuration (shared with other services)
WS_GATEWAY_HOST=ws-gateway
WS_GATEWAY_PORT=4400
WS_GATEWAY_API_KEY=your-ws-gateway-api-key

# Position Manager Configuration (required for position queries and risk checks)
POSITION_MANAGER_HOST=position-manager
POSITION_MANAGER_PORT=4800
POSITION_MANAGER_API_KEY=your-position-manager-api-key

# Order Execution Configuration
ORDERMANAGER_ENABLE_DRY_RUN=false
ORDERMANAGER_MAX_SINGLE_ORDER_SIZE=10000.0
ORDERMANAGER_ENABLE_ORDER_SPLITTING=false
ORDERMANAGER_ORDER_EXECUTION_TIMEOUT=30

# Risk Limits Configuration
ORDERMANAGER_MAX_POSITION_SIZE=1.0
ORDERMANAGER_MAX_EXPOSURE=50000.0
ORDERMANAGER_MAX_ORDER_SIZE_RATIO=0.1

# Bybit API Retry Configuration
ORDERMANAGER_BYBIT_API_RETRY_MAX_ATTEMPTS=3
ORDERMANAGER_BYBIT_API_RETRY_BASE_DELAY=1.0
ORDERMANAGER_BYBIT_API_RETRY_MAX_DELAY=30.0
ORDERMANAGER_BYBIT_API_RETRY_MULTIPLIER=2.0

# Order Type Selection Configuration
ORDERMANAGER_MARKET_ORDER_CONFIDENCE_THRESHOLD=0.9
ORDERMANAGER_MARKET_ORDER_SPREAD_THRESHOLD=0.1
ORDERMANAGER_LIMIT_ORDER_PRICE_OFFSET_RATIO=0.5

# Position Management Configuration
# Position snapshots and validation are now handled by position-manager service

# Order Cancellation Configuration
ORDERMANAGER_CANCEL_OPPOSITE_ORDERS_ONLY=false
ORDERMANAGER_CANCEL_STALE_ORDER_TIMEOUT=3600

# Risk Management Configuration
ORDERMANAGER_UNREALIZED_LOSS_WARNING_THRESHOLD=10.0
```

See `env.example` for complete configuration options.

### Docker Setup

1. Ensure dependencies are running:

```bash
# Start PostgreSQL, RabbitMQ, WebSocket Gateway, and Position Manager
docker compose up -d postgres rabbitmq ws-gateway position-manager

# Wait for services to be ready
docker compose logs -f postgres rabbitmq ws-gateway position-manager
# Press Ctrl+C when services are ready
```

**Note**: Order Manager requires Position Manager service to be running. Position Manager is the single source of truth for position data. Order Manager's position endpoints and risk checks delegate to Position Manager REST API.

2. Run database migrations:

Database migrations for Order Manager tables are managed in the `ws-gateway` service per constitution requirement. Migrations should already be applied when ws-gateway starts, but you can verify:

```bash
docker compose exec postgres psql -U ytrader -d ytrader -c "\dt"
```

You should see tables: `orders`, `signal_order_relationships`, `positions`, `position_snapshots`.

3. Build and start the service:

```bash
docker compose build order-manager
docker compose up -d order-manager
```

4. Check service health:

```bash
# Basic health check
curl http://localhost:4600/health

# Check readiness (includes dependency status)
curl http://localhost:4600/ready
```

5. View logs:

```bash
# Follow logs
docker compose logs -f order-manager

# View last 100 lines
docker compose logs --tail 100 order-manager
```

### Local Development Setup

1. Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install development tools:

```bash
pip install black flake8 mypy
```

4. Run linting:

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

5. Run tests:

```bash
pytest
```

6. Run the service locally:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 4600 --reload
```

## API Endpoints

All API endpoints require authentication via `X-API-Key` header. Set the API key in your `.env` file as `ORDERMANAGER_API_KEY`.

### Health Check

Check service health and dependency status:

```bash
# Basic health check
curl http://localhost:4600/health

# Liveness probe (service is running)
curl http://localhost:4600/live

# Readiness probe (dependencies available)
curl http://localhost:4600/ready
```

Example response from `/ready`:

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

### Order Queries

Query orders with filtering, pagination, and sorting:

```bash
# List all orders with pagination
curl -X GET "http://localhost:4600/api/v1/orders?page=1&page_size=20" \
  -H "X-API-Key: your-api-key-here"

# Filter by asset and status
curl -X GET "http://localhost:4600/api/v1/orders?asset=BTCUSDT&status=filled&page=1&page_size=10" \
  -H "X-API-Key: your-api-key-here"

# Filter by date range
curl -X GET "http://localhost:4600/api/v1/orders?date_from=2025-01-01T00:00:00Z&date_to=2025-01-31T23:59:59Z" \
  -H "X-API-Key: your-api-key-here"

# Get order by Bybit order ID
curl -X GET "http://localhost:4600/api/v1/orders/12345678" \
  -H "X-API-Key: your-api-key-here"
```

Query parameters:
- `asset`: Trading pair (e.g., BTCUSDT)
- `status`: Order status (pending, partially_filled, filled, cancelled, rejected, dry_run)
- `signal_id`: Trading signal UUID
- `order_id`: Bybit order ID
- `side`: Order side (Buy, Sell)
- `date_from`: Start date (ISO 8601)
- `date_to`: End date (ISO 8601)
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20, max: 100)
- `sort_by`: Field to sort by (created_at, updated_at, executed_at)
- `sort_order`: Sort direction (asc, desc)

### Position Queries

Query current trading positions:

```bash
# List all positions
curl -X GET "http://localhost:4600/api/v1/positions" \
  -H "X-API-Key: your-api-key-here"

# Get position for specific asset
curl -X GET "http://localhost:4600/api/v1/positions/BTCUSDT" \
  -H "X-API-Key: your-api-key-here"
```

### Manual Synchronization

Manually trigger order state synchronization with Bybit exchange:

```bash
# Sync all active orders
curl -X POST "http://localhost:4600/api/v1/sync" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"scope": "active"}'

# Sync all orders (active and completed)
curl -X POST "http://localhost:4600/api/v1/sync" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"scope": "all"}'
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

## Message Queues

### Consumed Queues

- `model-service.trading_signals`: Trading signals from model service

The service automatically consumes trading signals from this queue. Signals should follow this format:

```json
{
  "signal_id": "uuid",
  "signal_type": "buy",
  "asset": "BTCUSDT",
  "amount": 1000.0,
  "confidence": 0.85,
  "timestamp": "2025-01-27T10:00:00Z",
  "strategy_id": "momentum_v1",
  "model_version": "v1.0.0",
  "is_warmup": false,
  "market_data_snapshot": {
    "price": 50000.0,
    "spread": 1.5,
    "volume_24h": 1000000.0,
    "volatility": 0.02
  },
  "metadata": {},
  "trace_id": "optional-trace-id"
}
```

### Published Queues

- `order-manager.order_events`: Enriched order execution events

The service publishes enriched order events when order states change (filled, partially_filled, cancelled, rejected). Event format:

```json
{
  "event_id": "uuid",
  "event_type": "filled",
  "order_id": "bybit-order-id",
  "signal_id": "uuid",
  "asset": "BTCUSDT",
  "side": "Buy",
  "order_type": "Market",
  "status": "filled",
  "quantity": "0.1",
  "filled_quantity": "0.1",
  "average_price": "50000.0",
  "fees": "2.5",
  "executed_at": "2025-01-27T10:00:00Z",
  "trace_id": "trace-id"
}
```

## Usage Examples

### Testing with Dry-Run Mode

Enable dry-run mode to test order logic without executing real orders:

1. Set in `.env`:
```bash
ORDERMANAGER_ENABLE_DRY_RUN=true
```

2. Restart the service:
```bash
docker compose restart order-manager
```

3. Send a test trading signal to RabbitMQ:
```bash
docker compose exec rabbitmq rabbitmqadmin publish \
  routing_key=model-service.trading_signals \
  payload='{"signal_id":"test-001","signal_type":"buy","asset":"BTCUSDT","amount":1000.0,"confidence":0.85,"timestamp":"2025-01-27T10:00:00Z","strategy_id":"test","is_warmup":true}'
```

4. Check logs to see order processing:
```bash
docker compose logs -f order-manager
```

5. Query the order (will have `status: "dry_run"`):
```bash
curl -X GET "http://localhost:4600/api/v1/orders?asset=BTCUSDT" \
  -H "X-API-Key: your-api-key-here"
```

### Monitoring Order Processing

Monitor order processing in real-time:

```bash
# Watch service logs
docker compose logs -f order-manager

# Filter by trace ID
docker compose logs order-manager | grep "trace-12345"

# Check RabbitMQ queue status
docker compose exec rabbitmq rabbitmqctl list_queues name messages
```

### Troubleshooting

**Service won't start:**
- Check logs: `docker compose logs order-manager`
- Verify database connection: Ensure PostgreSQL is running and credentials are correct
- Verify RabbitMQ connection: Ensure RabbitMQ is running and credentials are correct
- Check Bybit API credentials: Verify API key and secret in `.env`

**Orders not being created:**
- Verify trading signals are being published to `model-service.trading_signals` queue
- Check service logs for signal processing errors
- Ensure dry-run mode is disabled if you want real orders
- Verify Bybit API credentials are valid and have trading permissions
- Check available balance for orders

**Order state not updating:**
- Verify WebSocket Gateway is operational
- Check that Order Manager has subscribed to order execution events
- Verify WebSocket connection is active (check WebSocket Gateway logs)
- Try manual sync: `POST /api/v1/sync`

**Database connection issues:**
- Verify PostgreSQL is accessible:
  ```bash
  docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT 1;"
  ```
- Check migration status:
  ```bash
  docker compose exec postgres psql -U ytrader -d ytrader -c "\dt"
  ```

## Database

The service uses a shared PostgreSQL database for orders, signal-order relationships, and positions. Database migrations are managed in the `ws-gateway` service per constitution requirement.

## Development

### Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Flake8 (max line length: 100)
- **Type Checking**: mypy (Python 3.11)

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Contract tests only
pytest tests/contract/
```

## License

See repository root for license information.

