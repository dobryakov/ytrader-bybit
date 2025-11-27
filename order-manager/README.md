# Order Manager Microservice

A microservice that receives high-level trading signals from the model service via RabbitMQ, executes them as orders on Bybit exchange, maintains accurate order state through WebSocket event subscriptions, and publishes enriched order execution events.

## Features

- **Signal Processing**: Receive and process trading signals from model service via RabbitMQ
- **Order Execution**: Execute orders on Bybit exchange with configurable order type selection (market vs limit)
- **Order State Management**: Maintain accurate order state through WebSocket event subscriptions
- **Position Management**: Track and manage trading positions with periodic snapshots and validation
- **Risk Management**: Enforce risk limits (max exposure, max order size, position size limits)
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
ORDERMANAGER_POSITION_SNAPSHOT_INTERVAL=300
ORDERMANAGER_POSITION_VALIDATION_INTERVAL=3600

# Order Cancellation Configuration
ORDERMANAGER_CANCEL_OPPOSITE_ORDERS_ONLY=false
ORDERMANAGER_CANCEL_STALE_ORDER_TIMEOUT=3600

# Risk Management Configuration
ORDERMANAGER_UNREALIZED_LOSS_WARNING_THRESHOLD=10.0
```

See `env.example` for complete configuration options.

### Docker Setup

1. Build and start the service:

```bash
docker compose build order-manager
docker compose up -d order-manager
```

2. Check service health:

```bash
curl http://localhost:4600/health
```

3. View logs:

```bash
docker compose logs -f order-manager
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

### Health Check

```bash
GET /health
GET /live
GET /ready
```

Returns service health status including database, message queue, and external service checks.

### Order Queries

```bash
GET /api/v1/orders?asset=BTCUSDT&status=filled&limit=10&offset=0
GET /api/v1/orders/{order_id}
```

Query orders with filtering, pagination, and sorting.

### Position Queries

```bash
GET /api/v1/positions?asset=BTCUSDT
GET /api/v1/positions/{asset}
```

Query current positions.

### Manual Synchronization

```bash
POST /api/v1/sync?scope=orders
```

Manually trigger order state synchronization with Bybit exchange.

## Message Queues

### Consumed Queues

- `model-service.trading_signals`: Trading signals from model service

### Published Queues

- `order-manager.order_events`: Enriched order execution events

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

