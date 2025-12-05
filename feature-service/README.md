# Feature Service

A microservice that receives market data streams from ws-gateway via RabbitMQ queues, computes real-time features (price, orderflow, orderbook, perpetual, temporal features) with latency ≤ 70ms, stores raw market data in Parquet format for 90+ days, and rebuilds features from historical data for model training datasets with explicit train/validation/test splits.

## Features

- **Real-Time Feature Computation**: Computes features from market data streams with latency ≤ 70ms
- **Dataset Building**: Builds training datasets from historical data with explicit train/validation/test splits
- **Raw Data Storage**: Stores raw market data in Parquet format for 90+ days
- **Data Quality Monitoring**: Tracks data quality metrics and provides quality reports
- **Feature Registry**: Manages feature configuration with validation and versioning
- **REST API**: Provides endpoints for feature retrieval, dataset building, and configuration management

## Prerequisites

- Docker and Docker Compose V2
- Python 3.11+ (for local development)
- PostgreSQL 15+
- RabbitMQ 3+

## Quick Start

### Using Docker Compose

1. Copy environment variables from project root:

   ```bash
   cp ../env.example ../.env
   ```

2. Update `../.env` with your database passwords and API keys

3. Start services:

   ```bash
   docker compose up -d feature-service
   ```

4. Check service health:

   ```bash
   curl http://localhost:4900/health
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

3. Set environment variables (see `../env.example` in project root)

4. Run the service:

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 4900 --reload
   ```

## Configuration

All configuration is done via environment variables. See `../env.example` in project root for available options.

### Key Configuration Variables

- `FEATURE_SERVICE_PORT`: REST API port (default: 4900)
- `FEATURE_SERVICE_API_KEY`: API key for REST API authentication
- `POSTGRES_HOST`: PostgreSQL hostname
- `RABBITMQ_HOST`: RabbitMQ hostname
- `WS_GATEWAY_HOST`: WebSocket Gateway hostname for subscription management
- `RAW_DATA_STORAGE_PATH`: Base directory for raw market data storage
- `DATASET_STORAGE_PATH`: Base directory for dataset storage

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status including database and message queue connectivity.

### Feature Retrieval

```bash
GET /features/latest?symbol=BTCUSDT
```

Returns latest computed features for a symbol.

### Dataset Building

```bash
POST /dataset/build
```

Creates a new dataset build request with train/validation/test period specifications.

### Feature Registry

```bash
GET /feature-registry
POST /feature-registry/reload
GET /feature-registry/validate
```

Manages Feature Registry configuration.

### Data Quality

```bash
GET /data-quality/report?symbol=BTCUSDT&from=2025-01-01T00:00:00Z&to=2025-01-02T00:00:00Z
```

Returns data quality report for a symbol over a specified time period.

## Project Structure

```text
feature-service/
├── src/
│   ├── api/                    # REST API endpoints
│   ├── services/               # Business logic services
│   ├── models/                 # Data models
│   ├── consumers/              # Message queue consumers
│   ├── publishers/             # Message queue publishers
│   ├── features/               # Feature computation modules
│   ├── storage/                # Storage services (Parquet, PostgreSQL)
│   └── main.py                 # Application entry point
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── contract/               # Contract tests
│   └── fixtures/               # Test fixtures
├── config/
│   └── feature_registry.yaml   # Default Feature Registry configuration
├── Dockerfile
├── requirements.txt
└── README.md
```

## Testing

Run tests inside Docker container with **ONLY** this command:

```bash
docker compose run --rm feature-service pytest tests/ -v
```

## License

See project root LICENSE file.
