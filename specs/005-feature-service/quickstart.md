# Quickstart: Feature Service

**Feature**: Feature Service for Real-Time Feature Computation and Dataset Building  
**Date**: 2025-01-27

## Overview

This guide provides step-by-step instructions to set up and run the Feature Service locally for development and testing. The service receives market data from ws-gateway, computes real-time features, stores raw data, and builds datasets for model training.

## Prerequisites

- Docker and Docker Compose V2 installed
- Access to PostgreSQL database (shared database, managed by ws-gateway)
- Access to RabbitMQ (for consuming market data and publishing features)
- ws-gateway service running and publishing market data to queues

## Quick Setup

### 1. Clone and Navigate

```bash
cd /home/ubuntu/ytrader
git checkout 005-feature-service
```

### 2. Configure Environment

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` and set the following variables:

```bash
# Database Configuration (shared PostgreSQL)
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

# Feature Service Configuration
FEATURE_SERVICE_PORT=4500
FEATURE_SERVICE_API_KEY=your-feature-service-api-key  # For REST API authentication
FEATURE_SERVICE_LOG_LEVEL=INFO

# Raw Data Storage
FEATURE_SERVICE_DATA_DIR=/data/feature-service  # Mounted volume for Parquet files
FEATURE_SERVICE_RETENTION_DAYS=90  # Minimum retention period


# Feature Registry
FEATURE_REGISTRY_PATH=/app/config/feature_registry.yaml
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

### 4. Ensure ws-gateway is Running

Feature Service consumes market data from ws-gateway queues. Ensure ws-gateway is running and publishing data:

```bash
docker compose ps ws-gateway
# Should show "Up" status

# Verify queues exist
docker compose exec rabbitmq rabbitmqadmin list queues name | grep "ws-gateway"
```

### 5. Run Database Migrations

**Note**: Per constitution principle II (Shared Database Strategy), PostgreSQL migrations for Feature Service are located in `ws-gateway/migrations/` (ws-gateway is the designated owner of all PostgreSQL migrations).

```bash
# Migrations should be run from ws-gateway service
docker compose run --rm ws-gateway python -m migrations.run
```

Or if migrations are run automatically on startup, skip this step.

### 6. Start Feature Service

```bash
docker compose up -d feature-service
```

### 7. Verify Service Health

```bash
curl http://localhost:4500/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:00:00Z",
  "services": {
    "database": "connected",
    "message_queue": "connected",
  }
}
```

## Basic Usage

### Get Latest Features

Retrieve the latest computed feature vector for a symbol:

```bash
curl -X GET "http://localhost:4500/features/latest?symbol=BTCUSDT" \
  -H "X-API-Key: your-feature-service-api-key"
```

Response:

```json
{
  "timestamp": "2025-01-27T10:00:00.000Z",
  "symbol": "BTCUSDT",
  "features": {
    "mid_price": 50000.0,
    "spread_abs": 1.0,
    "spread_rel": 0.00002,
    "returns_1s": 0.0001,
    "returns_3s": 0.0003,
    "returns_1m": 0.001,
    "vwap_3s": 50000.5,
    "volume_3s": 10.5,
    ...
  },
  "feature_registry_version": "1.0.0",
  "trace_id": "abc123"
}
```

### Build a Dataset

Request dataset building with time-based split:

```bash
curl -X POST http://localhost:4500/dataset/build \
  -H "X-API-Key: your-feature-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "split_strategy": "time_based",
    "train_period_start": "2025-01-01T00:00:00Z",
    "train_period_end": "2025-01-20T00:00:00Z",
    "validation_period_start": "2025-01-20T00:00:00Z",
    "validation_period_end": "2025-01-25T00:00:00Z",
    "test_period_start": "2025-01-25T00:00:00Z",
    "test_period_end": "2025-01-27T00:00:00Z",
    "target_config": {
      "type": "regression",
      "horizon": "1m"
    },
    "feature_registry_version": "1.0.0",
    "data_leakage_check": true,
    "output_format": "parquet"
  }'
```

Response:

```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "building",
  "estimated_completion": "2025-01-27T10:30:00Z",
  "splits_info": {
    "train": {
      "records": 0,
      "period_start": "2025-01-01T00:00:00Z",
      "period_end": "2025-01-20T00:00:00Z"
    },
    "validation": {
      "records": 0,
      "period_start": "2025-01-20T00:00:00Z",
      "period_end": "2025-01-25T00:00:00Z"
    },
    "test": {
      "records": 0,
      "period_start": "2025-01-25T00:00:00Z",
      "period_end": "2025-01-27T00:00:00Z"
    }
  }
}
```

### Build Dataset with Walk-Forward Validation

```bash
curl -X POST http://localhost:4500/dataset/build \
  -H "X-API-Key: your-feature-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "split_strategy": "walk_forward",
    "walk_forward_config": {
      "train_window_days": 30,
      "validation_window_days": 7,
      "step_days": 7,
      "start_date": "2025-01-01T00:00:00Z",
      "end_date": "2025-01-27T00:00:00Z",
      "min_train_samples": 1000
    },
    "target_config": {
      "type": "classification",
      "horizon": "5m",
      "threshold": 0.001
    },
    "feature_registry_version": "1.0.0",
    "output_format": "parquet"
  }'
```

### List Datasets

```bash
curl -X GET "http://localhost:4500/dataset/list?status=ready&symbol=BTCUSDT" \
  -H "X-API-Key: your-feature-service-api-key"
```

### Get Dataset Metadata

```bash
curl -X GET "http://localhost:4500/dataset/{dataset_id}" \
  -H "X-API-Key: your-feature-service-api-key"
```

### Download Dataset Split

```bash
curl -X GET "http://localhost:4500/dataset/{dataset_id}/download?split=train" \
  -H "X-API-Key: your-feature-service-api-key" \
  -o train.parquet
```

### Get Feature Registry

```bash
curl -X GET "http://localhost:4500/feature-registry" \
  -H "X-API-Key: your-feature-service-api-key"
```

### Reload Feature Registry

```bash
curl -X POST "http://localhost:4500/feature-registry/reload" \
  -H "X-API-Key: your-feature-service-api-key"
```

### Validate Feature Registry

```bash
curl -X GET "http://localhost:4500/feature-registry/validate" \
  -H "X-API-Key: your-feature-service-api-key"
```

### Get Data Quality Report

```bash
curl -X GET "http://localhost:4500/data-quality/report?symbol=BTCUSDT&from=2025-01-27T00:00:00Z&to=2025-01-27T23:59:59Z" \
  -H "X-API-Key: your-feature-service-api-key"
```

Response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "BTCUSDT",
  "period_start": "2025-01-27T00:00:00Z",
  "period_end": "2025-01-27T23:59:59Z",
  "missing_rate": 0.001,
  "anomaly_rate": 0.0001,
  "sequence_gaps": 2,
  "desynchronization_events": 1,
  "anomaly_details": {...},
  "sequence_gap_details": {...},
  "recommendations": [
    "Monitor orderbook sequence gaps",
    "Check ws-gateway connection stability"
  ],
  "created_at": "2025-01-27T23:59:59Z"
}
```

### Consume Features from Message Queue

Features are published to `features.live` queue. To consume:

```bash
# Install RabbitMQ client tools (if not already installed)
docker compose exec rabbitmq rabbitmqadmin list queues

# Consume features (example using Python)
python -c "
import pika
import json
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672))
channel = connection.channel()
channel.queue_declare(queue='features.live')

def callback(ch, method, properties, body):
    feature_vector = json.loads(body.decode())
    print(f'Received features for {feature_vector[\"symbol\"]} at {feature_vector[\"timestamp\"]}')

channel.basic_consume(queue='features.live', on_message_callback=callback, auto_ack=True)
print('Waiting for feature vectors...')
channel.start_consuming()
"
```

## Development Workflow

### Running Tests

Unit tests (run in service container):

```bash
docker compose run --rm feature-service pytest tests/unit -v
```

Integration tests (run in test container):

```bash
docker compose run --rm test-feature-service pytest tests/integration -v
```

Feature identity tests (online vs offline):

```bash
docker compose run --rm test-feature-service pytest tests/integration/test_feature_identity.py -v
```

End-to-end tests:

```bash
docker compose run --rm test-feature-service pytest tests/e2e -v
```

### Viewing Logs

```bash
# All logs
docker compose logs -f feature-service

# Last 100 lines
docker compose logs --tail 100 feature-service

# Filter by trace ID
docker compose logs feature-service | grep "trace_id=abc123"

# Filter by symbol
docker compose logs feature-service | grep "symbol=BTCUSDT"
```

### Database Access

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U ytrader -d ytrader

# View datasets
SELECT * FROM datasets WHERE status = 'ready' ORDER BY created_at DESC LIMIT 10;

# View Feature Registry versions
SELECT * FROM feature_registry_versions WHERE is_active = true;

# View data quality reports
SELECT * FROM data_quality_reports ORDER BY created_at DESC LIMIT 10;
```

### Inspect Raw Data Files

```bash
# List Parquet files
docker compose exec feature-service ls -lh /data/feature-service/orderbook/snapshots/2025-01-27/

# Read Parquet file (using Python in container)
docker compose exec feature-service python -c "
import pyarrow.parquet as pq
table = pq.read_table('/data/feature-service/trades/2025-01-27/BTCUSDT.parquet')
print(table.to_pandas().head())
"
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

### Features Not Computing

1. Check market data is being received:
   ```bash
   docker compose logs feature-service | grep -i "market data\|consumer"
   ```

2. Verify queues exist and have messages:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name messages | grep "ws-gateway"
   ```

3. Check Feature Registry is loaded:
   ```bash
   curl -X GET "http://localhost:4500/feature-registry" \
     -H "X-API-Key: your-feature-service-api-key"
   ```

### High Latency (>70ms)

1. Check service health:
   ```bash
   curl http://localhost:4500/health
   ```

2. Monitor logs for performance issues:
   ```bash
   docker compose logs --tail 100 -f feature-service | grep -i "latency\|performance"
   ```

3. Check database connection:
   ```bash
   docker compose logs feature-service | grep -i "postgres\|database"
   ```

### Dataset Building Fails

1. Check dataset status:
   ```bash
   curl -X GET "http://localhost:4500/dataset/{dataset_id}" \
     -H "X-API-Key: your-feature-service-api-key"
   ```

2. Verify raw data exists for requested period:
   ```bash
   docker compose exec feature-service ls -lh /data/feature-service/trades/
   ```

3. Check logs for errors:
   ```bash
   docker compose logs feature-service | grep -i "dataset\|error"
   ```


### Orderbook Desynchronization

1. Check data quality report:
   ```bash
   curl -X GET "http://localhost:4500/data-quality/report?symbol=BTCUSDT&from=...&to=..." \
     -H "X-API-Key: your-feature-service-api-key"
   ```

2. Monitor logs for desynchronization events:
   ```bash
   docker compose logs feature-service | grep -i "desync\|sequence.*gap"
   ```

3. Verify orderbook snapshots are being stored:
   ```bash
   docker compose exec feature-service ls -lh /data/feature-service/orderbook/snapshots/
   ```

## Production Deployment

### Environment Configuration

1. Use strong API keys and secrets
2. Configure proper logging levels (`FEATURE_SERVICE_LOG_LEVEL=WARNING`)
3. Use secure database credentials
4. Enable SSL/TLS for RabbitMQ and PostgreSQL connections
5. Configure data retention policies (`FEATURE_SERVICE_RETENTION_DAYS=90`)
6. Set up volume mounts for Parquet data storage with sufficient capacity

### Monitoring

- Monitor feature computation latency (target: ≤70ms at 95th percentile)
- Track dataset building completion time (target: ≤2 hours for 1 month of data)
- Monitor data quality metrics (missing rate, anomaly rate, sequence gaps)
- Set up alerts for high latency, dataset build failures, and data quality issues
- Monitor raw data storage usage and retention

### Scaling

- Service can scale horizontally by symbol (each instance handles subset of symbols)
- Raw data storage: Partition by date and symbol for efficient access
- Dataset building: Can run multiple builds in parallel (with resource limits)
- Feature computation: In-memory state per symbol, can be distributed

## Next Steps

- Read the [full specification](./spec.md) for detailed requirements
- Review the [data model](./data-model.md) for database schema and data structures
- Check [API contracts](./contracts/) for complete API documentation
- See [implementation plan](./plan.md) for architecture details
- Review [research](./research.md) for technology decisions

## Support

For issues or questions:
1. Check service logs: `docker compose logs feature-service`
2. Review health endpoint: `curl http://localhost:4500/health`
3. Check data quality reports for data issues
4. Consult the specification and documentation in this directory

