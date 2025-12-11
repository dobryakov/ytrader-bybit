# Feature Service

A microservice that receives market data streams from ws-gateway via RabbitMQ queues, computes real-time features (price, orderflow, orderbook, perpetual, temporal features) with latency ≤ 70ms, stores raw market data in Parquet format for 90+ days, and rebuilds features from historical data for model training datasets with explicit train/validation/test splits.

## Features

- **Real-Time Feature Computation**: Computes features from market data streams with latency ≤ 70ms
- **Dataset Building**: Builds training datasets from historical data with explicit train/validation/test splits
- **Historical Data Backfilling**: Fetches historical market data from Bybit REST API when insufficient data is available
- **Raw Data Storage**: Stores raw market data in Parquet format for 90+ days
- **Data Quality Monitoring**: Tracks data quality metrics and provides quality reports
- **Feature Registry**: Manages feature configuration with validation and versioning
- **REST API**: Provides endpoints for feature retrieval, dataset building, backfilling, and configuration management

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
- `FEATURE_SERVICE_BACKFILL_ENABLED`: Enable/disable backfilling feature (default: `true`)
- `FEATURE_SERVICE_BACKFILL_AUTO`: Enable/disable automatic backfilling (default: `true`)
- `FEATURE_SERVICE_BACKFILL_MAX_DAYS`: Maximum days to backfill in one operation (default: `90`)
- `BYBIT_API_KEY`: Bybit API key (optional, not required for public market data endpoints)
- `BYBIT_API_SECRET`: Bybit API secret (optional, not required for public market data endpoints)
- `BYBIT_ENVIRONMENT`: Bybit environment: `mainnet` or `testnet` (default: `mainnet`)
- `FEATURE_REGISTRY_CONFIG_PATH`: Path to legacy Feature Registry YAML file (default: `/app/config/feature_registry.yaml`)
- `FEATURE_REGISTRY_VERSIONS_DIR`: Directory for Feature Registry version files (default: `/app/config/versions`)
- `FEATURE_REGISTRY_USE_DB`: Use database-driven version management (default: `true`)
- `FEATURE_REGISTRY_AUTO_MIGRATE`: Automatically migrate legacy feature_registry.yaml to versioned storage on startup (default: `true`)

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

### Historical Data Backfilling

Feature Service can automatically fetch historical market data from Bybit REST API when insufficient data is available for model training. This allows immediate model training without waiting for data accumulation through WebSocket streams.

#### Manual Backfilling

Backfill historical data for a specific date range:

```bash
curl -X POST "http://localhost:4900/backfill/historical" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${FEATURE_SERVICE_API_KEY}" \
  -d '{
    "symbol": "BTCUSDT",
    "start_date": "2025-12-04",
    "end_date": "2025-12-07"
  }'
```

**Request Parameters:**
- `symbol` (required): Trading pair symbol (e.g., "BTCUSDT")
- `start_date` (required): Start date in YYYY-MM-DD format
- `end_date` (required): End date in YYYY-MM-DD format
- `data_types` (optional): List of data types to backfill (e.g., `["klines"]`). If not provided, uses Feature Registry to determine required data types.

**Response:**
```json
{
  "job_id": "backfill-job-123",
  "status": "pending",
  "message": "Backfilling job started"
}
```

#### Automatic Backfilling

Automatically backfill missing data up to a specified number of days. Uses Feature Registry to determine which data types to backfill:

```bash
curl -X POST "http://localhost:4900/backfill/auto" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${FEATURE_SERVICE_API_KEY}" \
  -d '{
    "symbol": "BTCUSDT",
    "max_days": 3
  }'
```

**Request Parameters:**
- `symbol` (required): Trading pair symbol
- `max_days` (optional): Maximum days to backfill (defaults to `FEATURE_SERVICE_BACKFILL_MAX_DAYS` from config)

#### Check Backfilling Job Status

```bash
JOB_ID="backfill-job-123"

curl -X GET "http://localhost:4900/backfill/status/${JOB_ID}" \
  -H "X-API-Key: ${FEATURE_SERVICE_API_KEY}"
```

**Response:**

Status: `in_progress` (job is currently running):
```json
{
  "job_id": "151a2a02-f59c-4273-ab96-c21ef652b13b",
  "symbol": "BTCUSDT",
  "start_date": "2025-12-04",
  "end_date": "2025-12-07",
  "data_types": ["klines"],
  "status": "in_progress",
  "progress": {
    "dates_completed": 0,
    "dates_total": 1,
    "current_date": "2025-12-04"
  },
  "start_time": "2025-12-07T12:53:26.668331+00:00",
  "end_time": null,
  "error_message": null,
  "completed_dates": [],
  "failed_dates": []
}
```

Status: `completed` (job finished successfully):
```json
{
  "job_id": "backfill-job-123",
  "symbol": "BTCUSDT",
  "start_date": "2025-12-04",
  "end_date": "2025-12-07",
  "data_types": ["klines"],
  "status": "completed",
  "progress": {
    "dates_completed": 4,
    "dates_total": 4
  },
  "start_time": "2025-12-07T12:53:26.668331+00:00",
  "end_time": "2025-12-07T12:55:10.123456+00:00",
  "error_message": null,
  "completed_dates": ["2025-12-04", "2025-12-05", "2025-12-06", "2025-12-07"],
  "failed_dates": []
}
```

Status: `failed` (job encountered errors):
```json
{
  "job_id": "backfill-job-456",
  "symbol": "BTCUSDT",
  "start_date": "2025-12-04",
  "end_date": "2025-12-07",
  "data_types": ["klines"],
  "status": "failed",
  "progress": {
    "dates_completed": 2,
    "dates_total": 4
  },
  "start_time": "2025-12-07T12:53:26.668331+00:00",
  "end_time": "2025-12-07T12:54:15.789012+00:00",
  "error_message": "Rate limit exceeded for date 2025-12-06",
  "completed_dates": ["2025-12-04", "2025-12-05"],
  "failed_dates": ["2025-12-06", "2025-12-07"]
}
```

**Response Fields:**
- `job_id`: Unique job identifier
- `symbol`: Trading pair symbol
- `start_date`, `end_date`: Date range for backfilling
- `data_types`: List of data types being backfilled
- `status`: Job status (`pending`, `in_progress`, `completed`, `failed`)
- `progress`: Progress information with `dates_completed`, `dates_total`, and optionally `current_date`
- `start_time`, `end_time`: Job start and end timestamps (ISO format)
- `error_message`: Error message if job failed (null otherwise)
- `completed_dates`: List of successfully processed dates
- `failed_dates`: List of dates that failed to process

#### Example: Backfill Last 3 Days

```bash
# Calculate dates
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "3 days ago" +%Y-%m-%d 2>/dev/null || \
  python3 -c "from datetime import date, timedelta; print((date.today() - timedelta(days=3)).isoformat())")

# Start backfilling
curl -X POST "http://localhost:4900/backfill/historical" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${FEATURE_SERVICE_API_KEY}" \
  -d "{
    \"symbol\": \"BTCUSDT\",
    \"start_date\": \"${START_DATE}\",
    \"end_date\": \"${END_DATE}\"
  }"
```

#### Configuration

Backfilling behavior is controlled by environment variables:

- `FEATURE_SERVICE_BACKFILL_ENABLED`: Enable/disable backfilling feature (default: `true`)
- `FEATURE_SERVICE_BACKFILL_AUTO`: Enable/disable automatic backfilling when data insufficient (default: `true`)
- `FEATURE_SERVICE_BACKFILL_MAX_DAYS`: Maximum days to backfill in one operation (default: `90`)
- `FEATURE_SERVICE_BACKFILL_RATE_LIMIT_DELAY_MS`: Delay between API requests in milliseconds (default: `100`)
- `FEATURE_SERVICE_BACKFILL_DEFAULT_INTERVAL`: Default kline interval in minutes (default: `1`)

**Note:** Backfilling uses Bybit REST API public endpoints and does not require API keys for market data. API keys are optional and only needed for authenticated endpoints (future use).

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
