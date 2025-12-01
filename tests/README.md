# Integration and E2E Tests

This directory contains integration tests for the Grafana monitoring service and end-to-end tests for the complete trading chain.

## Test Structure

### Integration Tests
- `integration/test_grafana_container.py` - Tests for Grafana container startup and health endpoints
- `integration/test_grafana_datasources.py` - Tests for Grafana data source connectivity and configuration

### E2E Tests
- `e2e/test_trading_chain_e2e.py` - Complete end-to-end test for trading chain (signal → order → position → execution)
- `e2e/test_training_orchestrator_e2e.py` - End-to-end test for training orchestrator (execution events → training pipeline)

## Running Tests

### In Docker Container (Recommended)

Tests are designed to run in a Docker container connected to the main `docker-compose.yml` network. This ensures tests can access all services (Grafana, PostgreSQL, RabbitMQ, etc.) using Docker service names.

**Start required services:**
```bash
docker compose up -d grafana postgres rabbitmq
```

**Wait for services to be healthy:**
```bash
docker compose ps
# Wait until all services show "healthy" status
```

**Run integration tests:**
```bash
# Run all integration tests
docker compose --profile test run --rm test-grafana pytest tests/integration -v

# Run specific test file
docker compose --profile test run --rm test-grafana pytest tests/integration/test_grafana_container.py -v

# Run with markers
docker compose --profile test run --rm test-grafana pytest tests/integration -v -m integration
```

**Run E2E tests:**
```bash
# Start all required services
docker compose up -d postgres rabbitmq order-manager position-manager model-service ws-gateway

# Wait for services to be healthy
docker compose ps

# Run E2E test for BUY order flow
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_trading_chain_e2e.py::test_buy_order_e2e -v

# Run E2E test for complete buy-sell cycle
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_trading_chain_e2e.py::test_buy_sell_cycle_e2e -v

# Run all E2E tests
docker compose --profile test run --rm test-grafana pytest tests/e2e -v

# Run training orchestrator E2E test
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_training_orchestrator_e2e.py -v

# Run specific training orchestrator test
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_training_orchestrator_e2e.py::test_training_orchestrator_event_buffering -v

# Run as standalone script (with custom parameters)
docker compose --profile test run --rm test-grafana python tests/e2e/test_trading_chain_e2e.py --asset ETHUSDT --amount 50.0

# Run buy-sell cycle
docker compose --profile test run --rm test-grafana python tests/e2e/test_trading_chain_e2e.py --asset ETHUSDT --amount 50.0 --buy-sell
```

### Local Execution (Development)

Tests can also run locally if Grafana is accessible on `localhost:4700`:

```bash
# Install dependencies
pip install -r tests/requirements.txt

# Set environment variables
export GRAFANA_ADMIN_PASSWORD=your_password
export GRAFANA_PORT=4700

# Run tests
pytest tests/integration -v
```

**Note**: Local execution requires Grafana to be running and accessible. The tests will use `localhost:4700` by default.

## Test Environment Variables

Tests use the following environment variables:

- `DOCKER_ENV` - Set to `true` when running in Docker (automatically set by test container)
- `GRAFANA_HOST` - Grafana hostname (default: `localhost`, in Docker: `grafana`)
- `GRAFANA_PORT` - Grafana port (default: `4700`, in Docker: `3000` internally)
- `GRAFANA_ADMIN_USER` - Grafana admin username (default: `admin`)
- `GRAFANA_ADMIN_PASSWORD` - Grafana admin password (required for authenticated tests)

## Test Container Details

The test container (`test-grafana`) is defined in `docker-compose.yml`:

- **Base image**: `python:3.11-slim`
- **Network**: Connected to `ytrader-network` (same as all services)
- **Profile**: `test` (only starts when explicitly requested)
- **Dependencies**: Waits for `grafana` service to be ready

## Test Coverage

### Integration Tests

#### Container Tests (`test_grafana_container.py`)
- ✅ Grafana container starts successfully
- ✅ Health endpoint responds correctly
- ✅ API is accessible

#### Data Source Tests (`test_grafana_datasources.py`)
- ✅ PostgreSQL data source is configured
- ✅ RabbitMQ HTTP API data source is configured
- ✅ Service health data sources are configured (ws-gateway, model-service, order-manager)
- ✅ PostgreSQL data source connectivity

### E2E Tests

#### Trading Chain E2E (`test_trading_chain_e2e.py`)
- ✅ Send trading signal to RabbitMQ queue `model-service.trading_signals`
- ✅ Order-manager processes signal and creates order in database
- ✅ Order is placed on Bybit (or in dry-run mode)
- ✅ Position-manager receives position updates via WebSocket
- ✅ Model-service receives execution events
- ✅ Complete buy-sell cycle verification

**Test Flow:**
1. **Signal Generation**: Test sends BUY/SELL signal to RabbitMQ (simulating model service)
2. **Order Processing**: Verifies order-manager consumes signal and creates order
3. **Order Execution**: Verifies order is placed on Bybit (or dry-run mode)
4. **Position Tracking**: Verifies position-manager updates position from WebSocket events
5. **Execution Events**: Verifies execution events are published and consumed by model-service

#### Training Orchestrator E2E (`test_training_orchestrator_e2e.py`)
- ✅ Publishes mock execution events to RabbitMQ (simulating order-manager)
- ✅ Verifies events are consumed by model-service and buffered in training orchestrator
- ✅ Verifies training is triggered when conditions are met (enough events, schedule, etc.)
- ✅ Verifies training pipeline (dataset building, model training, quality evaluation)
- ✅ Verifies buffer is cleared after training starts

**Test Flow:**
1. **Event Publishing**: Test publishes mock execution events to `order-manager.order_events` queue
2. **Event Consumption**: Verifies model-service consumes events via `ExecutionEventConsumer`
3. **Event Buffering**: Verifies events are added to training orchestrator buffer
4. **Training Trigger**: Verifies training starts when conditions are met (min_dataset_size, schedule, etc.)
5. **Training Pipeline**: Verifies training completes and creates model version

**Key Features:**
- Does NOT use real orders or Bybit communication
- Tests only the training pipeline by simulating execution events
- Can test with different event counts and configurations
- Verifies training orchestrator API endpoints

## Troubleshooting

### Tests fail with connection errors

1. **Check services are running:**
   ```bash
   docker compose ps
   ```

2. **Check Grafana is healthy:**
   ```bash
   docker compose logs grafana
   curl http://localhost:4700/api/health
   ```

3. **Verify network connectivity:**
   ```bash
   docker compose exec test-grafana ping -c 3 grafana
   ```

### Tests fail with authentication errors

Ensure `GRAFANA_ADMIN_PASSWORD` is set in your `.env` file:
```bash
grep GRAFANA_ADMIN_PASSWORD .env
```

### Tests timeout

Increase timeout in test files or check if Grafana is slow to start:
```bash
docker compose logs grafana | tail -50
```

