# Integration Tests for Grafana Monitoring

This directory contains integration tests for the Grafana monitoring service.

## Test Structure

- `test_grafana_container.py` - Tests for Grafana container startup and health endpoints
- `test_grafana_datasources.py` - Tests for Grafana data source connectivity and configuration

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

### Container Tests (`test_grafana_container.py`)

- ✅ Grafana container starts successfully
- ✅ Health endpoint responds correctly
- ✅ API is accessible

### Data Source Tests (`test_grafana_datasources.py`)

- ✅ PostgreSQL data source is configured
- ✅ RabbitMQ HTTP API data source is configured
- ✅ Service health data sources are configured (ws-gateway, model-service, order-manager)
- ✅ PostgreSQL data source connectivity

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

