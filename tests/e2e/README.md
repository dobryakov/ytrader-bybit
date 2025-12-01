# End-to-End Tests for Trading Chain

## Overview

This directory contains end-to-end tests that verify the complete trading flow from signal generation to order execution and position tracking.

## Test: `test_trading_chain_e2e.py`

This test verifies the entire trading chain:

1. **Signal Generation**: Sends a trading signal to RabbitMQ queue `model-service.trading_signals` (simulating model service trigger)
2. **Order Processing**: Verifies order-manager consumes the signal and creates an order in the database
3. **Order Execution**: Verifies order is placed on Bybit (or in dry-run mode)
4. **Position Tracking**: Verifies position-manager receives position updates via WebSocket
5. **Execution Events**: Verifies model-service receives execution events

## Prerequisites

Before running E2E tests, ensure all required services are running:

```bash
# Start all required services
docker compose up -d postgres rabbitmq order-manager position-manager model-service ws-gateway

# Wait for services to be healthy
docker compose ps
# Wait until all services show "healthy" or "running" status
```

## Running Tests

### Using pytest (Recommended)

```bash
# Run BUY order E2E test
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_trading_chain_e2e.py::test_buy_order_e2e -v

# Run complete buy-sell cycle E2E test
docker compose --profile test run --rm test-grafana pytest tests/e2e/test_trading_chain_e2e.py::test_buy_sell_cycle_e2e -v

# Run all E2E tests
docker compose --profile test run --rm test-grafana pytest tests/e2e -v
```

### Using Standalone Script

```bash
# Run BUY order test with custom parameters
docker compose --profile test run --rm test-grafana python tests/e2e/test_trading_chain_e2e.py \
  --asset ETHUSDT \
  --amount 50.0

# Run buy-sell cycle
docker compose --profile test run --rm test-grafana python tests/e2e/test_trading_chain_e2e.py \
  --asset ETHUSDT \
  --amount 50.0 \
  --buy-sell

# Run in real Bybit mode (not dry-run)
docker compose --profile test run --rm test-grafana python tests/e2e/test_trading_chain_e2e.py \
  --asset ETHUSDT \
  --amount 50.0 \
  --dry-run=false
```

## Test Parameters

- `--asset`: Trading pair (default: `ETHUSDT`)
- `--amount`: Order amount in USDT (default: `50.0`)
- `--buy-sell`: Run complete buy-sell cycle (default: False, only BUY)
- `--dry-run`: Run in dry-run mode (default: True)

## Test Flow

### BUY Order Flow

1. **Send Signal**: Test sends BUY signal to `model-service.trading_signals` queue
2. **Order Creation**: Order-manager consumes signal and creates order in database
3. **Order Validation**: Test verifies order was created with correct parameters
4. **Order Execution**: Test waits for order execution (if not dry-run)
5. **Position Update**: Test verifies position was updated in database
6. **Execution Event**: Test verifies execution event was published

### Buy-Sell Cycle Flow

1. **BUY Flow**: Complete BUY order flow (as above)
2. **Wait**: 5 second delay between BUY and SELL
3. **SELL Flow**: Complete SELL order flow (same as BUY, but with SELL signal)

## Expected Results

### Success Criteria

- ✅ Signal sent to RabbitMQ
- ✅ Order created in database
- ✅ Order status is not "rejected"
- ✅ Order executed (if not dry-run)
- ✅ Position updated (if order executed)
- ✅ Execution event published (if order executed)

### Dry-Run Mode

When running in dry-run mode (`ORDERMANAGER_ENABLE_DRY_RUN=true`):
- Orders are created in database but not sent to Bybit
- Order status will be "dry_run"
- Execution and position checks are skipped (marked as N/A)

## Troubleshooting

### Test fails with "Order not created"

1. **Check order-manager logs**:
   ```bash
   docker compose logs order-manager --tail 100
   ```

2. **Check RabbitMQ queue**:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues
   ```

3. **Verify signal was sent**:
   ```bash
   docker compose logs test-grafana | grep "Sent.*signal"
   ```

### Test fails with "Order was rejected"

1. **Check rejection reason**:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c \
     "SELECT order_id, rejection_reason FROM orders WHERE status = 'rejected' ORDER BY created_at DESC LIMIT 1;"
   ```

2. **Common rejection reasons**:
   - Insufficient balance
   - Risk limits exceeded
   - Invalid order parameters
   - Bybit API errors

### Test fails with connection errors

1. **Check services are running**:
   ```bash
   docker compose ps
   ```

2. **Check network connectivity**:
   ```bash
   docker compose exec test-grafana ping -c 3 rabbitmq
   docker compose exec test-grafana ping -c 3 postgres
   ```

3. **Verify environment variables**:
   ```bash
   docker compose exec test-grafana env | grep -E "POSTGRES|RABBITMQ"
   ```

### Position not updated

1. **Check position-manager logs**:
   ```bash
   docker compose logs position-manager --tail 100
   ```

2. **Check WebSocket connection**:
   ```bash
   docker compose logs ws-gateway --tail 100 | grep -i position
   ```

3. **Verify position in database**:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c \
     "SELECT asset, size, last_updated FROM positions WHERE asset = 'ETHUSDT';"
   ```

## Environment Variables

The test uses the following environment variables (set in docker-compose.yml):

- `POSTGRES_HOST`: PostgreSQL host (default: `postgres`)
- `POSTGRES_PORT`: PostgreSQL port (default: `5432`)
- `POSTGRES_DB`: Database name (default: `ytrader`)
- `POSTGRES_USER`: Database user (default: `ytrader`)
- `POSTGRES_PASSWORD`: Database password
- `RABBITMQ_HOST`: RabbitMQ host (default: `rabbitmq`)
- `RABBITMQ_PORT`: RabbitMQ port (default: `5672`)
- `RABBITMQ_USER`: RabbitMQ user (default: `guest`)
- `RABBITMQ_PASSWORD`: RabbitMQ password (default: `guest`)
- `ORDERMANAGER_ENABLE_DRY_RUN`: Enable dry-run mode (default: `true`)

## Notes

- Tests are designed to run in Docker containers connected to the main `docker-compose.yml` network
- Tests use real database and RabbitMQ connections (not mocks)
- In dry-run mode, orders are not actually placed on Bybit
- Test timeouts are configurable (default: 30-60 seconds per step)
- Tests clean up after themselves (no manual cleanup required)

