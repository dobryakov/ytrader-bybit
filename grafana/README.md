# Grafana Monitoring Service

Grafana monitoring dashboard service for the YTrader trading system. Provides visual monitoring of trading signals, order execution, model state, queue health, system health, WebSocket connections, and event history.

## Overview

The Grafana service runs in a Docker container and provides a web-based dashboard interface for monitoring the trading system. It connects to:

- **PostgreSQL**: Read-only access to trading data (signals, orders, execution events, model versions, quality metrics, subscriptions)
- **RabbitMQ Management API**: Queue metrics and health monitoring
- **Service REST APIs**: Health endpoints for ws-gateway, model-service, and order-manager

## Features

### Dashboard Panels

1. **Trading Signals** - Recent trading signals with asset, side, price, confidence, timestamp, strategy ID
2. **Order Execution** - Recent orders with execution status, prices, quantities, fees, and performance metrics
3. **Model State** - Active model versions, training status, warmup mode status
4. **Model Quality Metrics** - Win rate, total orders, successful orders, total PnL
5. **Queue Metrics** - RabbitMQ queue lengths, publish/consume rates, consumer counts, lag detection
6. **System Health** - Health status for all services (ws-gateway, model-service, order-manager)
7. **WebSocket Connection** - Connection status, environment, duration, heartbeat, reconnection count
8. **Event History** - Chronological history of key system events (signals, orders, model training, subscriptions)

## Configuration

### Environment Variables

Configure Grafana via `.env` file:

```bash
# Grafana UI port (non-standard port starting from 4700)
GRAFANA_PORT=4700

# Grafana admin credentials
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_password

# PostgreSQL read-only user for Grafana
GRAFANA_POSTGRES_USER=grafana_monitor
GRAFANA_POSTGRES_PASSWORD=your_db_password

# Existing service configurations (should already be present)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

### Data Sources

Data sources are automatically provisioned on container startup via `grafana/provisioning/datasources/datasources.yml`:

- **PostgreSQL** - Default data source for trading data queries
- **RabbitMQ HTTP API** - Infinity data source for queue metrics
- **ws-gateway Health** - Service health endpoint
- **model-service Health** - Service health endpoint
- **order-manager Health** - Service health endpoint

### Dashboards

Dashboards are automatically provisioned from `grafana/dashboards/` directory:

- **Trading System Monitoring** - Main dashboard with all monitoring panels

## Access

### Web UI

Access Grafana web interface:

```
http://localhost:4700
```

**Default Credentials** (change in production):
- Username: `admin` (or value from `GRAFANA_ADMIN_USER`)
- Password: `admin` (or value from `GRAFANA_ADMIN_PASSWORD`)

### Health Endpoint

Check Grafana health:

```bash
curl http://localhost:4700/api/health
```

## Deployment

### Start Service

```bash
docker compose up -d grafana
```

### View Logs

```bash
docker compose logs -f grafana
```

### Stop Service

```bash
docker compose stop grafana
```

## Dashboard Configuration

### Auto-Refresh

- Dashboard-level: 60 seconds (configurable)
- Panel-level: 60 seconds (configurable per panel)

### Time Range Presets

Available time range presets:
- Last 1 hour
- Last 24 hours
- Last 7 days
- Custom range

### Panel Queries

All panels use optimized SQL queries with:
- Time range filters (`$__timeFilter`)
- Result limits (100-200 records)
- Indexed column usage for performance

## Data Source Queries

### Trading Signals

```sql
SELECT 
    signal_id, asset, side, price, confidence, timestamp, 
    strategy_id, model_version, is_warmup, market_data_snapshot
FROM trading_signals
WHERE $__timeFilter(timestamp)
ORDER BY timestamp DESC
LIMIT 100;
```

### Order Execution

```sql
SELECT 
    o.order_id, e.signal_id, e.asset, e.side,
    e.execution_price, e.execution_quantity, e.execution_fees,
    e.executed_at, o.status as closure_status
FROM execution_events e
LEFT JOIN orders o ON e.signal_id = o.signal_id
WHERE $__timeFilter(e.executed_at)
ORDER BY e.executed_at DESC
LIMIT 100;
```

### Model Quality Metrics

```sql
SELECT 
    COUNT(*) as total_orders_count,
    SUM(CASE WHEN (performance->>'realized_pnl')::DECIMAL > 0 THEN 1 ELSE 0 END) as successful_orders_count,
    ROUND(SUM(CASE WHEN (performance->>'realized_pnl')::DECIMAL > 0 THEN 1 ELSE 0 END)::DECIMAL / 
          NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    SUM((performance->>'realized_pnl')::DECIMAL) as total_pnl
FROM execution_events
WHERE executed_at >= NOW() - INTERVAL '7 days'
GROUP BY strategy_id;
```

## Troubleshooting

### Grafana Cannot Connect to PostgreSQL

1. Verify PostgreSQL user exists:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "\du grafana_monitor"
   ```

2. Check credentials in `.env` file

3. Test connection:
   ```bash
   docker compose exec grafana curl http://postgres:5432
   ```

### Dashboard Panels Show "No Data"

1. Verify data sources are configured: Configuration → Data Sources
2. Test data source connectivity
3. Check SQL query syntax
4. Verify tables have data:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT COUNT(*) FROM trading_signals;"
   ```

### RabbitMQ Queue Metrics Not Displaying

1. Verify RabbitMQ Management API is enabled:
   ```bash
   curl http://localhost:15672/api/overview
   ```

2. Check RabbitMQ credentials in `.env`

3. Verify Infinity data source plugin is installed:
   ```bash
   docker compose exec grafana ls /var/lib/grafana/plugins/
   ```

## Security

### Change Default Credentials

**IMPORTANT**: Change default admin credentials in production:

1. Update `.env` file:
   ```bash
   GRAFANA_ADMIN_USER=your_secure_username
   GRAFANA_ADMIN_PASSWORD=your_secure_password
   ```

2. Restart Grafana:
   ```bash
   docker compose restart grafana
   ```

### Network Access

- Grafana UI is accessible on port 4700
- Consider firewall rules to restrict access
- Use reverse proxy with HTTPS in production
- Consider VPN or SSH tunnel for remote access

### PostgreSQL Read-Only User

The `grafana_monitor` user has:
- SELECT permissions on required tables only
- No INSERT, UPDATE, DELETE, or DDL permissions
- Limited to `public` schema

## Performance

### Query Optimization

- Queries use indexed columns (`timestamp`, `executed_at`, `is_active`)
- Result sets limited to 100-200 records
- Time range filters reduce data scanned
- Aggregations computed at query time

### Dashboard Refresh

- Default refresh: 60 seconds
- Adjust based on data update frequency
- Longer intervals for heavy queries
- Shorter intervals for critical metrics

## Maintenance

### Backup Configuration

```bash
# Backup Grafana data volume
docker compose exec grafana tar czf /tmp/grafana-backup.tar.gz /var/lib/grafana
docker cp grafana:/tmp/grafana-backup.tar.gz ./grafana-backup.tar.gz
```

### Update Grafana Version

1. Update `grafana/Dockerfile` with new image version
2. Pull new image: `docker compose pull grafana`
3. Restart service: `docker compose up -d grafana`

## Testing

### Integration Tests

Run integration tests:

```bash
docker compose run --rm test-grafana pytest tests/integration/test_grafana*.py -v
```

Tests verify:
- Container startup and health endpoint
- Data source connectivity
- Dashboard loading and panel rendering

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [PostgreSQL Data Source](https://grafana.com/docs/grafana/latest/datasources/postgres/)
- [Infinity Data Source Plugin](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/)
- [RabbitMQ Management API](https://www.rabbitmq.com/management.html)

