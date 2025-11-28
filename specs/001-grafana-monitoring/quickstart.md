# Quickstart: Grafana Monitoring Dashboard

**Feature**: Grafana Monitoring Dashboard  
**Date**: 2025-01-27

## Overview

This guide provides step-by-step instructions to set up and run the Grafana monitoring dashboard service for the YTrader trading system. Grafana provides visual monitoring of trading signals, order execution, model state, queue health, system health, and event history.

## Prerequisites

- Docker and Docker Compose V2 installed
- Existing YTrader services running (ws-gateway, model-service, order-manager, postgres, rabbitmq)
- PostgreSQL database with monitoring tables populated
- RabbitMQ with management plugin enabled

## Quick Setup

### 1. Clone and Navigate

```bash
cd /home/ubuntu/ytrader
git checkout 001-grafana-monitoring
```

### 2. Create Read-Only PostgreSQL User

Create a read-only PostgreSQL user for Grafana monitoring:

```bash
# Connect to PostgreSQL container
docker compose exec postgres psql -U ytrader -d ytrader

# In PostgreSQL prompt, create read-only user:
CREATE USER grafana_monitor WITH PASSWORD 'your_grafana_db_password';
GRANT CONNECT ON DATABASE ytrader TO grafana_monitor;
GRANT USAGE ON SCHEMA public TO grafana_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_monitor;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO grafana_monitor;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_monitor;
\q
```

**Note**: Replace `your_grafana_db_password` with a secure password. This will be used in `.env` file.

### 3. Configure Environment

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` and add the following Grafana configuration variables:

```bash
# =============================================================================
# Grafana Monitoring Configuration
# =============================================================================
# Grafana UI port (non-standard port starting from 4700)
GRAFANA_PORT=4700

# Grafana admin credentials (default: admin/admin)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# PostgreSQL read-only user for Grafana
GRAFANA_POSTGRES_USER=grafana_monitor
GRAFANA_POSTGRES_PASSWORD=your_grafana_db_password

# Existing service configurations (should already be present)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

**Security Note**: Change default admin credentials (`GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD`) in production.

### 4. Create Grafana Directory Structure

Create the Grafana configuration directories:

```bash
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
```

### 5. Configure Data Sources

Create data source provisioning file:

```bash
cat > grafana/provisioning/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: ytrader
    user: grafana_monitor
    secureJsonData:
      password: ${GRAFANA_POSTGRES_PASSWORD}
    jsonData:
      sslmode: disable
      maxOpenConns: 100
      maxIdleConns: 100
      connMaxLifetime: 14400
      postgresVersion: 1500
EOF
```

**Note**: Environment variable substitution in datasources.yml requires special handling. You may need to use a configuration script or set password directly (ensure `.env` is secure).

### 6. Configure Dashboard Provisioning

Create dashboard provisioning file:

```bash
cat > grafana/provisioning/dashboards/dashboards.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'Trading System Monitoring'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
```

### 7. Add Grafana Service to docker-compose.yml

Add Grafana service to `docker-compose.yml`:

```yaml
  grafana:
    image: grafana/grafana:10.4.0
    container_name: grafana
    ports:
      - "${GRAFANA_PORT:-4700}:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=
      - GF_PATHS_PROVISIONING=/etc/grafana/provisioning
      - GF_PATHS_DATA=/var/lib/grafana
      - POSTGRES_HOST=${POSTGRES_HOST:-postgres}
      - POSTGRES_PORT=${POSTGRES_PORT:-5432}
      - POSTGRES_DB=${POSTGRES_DB:-ytrader}
      - GRAFANA_POSTGRES_USER=${GRAFANA_POSTGRES_USER:-grafana_monitor}
      - GRAFANA_POSTGRES_PASSWORD=${GRAFANA_POSTGRES_PASSWORD}
      - RABBITMQ_USER=${RABBITMQ_USER:-guest}
      - RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD:-guest}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    networks:
      - ytrader-network
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Add Grafana volume to volumes section:

```yaml
volumes:
  postgres_data:
  rabbitmq_data:
  model_storage:
  grafana_data:
```

### 8. Start Grafana Service

Start Grafana using Docker Compose:

```bash
docker compose up -d grafana
```

Wait for Grafana to be ready:

```bash
docker compose logs -f grafana
# Press Ctrl+C when Grafana is ready (look for "HTTP Server Listen" message)
```

### 9. Verify Grafana is Running

Check Grafana health:

```bash
curl http://localhost:4700/api/health
```

Expected response:
```json
{
  "commit": "abc123",
  "database": "ok",
  "version": "10.4.0"
}
```

### 10. Access Grafana UI

Open Grafana in your browser:

```
http://localhost:4700
```

**Login Credentials**:
- Username: `admin` (or value from `GRAFANA_ADMIN_USER`)
- Password: `admin` (or value from `GRAFANA_ADMIN_PASSWORD`)

Grafana will prompt you to change the password on first login (optional).

### 11. Verify Data Sources

1. Navigate to **Configuration** → **Data Sources**
2. Verify the following data sources are configured:
   - **PostgreSQL** - Connected and tested
   - **RabbitMQ HTTP API** - (if configured)
   - **Service Health Endpoints** - (if configured)

### 12. Import or Create Dashboards

#### Option A: Create Dashboards Manually

1. Navigate to **Dashboards** → **New** → **New Dashboard**
2. Add panels for:
   - Recent Trading Signals (PostgreSQL query)
   - Recent Orders (PostgreSQL query)
   - Model State (PostgreSQL query)
   - Queue Metrics (RabbitMQ Management API)
   - System Health (Service REST APIs)
   - WebSocket Connection State (ws-gateway health endpoint)
   - Event History (PostgreSQL aggregated query)

See `data-model.md` for SQL query examples.

#### Option B: Import Dashboard JSON

If dashboard JSON files are available:

1. Navigate to **Dashboards** → **Import**
2. Upload dashboard JSON file
3. Select data sources
4. Click **Import**

### 13. Configure Dashboard Queries

#### Trading Signals Query

```sql
SELECT 
    signal_id,
    asset,
    side,
    execution_price as price,
    executed_at as time,
    strategy_id,
    (performance->>'confidence')::decimal as confidence
FROM execution_events
WHERE executed_at >= NOW() - INTERVAL '24 hours'
ORDER BY executed_at DESC
LIMIT 100;
```

#### Recent Orders Query

```sql
SELECT 
    e.id,
    o.order_id,
    e.signal_id,
    e.asset,
    e.side,
    e.execution_price,
    e.execution_quantity,
    e.execution_fees,
    e.executed_at as time,
    o.status as closure_status
FROM execution_events e
LEFT JOIN orders o ON e.signal_id = o.signal_id
WHERE e.executed_at >= NOW() - INTERVAL '24 hours'
ORDER BY e.executed_at DESC
LIMIT 100;
```

#### Queue Metrics Query (HTTP Data Source)

Query RabbitMQ Management API:
- URL: `http://rabbitmq:15672/api/queues`
- Method: GET
- Authentication: Basic Auth
- Username: `${RABBITMQ_USER}`
- Password: `${RABBITMQ_PASSWORD}`

Parse JSON response to extract queue metrics.

#### Health Status Query (HTTP Data Source)

Query service health endpoints:
- ws-gateway: `http://ws-gateway:4400/health`
- model-service: `http://model-service:4500/health`
- order-manager: `http://order-manager:4600/health`

See `contracts/api-endpoints.md` for detailed endpoint specifications.

---

## Configuration Examples

### PostgreSQL Data Source Configuration

**Via Grafana UI**:
1. Go to **Configuration** → **Data Sources** → **Add data source**
2. Select **PostgreSQL**
3. Configure:
   - **Host**: `postgres:5432`
   - **Database**: `ytrader`
   - **User**: `grafana_monitor`
   - **Password**: (from `.env` file)
   - **SSL Mode**: `disable`
4. Click **Save & Test**

**Via Provisioning File**:
See `contracts/datasources.yml` for YAML configuration.

### HTTP Data Source Configuration

**RabbitMQ Management API**:
1. Go to **Configuration** → **Data Sources** → **Add data source**
2. Select **JSON API** or **Prometheus** (used as proxy for HTTP)
3. Configure:
   - **URL**: `http://rabbitmq:15672`
   - **Basic Auth**: Enabled
   - **User**: `${RABBITMQ_USER}`
   - **Password**: `${RABBITMQ_PASSWORD}`
4. Click **Save & Test**

**Service Health Endpoints**:
1. Create HTTP data source for each service
2. Configure URLs:
   - `http://ws-gateway:4400`
   - `http://model-service:4500`
   - `http://order-manager:4600`

---

## Dashboard Panel Examples

### Trading Signals Panel

**Panel Type**: Table

**Query**:
```sql
SELECT 
    signal_id,
    asset,
    side,
    execution_price as "Price",
    executed_at as "Time",
    strategy_id as "Strategy",
    (performance->>'confidence')::decimal as "Confidence"
FROM execution_events
WHERE $__timeFilter(executed_at)
ORDER BY executed_at DESC
LIMIT 100;
```

**Visualization**: Table with columns: Signal ID, Asset, Side, Price, Time, Strategy, Confidence

### Queue Lag Detection Panel

**Panel Type**: Stat

**Query** (HTTP Data Source):
- Endpoint: `GET http://rabbitmq:15672/api/queues`
- Parse JSON: Extract `messages` field for each queue
- Calculate: Count queues with `messages > 1000`

**Visualization**: Single stat showing number of queues with lag

### System Health Panel

**Panel Type**: Table

**Queries** (Multiple HTTP Data Sources):
- Query each service health endpoint
- Extract `status` field
- Combine into table: Service Name, Status, Components

**Visualization**: Table with color coding (green for healthy, red for unhealthy)

---

## Troubleshooting

### Grafana Cannot Connect to PostgreSQL

**Symptoms**: Data source test fails

**Solutions**:
1. Verify PostgreSQL user exists and has SELECT permissions:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT 1 FROM information_schema.tables LIMIT 1;" -U grafana_monitor
   ```

2. Check PostgreSQL credentials in `.env` file

3. Verify network connectivity:
   ```bash
   docker compose exec grafana ping -c 3 postgres
   ```

### Grafana Cannot Connect to RabbitMQ Management API

**Symptoms**: Queue metrics not displaying

**Solutions**:
1. Verify RabbitMQ Management API is enabled:
   ```bash
   docker compose exec rabbitmq rabbitmq-diagnostics ping
   curl http://localhost:15672/api/overview
   ```

2. Check RabbitMQ credentials in `.env` file

3. Verify Management API port is accessible:
   ```bash
   docker compose exec grafana curl http://rabbitmq:15672/api/overview
   ```

### Dashboards Not Loading

**Symptoms**: Dashboard panels show "No data" or errors

**Solutions**:
1. Verify data sources are configured and tested
2. Check SQL queries for syntax errors
3. Verify tables exist and have data:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT COUNT(*) FROM execution_events;"
   ```

4. Check time range filters (use appropriate time ranges)

### Grafana Container Fails to Start

**Symptoms**: Container exits immediately

**Solutions**:
1. Check container logs:
   ```bash
   docker compose logs grafana
   ```

2. Verify volume mounts are correct:
   ```bash
   ls -la grafana/provisioning/datasources/
   ls -la grafana/provisioning/dashboards/
   ```

3. Check environment variables:
   ```bash
   docker compose config | grep -A 20 grafana
   ```

---

## Security Considerations

### Change Default Credentials

**Grafana Admin**:
- Change default `admin/admin` credentials in production
- Use strong passwords
- Store credentials securely (`.env` file, not in git)

### PostgreSQL Read-Only User

- Use dedicated read-only user for Grafana
- Limit permissions to SELECT only
- Use strong password

### Network Access

- Grafana UI is accessible on port 4700
- Consider firewall rules to restrict access
- Use reverse proxy with HTTPS in production
- Consider VPN or SSH tunnel for remote access

### Data Source Credentials

- Store all credentials in `.env` file (not committed to git)
- Use secure password storage mechanisms
- Rotate credentials periodically

---

## Performance Optimization

### Query Optimization

- Use indexed columns for filtering (`executed_at`, `created_at`)
- Limit result sets (LIMIT 100-200)
- Use time-range filters
- Aggregate metrics at query time

### Dashboard Refresh Intervals

- Default: 60 seconds
- Adjust based on data update frequency
- Use longer intervals for heavy queries
- Use shorter intervals for critical metrics

### Connection Pooling

- PostgreSQL: Max 100 connections, idle timeout 4 hours
- HTTP data sources: Connection reuse enabled

---

## Maintenance

### Backup Grafana Configuration

```bash
# Backup Grafana data volume
docker compose exec grafana tar czf /tmp/grafana-backup.tar.gz /var/lib/grafana

# Copy from container
docker cp grafana:/tmp/grafana-backup.tar.gz ./grafana-backup.tar.gz
```

### Update Grafana Version

```bash
# Update docker-compose.yml with new image version
# grafana/grafana:10.4.0 -> grafana/grafana:10.5.0

# Pull new image
docker compose pull grafana

# Restart service
docker compose up -d grafana
```

### View Grafana Logs

```bash
# Follow logs
docker compose logs -f grafana

# View last 100 lines
docker compose logs --tail 100 grafana
```

---

## Next Steps

1. **Customize Dashboards**: Create custom dashboards for your specific monitoring needs
2. **Set Up Alerts**: Configure Grafana alerts for critical metrics (queue lag, service health)
3. **Add More Metrics**: Extend dashboards with additional metrics as needed
4. **Performance Tuning**: Optimize queries and refresh intervals based on usage

---

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [PostgreSQL Data Source](https://grafana.com/docs/grafana/latest/datasources/postgres/)
- [HTTP Data Source](https://grafana.com/docs/grafana/latest/datasources/http-json/)
- [RabbitMQ Management API](https://www.rabbitmq.com/management.html)

---

## Support

For issues or questions:
1. Check logs: `docker compose logs grafana`
2. Verify data sources: Configuration → Data Sources
3. Review dashboard queries: Edit panel → Query inspector
4. Check service health: `curl http://localhost:4700/api/health`

