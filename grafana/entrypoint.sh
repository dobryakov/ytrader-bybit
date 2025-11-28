#!/bin/sh
# Grafana entrypoint script to substitute environment variables in provisioning files
# This script runs before Grafana starts to replace ${VAR} placeholders with actual values

set -e

# Substitute environment variables in datasources.yml
# Export required environment variables for envsubst
export POSTGRES_DB="${POSTGRES_DB:-ytrader}"
export GRAFANA_POSTGRES_USER="${GRAFANA_POSTGRES_USER:-grafana_monitor}"
export GRAFANA_POSTGRES_PASSWORD="${GRAFANA_POSTGRES_PASSWORD}"
export RABBITMQ_USER="${RABBITMQ_USER:-guest}"
export RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-guest}"

if [ -f /etc/grafana/provisioning/datasources/datasources.yml ]; then
    envsubst < /etc/grafana/provisioning/datasources/datasources.yml > /tmp/datasources.yml
    mv /tmp/datasources.yml /etc/grafana/provisioning/datasources/datasources.yml
fi

# Execute the original Grafana entrypoint
exec /run.sh "$@"

