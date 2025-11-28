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

# Process datasources.yml if it exists
# We're running as root, so we can write directly
# Use UID 472 (grafana user) instead of username (user may not exist yet)
if [ -f /etc/grafana/provisioning/datasources/datasources.yml ]; then
    envsubst < /etc/grafana/provisioning/datasources/datasources.yml > /tmp/datasources.yml.tmp
    mv /tmp/datasources.yml.tmp /etc/grafana/provisioning/datasources/datasources.yml
    chown 472:472 /etc/grafana/provisioning/datasources/datasources.yml 2>/dev/null || true
fi

# Switch to grafana user before starting Grafana
# The original /run.sh will handle user switching, but we do it explicitly here
export GF_PATHS_HOME=/usr/share/grafana
export GF_PATHS_DATA=/var/lib/grafana
export GF_PATHS_LOGS=/var/log/grafana
export GF_PATHS_PLUGINS=/var/lib/grafana/plugins
export GF_PATHS_PROVISIONING=/etc/grafana/provisioning

# Execute the original Grafana entrypoint (it will switch to grafana user)
exec /run.sh "$@"

