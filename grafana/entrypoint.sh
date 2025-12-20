#!/bin/sh
# Grafana entrypoint script to substitute environment variables in provisioning files
# This script runs before Grafana starts to replace ${VAR} placeholders with actual values
# It processes files as root, then switches to grafana user (UID 472) before starting Grafana

set -e

# Check current user ID
CURRENT_UID=$(id -u)

# Export required environment variables for envsubst
export POSTGRES_DB="${POSTGRES_DB:-ytrader}"
export GRAFANA_POSTGRES_USER="${GRAFANA_POSTGRES_USER:-grafana_monitor}"
export GRAFANA_POSTGRES_PASSWORD="${GRAFANA_POSTGRES_PASSWORD}"
export RABBITMQ_USER="${RABBITMQ_USER:-guest}"
export RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-guest}"
export POSITION_MANAGER_API_KEY="${POSITION_MANAGER_API_KEY}"
export MODEL_SERVICE_API_KEY="${MODEL_SERVICE_API_KEY}"

# Process datasources.yml if it exists
# We need root privileges to write to /etc/grafana/provisioning
if [ -f /etc/grafana/provisioning/datasources/datasources.yml ]; then
    if [ "$CURRENT_UID" = "0" ]; then
        # Running as root - process file and set ownership
        envsubst '$POSTGRES_DB $GRAFANA_POSTGRES_USER $GRAFANA_POSTGRES_PASSWORD $RABBITMQ_USER $RABBITMQ_PASSWORD $POSITION_MANAGER_API_KEY $MODEL_SERVICE_API_KEY' < /etc/grafana/provisioning/datasources/datasources.yml > /tmp/datasources.yml.tmp
        mv /tmp/datasources.yml.tmp /etc/grafana/provisioning/datasources/datasources.yml
        chown 472:472 /etc/grafana/provisioning/datasources/datasources.yml 2>/dev/null || true
        chmod 644 /etc/grafana/provisioning/datasources/datasources.yml 2>/dev/null || true
    else
        # Running as non-root - try with sudo if available
        if command -v sudo >/dev/null 2>&1; then
            envsubst '$POSTGRES_DB $GRAFANA_POSTGRES_USER $GRAFANA_POSTGRES_PASSWORD $RABBITMQ_USER $RABBITMQ_PASSWORD $POSITION_MANAGER_API_KEY $MODEL_SERVICE_API_KEY' < /etc/grafana/provisioning/datasources/datasources.yml | sudo tee /etc/grafana/provisioning/datasources/datasources.yml > /dev/null
            sudo chown 472:472 /etc/grafana/provisioning/datasources/datasources.yml 2>/dev/null || true
            sudo chmod 644 /etc/grafana/provisioning/datasources/datasources.yml 2>/dev/null || true
        fi
    fi
fi

# Set Grafana paths
export GF_PATHS_HOME=/usr/share/grafana
export GF_PATHS_DATA=/var/lib/grafana
export GF_PATHS_LOGS=/var/log/grafana
export GF_PATHS_PLUGINS=/var/lib/grafana/plugins
export GF_PATHS_PROVISIONING=/etc/grafana/provisioning

# Always switch to grafana user (UID 472) before starting Grafana
# This ensures Grafana never runs as root
if [ "$CURRENT_UID" = "0" ]; then
    # Running as root - switch to grafana user using su-exec (preferred for Alpine) or su
    if command -v su-exec >/dev/null 2>&1; then
        # su-exec is the Alpine Linux equivalent of gosu
        exec su-exec grafana /run.sh "$@"
    else
        # Fallback to su (should work but less ideal)
        # Note: su-exec should be available since it's installed in Dockerfile
        exec su -s /bin/sh grafana -c "exec /run.sh \"\$@\"" sh "$@"
    fi
else
    # Already running as non-root (should be grafana user)
    # Execute directly - /run.sh will verify user if needed
    exec /run.sh "$@"
fi

