#!/bin/sh
# Script to run migration 011 with environment variable substitution
# Usage: ./run_migration.sh [migration_file]

set -e

MIGRATION_FILE="${1:-011_create_grafana_monitor_user.sql}"

# Check if envsubst is available
if ! command -v envsubst >/dev/null 2>&1; then
    echo "Error: envsubst is required but not installed."
    echo "Install with: apt-get install gettext-base (Debian/Ubuntu) or apk add gettext (Alpine)"
    exit 1
fi

# Check if migration file exists
if [ ! -f "$MIGRATION_FILE" ]; then
    echo "Error: Migration file '$MIGRATION_FILE' not found"
    exit 1
fi

# Export required environment variables with defaults
# If GRAFANA_POSTGRES_PASSWORD is not set, use a default value
# WARNING: Using default password is insecure - set GRAFANA_POSTGRES_PASSWORD in .env
if [ -z "$GRAFANA_POSTGRES_PASSWORD" ]; then
    echo "Warning: GRAFANA_POSTGRES_PASSWORD not set, using default (insecure)"
    export GRAFANA_POSTGRES_PASSWORD="CHANGE_ME_IN_ENV"
fi

# Substitute variables and execute via psql
# Note: This assumes psql is configured via PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
echo "Running migration: $MIGRATION_FILE"
envsubst < "$MIGRATION_FILE" | psql -v ON_ERROR_STOP=1

echo "Migration completed successfully"

