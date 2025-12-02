-- Migration: Create read-only PostgreSQL user for Grafana monitoring
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Create grafana_monitor user with read-only permissions for Grafana dashboard queries

-- Create read-only user for Grafana (if not exists)
-- Note: Password is substituted via envsubst from GRAFANA_POSTGRES_PASSWORD environment variable
-- Usage: 
--   export GRAFANA_POSTGRES_PASSWORD=your_password
--   envsubst < 011_create_grafana_monitor_user.sql | psql -d ytrader
-- Or use the helper script: ./run_migration.sh 011_create_grafana_monitor_user.sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'grafana_monitor') THEN
        CREATE USER grafana_monitor WITH PASSWORD '${GRAFANA_POSTGRES_PASSWORD}';
    END IF;
END
$$;

-- Update password if user already exists (idempotent migration)
-- This allows re-running the migration to update the password
ALTER USER grafana_monitor WITH PASSWORD '${GRAFANA_POSTGRES_PASSWORD}';

-- Grant connection to database
GRANT CONNECT ON DATABASE ytrader TO grafana_monitor;

-- Grant usage on public schema
GRANT USAGE ON SCHEMA public TO grafana_monitor;

-- Grant SELECT on all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_monitor;

-- Grant SELECT on all sequences (for future use)
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO grafana_monitor;

-- Set default privileges for future tables (so new tables are automatically accessible)
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_monitor;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO grafana_monitor;

-- Explicitly grant SELECT on positions and position_snapshots tables (if they exist)
-- This ensures access even if tables were created before default privileges were set
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'positions') THEN
        GRANT SELECT ON TABLE public.positions TO grafana_monitor;
    END IF;
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'position_snapshots') THEN
        GRANT SELECT ON TABLE public.position_snapshots TO grafana_monitor;
    END IF;
END
$$;

-- Rollback (reverse migration):
-- REVOKE SELECT ON TABLE public.positions FROM grafana_monitor;
-- REVOKE SELECT ON TABLE public.position_snapshots FROM grafana_monitor;
-- REVOKE SELECT ON ALL SEQUENCES IN SCHEMA public FROM grafana_monitor;
-- REVOKE SELECT ON ALL TABLES IN SCHEMA public FROM grafana_monitor;
-- REVOKE USAGE ON SCHEMA public FROM grafana_monitor;
-- REVOKE CONNECT ON DATABASE ytrader FROM grafana_monitor;
-- DROP USER IF EXISTS grafana_monitor;

