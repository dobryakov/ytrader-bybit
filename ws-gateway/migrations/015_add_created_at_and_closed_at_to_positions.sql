-- Migration: 015_add_created_at_and_closed_at_to_positions.sql
-- Purpose: align positions table schema with Position Manager data model
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)

BEGIN;

-- Forward migration: add audit/closure timestamps if missing
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS closed_at TIMESTAMP NULL;

COMMIT;

-- Rollback section
-- WARNING: Dropping columns will remove data. Use with caution.
-- To rollback this migration, run:
--
-- BEGIN;
-- ALTER TABLE positions DROP COLUMN IF EXISTS closed_at;
-- ALTER TABLE positions DROP COLUMN IF EXISTS created_at;
-- COMMIT;



