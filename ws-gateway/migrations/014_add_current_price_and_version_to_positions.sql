-- Migration: 014_add_current_price_and_version_to_positions.sql
-- Purpose: add current_price and version fields to positions table for Position Manager
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)

BEGIN;

-- Forward migration: add columns and indexes
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS current_price DECIMAL(20, 8) NULL,
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_positions_current_price ON positions(current_price);
CREATE INDEX IF NOT EXISTS idx_positions_version ON positions(version);

-- Ensure version is set for existing rows
UPDATE positions SET version = 1 WHERE version IS NULL;

COMMIT;

-- Rollback section
-- Warning: Dropping columns will remove data. Use with caution in non-production environments.
-- To rollback this migration, run:
--
-- BEGIN;
-- DROP INDEX IF EXISTS idx_positions_current_price;
-- DROP INDEX IF EXISTS idx_positions_version;
-- ALTER TABLE positions DROP COLUMN IF EXISTS current_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS version;
-- COMMIT;


