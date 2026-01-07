-- Migration: 051_add_opened_at_to_positions.sql
-- Purpose: Add opened_at field to track when position was last opened (size changed from 0 to non-zero)
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)

BEGIN;

-- Forward migration: add opened_at timestamp to track last position opening
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS opened_at TIMESTAMP NULL;

-- Create index for efficient queries by opened_at
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at DESC);

-- For existing positions that are currently open (closed_at IS NULL AND size != 0),
-- set opened_at to last_updated as best approximation
UPDATE positions
SET opened_at = last_updated
WHERE closed_at IS NULL 
  AND size != 0 
  AND opened_at IS NULL;

-- For existing positions that are closed but were reopened (closed_at IS NOT NULL),
-- we cannot determine exact opened_at, so leave it NULL
-- (it will be set correctly on next reopen)

COMMIT;

-- Rollback section
-- WARNING: Dropping columns will remove data. Use with caution.
-- To rollback this migration, run:
--
-- BEGIN;
-- DROP INDEX IF EXISTS idx_positions_opened_at;
-- ALTER TABLE positions DROP COLUMN IF EXISTS opened_at;
-- COMMIT;

