-- Migration: 042_fix_closed_positions_realized_pnl.sql
-- Purpose: Fix realized_pnl in closed_positions table - use unrealized_pnl_at_close instead of cumulative value
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)
--
-- Problem: realized_pnl was incorrectly storing cumulative realized PnL from Bybit (cumRealisedPnl)
-- which is cumulative across all positions for the asset, not specific to a single position.
--
-- Solution: For a fully closed position, realized_pnl should equal unrealized_pnl_at_close
-- because all unrealized PnL becomes realized when position is closed.

BEGIN;

-- Update existing closed_positions records:
-- Set realized_pnl = unrealized_pnl_at_close for all records
-- This corrects the logic where cumulative PnL was incorrectly stored.
-- For a fully closed position, all unrealized PnL becomes realized PnL.
UPDATE closed_positions
SET realized_pnl = unrealized_pnl_at_close
WHERE unrealized_pnl_at_close IS NOT NULL;

-- Log the number of updated records
DO $$
DECLARE
    updated_count INTEGER;
BEGIN
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RAISE NOTICE 'Updated % closed_positions records: set realized_pnl = unrealized_pnl_at_close', updated_count;
END $$;

COMMIT;

-- Rollback section
-- WARNING: This rollback will restore the old (incorrect) values, which may not be recoverable
-- if the original cumulative values were not backed up.
-- To rollback this migration, you would need to restore from a backup.
--
-- Note: Rollback is not provided as the original cumulative values are not stored separately.
-- If rollback is needed, restore from database backup taken before this migration.

