-- Migration: Add snapshot_data JSONB column to position_snapshots table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store complete position state as JSONB for historical analysis and ML training

-- Add snapshot_data column as nullable first
ALTER TABLE position_snapshots ADD COLUMN IF NOT EXISTS snapshot_data JSONB;

-- Migrate existing data: construct snapshot_data from existing columns
-- This preserves historical data if any exists
UPDATE position_snapshots
SET snapshot_data = jsonb_build_object(
    'id', id::text,
    'position_id', position_id::text,
    'asset', asset,
    'mode', mode,
    'size', size::text,
    'average_entry_price', CASE WHEN average_entry_price IS NULL THEN NULL ELSE average_entry_price::text END,
    'current_price', NULL,
    'unrealized_pnl', CASE WHEN unrealized_pnl IS NULL THEN NULL ELSE unrealized_pnl::text END,
    'realized_pnl', CASE WHEN realized_pnl IS NULL THEN NULL ELSE realized_pnl::text END,
    'long_size', CASE WHEN long_size IS NULL THEN NULL ELSE long_size::text END,
    'short_size', CASE WHEN short_size IS NULL THEN NULL ELSE short_size::text END,
    'unrealized_pnl_pct', NULL,
    'time_held_minutes', NULL,
    'position_size_norm', NULL,
    'last_updated', snapshot_timestamp::text,
    'closed_at', NULL
)
WHERE snapshot_data IS NULL;

-- Make column NOT NULL after data migration
ALTER TABLE position_snapshots ALTER COLUMN snapshot_data SET NOT NULL;

-- Rollback (reverse migration):
-- ALTER TABLE position_snapshots ALTER COLUMN snapshot_data DROP NOT NULL;
-- ALTER TABLE position_snapshots DROP COLUMN IF EXISTS snapshot_data;

