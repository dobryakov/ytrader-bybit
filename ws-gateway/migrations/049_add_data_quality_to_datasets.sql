-- Add data_quality column to datasets table
-- This column will store information about problematic periods that were excluded from the dataset
-- Reversible: Yes (see rollback section at bottom)

BEGIN;

ALTER TABLE datasets ADD COLUMN IF NOT EXISTS data_quality JSONB;

COMMENT ON COLUMN datasets.data_quality IS 'JSON object storing data quality information including excluded problematic periods (periods with identical OHLC values)';

COMMIT;

-- Rollback (reverse migration):
-- BEGIN;
-- ALTER TABLE datasets DROP COLUMN IF EXISTS data_quality;
-- COMMIT;

