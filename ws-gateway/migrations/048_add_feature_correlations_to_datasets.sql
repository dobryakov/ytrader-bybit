-- Add feature_correlations column to datasets table
-- This column will store the correlation of each feature with the target as a JSONB object

ALTER TABLE datasets ADD COLUMN IF NOT EXISTS feature_correlations JSONB;

COMMENT ON COLUMN datasets.feature_correlations IS 'JSON object storing the correlation of each feature with the target';
