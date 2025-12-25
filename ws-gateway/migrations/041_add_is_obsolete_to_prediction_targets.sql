-- Migration: Add is_obsolete flag to prediction_targets table
-- Reversible: Yes
-- Purpose: Mark very old targets as obsolete to stop processing attempts

ALTER TABLE prediction_targets
ADD COLUMN IF NOT EXISTS is_obsolete BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for filtering out obsolete targets from pending computations
CREATE INDEX IF NOT EXISTS idx_prediction_targets_not_obsolete 
ON prediction_targets(target_timestamp) 
WHERE is_obsolete = FALSE AND actual_values_computed_at IS NULL;

-- Update partial index for pending computations to exclude obsolete
DROP INDEX IF EXISTS idx_prediction_targets_pending_computation;
CREATE INDEX IF NOT EXISTS idx_prediction_targets_pending_computation 
ON prediction_targets(target_timestamp) 
WHERE actual_values_computed_at IS NULL AND is_obsolete = FALSE;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_prediction_targets_pending_computation;
-- DROP INDEX IF EXISTS idx_prediction_targets_not_obsolete;
-- ALTER TABLE prediction_targets DROP COLUMN IF EXISTS is_obsolete;

