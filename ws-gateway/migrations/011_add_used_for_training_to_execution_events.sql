-- Migration: Add used_for_training and training_id columns to execution_events table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Enable persistent buffer storage for training orchestrator by tracking
--          which execution events have already been used for training.
--
-- Changes:
--  - Add used_for_training BOOLEAN flag (default FALSE)
--  - Add training_id UUID column referencing model_versions(id)
--  - Add composite index on (used_for_training, strategy_id) for efficient queries

ALTER TABLE execution_events
ADD COLUMN IF NOT EXISTS used_for_training BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE execution_events
ADD COLUMN IF NOT EXISTS training_id UUID NULL;

-- Add foreign key constraint (check if it doesn't exist first)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_execution_events_training_model_version'
    ) THEN
        ALTER TABLE execution_events
        ADD CONSTRAINT fk_execution_events_training_model_version
        FOREIGN KEY (training_id) REFERENCES model_versions(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Index to efficiently query unused events per strategy
CREATE INDEX IF NOT EXISTS idx_execution_events_used_for_training_strategy
ON execution_events (used_for_training, strategy_id);

-- Rollback (reverse migration):
-- NOTE: Drop foreign key constraint before dropping column
-- ALTER TABLE execution_events
--     DROP CONSTRAINT IF EXISTS fk_execution_events_training_model_version;
-- ALTER TABLE execution_events
--     DROP COLUMN IF EXISTS training_id;
-- ALTER TABLE execution_events
--     DROP COLUMN IF EXISTS used_for_training;
-- DROP INDEX IF EXISTS idx_execution_events_used_for_training_strategy;


