-- Migration: Make target_config nullable in datasets table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Allow NULL values in target_config since we now use target_registry_version

-- Make target_config nullable
ALTER TABLE datasets ALTER COLUMN target_config DROP NOT NULL;

-- Rollback (reverse migration):
-- ALTER TABLE datasets ALTER COLUMN target_config SET NOT NULL;

