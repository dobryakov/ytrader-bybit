-- Migration: Add target_registry_version to datasets table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Replace target_config JSONB with target_registry_version reference

-- Add new column
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS target_registry_version VARCHAR(50);

-- Create index
CREATE INDEX IF NOT EXISTS idx_datasets_target_registry_version ON datasets(target_registry_version);

-- Add foreign key constraint (optional, for data integrity)
-- ALTER TABLE datasets ADD CONSTRAINT fk_datasets_target_registry_version 
--     FOREIGN KEY (target_registry_version) REFERENCES target_registry_versions(version);

-- Note: target_config column is kept for now to allow gradual migration
-- It will be removed in a future migration after all datasets are migrated

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_datasets_target_registry_version;
-- ALTER TABLE datasets DROP COLUMN IF EXISTS target_registry_version;

