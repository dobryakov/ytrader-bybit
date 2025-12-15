-- Migration: Modify target_registry_versions table for file-based storage
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Change from storing config JSONB in DB to storing file_path, making YAML files the source of truth
-- Note: This migration removes config JSONB column and makes file_path NOT NULL

-- Step 1: Ensure file_path is set for existing records
-- Backfill file_path from version if not set
UPDATE target_registry_versions
SET file_path = '/app/config/versions/target_registry_v' || version || '.yaml'
WHERE file_path IS NULL;

-- Step 2: Make file_path NOT NULL (after backfill)
ALTER TABLE target_registry_versions
ALTER COLUMN file_path SET NOT NULL;

-- Step 3: Add index on file_path for quick lookups
CREATE INDEX IF NOT EXISTS idx_target_registry_file_path ON target_registry_versions(file_path);

-- Step 4: Remove CHECK constraints that depend on config JSONB
ALTER TABLE target_registry_versions
DROP CONSTRAINT IF EXISTS chk_target_config_type;

ALTER TABLE target_registry_versions
DROP CONSTRAINT IF EXISTS chk_target_config_horizon;

-- Step 5: Remove config JSONB column
-- Note: This will fail if there are rows with config but no file_path (should not happen after backfill)
ALTER TABLE target_registry_versions
DROP COLUMN IF EXISTS config;

-- Rollback (reverse migration):
-- Step 1: Add config JSONB column back (nullable first)
-- ALTER TABLE target_registry_versions
-- ADD COLUMN IF NOT EXISTS config JSONB;
--
-- Step 2: Backfill config from file_path if files exist (requires application logic)
-- Note: This would require reading YAML files and converting to JSONB
-- This is a complex operation that should be done via application migration script
--
-- Step 3: Make config NOT NULL (after backfill)
-- ALTER TABLE target_registry_versions
-- ALTER COLUMN config SET NOT NULL;
--
-- Step 4: Add CHECK constraints back
-- ALTER TABLE target_registry_versions
-- ADD CONSTRAINT chk_target_config_type CHECK (
--     config ? 'type' AND 
--     config->>'type' IN ('regression', 'classification', 'risk_adjusted')
-- );
-- ALTER TABLE target_registry_versions
-- ADD CONSTRAINT chk_target_config_horizon CHECK (
--     config ? 'horizon' AND 
--     (config->>'horizon')::integer > 0
-- );
--
-- Step 5: Drop file_path column
-- ALTER TABLE target_registry_versions
-- DROP COLUMN IF EXISTS file_path;
--
-- Step 6: Drop index on file_path
-- DROP INDEX IF EXISTS idx_target_registry_file_path;

