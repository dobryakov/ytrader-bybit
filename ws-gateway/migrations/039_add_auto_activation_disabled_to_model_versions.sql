-- Migration: Add auto_activation_disabled field to model_versions table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Allow manual blocking of automatic model activation by mode_transition service

ALTER TABLE model_versions 
ADD COLUMN IF NOT EXISTS auto_activation_disabled BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN model_versions.auto_activation_disabled IS 
'If true, prevents automatic activation of this model by mode_transition service. Model can still be manually activated via API.';

-- Rollback (reverse migration):
-- ALTER TABLE model_versions DROP COLUMN IF EXISTS auto_activation_disabled;

