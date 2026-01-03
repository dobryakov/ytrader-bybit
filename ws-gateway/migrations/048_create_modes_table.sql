-- Migration: Create modes table for Dashboard API
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Stores mode configurations for dataset building with duration-based periods

CREATE TABLE IF NOT EXISTS modes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    asset VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    target_registry_version VARCHAR(50) NOT NULL,
    train_duration_days INTEGER NOT NULL,
    validation_duration_days INTEGER NOT NULL,
    test_duration_days INTEGER NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT chk_durations_positive CHECK (
        train_duration_days > 0 AND
        validation_duration_days > 0 AND
        test_duration_days > 0
    )
);

CREATE INDEX IF NOT EXISTS idx_modes_asset ON modes(asset);
CREATE INDEX IF NOT EXISTS idx_modes_strategy ON modes(strategy_id);
CREATE INDEX IF NOT EXISTS idx_modes_active ON modes(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_modes_created_at ON modes(created_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_modes_created_at;
-- DROP INDEX IF EXISTS idx_modes_active;
-- DROP INDEX IF EXISTS idx_modes_strategy;
-- DROP INDEX IF EXISTS idx_modes_asset;
-- DROP TABLE IF EXISTS modes;

