-- Migration: Create datasets table for Feature Service
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Stores dataset metadata for model training datasets built by Feature Service

CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'building',
    split_strategy VARCHAR(50) NOT NULL,
    train_period_start TIMESTAMP,
    train_period_end TIMESTAMP,
    validation_period_start TIMESTAMP,
    validation_period_end TIMESTAMP,
    test_period_start TIMESTAMP,
    test_period_end TIMESTAMP,
    walk_forward_config JSONB,
    target_config JSONB NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    train_records INTEGER DEFAULT 0,
    validation_records INTEGER DEFAULT 0,
    test_records INTEGER DEFAULT 0,
    output_format VARCHAR(20) NOT NULL DEFAULT 'parquet',
    storage_path VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    estimated_completion TIMESTAMP,
    error_message TEXT,
    
    CONSTRAINT chk_status CHECK (status IN ('building', 'ready', 'failed')),
    CONSTRAINT chk_split_strategy CHECK (split_strategy IN ('time_based', 'walk_forward')),
    CONSTRAINT chk_periods_order CHECK (
        train_period_start IS NULL OR train_period_end IS NULL OR
        train_period_start < train_period_end
    ),
    CONSTRAINT chk_validation_periods_order CHECK (
        validation_period_start IS NULL OR validation_period_end IS NULL OR
        validation_period_start < validation_period_end
    ),
    CONSTRAINT chk_test_periods_order CHECK (
        test_period_start IS NULL OR test_period_end IS NULL OR
        test_period_start < test_period_end
    )
);

CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_symbol ON datasets(symbol);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_datasets_created_at;
-- DROP INDEX IF EXISTS idx_datasets_symbol;
-- DROP INDEX IF EXISTS idx_datasets_status;
-- DROP TABLE IF EXISTS datasets;

