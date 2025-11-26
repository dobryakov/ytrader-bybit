-- Migration: Create model_quality_metrics table
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS model_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    evaluated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluation_dataset_size INTEGER,
    metadata JSONB,
    
    CONSTRAINT chk_metric_type CHECK (metric_type IN ('classification', 'regression', 'trading_performance')),
    CONSTRAINT chk_metric_value_bounds CHECK (
        (metric_type = 'classification' AND metric_value >= 0 AND metric_value <= 1) OR
        (metric_type IN ('regression', 'trading_performance'))
    )
);

CREATE INDEX IF NOT EXISTS idx_model_quality_metrics_model_version_id ON model_quality_metrics(model_version_id);
CREATE INDEX IF NOT EXISTS idx_model_quality_metrics_metric_name ON model_quality_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_model_quality_metrics_evaluated_at ON model_quality_metrics(evaluated_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_quality_metrics_evaluated_at;
-- DROP INDEX IF EXISTS idx_model_quality_metrics_metric_name;
-- DROP INDEX IF EXISTS idx_model_quality_metrics_model_version_id;
-- DROP TABLE IF EXISTS model_quality_metrics;

