-- Migration: Create model_predictions table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store raw model predictions (probabilities) for test split analysis

CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(255) NOT NULL,
    dataset_id UUID NOT NULL,
    split VARCHAR(50) NOT NULL,  -- 'train', 'validation', 'test'
    training_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Массив предсказаний (JSONB)
    -- Формат: [{"y_true": -1, "probabilities": [0.2, 0.3, 0.5], "confidence": 0.5}, ...]
    predictions JSONB NOT NULL,
    
    -- Метаданные (task_type, task_variant, num_classes, etc.)
    metadata JSONB,
    
    CONSTRAINT chk_split CHECK (split IN ('train', 'validation', 'test'))
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_model_version ON model_predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_model_predictions_dataset_id ON model_predictions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_split ON model_predictions(split);
CREATE INDEX IF NOT EXISTS idx_model_predictions_created_at ON model_predictions(created_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_predictions_created_at;
-- DROP INDEX IF EXISTS idx_model_predictions_split;
-- DROP INDEX IF EXISTS idx_model_predictions_dataset_id;
-- DROP INDEX IF EXISTS idx_model_predictions_model_version;
-- DROP TABLE IF EXISTS model_predictions;

