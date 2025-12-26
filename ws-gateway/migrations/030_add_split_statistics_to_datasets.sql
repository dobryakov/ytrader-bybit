-- Migration: Add split_statistics to datasets table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store statistics for each split (train/validation/test) including class distribution and target statistics

-- Add split_statistics JSONB field to store statistics for each split
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS split_statistics JSONB;

-- Add comment explaining the structure
COMMENT ON COLUMN datasets.split_statistics IS 'Statistics for each split: {
  "train": {
    "class_distribution": {"0": 1000, "1": 500, "-1": 300},
    "target_statistics": {"mean": 0.001, "median": 0.0, "std": 0.05, "min": -0.1, "max": 0.1},
    "class_balance_ratio": 0.5,
    "minority_class_size": 300
  },
  "validation": {...},
  "test": {...}
}';

-- Rollback (reverse migration):
-- ALTER TABLE datasets DROP COLUMN IF EXISTS split_statistics;

