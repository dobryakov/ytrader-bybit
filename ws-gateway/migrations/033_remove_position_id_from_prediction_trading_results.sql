-- Migration: Remove position_id from prediction_trading_results
-- Reversible: Yes
-- Purpose: Normalize data model - position_id can be obtained via JOIN through signal_order_relationships and position_orders

-- Drop index on position_id
DROP INDEX IF EXISTS idx_prediction_trading_results_position_id;

-- Drop foreign key constraint if exists (PostgreSQL doesn't name it explicitly, but we check)
DO $$
BEGIN
    -- Drop foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE constraint_name = 'prediction_trading_results_position_id_fkey'
        AND table_name = 'prediction_trading_results'
    ) THEN
        ALTER TABLE prediction_trading_results 
        DROP CONSTRAINT prediction_trading_results_position_id_fkey;
    END IF;
END $$;

-- Drop column position_id
ALTER TABLE prediction_trading_results 
DROP COLUMN IF EXISTS position_id;

-- Rollback (reverse migration):
-- ALTER TABLE prediction_trading_results 
-- ADD COLUMN position_id UUID REFERENCES positions(id);
-- CREATE INDEX idx_prediction_trading_results_position_id ON prediction_trading_results(position_id);

