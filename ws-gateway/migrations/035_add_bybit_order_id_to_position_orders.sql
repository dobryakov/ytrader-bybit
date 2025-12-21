-- Migration: Add bybit_order_id to position_orders table
-- Reversible: Yes
-- Purpose: Add bybit_order_id field to support creating position_orders before order is created in DB
--          This allows position-manager to create position_orders when order_id is not yet available

-- Add bybit_order_id column (compatible with current schema)
ALTER TABLE position_orders 
ADD COLUMN IF NOT EXISTS bybit_order_id VARCHAR(255);

-- Create index for fast lookup by bybit_order_id (partial index)
CREATE INDEX IF NOT EXISTS idx_position_orders_bybit_order_id 
ON position_orders(bybit_order_id) 
WHERE bybit_order_id IS NOT NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_position_orders_bybit_order_id;
-- ALTER TABLE position_orders DROP COLUMN IF EXISTS bybit_order_id;

