-- Migration: Make order_id nullable in position_orders and update unique constraints
-- Reversible: Yes (with data loss if NULL values exist)
-- Purpose: Allow position_orders to be created before order is created in DB
--          This supports asynchronous event processing where order and position_orders
--          can be created in any order
-- Depends on: 035_add_bybit_order_id_to_position_orders.sql

-- 1. Drop old unique constraint (will be replaced with partial index)
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS uq_position_order;

-- 2. Drop foreign key for order_id (will be recreated with NULL support)
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS position_orders_order_id_fkey;

-- 3. Change order_id to NULLABLE
ALTER TABLE position_orders 
ALTER COLUMN order_id DROP NOT NULL;

-- 4. Create partial unique index for order_id IS NOT NULL
--    (NULL values won't be checked for uniqueness)
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_order_unique 
ON position_orders(position_id, order_id) 
WHERE order_id IS NOT NULL;

-- 5. Create unique index for order_id IS NULL with bybit_order_id
--    (prevents duplicate position_orders with NULL order_id for same position)
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_bybit_order_unique 
ON position_orders(position_id, bybit_order_id) 
WHERE order_id IS NULL AND bybit_order_id IS NOT NULL;

-- 6. Recreate foreign key with NULL support
ALTER TABLE position_orders 
ADD CONSTRAINT position_orders_order_id_fkey 
FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE;

-- Rollback (reverse migration):
-- Note: Rollback may fail if there are NULL order_id values
-- ALTER TABLE position_orders DROP CONSTRAINT IF EXISTS position_orders_order_id_fkey;
-- DROP INDEX IF EXISTS position_orders_position_bybit_order_unique;
-- DROP INDEX IF EXISTS position_orders_position_order_unique;
-- ALTER TABLE position_orders ALTER COLUMN order_id SET NOT NULL;
-- ALTER TABLE position_orders ADD CONSTRAINT uq_position_order UNIQUE (position_id, order_id);
-- ALTER TABLE position_orders ADD CONSTRAINT position_orders_order_id_fkey 
--     FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE;

