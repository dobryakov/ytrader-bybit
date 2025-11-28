-- Migration: Extend account_balances table with additional fields
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store additional coin-level information needed for margin calculations
-- 
-- This migration adds fields from Bybit wallet messages that are used for:
-- - Determining which coins can be used as margin collateral
-- - Calculating available margin for trading
-- - Tracking locked margin in orders and positions

ALTER TABLE account_balances
ADD COLUMN IF NOT EXISTS equity DECIMAL(20, 8),  -- Coin equity value
ADD COLUMN IF NOT EXISTS usd_value DECIMAL(20, 8),  -- USD value of the coin
ADD COLUMN IF NOT EXISTS margin_collateral BOOLEAN DEFAULT false,  -- Can be used as margin collateral
ADD COLUMN IF NOT EXISTS total_order_im DECIMAL(20, 8) DEFAULT 0,  -- Initial margin locked in orders for this coin
ADD COLUMN IF NOT EXISTS total_position_im DECIMAL(20, 8) DEFAULT 0;  -- Initial margin locked in positions for this coin

-- Add constraints (use DO block to handle IF NOT EXISTS)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_equity_non_negative'
    ) THEN
        ALTER TABLE account_balances
        ADD CONSTRAINT chk_equity_non_negative 
            CHECK (equity IS NULL OR equity >= 0);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_usd_value_non_negative'
    ) THEN
        ALTER TABLE account_balances
        ADD CONSTRAINT chk_usd_value_non_negative 
            CHECK (usd_value IS NULL OR usd_value >= 0);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_total_order_im_non_negative'
    ) THEN
        ALTER TABLE account_balances
        ADD CONSTRAINT chk_total_order_im_non_negative 
            CHECK (total_order_im >= 0);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_total_position_im_non_negative'
    ) THEN
        ALTER TABLE account_balances
        ADD CONSTRAINT chk_total_position_im_non_negative 
            CHECK (total_position_im >= 0);
    END IF;
END $$;

-- Rollback (reverse migration):
-- ALTER TABLE account_balances
-- DROP CONSTRAINT IF EXISTS chk_total_position_im_non_negative,
-- DROP CONSTRAINT IF EXISTS chk_total_order_im_non_negative,
-- DROP CONSTRAINT IF EXISTS chk_usd_value_non_negative,
-- DROP CONSTRAINT IF EXISTS chk_equity_non_negative;
-- ALTER TABLE account_balances
-- DROP COLUMN IF EXISTS total_position_im,
-- DROP COLUMN IF EXISTS total_order_im,
-- DROP COLUMN IF EXISTS margin_collateral,
-- DROP COLUMN IF EXISTS usd_value,
-- DROP COLUMN IF EXISTS equity;

