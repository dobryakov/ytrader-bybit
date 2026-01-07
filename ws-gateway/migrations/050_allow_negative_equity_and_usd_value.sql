-- Migration: Allow negative balances, equity and usd_value in account_balances
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Remove constraints that prevent negative balance, equity and usd_value values
-- 
-- In margin trading, balances, equity and usd_value can be negative when:
-- - Positions are underwater (unrealized losses exceed available balance)
-- - Margin is used for open positions
-- - Borrowed funds are used
--
-- This is normal behavior for unified accounts with active positions.

-- Remove constraints that prevent negative values
DO $$
BEGIN
    -- Remove constraint from initial migration (002)
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_non_negative'
    ) THEN
        ALTER TABLE account_balances
        DROP CONSTRAINT chk_non_negative;
    END IF;
    
    -- Remove constraints from extended fields migration (013)
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_equity_non_negative'
    ) THEN
        ALTER TABLE account_balances
        DROP CONSTRAINT chk_equity_non_negative;
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_usd_value_non_negative'
    ) THEN
        ALTER TABLE account_balances
        DROP CONSTRAINT chk_usd_value_non_negative;
    END IF;
END $$;

-- Rollback (reverse migration):
-- DO $$
-- BEGIN
--     IF NOT EXISTS (
--         SELECT 1 FROM pg_constraint 
--         WHERE conname = 'chk_non_negative'
--     ) THEN
--         ALTER TABLE account_balances
--         ADD CONSTRAINT chk_non_negative 
--             CHECK (wallet_balance >= 0 AND available_balance >= 0 AND frozen >= 0);
--     END IF;
--     
--     IF NOT EXISTS (
--         SELECT 1 FROM pg_constraint 
--         WHERE conname = 'chk_equity_non_negative'
--     ) THEN
--         ALTER TABLE account_balances
--         ADD CONSTRAINT chk_equity_non_negative 
--             CHECK (equity IS NULL OR equity >= 0);
--     END IF;
--     
--     IF NOT EXISTS (
--         SELECT 1 FROM pg_constraint 
--         WHERE conname = 'chk_usd_value_non_negative'
--     ) THEN
--         ALTER TABLE account_balances
--         ADD CONSTRAINT chk_usd_value_non_negative 
--             CHECK (usd_value IS NULL OR usd_value >= 0);
--     END IF;
-- END $$;

