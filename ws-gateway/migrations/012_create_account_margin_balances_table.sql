-- Migration: Create account_margin_balances table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store account-level margin and balance information for unified accounts
-- This allows order-manager to check available margin without calling Bybit API
-- 
-- Note: total_available_balance can be negative when margin is used for open positions.
-- This is normal for unified accounts with active positions and is allowed by design.

CREATE TABLE IF NOT EXISTS account_margin_balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_type VARCHAR(20) NOT NULL,  -- 'UNIFIED', 'SPOT', etc.
    total_equity DECIMAL(20, 8) NOT NULL,
    total_wallet_balance DECIMAL(20, 8) NOT NULL,
    total_margin_balance DECIMAL(20, 8) NOT NULL,
    total_available_balance DECIMAL(20, 8) NOT NULL,  -- Available margin for trading (can be negative)
    total_initial_margin DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- Locked in positions
    total_maintenance_margin DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- Maintenance margin for positions
    total_order_im DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- Locked in pending orders
    base_currency VARCHAR(10) NOT NULL,  -- Base currency for margin (USDT, USD, etc.)
    event_timestamp TIMESTAMP NOT NULL,
    received_at TIMESTAMP NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(100),
    
    CONSTRAINT chk_non_negative CHECK (
        total_equity >= 0 AND 
        total_wallet_balance >= 0 AND 
        total_margin_balance >= 0 AND 
        -- total_available_balance can be negative when margin is used for positions
        total_initial_margin >= 0 AND
        total_maintenance_margin >= 0 AND
        total_order_im >= 0
    )
);

CREATE INDEX IF NOT EXISTS idx_account_margin_balances_received_at 
    ON account_margin_balances(received_at DESC);
CREATE INDEX IF NOT EXISTS idx_account_margin_balances_account_type 
    ON account_margin_balances(account_type);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_account_margin_balances_account_type;
-- DROP INDEX IF EXISTS idx_account_margin_balances_received_at;
-- DROP TABLE IF EXISTS account_margin_balances;

