-- Migration: Create account_balances table
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS account_balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    coin VARCHAR(10) NOT NULL,
    wallet_balance DECIMAL(20, 8) NOT NULL,
    available_balance DECIMAL(20, 8) NOT NULL,
    frozen DECIMAL(20, 8) NOT NULL DEFAULT 0,
    event_timestamp TIMESTAMP NOT NULL,
    received_at TIMESTAMP NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(100),
    
    CONSTRAINT chk_balance_sum CHECK (wallet_balance = available_balance + frozen),
    CONSTRAINT chk_non_negative CHECK (wallet_balance >= 0 AND available_balance >= 0 AND frozen >= 0)
);

CREATE INDEX IF NOT EXISTS idx_account_balances_coin ON account_balances(coin);
CREATE INDEX IF NOT EXISTS idx_account_balances_received_at ON account_balances(received_at);
CREATE INDEX IF NOT EXISTS idx_account_balances_coin_received_at ON account_balances(coin, received_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_account_balances_coin_received_at;
-- DROP INDEX IF EXISTS idx_account_balances_received_at;
-- DROP INDEX IF EXISTS idx_account_balances_coin;
-- DROP TABLE IF EXISTS account_balances;

