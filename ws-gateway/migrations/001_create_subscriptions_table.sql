-- Migration: Create subscriptions table
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    topic VARCHAR(200) NOT NULL,
    requesting_service VARCHAR(100) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_event_at TIMESTAMP,
    
    CONSTRAINT chk_channel_type CHECK (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'balance', 'position', 'kline', 'liquidation')),
    CONSTRAINT chk_symbol_required CHECK (
        (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NOT NULL) OR
        (channel_type NOT IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_topic ON subscriptions(topic);
CREATE INDEX IF NOT EXISTS idx_subscriptions_active ON subscriptions(is_active);
CREATE INDEX IF NOT EXISTS idx_subscriptions_service ON subscriptions(requesting_service);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_subscriptions_service;
-- DROP INDEX IF EXISTS idx_subscriptions_active;
-- DROP INDEX IF EXISTS idx_subscriptions_topic;
-- DROP TABLE IF EXISTS subscriptions;

