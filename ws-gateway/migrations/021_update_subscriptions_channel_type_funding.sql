-- Migration: Ensure 'funding' is included in subscriptions.channel_type CHECK constraint
-- Reversible: Yes
--
-- Context:
-- Earlier deployments may have created the subscriptions table without the
-- 'funding' channel_type in the chk_channel_type and chk_symbol_required constraints.
-- This migration updates both constraints to include 'funding' while remaining safe
-- to run on databases where it is already present.

DO $$
BEGIN
    -- Drop existing chk_channel_type constraint if it exists.
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_channel_type'
          AND conrelid = 'subscriptions'::regclass
    ) THEN
        ALTER TABLE subscriptions DROP CONSTRAINT chk_channel_type;
    END IF;

    -- Re-create chk_channel_type constraint with the full, expected set of channel types,
    -- including 'funding'.
    ALTER TABLE subscriptions
    ADD CONSTRAINT chk_channel_type CHECK (
        channel_type IN (
            'trades',
            'ticker',
            'orderbook',
            'order',
            'balance',
            'position',
            'kline',
            'liquidation',
            'funding'
        )
    );

    -- Drop existing chk_symbol_required constraint if it exists.
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_symbol_required'
          AND conrelid = 'subscriptions'::regclass
    ) THEN
        ALTER TABLE subscriptions DROP CONSTRAINT chk_symbol_required;
    END IF;

    -- Re-create chk_symbol_required constraint with 'funding' included in symbol-required channels.
    ALTER TABLE subscriptions
    ADD CONSTRAINT chk_symbol_required CHECK (
        (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'kline', 'funding') AND symbol IS NOT NULL) OR
        (channel_type NOT IN ('trades', 'ticker', 'orderbook', 'order', 'kline', 'funding') AND symbol IS NULL)
    );
END $$;

-- Rollback (reverse migration):
-- NOTE: This rollback restores the constraint to the pre-funding state.
-- Use with caution if other channel types have been added after this migration.
--
-- DO $$
-- BEGIN
--     IF EXISTS (
--         SELECT 1
--         FROM pg_constraint
--         WHERE conname = 'chk_channel_type'
--           AND conrelid = 'subscriptions'::regclass
--     ) THEN
--         ALTER TABLE subscriptions DROP CONSTRAINT chk_channel_type;
--     END IF;
--
--     ALTER TABLE subscriptions
--     ADD CONSTRAINT chk_channel_type CHECK (
--         channel_type IN (
--             'trades',
--             'ticker',
--             'orderbook',
--             'order',
--             'balance',
--             'position',
--             'kline',
--             'liquidation'
--         )
--     );
--
--     IF EXISTS (
--         SELECT 1
--         FROM pg_constraint
--         WHERE conname = 'chk_symbol_required'
--           AND conrelid = 'subscriptions'::regclass
--     ) THEN
--         ALTER TABLE subscriptions DROP CONSTRAINT chk_symbol_required;
--     END IF;
--
--     ALTER TABLE subscriptions
--     ADD CONSTRAINT chk_symbol_required CHECK (
--         (channel_type IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NOT NULL) OR
--         (channel_type NOT IN ('trades', 'ticker', 'orderbook', 'order', 'kline') AND symbol IS NULL)
--     );
-- END $$;

