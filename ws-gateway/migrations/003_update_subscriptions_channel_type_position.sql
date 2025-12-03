-- Migration: Ensure 'position' is included in subscriptions.channel_type CHECK constraint
-- Reversible: Yes
--
-- Context:
-- Earlier deployments may have created the subscriptions table without the
-- 'position' channel_type in the chk_channel_type constraint. This migration
-- updates the constraint to include 'position' while remaining safe to run
-- on databases where it is already present.

DO $$
BEGIN
    -- Drop existing constraint if it exists.
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_channel_type'
          AND conrelid = 'subscriptions'::regclass
    ) THEN
        ALTER TABLE subscriptions DROP CONSTRAINT chk_channel_type;
    END IF;

    -- Re-create constraint with the full, expected set of channel types,
    -- including 'position'.
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
            'liquidation'
        )
    );
END $$;

-- Rollback (reverse migration):
-- NOTE: This rollback restores the constraint to the pre-position state.
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
--             'kline',
--             'liquidation'
--         )
--     );
-- END $$;


