-- Migration: 053_migrate_closed_positions_to_positions.sql
-- Purpose: Migrate data from closed_positions table to positions table
--          Set closed_at for migrated records
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)
-- WARNING: This migration should be run AFTER 052_extend_positions_table_for_unified_architecture.sql

BEGIN;

-- Check if closed_positions table exists
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'closed_positions'
    ) THEN
        -- Migrate data from closed_positions to positions
        -- Map fields from closed_positions to positions structure
        INSERT INTO positions (
            id,
            asset,
            mode,
            size,  -- Always 0 for closed positions
            average_entry_price,
            current_price,
            exit_price,
            unrealized_pnl,  -- Set to unrealized_pnl_at_close
            realized_pnl,
            long_size,
            short_size,
            long_avg_price,
            short_avg_price,
            total_fees,
            created_at,
            closed_at,
            version,
            -- Set other fields to NULL or defaults
            leverage,
            position_value,
            liq_price,
            bust_price,
            take_profit,
            stop_loss,
            cum_realised_pnl,
            cum_unrealised_pnl,
            opening_fees,
            closing_fees,
            margin_used,
            available_margin,
            maintenance_margin,
            max_size,
            min_size,
            total_volume_traded,
            first_entry_price,
            last_entry_price,
            peak_unrealized_pnl,
            peak_unrealized_pnl_at,
            worst_unrealized_pnl,
            worst_unrealized_pnl_at,
            last_updated,
            source,
            last_sync_with_bybit,
            bybit_position_data
        )
        SELECT 
            original_position_id as id,  -- Use original_position_id as position id
            asset,
            mode,
            0 as size,  -- Closed positions always have size = 0
            average_entry_price,
            current_price,
            exit_price,
            unrealized_pnl_at_close as unrealized_pnl,
            realized_pnl,
            long_size,
            short_size,
            long_avg_price,
            short_avg_price,
            COALESCE(total_fees, 0) as total_fees,
            opened_at as created_at,  -- Use opened_at as created_at (approximation)
            closed_at,
            version,
            -- Other fields set to NULL
            NULL as leverage,
            NULL as position_value,
            NULL as liq_price,
            NULL as bust_price,
            NULL as take_profit,
            NULL as stop_loss,
            NULL as cum_realised_pnl,
            NULL as cum_unrealised_pnl,
            NULL as opening_fees,
            NULL as closing_fees,
            NULL as margin_used,
            NULL as available_margin,
            NULL as maintenance_margin,
            NULL as max_size,
            NULL as min_size,
            0 as total_volume_traded,
            average_entry_price as first_entry_price,  -- Use average_entry_price as first_entry_price
            average_entry_price as last_entry_price,  -- Use average_entry_price as last_entry_price
            unrealized_pnl_at_close as peak_unrealized_pnl,  -- Use unrealized_pnl_at_close as peak
            closed_at as peak_unrealized_pnl_at,
            unrealized_pnl_at_close as worst_unrealized_pnl,  -- Use unrealized_pnl_at_close as worst
            closed_at as worst_unrealized_pnl_at,
            closed_at as last_updated,  -- Use closed_at as last_updated
            'migration_from_closed_positions' as source,
            NULL as last_sync_with_bybit,
            NULL as bybit_position_data
        FROM closed_positions
        WHERE NOT EXISTS (
            -- Skip if position with same id already exists in positions
            SELECT 1 FROM positions 
            WHERE positions.id = closed_positions.original_position_id
        )
        ON CONFLICT (id) DO NOTHING;  -- Skip if position already exists
        
        RAISE NOTICE 'Migrated % rows from closed_positions to positions', 
            (SELECT COUNT(*) FROM closed_positions);
    ELSE
        RAISE NOTICE 'closed_positions table does not exist, skipping migration';
    END IF;
END $$;

COMMIT;

-- Rollback section
-- WARNING: This rollback will DELETE migrated positions from positions table
-- To rollback this migration, run:
--
-- BEGIN;
-- DELETE FROM positions 
-- WHERE source = 'migration_from_closed_positions' 
--   AND closed_at IS NOT NULL;
-- COMMIT;

