-- Migration: Fix incorrect entry_price and recalculate PnL in prediction_trading_results
-- Reversible: No (one-time data fix)
-- Purpose: Fix entry_price values that were incorrectly saved from execution_events.execution_price
--          and recalculate realized_pnl based on correct prices

-- Step 1: Create temporary table with correct prices
CREATE TEMP TABLE IF NOT EXISTS price_fixes AS
SELECT 
    ptr.id,
    ptr.signal_id,
    ptr.entry_signal_id,
    ptr.entry_price as current_entry_price,
    ptr.exit_price as current_exit_price,
    ptr.realized_pnl as current_realized_pnl,
    ptr.position_size_at_entry,
    ptr.is_closed,
    -- Get correct entry price: prefer orders.price (most reliable), then orders.average_price, then position_orders.execution_price
    -- Note: position_orders.execution_price may contain incorrect values, so we prioritize orders.price
    COALESCE(
        o_entry.price,
        o_entry.average_price,
        por_entry.execution_price,
        ptr.entry_price  -- Fallback to current if nothing found
    ) as correct_entry_price,
    -- Get correct exit price: prefer orders.price, then orders.average_price, then position_orders.execution_price
    COALESCE(
        o_exit.price,
        o_exit.average_price,
        por_exit.execution_price,
        ptr.exit_price  -- Fallback to current if nothing found
    ) as correct_exit_price,
    -- Get position average entry price for PnL calculation
    p.average_entry_price as position_avg_entry_price
FROM prediction_trading_results ptr
JOIN trading_signals ts ON ptr.entry_signal_id = ts.signal_id
-- Get entry order
LEFT JOIN signal_order_relationships sor_entry ON sor_entry.signal_id = ptr.signal_id
LEFT JOIN orders o_entry ON o_entry.id = sor_entry.order_id
LEFT JOIN position_orders por_entry ON por_entry.order_id = o_entry.id AND por_entry.relationship_type IN ('opened', 'increased')
-- Get exit order (if closed)
LEFT JOIN signal_order_relationships sor_exit ON sor_exit.signal_id = ptr.signal_id
LEFT JOIN orders o_exit ON o_exit.id = sor_exit.order_id
LEFT JOIN position_orders por_exit ON por_exit.order_id = o_exit.id AND por_exit.relationship_type IN ('closed', 'decreased')
-- Get position for average entry price
LEFT JOIN position_orders por_pos ON por_pos.order_id = sor_entry.order_id
LEFT JOIN positions p ON p.id = por_pos.position_id
WHERE ptr.entry_price IS NOT NULL;

-- Step 2: Update entry_price with correct values
UPDATE prediction_trading_results ptr
SET entry_price = pf.correct_entry_price
FROM price_fixes pf
WHERE ptr.id = pf.id
  AND pf.correct_entry_price IS NOT NULL
  AND ABS(pf.current_entry_price - pf.correct_entry_price) > 10;  -- Only update if difference > 10 USDT

-- Step 3: Update exit_price with correct values (if closed)
UPDATE prediction_trading_results ptr
SET exit_price = pf.correct_exit_price
FROM price_fixes pf
WHERE ptr.id = pf.id
  AND ptr.is_closed = true
  AND pf.correct_exit_price IS NOT NULL
  AND (ptr.exit_price IS NULL OR ABS(ptr.exit_price - pf.correct_exit_price) > 10);

-- Step 4: Recalculate realized_pnl for closed positions
UPDATE prediction_trading_results ptr
SET realized_pnl = (
    CASE 
        WHEN ptr.position_size_at_entry > 0 THEN
            -- Long position: (exit_price - entry_price) * quantity - fees
            (ptr.exit_price - ptr.entry_price) * ABS(ptr.position_size_at_entry) - COALESCE(ptr.fees, 0)
        WHEN ptr.position_size_at_entry < 0 THEN
            -- Short position: (entry_price - exit_price) * quantity - fees
            (ptr.entry_price - ptr.exit_price) * ABS(ptr.position_size_at_entry) - COALESCE(ptr.fees, 0)
        ELSE 0
    END
),
total_pnl = (
    CASE 
        WHEN ptr.position_size_at_entry > 0 THEN
            (ptr.exit_price - ptr.entry_price) * ABS(ptr.position_size_at_entry) - COALESCE(ptr.fees, 0)
        WHEN ptr.position_size_at_entry < 0 THEN
            (ptr.entry_price - ptr.exit_price) * ABS(ptr.position_size_at_entry) - COALESCE(ptr.fees, 0)
        ELSE 0
    END
) + COALESCE(ptr.unrealized_pnl, 0)
FROM price_fixes pf
WHERE ptr.id = pf.id
  AND ptr.is_closed = true
  AND ptr.entry_price IS NOT NULL
  AND ptr.exit_price IS NOT NULL
  AND ptr.position_size_at_entry IS NOT NULL
  AND ptr.position_size_at_entry != 0;

-- Step 5: Show summary of changes
SELECT 
    'Summary of fixes' as info,
    COUNT(*) FILTER (WHERE ABS(current_entry_price - correct_entry_price) > 10) as entry_prices_fixed,
    COUNT(*) FILTER (WHERE is_closed AND ABS(current_exit_price - COALESCE(correct_exit_price, current_exit_price)) > 10) as exit_prices_fixed,
    COUNT(*) FILTER (WHERE is_closed) as closed_positions_recalculated
FROM price_fixes;

