-- Упрощенный SQL запрос для расчета процента успешности сигналов
-- с группировкой по часам
--
-- Использование:
-- Замените :model_version, :asset, :strategy_id на конкретные значения
-- Например: 'v1.0', 'BTCUSDT', 'strategy_1'

SELECT 
    -- Группировка по часам
    DATE_TRUNC('hour', ts.timestamp) as hour,
    
    -- Основные метрики
    COUNT(*) as total_signals,
    COUNT(CASE WHEN pt.actual_values IS NOT NULL AND pt.actual_values != '{}'::jsonb THEN 1 END) as evaluated_signals,
    
    -- Успешность по направлению (direction)
    COUNT(CASE 
        WHEN pt.predicted_values->>'direction' IS NOT NULL 
         AND pt.actual_values->>'direction' IS NOT NULL
         AND pt.predicted_values->>'direction' = pt.actual_values->>'direction'
        THEN 1 
    END) as successful_by_direction,
    
    -- Успешность по финансовому результату (PnL)
    COUNT(CASE 
        WHEN ptr.total_pnl IS NOT NULL AND ptr.total_pnl > 0 
        THEN 1 
    END) as successful_by_pnl,
    
    -- Процент успешности по направлению
    ROUND(
        100.0 * COUNT(CASE 
            WHEN pt.predicted_values->>'direction' IS NOT NULL 
             AND pt.actual_values->>'direction' IS NOT NULL
             AND pt.predicted_values->>'direction' = pt.actual_values->>'direction'
            THEN 1 
        END)::numeric / 
        NULLIF(COUNT(CASE 
            WHEN pt.predicted_values->>'direction' IS NOT NULL 
             AND pt.actual_values->>'direction' IS NOT NULL
            THEN 1 
        END), 0),
        2
    ) as success_rate_direction_percent,
    
    -- Процент успешности по PnL
    ROUND(
        100.0 * COUNT(CASE 
            WHEN ptr.total_pnl IS NOT NULL AND ptr.total_pnl > 0 
            THEN 1 
        END)::numeric / 
        NULLIF(COUNT(CASE WHEN ptr.total_pnl IS NOT NULL THEN 1 END), 0),
        2
    ) as success_rate_pnl_percent,
    
    -- Дополнительная статистика
    AVG(ts.confidence) as avg_confidence,
    COUNT(CASE WHEN ts.side = 'buy' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN ts.side = 'sell' THEN 1 END) as sell_signals,
    SUM(COALESCE(ptr.total_pnl, 0)) as total_pnl_sum,
    AVG(ptr.total_pnl) FILTER (WHERE ptr.total_pnl IS NOT NULL) as avg_pnl
    
FROM trading_signals ts
INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id

WHERE 
    -- Фильтры (замените на нужные значения)
    ts.model_version = :model_version
    AND ts.asset = :asset
    AND ts.strategy_id = :strategy_id
    -- Только вычисленные таргеты
    AND pt.actual_values IS NOT NULL
    AND pt.actual_values != '{}'::jsonb
    AND pt.actual_values_computed_at IS NOT NULL
    -- Исключаем устаревшие
    AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
    -- Опционально: фильтр по дате
    -- AND ts.timestamp >= '2024-01-01'::timestamp
    -- AND ts.timestamp <= '2024-12-31'::timestamp

GROUP BY 
    DATE_TRUNC('hour', ts.timestamp),
    ts.model_version,
    ts.asset,
    ts.strategy_id

ORDER BY 
    hour DESC;

