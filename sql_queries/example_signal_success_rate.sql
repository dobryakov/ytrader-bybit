-- Пример использования SQL запроса для расчета процента успешности сигналов
-- 
-- Замените значения в WHERE на нужные:
-- - 'v1.0' -> ваша версия модели
-- - 'BTCUSDT' -> ваш торговый актив
-- - 'strategy_1' -> ваш идентификатор стратегии

-- Пример 1: Простой запрос с конкретными значениями
SELECT 
    DATE_TRUNC('hour', ts.timestamp) as hour,
    COUNT(*) as total_signals,
    COUNT(CASE 
        WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
        THEN 1 
    END) as successful_signals,
    ROUND(
        100.0 * COUNT(CASE 
            WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
            THEN 1 
        END)::numeric / 
        NULLIF(COUNT(CASE 
            WHEN pt.predicted_values->>'direction' IS NOT NULL 
             AND pt.actual_values->>'direction' IS NOT NULL
            THEN 1 
        END), 0),
        2
    ) as success_rate_percent,
    AVG(ts.confidence) as avg_confidence,
    SUM(COALESCE(ptr.total_pnl, 0)) as total_pnl
FROM trading_signals ts
INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
WHERE 
    ts.model_version = 'v1.0'  -- Замените на вашу версию модели
    AND ts.asset = 'BTCUSDT'   -- Замените на ваш актив
    AND ts.strategy_id = 'strategy_1'  -- Замените на ваш strategy_id
    AND pt.actual_values IS NOT NULL
    AND pt.actual_values != '{}'::jsonb
    AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
GROUP BY DATE_TRUNC('hour', ts.timestamp)
ORDER BY hour DESC;

-- Пример 2: С фильтром по датам
SELECT 
    DATE_TRUNC('hour', ts.timestamp) as hour,
    COUNT(*) as total_signals,
    COUNT(CASE 
        WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
        THEN 1 
    END) as successful_signals,
    ROUND(
        100.0 * COUNT(CASE 
            WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
            THEN 1 
        END)::numeric / 
        NULLIF(COUNT(CASE 
            WHEN pt.predicted_values->>'direction' IS NOT NULL 
             AND pt.actual_values->>'direction' IS NOT NULL
            THEN 1 
        END), 0),
        2
    ) as success_rate_percent
FROM trading_signals ts
INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
WHERE 
    ts.model_version = 'v1.0'
    AND ts.asset = 'BTCUSDT'
    AND ts.strategy_id = 'strategy_1'
    AND ts.timestamp >= '2024-01-01'::timestamp  -- Начальная дата
    AND ts.timestamp <= '2024-12-31'::timestamp  -- Конечная дата
    AND pt.actual_values IS NOT NULL
    AND pt.actual_values != '{}'::jsonb
    AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
GROUP BY DATE_TRUNC('hour', ts.timestamp)
ORDER BY hour DESC;

-- Пример 3: С дополнительной статистикой по типам таргетов
SELECT 
    DATE_TRUNC('hour', ts.timestamp) as hour,
    pt.target_config->>'type' as target_type,
    COUNT(*) as total_signals,
    COUNT(CASE 
        WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
        THEN 1 
    END) as successful_signals,
    ROUND(
        100.0 * COUNT(CASE 
            WHEN pt.predicted_values->>'direction' = pt.actual_values->>'direction'
            THEN 1 
        END)::numeric / 
        NULLIF(COUNT(CASE 
            WHEN pt.predicted_values->>'direction' IS NOT NULL 
             AND pt.actual_values->>'direction' IS NOT NULL
            THEN 1 
        END), 0),
        2
    ) as success_rate_percent
FROM trading_signals ts
INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
WHERE 
    ts.model_version = 'v1.0'
    AND ts.asset = 'BTCUSDT'
    AND ts.strategy_id = 'strategy_1'
    AND pt.actual_values IS NOT NULL
    AND pt.actual_values != '{}'::jsonb
    AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
GROUP BY 
    DATE_TRUNC('hour', ts.timestamp),
    pt.target_config->>'type'
ORDER BY 
    hour DESC,
    target_type;

