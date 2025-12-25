-- SQL запрос для расчета процента успешности сигналов
-- с фильтрацией по модели, ассету и стратегии
-- с градацией по часам

-- Параметры запроса (замените на нужные значения):
-- :model_version - версия модели (например, 'v1.0')
-- :asset - торговый актив (например, 'BTCUSDT')
-- :strategy_id - идентификатор стратегии (например, 'strategy_1')
-- :start_date - начальная дата (опционально, например '2024-01-01')
-- :end_date - конечная дата (опционально, например '2024-12-31')

WITH signal_predictions AS (
    SELECT 
        ts.signal_id,
        ts.asset,
        ts.strategy_id,
        ts.model_version,
        ts.timestamp as signal_timestamp,
        ts.side,
        ts.confidence,
        -- Предсказанные значения
        pt.predicted_values->>'direction' as predicted_direction,
        pt.predicted_values->>'value' as predicted_value,
        pt.predicted_values->>'class' as predicted_class,
        -- Фактические значения
        pt.actual_values->>'direction' as actual_direction,
        pt.actual_values->>'value' as actual_value,
        pt.actual_values->>'class' as actual_class,
        pt.actual_values->>'return_value' as actual_return,
        pt.actual_values->>'candle_close' as actual_price,
        -- Метаданные
        pt.target_config->>'type' as target_type,
        pt.actual_values_computed_at,
        pt.is_obsolete,
        -- Финансовые результаты (если доступны)
        ptr.total_pnl,
        ptr.realized_pnl,
        ptr.unrealized_pnl,
        ptr.is_closed
    FROM trading_signals ts
    INNER JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
    LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
    WHERE 
        -- Фильтры по параметрам
        ts.model_version = :model_version
        AND ts.asset = :asset
        AND ts.strategy_id = :strategy_id
        -- Только вычисленные таргеты (с actual_values)
        AND pt.actual_values IS NOT NULL
        AND pt.actual_values != '{}'::jsonb
        AND pt.actual_values_computed_at IS NOT NULL
        -- Исключаем устаревшие записи
        AND (pt.is_obsolete IS NULL OR pt.is_obsolete = false)
        -- Опциональные фильтры по дате
        AND (:start_date IS NULL OR ts.timestamp >= :start_date::timestamp)
        AND (:end_date IS NULL OR ts.timestamp <= :end_date::timestamp)
),
success_calculation AS (
    SELECT 
        signal_id,
        asset,
        strategy_id,
        model_version,
        signal_timestamp,
        side,
        confidence,
        predicted_direction,
        actual_direction,
        predicted_value,
        actual_value,
        predicted_class,
        actual_class,
        actual_return,
        actual_price,
        target_type,
        total_pnl,
        realized_pnl,
        is_closed,
        -- Определение успешности в зависимости от типа таргета
        CASE 
            -- Для classification: сравниваем direction или class
            WHEN target_type = 'classification' THEN
                CASE 
                    WHEN predicted_direction IS NOT NULL AND actual_direction IS NOT NULL 
                        THEN predicted_direction = actual_direction
                    WHEN predicted_class IS NOT NULL AND actual_class IS NOT NULL 
                        THEN predicted_class = actual_class
                    ELSE NULL
                END
            -- Для regression: сравниваем направление (знак) возврата
            WHEN target_type = 'regression' THEN
                CASE 
                    WHEN predicted_value IS NOT NULL AND actual_return IS NOT NULL THEN
                        -- Сравниваем знаки: если оба положительные или оба отрицательные - успех
                        (predicted_value::numeric > 0 AND actual_return::numeric > 0)
                        OR (predicted_value::numeric < 0 AND actual_return::numeric < 0)
                        OR (predicted_value::numeric = 0 AND actual_return::numeric = 0)
                    ELSE NULL
                END
            -- Для risk_adjusted или других типов: используем финансовый результат
            WHEN target_type = 'risk_adjusted' OR target_type IS NULL THEN
                CASE 
                    WHEN total_pnl IS NOT NULL THEN total_pnl > 0
                    WHEN actual_return IS NOT NULL THEN actual_return::numeric > 0
                    ELSE NULL
                END
            ELSE NULL
        END as is_successful,
        -- Альтернативный вариант: успешность по финансовому результату
        CASE 
            WHEN total_pnl IS NOT NULL THEN total_pnl > 0
            WHEN actual_return IS NOT NULL THEN actual_return::numeric > 0
            ELSE NULL
        END as is_successful_by_pnl
    FROM signal_predictions
)
SELECT 
    -- Группировка по часам
    DATE_TRUNC('hour', signal_timestamp) as hour,
    -- Статистика по сигналам
    COUNT(*) as total_signals,
    COUNT(CASE WHEN is_successful = true THEN 1 END) as successful_signals,
    COUNT(CASE WHEN is_successful = false THEN 1 END) as failed_signals,
    COUNT(CASE WHEN is_successful IS NULL THEN 1 END) as unknown_signals,
    -- Процент успешности
    ROUND(
        100.0 * COUNT(CASE WHEN is_successful = true THEN 1 END)::numeric / 
        NULLIF(COUNT(CASE WHEN is_successful IS NOT NULL THEN 1 END), 0),
        2
    ) as success_rate_percent,
    -- Альтернативный процент успешности по PnL
    ROUND(
        100.0 * COUNT(CASE WHEN is_successful_by_pnl = true THEN 1 END)::numeric / 
        NULLIF(COUNT(CASE WHEN is_successful_by_pnl IS NOT NULL THEN 1 END), 0),
        2
    ) as success_rate_by_pnl_percent,
    -- Дополнительная статистика
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN side = 'buy' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN side = 'sell' THEN 1 END) as sell_signals,
    -- Финансовые метрики (если доступны)
    SUM(total_pnl) as total_pnl_sum,
    AVG(total_pnl) as avg_pnl,
    COUNT(CASE WHEN is_closed = true THEN 1 END) as closed_positions,
    -- Метаданные
    model_version,
    asset,
    strategy_id
FROM success_calculation
WHERE is_successful IS NOT NULL OR is_successful_by_pnl IS NOT NULL
GROUP BY 
    DATE_TRUNC('hour', signal_timestamp),
    model_version,
    asset,
    strategy_id
ORDER BY 
    hour DESC,
    success_rate_percent DESC;

