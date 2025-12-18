-- Проверка наличия label_mapping в активных моделях
-- Использование: docker compose exec ws-gateway psql -U postgres -d ytrader -f /path/to/check_model_label_mappings.sql

-- Проверить активные модели на наличие label_mapping
SELECT 
    version,
    strategy_id,
    is_active,
    created_at,
    CASE 
        WHEN training_config->>'label_mapping_for_inference' IS NOT NULL 
        THEN 'HAS_LABEL_MAPPING' 
        ELSE 'NO_LABEL_MAPPING' 
    END as mapping_status,
    training_config->>'label_mapping_for_inference' as label_mapping_json,
    training_config->>'task_variant' as task_variant,
    training_config->>'target_registry_version' as target_registry_version
FROM model_versions
WHERE is_active = true
ORDER BY created_at DESC;

-- Подсчет моделей с/без label_mapping
SELECT 
    CASE 
        WHEN training_config->>'label_mapping_for_inference' IS NOT NULL 
        THEN 'HAS_LABEL_MAPPING' 
        ELSE 'NO_LABEL_MAPPING' 
    END as mapping_status,
    COUNT(*) as model_count
FROM model_versions
WHERE is_active = true
GROUP BY mapping_status;

