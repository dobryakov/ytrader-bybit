# Optimized Dataset Builder

## Обзор

OptimizedDatasetBuilder - это полностью переписанная версия DatasetBuilder, обеспечивающая ускорение построения датасетов в 10-50 раз за счет:

- **Векторизованных вычислений** - использование pandas/numpy для массовых операций
- **Многоуровневого кэширования** - локальный кэш + Redis для минимизации I/O
- **Потоковой обработки** - обработка данных по дням вместо загрузки всего периода
- **Адаптивной стратегии кэширования** - автоматический выбор оптимальной стратегии
- **Инкрементальных обновлений** - переиспользование состояния orderbook и rolling windows

## Архитектура

### Основные компоненты

1. **FeatureRequirementsAnalyzer** - анализирует Feature Registry и определяет требования к данным
2. **OptimizedRollingWindow** - оптимизированный rolling window с фиксированным буфером
3. **AdaptiveCacheStrategy** - адаптивная стратегия кэширования
4. **OptimizedDailyDataCache** - многоуровневый кэш (локальный + Redis)
5. **AdaptivePrefetcher** - адаптивная предзагрузка данных
6. **HybridFeatureComputer** - гибридный подход к вычислению фич (векторизация + streaming)
7. **IncrementalOrderbookManager** - инкрементальное управление orderbook
8. **StreamingDatasetBuilder** - потоковая обработка данных по дням
9. **OptimizedDatasetBuilder** - финальная интеграция всех компонентов

## Использование

### Включение оптимизированного builder

Добавьте в `.env`:

```bash
DATASET_BUILDER_USE_OPTIMIZED=true
DATASET_BUILDER_BATCH_SIZE=1000
```

### API

API остается тем же самым - все endpoints работают одинаково:

```bash
POST /dataset/build
{
  "symbol": "BTCUSDT",
  "split_strategy": "time_based",
  "target_registry_version": "1.0.0",
  "train_period_start": "2024-01-01T00:00:00Z",
  "train_period_end": "2024-01-02T00:00:00Z",
  ...
}
```

### Программное использование

```python
from src.services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder

builder = OptimizedDatasetBuilder(
    metadata_storage=metadata_storage,
    parquet_storage=parquet_storage,
    dataset_storage_path="/data/datasets",
    cache_service=cache_service,
    feature_registry_loader=feature_registry_loader,
    target_registry_version_manager=target_registry_version_manager,
    dataset_publisher=dataset_publisher,
    batch_size=1000,
)

dataset_id = await builder.build_dataset(
    symbol="BTCUSDT",
    split_strategy=SplitStrategy.TIME_BASED,
    target_registry_version="1.0.0",
    train_period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    train_period_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
    ...
)
```

## Стратегии кэширования

### Короткие периоды (1-3 дня)
- Кэшируется весь период
- Предзагрузка отключена

### Средние периоды (4-7 дней)
- Кэшируется 2 дня
- Предзагрузка следующего дня (24 часа вперед)

### Длинные периоды (8+ дней)
- Кэшируется 1 день
- Адаптивная предзагрузка (2 часа вперед, подстраивается под скорость обработки)

## Производительность

### Ожидаемое ускорение

- **Короткие периоды (1-3 дня)**: 10-20x быстрее
- **Средние периоды (4-7 дней)**: 20-30x быстрее
- **Длинные периоды (8+ дней)**: 30-50x быстрее

### Факторы ускорения

1. **Векторизация**: 5-10x ускорение вычислений фич
2. **Кэширование**: 2-5x ускорение за счет минимизации I/O
3. **Потоковая обработка**: 2-3x ускорение за счет эффективного использования памяти
4. **Инкрементальные обновления**: 1.5-2x ускорение за счет переиспользования состояния

## Обратная совместимость

OptimizedDatasetBuilder полностью совместим с существующим API и форматами данных:

- Те же endpoints
- Те же форматы выходных данных (parquet, csv)
- Те же split стратегии (time_based, walk_forward)
- Те же форматы метаданных

## Тестирование

### Unit-тесты

```bash
pytest tests/unit/test_requirements_analyzer.py
pytest tests/unit/test_optimized_rolling_window.py
pytest tests/unit/test_cache_strategy.py
pytest tests/unit/test_incremental_orderbook.py
```

### Integration-тесты

```bash
pytest tests/integration/test_optimized_dataset_building.py
```

## Конфигурация

### Переменные окружения

- `DATASET_BUILDER_USE_OPTIMIZED` - использовать оптимизированный builder (default: false)
- `DATASET_BUILDER_BATCH_SIZE` - размер батча для обработки timestamp'ов (default: 1000)
- `CACHE_REDIS_ENABLED` - включить Redis кэш (default: true)
- `DATASET_BUILDER_CACHE_ENABLED` - включить кэширование для dataset builder (default: true)

## Ограничения

1. **Recovery support**: Оптимизированный builder пока не поддерживает `recover_incomplete_builds()` (в разработке)
2. **Memory usage**: Для очень больших периодов (>30 дней) может потребоваться больше памяти
3. **Redis dependency**: Для максимальной производительности рекомендуется использовать Redis

## Миграция

### С старого builder на новый

1. Добавьте `DATASET_BUILDER_USE_OPTIMIZED=true` в `.env`
2. Перезапустите сервис
3. Проверьте, что датасеты строятся корректно
4. При необходимости настройте `DATASET_BUILDER_BATCH_SIZE`

### Откат

Просто удалите или установите `DATASET_BUILDER_USE_OPTIMIZED=false` и перезапустите сервис.

## Мониторинг

OptimizedDatasetBuilder логирует:

- Статистику кэша (hit rate, misses, parquet reads)
- Скорость обработки (timestamps per second)
- Использование памяти
- Время выполнения этапов

Пример логов:

```
INFO: cache_strategy_determined strategy=full_period period_days=1
INFO: streaming_dataset_build_started symbol=BTCUSDT days=3
INFO: cache_statistics cache_hits=100 cache_misses=10 hit_rate=90.91%
INFO: optimized_dataset_build_completed dataset_id=... train_records=1000
```

## Поддержка

При возникновении проблем:

1. Проверьте логи на наличие ошибок
2. Убедитесь, что Redis доступен (если используется)
3. Проверьте, что достаточно памяти
4. Попробуйте уменьшить `DATASET_BUILDER_BATCH_SIZE`

