#!/usr/bin/env python3
"""
Тестовый скрипт для проверки реального метода create_dataset.

Этот скрипт использует реальный код MetadataStorage для проверки проблемы.
"""
import asyncio
import sys
import os
from pathlib import Path

# Добавляем путь к корню проекта
# В контейнере рабочая директория - /app, поэтому добавляем /app в sys.path
# Это позволяет импортировать модули как "from src.storage.metadata_storage import ..."
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from datetime import datetime, timezone
from src.storage.metadata_storage import MetadataStorage
import os


async def test_real_create_dataset():
    """Тестирует реальный метод create_dataset."""
    
    print("=" * 80)
    print("ТЕСТ РЕАЛЬНОГО МЕТОДА create_dataset")
    print("=" * 80)
    print()
    
    # Создаем экземпляр MetadataStorage
    storage = MetadataStorage()
    
    try:
        # Инициализируем подключение
        await storage.initialize()
        print("✓ MetadataStorage инициализирован")
        print()
        
        # Тест 1: Все datetime timezone-naive (как могут прийти из API)
        print("ТЕСТ 1: Все datetime timezone-naive (как из API)")
        print("-" * 80)
        dataset_data_naive = {
            "symbol": "BTCUSDT",
            "status": "building",
            "split_strategy": "time_based",
            "train_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "train_period_end": datetime(2025, 10, 28, 18, 46, 0),  # naive
            "validation_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "validation_period_end": datetime(2025, 10, 28, 18, 46, 0),  # naive
            "test_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "test_period_end": datetime(2025, 10, 28, 18, 46, 0),  # naive
            "walk_forward_config": None,
            "target_config": {"type": "regression", "horizon": 3600},
            "feature_registry_version": "1.0.0",
            "output_format": "parquet",
        }
        
        print("Исходные данные (все naive):")
        for key, value in dataset_data_naive.items():
            if isinstance(value, datetime):
                print(f"  {key}: {value} (naive: {value.tzinfo is None})")
        
        try:
            dataset_id = await storage.create_dataset(dataset_data_naive)
            print(f"\n✓ Успешно создан dataset с ID: {dataset_id}")
            
            # Проверяем, что данные сохранились правильно
            dataset = await storage.get_dataset(dataset_id)
            print("\nДанные из базы:")
            for key in ["train_period_start", "train_period_end", "validation_period_start", 
                       "validation_period_end", "test_period_start", "test_period_end"]:
                if key in dataset and dataset[key]:
                    dt = dataset[key]
                    print(f"  {key}: {dt} (type: {type(dt).__name__}, naive: {dt.tzinfo is None if hasattr(dt, 'tzinfo') else 'N/A'})")
            
            # Удаляем тестовый dataset
            async with storage.transaction() as conn:
                await conn.execute("DELETE FROM datasets WHERE id = $1", dataset_id)
            print("\n✓ Тестовый dataset удален")
            
        except Exception as e:
            print(f"\n✗ ОШИБКА: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Тест 2: Смешанные datetime (naive и aware)
        print("ТЕСТ 2: Смешанные datetime (naive и aware)")
        print("-" * 80)
        dataset_data_mixed = {
            "symbol": "BTCUSDT",
            "status": "building",
            "split_strategy": "time_based",
            "train_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "train_period_end": datetime(2025, 10, 28, 18, 46, 0, tzinfo=timezone.utc),  # aware
            "validation_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "validation_period_end": datetime(2025, 10, 28, 18, 46, 0, tzinfo=timezone.utc),  # aware
            "test_period_start": datetime(2025, 10, 27, 18, 46, 0),  # naive
            "test_period_end": datetime(2025, 10, 28, 18, 46, 0, tzinfo=timezone.utc),  # aware
            "walk_forward_config": None,
            "target_config": {"type": "regression", "horizon": 3600},
            "feature_registry_version": "1.0.0",
            "output_format": "parquet",
        }
        
        print("Исходные данные (смешанные):")
        for key, value in dataset_data_mixed.items():
            if isinstance(value, datetime):
                print(f"  {key}: {value} (naive: {value.tzinfo is None})")
        
        try:
            dataset_id = await storage.create_dataset(dataset_data_mixed)
            print(f"\n✓ Успешно создан dataset с ID: {dataset_id}")
            
            # Удаляем тестовый dataset
            async with storage.transaction() as conn:
                await conn.execute("DELETE FROM datasets WHERE id = $1", dataset_id)
            print("✓ Тестовый dataset удален")
            
        except Exception as e:
            print(f"\n✗ ОШИБКА: {type(e).__name__}: {e}")
            if "can't subtract offset-naive and offset-aware" in str(e):
                print("  → Это именно та ошибка, которую мы пытаемся исправить!")
            import traceback
            traceback.print_exc()
        print()
        
        # Тест 3: datetime в walk_forward_config (если они там есть как объекты)
        print("ТЕСТ 3: datetime в walk_forward_config")
        print("-" * 80)
        dataset_data_with_wf = {
            "symbol": "BTCUSDT",
            "status": "building",
            "split_strategy": "walk_forward",
            "train_period_start": None,
            "train_period_end": None,
            "validation_period_start": None,
            "validation_period_end": None,
            "test_period_start": None,
            "test_period_end": None,
            "walk_forward_config": {
                "start_date": "2025-10-27T18:46:00",  # строка без timezone
                "end_date": "2025-10-28T18:46:00",
                "train_window_days": 30,
                "validation_window_days": 7,
                "test_window_days": 7,
                "step_days": 7,
                # Если бы здесь были datetime объекты (что маловероятно, но возможно)
                "some_datetime": datetime(2025, 10, 27, 18, 46, 0),  # naive datetime в конфиге
            },
            "target_config": {"type": "regression", "horizon": 3600},
            "feature_registry_version": "1.0.0",
            "output_format": "parquet",
        }
        
        print("Исходные данные (с datetime в walk_forward_config):")
        print(f"  walk_forward_config: {dataset_data_with_wf['walk_forward_config']}")
        
        try:
            dataset_id = await storage.create_dataset(dataset_data_with_wf)
            print(f"\n✓ Успешно создан dataset с ID: {dataset_id}")
            
            # Удаляем тестовый dataset
            async with storage.transaction() as conn:
                await conn.execute("DELETE FROM datasets WHERE id = $1", dataset_id)
            print("✓ Тестовый dataset удален")
            
        except Exception as e:
            print(f"\n✗ ОШИБКА: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        await storage.close()
        print("✓ MetadataStorage закрыт")
        
    except Exception as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_real_create_dataset())
