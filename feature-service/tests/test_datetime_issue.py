#!/usr/bin/env python3
"""
Тестовый скрипт для диагностики проблемы с datetime в asyncpg.

Этот скрипт проверяет различные сценарии смешивания timezone-aware и timezone-naive datetime.
"""
import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta
import json
import os
from typing import Any, Optional


def normalize_datetime(dt: Any) -> Optional[datetime]:
    """Нормализует datetime к timezone-aware UTC."""
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        return dt
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_datetime_in_dict(data: Any) -> Any:
    """Рекурсивно нормализует все datetime в структуре."""
    if data is None:
        return None
    if isinstance(data, datetime):
        return normalize_datetime(data)
    if isinstance(data, dict):
        return {key: normalize_datetime_in_dict(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        normalized = [normalize_datetime_in_dict(item) for item in data]
        return type(data)(normalized) if isinstance(data, tuple) else normalized
    return data


async def test_datetime_scenarios():
    """Тестирует различные сценарии с datetime."""
    
    # Получаем параметры подключения из переменных окружения
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db = os.getenv("POSTGRES_DB", "ytrader")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    print("=" * 80)
    print("ТЕСТ ПРОБЛЕМЫ С DATETIME В ASYNCPG")
    print("=" * 80)
    print()
    
    try:
        # Подключаемся к базе данных
        print(f"Подключение к PostgreSQL: {postgres_host}:{postgres_port}/{postgres_db}")
        conn = await asyncpg.connect(
            host=postgres_host,
            port=postgres_port,
            database=postgres_db,
            user=postgres_user,
            password=postgres_password,
        )
        print("✓ Подключение успешно")
        print()
        
        # Создаем тестовую таблицу, если её нет
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_datasets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                split_strategy TEXT NOT NULL,
                train_period_start TIMESTAMPTZ,
                train_period_end TIMESTAMPTZ,
                validation_period_start TIMESTAMPTZ,
                validation_period_end TIMESTAMPTZ,
                test_period_start TIMESTAMPTZ,
                test_period_end TIMESTAMPTZ,
                walk_forward_config JSONB,
                target_config JSONB,
                feature_registry_version TEXT,
                output_format TEXT DEFAULT 'parquet',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("✓ Тестовая таблица создана/проверена")
        print()
        
        # Тест 1: Все datetime timezone-naive
        print("ТЕСТ 1: Все datetime timezone-naive (без timezone)")
        print("-" * 80)
        naive_start = datetime(2025, 10, 27, 18, 46, 0)
        naive_end = datetime(2025, 10, 28, 18, 46, 0)
        print(f"train_period_start: {naive_start} (naive: {naive_start.tzinfo is None})")
        print(f"train_period_end: {naive_end} (naive: {naive_end.tzinfo is None})")
        
        try:
            await conn.execute("""
                INSERT INTO test_datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                "BTCUSDT",
                "building",
                "time_based",
                naive_start,  # $4
                naive_end,
                naive_start,
                naive_end,
                naive_start,
                naive_end,
                None,
                {"type": "regression", "horizon": "1h"},  # asyncpg автоматически конвертирует dict в JSONB
                "1.0.0",
                "parquet",
            )
            print("✗ ОШИБКА: Запрос прошел успешно, но не должен был!")
            print("  → Это означает, что naive datetime были приняты (не должно быть!)")
        except Exception as e:
            error_msg = str(e)
            print(f"✓ Ожидаемая ошибка: {type(e).__name__}: {error_msg}")
            if "can't subtract offset-naive and offset-aware" in error_msg:
                print("  → ✓ Это именно та ошибка с datetime, которую мы пытаемся исправить!")
            elif "expected str, got dict" in error_msg or "jsonb" in error_msg.lower():
                print("  → Это ошибка с JSONB форматом (не относится к datetime проблеме)")
                print("  → Попробуем с json.dumps()...")
                # Попробуем с json.dumps
                try:
                    await conn.execute("""
                        INSERT INTO test_datasets (
                            symbol, status, split_strategy,
                            train_period_start, train_period_end,
                            validation_period_start, validation_period_end,
                            test_period_start, test_period_end,
                            walk_forward_config, target_config,
                            feature_registry_version, output_format
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        "BTCUSDT",
                        "building",
                        "time_based",
                        naive_start,  # $4
                        naive_end,
                        naive_start,
                        naive_end,
                        naive_start,
                        naive_end,
                        None,
                        json.dumps({"type": "regression", "horizon": "1h"}),
                        "1.0.0",
                        "parquet",
                    )
                    print("    ✗ С json.dumps() запрос прошел, но не должен был (naive datetime!)")
                except Exception as e2:
                    error_msg2 = str(e2)
                    if "can't subtract offset-naive and offset-aware" in error_msg2:
                        print(f"    ✓ С json.dumps() получили datetime ошибку: {error_msg2}")
                        print("    → Это именно та ошибка с datetime, которую мы пытаемся исправить!")
        print()
        
        # Тест 2: Все datetime timezone-aware UTC
        print("ТЕСТ 2: Все datetime timezone-aware UTC")
        print("-" * 80)
        aware_start = datetime(2025, 10, 27, 18, 46, 0, tzinfo=timezone.utc)
        aware_end = datetime(2025, 10, 28, 18, 46, 0, tzinfo=timezone.utc)
        print(f"train_period_start: {aware_start} (naive: {aware_start.tzinfo is None})")
        print(f"train_period_end: {aware_end} (naive: {aware_end.tzinfo is None})")
        
        try:
            await conn.execute("""
                INSERT INTO test_datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                "BTCUSDT",
                "building",
                "time_based",
                aware_start,  # $4
                aware_end,
                aware_start,
                aware_end,
                aware_start,
                aware_end,
                None,
                {"type": "regression", "horizon": "1h"},  # asyncpg автоматически конвертирует dict в JSONB
                "1.0.0",
                "parquet",
            )
            print("✓ Запрос прошел успешно (все datetime timezone-aware)")
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Неожиданная ошибка: {type(e).__name__}: {error_msg}")
            if "can't subtract offset-naive and offset-aware" in error_msg:
                print("  → Это ошибка с datetime!")
        print()
        
        # Тест 3: Смешанные datetime (naive и aware)
        print("ТЕСТ 3: Смешанные datetime (naive и aware) - ВОСПРОИЗВЕДЕНИЕ ПРОБЛЕМЫ")
        print("-" * 80)
        mixed_start = datetime(2025, 10, 27, 18, 46, 0)  # naive
        mixed_end = datetime(2025, 10, 28, 18, 46, 0, tzinfo=timezone.utc)  # aware
        print(f"train_period_start: {mixed_start} (naive: {mixed_start.tzinfo is None})")
        print(f"train_period_end: {mixed_end} (naive: {mixed_end.tzinfo is None})")
        
        try:
            await conn.execute("""
                INSERT INTO test_datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                "BTCUSDT",
                "building",
                "time_based",
                mixed_start,  # $4 - naive
                mixed_end,    # $5 - aware
                mixed_start,
                mixed_end,
                mixed_start,
                mixed_end,
                None,
                {"type": "regression", "horizon": "1h"},  # asyncpg автоматически конвертирует dict в JSONB
                "1.0.0",
                "parquet",
            )
            print("✗ ОШИБКА: Запрос прошел успешно, но не должен был!")
            print("  → Это означает, что смешанные datetime были приняты (не должно быть!)")
        except Exception as e:
            error_msg = str(e)
            print(f"✓ Ожидаемая ошибка (воспроизводит проблему): {type(e).__name__}: {error_msg}")
            if "can't subtract offset-naive and offset-aware" in error_msg:
                print("  → ✓✓✓ Это именно та ошибка с datetime, которую мы пытаемся исправить!")
                print("  → ПРОБЛЕМА ВОСПРОИЗВЕДЕНА УСПЕШНО!")
            elif "expected str, got dict" in error_msg or "jsonb" in error_msg.lower():
                print("  → Это ошибка с JSONB форматом, попробуем с json.dumps()...")
                try:
                    await conn.execute("""
                        INSERT INTO test_datasets (
                            symbol, status, split_strategy,
                            train_period_start, train_period_end,
                            validation_period_start, validation_period_end,
                            test_period_start, test_period_end,
                            walk_forward_config, target_config,
                            feature_registry_version, output_format
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        "BTCUSDT",
                        "building",
                        "time_based",
                        mixed_start,  # $4 - naive
                        mixed_end,    # $5 - aware
                        mixed_start,
                        mixed_end,
                        mixed_start,
                        mixed_end,
                        None,
                        json.dumps({"type": "regression", "horizon": "1h"}),
                        "1.0.0",
                        "parquet",
                    )
                    print("    ✗ С json.dumps() запрос прошел, но не должен был (смешанные datetime!)")
                except Exception as e2:
                    error_msg2 = str(e2)
                    if "can't subtract offset-naive and offset-aware" in error_msg2:
                        print(f"    ✓✓✓ С json.dumps() получили datetime ошибку: {error_msg2}")
                        print("    → ПРОБЛЕМА ВОСПРОИЗВЕДЕНА УСПЕШНО!")
        print()
        
        # Тест 4: Нормализованные datetime
        print("ТЕСТ 4: Нормализованные datetime (все timezone-aware UTC)")
        print("-" * 80)
        raw_start = datetime(2025, 10, 27, 18, 46, 0)  # naive из источника
        raw_end = datetime(2025, 10, 28, 18, 46, 0)  # naive из источника
        normalized_start = normalize_datetime(raw_start)
        normalized_end = normalize_datetime(raw_end)
        print(f"Исходный train_period_start: {raw_start} (naive: {raw_start.tzinfo is None})")
        print(f"Нормализованный train_period_start: {normalized_start} (naive: {normalized_start.tzinfo is None})")
        print(f"Исходный train_period_end: {raw_end} (naive: {raw_end.tzinfo is None})")
        print(f"Нормализованный train_period_end: {normalized_end} (naive: {normalized_end.tzinfo is None})")
        
        try:
            await conn.execute("""
                INSERT INTO test_datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                "BTCUSDT",
                "building",
                "time_based",
                normalized_start,  # $4
                normalized_end,
                normalized_start,
                normalized_end,
                normalized_start,
                normalized_end,
                None,
                {"type": "regression", "horizon": "1h"},  # asyncpg автоматически конвертирует dict в JSONB
                "1.0.0",
                "parquet",
            )
            print("✓ Запрос прошел успешно после нормализации (все datetime timezone-aware)")
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Неожиданная ошибка: {type(e).__name__}: {error_msg}")
            if "can't subtract offset-naive and offset-aware" in error_msg:
                print("  → Это ошибка с datetime! Нормализация не сработала!")
        print()
        
        # Тест 5: datetime в JSONB (walk_forward_config)
        print("ТЕСТ 5: datetime в JSONB структуре (walk_forward_config)")
        print("-" * 80)
        wf_config_with_naive = {
            "start_date": "2025-10-27T18:46:00",  # без timezone
            "end_date": "2025-10-28T18:46:00",
            "train_window_days": 30,
            "validation_window_days": 7,
            "test_window_days": 7,
            "step_days": 7,
        }
        print(f"walk_forward_config (со строками без timezone): {wf_config_with_naive}")
        
        # Парсим строки в datetime
        wf_start = datetime.fromisoformat(wf_config_with_naive["start_date"])
        wf_end = datetime.fromisoformat(wf_config_with_naive["end_date"])
        print(f"Парсинг start_date: {wf_start} (naive: {wf_start.tzinfo is None})")
        print(f"Парсинг end_date: {wf_end} (naive: {wf_end.tzinfo is None})")
        
        # Если datetime в JSONB, они должны быть строками, но проверим случай, если они объекты
        # В реальности JSONB хранит datetime как строки, но проверим нормализацию
        
        # Тест 6: Реальный сценарий - как данные приходят из API
        print("ТЕСТ 6: Реальный сценарий - данные как из API")
        print("-" * 80)
        # Симулируем данные, которые приходят из FastAPI (Pydantic может парсить ISO строки)
        api_data = {
            "symbol": "BTCUSDT",
            "status": "building",
            "split_strategy": "time_based",
            "train_period_start": datetime(2025, 10, 27, 18, 46, 0),  # может быть naive
            "train_period_end": datetime(2025, 10, 28, 18, 46, 0),
            "validation_period_start": datetime(2025, 10, 27, 18, 46, 0),
            "validation_period_end": datetime(2025, 10, 28, 18, 46, 0),
            "test_period_start": datetime(2025, 10, 27, 18, 46, 0),
            "test_period_end": datetime(2025, 10, 28, 18, 46, 0),
            "walk_forward_config": None,
            "target_config": {"type": "regression", "horizon": "1h"},
            "feature_registry_version": "1.0.0",
            "output_format": "parquet",
        }
        
        print("Исходные данные из API:")
        for key, value in api_data.items():
            if isinstance(value, datetime):
                print(f"  {key}: {value} (naive: {value.tzinfo is None})")
            else:
                print(f"  {key}: {value}")
        
        # Нормализуем как в metadata_storage.create_dataset
        normalized_data = normalize_datetime_in_dict(api_data)
        print("\nПосле нормализации:")
        for key, value in normalized_data.items():
            if isinstance(value, datetime):
                print(f"  {key}: {value} (naive: {value.tzinfo is None})")
        
        try:
            await conn.execute("""
                INSERT INTO test_datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                normalized_data["symbol"],
                normalized_data["status"],
                normalized_data["split_strategy"],
                normalized_data["train_period_start"],  # $4
                normalized_data["train_period_end"],
                normalized_data["validation_period_start"],
                normalized_data["validation_period_end"],
                normalized_data["test_period_start"],
                normalized_data["test_period_end"],
                normalized_data["walk_forward_config"],  # asyncpg автоматически конвертирует dict в JSONB
                normalized_data["target_config"],  # asyncpg автоматически конвертирует dict в JSONB
                normalized_data["feature_registry_version"],
                normalized_data["output_format"],
            )
            print("\n✓ Запрос прошел успешно с нормализованными данными")
            print("  → Все datetime были нормализованы к timezone-aware UTC")
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Ошибка: {type(e).__name__}: {error_msg}")
            if "can't subtract offset-naive and offset-aware" in error_msg:
                print("  → Это именно та ошибка с datetime, которую мы пытаемся исправить!")
                print("  → Нормализация не сработала должным образом!")
            elif "expected str, got dict" in error_msg or "jsonb" in error_msg.lower():
                print("  → Это ошибка с JSONB форматом, попробуем с json.dumps()...")
                try:
                    await conn.execute("""
                        INSERT INTO test_datasets (
                            symbol, status, split_strategy,
                            train_period_start, train_period_end,
                            validation_period_start, validation_period_end,
                            test_period_start, test_period_end,
                            walk_forward_config, target_config,
                            feature_registry_version, output_format
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        normalized_data["symbol"],
                        normalized_data["status"],
                        normalized_data["split_strategy"],
                        normalized_data["train_period_start"],  # $4
                        normalized_data["train_period_end"],
                        normalized_data["validation_period_start"],
                        normalized_data["validation_period_end"],
                        normalized_data["test_period_start"],
                        normalized_data["test_period_end"],
                        json.dumps(normalized_data["walk_forward_config"]) if normalized_data["walk_forward_config"] else None,
                        json.dumps(normalized_data["target_config"]) if normalized_data["target_config"] else None,
                        normalized_data["feature_registry_version"],
                        normalized_data["output_format"],
                    )
                    print("    ✓ С json.dumps() запрос прошел успешно")
                    print("    → Все datetime были нормализованы к timezone-aware UTC")
                except Exception as e2:
                    error_msg2 = str(e2)
                    print(f"    ✗ С json.dumps() тоже ошибка: {error_msg2}")
                    if "can't subtract offset-naive and offset-aware" in error_msg2:
                        print("    → Это ошибка с datetime! Нормализация не сработала!")
            import traceback
            traceback.print_exc()
        print()
        
        # Очистка
        await conn.execute("DELETE FROM test_datasets")
        print("✓ Тестовые данные удалены")
        
        await conn.close()
        print("✓ Соединение закрыто")
        
    except Exception as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_datetime_scenarios())
