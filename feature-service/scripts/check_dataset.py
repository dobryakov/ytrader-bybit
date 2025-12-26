#!/usr/bin/env python3
"""
Скрипт для проверки датасета.
Проверяет метаданные из базы данных и фактическое содержимое файлов.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncpg

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


async def get_dataset_from_db(dataset_id: str) -> dict:
    """Получить информацию о датасете из базы данных."""
    conn = await asyncpg.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        database=config.postgres_db,
        user=config.postgres_user,
        password=config.postgres_password,
    )
    
    try:
        row = await conn.fetchrow(
            """
            SELECT * FROM datasets WHERE id = $1
            """,
            dataset_id,
        )
        
        if row is None:
            return None
        
        # Преобразуем в словарь
        dataset_dict = dict(row)
        
        # Нормализуем datetime
        for key, value in dataset_dict.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    dataset_dict[key] = value.replace(tzinfo=datetime.now().astimezone().tzinfo)
        
        return dataset_dict
    finally:
        await conn.close()


def check_dataset_files(storage_path: str, dataset_id: str, output_format: str = "parquet") -> dict:
    """Проверить файлы датасета."""
    dataset_dir = Path(storage_path) / dataset_id
    
    if not dataset_dir.exists():
        return {
            "exists": False,
            "error": f"Директория датасета не найдена: {dataset_dir}",
        }
    
    result = {
        "exists": True,
        "directory": str(dataset_dir),
        "files": {},
        "total_records": 0,
    }
    
    splits = ["train", "validation", "test"]
    
    for split in splits:
        file_path = dataset_dir / f"{split}.{output_format}"
        
        if not file_path.exists():
            result["files"][split] = {
                "exists": False,
                "path": str(file_path),
            }
            continue
        
        try:
            # Читаем файл
            if output_format == "parquet":
                df = pd.read_parquet(file_path)
            elif output_format == "csv":
                df = pd.read_csv(file_path)
            else:
                result["files"][split] = {
                    "exists": True,
                    "error": f"Неподдерживаемый формат: {output_format}",
                }
                continue
            
            # Проверяем данные
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Проверяем наличие timestamp
            has_timestamp = "timestamp" in df.columns
            
            # Подсчитываем NaN по колонкам
            nan_counts = df.isna().sum()
            nan_ratios = (nan_counts / total_rows * 100).round(2) if total_rows > 0 else {}
            
            # Находим колонки с высоким процентом NaN (>50%)
            high_nan_cols = {
                col: {
                    "nan_count": int(nan_counts[col]),
                    "nan_ratio": float(nan_ratios[col]),
                }
                for col in df.columns
                if nan_ratios.get(col, 0) > 50
            }
            
            # Проверяем наличие target
            has_target = "target" in df.columns
            
            # Получаем диапазон timestamp
            timestamp_range = None
            if has_timestamp and total_rows > 0:
                try:
                    timestamps = pd.to_datetime(df["timestamp"])
                    timestamp_range = {
                        "min": timestamps.min().isoformat(),
                        "max": timestamps.max().isoformat(),
                    }
                except Exception as e:
                    timestamp_range = {"error": str(e)}
            
            result["files"][split] = {
                "exists": True,
                "path": str(file_path),
                "total_rows": total_rows,
                "total_cols": total_cols,
                "has_timestamp": has_timestamp,
                "has_target": has_target,
                "timestamp_range": timestamp_range,
                "high_nan_cols": high_nan_cols,
                "high_nan_cols_count": len(high_nan_cols),
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            }
            
            result["total_records"] += total_rows
            
        except Exception as e:
            result["files"][split] = {
                "exists": True,
                "error": f"Ошибка при чтении файла: {str(e)}",
            }
    
    return result


async def main():
    """Основная функция."""
    if len(sys.argv) < 2:
        print("Использование: python check_dataset.py <dataset_id>")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    
    print("=" * 80)
    print(f"ПРОВЕРКА ДАТАСЕТА: {dataset_id}")
    print("=" * 80)
    print()
    
    # Получаем информацию из базы данных
    print("1. Получение метаданных из базы данных...")
    try:
        dataset_meta = await get_dataset_from_db(dataset_id)
        
        if dataset_meta is None:
            print(f"❌ Датасет {dataset_id} не найден в базе данных")
            sys.exit(1)
        
        print("✓ Метаданные получены")
        print()
        print("Метаданные датасета:")
        print(f"  ID: {dataset_meta.get('id')}")
        print(f"  Symbol: {dataset_meta.get('symbol')}")
        print(f"  Status: {dataset_meta.get('status')}")
        print(f"  Split Strategy: {dataset_meta.get('split_strategy')}")
        print(f"  Train Records: {dataset_meta.get('train_records', 0)}")
        print(f"  Validation Records: {dataset_meta.get('validation_records', 0)}")
        print(f"  Test Records: {dataset_meta.get('test_records', 0)}")
        print(f"  Storage Path: {dataset_meta.get('storage_path')}")
        print(f"  Output Format: {dataset_meta.get('output_format', 'parquet')}")
        print(f"  Feature Registry Version: {dataset_meta.get('feature_registry_version')}")
        print(f"  Target Registry Version: {dataset_meta.get('target_registry_version')}")
        print(f"  Created At: {dataset_meta.get('created_at')}")
        print(f"  Completed At: {dataset_meta.get('completed_at')}")
        if dataset_meta.get('error_message'):
            print(f"  Error Message: {dataset_meta.get('error_message')}")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка при получении метаданных: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Проверяем файлы
    print("2. Проверка файлов датасета...")
    storage_path = dataset_meta.get('storage_path')
    output_format = dataset_meta.get('output_format', 'parquet')
    
    if not storage_path:
        print("❌ Storage path не указан в метаданных")
        sys.exit(1)
    
    try:
        files_info = check_dataset_files(storage_path, dataset_id, output_format)
        
        if not files_info.get("exists"):
            print(f"❌ {files_info.get('error')}")
            sys.exit(1)
        
        print("✓ Файлы проверены")
        print()
        print("Информация о файлах:")
        print(f"  Директория: {files_info['directory']}")
        print(f"  Всего записей в файлах: {files_info['total_records']}")
        print()
        
        for split in ["train", "validation", "test"]:
            split_info = files_info["files"].get(split, {})
            
            if not split_info.get("exists"):
                print(f"  {split.upper()}: ❌ Файл не найден ({split_info.get('path')})")
                continue
            
            if "error" in split_info:
                print(f"  {split.upper()}: ❌ Ошибка: {split_info['error']}")
                continue
            
            print(f"  {split.upper()}:")
            print(f"    Путь: {split_info['path']}")
            print(f"    Размер: {split_info['file_size_mb']} MB")
            print(f"    Строк: {split_info['total_rows']}")
            print(f"    Колонок: {split_info['total_cols']}")
            print(f"    Есть timestamp: {'✓' if split_info['has_timestamp'] else '❌'}")
            print(f"    Есть target: {'✓' if split_info['has_target'] else '❌'}")
            
            if split_info.get('timestamp_range'):
                if 'error' in split_info['timestamp_range']:
                    print(f"    Диапазон timestamp: ❌ {split_info['timestamp_range']['error']}")
                else:
                    print(f"    Диапазон timestamp: {split_info['timestamp_range']['min']} - {split_info['timestamp_range']['max']}")
            
            if split_info.get('high_nan_cols_count', 0) > 0:
                print(f"    ⚠️  Колонок с >50% NaN: {split_info['high_nan_cols_count']}")
                # Показываем топ-10 колонок с высоким NaN
                high_nan_sorted = sorted(
                    split_info['high_nan_cols'].items(),
                    key=lambda x: x[1]['nan_ratio'],
                    reverse=True
                )[:10]
                for col, info in high_nan_sorted:
                    print(f"      - {col}: {info['nan_ratio']}% ({info['nan_count']}/{split_info['total_rows']})")
            else:
                print(f"    ✓ Нет колонок с >50% NaN")
            
            print()
        
        # Сравнение с метаданными
        print("3. Сравнение с метаданными:")
        meta_train = dataset_meta.get('train_records', 0)
        meta_val = dataset_meta.get('validation_records', 0)
        meta_test = dataset_meta.get('test_records', 0)
        
        file_train = files_info["files"].get("train", {}).get("total_rows", 0)
        file_val = files_info["files"].get("validation", {}).get("total_rows", 0)
        file_test = files_info["files"].get("test", {}).get("total_rows", 0)
        
        print(f"  Train: метаданные={meta_train}, файл={file_train}, {'✓' if meta_train == file_train else '❌ НЕ СОВПАДАЕТ'}")
        print(f"  Validation: метаданные={meta_val}, файл={file_val}, {'✓' if meta_val == file_val else '❌ НЕ СОВПАДАЕТ'}")
        print(f"  Test: метаданные={meta_test}, файл={file_test}, {'✓' if meta_test == file_test else '❌ НЕ СОВПАДАЕТ'}")
        print()
        
        # Итоговая оценка
        print("=" * 80)
        print("ИТОГОВАЯ ОЦЕНКА:")
        print("=" * 80)
        
        total_file_records = file_train + file_val + file_test
        total_meta_records = meta_train + meta_val + meta_test
        
        if total_file_records == 0:
            print("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: В файлах нет данных!")
        elif total_file_records < total_meta_records * 0.5:
            print(f"⚠️  ПРОБЛЕМА: В файлах значительно меньше данных, чем в метаданных")
            print(f"   Метаданные: {total_meta_records}, Файлы: {total_file_records}")
            print(f"   Разница: {total_meta_records - total_file_records} записей")
        elif total_file_records != total_meta_records:
            print(f"⚠️  ВНИМАНИЕ: Несоответствие количества записей")
            print(f"   Метаданные: {total_meta_records}, Файлы: {total_file_records}")
            print(f"   Разница: {abs(total_meta_records - total_file_records)} записей")
        else:
            print("✓ Количество записей соответствует метаданным")
        
        # Проверка на высокий процент NaN
        all_high_nan = sum(
            files_info["files"].get(split, {}).get("high_nan_cols_count", 0)
            for split in ["train", "validation", "test"]
        )
        
        if all_high_nan > 0:
            print(f"⚠️  ВНИМАНИЕ: Обнаружено {all_high_nan} колонок с >50% NaN значений")
        else:
            print("✓ Нет колонок с критически высоким процентом NaN")
        
        print()
        
    except Exception as e:
        print(f"❌ Ошибка при проверке файлов: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

