#!/usr/bin/env python3
"""
Проверка и исправление схемы таблицы datasets.
"""
import asyncio
import asyncpg
import os
import sys

# Добавляем путь к корню проекта
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

async def check_and_fix_schema():
    """Проверяет схему таблицы и исправляет при необходимости."""
    
    # Получаем параметры подключения из переменных окружения
    postgres_host = os.getenv("POSTGRES_HOST", "postgres")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db = os.getenv("POSTGRES_DB", "ytrader")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    print("=" * 80)
    print("ПРОВЕРКА И ИСПРАВЛЕНИЕ СХЕМЫ ТАБЛИЦЫ datasets")
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
        
        # Проверяем текущую схему таблицы
        print("Проверка текущей схемы таблицы datasets:")
        print("-" * 80)
        
        columns = await conn.fetch("""
            SELECT 
                column_name,
                data_type,
                udt_name
            FROM information_schema.columns
            WHERE table_name = 'datasets'
            AND column_name IN (
                'train_period_start', 'train_period_end',
                'validation_period_start', 'validation_period_end',
                'test_period_start', 'test_period_end'
            )
            ORDER BY column_name
        """)
        
        if not columns:
            print("✗ Таблица datasets не найдена или не содержит datetime колонок")
            await conn.close()
            return
        
        print("Текущие типы колонок:")
        needs_fix = False
        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            udt_name = col['udt_name']
            print(f"  {col_name}: {data_type} (udt: {udt_name})")
            
            # Проверяем, является ли тип timestamptz
            if udt_name != 'timestamptz' and 'timestamp' in data_type.lower():
                print(f"    ⚠ Колонка {col_name} имеет тип {data_type}, должен быть timestamptz")
                needs_fix = True
        
        print()
        
        if not needs_fix:
            print("✓ Все колонки имеют правильный тип timestamptz")
        else:
            print("Исправление типов колонок...")
            print("-" * 80)
            
            # Исправляем каждую колонку
            datetime_columns = [
                'train_period_start', 'train_period_end',
                'validation_period_start', 'validation_period_end',
                'test_period_start', 'test_period_end'
            ]
            
            for col_name in datetime_columns:
                try:
                    # Проверяем текущий тип
                    col_info = await conn.fetchrow("""
                        SELECT data_type, udt_name
                        FROM information_schema.columns
                        WHERE table_name = 'datasets'
                        AND column_name = $1
                    """, col_name)
                    
                    if col_info and col_info['udt_name'] != 'timestamptz':
                        print(f"Исправление колонки {col_name}...")
                        
                        # Используем предложенный пользователем подход
                        await conn.execute(f"""
                            ALTER TABLE datasets
                            ALTER COLUMN {col_name} TYPE timestamptz
                            USING {col_name} AT TIME ZONE 'UTC'
                        """)
                        
                        print(f"  ✓ Колонка {col_name} исправлена")
                    else:
                        print(f"  - Колонка {col_name} уже имеет правильный тип")
                        
                except Exception as e:
                    print(f"  ✗ Ошибка при исправлении колонки {col_name}: {e}")
        
        print()
        
        # Проверяем финальное состояние
        print("Финальное состояние схемы:")
        print("-" * 80)
        columns = await conn.fetch("""
            SELECT 
                column_name,
                data_type,
                udt_name
            FROM information_schema.columns
            WHERE table_name = 'datasets'
            AND column_name IN (
                'train_period_start', 'train_period_end',
                'validation_period_start', 'validation_period_end',
                'test_period_start', 'test_period_end'
            )
            ORDER BY column_name
        """)
        
        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            udt_name = col['udt_name']
            status = "✓" if udt_name == 'timestamptz' else "✗"
            print(f"  {status} {col_name}: {data_type} (udt: {udt_name})")
        
        await conn.close()
        print()
        print("✓ Проверка завершена")
        
    except Exception as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(check_and_fix_schema())
