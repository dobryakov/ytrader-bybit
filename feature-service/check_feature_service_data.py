#!/usr/bin/env python3
"""
Скрипт для проверки накопленных данных в feature-service.
"""
import subprocess
import json
from datetime import datetime, timedelta

def run_cmd(cmd):
    """Выполнить команду и вернуть результат."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_logs_count(pattern, since="1h"):
    """Подсчитать количество вхождений в логах."""
    cmd = f"docker compose logs feature-service --since {since} 2>&1 | grep -c '{pattern}' || echo '0'"
    result = run_cmd(cmd)
    # Взять только первое число, если есть несколько строк
    try:
        return int(result.split()[0])
    except (ValueError, IndexError):
        return 0

def format_size(size_bytes):
    """Форматировать размер в читаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

print("=" * 60)
print("СТАТИСТИКА ДАННЫХ FEATURE-SERVICE")
print("=" * 60)
print()

# 1. Проверка хранилища данных
print("📁 ХРАНИЛИЩЕ ДАННЫХ:")
print("-" * 60)
raw_size = run_cmd("docker compose exec -T feature-service du -sb /data/raw 2>/dev/null | awk '{print $1}' || echo '0'")
dataset_size = run_cmd("docker compose exec -T feature-service du -sb /data/datasets 2>/dev/null | awk '{print $1}' || echo '0'")
raw_files = run_cmd("docker compose exec -T feature-service find /data/raw -type f 2>/dev/null | wc -l || echo '0'")
dataset_files = run_cmd("docker compose exec -T feature-service find /data/datasets -type f 2>/dev/null | wc -l || echo '0'")

print(f"  Raw Data Storage: {format_size(int(raw_size or 0))} ({raw_files} файлов)")
print(f"  Dataset Storage: {format_size(int(dataset_size or 0))} ({dataset_files} файлов)")
print()

# 2. Статистика обработки событий
print("📊 СТАТИСТИКА ОБРАБОТКИ:")
print("-" * 60)
features_1h = get_logs_count("features_computed", "1h")
features_24h = get_logs_count("features_computed", "24h")
events_1h = get_logs_count("market_data_event_processed", "1h")
events_24h = get_logs_count("market_data_event_processed", "24h")

print(f"  Вычислено фич за последний час: {features_1h}")
print(f"  Вычислено фич за 24 часа: {features_24h}")
print(f"  Обработано событий за час: {events_1h}")
print(f"  Обработано событий за 24 часа: {events_24h}")
print()

# 3. Информация о символах
print("💰 ОТСЛЕЖИВАЕМЫЕ СИМВОЛЫ:")
print("-" * 60)
symbols = run_cmd("docker compose exec -T feature-service env | grep FEATURE_SERVICE_SYMBOLS | cut -d'=' -f2 || echo 'не указано'")
symbol_list = symbols.split(",") if symbols != "не указано" else []
print(f"  Символы: {', '.join(symbol_list) if symbol_list else 'не указано'} ({len(symbol_list)} символов)")
print()

# 4. Rolling Windows в памяти
print("🔄 ROLLING WINDOWS (в памяти):")
print("-" * 60)
print("  Окна времени: 1s, 3s, 15s, 1m")
print("  Тип данных: Trades и Klines")
print("  Хранение: только в памяти, данные автоматически удаляются после истечения окна")
print()
print("  Примечание: Rolling windows хранят данные только в рамках временных окон:")
print("    - 1s: данные за последнюю 1 секунду")
print("    - 3s: данные за последние 3 секунды")
print("    - 15s: данные за последние 15 секунд")
print("    - 1m: данные за последнюю 1 минуту")
print()

# 5. Статус сервиса
print("🏥 СТАТУС СЕРВИСА:")
print("-" * 60)
health = run_cmd("docker compose exec -T feature-service curl -s http://localhost:4900/health 2>/dev/null || echo '{}'")
try:
    health_data = json.loads(health)
    status = health_data.get("status", "unknown")
    timestamp = health_data.get("timestamp", "unknown")
    print(f"  Статус: {status}")
    print(f"  Последняя проверка: {timestamp}")
except:
    print("  Статус: недоступен")
print()

# 6. Очереди RabbitMQ
print("📨 ПОДПИСКИ НА ОЧЕРЕДИ:")
print("-" * 60)
queue_info = run_cmd("docker compose exec -T rabbitmq rabbitmqadmin list queues name messages consumers 2>/dev/null | grep 'ws-gateway.trades' || echo 'не найдено'")
if queue_info and queue_info != "не найдено":
    parts = queue_info.split("|")
    if len(parts) >= 3:
        queue_name = parts[0].strip()
        messages = parts[1].strip()
        consumers = parts[2].strip()
        print(f"  Очередь {queue_name}:")
        print(f"    Сообщений в очереди: {messages}")
        print(f"    Активных потребителей: {consumers}")
else:
    print("  Информация об очередях недоступна")
print()

print("=" * 60)
print("ЗАКЛЮЧЕНИЕ:")
print("-" * 60)
print("Feature-service хранит данные в двух местах:")
print("1. В памяти (Rolling Windows): данные за последние 1s-1m для вычисления фич")
print("2. На диске (Raw Data Storage): долгосрочное хранилище (сейчас пусто)")
print()
if int(raw_size or 0) == 0:
    print("⚠️  Хранилище данных пока пусто - возможно, сохранение данных еще не настроено")
else:
    print(f"✅ На диске сохранено {format_size(int(raw_size or 0))} данных")
print()
print(f"📈 За последний час обработано ~{features_1h} вычислений фич")
print(f"📈 За 24 часа обработано ~{features_24h} вычислений фич")

