# Анализ проблем переподключений RabbitMQ в feature-service

## Обзор проблемы

В feature-service наблюдаются периодические ошибки переподключения к RabbitMQ:

- `ConnectionResetError: [Errno 104] Connection reset by peer`
- `AMQPConnectionError: Server connection reset`
- `ChannelInvalidStateError: No active transport in channel`
- `RuntimeError: Connection closed`

Эти ошибки приводят к потере сообщений и нестабильной работе сервиса.

## Анализ ошибок

### 1. Отсутствие настройки heartbeat при создании соединения

**Проблема:**
- В `docker-compose.yml` установлен `RABBITMQ_HEARTBEAT=60` (60 секунд)
- При создании соединения через `connect_robust()` параметр `heartbeat` не передается
- Без heartbeat RabbitMQ может закрывать неактивные соединения, считая их "мертвыми"

**Текущий код:**
```python
# feature-service/src/mq/connection.py:83-88
aio_pika.connect_robust(
    host=self._host,
    port=self._port,
    login=self._user,
    password=self._password,
    # heartbeat не указан!
)
```

**Последствия:**
- RabbitMQ закрывает соединения, которые не отправляют heartbeat в течение 60 секунд
- Клиент не знает о закрытии соединения до попытки использовать его
- Возникают ошибки `Connection reset by peer`

### 2. Создание нового канала при каждом вызове get_channel()

**Проблема:**
- Метод `get_channel()` всегда вызывает `self._connection.channel()`, создавая новый канал
- Это приводит к созданию множества каналов на одном соединении
- RabbitMQ имеет ограничения на количество каналов на соединение

**Текущий код:**
```python
# feature-service/src/mq/connection.py:170-173
return await asyncio.wait_for(
    self._connection.channel(),  # Всегда создает новый канал!
    timeout=10.0,
)
```

**Последствия:**
- Перегрузка RabbitMQ множественными каналами
- Увеличение потребления памяти
- Потенциальные проблемы с производительностью

### 3. Неполная обработка ошибок соединения

**Проблема:**
- В `feature_publisher.py` обрабатываются только `ChannelInvalidStateError`, `AttributeError`, `RuntimeError`
- `ConnectionResetError` и `AMQPConnectionError` попадают в общий `except Exception`
- При этих ошибках не выполняется повторная инициализация канала

**Текущий код:**
```python
# feature-service/src/publishers/feature_publisher.py:84-166
except (aiormq.exceptions.ChannelInvalidStateError, AttributeError, RuntimeError) as e:
    # Обработка с повторной инициализацией
    ...
except Exception as e:
    # Только логирование, без повторной попытки
    logger.error("feature_publish_error", ...)
    return
```

**Последствия:**
- При `ConnectionResetError` сообщения теряются без повторной попытки
- Нет автоматического восстановления соединения

### 4. Отсутствие настройки reconnect_interval

**Проблема:**
- `connect_robust()` не получает параметр `reconnect_interval`
- Используется значение по умолчанию, которое может быть слишком агрессивным
- Это может приводить к слишком частым попыткам переподключения

**Последствия:**
- Избыточная нагрузка на RabbitMQ при переподключениях
- Потенциальные проблемы с производительностью

### 5. Логи RabbitMQ показывают неожиданные закрытия соединений

**Наблюдения из логов:**
```
[warning] client unexpectedly closed TCP connection
```

Это указывает на:
- Неожиданное закрытие соединения клиентом (возможно, из-за отсутствия heartbeat)
- Проблемы с таймаутами
- Нестабильность сетевого соединения

### 6. Каналы не переиспользуются между вызовами publish

**Проблема:**
- В `FeaturePublisher` канал создается в `initialize()` и хранится
- При ошибках канал сбрасывается в `None`, и создается новый канал
- При множественных ошибках это приводит к созданию множества каналов

**Текущий код:**
```python
# feature-service/src/publishers/feature_publisher.py:108-120
if attempt < max_retries - 1:
    # Reset channel and queue references before reinitialization
    self._channel = None
    self._queue = None
    ...
    await self.initialize()  # Создает новый канал
```

**Последствия:**
- Накопление неиспользуемых каналов
- Увеличение потребления ресурсов

### 7. Race condition при конкурентном создании каналов

**Проблема:**
- Есть `_channel_lock` для предотвращения конкурентного создания каналов
- Однако при разрыве соединения несколько компонентов могут одновременно пытаться переподключиться
- Это может приводить к созданию нескольких соединений

**Последствия:**
- Множественные соединения к RabbitMQ
- Потенциальные проблемы с производительностью

## Рекомендации по исправлению

### 1. Добавить параметр heartbeat при создании соединения

```python
# feature-service/src/mq/connection.py
self._connection = await asyncio.wait_for(
    aio_pika.connect_robust(
        host=self._host,
        port=self._port,
        login=self._user,
        password=self._password,
        heartbeat=60,  # Соответствует RABBITMQ_HEARTBEAT в docker-compose.yml
    ),
    timeout=10.0,
)
```

**Обоснование:**
- Обеспечивает соответствие настройкам сервера
- Предотвращает закрытие соединений из-за отсутствия активности
- Позволяет RabbitMQ и клиенту обнаруживать разорванные соединения

### 2. Добавить reconnect_interval для контроля частоты переподключений

```python
aio_pika.connect_robust(
    host=self._host,
    port=self._port,
    login=self._user,
    password=self._password,
    heartbeat=60,
    reconnect_interval=5.0,  # Интервал между попытками переподключения (секунды)
)
```

**Обоснование:**
- Предотвращает слишком частые попытки переподключения
- Снижает нагрузку на RabbitMQ
- Улучшает стабильность системы

### 3. Реализовать переиспользование каналов

**Вариант A: Кэширование каналов в MQConnectionManager**

```python
class MQConnectionManager:
    def __init__(self, ...):
        ...
        self._cached_channel: Optional[aio_pika.Channel] = None
    
    async def get_channel(self) -> aio_pika.Channel:
        # Проверяем, есть ли валидный кэшированный канал
        if self._cached_channel is not None:
            try:
                if not self._cached_channel.is_closed:
                    return self._cached_channel
            except (RuntimeError, AttributeError):
                pass
        
        # Создаем новый канал только если кэшированный невалиден
        self._cached_channel = await self._connection.channel()
        return self._cached_channel
```

**Вариант B: Использование пула каналов**

```python
from aio_pika.pool import Pool

class MQConnectionManager:
    def __init__(self, ...):
        ...
        self._channel_pool: Optional[Pool] = None
    
    async def get_channel(self) -> aio_pika.Channel:
        if self._channel_pool is None:
            self._channel_pool = Pool(
                lambda: self._connection.channel(),
                max_size=10,  # Максимальное количество каналов в пуле
            )
        
        async with self._channel_pool.acquire() as channel:
            return channel
```

**Обоснование:**
- Снижает количество создаваемых каналов
- Улучшает производительность
- Снижает нагрузку на RabbitMQ

### 4. Расширить обработку ошибок соединения

```python
# feature-service/src/publishers/feature_publisher.py
except (
    aiormq.exceptions.ChannelInvalidStateError,
    aiormq.exceptions.AMQPConnectionError,
    ConnectionResetError,
    AttributeError,
    RuntimeError,
) as e:
    # Обработка с повторной инициализацией
    ...
```

**Обоснование:**
- Обеспечивает обработку всех типов ошибок соединения
- Предотвращает потерю сообщений
- Улучшает надежность системы

### 5. Улучшить обработку ошибок соединения

```python
async def publish(self, feature_vector: FeatureVector) -> None:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Проверка и инициализация канала
            if self._queue is None or self._channel is None:
                await self.initialize()
            
            # Публикация сообщения
            await self._channel.default_exchange.publish(...)
            return  # Успех
        
        except (
            aiormq.exceptions.ChannelInvalidStateError,
            aiormq.exceptions.AMQPConnectionError,
            ConnectionResetError,
        ) as e:
            # Сброс соединения при ошибках соединения
            if self._mq_manager is not None:
                # Принудительно закрываем соединение для переподключения
                await self._mq_manager.close()
                self._mq_manager._connection = None
            
            # Сброс канала и очереди
            self._channel = None
            self._queue = None
            
            if attempt < max_retries - 1:
                # Экспоненциальный backoff
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            else:
                logger.error("feature_publish_failed_after_retries", ...)
                return
```

**Обоснование:**
- Обеспечивает полный сброс соединения при критических ошибках
- Использует экспоненциальный backoff для снижения нагрузки
- Улучшает восстановление после ошибок

### 6. Добавить мониторинг и логирование

```python
# Логирование количества активных каналов
logger.info(
    "channel_created",
    channel_count=len(self._connection.channels) if hasattr(self._connection, 'channels') else 'unknown',
)

# Логирование переподключений
logger.warning(
    "connection_reconnecting",
    reconnect_count=self._reconnect_count,
    last_error=str(e),
)
```

**Обоснование:**
- Позволяет отслеживать проблемы в реальном времени
- Помогает в диагностике проблем
- Улучшает наблюдаемость системы

## Дополнительные наблюдения

### Логи RabbitMQ

Из логов видно множественные переподключения:
```
[warning] client unexpectedly closed TCP connection
```

Это указывает на:
- Нестабильность соединения
- Проблемы с heartbeat
- Возможные проблемы с сетью

### Callback ошибки

Ошибки `Callback <OneShotCallback>` указывают на проблемы с обработчиками закрытия каналов в `RobustChannel`:
- Обработчики вызываются после закрытия канала
- Это может приводить к ошибкам при попытке использования закрытого канала

### Сообщение "Connection was not opened"

Это указывает на попытку использования соединения до его полной инициализации:
- Необходимо добавить проверку состояния соединения перед использованием
- Использовать `asyncio.wait_for()` с таймаутом для ожидания готовности соединения

## Приоритет исправлений

1. **Критично:** Добавить `heartbeat` при создании соединения
2. **Критично:** Расширить обработку ошибок (`ConnectionResetError`, `AMQPConnectionError`)
3. **Важно:** Добавить `reconnect_interval`
4. **Важно:** Реализовать переиспользование каналов
5. **Желательно:** Улучшить мониторинг и логирование

## Ссылки

- [Документация aio-pika](https://aio-pika.readthedocs.io/)
- [RabbitMQ Heartbeat Guide](https://www.rabbitmq.com/heartbeats.html)
- [RabbitMQ Connection Management](https://www.rabbitmq.com/connections.html)

