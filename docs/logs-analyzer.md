# План реализации централизованного логирования с Graylog

## Цели и требования

### Цели
- Сохранить возможность просмотра логов "глазами" через `docker compose logs`
- Отправлять все логи в Graylog для централизованного хранения и анализа
- Решение должно быть минимально инвазивным, без изменения приложений
- Логи должны передаваться в реальном времени

### Требования
- Не изменять код приложений
- Сохранить локальный просмотр логов через Docker
- Обеспечить централизованное хранение и поиск
- Поддержка структурированных логов (JSON из structlog)

## Архитектура решения

### Компоненты

1. **Docker GELF Logging Driver**
   - Дублирует stdout/stderr контейнеров в Graylog
   - Не требует изменений в коде приложений
   - Работает параллельно с `docker compose logs`

2. **Graylog Stack**
   - **MongoDB** — хранение метаданных и конфигурации
   - **Elasticsearch** — индексирование и хранение логов
   - **Graylog** — обработка, парсинг и поиск логов

### Схема потока данных

```
Приложение (stdout) 
    ↓
Docker GELF Driver (дублирование)
    ├─→ docker compose logs (локальный просмотр)
    └─→ Graylog (UDP 4711)
            ↓
        Elasticsearch (индексирование)
            ↓
        Graylog Web UI (поиск и анализ)
```

## Этапы реализации

### Этап 1: Подготовка инфраструктуры

#### 1.1. Добавление сервисов в docker-compose.yml

Добавить три новых сервиса:

**MongoDB:**
- Образ: `mongo:7`
- Порт: 4702 (внутренний, не маппится наружу)
- Volume: `mongo_data` для персистентности
- Health check: `mongosh --eval "db.adminCommand('ping')"`

**Elasticsearch:**
- Образ: `docker.elastic.co/elasticsearch/elasticsearch:7.17.0`
- Порт: 4703 (внутренний, не маппится наружу)
- Volume: `elasticsearch_data` для персистентности
- Переменные окружения:
  - `discovery.type=single-node`
  - `ES_JAVA_OPTS=-Xms512m -Xmx512m` (ограничение памяти)
  - `xpack.security.enabled=false` (для упрощения, в проде включить)
  - `http.port=4703` (изменение стандартного порта 9200)
- Health check: `curl -f http://localhost:4703/_cluster/health`

**Graylog:**
- Образ: `graylog/graylog:5.2`
- Порты:
  - **4701** (внешний) → **4701** (внутренний) — веб-интерфейс
  - **4711** (UDP, внутренний) — GELF прием логов
- Переменные окружения:
  - `GRAYLOG_PASSWORD_SECRET` (из .env)
  - `GRAYLOG_ROOT_PASSWORD_SHA2` (хеш пароля админа)
  - `GRAYLOG_HTTP_EXTERNAL_URI=http://graylog:4701/`
  - `GRAYLOG_HTTP_BIND_ADDRESS=0.0.0.0:4701`
  - `GRAYLOG_ELASTICSEARCH_HOSTS=http://elasticsearch:4703`
  - `GRAYLOG_MONGODB_URI=mongodb://mongo:4702/graylog`
- Health check: `curl -f http://localhost:4701/api/system/lbstatus`

#### 1.2. Добавление переменных окружения в .env

```bash
# Graylog Configuration
GRAYLOG_PASSWORD_SECRET=<генерировать случайную строку>
GRAYLOG_ROOT_PASSWORD=<пароль для админа>
GRAYLOG_ROOT_PASSWORD_SHA2=<SHA256 хеш пароля>
GRAYLOG_WEB_PORT=4701  # Внешний порт для веб-интерфейса
GRAYLOG_GELF_PORT=4711  # Внутренний UDP порт для приема логов
```

#### 1.3. Создание volumes

Добавить в секцию `volumes`:
- `mongo_data`
- `elasticsearch_data`
- `graylog_data` (опционально, для конфигураций)

### Этап 2: Настройка GELF driver для контейнеров

#### 2.1. Обновление logging конфигурации

Для каждого сервиса, который пишет логи, заменить или дополнить секцию `logging`:

**Текущая конфигурация:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "1m"
    max-file: "3"
```

**Новая конфигурация (GELF + json-file):**
```yaml
logging:
  driver: "gelf"
  options:
    gelf-address: "udp://graylog:4711"
    tag: "{{.Name}}"  # Имя контейнера
    labels: "service"  # Метки для фильтрации
    env: "SERVICE_NAME"  # Переменные окружения для тегов
```

**Важно:** Docker поддерживает только один logging driver. Для дублирования нужно использовать альтернативный подход (см. ниже).

#### 2.2. Решение проблемы дублирования

Docker не поддерживает несколько logging drivers одновременно. Варианты:

**Вариант A: Использовать только GELF (рекомендуется)**
- GELF driver отправляет логи в Graylog
- Локальный просмотр через `docker compose logs` продолжит работать (Docker читает из внутреннего буфера)
- Минус: локальные логи не сохраняются на диск

**Вариант B: Использовать Fluentd/Fluent Bit**
- Fluent Bit читает из `json-file` и отправляет в Graylog
- Сохраняет локальные логи
- Требует дополнительный контейнер

**Выбор: Вариант A** (проще, соответствует требованию минимальной инвазивности)

### Этап 3: Настройка Graylog

#### 3.1. Создание GELF UDP Input

После запуска Graylog, через веб-интерфейс или API:

1. Перейти в **System → Inputs**
2. Выбрать **GELF UDP**
3. Настроить:
   - **Title**: `Docker Containers GELF`
   - **Bind address**: `0.0.0.0`
   - **Port**: `4711`
   - **Deploy**: запустить input

#### 3.2. Настройка парсинга JSON логов

Graylog автоматически парсит JSON из поля `message`, но можно улучшить:

1. Создать **Extractor** для поля `message`:
   - Тип: **JSON**
   - Применить ко всем сообщениям

2. Создать **Pipeline Rule** для структурированных логов:
   - Извлечь поля из JSON (level, timestamp, service_name, trace_id и т.д.)
   - Добавить метаданные контейнера (container_name, image)

#### 3.3. Создание Streams (опционально)

Разделить логи по сервисам:
- `ws-gateway-logs`
- `model-service-logs`
- `feature-service-logs`
- `order-manager-logs`
- `position-manager-logs`
- `infrastructure-logs` (postgres, rabbitmq, redis)

Правила для streams:
```
container_name:ws-gateway
container_name:model-service
...
```

### Этап 4: Тестирование

#### 4.1. Проверка отправки логов

1. Запустить сервисы:
   ```bash
   docker compose up -d
   ```

2. Проверить, что Graylog получает логи:
   - Открыть Graylog Web UI: `http://localhost:4701`
   - Перейти в **Search**
   - Должны появиться логи из контейнеров

3. Проверить локальный просмотр:
   ```bash
   docker compose logs -f ws-gateway
   ```
   Должны отображаться логи в реальном времени

#### 4.2. Проверка структурированных полей

1. В Graylog найти лог с JSON
2. Проверить, что поля извлечены (level, timestamp, service_name)
3. Убедиться, что можно фильтровать по полям

#### 4.3. Проверка производительности

1. Мониторинг загрузки Elasticsearch
2. Проверка задержки доставки логов
3. Нагрузочное тестирование (генерация большого объема логов)

### Этап 5: Настройка retention и оптимизация

#### 5.1. Retention Policy

Настроить удаление старых логов в Elasticsearch:

1. В Graylog: **System → Indices**
2. Настроить:
   - **Index retention**: количество дней (например, 30)
   - **Index rotation**: по размеру или времени

#### 5.2. Оптимизация Elasticsearch

Для небольших проектов:
- `ES_JAVA_OPTS=-Xms512m -Xmx512m` (достаточно для ~10GB логов)
- Использовать single-node режим

Для больших проектов:
- Увеличить heap size
- Рассмотреть multi-node кластер

#### 5.3. Мониторинг Graylog

Добавить health checks для всех компонентов:
- MongoDB
- Elasticsearch
- Graylog

Настроить алерты при недоступности сервисов.

#### 5.4. Обработка разных форматов логов

В проекте используются разные форматы логирования в разных сервисах:

**Текущая ситуация:**
- **JSON логи**: `ws-gateway` (в DEBUG), `order-manager` (production), `position-manager` (production)
- **Текстовые логи**: `model-service`, `feature-service`, `ws-gateway` (production)
- **Разные наборы полей**: некоторые сервисы имеют `trace_id`, `logger_name`, другие нет

**Решение: Настройка Extractors и Pipeline Rules**

##### 5.4.1. Автоматический парсинг JSON логов

Graylog автоматически парсит JSON из поля `message`, но можно улучшить:

1. **Создать JSON Extractor** (если автоматический парсинг не работает):
   - Перейти в **System → Inputs → GELF UDP → Manage extractors**
   - Добавить **JSON Extractor** для поля `message`
   - Условие: `message` содержит JSON (начинается с `{`)

2. **Создать Pipeline Rule для автоматического определения формата**:
   ```
   rule "Parse JSON logs"
   when
     has_field("message") && 
     to_string($message.message) =~ /^\{.*\}$/
   then
     let parsed = parse_json(to_string($message.message));
     set_fields(parsed);
   end
   ```

##### 5.4.2. Обработка текстовых логов

Для сервисов с текстовым форматом (model-service, feature-service):

1. **Создать Regex Extractor** для извлечения полей:
   - Пример для model-service:
     ```
     Регулярное выражение: ^(\d{4}-\d{2}-\d{2}T[\d:\.]+Z)\s+\[(\w+)\s+\]\s+(.+?)\s+\[([^\]]+)\](?:\s+trace_id=(\w+))?$
     Поля:
       - timestamp: $1
       - level: $2
       - message: $3
       - logger_name: $4
       - trace_id: $5 (опционально)
     ```

2. **Создать Grok Pattern** для структурированных текстовых логов:
   - Использовать встроенные Grok patterns или создать кастомные
   - Пример: `%{TIMESTAMP_ISO8601:timestamp} \[%{LOGLEVEL:level}\s+\] %{GREEDYDATA:message}`

##### 5.4.3. Унификация полей

Создать Pipeline Rule для добавления недостающих полей:

```
rule "Add service_name from container"
when
  has_field("container_name")
then
  let service_name = regex("([^-]+)", to_string($message.container_name));
  set_field("service_name", service_name[1]);
end

rule "Normalize log level"
when
  has_field("level")
then
  let level_upper = uppercase(to_string($message.level));
  set_field("level", level_upper);
end
```

##### 5.4.4. Обработка ANSI кодов

Для сервисов с цветным выводом (ConsoleRenderer):

1. **Создать Pipeline Rule для удаления ANSI кодов**:
   ```
   rule "Remove ANSI codes"
   when
     has_field("message")
   then
     let clean_message = regex_replace(
       to_string($message.message),
       /\x1b\[[0-9;]*m/g,
       ""
     );
     set_field("message", clean_message);
   end
   ```

##### 5.4.5. Создание Streams по сервисам

Для удобной фильтрации создать Streams:

1. **ws-gateway-logs**:
   - Условие: `container_name:ws-gateway`
   - Автоматически применять JSON extractor

2. **model-service-logs**:
   - Условие: `container_name:model-service`
   - Применить Regex extractor для текстовых логов

3. **feature-service-logs**:
   - Условие: `container_name:feature-service`
   - Применить Regex extractor

4. **order-manager-logs**:
   - Условие: `container_name:order-manager`
   - Автоматически применять JSON extractor

5. **position-manager-logs**:
   - Условие: `container_name:position-manager`
   - Автоматически применять JSON extractor

##### 5.4.6. Рекомендации по унификации (долгосрочно)

Для упрощения обработки рекомендуется:

1. **Перевести все сервисы на JSON формат в production**:
   - Использовать `structlog.processors.JSONRenderer()` вместо `ConsoleRenderer()`
   - Это упростит парсинг и поиск

2. **Унифицировать набор полей**:
   - Все сервисы должны логировать: `level`, `timestamp`, `service_name`, `message`
   - Опционально: `trace_id`, `logger_name`, `request_id`

3. **Пример унифицированного формата**:
   ```json
   {
     "level": "info",
     "timestamp": "2024-01-01T12:00:00.123456Z",
     "service_name": "model-service",
     "logger_name": "model_service.api.orders",
     "trace_id": "abc123",
     "message": "Order created",
     "order_id": "order_123",
     "asset": "BTCUSDT"
   }
   ```

##### 5.4.7. Тестирование парсинга

После настройки Extractors и Pipeline Rules:

1. Проверить, что JSON логи правильно парсятся:
   ```bash
   # В Graylog Search
   container_name:order-manager AND level:info
   ```

2. Проверить, что текстовые логи обрабатываются:
   ```bash
   # В Graylog Search
   container_name:model-service AND trace_id:*
   ```

3. Проверить извлечение полей:
   - Открыть любой лог
   - Убедиться, что поля извлечены (level, timestamp, trace_id и т.д.)
   - Проверить, что можно фильтровать по этим полям

## Список изменений

### Файлы для изменения

1. **docker-compose.yml**
   - Добавить сервисы: `mongo`, `elasticsearch`, `graylog`
   - Обновить `logging` секции для всех сервисов с логами
   - Добавить volumes для персистентности

2. **env.example**
   - Добавить секцию с переменными Graylog
   - Добавить инструкции по генерации `GRAYLOG_PASSWORD_SECRET` и `GRAYLOG_ROOT_PASSWORD_SHA2`

3. **README.md** (обновить)
   - Добавить раздел о централизованном логировании
   - Инструкции по доступу к Graylog Web UI
   - Примеры поиска и фильтрации логов

### Сервисы для обновления logging

Следующие сервисы имеют явную конфигурацию `logging` и требуют обновления:

1. `ws-gateway` (строки 41-45)
2. `feature-service` (строки 304-308)
3. `position-manager` (строки 455-459)

Остальные сервисы используют дефолтный logging driver и также требуют обновления:
- `model-service`
- `order-manager`
- `postgres` (опционально)
- `rabbitmq` (опционально)
- `redis` (опционально)

## Команды для реализации

### Генерация паролей и секретов

```bash
# Генерация GRAYLOG_PASSWORD_SECRET (случайная строка)
openssl rand -base64 32

# Генерация SHA256 хеша пароля для GRAYLOG_ROOT_PASSWORD_SHA2
echo -n "your_password" | sha256sum | awk '{print $1}'
```

### Запуск и проверка

```bash
# Запуск всех сервисов
docker compose up -d

# Проверка статуса Graylog
docker compose ps graylog mongo elasticsearch

# Просмотр логов Graylog
docker compose logs -f graylog

# Проверка доступности Elasticsearch
curl http://localhost:4703/_cluster/health

# Проверка доступности Graylog API (внешний порт)
curl http://localhost:4701/api/system/lbstatus

# Проверка доступности Graylog API (изнутри контейнера)
docker compose exec graylog curl -f http://localhost:4701/api/system/lbstatus
```

### Тестирование отправки логов

```bash
# Генерация тестового лога
docker compose exec ws-gateway echo "Test log message"

# Проверка в Graylog Web UI
# Открыть http://localhost:4701
# Перейти в Search → найти "Test log message"
```

## Риски и митигация

### Риск 1: Высокое потребление ресурсов

**Митигация:**
- Настроить ограничения памяти для Elasticsearch
- Настроить retention policy для удаления старых логов
- Мониторить использование ресурсов

### Риск 2: Потеря логов при недоступности Graylog

**Митигация:**
- Docker буферизует логи при недоступности GELF endpoint
- Настроить health checks и автоматический restart
- Рассмотреть использование persistent queue (Fluent Bit)

### Риск 3: Проблемы с парсингом структурированных логов

**Митигация:**
- Протестировать парсинг JSON логов на этапе разработки
- Создать extractors для каждого типа логов
- Документировать формат логов

## Мониторинг и обслуживание

### Регулярные задачи

1. **Мониторинг использования диска**
   - Проверять размер volumes Elasticsearch
   - Настраивать автоматическую очистку старых логов

2. **Проверка производительности**
   - Мониторить задержку доставки логов
   - Проверять нагрузку на Elasticsearch

3. **Резервное копирование**
   - Настроить backup конфигураций Graylog
   - Рассмотреть backup индексов Elasticsearch (для критичных логов)

### Метрики для мониторинга

- Количество логов в секунду
- Размер индексов Elasticsearch
- Задержка доставки логов (latency)
- Доступность сервисов (uptime)
- Использование памяти и CPU

## Дальнейшие улучшения

1. **Интеграция с Grafana**
   - Использовать Graylog API для запросов из Grafana

2. **Алертинг**
   - Настроить алерты на критические ошибки
   - Интеграция с системами уведомлений

3. **Дашборды**
   - Создать дашборды для анализа логов по сервисам
   - Визуализация ошибок и предупреждений

4. **Расширенный парсинг**
   - Создать extractors для специфичных форматов логов
   - Настроить pipelines для автоматической обработки

## Используемые порты

### Портовая карта

#### Внешние порты (доступны с хоста)

| Сервис | Внешний порт | Внутренний порт | Протокол | Назначение |
|--------|--------------|----------------|----------|------------|
| ws-gateway | 4400 | 4400 | HTTP | REST API |
| model-service | 4500 | 4500 | HTTP | REST API |
| order-manager | 4600 | 4600 | HTTP | REST API |
| grafana | 4700 | 3000 | HTTP | Веб-интерфейс |
| **graylog** | **4701** | **4701** | **HTTP** | **Веб-интерфейс** |
| position-manager | 4800 | 4800 | HTTP | REST API |
| feature-service | 4900 | 4900 | HTTP | REST API |

#### Внутренние порты (только внутри Docker сети)

| Сервис | Порт | Протокол | Назначение |
|--------|------|----------|------------|
| postgres | 5432 | TCP | База данных |
| rabbitmq | 5672 | TCP | Message broker |
| redis | 6379 | TCP | Кэш и блокировки |
| mongo | 4702 | TCP | Метаданные Graylog |
| elasticsearch | 4703 | HTTP | Индексирование логов |
| **graylog (GELF)** | **4711** | **UDP** | **Прием логов** |

### Проверка конфликтов портов

Все порты проверены на конфликты:

- ✅ **4701** — используется для веб-интерфейса Graylog
- ✅ **4702** — используется для MongoDB (метаданные Graylog)
- ✅ **4703** — используется для Elasticsearch (индексирование логов)
- ✅ **4711** — используется для GELF UDP (прием логов)

### Доступ к сервисам

**Graylog Web UI:**
```bash
http://localhost:4701
```

**Проверка GELF input (из контейнера):**
```bash
docker compose exec graylog curl -f http://localhost:4701/api/system/inputs
```

**Проверка Elasticsearch (из контейнера):**
```bash
docker compose exec elasticsearch curl -f http://localhost:4703/_cluster/health
```

**Проверка MongoDB (из контейнера):**
```bash
docker compose exec mongo mongosh --eval "db.adminCommand('ping')"
```

## Заключение

Данный план обеспечивает минимально инвазивную интеграцию централизованного логирования с сохранением возможности локального просмотра логов. Реализация разбита на этапы для постепенного внедрения и тестирования.

Основные преимущества решения:
- ✅ Не требует изменений в коде приложений
- ✅ Сохраняет локальный просмотр логов
- ✅ Обеспечивает централизованное хранение и поиск
- ✅ Поддерживает структурированные логи
- ✅ Работает в реальном времени

