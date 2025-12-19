# Централизованный анализ логов

## Обзор

Документация описывает настройку системы централизованного сбора и анализа логов на основе **Fluent Bit + Loki + Grafana**.

### Архитектура

```
┌─────────────────┐
│   Приложения    │
│ (ws-gateway,    │
│  model-service, │
│  и т.д.)        │
└────────┬────────┘
         │ stdout
         ▼
┌─────────────────┐
│     Docker     │
│  (json-file    │
│   driver)      │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│ docker compose  │  │  Fluent Bit  │
│     logs        │  │  (собирает   │
│  (читает файлы) │  │   логи через │
└─────────────────┘  │ Docker API)  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │     Loki     │
                     │  (хранилище  │
                     │    логов)    │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Grafana    │
                     │ (визуализация│
                     │  и запросы)  │
                     └──────────────┘
```

### Преимущества решения

1. **Двойной доступ к логам:**
   - `docker compose logs` продолжает работать (читает файлы Docker)
   - Grafana предоставляет продвинутый анализ через Loki

2. **Не требует изменений в коде приложений:**
   - Все приложения уже пишут логи в stdout
   - Fluent Bit автоматически собирает логи всех контейнеров

3. **Легковесность:**
   - Fluent Bit - легковесный сборщик (C, минимальные ресурсы)
   - Loki - оптимизирован для логов (меньше ресурсов, чем Elasticsearch)

4. **Структурированные логи:**
   - Автоматический парсинг JSON-логов из structlog
   - Поддержка trace_id для отслеживания запросов

5. **Универсальность:**
   - Fluent Bit может отправлять логи в разные системы (Loki, Elasticsearch, Graylog)
   - Легко переключиться на другую систему хранения

## Компоненты

### 1. Fluent Bit

**Роль:** Сборщик логов из Docker контейнеров

**Как работает:**
- Подключается к Docker socket хоста
- Читает логи всех контейнеров через Docker API
- Парсит JSON-логи из structlog
- Отправляет в Loki

**Преимущества:**
- Работает внутри docker-compose
- Автоматически видит все контейнеры
- Не требует изменений в других сервисах

### 2. Loki

**Роль:** Хранилище и индексация логов

**Особенности:**
- Оптимизирован для логов (не для документов, как Elasticsearch)
- Использует метки (labels) для индексации
- Поддерживает LogQL (язык запросов, похожий на PromQL)
- Эффективное хранение и быстрый поиск

### 3. Grafana

**Роль:** Визуализация и анализ логов

**Возможности:**
- Запросы LogQL для фильтрации и поиска
- Дашборды с графиками и таблицами
- Алерты на основе логов
- Интеграция с существующими дашбордами Grafana

## Установка и настройка

### Шаг 1: Добавление сервисов в docker-compose.yml

Добавьте следующие сервисы в `docker-compose.yml`:

```yaml
  # Loki - хранилище логов
  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "${LOKI_PORT:-3100}:3100"
    volumes:
      - ./loki/config:/etc/loki
      - loki_data:/loki
    command: -config.file=/etc/loki/loki-config.yaml
    networks:
      - ytrader-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Fluent Bit - сборщик логов
  fluent-bit:
    image: fluent/fluent-bit:latest
    container_name: fluent-bit
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./fluent-bit/config:/fluent-bit/etc
    depends_on:
      loki:
        condition: service_healthy
    networks:
      - ytrader-network
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "1m"
        max-file: "3"
```

Добавьте volume для Loki в секцию `volumes`:

```yaml
volumes:
  # ... существующие volumes ...
  loki_data:
```

### Шаг 2: Создание конфигурации Loki

Создайте директорию и файл конфигурации:

```bash
mkdir -p loki/config
```

Создайте файл `loki/config/loki-config.yaml`:

```yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  instance_addr: 127.0.0.1
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093

# Настройки хранения
limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h  # 7 дней
  retention_period: 168h  # 7 дней хранения логов
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
  max_query_length: 721h
  max_query_parallelism: 32
  max_streams_per_user: 10000
  max_line_size: 256KB

# Настройки сжатия
compactor:
  working_directory: /loki/compactor
  shared_store: filesystem
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
```

### Шаг 3: Создание конфигурации Fluent Bit

Создайте директорию и файл конфигурации:

```bash
mkdir -p fluent-bit/config
```

Создайте файл `fluent-bit/config/fluent-bit.conf`:

```ini
[SERVICE]
    Flush         1
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf
    HTTP_Server   On
    HTTP_Listen   0.0.0.0
    HTTP_Port     2020

[INPUT]
    Name              docker_events
    Tag               docker.*
    Docker_Mode       On
    Docker_Mode_Flush 5
    Docker_Mode_Parser_Firstline container_name
    Parser            docker
    Parser            docker_no_time
    Parser            docker_no_time_parse

[INPUT]
    Name              tail
    Path              /var/lib/docker/containers/*/*-json.log
    Parser            docker
    Tag               docker.*
    Refresh_Interval  5
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On
    Skip_Empty_Lines  On

[FILTER]
    Name                kubernetes
    Match               docker.*
    Kube_URL            https://kubernetes.default.svc:443
    Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
    Kube_Tag_Prefix     docker.var.log.containers.
    Merge_Log           On
    Keep_Log           Off
    K8S-Logging.Parser  On
    K8S-Logging.Exclude Off

# Альтернативный фильтр для Docker (без Kubernetes)
[FILTER]
    Name                modify
    Match               docker.*
    Add                 hostname ${HOSTNAME}
    Add                 environment production

# Парсинг JSON-логов из structlog
[FILTER]
    Name                parser
    Match               docker.*
    Key_Name            log
    Parser              json
    Reserve_Data        On
    Preserve_Key        On

# Добавление меток из JSON
[FILTER]
    Name                nest
    Match               docker.*
    Operation           lift
    Nested_under        log
    Add_prefix          log_

# Удаление префикса для удобства
[FILTER]
    Name                modify
    Match               docker.*
    Rename              log_service_name service
    Rename              log_level level
    Rename              log_trace_id trace_id
    Rename              log_event event
    Rename              log_message message
    Rename              log_timestamp timestamp

# Форматирование для Loki
[FILTER]
    Name                record_modifier
    Match               docker.*
    Record              container_name ${HOSTNAME}
    Record              source fluent-bit

[OUTPUT]
    Name        http
    Match       docker.*
    Host        loki
    Port        3100
    URI         /loki/api/v1/push
    Format      json
    Json_date_key    timestamp
    Json_date_format %Y-%m-%dT%H:%M:%S.%L
    HTTP_Header X-Scope-OrgID tenant1
```

Создайте файл `fluent-bit/config/parsers.conf`:

```ini
[PARSER]
    Name        docker
    Format      json
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L
    Time_Keep   On
    Decode_Field_As   escaped_utf8    log    do_next
    Decode_Field_As   escaped         log

[PARSER]
    Name        docker_no_time
    Format      json
    Time_Keep   Off
    Decode_Field_As   escaped_utf8    log    do_next
    Decode_Field_As   escaped         log

[PARSER]
    Name        docker_no_time_parse
    Format      json
    Time_Keep   Off
    Decode_Field_As   escaped_utf8    log    do_next
    Decode_Field_As   escaped         log

[PARSER]
    Name        json
    Format      json
    Time_Key    timestamp
    Time_Format %Y-%m-%dT%H:%M:%S.%L
    Time_Keep   On
```

**Примечание:** Если Fluent Bit не может читать файлы напрямую (нет доступа к `/var/lib/docker/containers`), используйте альтернативную конфигурацию с Docker API через плагин `docker_events`.

### Шаг 4: Настройка Grafana для работы с Loki

Добавьте Loki как data source в Grafana. Обновите файл `grafana/provisioning/datasources/datasources.yml`:

```yaml
datasources:
  # ... существующие data sources ...
  
  # Loki Data Source для логов
  - name: Loki
    uid: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
      derivedFields:
        - datasourceUid: PostgreSQL
          matcherRegex: "trace_id=(\\w+)"
          name: TraceID
          url: "$${__value.raw}"
    isDefault: false
```

### Шаг 5: Добавление переменных окружения

Добавьте в `env.example`:

```bash
# =============================================================================
# Loki Logging Configuration
# =============================================================================
# Loki HTTP API port (non-standard port starting from 3100)
LOKI_PORT=3100
```

## Использование

### Просмотр логов через docker compose

Команда `docker compose logs` продолжает работать как раньше:

```bash
# Все логи
docker compose logs

# Логи конкретного сервиса
docker compose logs ws-gateway

# Последние 100 строк с follow
docker compose logs --tail 100 -f model-service
```

### Просмотр логов через Grafana

1. Откройте Grafana: `http://localhost:4700`
2. Перейдите в **Explore** (иконка компаса слева)
3. Выберите data source **Loki**
4. Используйте LogQL для запросов

### Примеры запросов LogQL

#### Все логи за последний час

```logql
{job="docker"}
```

#### Логи конкретного сервиса

```logql
{container_name="ws-gateway"}
```

#### Логи с уровнем ERROR

```logql
{container_name=~".+"} |= "ERROR"
```

#### Логи с конкретным trace_id

```logql
{container_name=~".+"} | json | trace_id="abc123"
```

#### Логи с фильтрацией по JSON-полям

```logql
{container_name="model-service"} | json | level="ERROR" | message=~".*signal.*"
```

#### Подсчет логов по уровням

```logql
sum by (level) (count_over_time({container_name=~".+"} | json [5m]))
```

#### Логи за последние 15 минут с фильтрацией

```logql
{container_name=~".+"} | json | level="ERROR" | message=~".*error.*"
```

#### Поиск по тексту в сообщении

```logql
{container_name="order-manager"} |= "insufficient balance"
```

#### Логи с группировкой по сервису и уровню

```logql
sum by (container_name, level) (count_over_time({container_name=~".+"} | json [1h]))
```

### Создание дашбордов в Grafana

1. Создайте новый дашборд: **Dashboards → New Dashboard**
2. Добавьте панель: **Add → Visualization**
3. Выберите data source **Loki**
4. Настройте запрос LogQL
5. Выберите тип визуализации (Logs, Time series, Table и т.д.)

#### Пример панели: Логи по уровням

**Query:**
```logql
sum by (level) (count_over_time({container_name=~".+"} | json [5m]))
```

**Visualization:** Time series
**Legend:** `{{level}}`

#### Пример панели: Последние ошибки

**Query:**
```logql
{container_name=~".+"} | json | level="ERROR"
```

**Visualization:** Logs
**Limit:** 100

#### Пример панели: Логи по сервисам

**Query:**
```logql
sum by (container_name) (count_over_time({container_name=~".+"} [1h]))
```

**Visualization:** Bar chart
**Legend:** `{{container_name}}`

## Мониторинг и обслуживание

### Проверка статуса сервисов

```bash
# Статус всех сервисов
docker compose ps

# Логи Fluent Bit
docker compose logs fluent-bit

# Логи Loki
docker compose logs loki

# Проверка здоровья Loki
curl http://localhost:3100/ready
```

### Метрики Fluent Bit

Fluent Bit предоставляет метрики на порту 2020:

```bash
curl http://localhost:2020/api/v1/metrics
```

### Очистка старых логов

Loki автоматически удаляет логи старше 7 дней (настройка `retention_period` в `loki-config.yaml`).

Для ручной очистки:

```bash
# Войти в контейнер Loki
docker compose exec loki sh

# Очистить старые данные (внутри контейнера)
# Loki автоматически управляет retention, но можно проверить размер
du -sh /loki/chunks
```

### Резервное копирование

Логи хранятся в volume `loki_data`. Для резервного копирования:

```bash
# Остановить Loki
docker compose stop loki

# Создать backup
docker run --rm -v ytrader_loki_data:/data -v $(pwd):/backup alpine tar czf /backup/loki-backup-$(date +%Y%m%d).tar.gz /data

# Запустить Loki
docker compose start loki
```

## Устранение неполадок

### Fluent Bit не собирает логи

1. Проверьте доступ к Docker socket:
   ```bash
   docker compose exec fluent-bit ls -la /var/run/docker.sock
   ```

2. Проверьте логи Fluent Bit:
   ```bash
   docker compose logs fluent-bit
   ```

3. Проверьте конфигурацию:
   ```bash
   docker compose exec fluent-bit cat /fluent-bit/etc/fluent-bit.conf
   ```

### Loki не принимает логи

1. Проверьте доступность Loki:
   ```bash
   curl http://localhost:3100/ready
   ```

2. Проверьте логи Loki:
   ```bash
   docker compose logs loki
   ```

3. Проверьте подключение из Fluent Bit:
   ```bash
   docker compose exec fluent-bit wget -O- http://loki:3100/ready
   ```

### Логи не отображаются в Grafana

1. Проверьте data source в Grafana:
   - Settings → Data sources → Loki
   - Test connection

2. Проверьте запрос LogQL:
   - Убедитесь, что используете правильный синтаксис
   - Попробуйте простой запрос: `{job="docker"}`

3. Проверьте временной диапазон:
   - Убедитесь, что выбран правильный период времени

### JSON-логи не парсятся

1. Проверьте формат логов:
   ```bash
   docker compose logs ws-gateway | head -5
   ```

2. Убедитесь, что structlog выводит JSON (не DEBUG режим):
   - В DEBUG режиме используется ConsoleRenderer (не JSON)
   - Для Loki нужен JSONRenderer

3. Проверьте парсер в Fluent Bit:
   - Убедитесь, что используется правильный парсер JSON

## Расширенные настройки

### Настройка retention для разных сервисов

Loki поддерживает разные retention для разных меток. Обновите `loki-config.yaml`:

```yaml
limits_config:
  retention_period: 168h  # По умолчанию 7 дней
  per_stream_rate_limit: 3MB
  per_stream_rate_limit_burst: 15MB
  split_queries_by_interval: 15m
  max_query_series: 500
  max_query_parallelism: 32
```

### Настройка алертов на основе логов

Создайте файл `loki/config/rules/alerts.yaml`:

```yaml
groups:
  - name: log_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate({container_name=~".+"} | json | level="ERROR" [5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"
```

### Интеграция с существующими дашбордами Grafana

Добавьте панели с логами в существующие дашборды:

1. Откройте существующий дашборд
2. Добавьте панель с data source Loki
3. Настройте запрос LogQL для фильтрации по нужному сервису

## Производительность

### Рекомендуемые ресурсы

- **Fluent Bit:** 100-200 MB RAM, 0.1 CPU
- **Loki:** 500 MB - 2 GB RAM (зависит от объема логов), 0.5-1 CPU
- **Хранилище:** ~100 MB на 1 GB логов (сжатие)

### Оптимизация

1. **Ограничение объема логов:**
   - Настройте `max-size` и `max-file` в docker-compose.yml
   - Используйте ротацию логов

2. **Фильтрация в Fluent Bit:**
   - Добавьте фильтры для исключения ненужных логов
   - Используйте `exclude` в парсерах

3. **Retention в Loki:**
   - Уменьшите `retention_period` для старых логов
   - Используйте разные retention для разных сервисов

## Альтернативные решения

### Переход на Graylog

Если нужно переключиться на Graylog:

1. Измените OUTPUT в Fluent Bit:
   ```ini
   [OUTPUT]
       Name        forward
       Match       docker.*
       Host        graylog
       Port        12201
   ```

2. Или используйте GELF:
   ```ini
   [OUTPUT]
       Name        gelf
       Match       docker.*
       Host        graylog
       Port        12201
       Mode        tcp
   ```

### Переход на Elasticsearch

Если нужно переключиться на Elasticsearch:

```ini
[OUTPUT]
    Name        es
    Match       docker.*
    Host        elasticsearch
    Port        9200
    Index       docker-logs
    Type        _doc
```

## Дополнительные ресурсы

- [Документация Loki](https://grafana.com/docs/loki/latest/)
- [Документация Fluent Bit](https://docs.fluentbit.io/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
- [Grafana Logs Panel](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/logs/)

## Поддержка

При возникновении проблем:

1. Проверьте логи всех компонентов:
   ```bash
   docker compose logs fluent-bit loki grafana
   ```

2. Проверьте метрики Fluent Bit:
   ```bash
   curl http://localhost:2020/api/v1/metrics
   ```

3. Проверьте статус Loki:
   ```bash
   curl http://localhost:3100/ready
   curl http://localhost:3100/metrics
   ```

