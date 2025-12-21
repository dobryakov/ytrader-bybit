# Анализ защиты от Path Traversal атак

**Дата анализа:** 2025-12-21  
**Статус:** feature-service защищен, требуется защита в других сервисах

## Резюме

Проанализированы все сервисы проекта на предмет уязвимостей path traversal. Обнаружены потенциальные уязвимости в нескольких сервисах, требующие внедрения защиты.

## Статус защиты по сервисам

### ✅ feature-service — ЗАЩИЩЕН

**Статус:** Защита внедрена и протестирована

**Реализовано:**
- Security middleware для проверки путей в URL
- Валидация путей в эндпоинтах (`download_dataset`, `sync_feature_registry_file`)
- Обработчики исключений для возврата 404 вместо 500
- Функции `is_path_traversal_attempt()` и `validate_path_safe()`

**Файлы с защитой:**
- `feature-service/src/api/middleware/security.py` — security middleware
- `feature-service/src/main.py` — обработчики исключений
- `feature-service/src/api/dataset.py` — валидация путей в download_dataset
- `feature-service/src/api/feature_registry.py` — валидация file_path

---

### ⚠️ model-service — ТРЕБУЕТ ЗАЩИТЫ

**Критичность:** ВЫСОКАЯ

**Уязвимые места:**

1. **API эндпоинты с параметром `version` в пути:**
   - `GET /api/v1/models/{version}` — получение деталей модели
   - `POST /api/v1/models/{version}/activate` — активация модели
   - `GET /api/v1/models/{version}/metrics` — метрики модели
   - `GET /api/v1/models/{version}/metrics/time-series` — временные ряды метрик

2. **Методы работы с файлами:**
   - `ModelStorage.get_model_path(version, filename)` — строит путь: `base_path / f"v{version.lstrip('v')}" / filename`
   - `ModelStorage.get_version_directory(version)` — строит путь: `base_path / f"v{version.lstrip('v')}"`
   - `ModelStorage.load_model(version, filename)` — загружает модель по пути
   - `ModelStorage.save_model(version, filename)` — сохраняет модель по пути
   - `ModelStorage.delete_model(version, filename)` — удаляет модель

**Риск:**
Если параметр `version` содержит path traversal последовательности (например, `../../etc/passwd`), это может привести к:
- Чтению файлов вне директории моделей
- Записи файлов в произвольные места
- Удалению файлов вне директории моделей

**Пример уязвимого запроса:**
```bash
GET /api/v1/models/../../etc/passwd
# Может привести к попытке доступа к /path/to/models/v../../etc/passwd
```

**Рекомендации:**
1. Добавить security middleware (аналогично feature-service)
2. Валидировать параметр `version` в API эндпоинтах
3. Добавить проверку в `ModelStorage.get_model_path()` и `get_version_directory()`
4. Использовать `validate_path_safe()` для проверки, что результирующий путь находится в пределах `base_path`
5. Добавить обработчики исключений для возврата 404 вместо 500

**Файлы для изменения:**
- `model-service/src/api/middleware/security.py` — создать security middleware
- `model-service/src/main.py` — добавить middleware и обработчики исключений
- `model-service/src/api/models.py` — валидация version
- `model-service/src/api/metrics.py` — валидация version
- `model-service/src/services/storage.py` — валидация путей в методах работы с файлами

---

### ⚠️ order-manager — ТРЕБУЕТ ЗАЩИТЫ

**Критичность:** СРЕДНЯЯ

**Уязвимые места:**
- API эндпоинты с параметрами в пути (например, `{order_id}`, `{asset}`)
- Нет прямой работы с файлами через API, но возможны общие path traversal атаки через URL

**Риск:**
- Path traversal в URL может привести к ошибкам 500 вместо 404
- Потенциальная утечка информации через stack traces

**Рекомендации:**
1. Добавить security middleware для проверки путей в URL
2. Добавить обработчики исключений для возврата 404 вместо 500

**Файлы для изменения:**
- `order-manager/src/api/middleware/security.py` — создать security middleware
- `order-manager/src/main.py` — добавить middleware и обработчики исключений

---

### ⚠️ position-manager — ТРЕБУЕТ ЗАЩИТЫ

**Критичность:** СРЕДНЯЯ

**Уязвимые места:**
- API эндпоинты с параметрами в пути (например, `{asset}`)
- Нет прямой работы с файлами через API, но возможны общие path traversal атаки через URL

**Риск:**
- Path traversal в URL может привести к ошибкам 500 вместо 404
- Потенциальная утечка информации через stack traces

**Рекомендации:**
1. Добавить security middleware для проверки путей в URL
2. Добавить обработчики исключений для возврата 404 вместо 500

**Файлы для изменения:**
- `position-manager/src/api/middleware/security.py` — создать security middleware
- `position-manager/src/main.py` — добавить middleware и обработчики исключений

---

### ⚠️ ws-gateway — ТРЕБУЕТ ЗАЩИТЫ

**Критичность:** СРЕДНЯЯ

**Уязвимые места:**
- API эндпоинты с параметрами в пути (например, `{subscription_id}`)
- Нет прямой работы с файлами через API, но возможны общие path traversal атаки через URL

**Риск:**
- Path traversal в URL может привести к ошибкам 500 вместо 404
- Потенциальная утечка информации через stack traces

**Рекомендации:**
1. Добавить security middleware для проверки путей в URL
2. Добавить обработчики исключений для возврата 404 вместо 500

**Файлы для изменения:**
- `ws-gateway/src/api/middleware/security.py` — создать security middleware
- `ws-gateway/src/main.py` — добавить middleware и обработчики исключений

---

## Общие рекомендации

### 1. Security Middleware

Создать универсальный security middleware для всех сервисов:

```python
# src/api/middleware/security.py
- is_path_traversal_attempt(path: str) -> bool
- validate_path_safe(base_path: Path, file_path: Path) -> bool
- security_middleware(request: Request, call_next) -> Response
```

### 2. Обработчики исключений

Добавить в каждый сервис:
- Обработчик `RequestValidationError` для возврата 404 вместо 422/500
- Обработчик общих исключений для подозрительных путей

### 3. Валидация параметров пути

Валидировать все параметры пути, которые используются для работы с файлами:
- Проверка на path traversal последовательности (`..`, `//`, `~`, null bytes)
- Проверка, что результирующий путь находится в пределах базовой директории

### 4. Приоритет внедрения

1. **model-service** — ВЫСОКИЙ (работа с файлами)
2. **order-manager** — СРЕДНИЙ (общая защита)
3. **position-manager** — СРЕДНИЙ (общая защита)
4. **ws-gateway** — СРЕДНИЙ (общая защита)

---

## Шаблон для внедрения

Для каждого сервиса необходимо:

1. **Создать security middleware:**
   - Скопировать `feature-service/src/api/middleware/security.py`
   - Адаптировать под структуру сервиса

2. **Добавить middleware в main.py:**
   ```python
   from src.api.middleware.security import security_middleware
   
   @app.middleware("http")
   async def security_middleware_wrapper(request, call_next):
       return await security_middleware(request, call_next)
   ```

3. **Добавить обработчики исключений:**
   ```python
   @app.exception_handler(RequestValidationError)
   async def validation_exception_handler(...)
   
   @app.exception_handler(Exception)
   async def general_exception_handler(...)
   ```

4. **Валидировать параметры в эндпоинтах:**
   - Для model-service: валидация `version` и `filename`
   - Для других сервисов: валидация параметров пути при необходимости

---

## Тестирование

После внедрения защиты протестировать:

```bash
# Path traversal атаки
curl "http://localhost:PORT/../../../../../../etc/passwd"
curl "http://localhost:PORT/api/v1/models/../../etc/passwd"

# Должны возвращать 404, а не 500
```

---

## Заключение

- ✅ **feature-service** — защищен
- ⚠️ **model-service** — требует срочной защиты (работа с файлами)
- ⚠️ **order-manager, position-manager, ws-gateway** — требуют общей защиты

Рекомендуется начать с model-service как наиболее критичного сервиса.

