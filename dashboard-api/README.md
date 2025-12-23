# Dashboard API

FastAPI сервис для предоставления данных из БД для дашборда.

## Технологии

- FastAPI
- asyncpg (PostgreSQL)
- Pydantic для валидации
- structlog для логирования

## Разработка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск dev сервера
uvicorn src.main:app --reload --host 0.0.0.0 --port 4050
```

## Переменные окружения

Настройте переменные в `.env`:

```env
DASHBOARD_API_PORT=4050
DASHBOARD_API_KEY=your-api-key-here
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=your-password
```

## API Endpoints

- `GET /api/v1/positions` - список позиций
- `GET /api/v1/positions/{asset}` - позиция по asset
- `GET /api/v1/orders` - список ордеров
- `GET /api/v1/signals` - список сигналов
- `GET /api/v1/models` - список моделей
- `GET /api/v1/metrics/overview` - общие метрики
- `GET /api/v1/metrics/portfolio` - метрики портфолио
- `GET /api/v1/charts/pnl` - данные для графика PnL
- `GET /api/v1/charts/signals-confidence` - график confidence сигналов

## Аутентификация

Все `/api/*` endpoints требуют заголовок `X-API-Key` с валидным ключом.

