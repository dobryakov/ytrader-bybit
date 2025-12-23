# Dashboard Frontend

React дашборд для визуализации данных торговой системы YTrader.

## Технологии

- React 18 + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui компоненты
- Recharts для графиков
- React Query для управления состоянием
- Axios для HTTP запросов

## Разработка

### Локальная разработка

```bash
# Установка зависимостей
npm install

# Запуск dev сервера с hot reload
npm run dev

# Сборка для production
npm run build

# Просмотр production сборки
npm run preview
```

### Разработка в Docker (с live reload)

По умолчанию используется `Dockerfile.dev` для разработки с live reload:

```bash
# Запуск в dev режиме (с live reload)
docker compose up -d dashboard-frontend

# Просмотр логов
docker compose logs -f dashboard-frontend

# Перезапуск при изменениях (автоматически через volumes)
docker compose restart dashboard-frontend
```

При изменении файлов в `src/`, `public/` или конфигурационных файлах Vite автоматически перезагрузит приложение.

### Production сборка

Для production используйте `Dockerfile.prod`:

```bash
# В docker-compose.yml измените:
# dockerfile: Dockerfile.dev -> dockerfile: Dockerfile.prod
docker compose build dashboard-frontend
docker compose up -d dashboard-frontend
```

## Переменные окружения

В dev режиме через Docker используется прокси `/api`, который автоматически перенаправляется на `dashboard-api:4050`.

Переменные окружения для Docker (в `docker-compose.yml`):

```env
VITE_API_KEY=your-api-key-here  # API ключ для аутентификации
```

**Важно:** `VITE_API_URL` не нужно устанавливать в dev режиме. Фронтенд использует относительный путь `/api`, который проксируется через Vite на бэкенд. В production режиме может потребоваться установить `VITE_API_URL` на полный URL бэкенда.

## Структура

```
src/
├── components/
│   ├── ui/          # shadcn/ui компоненты
│   ├── charts/      # Компоненты графиков
│   ├── tables/      # Компоненты таблиц
│   └── metrics/     # Компоненты метрик
├── pages/           # Страницы приложения
├── hooks/           # React Query хуки
└── lib/             # Утилиты и конфигурация
```

## Live Reload

Frontend настроен для работы с live reload в Docker:

- Файлы монтируются как volumes для мгновенного отображения изменений
- Vite HMR (Hot Module Replacement) работает через WebSocket
- Polling включен для надежной работы с Docker volumes
- Порт 4051 проброшен для доступа из браузера

Изменения в `src/` применяются автоматически без пересборки контейнера.

