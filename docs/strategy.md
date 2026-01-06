# Спецификация: Сервис управления капиталом и размером позиций (Capital Management Service)

**Дата создания**: 2026-01-06  
**Статус**: Draft  
**Приоритет**: P1  
**Зависимости**: Model Service, Order Manager, Position Manager

## 1. Обзор

### 1.1. Цель проекта

Создать отдельный сервис **Capital Management Service** для управления размером позиций и распределения капитала. Сервис будет отвечать за:

- Расчет размера позиции (position sizing) на основе управления капиталом
- Распределение капитала между стратегиями и активами
- Управление рисками на уровне портфолио
- Определение размера ордера на основе доступного капитала и риск-профиля

### 1.2. Проблема текущей архитектуры

В текущей архитектуре определение размера позиции (`amount`) смешано с логикой предсказания направления:

#### 1.2.1. Model Service определяет amount

**Проблема**: `IntelligentSignalGenerator` в model-service рассчитывает `amount` на основе:
- `confidence` (уверенность модели)
- `predicted_return` (для регрессионных моделей)
- Фиксированных диапазонов `min_amount` / `max_amount`

**Код проблемы** (`model-service/src/services/intelligent_signal_generator.py:884-949`):

```python
def _calculate_amount(
    self,
    current_price: float,
    confidence: float,
    prediction_result: Optional[Dict[str, Any]] = None,
) -> float:
    """Calculate order amount based on confidence."""
    base_amount = (self.min_amount + self.max_amount) / 2
    
    # For regression models, use predicted_return for position sizing
    if prediction_result is not None:
        prediction = prediction_result.get("prediction")
        if isinstance(prediction, float):
            predicted_return = float(prediction)
            max_expected_return = settings.model_regression_max_expected_return
            return_magnitude = abs(predicted_return)
            return_based_multiplier = min(1.0, return_magnitude / max_expected_return)
            confidence_multiplier = 0.5 + (max(confidence, return_based_multiplier) * 0.5)
    
    amount = base_amount * confidence_multiplier
    amount = max(self.min_amount, min(self.max_amount, amount))
    return round(amount, 2)
```

**Почему это неправильно**:
- Модель должна предсказывать только **направление** (buy/sell), а не размер позиции
- Размер позиции — это задача **управления капиталом**, а не предсказания
- Не учитывается доступный баланс, риск-профиль стратегии, корреляции между активами

#### 1.2.2. Order Manager не пересчитывает amount

**Проблема**: Order Manager только:
- Конвертирует `amount` → `quantity` (через `QuantityCalculator`)
- Ограничивает `quantity` по лимитам (через `RiskManager.check_and_adapt_order_size()`)
- Проверяет баланс на основе уже рассчитанного `quantity`

**Что отсутствует**:
- Пересчет `amount` на основе доступного капитала
- Учет риск-профиля стратегии при определении размера
- Распределение капитала между позициями
- Учет корреляций между активами

#### 1.2.3. Разделение ответственностей нарушено

**Текущее состояние**:
```
Model Service:
  ├─ Предсказывает направление (buy/sell) ✅
  └─ Определяет amount ❌ (не должно)

Order Manager:
  ├─ Конвертирует amount → quantity ✅
  ├─ Ограничивает quantity по лимитам ✅
  └─ Пересчитывает amount на основе капитала ❌ (не делает)
```

**Правильное разделение**:
```
Model Service:
  └─ Предсказывает только направление (buy/sell) ✅

Capital Management Service:
  ├─ Рассчитывает amount на основе:
  │  ├─ Доступного баланса
  │  ├─ Риск-профиля стратегии
  │  ├─ Размера портфолио
  │  ├─ Максимального риска на сделку
  │  └─ Корреляций между позициями
  └─ Возвращает amount для сигнала

Order Manager:
  ├─ Конвертирует amount → quantity ✅
  └─ Ограничивает quantity по лимитам ✅
```

## 2. Требования к сервису

### 2.1. Функциональные требования

#### FR1: Расчет размера позиции на основе управления капиталом

**Описание**: Сервис должен рассчитывать оптимальный размер позиции (`amount` в USDT) на основе:

- **FR1.1**: Доступного баланса (USDT и других валют)
- **FR1.2**: Риск-профиля стратегии (максимальный риск на сделку в %)
- **FR1.3**: Размера портфолио (общий exposure)
- **FR1.4**: Максимального риска на сделку (в % от капитала)
- **FR1.5**: Корреляций между активами (для диверсификации)
- **FR1.6**: Текущих позиций (избегать переэкспозиции)

**Входные данные**:
- `signal_type`: "buy" или "sell"
- `asset`: Trading pair (например, "ETHUSDT")
- `strategy_id`: Идентификатор стратегии
- `confidence`: Уверенность модели (0-1) - опционально, для адаптивного sizing
- `predicted_return`: Предсказанный возврат (для регрессии) - опционально

**Выходные данные**:
- `amount`: Размер позиции в USDT (quote currency)
- `risk_percentage`: Процент риска от капитала
- `reasoning`: Объяснение расчета (для логирования)

#### FR2: Интеграция с существующими сервисами

**FR2.1**: Интеграция с Model Service
- Model Service вызывает Capital Management Service перед созданием `TradingSignal`
- Model Service передает только `signal_type` (buy/sell), без `amount`
- Capital Management Service возвращает `amount`

**FR2.2**: Интеграция с Position Manager
- Получение текущих позиций для расчета exposure
- Получение портфолио-метрик (total_exposure, total_pnl)

**FR2.3**: Интеграция с Order Manager
- Order Manager может вызывать Capital Management Service для пересчета `amount` при недостатке баланса
- Order Manager может использовать сервис для адаптации размера при изменении условий

#### FR3: Управление риск-профилями стратегий

**FR3.1**: Конфигурация риск-профилей
- Максимальный риск на сделку (% от капитала)
- Максимальный размер позиции (% от портфолио)
- Максимальный exposure на актив (% от портфолио)

**FR3.2**: Разные профили для разных стратегий
- Консервативный (низкий риск)
- Умеренный (средний риск)
- Агрессивный (высокий риск)

#### FR4: Адаптивный position sizing

**FR4.1**: Адаптация на основе confidence
- Высокая confidence → больший размер позиции (в пределах лимитов)
- Низкая confidence → меньший размер позиции

**FR4.2**: Адаптация на основе predicted_return (для регрессии)
- Больший predicted_return → больший размер позиции
- Меньший predicted_return → меньший размер позиции

**FR4.3**: Адаптация на основе волатильности
- Высокая волатильность → меньший размер позиции
- Низкая волатильность → больший размер позиции

### 2.2. Нефункциональные требования

#### NFR1: Производительность
- Расчет `amount` должен выполняться за < 100ms (95-й перцентиль)
- Поддержка до 1000 запросов в минуту
- Кэширование данных о балансе и позициях (TTL: 5 секунд)

#### NFR2: Надежность
- Graceful degradation: при недоступности сервиса Model Service использует fallback (фиксированный amount)
- Retry механизм для внешних API вызовов
- Логирование всех расчетов для аудита

#### NFR3: Масштабируемость
- Stateless архитектура (можно горизонтально масштабировать)
- Использование Redis для кэширования
- Асинхронная обработка запросов

## 3. Архитектура

### 3.1. Компоненты сервиса

```
Capital Management Service
├─ API Layer
│  ├─ REST API (FastAPI)
│  └─ gRPC API (опционально, для высокой производительности)
├─ Business Logic
│  ├─ PositionSizeCalculator
│  │  ├─ calculate_amount()
│  │  ├─ calculate_risk_percentage()
│  │  └─ adapt_to_volatility()
│  ├─ CapitalAllocator
│  │  ├─ get_available_capital()
│  │  ├─ allocate_capital()
│  │  └─ check_portfolio_limits()
│  └─ RiskProfileManager
│     ├─ get_risk_profile()
│     ├─ validate_risk_limits()
│     └─ calculate_max_position_size()
├─ Integration Layer
│  ├─ PositionManagerClient
│  ├─ BalanceServiceClient
│  └─ MarketDataClient (для волатильности)
└─ Data Layer
   ├─ RiskProfileRepository
   └─ Cache (Redis)
```

### 3.2. API Endpoints

#### POST `/api/v1/position-size/calculate`

**Описание**: Рассчитывает размер позиции для сигнала

**Request Body**:
```json
{
  "signal_type": "buy",
  "asset": "ETHUSDT",
  "strategy_id": "test-strategy",
  "confidence": 0.85,
  "predicted_return": 0.02,
  "current_price": 3318.84
}
```

**Response**:
```json
{
  "amount": 5000.0,
  "risk_percentage": 2.5,
  "reasoning": {
    "available_capital": 200000.0,
    "risk_per_trade": 2.5,
    "position_size_limit": 0.1,
    "volatility_adjustment": 0.95,
    "confidence_multiplier": 1.0
  },
  "warnings": []
}
```

#### GET `/api/v1/risk-profiles/{strategy_id}`

**Описание**: Получает риск-профиль стратегии

**Response**:
```json
{
  "strategy_id": "test-strategy",
  "max_risk_per_trade": 2.5,
  "max_position_size_pct": 10.0,
  "max_exposure_per_asset_pct": 20.0,
  "profile_type": "moderate"
}
```

#### PUT `/api/v1/risk-profiles/{strategy_id}`

**Описание**: Обновляет риск-профиль стратегии

**Request Body**:
```json
{
  "max_risk_per_trade": 3.0,
  "max_position_size_pct": 15.0,
  "max_exposure_per_asset_pct": 25.0,
  "profile_type": "aggressive"
}
```

### 3.3. Интеграция с Model Service

#### Изменения в Model Service

**Текущий код** (`intelligent_signal_generator.py:487-493`):
```python
# Calculate order amount from model
model_amount = self._calculate_amount(
    current_price, 
    confidence,
    prediction_result=prediction_result
)
```

**Новый код**:
```python
# Get position size from Capital Management Service
from ..services.capital_management_client import capital_management_client

position_size_result = await capital_management_client.calculate_position_size(
    signal_type=signal_type,
    asset=asset,
    strategy_id=strategy_id,
    confidence=confidence,
    predicted_return=prediction_result.get("prediction") if isinstance(prediction_result.get("prediction"), float) else None,
    current_price=current_price,
    trace_id=trace_id,
)

if position_size_result:
    amount = position_size_result.amount
else:
    # Fallback to fixed amount if service unavailable
    logger.warning("Capital Management Service unavailable, using fallback amount")
    amount = self.min_amount
```

#### Удаление метода `_calculate_amount`

Метод `_calculate_amount()` должен быть удален из `IntelligentSignalGenerator`, так как расчет размера позиции теперь выполняется в Capital Management Service.

### 3.4. Интеграция с Order Manager

Order Manager может использовать Capital Management Service для пересчета `amount` при недостатке баланса:

```python
# In signal_processor.py, after balance check fails
if balance_insufficient:
    # Try to recalculate amount with available balance
    recalculated_amount = await capital_management_client.calculate_position_size(
        signal_type=signal.signal_type,
        asset=signal.asset,
        strategy_id=signal.strategy_id,
        available_capital=available_balance,
        trace_id=trace_id,
    )
    if recalculated_amount:
        # Update signal amount and retry
        signal.amount = recalculated_amount.amount
        quantity = await self.quantity_calculator.calculate_quantity(signal)
```

## 4. Алгоритм расчета размера позиции

### 4.1. Базовый алгоритм

```
1. Получить риск-профиль стратегии
   ├─ max_risk_per_trade (% от капитала)
   ├─ max_position_size_pct (% от портфолио)
   └─ max_exposure_per_asset_pct (% от портфолио)

2. Получить доступный капитал
   ├─ USDT balance
   └─ Margin available

3. Рассчитать базовый размер позиции
   amount_base = available_capital * max_risk_per_trade / 100

4. Применить адаптеры
   ├─ Confidence multiplier (0.5 - 1.0)
   ├─ Predicted return multiplier (для регрессии)
   └─ Volatility adjustment (0.8 - 1.0)

5. Проверить лимиты
   ├─ max_position_size_pct
   ├─ max_exposure_per_asset_pct
   └─ min_order_value (из instrument info)

6. Вернуть финальный amount
```

### 4.2. Пример расчета

**Входные данные**:
- `available_capital`: 200,000 USDT
- `max_risk_per_trade`: 2.5%
- `confidence`: 0.85
- `predicted_return`: 0.02 (2%)
- `current_exposure_eth`: 30,000 USDT
- `max_exposure_per_asset_pct`: 20%
- `total_portfolio_value`: 250,000 USDT

**Расчет**:
```
1. Базовый размер: 200,000 * 0.025 = 5,000 USDT

2. Confidence multiplier: 0.5 + (0.85 * 0.5) = 0.925
   Adjusted: 5,000 * 0.925 = 4,625 USDT

3. Predicted return multiplier: min(1.0, 0.02 / 0.05) = 0.4
   (если max_expected_return = 5%)
   Adjusted: 4,625 * 0.4 = 1,850 USDT

4. Проверка лимитов:
   - max_exposure_per_asset: 250,000 * 0.20 = 50,000 USDT
   - Текущий exposure: 30,000 USDT
   - Доступно: 50,000 - 30,000 = 20,000 USDT
   - Final amount: min(1,850, 20,000) = 1,850 USDT

5. Результат: amount = 1,850 USDT
```

## 5. Конфигурация

### 5.1. Настройки сервиса

```python
# .env
CAPITAL_MANAGEMENT_SERVICE_HOST=capital-management
CAPITAL_MANAGEMENT_SERVICE_PORT=4700
CAPITAL_MANAGEMENT_SERVICE_API_KEY=...

# Risk profiles (default)
CAPITAL_MANAGEMENT_DEFAULT_MAX_RISK_PER_TRADE=2.5
CAPITAL_MANAGEMENT_DEFAULT_MAX_POSITION_SIZE_PCT=10.0
CAPITAL_MANAGEMENT_DEFAULT_MAX_EXPOSURE_PER_ASSET_PCT=20.0

# Cache settings
CAPITAL_MANAGEMENT_CACHE_TTL_SECONDS=5
CAPITAL_MANAGEMENT_CACHE_ENABLED=true
```

### 5.2. База данных

**Таблица `risk_profiles`**:
```sql
CREATE TABLE risk_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(100) NOT NULL UNIQUE,
    max_risk_per_trade DECIMAL(5, 2) NOT NULL DEFAULT 2.5,
    max_position_size_pct DECIMAL(5, 2) NOT NULL DEFAULT 10.0,
    max_exposure_per_asset_pct DECIMAL(5, 2) NOT NULL DEFAULT 20.0,
    profile_type VARCHAR(20) NOT NULL DEFAULT 'moderate',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## 6. Миграция

### 6.1. Этап 1: Создание сервиса

1. Создать структуру проекта `capital-management/`
2. Настроить Docker, docker-compose
3. Реализовать базовую инфраструктуру (logging, database, Redis)
4. Создать API endpoints
5. Реализовать алгоритм расчета размера позиции

### 6.2. Этап 2: Интеграция с Model Service

1. Создать `CapitalManagementClient` в model-service
2. Заменить `_calculate_amount()` на вызов Capital Management Service
3. Добавить fallback на фиксированный amount при недоступности сервиса
4. Протестировать интеграцию

### 6.3. Этап 3: Удаление старого кода

1. Удалить метод `_calculate_amount()` из `IntelligentSignalGenerator`
2. Удалить параметры `min_amount`, `max_amount` из конструктора
3. Обновить документацию

### 6.4. Этап 4: Интеграция с Order Manager

1. Добавить опциональную интеграцию в Order Manager для пересчета amount
2. Протестировать сценарии недостатка баланса

## 7. Тестирование

### 7.1. Unit тесты

- Расчет размера позиции с различными входными данными
- Проверка лимитов (max_position_size, max_exposure)
- Адаптация на основе confidence и predicted_return

### 7.2. Integration тесты

- Интеграция с Model Service (полный цикл генерации сигнала)
- Интеграция с Position Manager (получение позиций и exposure)
- Интеграция с Balance Service (получение баланса)

### 7.3. E2E тесты

- Генерация сигнала с расчетом amount через Capital Management Service
- Обработка недоступности сервиса (fallback)
- Пересчет amount при недостатке баланса

## 8. Мониторинг и метрики

### 8.1. Метрики

- `capital_management_calculate_duration_seconds` - время расчета amount
- `capital_management_requests_total` - общее количество запросов
- `capital_management_errors_total` - количество ошибок
- `capital_management_fallback_used_total` - использование fallback

### 8.2. Логирование

- Логировать все расчеты amount с reasoning
- Логировать использование лимитов
- Логировать предупреждения (warnings)

## 9. Дальнейшее развитие

### 9.1. Продвинутые функции

- Kelly Criterion для оптимизации размера позиции
- Корреляционный анализ для диверсификации
- Динамическая адаптация риск-профилей на основе производительности
- Machine Learning для оптимизации position sizing

### 9.2. Интеграции

- Интеграция с внешними источниками данных о волатильности
- Интеграция с системами управления рисками
- Интеграция с портфолио-аналитикой

## 10. Ссылки

- [Model Service Signal Generation](../model-service/src/services/intelligent_signal_generator.py)
- [Order Manager Risk Manager](../order-manager/src/services/risk_manager.py)
- [Position Manager Service](../docs/position-manager.md)

