# План диагностики и улучшения качества модели

## Проблема

Модель имеет ROC AUC > 0.6, что указывает на способность к ранжированию, но на этапе инференса сигналы почти полностью отсекаются из-за:

1. **min_confidence_threshold = 0.6** - высокий порог уверенности
2. **min_probability_diff** (гистерезис) - требование минимальной разницы между buy/sell вероятностями
3. **Отсутствие анализа score без порогов** - нет понимания, где именно теряется предсказательная сила

## Цель

Понять, есть ли предсказательная сила в модели до применения пост-фильтров и где именно она теряется. Перейти от оценки модели как классификатора к оценке как ranking-модели.

---

## Фаза 1: Базовая диагностика (Приоритет: Высокий)

### 1.1. Сохранение raw probabilities/scores на test split

**Задача:** Сохранить все raw предсказания модели (probabilities для каждого класса) на test split для последующего анализа.

**Реализация:**
- Добавить сохранение probabilities в БД или файлы при оценке на test split
- Формат данных:
  ```python
  {
      "model_version": str,
      "dataset_id": str,
      "split": "test",
      "timestamp": datetime,
      "predictions": [
          {
              "y_true": int,  # истинный класс
              "y_pred": int,  # предсказанный класс
              "probabilities": [float, float, float],  # [P(-1), P(0), P(1)] или [P(0), P(1)] для binary
              "confidence": float,  # max(probabilities)
              "buy_probability": float,
              "sell_probability": float,
              "hold_probability": float,  # для 3-классовой
          },
          ...
      ]
  }
  ```

**Место реализации:**
- `model-service/src/services/training_orchestrator.py` - в методе `_train_model_async` после оценки на test split
- Создать новый репозиторий `ModelPredictionRepository` или расширить `ModelQualityMetricsRepository`

**Формат хранения:**
- Вариант 1: PostgreSQL таблица `model_predictions` (JSONB для probabilities)
- Вариант 2: Parquet файлы в S3/локальном хранилище
- Вариант 3: Оба варианта (БД для быстрого доступа, файлы для глубокого анализа)

### 1.2. Расчет baseline метрик (majority class)

**Задача:** Посчитать метрики для baseline стратегии, которая всегда предсказывает majority class.

**Реализация:**
- Добавить метод `calculate_baseline_metrics()` в `QualityEvaluator`
- Baseline стратегия:
  - Найти majority class в test split
  - Всегда предсказывать этот класс
  - Посчитать accuracy, precision, recall, F1, balanced_accuracy
- Сохранить baseline метрики в `model_quality_metrics` с `metric_name="baseline_*"`

**Метрики для сравнения:**
- `baseline_accuracy` - accuracy majority class стратегии
- `baseline_precision` - precision для majority class
- `baseline_recall` - recall для majority class (должен быть 1.0 для majority class, 0.0 для остальных)
- `baseline_f1_score` - F1 для majority class
- `baseline_balanced_accuracy` - balanced accuracy (должна быть близка к 1/n_classes)

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - новый метод
- `model-service/src/services/training_orchestrator.py` - вызов после оценки на test

### 1.3. Анализ edge в top-k без фильтров

**Задача:** Проверить, есть ли положительный edge в top-k% предсказаний (например, top-20%) без применения confidence threshold и hysteresis.

**Реализация:**
- Добавить метод `analyze_top_k_performance()` в `QualityEvaluator`
- Алгоритм:
  1. Отсортировать все предсказания на test по confidence (max probability) по убыванию
  2. Взять top-k% (k = 10, 20, 30, 50)
  3. Посчитать метрики только для top-k%:
     - Accuracy
     - Precision, Recall, F1 для каждого класса
     - Win rate (если есть исторические данные о PnL)
  4. Сравнить с baseline и с полным test set

**Метрики:**
- `top_k_accuracy` - accuracy в top-k%
- `top_k_precision_class_{label}` - precision для каждого класса в top-k%
- `top_k_recall_class_{label}` - recall для каждого класса в top-k%
- `top_k_f1_score` - F1 в top-k%
- `top_k_lift` - отношение accuracy в top-k% к baseline accuracy
- `top_k_coverage` - доля samples в top-k% от общего количества

**Параметры:**
- k_values = [10, 20, 30, 50] - разные проценты для анализа
- Сохранить метрики для каждого k в `model_quality_metrics`

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - новый метод
- `model-service/src/services/training_orchestrator.py` - вызов после сохранения predictions

---

## Фаза 2: Расширенная диагностика (Приоритет: Средний)

### 2.1. Метрики ранжирования (Ranking Metrics)

**Задача:** Оценить модель как ranking-систему, а не только как классификатор.

**Реализация:**
- Добавить методы для расчета ranking метрик:
  - **Precision@K** - precision для top-K предсказаний
  - **Recall@K** - recall для top-K предсказаний
  - **NDCG@K** (Normalized Discounted Cumulative Gain) - для оценки качества ранжирования
  - **MAP** (Mean Average Precision) - средняя precision по всем классам
  - **Lift curves** - график lift в зависимости от процента выборки

**Метрики:**
- `precision_at_k` для k = [10, 20, 30, 50, 100]
- `recall_at_k` для k = [10, 20, 30, 50, 100]
- `ndcg_at_k` для k = [10, 20, 30, 50, 100]
- `mean_average_precision` - MAP для всех классов

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - новые методы
- Использовать библиотеки: `sklearn.metrics` или специализированные для ranking

### 2.2. Анализ калибровки вероятностей

**Задача:** Проверить, насколько хорошо откалиброваны вероятности модели.

**Реализация:**
- Добавить методы для анализа калибровки:
  - **Calibration plots** - график predicted probability vs actual frequency
  - **Brier score** - мера калибровки (меньше = лучше)
  - **Expected Calibration Error (ECE)** - средняя разница между predicted и actual
  - **Reliability diagrams** - визуализация калибровки

**Метрики:**
- `brier_score` - для каждого класса и средний
- `expected_calibration_error` - ECE для каждого класса
- `calibration_slope` - наклон калибровочной кривой (1.0 = идеально)

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - новые методы
- Использовать `sklearn.calibration` для расчета метрик

### 2.3. Поэтапный анализ потерь сигналов

**Задача:** Понять, на каком этапе фильтрации теряется больше всего сигналов и предсказательной силы.

**Реализация:**
- Посчитать метрики на каждом этапе фильтрации:
  1. **Без фильтров** - все предсказания модели
  2. **После min_confidence_threshold** - только предсказания с confidence >= threshold
  3. **После min_probability_diff** - только предсказания с достаточной разницей вероятностей
  4. **После обоих фильтров** - финальные сигналы

**Метрики для каждого этапа:**
- Количество сигналов (абсолютное и процент от test set)
- Accuracy, Precision, Recall, F1
- Coverage - доля samples, прошедших фильтр
- Lift относительно baseline

**Визуализация:**
- Таблица с метриками на каждом этапе
- График потерь сигналов на каждом этапе

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - метод `analyze_filtering_stages()`
- `model-service/src/services/training_orchestrator.py` - вызов после сохранения predictions

---

## Фаза 3: Анализ по классам и торговые метрики (Приоритет: Низкий)

### 3.1. Детальный анализ по классам

**Задача:** Понять, для каких классов модель работает лучше/хуже.

**Реализация:**
- Для каждого класса отдельно:
  - Precision, Recall, F1 в top-k%
  - Распределение вероятностей для правильных/неправильных предсказаний
  - Confusion matrix для top-k%
  - Анализ ошибок (какие классы путаются чаще всего)

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - метод `analyze_per_class_metrics()`

### 3.2. Trading-метрики на top-k (если доступны исторические данные)

**Задача:** Если есть исторические данные о PnL, посчитать торговые метрики для top-k% без фильтров.

**Реализация:**
- Для top-k% предсказаний:
  - Win rate
  - Average PnL
  - Sharpe ratio
  - Profit factor
  - Max drawdown

**Условие:** Требуется связь между предсказаниями и историческими сделками (через timestamp или signal_id).

**Место реализации:**
- `model-service/src/services/quality_evaluator.py` - расширить `calculate_trading_metrics()`

---

## Технические детали реализации

### Структура данных для сохранения predictions

**Вариант 1: PostgreSQL (рекомендуется для быстрого доступа)**

```sql
CREATE TABLE model_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(255) NOT NULL,
    dataset_id UUID NOT NULL,
    split VARCHAR(50) NOT NULL,  -- 'train', 'validation', 'test'
    training_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Массив предсказаний (JSONB)
    predictions JSONB NOT NULL,
    -- Формат: [{"y_true": -1, "y_pred": 1, "probabilities": [0.2, 0.3, 0.5], ...}, ...]
    
    -- Метаданные
    metadata JSONB,
    
    INDEX idx_model_version (model_version),
    INDEX idx_dataset_id (dataset_id),
    INDEX idx_split (split)
);
```

**Вариант 2: Parquet файлы (для глубокого анализа)**

- Сохранять в `data/predictions/{model_version}/{dataset_id}/{split}.parquet`
- Использовать pandas для записи/чтения

### Новые методы в QualityEvaluator

```python
class QualityEvaluator:
    # Фаза 1
    def calculate_baseline_metrics(self, y_true: pd.Series) -> Dict[str, float]
    def analyze_top_k_performance(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series, 
        y_pred_proba: np.ndarray,
        k_values: List[int] = [10, 20, 30, 50]
    ) -> Dict[str, Any]
    
    # Фаза 2
    def calculate_ranking_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        k_values: List[int] = [10, 20, 30, 50, 100]
    ) -> Dict[str, float]
    
    def analyze_calibration(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]
    
    def analyze_filtering_stages(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: np.ndarray,
        confidence_threshold: float = 0.6,
        min_probability_diff: float = 0.05
    ) -> Dict[str, Any]
    
    # Фаза 3
    def analyze_per_class_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: np.ndarray,
        top_k_percent: Optional[int] = None
    ) -> Dict[str, Any]
```

### Интеграция в TrainingOrchestrator

В методе `_train_model_async` после оценки на test split:

```python
# После test_metrics = quality_evaluator.evaluate(...)

# 1. Сохранить predictions
await self._save_test_predictions(
    model_version=version,
    dataset_id=dataset_id,
    y_true=test_labels,
    y_pred=test_y_pred,
    y_pred_proba=test_y_pred_proba,
    model=model,
    task_type=task_type,
    task_variant=task_variant,
)

# 2. Посчитать baseline
baseline_metrics = quality_evaluator.calculate_baseline_metrics(test_labels)
await self._save_metrics(model_version_id, baseline_metrics, "test")

# 3. Анализ top-k
top_k_results = quality_evaluator.analyze_top_k_performance(
    y_true=test_labels,
    y_pred=pd.Series(test_y_pred),
    y_pred_proba=test_y_pred_proba,
    k_values=[10, 20, 30, 50]
)
await self._save_top_k_metrics(model_version_id, top_k_results, "test")

# 4. (Фаза 2) Ranking метрики
ranking_metrics = quality_evaluator.calculate_ranking_metrics(
    y_true=test_labels,
    y_pred_proba=test_y_pred_proba,
    k_values=[10, 20, 30, 50, 100]
)
await self._save_metrics(model_version_id, ranking_metrics, "test")

# 5. (Фаза 2) Анализ калибровки
calibration_metrics = quality_evaluator.analyze_calibration(
    y_true=test_labels,
    y_pred_proba=test_y_pred_proba
)
await self._save_metrics(model_version_id, calibration_metrics, "test")

# 6. (Фаза 2) Поэтапный анализ фильтрации
filtering_analysis = quality_evaluator.analyze_filtering_stages(
    y_true=test_labels,
    y_pred=pd.Series(test_y_pred),
    y_pred_proba=test_y_pred_proba,
    confidence_threshold=settings.model_activation_threshold,
    min_probability_diff=settings.model_min_probability_diff
)
await self._save_filtering_analysis(model_version_id, filtering_analysis, "test")
```

---

## Ожидаемые результаты

После реализации Фазы 1 мы получим:

1. **Понимание предсказательной силы модели:**
   - Есть ли edge в top-20% предсказаний?
   - Насколько модель лучше baseline?

2. **Где теряется сигнал:**
   - Сколько сигналов отсекается на каждом этапе фильтрации?
   - Какие метрики ухудшаются после применения фильтров?

3. **Рекомендации по оптимизации:**
   - Нужно ли снижать min_confidence_threshold?
   - Нужно ли уменьшать min_probability_diff?
   - Стоит ли использовать адаптивные пороги в зависимости от score?

---

## Порядок реализации

1. **Фаза 1.1** - Сохранение predictions (критично для всего остального)
2. **Фаза 1.2** - Baseline метрики
3. **Фаза 1.3** - Top-k анализ
4. **Фаза 2.1** - Ranking метрики
5. **Фаза 2.2** - Анализ калибровки
6. **Фаза 2.3** - Поэтапный анализ фильтрации
7. **Фаза 3** - Дополнительные анализы (по необходимости)

---

## Примечания

- Все метрики должны сохраняться в `model_quality_metrics` с соответствующими `metric_name` и `dataset_split="test"`
- Для визуализации можно использовать существующие инструменты или добавить простые графики через matplotlib
- Анализ должен запускаться автоматически после каждого обучения модели на test split
- Результаты должны быть доступны через API или логи для мониторинга

