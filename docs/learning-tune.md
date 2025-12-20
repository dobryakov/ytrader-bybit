# Настройка обучения моделей: решение проблемы дисбаланса классов

## Проблема

При обучении моделей наблюдается сильный дисбаланс классов в обучающем датасете:

- **Train dataset (после удаления нулевого класса):**
  - Класс `1` (BUY): 5770 (82.05%)
  - Класс `-1` (SELL): 1150 (16.35%)
  - Соотношение: ~5:1

- **Test dataset:**
  - Класс `-1` (SELL): 668 (52.85%)
  - Класс `1` (BUY): 596 (47.15%)
  - Соотношение: ~1.1:1

**Ключевая проблема:** Несоответствие распределений между train и test датасетами приводит к низкой accuracy (15.7%) и плохому обобщению модели.

## Текущая реализация

### 1. Динамический `scale_pos_weight` для бинарной классификации

**Местоположение:** `model-service/src/services/model_trainer.py:522-548`

**Формула:** `scale_pos_weight = n_neg / n_pos` (с ограничением до 20.0)

**Проблема:** После ремаппинга `{-1, 1} → {0, 1}`:
- Класс `0` (бывший `-1`) = 1150
- Класс `1` (бывший `1`) = 5770
- `scale_pos_weight = 1150 / 5770 ≈ 0.2`

Это **увеличивает вес majority класса**, что не решает проблему дисбаланса.

### 2. SMOTE (отключен)

**Конфигурация:** `MODEL_TRAINING_USE_SMOTE=false`

**Статус:** Пробовали, не помогло.

### 3. Метод весов классов

**Конфигурация:** `MODEL_TRAINING_CLASS_WEIGHT_METHOD=inverse_frequency`

**Применение:** Только для мультиклассовой классификации (3+ классов), не используется для бинарной.

---

## Варианты решения

### 1. Undersampling Majority Class

**Идея:** Уменьшить количество примеров мажоритарного класса до уровня минорного.

**Методы:**
- **Случайное удаление:** Удалить случайные примеры класса `1` до ~1150
- **Стратифицированное удаление:** Сохранить временную структуру данных
- **Tomek Links:** Удалить примеры, которые находятся близко к границе классов

**Плюсы:**
- Быстрее обучение (меньше данных)
- Меньше переобучения на мажоритарном классе
- Простая реализация

**Минусы:**
- Потеря данных (теряем 4620 примеров класса `1`)
- Может ухудшить качество, если мажоритарный класс важен

**Реализация:**
```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Случайное undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Tomek Links (удаление примеров на границе)
tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X, y)
```

**Конфигурация:**
```yaml
# model-service/config/model_hyperparams.yaml
model_training_use_undersampling: true
model_training_undersampling_method: "random"  # "random" | "tomek"
```

---

### 2. Простое Oversampling (без SMOTE)

**Идея:** Дублировать примеры минорного класса.

**Методы:**
- **Случайное дублирование:** Дублировать случайные примеры класса `-1` до ~5770
- **Стратифицированное дублирование:** Дублировать по временным периодам
- **ADASYN:** Адаптивный синтез (более умный, чем простое дублирование)

**Плюсы:**
- Не теряем данные
- Проще, чем SMOTE
- Быстрая реализация

**Минусы:**
- Переобучение на дублированных примерах
- Может не помочь, если данных минорного класса мало

**Реализация:**
```python
from imblearn.over_sampling import RandomOverSampler, ADASYN

# Случайное oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# ADASYN (адаптивный синтез)
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

**Конфигурация:**
```yaml
model_training_use_oversampling: true
model_training_oversampling_method: "random"  # "random" | "adasyn"
```

---

### 3. Настройка порога классификации (Threshold Tuning)

**Идея:** Не использовать порог 0.5, а подобрать оптимальный по метрике.

**Текущая ситуация:**
В логах видно калибровку порогов:
```
thresholds={-1: 0.9693442583084106, 1: 0.14441588521003723}
```

**Методы:**
- **Подбор по PR-AUC:** Найти порог, максимизирующий PR-AUC на validation
- **Подбор по F1-score:** Найти порог, максимизирующий F1-score
- **ROC-кривая:** Использовать ROC-кривую для выбора оптимального порога
- **Калибровка вероятностей:** Platt scaling, Isotonic regression

**Плюсы:**
- Не меняет данные
- Быстро (пост-обработка)
- Улучшает precision/recall баланс

**Минусы:**
- Не решает проблему обучения на несбалансированных данных
- Требует validation set для подбора

**Реализация:**
```python
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

# Подбор порога по PR-AUC
y_pred_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Использование оптимального порога
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
```

**Конфигурация:**
```yaml
model_training_threshold_tuning: true
model_training_threshold_metric: "f1"  # "f1" | "pr_auc" | "roc_auc"
```

---

### 4. Изменение метрики оптимизации

**Идея:** Оптимизировать метрику, устойчивую к дисбалансу.

**Текущая конфигурация:**
```yaml
# model-service/config/model_hyperparams.yaml
eval_metric: aucpr  # PR-AUC уже используется
```

**Дополнительные варианты:**
- **F1-score (macro/micro):** Устойчив к дисбалансу
- **Balanced accuracy:** Среднее recall по классам
- **Matthews Correlation Coefficient (MCC):** Учитывает все элементы confusion matrix
- **Custom loss function:** Focal loss

**Реализация:**
```python
# Изменить eval_metric в XGBoost
hyperparameters = {
    "eval_metric": "aucpr",  # Уже используется
    # Или использовать custom objective для focal loss
}
```

**Конфигурация:**
```yaml
eval_metric: aucpr  # Текущее значение
# Альтернативы: "logloss", "error", "merror", "mlogloss"
```

---

### 5. Focal Loss

**Идея:** Фокус на сложных примерах, автоматическое снижение веса легких.

**Формула:** `FL = -α(1-p)^γ * log(p)`

Где:
- `α` — баланс классов (аналог `scale_pos_weight`)
- `γ` — фокус на сложных примерах (focusing parameter)
- `p` — предсказанная вероятность

**Плюсы:**
- Эффективно для дисбаланса
- Автоматически фокусируется на сложных примерах
- Не требует изменения данных

**Минусы:**
- Требует реализации custom objective для XGBoost
- Нужна настройка гиперпараметров `α` и `γ`

**Реализация:**
```python
import xgboost as xgb
import numpy as np

def focal_loss_objective(y_true, y_pred):
    """
    Focal Loss для XGBoost.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные вероятности (raw scores)
    
    Returns:
        grad, hess
    """
    alpha = 0.25  # Баланс классов
    gamma = 2.0   # Focusing parameter
    
    # Преобразуем raw scores в вероятности
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Вычисляем градиент и гессиан
    grad = alpha * (1 - p) ** gamma * (y_true - p) * (gamma * np.log(p) + 1)
    hess = alpha * (1 - p) ** gamma * p * (1 - p) * (gamma * np.log(p) + 2)
    
    return grad, hess

# Использование в XGBoost
model = xgb.XGBClassifier(
    objective=focal_loss_objective,
    eval_metric="aucpr",
    ...
)
```

**Конфигурация:**
```yaml
model_training_use_focal_loss: true
model_training_focal_loss_alpha: 0.25
model_training_focal_loss_gamma: 2.0
```

---

### 6. Изменение вычисления Target (Target Computation)

**Идея:** Проблема может быть в том, как формируется target.

**Текущая конфигурация:**
```yaml
# feature-service/config/versions/target_registry_v1.6.0.yaml
threshold: null  # Нет dead-zone для flat
task_variant: "binary_classification"
min_positive_support: 0.05
min_negative_support: 0.05
horizon: 1800  # 30 минут
```

**Варианты:**
1. **Добавить dead-zone для flat:**
   ```yaml
   threshold: 0.001  # Порог для нейтрального класса
   ```

2. **Изменить минимальную поддержку классов:**
   ```yaml
   min_positive_support: 0.15  # Увеличить минимальную поддержку
   min_negative_support: 0.15
   ```

3. **Изменить горизонт предсказания:**
   ```yaml
   horizon: 3600  # 1 час вместо 30 минут
   ```

4. **Изменить метод нормализации:**
   ```yaml
   normalize: "zscore"  # Вместо "sharpe"
   ```

**Плюсы:**
- Решает проблему на уровне данных
- Может улучшить качество target'ов

**Минусы:**
- Требует пересборки датасетов
- Может изменить семантику задачи

**Реализация:**
Изменить `feature-service/config/versions/target_registry_v1.6.0.yaml`:
```yaml
version: "1.6.0"
config:
  type: "classification"
  horizon: 1800
  threshold: 0.001  # Добавить dead-zone
  computation:
    preset: "next_candle_direction"
    options:
      min_positive_support: 0.15  # Увеличить
      min_negative_support: 0.15  # Увеличить
      task_variant: "binary_classification"
```

---

### 7. Стратифицированная выборка для Train/Test Split

**Идея:** Обеспечить одинаковое распределение классов в train и test.

**Проблема:**
- Train: 82% vs 16%
- Test: 52.85% vs 47.15%

**Методы:**
- **Стратифицированный split по классам:** Обеспечить одинаковое распределение
- **Временная стратификация:** Сохранить временной порядок, но балансировать классы

**Плюсы:**
- Решает проблему несоответствия распределений
- Улучшает обобщение

**Минусы:**
- Требует изменения логики split'а в Feature Service
- Может быть сложно для временных данных

**Реализация:**
В `feature-service/src/services/optimized_dataset/optimized_builder.py`:
```python
from sklearn.model_selection import train_test_split

# Стратифицированный split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Стратификация по классам
    random_state=42
)
```

---

### 8. Ensemble методы с балансировкой

**Идея:** Несколько моделей на сбалансированных подвыборках.

**Методы:**
- **Bagging с undersampling:** Каждая модель на сбалансированной выборке
- **Easy Ensemble:** Несколько моделей на разных undersampled выборках
- **Balance Cascade:** Последовательное обучение с удалением правильно классифицированных примеров

**Плюсы:**
- Комбинирует преимущества нескольких подходов
- Улучшает стабильность

**Минусы:**
- Сложная реализация
- Медленнее обучение (несколько моделей)

**Реализация:**
```python
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier

# Easy Ensemble
easy_ensemble = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
)
easy_ensemble.fit(X_train, y_train)

# Balanced Bagging
balanced_bagging = BalancedBaggingClassifier(
    n_estimators=10,
    random_state=42
)
balanced_bagging.fit(X_train, y_train)
```

---

### 9. Cost-Sensitive Learning

**Идея:** Явно задать стоимость ошибок для разных классов.

**Текущая реализация:**
- `scale_pos_weight` вычисляется динамически (~0.2)
- Это **увеличивает вес majority класса**, что неправильно

**Варианты:**
1. **Увеличить `scale_pos_weight` вручную:**
   ```yaml
   # model-service/config/model_hyperparams.yaml
   scale_pos_weight: 5.0  # Вместо динамического 0.2
   ```

2. **Использовать `sample_weight` для бинарной классификации:**
   Сейчас `sample_weight` используется только для multi-class.

3. **Добавить параметр `cost_matrix`:**
   ```yaml
   cost_matrix:
     false_positive: 1.0   # Стоимость FP
     false_negative: 5.0    # Стоимость FN (выше, т.к. минорный класс)
   ```

**Плюсы:**
- Простая реализация
- Прямой контроль над балансом

**Минусы:**
- Требует ручной настройки
- Может привести к переобучению

**Реализация:**
```python
# В model_trainer.py для бинарной классификации
if len(unique_labels) == 2:
    # Использовать sample_weight вместо scale_pos_weight
    class_weights = {
        0: 5.0,  # Минорный класс (SELL) - больший вес
        1: 1.0   # Мажоритарный класс (BUY) - меньший вес
    }
    sample_weight = y.map(class_weights).values
    # Использовать sample_weight в model.fit()
```

**Конфигурация:**
```yaml
# model-service/config/model_hyperparams.yaml
binary_classification:
  common:
    scale_pos_weight: 5.0  # Фиксированное значение вместо динамического
    # Или использовать sample_weight
    use_sample_weight_for_binary: true
```

---

### 10. Изменение метода весов классов

**Текущая конфигурация:**
```python
MODEL_TRAINING_CLASS_WEIGHT_METHOD=inverse_frequency
```

**Варианты:**
1. **`balanced` (sklearn):** Более агрессивная балансировка
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       "balanced",
       classes=np.array(sorted(unique_labels)),
       y=y.values
   )
   ```

2. **Custom weights:** Явно задать веса для каждого класса
   ```yaml
   model_training_class_weights:
     -1: 5.0  # SELL - больший вес
     1: 1.0   # BUY - меньший вес
   ```

3. **Адаптивные веса:** Менять веса в процессе обучения

**Реализация:**
```python
# В model_trainer.py
if weight_method == "custom":
    # Читать из конфигурации
    custom_weights = settings.model_training_custom_class_weights
    class_weights = {label: custom_weights.get(label, 1.0) for label in unique_labels}
```

**Конфигурация:**
```yaml
MODEL_TRAINING_CLASS_WEIGHT_METHOD=custom
MODEL_TRAINING_CUSTOM_CLASS_WEIGHTS='{"-1": 5.0, "1": 1.0}'
```

---

## Рекомендации по приоритетам

### Высокий приоритет (быстро, эффективно):

1. **Настройка порога классификации** ⭐⭐⭐
   - Быстро (пост-обработка)
   - Не требует изменения данных
   - Может сразу улучшить метрики

2. **Cost-Sensitive Learning** ⭐⭐⭐
   - Простая реализация
   - Прямой контроль над балансом
   - Исправить текущую логику `scale_pos_weight`

3. **Изменение Target Computation** ⭐⭐
   - Может решить проблему на уровне данных
   - Требует пересборки датасетов

### Средний приоритет:

4. **Undersampling Majority Class** ⭐⭐
   - Если данных достаточно
   - Простая реализация

5. **Focal Loss** ⭐⭐
   - Эффективно, но требует реализации custom objective

### Низкий приоритет (сложно, долго):

6. **Ensemble методы** ⭐
   - Сложная реализация
   - Медленнее обучение

7. **Стратифицированная выборка** ⭐
   - Требует изменения логики split'а в Feature Service

---

## Вопросы для уточнения

1. **Почему train и test имеют разное распределение?**
   - Это может быть основной проблемой
   - Нужно проверить логику split'а в Feature Service

2. **Можно ли изменить вычисление target?**
   - Добавить dead-zone для flat?
   - Изменить пороги `min_positive_support` / `min_negative_support`?

3. **Какой метрикой важнее оптимизировать?**
   - Precision (меньше ложных сигналов)?
   - Recall (больше правильных сигналов)?
   - F1-score (баланс)?
   - PR-AUC (уже используется)?

---

## План действий

### Этап 1: Быстрые улучшения (1-2 дня)

1. ✅ Исправить логику `scale_pos_weight` для бинарной классификации
   - Использовать `sample_weight` вместо `scale_pos_weight`
   - Или инвертировать формулу: `scale_pos_weight = n_pos / n_neg` (вместо `n_neg / n_pos`)

2. ✅ Реализовать настройку порога классификации
   - Подбор оптимального порога по F1-score на validation

### Этап 2: Среднесрочные улучшения (3-5 дней)

3. ✅ Добавить поддержку undersampling/oversampling
   - Реализовать в `model_trainer.py`
   - Добавить конфигурацию

4. ✅ Изменить Target Computation
   - Добавить dead-zone для flat
   - Увеличить `min_positive_support` / `min_negative_support`

### Этап 3: Долгосрочные улучшения (1-2 недели)

5. ✅ Реализовать Focal Loss
   - Custom objective для XGBoost
   - Настройка гиперпараметров

6. ✅ Исправить стратифицированный split
   - Обеспечить одинаковое распределение в train/test

---

## Ссылки на код

- **Model Trainer:** `model-service/src/services/model_trainer.py`
- **Target Registry:** `feature-service/config/versions/target_registry_v1.6.0.yaml`
- **Model Hyperparameters:** `model-service/config/model_hyperparams.yaml`
- **Settings:** `model-service/src/config/settings.py`

---

## История изменений

- **2025-12-19:** Создан документ с описанием всех вариантов решения проблемы дисбаланса классов
- SMOTE пробовали, не помогло
- Текущая проблема: train (82% vs 16%) vs test (52.85% vs 47.15%)

