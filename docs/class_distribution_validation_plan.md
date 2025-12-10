# План реализации проверки распределения классов в датасете

## Цель
Добавить автоматическую проверку распределения классов в датасете перед обучением модели, чтобы выявлять критические дисбалансы классов на раннем этапе и предотвращать обучение моделей на некачественных данных.

## Проблема
Текущая ситуация:
- Модели могут обучаться на датасетах с критическим дисбалансом классов (например, 100% класс 0)
- Это приводит к обманчиво высокой точности (accuracy ~1.0), но модель просто предсказывает один класс
- Проблема обнаруживается только после обучения, что тратит время и ресурсы

## Текущее состояние (на основе последнего обучения)
**Датасет:** `8b39b294-b70c-4a70-8868-e0ae51885bf4` (2025-12-09)

**Распределение классов:**
- **Train split:** Класс 0 (flat): 91.49%, Класс -1 (down): 4.42%, Класс 1 (up): 4.09%
- **Validation split:** Класс 0 (flat): 91.79%, Класс -1 (down): 4.27%, Класс 1 (up): 3.95%
- **Test split:** Класс 0 (flat): 94.30%, Класс -1 (down): 3.06%, Класс 1 (up): 2.64%
- **Соотношение дисбаланса:** 22.35x (класс 0 в 22 раза чаще минорных классов)

**Метрики модели:**
- Accuracy: ~32% (низкая, но лучше чем просто предсказывать класс 0)
- Precision: ~83-89% (хорошо: когда модель предсказывает движение, она часто права)
- Recall: ~30-32% (плохо: модель пропускает ~70% примеров движений)
- ROC AUC: ~0.53-0.60 (близко к случайному, плохо разделяет классы)

**Вывод:** Критический дисбаланс (91-94% класса 0) находится в зоне предупреждения. Модель пытается предсказывать движения, но качество низкое из-за дисбаланса и, возможно, недостаточно информативных признаков.

## Решение
Добавить валидацию распределения классов на этапе загрузки датасета, до начала обучения модели.

---

## Задача 1: Добавить метод validate_class_distribution() в TrainingDataset

**Файл:** `model-service/src/models/training_dataset.py`

### Детали реализации:

1. **Сигнатура метода:**
```python
def validate_class_distribution(
    self,
    warning_threshold: float = 0.9,
    error_threshold: float = 0.95,
    min_class_ratio: float = 0.01,
    min_class_count: int = 10,
) -> Dict[str, Any]:
```

2. **Логика проверки:**
   - Вычислить распределение классов через `self.labels.value_counts()`
   - Определить общее количество записей
   - Вычислить процент доминирующего класса
   - Проверить количество уникальных классов (ошибка, если только 1 класс)
   - Проверить минимальное количество примеров для каждого класса
   - Вычислить соотношение дисбаланса (max/min)

3. **Проверки и действия:**
   - **Критический дисбаланс (>error_threshold)**: Вызвать `ValueError` с понятным сообщением
   - **Умеренный дисбаланс (>warning_threshold)**: Вернуть предупреждение в результатах
   - **Минимальный класс**: Проверить, что каждый класс имеет минимум `min_class_count` примеров или `min_class_ratio * total` примеров

4. **Возвращаемое значение:**
```python
{
    "class_distribution": {class: count, ...},  # Словарь с распределением
    "class_percentages": {class: percentage, ...},  # Проценты для каждого класса
    "max_class": int,  # Класс с максимальным количеством
    "max_class_percentage": float,  # Процент доминирующего класса
    "min_class": int,  # Класс с минимальным количеством
    "min_class_percentage": float,  # Процент минорного класса
    "imbalance_ratio": float,  # Соотношение max/min
    "unique_classes_count": int,  # Количество уникальных классов
    "status": str,  # "balanced" | "moderate_imbalance" | "critical_imbalance"
    "warnings": List[str],  # Список предупреждений
    "is_valid": bool,  # True если можно продолжать обучение
}
```

5. **Импорты:**
   - Добавить `from ..config.logging import get_logger` для логирования (если нужно)
   - Использовать `logger` для предупреждений (не для ошибок - ошибки через исключения)

---

## Задача 2: Добавить конфигурируемые пороги в Settings

**Файл:** `model-service/src/config/settings.py`

### Детали реализации:

1. **Добавить поля в класс Settings (после model_classification_threshold):**
```python
# Class Distribution Validation Configuration
model_training_class_imbalance_warning_threshold: float = Field(
    default=0.9,
    alias="MODEL_TRAINING_CLASS_IMBALANCE_WARNING_THRESHOLD",
    description="Warning threshold for class imbalance (0.9 = 90%). Training continues but warning is logged."
)
model_training_class_imbalance_error_threshold: float = Field(
    default=0.95,
    alias="MODEL_TRAINING_CLASS_IMBALANCE_ERROR_THRESHOLD",
    description="Error threshold for critical class imbalance (0.95 = 95%). Training is aborted if exceeded."
)
model_training_min_class_ratio: float = Field(
    default=0.01,
    alias="MODEL_TRAINING_MIN_CLASS_RATIO",
    description="Minimum ratio of examples per class (0.01 = 1% of total). Classes below this ratio trigger warnings."
)
model_training_min_class_count: int = Field(
    default=10,
    alias="MODEL_TRAINING_MIN_CLASS_COUNT",
    description="Minimum absolute count of examples per class. Classes below this count trigger warnings."
)
```

2. **Добавить валидаторы:**
```python
@field_validator("model_training_class_imbalance_warning_threshold")
@classmethod
def validate_warning_threshold(cls, v: float) -> float:
    """Validate warning threshold is between 0 and 1."""
    if not 0.0 <= v <= 1.0:
        raise ValueError("Warning threshold must be between 0.0 and 1.0")
    return v

@field_validator("model_training_class_imbalance_error_threshold")
@classmethod
def validate_error_threshold(cls, v: float) -> float:
    """Validate error threshold is between 0 and 1."""
    if not 0.0 <= v <= 1.0:
        raise ValueError("Error threshold must be between 0.0 and 1.0")
    if v <= settings.model_training_class_imbalance_warning_threshold:
        raise ValueError("Error threshold must be greater than warning threshold")
    return v
```

3. **Обновить env.example и .env**
   - Добавить новые переменные окружения с описаниями в оба файла

---

## Задача 3: Вызвать проверку в training_orchestrator.py

**Файл:** `model-service/src/services/training_orchestrator.py`

### Детали реализации:

1. **Место вызова:** После строки 534 (после логирования "Dataset loaded successfully"), перед строкой 536 (проверка _training_cancelled)

2. **Код:**
```python
# Validate class distribution before training
try:
    from ..config.settings import settings
    distribution_result = dataset.validate_class_distribution(
        warning_threshold=settings.model_training_class_imbalance_warning_threshold,
        error_threshold=settings.model_training_class_imbalance_error_threshold,
        min_class_ratio=settings.model_training_min_class_ratio,
        min_class_count=settings.model_training_min_class_count,
    )
    
    # Log distribution statistics
    logger.info(
        "Class distribution validated",
        training_id=training_id,
        dataset_id=str(dataset_id),
        class_distribution=distribution_result["class_distribution"],
        max_class_percentage=distribution_result["max_class_percentage"],
        imbalance_ratio=distribution_result["imbalance_ratio"],
        status=distribution_result["status"],
        trace_id=trace_id,
    )
    
    # Log warnings if any
    if distribution_result.get("warnings"):
        for warning in distribution_result["warnings"]:
            logger.warning(
                "Class distribution warning",
                training_id=training_id,
                dataset_id=str(dataset_id),
                warning=warning,
                trace_id=trace_id,
            )
    
    # Check if training should continue
    if not distribution_result["is_valid"]:
        logger.error(
            "Training aborted due to critical class imbalance",
            training_id=training_id,
            dataset_id=str(dataset_id),
            max_class_percentage=distribution_result["max_class_percentage"],
            recommendations="Consider adjusting MODEL_CLASSIFICATION_THRESHOLD or MODEL_PREDICTION_HORIZON_SECONDS",
            trace_id=trace_id,
        )
        return  # Abort training
        
except ValueError as e:
    # Critical imbalance detected - abort training
    logger.error(
        "Training aborted due to critical class imbalance",
        training_id=training_id,
        dataset_id=str(dataset_id),
        error=str(e),
        recommendations="Consider adjusting MODEL_CLASSIFICATION_THRESHOLD or MODEL_PREDICTION_HORIZON_SECONDS",
        trace_id=trace_id,
    )
    return  # Abort training
```

3. **Импорты:**
   - Убедиться, что `settings` импортирован (вероятно, уже есть)

---

## Задача 4: Добавить проверку для validation и test splits

**Файл:** `model-service/src/services/training_orchestrator.py`

### Детали реализации:

1. **Для validation split (после строки 573):**
```python
# Validate validation split class distribution
if validation_labels is not None:
    try:
        val_distribution = validation_labels.value_counts()
        val_total = len(validation_labels)
        val_max_pct = val_distribution.max() / val_total if val_total > 0 else 0
        
        logger.info(
            "Validation split class distribution",
            training_id=training_id,
            class_distribution=val_distribution.to_dict(),
            max_class_percentage=val_max_pct,
            trace_id=trace_id,
        )
        
        # Compare with train split distribution
        train_distribution = dataset.labels.value_counts()
        train_max_class = train_distribution.idxmax()
        val_max_class = val_distribution.idxmax()
        
        if train_max_class != val_max_class:
            logger.warning(
                "Different dominant class in validation split",
                training_id=training_id,
                train_max_class=int(train_max_class),
                validation_max_class=int(val_max_class),
                trace_id=trace_id,
            )
    except Exception as e:
        logger.warning(
            "Failed to validate validation split distribution",
            training_id=training_id,
            error=str(e),
            trace_id=trace_id,
        )
```

2. **Для test split (после строки 660, после загрузки test_df):**
   - Аналогичная проверка для test split
   - Сравнение с train и validation splits

---

## Задача 5: Обновить validate_consistency() для автоматической проверки

**Файл:** `model-service/src/models/training_dataset.py`

### Детали реализации:

1. **Обновить метод validate_consistency():**
```python
def validate_consistency(
    self,
    validate_class_distribution: bool = True,
    **class_distribution_kwargs
) -> None:
    """
    Validate that features and labels have consistent dimensions.
    Optionally validate class distribution.

    Args:
        validate_class_distribution: If True, also validate class distribution
        **class_distribution_kwargs: Arguments to pass to validate_class_distribution()
    
    Raises:
        ValueError: If features and labels dimensions don't match or class distribution is invalid
    """
    if len(self.features) != len(self.labels):
        raise ValueError(
            f"Features and labels must have the same length: "
            f"features={len(self.features)}, labels={len(self.labels)}"
        )
    
    if validate_class_distribution:
        # This will raise ValueError if critical imbalance detected
        self.validate_class_distribution(**class_distribution_kwargs)
```

2. **Обновить вызов в model_trainer.py:**
   - Метод `validate_consistency()` уже вызывается в `model_trainer.py` (строка 71)
   - По умолчанию проверка распределения классов будет выполняться автоматически
   - Можно отключить через параметр `validate_class_distribution=False` если нужно

---

## Задача 6: Структурированное логирование

**Файлы:** `model-service/src/models/training_dataset.py`, `model-service/src/services/training_orchestrator.py`

### Детали реализации:

1. **В validate_class_distribution():**
   - Не логировать напрямую (метод должен быть чистым)
   - Возвращать структурированные данные для логирования
   - Исключения (ValueError) для критических ошибок

2. **В training_orchestrator.py:**
   - Использовать структурированное логирование с ключами:
     - `class_distribution`: словарь {class: count}
     - `class_percentages`: словарь {class: percentage}
     - `max_class_percentage`: float
     - `imbalance_ratio`: float
     - `status`: str
     - `warnings`: List[str]
     - `recommendations`: str (если есть проблемы)

3. **Формат сообщений:**
   - Info: нормальное распределение или умеренный дисбаланс
   - Warning: умеренный дисбаланс с рекомендациями
   - Error: критический дисбаланс, обучение прервано

---

## Задача 7: Документация

**Файлы:** `model-service/README.md`, docstrings в коде

### Детали реализации:

1. **Обновить README.md:**
   - Добавить раздел "Class Distribution Validation"
   - Описать, когда проверка выполняется
   - Описать пороги и их значения по умолчанию
   - Примеры сообщений об ошибках и предупреждениях
   - Как настроить пороги через environment variables

2. **Обновить docstrings:**
   - В `validate_class_distribution()`: подробное описание параметров и возвращаемого значения
   - В `validate_consistency()`: описание нового параметра
   - В Settings: описание новых полей

3. **Примеры в документации:**
```markdown
## Class Distribution Validation

The model service automatically validates class distribution before training to detect critical imbalances.

### Configuration

Set these environment variables to customize validation thresholds:

- `MODEL_TRAINING_CLASS_IMBALANCE_WARNING_THRESHOLD=0.9` (default: 0.9 = 90%)
- `MODEL_TRAINING_CLASS_IMBALANCE_ERROR_THRESHOLD=0.95` (default: 0.95 = 95%)
- `MODEL_TRAINING_MIN_CLASS_RATIO=0.01` (default: 0.01 = 1%)
- `MODEL_Training_MIN_CLASS_COUNT=10` (default: 10)

### Behavior

- **Warning (>90% one class)**: Training continues, but warning is logged
- **Error (>95% one class)**: Training is aborted with error message

### Example Error Message

```
Training aborted due to critical class imbalance
max_class_percentage: 100.0
recommendations: Consider adjusting MODEL_CLASSIFICATION_THRESHOLD or MODEL_PREDICTION_HORIZON_SECONDS
```
```

---

## Порядок реализации

1. Задача 2 (Settings) - добавить конфигурацию
2. Задача 1 (TrainingDataset) - добавить метод validate_class_distribution()
3. Задача 5 (TrainingDataset) - обновить validate_consistency()
4. Задача 3 (training_orchestrator) - вызвать проверку для train split
5. Задача 4 (training_orchestrator) - добавить проверку для validation/test splits
6. Задача 6 (логирование) - улучшить структурированное логирование
7. Задача 7 (документация) - обновить документацию

---

## Тестирование

После реализации нужно протестировать:

1. **Нормальное распределение:** Датасет с сбалансированными классами - обучение должно продолжиться
2. **Умеренный дисбаланс (90-95%):** Предупреждение должно быть залогировано, обучение продолжается
   - **Текущий случай:** 91-94% класса 0 → должно быть предупреждение, обучение продолжается
3. **Критический дисбаланс (>95%):** Обучение должно быть прервано с ошибкой
4. **Один класс (100%):** Обучение должно быть прервано с ошибкой
5. **Минорный класс (<1%):** Предупреждение должно быть залогировано

## Рекомендации на основе текущего анализа

### Текущая ситуация (91-94% класса 0)
- **Статус:** Умеренный дисбаланс (в зоне предупреждения)
- **Действие:** Предупреждение будет залогировано, обучение продолжится
- **Рекомендации:**
  1. Рассмотреть уменьшение `MODEL_CLASSIFICATION_THRESHOLD` (сейчас 0.02)
  2. Рассмотреть увеличение `MODEL_PREDICTION_HORIZON_SECONDS` (сейчас 120)
  3. Использовать `class_weight='balanced'` в XGBoost для балансировки классов
  4. Проверить качество признаков (features) - возможно, они недостаточно информативны

### Ожидаемое поведение после реализации
При текущем распределении (91-94% класса 0):
- ✅ Обучение продолжится (дисбаланс < 95%)
- ⚠️ Будет залогировано предупреждение с детальной статистикой
- 📊 В логах будет видно:
  - Распределение классов по всем splits
  - Соотношение дисбаланса
  - Рекомендации по исправлению

---

## Связанные файлы

- `model-service/src/models/training_dataset.py` - основная логика валидации
- `model-service/src/services/training_orchestrator.py` - вызов валидации
- `model-service/src/config/settings.py` - конфигурация порогов
- `model-service/src/services/model_trainer.py` - использование validate_consistency()
- `model-service/README.md` - документация
- `.env` и `env.example` - примеры конфигурации

