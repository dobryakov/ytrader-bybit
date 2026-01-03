## Краткий отчёт: лучшие результаты

### Лучшая модель (по Sharpe Ratio)

- Ассет: ETHUSDT
- Feature Registry: 1.7.2
- Период обучения: 60 дней
- Метрики (test set):
  - Sharpe Ratio: 0.691780
  - Information Coefficient: 0.124533
  - Directional Accuracy: 0.539960

---

### Оптимальные конфигурации по ассетам

#### ETHUSDT
- Feature Registry: 1.7.2
- Период: 60 дней
- Sharpe Ratio (test): 0.691780
- Information Coefficient (test): 0.124533
- Directional Accuracy (test): 0.539960

#### BTCUSDT
- Feature Registry: 1.7.2
- Период: 21 день
- Sharpe Ratio (test): 0.459227
- Information Coefficient (test): 0.426224 (лучший IC среди всех моделей)
- Directional Accuracy (test): 0.584680

---

### Лучшие результаты по метрикам

1. Sharpe Ratio: ETHUSDT, 1.7.2, 60 дней — 0.691780
2. Information Coefficient: BTCUSDT, 1.7.2, 21 день — 0.426224
3. Directional Accuracy: BTCUSDT, 1.7.2, 60 дней — 0.599198

---

### Выводы

- Лучшая общая модель: ETHUSDT с Feature Registry 1.7.2 и 60 днями обучения
- ETHUSDT лучше работает с 60 днями обучения
- BTCUSDT лучше работает с 21 днём обучения
- Feature Registry 1.7.2 используется во всех лучших моделях
