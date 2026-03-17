# Лог работ по Фазе 4 (Physical KAN-Transformer)

## [2026-03-17] Сессия 3: NaN fix, полный pretrain, fine-tuning pipeline

### Исправления

1. **Критический баг: NaN Loss в SpectralReconstructionLoss**
   - **Проблема**: `true_amp` и `true_phase` содержали NaN из-за отсутствующих каналов (IN absent в 96% файлов). При вычислении `cos(NaN)` / `sin(NaN)` результат = NaN, а `NaN * 0 = NaN` (IEEE 754) — маскирование через умножение не убирает NaN.
   - **Решение**: Добавлен `torch.nan_to_num(x, nan=0.0)` для всех 4 входных тензоров **перед** вычислениями в [osc_tools/ml/losses.py](osc_tools/ml/losses.py). Маска по-прежнему игнорирует эти позиции при агрегации Loss.
   - Исправлено в `SpectralReconstructionLoss.forward()` и `ComplexMSELoss.forward()`.

### Запущенный pretrain (100 эпох)

- Модель: PhysicalKANTransformer, 515,572 параметра
- d_model=64, 4 layers, 4 heads, batch_size=32
- ~20 сек/эпоха, AMP, RTX 3060 Ti (8 ГБ, использовано ~80 МБ VRAM)
- Кривая обучения (val_loss): 8.86 → 0.43 за 24 эпохи (продолжает падать)
- Сохранение: `experiments/phase4/pretrain_PhysicalKANTransformer_20260317_093504/`

### Реализовано: Fine-tuning pipeline

- Создан скрипт [scripts/phase4_experiments/run_phase4_finetune.py](scripts/phase4_experiments/run_phase4_finetune.py)
- **Ключевые решения:**
  - Инициализация из SSL-чекпоинта (`strict=False`, пропускает `cls_head`)
  - zone_size=1: каждый временной шаг (stride=16 ≈ полпериода) = зона
  - 4 класса: Target_Normal, Target_ML_1, Target_ML_2, Target_ML_3
  - `BCEWithLogitsLoss` с `pos_weight` для балансировки классов
  - Раздельный LR: backbone=1e-5, head=5e-4
  - Метрики: Macro-F1, ROC-AUC, per-class F1
  - `target_window_mode='any_in_window'` — окно помечается, если есть событие
- Smoke-test: запущен, проверяется

---

## [2026-03-16] Сессия 2: KAN-гейт, NaN конвенция, CSV, углы [0, 2π]

### Выполненные работы

1. **KAN-гейт в FFN**: заменён `nn.Linear` → `FastKANLayer` в `PhysicalKANFeedForward.angle_gate`
2. **Симметричные составляющие**: обрезаны до h1 в `_build_feature_columns`
3. **Конвенция отсутствующих каналов**: NaN (не -1). DatasetManager → NaN, DataSanitizer(missing_marker=None)
4. **Углы в [0, 2π]** в `polar.py`
5. **CSV регенерирован** (272K строк, IN absent в 96% файлов)
6. **§9.5 architectures_description.md** переписана
7. **448 тестов пройдены**, 3-epoch pretrain OK
8. Файлы: см. `docs/phase4_continuation_prompt.md`

---

## [2026-03-16] Сессия 1: SSL pipeline, AMP fix, Loss, Architecture

### Выполненные работы

1. Полный pretrain pipeline (`run_phase4_pretrain.py` ~790 строк)
2. AMP fix: комплексные блоки работают в float32 через `autocast(enabled=False)`
3. `SpectralReconstructionLoss` с 4-шаговой нормализацией
4. `SSLSpectralDataset`: маскирование 25% timesteps + предсказание 2 периодов
5. Baseline и Physical модели реализованы
