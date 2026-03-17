# Промт для продолжения Phase 4 (17.03.2026)

## Контекст: что уже сделано

### Сессия 1
- Полный **pretrain pipeline** (`run_phase4_pretrain.py` ~790 строк): CLI, CosineAnnealingLR + warmup, AMP, checkpointing, JSONL logging
- **AMP fix**: комплексные блоки (ComplexInteractionBlock, PhysicalStem, PhysicalKANFeedForward) работают в float32 через `autocast(enabled=False)`
- `SpectralReconstructionLoss` с 4-шаговой нормализацией + `build_channel_groups_phase_polar(separated=True)`
- `SSLSpectralDataset`: маскирование 25% timesteps + предсказание 2 будущих периодов

### Сессия 2
- **KAN-гейт в FFN**: заменил `nn.Linear` → `FastKANLayer` в `PhysicalKANFeedForward.angle_gate` (как в Stem)
- **Симметричные составляющие**: обрезаны до h1 в `_build_feature_columns` (symmetric/symmetric_polar)
- **Конвенция отсутствующих каналов**: NaN (не -1). DatasetManager детектирует `missing_ch` и пишет NaN. `_preload_data()` сохраняет NaN. DataSanitizer(missing_marker=None) по умолчанию.
- **Углы в [0, 2π]** в `polar.py`
- **CSV регенерирован** (272K строк, IN absent в 96% файлов)
- **§9.5 architectures_description.md** переписана без FIXME
- **448 тестов пройдены**, smoke test OK, 3-epoch pretrain OK (loss падает)

### Сессия 3 (17.03.2026)
- **Критический баг исправлен**: NaN Loss в `SpectralReconstructionLoss` из-за NaN в target (отсутствующие каналы). Добавлен `torch.nan_to_num(x, nan=0.0)` **перед** вычислениями. Также исправлено для `ComplexMSELoss`.
- **Полный pretrain запущен** (100 эпох): val_loss упал с 8.86 → 0.245 за 39 эпох (best). После 42-й эпохи val начинает слегка расти — модель может переобучаться. Чекпоинт: `experiments/phase4/pretrain_PhysicalKANTransformer_20260317_093504/`
- **Fine-tuning pipeline реализован** (`run_phase4_finetune.py`):
  - Инициализация из SSL-чекпоинта (`strict=False`, `cls_head` = random init)
  - zone_size=1, 4 класса (base), BCEWithLogitsLoss + pos_weight
  - Раздельный LR: backbone=1e-5, head=5e-4
  - Метрики: Macro-F1, ROC-AUC, per-class F1, exact_match
  - Loss вычисляется в float32 (fix для AMP overflow в BCEWithLogitsLoss)
  - Smoke-test без SSL: val F1=0.318, AUC=0.662 (baseline random init)
- **PHASE_4_WORK_LOG.md** создан
- **PHASE_4_PLAN.md** обновлён (Этап 4 частично закрыт)

## Текущее состояние файлов
- `osc_tools/ml/layers/transformer_blocks.py` — все блоки с AMP-фиксами + KAN-гейт
- `osc_tools/ml/models/transformer.py` — PhysicalKANTransformer + BaselineTransformer, NaN sanitizer
- `osc_tools/ml/losses.py` — SpectralReconstructionLoss + ComplexMSELoss (с NaN-to-num fix)
- `osc_tools/ml/ssl_dataset.py` — SSLSpectralDataset
- `osc_tools/ml/precomputed_dataset.py` — symmetric h1 only, NaN preserved in _preload_data
- `osc_tools/features/polar.py` — углы [0, 2π]
- `osc_tools/data_management/dataset_manager.py` — NaN маркеры для missing channels
- `scripts/phase4_experiments/run_phase4_pretrain.py` — полный pretrain pipeline
- `scripts/phase4_experiments/run_phase4_finetune.py` — полный fine-tuning pipeline

## С чего начать следующую сессию

### Приоритет 1: Анализ pretrain и запуск fine-tuning
- Проверить чекпоинты pretrain в `experiments/phase4/pretrain_PhysicalKANTransformer_20260317_093504/`
- Запустить fine-tuning с лучшим SSL-чекпоинтом:
  `python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_.../best_model.pt`
- Сравнить с fine-tuning БЕЗ SSL (random init) — baseline

### Приоритет 2: Baseline сравнение
- Запустить SSL pretrain для BaselineTransformer + fine-tuning
  `python scripts/phase4_experiments/run_phase4_pretrain.py --mode pretrain --model BaselineTransformer`
  `python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint ... --model BaselineTransformer`
- Сравнить Macro-F1 / ROC-AUC: PhysicalKAN vs Baseline

### Приоритет 3: Низшие гармоники (Этап 1)
- Добавить расчёт гармоник с периодами 2, 4, 6, 10 периодов промышленной частоты  
- Паддинг начальных точек: дублировать первое вычисленное значение
- Потребует изменений в: dataset_manager.py, precomputed_dataset.py, потенциально новый FFT-калькулятор

### Приоритет 4: Smearing width метрика
- Оценить точность локализации границ событий (task из PHASE_4_PLAN, Этап 4, п.2)
- Стресс-тест на stride=1
