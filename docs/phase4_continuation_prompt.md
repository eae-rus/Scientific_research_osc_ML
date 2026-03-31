# Промт для продолжения Phase 4 (17.03.2026, после сессии 4)

## Контекст: что уже сделано

### Сессии 1-2 (см. PHASE_4_WORK_LOG.md)
- Полный pretrain pipeline, AMP fixes, KAN-гейт, NaN→DataSanitizer, углы [0,2π], CSV 272K строк

### Сессия 3
- **NaN Loss fix**: `torch.nan_to_num(x, nan=0.0)` в SpectralReconstructionLoss + ComplexMSELoss
- **100-epoch pretrain**: val_loss 8.86 → 0.245 (best at epoch 39)
- **Fine-tuning pipeline**: zone_size=1, BCEWithLogitsLoss + pos_weight, раздельный LR, float32 loss

### Сессия 4 (текущая)
- **Низшие гармоники**: `compute_low_harmonics_fft()` в `filtering.py` — backward-looking FFT с окнами 2/4/6/10 периодов. Pad начальных точек.
- **AugmentedSpectralDataset** (`osc_tools/ml/augmented_dataset.py`): on-the-fly FFT с аугментацией на сырых данных ДО FFT. Поддерживает SSL + classify. 208 каналов.
- **Уровни сложности**: light (d_model=48, 6L), medium (64, 8L), heavy (64, 16L). CLI: `--complexity`
- **Gradient accumulation**: `--accumulation-steps 8` → effective batch 256
- **Скрипт оценки**: `scripts/phase4_experiments/evaluate_phase4.py` (порог 0.7, full metrics, latency)
- **Обновлены скрипты**: pretrain и finetune поддерживают `--no-augmentation`, `--no-low-harmonics`
- **`build_channel_groups_phase_polar`** расширена: +`num_low_harmonics` параметр

## Текущее состояние файлов
- `osc_tools/preprocessing/filtering.py` — `compute_low_harmonics_fft()` (NEW)
- `osc_tools/ml/augmented_dataset.py` — `AugmentedSpectralDataset` (NEW)
- `osc_tools/ml/losses.py` — `build_channel_groups_phase_polar(num_low_harmonics=)` (UPDATED)
- `scripts/phase4_experiments/run_phase4_pretrain.py` — complexity, augmentation, grad accum (UPDATED)
- `scripts/phase4_experiments/run_phase4_finetune.py` — complexity, augmentation, grad accum (UPDATED)
- `scripts/phase4_experiments/evaluate_phase4.py` — NEW evaluation script

## С чего начать следующую сессию

### Приоритет 1: Smoke test + запуск pretrain с новыми фичами
- Проверить smoke test: `python scripts/phase4_experiments/run_phase4_pretrain.py --mode smoke`
- Запустить pretrain с низшими гармониками + augmentation:
  `python scripts/phase4_experiments/run_phase4_pretrain.py --mode pretrain --complexity medium`
- Сравнить скорость: on-the-fly FFT vs precomputed (ожидается ~3-5x медленнее, но ~100 МБ VRAM)

### Приоритет 2: Fine-tuning с SSL чекпоинтом
- Использовать best_model.pt из pretrain session 3:
  `python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_PhysicalKANTransformer_20260317_093504/best_model.pt --complexity medium`
- ВАЖНО: pretrain session 3 использовал 144 канала (без low harmonics), finetune с 208 каналами будет несовместим.
  Нужен НОВЫЙ pretrain с 208 каналами, либо finetune с `--no-low-harmonics`

### Приоритет 3: Baseline сравнение
- Pretrain BaselineTransformer + fine-tune БЕЗ SSL (random init)
- Сравнить: KAN+SSL vs KAN-random vs Baseline+SSL vs Baseline-random

### Приоритет 4: Evaluation
- `python scripts/phase4_experiments/evaluate_phase4.py --compare-dir experiments/phase4/`
