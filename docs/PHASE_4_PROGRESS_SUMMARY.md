# Фаза 4: Сводка прогресса и план продолжения

> Дата: 31 марта 2026
> Контекст: Physical KAN-Transformer для классификации аварийных режимов

---

## Что выполнено

### Ядро модели (Этап 0–3) — было до текущей серии задач
- PhysicalKANTransformer + BaselineTransformer (полная реализация)
- SSL pretrain: реконструкция + masked modeling + future prediction
- Fine-tune: BCEWithLogitsLoss, зональная классификация
- DataSanitizer (NaN→0 + learnable MissingToken + mask)
- PhysicalStem с FastKAN, ComplexInteractionBlock, DirectionalRelayGate

### Инфраструктура обучения (Q5–Q14) ✅
| Задача | Файл | Суть |
|--------|------|------|
| Q5: Отчётность | `run_phase4_finetune.py` | Per-class F1 каждую эпоху, confusion matrix |
| Q6: Фикс. split | `run_phase4_finetune.py` | `split.json` авто + `--split-from` CLI |
| Q7: LR scheduler | `run_phase4_finetune.py` | CosineAnnealingWarmRestarts (T_0=50, T_mult=2) |
| Q8: Балансировка | `run_phase4_finetune.py` | WeightedRandomSampler + `--balanced-sampling` |
| Q9: 3 класса | `osc_tools/ml/labels.py` | `'base3'` → ML_1/ML_2/ML_3 (Normal = исключение) |
| Q12: Именование | `run_phase4_finetune.py` | `finetune_{model}_{level}_{ts}` + `run_name` |
| Q13: 2 чекпоинта | `run_phase4_finetune.py` | `best_model.pt` (F1) + `best_model_loss.pt` (loss) |
| Q14: Мониторинг | `run_phase4_finetune.py` | Overfit detection (patience=15), CSV-кривые |

### Верификация и исправления (Q1, Q3) ✅
| Задача | Файл | Суть |
|--------|------|------|
| Q1: NaN mask | `transformer_blocks.py` | `nan_to_num` защита при полной маскировке; PhysicalStem.mask TODO |
| Q3: future | `augmented_dataset.py` + `transformer.py` | X = только текущие данные; FuturePredictionHead (cross-attention) для будущих зон; рефакторинг `future_periods` → `future_zones` (привязка к шагам модели, не к SPP) |

### Метрики и оценка (Q10, Q11, B1) ✅
| Задача | Файл | Суть |
|--------|------|------|
| Q10: Edge handling | `evaluate_phase4.py` | `mark_oscillogram` → (probs, coverage), NaN для непокрытых |
| Q11: Пороги | `evaluate_phase4.py` | `find_optimal_thresholds()` per-class F1-optimal |
| B1: Boundary | `evaluate_phase4.py` | `compute_boundary_metrics()`: delay, smearing, false_alarm |

### Абляции (Q4, B2) ✅
| Задача | Файл | Суть |
|--------|------|------|
| Q4: Механизм | `transformer_blocks.py` | `set_ablation()` runtime, disable_kan/interaction/phase_shift |
| B2: Скрипты | `run_phase4_ablation.py` | 7 сценариев inference + режим train, MANUAL_RUN, таблица |

### Визуализация (Q15, Q16 = C1) ✅
| Задача | Файл | Суть |
|--------|------|------|
| Q15: zone viz | `plot_phase4_marking.py` | Coverage подграфик, серый фон для NaN |
| Q16: IFFT | `osc_tools/visualization/spectral_reconstruction.py` | Модуль: polar → complex → IFFT → overlap-add |
| Q16: pretrain | `plot_pretrain_reconstruction.py` | Скрипт: загрузка pretrain → SSL inference → 3 линии: оригинал/идеал/модель; + сравнение амплитуд гармоник |

### Тесты ✅
- 15+ новых тестов в `tests/unit/test_transformer_model.py`:
  TestFuturePredictionHead, TestPhysicalKANTransformerExtended, TestLabelsBase3, TestFindSegments, TestComputeBoundaryMetrics

---

## Что осталось

### Ожидает действий исследователя (требуется до продолжения)

**1. Запуск pretrain + finetune (B3 — Baseline comparison)**
- Запустить `run_phase4_pretrain.py` → `run_phase4_finetune.py` для PhysicalKANTransformer
- Запустить то же для BaselineTransformer (с `--model BaselineTransformer`)
- Сравнить F1, AUC, boundary metrics. Таблица формируется автоматически в evaluate

**2. Проверка pretrain-реконструкции (C1)**
- Запустить `plot_pretrain_reconstruction.py`:
  - Указать претрейн-чекпоинт в MANUAL_CONFIG
  - Указать имя осциллограммы из CSV
  - Визуально оценить: совпадают ли кривые «идеал» и «SSL-модель»?
- Это **главный критерий** качества pretrain

**3. Ответы на вопросы по C2 (расширение датасета)**
Подробный чеклист в `docs/PHASE_4_TASK_DETAILS.md`, раздел Q17:
- Какие CSV-файлы доступны? Пути, формат
- Какие частоты дискретизации (f_rate)? Какие f_network?
- Формат `unlabeled_50_1200.csv` — те же колонки что в `labeled.csv`?
- Файл `norm_coef_vXX.csv` — что в нём? (нормирующие коэффициенты)
- Файл `with_perturbations_vXX.csv` — индекс интересных осциллограмм?
- Симулированные данные: формат, структура, номиналы?
- Как обрабатывать отсутствующие каналы в новых данных?
- Один общий pretrain на всех f_rate, или раздельные модели?

**4. Информация для C3 (РНМ-разметка)**
- Уточнить формулы (уже описаны в PHASE_4_TASK_DETAILS.md)
- Указать, какие уставки использовать для начальной реализации

### Задачи агента (после ответов исследователя)

| Приоритет | Задача | Зависит от |
|-----------|--------|-----------|
| 1 | **C2: Адаптация пайплайна** под разные f_rate (пересчёт FFT_WINDOW, stride, т.д.) | Ответы на вопросы C2 |
| 2 | **C3: Модуль РНМ-разметки** — автоматическая метка аварий для finetune | Уточнение формул |
| 3 | **C4: Масштабирование** — streaming, реестр датасетов | Связано с C2 |
| 4 | **Этап 4.1: full_by_levels** — все подтипы событий | Наличие разметки |
| 5 | **Стресс-тест stride=1** — финальная проверка | Обученная модель |

### Исследовательские задачи (низкий приоритет, не для агента)
- Stem переписать (d_model=16) — «будет не так», ждём решение исследователя
- Итоги Фазы 3 — текст для статьи
- Написать про ОЗЗ/ДПОЗЗ аналитическими формулами

---

## Ключевые файлы для навигации

| Файл | Назначение |
|------|-----------|
| `docs/phase_discription/PHASE_4_PLAN.md` | Полный план с чекбоксами |
| `docs/PHASE_4_TASK_DETAILS.md` | Подробное описание всех задач |
| `scripts/phase4_experiments/run_phase4_pretrain.py` | Pretrain скрипт |
| `scripts/phase4_experiments/run_phase4_finetune.py` | Finetune скрипт |
| `scripts/phase4_experiments/evaluate_phase4.py` | Оценка: метрики, пороги, boundary |
| `scripts/phase4_experiments/run_phase4_ablation.py` | Абляционные эксперименты |
| `scripts/phase4_experiments/plot_pretrain_reconstruction.py` | Визуализация pretrain (C1) |
| `scripts/phase4_experiments/plot_phase4_marking.py` | Визуализация finetune разметки |
| `osc_tools/visualization/spectral_reconstruction.py` | Модуль обратного FFT |
| `osc_tools/ml/models/transformer.py` | Архитектуры моделей |
| `osc_tools/ml/augmented_dataset.py` | On-the-fly FFT dataset |
| `osc_tools/ml/labels.py` | Управление метками (base3/base/ozz) |
| `tests/unit/test_transformer_model.py` | Тесты моделей |
