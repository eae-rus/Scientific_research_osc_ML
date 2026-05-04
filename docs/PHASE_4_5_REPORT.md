# Этап 4.5: Обучение на симулированных данных RTDS — Отчёт

## 1. Обзор и мотивация

### 1.1 Цель
Обучить модель Physical KAN-Transformer на задачу классификации типов однофазных замыканий на землю (ОЗЗ) и дуговых перемежающихся замыканий (ДПОЗЗ) в сетях 6-35 кВ с изолированной/компенсированной нейтралью, используя симулированные данные RTDS (Real-Time Digital Simulator).

### 1.2 Задача
Модель должна классифицировать зоны осциллограммы по 4 бинарным признакам (multi-label):
1. **Stable SPGF** (Устойчивое ОЗЗ) — одиночный пробой изоляции
2. **Petersen AIGF** — дуговое перемежающееся ОЗЗ по теории Петерсена
3. **Peters-Slepyan AIGF** — ДПОЗЗ по теории Петерса и Слепян
4. **Belyakov AIGF** — ДПОЗЗ по теории Белякова Н.Н.

Когда все 4 выхода = 0, зона считается **нормальным режимом** (нет ОЗЗ). Это неявный 5-й класс, представленный:
- **участками до начала ОЗЗ** внутри каждой SimOZZ осциллограммы (target `[0,0,0,0]` автоматически),
- **реальными осциллограммами без ОЗЗ** — КЗ, пуски двигателей и пр. — для разнообразия «нормы».

### 1.3 Зачем нужны симулированные данные
Реальные осциллограммы ОЗЗ крайне редки и плохо размечены по типу дуги. RTDS-моделирование позволяет:
- Получить **тысячи** примеров каждого типа дуги с **точной разметкой**
- Варьировать параметры сети (R, L) для обобщения
- Контролировать момент и длительность аварии

---

## 2. Набор данных Simulated_OZZ_v1

### 2.1 Источник
~19 200 CSV-файлов, сгенерированных RTDS-симулятором для сети 10 кВ с изолированной нейтралью. Общий объём ~62 ГБ.

### 2.2 Имена файлов
Формат: `OZZ_X_R_L_P_I_T.csv` или `OZZ_X_R_L_P_T.csv`, где:
- **X** (1-4) — тип дуги (Stable, Petersen, PetersSlepian, Beliakov)
- **R** (1, 10, 100, 1000) — активное сопротивление утечки (Ом)
- **L** (10-100) — индуктивность (мГн)
- **P** (1-3) — фаза замыкания
- **I** — угол зажигания (опционально)
- **T** — момент замыкания (мкс)

### 2.3 Каналы
8 сигнальных каналов:
| Канал | Описание | Исходное имя в CSV |
|-------|----------|-------------------|
| IA, IB, IC | Фазные токи линии | `TLINE5|IA`, `IB`, `IC` |
| IN | Ток нулевой последовательности | `3I0` |
| UA, UB, UC | Фазные напряжения | `S1_VA`, `S1_VB`, `S1_VC` |
| UN | Напряжение нулевой последовательности | `3U0` |

**Важно:** IN (3I0) и UN (3U0) — **реальные измеренные** сигналы (не нули). Их максимальные значения: IN до ~0.055 A, UN до ~26 кВ.

### 2.4 Частота дискретизации
Fs зависит от параметра L (индуктивность) — это шаг моделирования RTDS:

| L (мГн) | dt (с) | Fs (Гц) | spp | stride (1/8 периода) | Доля файлов |
|---------|--------|---------|-----|---------------------|------------|
| 10, 20 | 2.6e-5 | 38 462 | 769 | 96 | ~20% |
| 30-100 | 2.5e-5 | 40 000 | 800 | 100 | ~80% |

### 2.5 Разметка
Бинарные столбцы OZZ-флагов: `OZZ`, `Stable`, `Petersen`, `Peters_Slepian`, `Beliakov2`.
Модель получает 4 целевых колонки (multi-label):
- `Target_OZZ_Stable` = OZZ × (X==1)
- `Target_OZZ_Petersen` = OZZ × (X==2)
- `Target_OZZ_PetersSlepian` = OZZ × (X==3)
- `Target_OZZ_Beliakov` = OZZ × (X==4)

Когда все 4 = 0 → состояние «не ОЗЗ» (нормальный режим).

---

## 3. Архитектура пайплайна

### 3.1 Загрузка данных (Lazy Loading)
62 ГБ данных не помещаются в ОЗУ (32 ГБ). Реализован ленивый загрузчик:
- **`SimOZZFileIndex`** — сканирует директорию, считывает только dt и n_samples (без данных)
- **`SimOZZLazyDataset`** — подгружает CSV по требованию с LRU-кэшем (500 файлов ≈ 500 МБ)
- **Per-file Fs** — параметры FFT (spp, stride, fft_window, window_size) вычисляются индивидуально для каждого файла через `_compute_params(fi.fs)`

### 3.2 Спектральная обработка (On-the-fly FFT)
Используется единая функция `compute_spectral_from_raw()`:
1. **FFT** по окну в 1 период (spp отсчётов) → 9 стандартных гармоник
2. **Низшие гармоники** (sub-periods [2,4,6,10]) через backward-looking DFT
3. **Polar conversion** (амплитуда + угол) с опорой на UA h1
4. **Симметричные составляющие** (I1, I2, I0, U1, U2, U0) для h1

Результат: **220 каналов** = 8 сигналов × (9+4) гармоник × 2 (mag+phase) + 6 симметричных × 2

### 3.3 Зонирование
- Stride = 1/8 периода (96-100 сырых отсчётов ≈ 2.5 мс)
- Окно = 10 периодов (0.2 с)
- Числа зон: **(10-1) × 8 = 72** — фиксировано, не зависит от Fs
- Зонные метки: max-агрегация OZZ-флагов внутри stride

### 3.4 Формат выхода
```
X: Tensor (220, 72)  — спектральные признаки
Y: Tensor (72, 4)    — метки зон
```

---

## 4. Аугментация

### 4.1 Базовая (TimeSeriesAugmenter)
Применяется до FFT на сырых 8-канальных данных:
- **Инверсия** (p=0.5): x → -x (физически корректно для AC)
- **Масштабирование** (p=0.2): ×0.9-1.1 для токов, ×0.98-1.02 для напряжений
- **Перетасовка фаз** (p=0.33): ABC → BCA или CAB (инвариантность к маркировке фаз)

### 4.2 Dropout каналов нулевой последовательности (NeutralChannelDropout)
Имитация отсутствия IN и/или UN (не все подстанции оснащены ТТ/ТН нулевой последовательности):
- С p=1/3: все 8 каналов присутствуют
- С p=1/3: IN обнулён (нет ТТ нулевой последовательности)
- С p=1/3: IN и UN обнулены (нет ТТ и ТН нулевой последовательности)

Маркер отсутствующего канала: `fill_value=0.0`.

### 4.3 Dropout фазного тока (PhaseCurrentDropout)
Имитация отсутствия ТТ одной из фаз. В реальных осциллограммах часто фаза B физически не подключена — на канале чистый шум (RMS ~0.025 против IA/IC ~0.5-1.0). Это ломает вычисление симметричных составляющих и ухудшает inference.

**Режимы** (выбираются случайно):
- **45%** — все 3 фазы тока присутствуют (без изменений)
- **45%** — зануляется IB (позиция 1) — имитация отсутствия ТТ фазы B
- **5%** — зануляется IA (позиция 0)
- **5%** — зануляется IC (позиция 2)

**Важно:** Применяется **после** PhaseShuffling (внутри `CompositeAugmenter` — `base_aug` → `phase_dropout` → `neutral_dropout`). Это означает, что зануляется то, что нейросеть видит как «позиция B», а не исходная фаза B до перетасовки. Физически корректно: пропадание канала ТТ — это пропадание конкретного **измерения**, а не конкретной **фазы** электроэнергии.

**Файл:** `osc_tools/ml/augmentation.py` → `PhaseCurrentDropout`.

### 4.4 Полная цепочка аугментации
```
CompositeAugmenter:
  1. TimeSeriesAugmenter (на сырых 8-канальных данных до FFT):
     ├── Инверсия (p=0.5): x → -x
     ├── Масштабирование (p=0.2): ×0.9-1.1 (токи), ×0.98-1.02 (напряжения)
     ├── PhaseShuffling (p=0.33): ABC → BCA или CAB
     └── DropChannel (p=0.0): отключён
  2. PhaseCurrentDropout: зануление одного фазного тока (45% B, 5% A, 5% C)
  3. NeutralChannelDropout: зануление IN и/или UN (1/3 каждый режим)
```
Всё применяется до FFT. Для val-аугментации используется только `NeutralChannelDropout` (без инверсии, масштабирования и фазного дропаута).

---

## 5. Комбинированное обучение (5 классов)

### 5.1 Проблема
Симулированные данные содержат только ОЗЗ (хотя внутри осциллограмм есть участки «нормы»). Для робастности нужен 5-й класс — реальные осциллограммы без ОЗЗ.

### 5.2 Источник 5-го класса
Из `data/ml_datasets/labeled_2025_12_03.csv` (990 реальных осциллограмм):
- **959 файлов** (~1.5M строк) не содержат ни одного OZZ-флага
- Отфильтрованы по колонкам: ML_2_1, ML_2_1_1, ML_2_1_2, ML_2_1_3, ML_2_2
- Содержат КЗ, пуски двигателей и другие неаварийные режимы

### 5.3 Совместимость форматов
Реальные данные: Fs=1600 Гц, spp=32, window_size=320 (10 периодов).
При stride_fraction=8: stride=4 → (320-32)/4 = **72 зоны** — совпадает с SimOZZ.
Число каналов: **220** — одинаковое.
Формат: `(220, 72)` — идентичный.

### 5.4 Балансировка
`BalancedEpochSampler` обеспечивает равномерную выборку:
- **samples_per_class** элементов из каждого из 5 классов за эпоху
- Для редких классов — oversampling с повтором
- Seed обновляется каждую эпоху для разнообразия

### 5.5 Диаграмма классов

| ID | Класс | Источник | Параметры | Target |
|---|-------|---------|-----------|--------|
| 0 | Stable SPGF | SimOZZ X=1 | ~4800 файлов | [1,0,0,0] |
| 1 | Petersen AIGF | SimOZZ X=2 | ~4800 файлов | [0,1,0,0] |
| 2 | Peters-Slepyan AIGF | SimOZZ X=3 | ~4800 файлов | [0,0,1,0] |
| 3 | Belyakov AIGF | SimOZZ X=4 | ~4800 файлов | [0,0,0,1] |
| 4 | Normal (не ОЗЗ) | Реальные осц. (КЗ, пуски и др.) | 959 файлов | [0,0,0,0] |

**Примечание:** `[0,0,0,0]` (Normal) встречается также в SimOZZ — это зоны до начала ОЗЗ внутри каждой осциллограммы. Таким образом, модель учится распознавать «нормальный режим» из двух источников: (1) синтетическая «норма» до момента пробоя и (2) реальные переходные процессы без ОЗЗ.

---

## 6. Нормализация

### 6.1 Проблема
Для реальных осциллограмм используется физическая нормализация:
```
Ip_nominal = 20 × base_val (фазные токи)
Iz_nominal = 5 × base_val  (ток нулевой посл.)
Ub_nominal = 3 × base_val  (напряжения)
```
где `base_val` — из `norm_coef_all_v1.4.csv` для каждой шины.

Для симулированных данных (сеть 10 кВ) номинальные значения определены по результатам сканирования всех 19 199 файлов.

### 6.2 Сбор статистики
Скрипт `scripts/phase4_experiments/scan_sim_ozz_statistics.py` просканировал все 19 199 файлов (~20 мин).

Результаты:
- Подробная таблица: `reports/sim_ozz_channel_stats.csv`
- Сводка: `reports/sim_ozz_channel_summary.txt`

| Канал | max\|h1\| global | median max\|h1\| | Номинал | Norm. median |
|-------|-----------------|-----------------|---------|-------------|
| IA    | 0.060           | 0.038           | **0.2** (ТТ 200А) | 0.19 |
| IB    | 0.059           | 0.038           | **0.2** | 0.19 |
| IC    | 0.060           | 0.038           | **0.2** | 0.19 |
| IN    | 0.023           | 0.009           | **0.03** (ТТНП 30А) | 0.28 |
| UA    | 13.48 кВ        | 9.39 кВ         | **10** кВ (U_лин) | 0.94 |
| UB    | 13.58 кВ        | 9.39 кВ         | **10** кВ | 0.94 |
| UC    | 13.37 кВ        | 9.39 кВ         | **10** кВ | 0.94 |
| UN    | 20.71 кВ        | 6.49 кВ         | **17.32** кВ (3Uф) | 0.37 |

### 6.3 Выбранные номиналы (SIM_NOMINAL) — обновлены

Для совместимости с `NormOsc` (нормализация реальных осциллограмм) номиналы домножены на те же коэффициенты:
```
NormOsc:  Ip_nominal = 20 × base_val,  Iz_nominal = 5 × base_val,  Ub_nominal = 3 × base_val
```

**Базовые значения** (физика сети 10 кВ):
- `_U_LIN_KV = 10.0` — линейное напряжение сети (кВ)
- `_I_NOM_CT_KA = 0.2` — номинал ТТ 200А (в кА)
- `_I_NOM_CTUP_KA = 0.03` — номинал ТТНП 30А (в кА)

**SIM_NOMINAL после применения множителей:**
| Канал | Формула | Значение | Пояснение |
|-------|---------|----------|----------|
| IA/IB/IC | 20 × 0.2 | **4.0** | 20× (как NormOsc) от ТТ 200А |
| IN | 5 × 0.03 | **0.15** | 5× (как NormOsc) от ТТНП 30А |
| UA/UB/UC | 3 × 10.0 | **30.0** | 3× (как NormOsc) от U_лин |
| UN | 3 × (10×√3) | **51.96** | 3× от 3Uф |

Этим обеспечивается **единый масштаб** между SimOZZ и реальными данными: медианное спектральное отношение voltage h1 ≈ 1.0 (ранее было 3.08× из-за рассогласования).

Константы хранятся в `osc_tools/ml/simulated_ozz_dataset.py` → `SIM_NOMINAL`.

---

## 7. Файловая структура

### 7.1 Новые модули
| Файл | Назначение |
|------|-----------|
| `osc_tools/ml/simulated_ozz_dataset.py` | Lazy-загрузчик SimOZZ с per-file Fs |
| `osc_tools/ml/balanced_dataset.py` | BalancedConcatDataset + BalancedEpochSampler |
| `osc_tools/data_management/no_ozz_filter.py` | Фильтрация реальных осциллограмм без ОЗЗ |
| `osc_tools/data_management/sim_ozz_split.py` | Стратифицированный split по типу дуги |
| `osc_tools/data_management/real_ozz_split.py` | Разделение real_OZZ по верификации |
| `scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py` | Тренировочный скрипт |
| `scripts/phase4_experiments/sim_ozz/scan_sim_ozz_statistics.py` | Сбор статистики каналов |
| `scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py` | Оценка на SimOZZ val |
| `scripts/phase4_experiments/real_ozz/inference_real_ozz.py` | Inference на реальных COMTRADE |
| `scripts/phase4_experiments/real_ozz/export_single_comtrade_marking.py` | Экспорт разметки одного COMTRADE в CSV/PNG |
| `scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py` | Статистика по реальным ОЗЗ |
| `scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py` | Запуск baseline-моделей для сравнения |
| `scripts/phase4_experiments/evaluation/model_complexity_report.py` | Отчёт сложности модели для статьи |
| `scripts/phase4_experiments/interpretability/channel_dropout_probing.py` | Channel Dropout Probing |
| `scripts/phase4_experiments/interpretability/gradient_attribution.py` | Gradient×Input Attribution |

### 7.2 Изменённые модули
| Файл | Изменение |
|------|----------|
| `osc_tools/ml/augmented_dataset.py` | Параметризация `compute_spectral_from_raw()` для произвольной Fs |
| `osc_tools/ml/augmentation.py` | `NeutralChannelDropout`, `PhaseCurrentDropout`, `CompositeAugmenter` |
| `osc_tools/ml/labels.py` | target_level='sim_ozz' |

---

## 8. Ключевые технические решения

### 8.1 Per-file Fs вместо глобального
Fs зависит от L (шаг моделирования RTDS). Единый Fs для всех файлов приводил бы к некорректному FFT для ~20% данных. Решение: `_compute_params(fi.fs)` вычисляет spp, stride, fft_window, window_size для каждого файла. Число зон всегда = (num_periods_window - 1) × stride_fraction = 72.

### 8.2 Совместимость с реальными данными
Stride_fraction=8 обеспечивает одинаковое число зон (72) для обоих типов данных, несмотря на разницу в Fs (1600 vs 38k-40k Гц).

### 8.3 Lazy loading + LRU cache
LRU-кэш на 500 файлов (~500 МБ) снижает I/O при sequential access в DataLoader. `num_workers=0` из-за thread-safety OrderedDict.

---

## 9. Инструкция по запуску обучения

### 9.1 Предусловия
- Данные `data/Simulated_OZZ_v1/` — ~19 200 CSV-файлов (~62 ГБ)
- Реальные осциллограммы `data/ml_datasets/labeled_2025_12_03.csv`
- CUDA-совместимый GPU (модель занимает ~200-500 МБ VRAM, основное потребление — батчи)
- Anaconda с PyTorch, CUDA

### 9.2 Smoke-test (быстрая проверка)
Команда (CLI):
```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py --smoke
```
Что делает: 2 эпохи, 20 файлов, batch=4, без симметричных — проверка работоспособности пайплайна.

### 9.3 Полное обучение
Команда (CLI):
```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py
```
Или с параметрами:
```bash
... python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py \
    --model PhysicalKANTransformer --complexity light --epochs 50
```

Допустимые аргументы:
| Аргумент | Описание | По умолчанию |
|----------|----------|-------------|
| `--model` | `PhysicalKANTransformer` или `BaselineTransformer` | `PhysicalKANTransformer` |
| `--complexity` | `light` / `medium` / `heavy` | из конфига |
| `--epochs` | Число эпох | 50 |
| `--batch-size` | Размер батча | 32 |
| `--max-files` | Ограничить число файлов (для отладки) | None (все) |
| `--accumulation-steps` | Gradient accumulation | 8 |
| `--smoke` | Быстрый smoke-test | — |
| `--resume` | Путь к чекпоинту для продолжения | None |
| `--reset-optimizer` | Сбросить оптимизатор при resume | — |

### 9.4 Ручной запуск (из секции `__main__`)
Можно запустить скрипт напрямую, отредактировав константы в конце файла:
```python
MODEL_TYPE = 'PhysicalKANTransformer'
SELECTED_COMPLEXITY = 'light'
EPOCHS = 50
MAX_FILES = None   # None = все 19199 файлов
```

### 9.5 Продолжение обучения
```bash
... python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py \
    --resume experiments/phase4/sim_ozz_finetune_.../latest_checkpoint.pt
```

### 9.6 Результаты
Сохраняются в `experiments/phase4/sim_ozz_finetune_<model>_<timestamp>/`:
- `config.json` — конфигурация запуска
- `split.json` — списки файлов train/val
- `training_log.jsonl` — метрики по эпохам
- `training_curves.csv` — то же в CSV
- `best_model.pt` — лучшая модель по F1
- `best_model_loss.pt` — лучшая модель по loss
- `latest_checkpoint.pt` — последний чекпоинт
- `confusion_matrix.txt` — confusion matrix
- `summary.json` — итоговые метрики

### 9.7 Что ожидать при запуске
1. Сканирование `~19 200` файлов (dt + n_samples) — ~30-60 сек
2. Стратифицированный split 80/20 по типу дуги
3. Подготовка реальных данных (959 файлов «не ОЗЗ»)
4. Создание BalancedConcatDataset (5 классов) и BalancedEpochSampler
5. Обучение с прогресс-баром (tqdm)
6. Чекпоинты каждые 5 эпох + best + latest

---

## 10. Оценка и визуализация на SimOZZ (Этап 10.1)

### 10.1 Реализованный инструментарий

**Скрипт:** `scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py`

Переиспользует метрики из `evaluate_phase4.py` (без дублирования) и добавляет SimOZZ-специфичную загрузку данных.

#### Метрики (window-level и zone-level):
- **Per-class F1, Precision, Recall** (4 класса ОЗЗ)
- **Macro-F1** с порогом 0.5 и оптимальным (grid search per-class)
- **ROC-AUC** per-class и macro
- **Confusion matrix** (per-class binary: TP/FP/FN/TN)
- **Exact match accuracy** (все 4 класса совпали)
- **Boundary-метрики** (zone-level): delay, smearing onset/offset, false alarm zones
- **Inference latency** (мс/sample)

#### Визуализация (сохраняется в `reports/phase4/sim_ozz_eval/`):
- `confusion_matrix.png` — per-class confusion matrices
- `roc_curves.png` — ROC-кривые с AUC
- `prob_distributions.png` — гистограммы P(class) для positive/negative samples
- `training_curves.png` — loss и F1 по эпохам из training_log.jsonl

#### Запуск:
```bash
python scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py --checkpoint experiments/phase4/sim_ozz_.../best_model.pt
```
Или ручной режим: скрипт автоматически находит последний SimOZZ эксперимент в `experiments/phase4/`.

**Статус:** Реализовано, ожидает завершения обучения. По результатам запуска будут получены количественные метрики и диаграммы.

---

## 11. Перенос на реальные ОЗЗ (Этап 10.3)

### 11.1 Набор данных real_OZZ

**Расположение:** `data/real_OZZ/osc_comtrade/` — 1865 COMTRADE файлов (.cfg + .dat).

**Разделение** по отчёту `overvoltage_report_T1_with_com_v1.7.csv` (поле «Проверка»):
- **Confirmed OZZ** (`+`): 830 записей — подтверждённые случаи ОЗЗ
- **False detection** (`-`): 1025 записей — ошибочно детектировано (нет ОЗЗ)

Один файл COMTRADE может содержать несколько секций (bus=1,2,...,5).
Распределение по секциям: bus 1 — 957, bus 2 — 822, bus 3 — 42, bus 4 — 31, bus 5 — 3.
По типу: СШ (сборная шина) — 1566, КЛ (кабельная линия) — 289.

### 11.2 Модуль разделения

**Файл:** `osc_tools/data_management/real_ozz_split.py`

Функции:
- `load_real_ozz_report()` — загрузка CSV отчёта с стандартизацией имён колонок
- `split_by_verification(report)` → `(confirmed_df, false_detection_df)` с проверкой наличия файлов на диске
- `get_bus_for_file(report, filename)` → список секций для файла
- `save_split(report)` → сохранение confirmed_ozz.csv и false_detection.csv

### 11.3 Inference пайплайн

**Скрипт:** `scripts/phase4_experiments/real_ozz/inference_real_ozz.py`

Пайплайн для каждого файла:
1. **Парсинг COMTRADE** через `ReadComtrade` → Polars DataFrame
2. **Определение секций** (bus) из overvoltage_report → для каждой секции отдельно
3. **Маппинг каналов**: `I | Bus-{bus} | phase: A` → IA, `U | BusBar-{bus} | phase: A` → UA и т.д.
4. **Нормализация** через `NormOsc` + `norm_coef_all_v1.4.csv`
5. **Адаптация Fs**: spp, stride, fft_window вычисляются из реальной частоты файла
6. **Inference** скользящим окном (шаг = 1 период) с пообразцовым усреднением
7. **Автокроп** вокруг обнаруженного ОЗЗ (±10 периодов от зоны с P > 0.3)

#### Визуализация:
Для каждого файла строится комбинированный график:
- Токи (IA, IB, IC, IN) — фазная раскраска
- Напряжения (UA, UB, UC, UN)
- 4 панели: вероятности классов ОЗЗ (Stable, Petersen, Peters-Slepyan, Belyakov)
- Coverage (число окон, покрывающих каждую точку)

Графики сохраняются раздельно:
- `reports/phase4/real_ozz_inference/confirmed_ozz/` — для подтверждённых ОЗЗ
- `reports/phase4/real_ozz_inference/false_detection/` — для ложных срабатываний

Имя файла графика: `{md5_hash}[_bus{N}].png` (суффикс bus только при мульти-секциях).

#### Запуск:
```bash
# Все файлы
python scripts/phase4_experiments/real_ozz/inference_real_ozz.py --checkpoint .../best_model.pt

# Только подтверждённые ОЗЗ, первые 50
python scripts/phase4_experiments/real_ozz/inference_real_ozz.py --checkpoint .../best_model.pt --subset confirmed --max-files 50

# Конкретная секция
python scripts/phase4_experiments/real_ozz/inference_real_ozz.py --checkpoint .../best_model.pt --bus 1
```

**Статус:** Реализовано, ожидает завершения обучения модели. По результатам запуска будут получены графики inference и статистика обнаружений.

### 11.4 Ожидаемые результаты и анализ

После запуска inference будет доступна следующая информация:
- **Detection Rate** на confirmed_ozz: доля файлов, где модель обнаружила ОЗЗ (max P > 0.5)
- **False Positive Rate** на false_detection: доля файлов, где модель ошибочно нашла ОЗЗ
- **Визуальный анализ**: типы дуг на подтверждённых ОЗЗ vs предсказания модели
- Статистика сохраняется в `inference_stats.json`

---

## 12. Интерпретируемость KAN-активаций (Этап 10.2)

Цель — понять, **на какие признаки и цепочки** модель опирается при классификации каждого из 4 типов ОЗЗ. Это ключевой этап для научной публикации: интерпретируемость KAN — одно из заявленных преимуществ архитектуры.

### 12.1 Channel Dropout Probing (Feature Importance)
**Идея:** Зануляем группы входных каналов (из 220) и измеряем падение метрик — это даёт "важность" группы для каждого класса.

**Группы для зануления:**
| Группа | Каналы | Вопрос |
|--------|--------|--------|
| Нулевые (IN) | IN_h1..h9, IN_lh* (26 ch) | Насколько критичен ток НП? |
| Нулевые (UN) | UN_h1..h9, UN_lh* (26 ch) | Насколько критично напряжение НП? |
| Все нулевые | IN + UN (52 ch) | Можно ли обойтись без 3I0 и 3U0? |
| Фазные токи (IA+IB+IC) | 78 ch | Насколько важны токи линии? |
| Фазные напряжения | 78 ch | Насколько важны напряжения? |
| Только h1 | Все не-h1 гармоники | Достаточно ли одной 1-й гармоники? |
| Только симметричные | 12 ch: I1,I2,I0,U1,U2,U0 | Хватит ли 12 каналов вместо 220? |
| Sub-periods | lh2,lh4,lh6,lh10 (64 ch) | Нужны ли низкочастотные компоненты? |
| Одиночные сигналы | IA, IB, IC, UA, UB, UC по 26 ch | Какой конкретный сигнал критичен? |

**Метрика:** Per-class F1 на val-выборке (SimOZZ) + macro-F1.

**Ожидания:**
- Для Stable SPGF: скорее всего критичны IN и UN (постоянное смещение нулевой)
- Для ДПОЗЗ: скорее всего критичны высшие гармоники и sub-periods (быстрые пробои)
- IA/IB/IC менее важны, чем IN/UN (потому что фазные токи при ОЗЗ в изолированной сети малы)

**Скрипт:** `scripts/phase4_experiments/interpretability/channel_dropout_probing.py`

### 12.2 Gradient-based Attribution (Saliency Maps)
**Идея:** Для каждого класса вычисляем $\nabla_{x} \hat{y}_c \cdot x$ (Gradient×Input) — это показывает, какие входные каналы в каких зонах сильнее всего влияют на предсказание.

**Визуализация:**
- Heatmap (220 каналов × 72 зоны) per-class, усреднённый по val-выборке
- Сгруппированный heatmap (8 сигналов × 13 гармоник × 2) per-class
- Top-20 каналов per-class (bar chart)

**Скрипт:** `scripts/phase4_experiments/interpretability/gradient_attribution.py`

### 12.3 Attention Weight Analysis
**Идея:** Из `ComplexMultiheadAttention` извлекаем веса внимания — какие временные зоны «смотрят» на какие. Для каждого класса паттерн внимания должен быть разным:
- **Stable SPGF**: широкое рассеянное внимание (постоянный процесс)
- **ДПОЗЗ**: сфокусированное внимание на моменты пробоя (spike-паттерн)

**Визуализация:** Attention heatmaps (T×T) per-head per-layer, усреднённые по классам.

### 12.4 KAN Activation Analysis
**Идея:** Forward-hooks на FastKAN-слои → для каждого нейрона строим φ(x) с наложенной гистограммой входов per-class.

**Точки снятия:**
- `model.stem.rphysics_kan` — какие polar-представления формирует stem
- `model.encoder_blocks[i].ffn.kan_layer` — KAN-нелинейности в FFN
- `model.cls_head.kan` — финальный KAN перед классификацией

**Визуализация:**
- ТОП-K нейронов: φ(x) curve + per-class distribution overlay
- Идентификация формы: пороговая, S-образная, линейная → физический смысл

### 12.5 Темпоральная визуализация
Heatmap: 72 зоны × ТОП-K каналов — момент «обнаружения» ОЗЗ моделью. Помогает увидеть:
- Через сколько зон после начала аварии модель «уверена»
- Различие скорости детекции между типами дуг (Stable быстрее? Belyakov медленнее?)

### 12.6 Pruning (прореживание) KAN-сетей
**Цель:** Уменьшить сложность модели, обнулив неинформативные связи.

**Подход для KAN:**
В отличие от стандартного pruning весов (MLP), для KAN-сетей прореживание означает
**обнуление целых рёбер** (B-spline функций) между узлами. Если амплитуда активации
ребра мала для всех входных данных, ребро можно безопасно удалить.

**План:**
1. **Анализ важности рёбер:**
   - Для каждого ребра (i→j) в KAN-слое: средняя абсолютная активация |φ_ij(x)| по val-выборке
   - Ранжирование рёбер по важности
2. **Мягкий pruning (L1-регуляризация):**
   - Добавить L1-штраф на амплитуды RBF-весов → рёбра с низкой важностью стремятся к нулю
   - Fine-tuning с L1 → естественное прореживание
   - Примечание: А так можно бы обучить модель, чуть другая ведь получится, она будет стараться использовать как можно меньше весов, и потом такую модель можно будет уже жёстким pruning проходить. Это пока мысли в слух, но как о мне, интересные.
3. **Жёсткий pruning (обнуление + пересборка):**
   - Установить порог: удалить рёбра с |φ| < ε
   - Пересобрать сеть с меньшим числом рёбер (новые слои FastKANLayer с уменьшенными размерностями)
   - Fine-tuning пересобранной сети (Вопрос, а не будут ли потеряны веса? Ведь может получится, что и входа некоторые не нужны будут. Т.е. можно будет как будто удалять вообще часть, т.е. пересобрать, но сохранив "веса", а далее, можно будет и попробовать дообучить её в общем-то.)
4. **Оценка:**
   - Сравнение: оригинал vs pruned (F1, latency, число параметров)
   - Визуализация: какие рёбра были удалены → физическая интерпретация

**Инструменты:** Частично реализовано в `osc_tools/visualization/kan_plot.py` (Фаза 2.6).
Потребуется адаптация под архитектуру Physical KAN-Transformer.

---

## 13. Статистика по реальным ОЗЗ (Этап 10.4)

После inference на реальных COMTRADE файлах можно собрать развёрнутую статистику, которую невозможно получить иным способом (разметка типа дуги в реальных данных отсутствует).

### 13.1 Файл-уровневая статистика
Для каждого файла:
- `max_prob[c]` — максимальная вероятность каждого из 4 классов по всем зонам
- `detected[c]` — бинарный флаг (max_prob > threshold)
- `any_ozz` — обнаружено ли хотя бы одно ОЗЗ любого типа

**Агрегирование:**
- **Detection Rate** на confirmed_ozz: доля файлов с `any_ozz = True`
- **False Positive Rate** на false_detection: доля файлов с `any_ozz = True`
- **Per-class distribution** confirmed: сколько файлов содержат каждый тип ОЗЗ
- **Confusion matrix**: TP / FP / FN / TN по файлам

### 13.2 Зон-уровневая статистика с темпоральной коррекцией

**Проблема:** Stable SPGF размечается широкими сплошными зонами, а ДПОЗЗ — узкими точечными всплесками (моменты пробоя). Прямое сравнение числа зон даёт несправедливое преимущество Stable.

**Решение — расширение зон (zone expansion):**
Для расчёта статистик (НЕ для обучения) каждый обнаруженный всплеск ДПОЗЗ расширяется на ±1 период промышленной частоты (±8 зон при stride_fraction=8). Это агрегирует серию коротких пробоёв в единое «событие».

**Алгоритм:**
1. Для каждого класса: бинаризовать prob > threshold → маска зон
2. Применить morphological dilation с ядром = 2 × stride_fraction (= 16 зон = 1 период)
3. Подсчитать connected components → «события»
4. Для каждого события: начало, конец, длительность (в мс), средняя вероятность

**Метрики:**
- Число событий per-class per-file
- Средняя длительность события per-class (мс)
- Доля осциллограммы, покрытая событием per-class (%)
- Multi-class co-occurrence: сколько файлов содержат >1 типа ОЗЗ одновременно

### 13.3 Распределение уверенности модели
- Гистограмма `max_prob[c]` для confirmed vs false_detection
- Помогает оценить калибровку: разделяет ли модель уверенно «есть ОЗЗ» от «нет»?
- ROC-подобная кривая для порога detection

### 13.4 Темпоральные паттерны
- Когда в осциллограмме начинается ОЗЗ (relative onset): в начале, середине или конце?
- Бывают ли множественные эпизоды ОЗЗ в одном файле?
- Как часто разные типы дуг чередуются внутри одного файла?

### 13.5 Итоговый отчёт
Весь собранный набор статистик сохраняется в:
- `reports/phase4/real_ozz_statistics/real_ozz_statistics.json` — машиночитаемый файл
- `reports/phase4/real_ozz_statistics/per_file_statistics.json` — per-file статистика для дальнейшего анализа
- `reports/phase4/real_ozz_statistics/` — графики (гистограммы, pie-charts, temporal heatmaps)

**Скрипт:** `scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py`

---

## 14. Чеклист текущего состояния

| # | Задача | Статус | Файл/ссылка |
|---|--------|--------|-------------|
| 1 | Lazy-загрузчик SimOZZ | ✅ | `osc_tools/ml/simulated_ozz_dataset.py` |
| 2 | Комбинированное обучение (5 классов) | ✅ | `osc_tools/ml/balanced_dataset.py` |
| 3 | Стратифицированный split | ✅ | `osc_tools/data_management/sim_ozz_split.py` |
| 4 | NeutralChannelDropout | ✅ | `osc_tools/ml/augmentation.py` |
| 5 | PhaseCurrentDropout | ✅ | `osc_tools/ml/augmentation.py` |
| 6 | Масштабирование SIM_NOMINAL (NormOsc-совместимое) | ✅ | `osc_tools/ml/simulated_ozz_dataset.py` |
| 7 | Обучение модели | 🔄 дообучение | `scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py` |
| 8 | Оценка на SimOZZ val | ✅ скрипт готов | `scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py` |
| 9 | Inference на real_OZZ | ✅ скрипт готов | `scripts/phase4_experiments/real_ozz/inference_real_ozz.py` |
| 10 | Channel Dropout Probing | ✅ скрипт готов | `scripts/phase4_experiments/interpretability/channel_dropout_probing.py` |
| 11 | Gradient Attribution | ✅ скрипт готов | `scripts/phase4_experiments/interpretability/gradient_attribution.py` |
| 12 | Статистика по реальным ОЗЗ | ✅ скрипт готов | `scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py` |

**Примечание:** Скрипты 8-12 готовы к запуску, ожидают завершения дообучения модели. 13 тренировочных запусков уже выполнено (последний: 28.04.2026). Графики inference на реальных данных уже визуально просмотрены.

---

## 15. Инструкция по запуску оставшихся опытов

Все команды выполняются через conda из корня проекта.
Подставьте `CHECKPOINT` = путь к лучшему чекпоинту (например `experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260428_174217/best_model.pt`).

### 15.1 Шаг 1: Оценка на SimOZZ val

```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py \
    --checkpoint CHECKPOINT
```

**Что получим:**
- `reports/phase4/sim_ozz_eval/confusion_matrix.png` — confusion matrices
- `reports/phase4/sim_ozz_eval/roc_curves.png` — ROC-кривые
- `reports/phase4/sim_ozz_eval/prob_distributions.png` — распределения P(class)
- `reports/phase4/sim_ozz_eval/training_curves.png` — кривые обучения
- Консольный отчёт: per-class F1/Precision/Recall, macro-F1, ROC-AUC, boundary metrics

### 15.2 Шаг 2: Inference на реальных ОЗЗ (графики)

```bash
# Все файлы (~1800 штук, ~30-60 мин)
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/real_ozz/inference_real_ozz.py \
    --checkpoint CHECKPOINT

# Или для быстрой проверки — только первые 50 confirmed
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/real_ozz/inference_real_ozz.py \
    --checkpoint CHECKPOINT --subset confirmed --max-files 50
```

**Что получим:**
- `reports/phase4/real_ozz_inference/confirmed_ozz/*.png` — графики confirmed
- `reports/phase4/real_ozz_inference/false_detection/*.png` — графики false_detection
- `reports/phase4/real_ozz_inference/inference_stats.json` — сводка обнаружений

### 15.3 Шаг 3: Статистика по реальным ОЗЗ

```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py \
    --checkpoint CHECKPOINT
```

**Что получим:**
- `reports/phase4/real_ozz_statistics/real_ozz_statistics.json` — агрегированная статистика
- `reports/phase4/real_ozz_statistics/per_file_statistics.json` — per-file метрики
- `reports/phase4/real_ozz_statistics/confidence_distributions.png` — гистограммы P(class)
- `reports/phase4/real_ozz_statistics/detection_pie.png` — Detection Rate (pie chart)
- `reports/phase4/real_ozz_statistics/per_class_detection.png` — per-class bar chart

### 15.4 Шаг 4: Channel Dropout Probing (интерпретируемость)

```bash
# Полный прогон (~30 мин, 31 группа × val inference)
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/interpretability/channel_dropout_probing.py \
    --checkpoint CHECKPOINT

# Быстрый тест (200 файлов вместо всех)
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/interpretability/channel_dropout_probing.py \
    --checkpoint CHECKPOINT --max-files 200
```

**Что получим:**
- `reports/phase4/interpretability/channel_dropout_probing.json` — ΔF1 для каждой группы
- `reports/phase4/interpretability/channel_dropout_importance.png` — ранжированный bar chart
- `reports/phase4/interpretability/channel_dropout_heatmap.png` — heatmap per-class

### 15.5 Шаг 5: Gradient Attribution (saliency maps)

```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/interpretability/gradient_attribution.py \
    --checkpoint CHECKPOINT --max-files 200
```

**Что получим:**
- `reports/phase4/interpretability/gradient_attribution.json` — ТОП-20 каналов per-class
- `reports/phase4/interpretability/gradient_attribution_arrays.npz` — полные массивы (220×72 per-class)
- `reports/phase4/interpretability/gradient_signal_heatmap.png` — важность 8 сигналов
- `reports/phase4/interpretability/gradient_temporal_heatmap.png` — ТОП-30 каналов × 72 зоны

### 15.6 Порядок запуска (рекомендуемый)

1. **Оценка SimOZZ** (15.1) — получить F1/AUC, убедиться что модель работает
2. **Inference real_OZZ** (15.2) — графики для визуального анализа
3. **Статистика real_OZZ** (15.3) — количественный анализ detection rate
4. **Channel Dropout** (15.4) — понять, какие каналы важны
5. **Gradient Attribution** (15.5) — детальные saliency maps

Шаги 2-3 можно запускать параллельно (*разные скрипты, не конфликтуют*).
Шаги 4-5 тоже параллельны друг другу, но каждый требует GPU.

---

## 16. Анализ сохранённых результатов

Все результаты сохраняются в машиночитаемом формате (JSON, NPZ) и могут быть
загружены для дополнительного анализа без повторного inference.

### 16.1 Анализ статистики по реальным ОЗЗ

```python
import json
import numpy as np

# Загрузка агрегированной статистики
with open('reports/phase4/real_ozz_statistics/real_ozz_statistics.json') as f:
    agg = json.load(f)

# Detection Rate
print(f"Confirmed: DR={agg['confirmed']['detection_rate_pct']:.1f}%")
print(f"False Det: FPR={agg['false_detection']['detection_rate_pct']:.1f}%")

# Per-class breakdown
for cls, info in agg['confirmed']['per_class_detection'].items():
    print(f"  {cls}: {info['count']} файлов ({info['pct']:.1f}%)")

# Per-file данные (для собственного анализа)
with open('reports/phase4/real_ozz_statistics/per_file_statistics.json') as f:
    per_file = json.load(f)

# Фильтр: файлы с co-occurrence >1 типа ОЗЗ
multi_type = [f for f in per_file if f.get('n_types_detected', 0) > 1]
print(f"Файлов с >1 типом ОЗЗ: {len(multi_type)}")
```

### 16.2 Анализ Channel Dropout Probing

```python
import json

with open('reports/phase4/interpretability/channel_dropout_probing.json') as f:
    cdp = json.load(f)

# Baseline
print(f"Baseline Macro-F1: {cdp['baseline']['macro_f1']:.4f}")

# ТОП-5 самых важных групп
for name in cdp['ranked'][:5]:
    g = cdp['groups'][name]
    print(f"  {name}: Δ={g['delta_macro_f1']:+.4f} ({g['num_channels_masked']} ch)")

# Per-class анализ: для какого класса UN важнее всего?
un = cdp['groups']['signal_UN']
for cls, delta in un['delta_per_class_f1'].items():
    print(f"  UN → {cls}: ΔF1={delta:+.4f}")
```

### 16.3 Анализ Gradient Attribution

```python
import json
import numpy as np

# JSON результаты
with open('reports/phase4/interpretability/gradient_attribution.json') as f:
    ga = json.load(f)

# ТОП-5 каналов для каждого класса
for cls, channels in ga['top_channels'].items():
    print(f"\n{cls}:")
    for ch in channels[:5]:
        print(f"  {ch['channel']}: {ch['value']:.6f}")

# NPZ массивы для кастомного анализа
data = np.load('reports/phase4/interpretability/gradient_attribution_arrays.npz')
attr = data['attribution_per_class']   # (4, 220) — per-class × per-channel
attr_t = data['attribution_temporal']   # (4, 220, 72) — полная карта
counts = data['counts_per_class']       # (4,) — число зон

# Пример: какой класс больше всего зависит от IN?
IN_indices = list(range(3*26, 4*26))  # каналы 78..103
for c in range(4):
    in_importance = attr[c, IN_indices].sum()
    total = attr[c].sum()
    print(f"  {ga['counts'][list(ga['counts'].keys())[c]]}: "
          f"IN = {in_importance/total*100:.1f}% от total attribution")
```

### 16.4 Файловая структура результатов

```
experiments/phase4/sim_ozz_finetune_<model>_<timestamp>/
├── config.json                              # Конфигурация обучения
├── split.json                               # Train/val split
├── training_log.jsonl                       # Метрики по эпохам
└── best_model.pt                            # Лучшая модель

reports/phase4/
├── sim_ozz_eval/                            # Оценка на SimOZZ (§15.1)
│   ├── sim_ozz_evaluation.json
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── prob_distributions.png
├── real_ozz_inference/                      # Inference графики (§15.2)
│   ├── confirmed_ozz/*.png
│   ├── false_detection/*.png
│   └── inference_stats.json
├── real_ozz_statistics/                     # Статистика (§15.3)
│   ├── real_ozz_statistics.json              # Агрегированная
│   ├── per_file_statistics.json              # Per-file
│   ├── confidence_distributions.png
│   ├── detection_pie.png
│   └── per_class_detection.png
└── interpretability/                        # Интерпретируемость (§15.4-15.5)
    ├── channel_dropout_probing.json
    ├── channel_dropout_importance.png
    ├── channel_dropout_heatmap.png
    ├── gradient_attribution.json
    ├── gradient_attribution_arrays.npz
    ├── gradient_signal_heatmap.png
    └── gradient_temporal_heatmap.png
```

---

## 17. Полученные результаты обученной модели PhysicalKANTransformer

Раздел фиксирует фактические результаты завершённого прогона модели
`PhysicalKANTransformer` на данных `Simulated_OZZ_v1` и последующего переноса на
реальные COMTRADE-осциллограммы. Эти выводы сформулированы в стиле, пригодном для
дальнейшей переработки в научную статью.

### 17.1 Конфигурация финальной модели

Основной чекпоинт эксперимента:
`experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260428_174217/latest_checkpoint.pt`.

Ключевые параметры модели и обучения:

| Параметр | Значение |
|----------|----------|
| Архитектура | `PhysicalKANTransformer` |
| Число параметров | ~263 тыс. |
| Вход | 220 спектральных каналов × 72 зоны |
| `d_model` | 48 |
| Число слоёв Transformer | 6 |
| Число голов внимания | 4 |
| KAN grid size | 5 |
| Голова классификации | KAN (`cls_head_type='kan'`) |
| Аугментация | инверсия, масштабирование, shuffle фаз, dropout фазного тока, dropout IN/UN |
| Оптимизируемая задача | zone-level multi-label классификация 4 типов ОЗЗ |
| Число эпох | 50 |
| Лучшая эпоха по validation F1 | 42 |
| Лучший validation F1 при обучении | 0.9609 |
| Лучший validation ROC-AUC при обучении | 0.99995 |

Модель имеет малый размер по числу параметров, но использует физически
ориентированное представление входов: спектральные признаки, фазовые углы,
симметричные составляющие и KAN-нелинейности. Это важно для дальнейшего сравнения
с baseline-моделями: сравнивать необходимо не только точность, но и сложность,
латентность и устойчивость переноса на реальные осциллограммы.

### 17.2 Оценка на SimOZZ validation subset

Оценка выполнена на 200 файлах SimOZZ, что соответствует 24 200 окнам. В отличие от
первоначальной версии оценки, window-level агрегация выполняется через `max` по
зонам, что соответствует физическому вопросу «есть ли данный тип ОЗЗ хотя бы в
одной зоне окна». Такая агрегация особенно важна для ДПОЗЗ, где аварийный процесс
может занимать малое число зон.

Сводные метрики при пороге 0.5:

| Метрика | Значение |
|---------|----------|
| Macro-F1 (window-level) | 0.9865 |
| Macro-Precision | 0.9746 |
| Macro-Recall | 0.9988 |
| Macro ROC-AUC | 0.99993 |
| Exact match | 0.9819 |
| Zone-level Macro-F1 | 0.9525 |
| Zone-level Exact match | 0.9881 |
| Latency | ~24.3 мс/sample |

Per-class результаты:

| Класс | F1 | Precision | Recall | ROC-AUC | TP | FP | FN | TN |
|-------|----|-----------|--------|---------|----|----|----|----|
| Stable SPGF | 0.9961 | 0.9928 | 0.9994 | 0.99990 | 16232 | 118 | 9 | 7841 |
| Petersen AIGF | 0.9906 | 0.9828 | 0.9986 | 0.99998 | 3538 | 62 | 5 | 20595 |
| Peters-Slepyan AIGF | 0.9797 | 0.9625 | 0.9976 | 0.99987 | 3386 | 132 | 8 | 20674 |
| Belyakov AIGF | 0.9794 | 0.9603 | 0.9994 | 0.99998 | 3118 | 129 | 2 | 20951 |

Оптимизация индивидуальных порогов по классам увеличивает macro-F1 до 0.9936:

| Класс | Оптимальный порог | F1 при оптимальном пороге |
|-------|-------------------|--------------------------|
| Stable SPGF | 0.7646 | 0.9966 |
| Petersen AIGF | 0.6960 | 0.9940 |
| Peters-Slepyan AIGF | 0.9214 | 0.9887 |
| Belyakov AIGF | 0.8332 | 0.9950 |

**Вывод.** На синтетическом validation subset модель уверенно разделяет все 4
типа ОЗЗ. Значения ROC-AUC близки к 1.0 для всех классов, что означает хорошее
ранжирование вероятностей. Основной запас улучшения связан не с самой
способностью модели обнаруживать классы на SimOZZ, а с переносом на реальные
осциллограммы и калибровкой порогов под реальные отрицательные примеры.

### 17.3 Перенос на реальные COMTRADE-осциллограммы

Inference на реальных осциллограммах обработал 1370 записей из 1855. По итоговой
статистике валидными оказались 1365 записей типа «файл/секция»:

| Группа | Записей | ОЗЗ обнаружено | ОЗЗ не обнаружено | Detection Rate |
|--------|---------|----------------|-------------------|----------------|
| Confirmed OZZ | 477 | 432 | 45 | 90.6% |
| False detection / no OZZ | 888 | 437 | 451 | 49.2% |

Per-class detection на подтверждённых ОЗЗ:

| Предсказанный тип | Count | Доля confirmed |
|-------------------|-------|----------------|
| Stable SPGF | 239 | 50.1% |
| Petersen AIGF | 123 | 25.8% |
| Peters-Slepyan AIGF | 213 | 44.7% |
| Belyakov AIGF | 281 | 58.9% |

Per-class detection на группе `false_detection`:

| Предсказанный тип | Count | Доля false_detection |
|-------------------|-------|----------------------|
| Stable SPGF | 136 | 15.3% |
| Petersen AIGF | 53 | 6.0% |
| Peters-Slepyan AIGF | 110 | 12.4% |
| Belyakov AIGF | 305 | 34.3% |

Для confirmed-группы медианные максимальные вероятности составили:
Stable — 0.508, Petersen — 0.073, Peters-Slepyan — 0.328, Belyakov — 0.760.
Для false_detection-группы медианы ниже: Stable — 0.097, Petersen — около 0,
Peters-Slepyan — около 0, Belyakov — 0.098. Однако 90-й перцентиль у части классов
остаётся высоким даже на false_detection: Stable ≈ 1.0, Peters-Slepyan ≈ 0.896,
Belyakov ≈ 0.963. Это указывает на наличие «сложных отрицательных» реальных
примеров, которые модель воспринимает как похожие на ОЗЗ.

Co-occurrence по confirmed-группе:

| Число типов ОЗЗ в файле | Count | Доля |
|-------------------------|-------|------|
| 0 типов | 45 | 9.4% |
| 1 тип | 138 | 28.9% |
| 2 типа | 182 | 38.2% |
| 3 типа | 94 | 19.7% |
| 4 типа | 18 | 3.8% |

Co-occurrence по false_detection-группе:

| Число типов ОЗЗ в файле | Count | Доля |
|-------------------------|-------|------|
| 0 типов | 451 | 50.8% |
| 1 тип | 296 | 33.3% |
| 2 типа | 119 | 13.4% |
| 3 типа | 18 | 2.0% |
| 4 типа | 4 | 0.5% |

**Интерпретация.** Высокий detection rate на confirmed OZZ (90.6%) показывает,
что признаки, выученные на RTDS-симуляциях, частично переносятся на реальные
осциллограммы. Одновременно высокая доля срабатываний на false_detection (49.2%)
показывает ограничение текущего обучения: модель хорошо знает разные формы ОЗЗ,
но недостаточно хорошо знает «не-ОЗЗ» режимы реальной сети. Это не является
противоречием: синтетический датасет целенаправленно покрывает аварии типа ОЗЗ,
а разнообразие реальных переходных процессов без ОЗЗ существенно шире.

### 17.4 Интерпретируемость: Channel Dropout Probing

Для анализа важности входных признаков были занулены группы каналов, после чего
измерялось падение Macro-F1 на validation subset. Baseline для данного анализа:
Macro-F1 = 0.9525.

Наиболее значимые группы каналов:

| Занулённая группа | Каналов | Macro-F1 после зануления | Δ Macro-F1 |
|-------------------|---------|--------------------------|------------|
| Все phase-polar признаки | 208 | 0.0000 | 0.9525 |
| Фазные напряжения UA/UB/UC | 78 | 0.0120 | 0.9405 |
| Все амплитуды | 110 | 0.0248 | 0.9277 |
| 1-я гармоника h1 | 16 | 0.0916 | 0.8609 |
| Все признаки кроме h1 | 192 | 0.1716 | 0.7809 |
| Симметричные составляющие | 12 | 0.1729 | 0.7796 |
| Гармоники h2-h9 | 128 | 0.1836 | 0.7689 |
| Все углы | 110 | 0.2036 | 0.7489 |
| UA | 26 | 0.4524 | 0.5001 |
| UC | 26 | 0.5516 | 0.4009 |
| UB | 26 | 0.5861 | 0.3664 |
| h3 | 16 | 0.6507 | 0.3018 |
| h2 | 16 | 0.7615 | 0.1910 |
| Sub-period признаки | 64 | 0.8253 | 0.1272 |

Отдельно важно, что зануление всех нейтральных каналов IN+UN приводит к малому
падению Macro-F1: Δ=0.0056. Это не означает физическую неважность нулевой
последовательности. Скорее, модель научилась восстанавливать информацию о
смещении нейтрали через фазные напряжения и их угловые соотношения. Такой эффект
согласуется с применённой аугментацией: в обучении модель регулярно видела
режимы с отсутствующими IN и/или UN, поэтому была вынуждена использовать
избыточность фазных напряжений.

**Научный вывод.** Критически важными для классификации оказались фазные
напряжения, первая гармоника и угловые признаки. Для ДПОЗЗ дополнительно заметна
роль h2, h3 и sub-period признаков, что физически согласуется с короткими
дуговыми пробоями и быстрыми переходными процессами.

### 17.5 Интерпретируемость: Gradient×Input Attribution

Gradient×Input attribution показывает, какие признаки сильнее всего влияют на
логиты соответствующих классов. Число положительных зон, участвовавших в
усреднении:

| Класс | Число положительных зон |
|-------|--------------------------|
| Stable SPGF | 468 450 |
| Petersen AIGF | 30 560 |
| Peters-Slepyan AIGF | 68 026 |
| Belyakov AIGF | 23 995 |

ТОП сигналов по суммарной важности:

| Класс | Наиболее важные сигналы |
|-------|--------------------------|
| Stable SPGF | UB, UC, UA, UN |
| Petersen AIGF | UB, UN, UC, UA |
| Peters-Slepyan AIGF | UB, UC, UA, UN |
| Belyakov AIGF | UB, UC, UA, UN |

ТОП индивидуальных каналов:

| Класс | Наиболее важные признаки |
|-------|--------------------------|
| Stable SPGF | U1_angle, UB_h1_angle, U1_mag, U0_angle, UA_h1_mag |
| Petersen AIGF | UB_h1_angle, U1_angle, U0_angle, U1_mag, U0_mag |
| Peters-Slepyan AIGF | U1_angle, U0_angle, UB_h1_angle, UC_h1_angle, U1_mag |
| Belyakov AIGF | UB_h1_angle, UC_h1_angle, U1_mag, U1_angle, UC_h1_mag |

**Интерпретация.** Attribution подтверждает вывод Channel Dropout Probing:
модель опирается прежде всего на геометрию напряжений, углы первой гармоники,
прямую и нулевую последовательности. Для исследования РЗА это положительный
результат: признаки, выделенные моделью как важные, имеют физический смысл и
связаны с классическими представлениями об ОЗЗ — смещение нейтрали, изменение
фазных напряжений и наличие переходных гармонических составляющих.

### 17.6 Общий вывод по этапу 4.5

На синтетических данных RTDS модель `PhysicalKANTransformer` демонстрирует
высокое качество классификации всех четырёх типов ОЗЗ: window-level Macro-F1 =
0.9865 при стандартном пороге 0.5 и до 0.9936 при индивидуальной оптимизации
порогов. При переносе на реальные осциллограммы модель обнаруживает ОЗЗ в 90.6%
подтверждённых случаев, что показывает работоспособность переноса физических
признаков из RTDS-среды на реальные данные. Главным ограничением остаётся высокая
доля срабатываний на группе false_detection (49.2%), что указывает на
необходимость дальнейшего обучения на реальных отрицательных примерах и более
строгой калибровки порогов.

С точки зрения интерпретируемости модель не выглядит «чёрным ящиком»: и Channel
Dropout Probing, и Gradient×Input attribution показывают, что решающую роль играют
физически осмысленные признаки — фазные напряжения, углы первой гармоники,
прямая/нулевая последовательности и низкие гармоники переходного процесса.

---

## 18. Ограничения, будущие задачи и план развития экспериментов

### 18.1 Главная текущая проблема: ложные срабатывания на реальных не-ОЗЗ

Существенная проблема текущей модели — частое обнаружение ОЗЗ в осциллограммах,
которые в отчёте относятся к false_detection / no OZZ. Наиболее вероятная причина
состоит в дисбалансе опыта модели: она обучалась преимущественно на
симулированных ОЗЗ и видела ограниченное разнообразие реальных режимов без ОЗЗ.
Даже с добавлением реальных «нормальных» окон модель не видела достаточного
числа сложных отрицательных примеров: пусков, коммутаций, насыщений ТТ,
нестандартных переходных процессов и шумовых/неполных каналов.

Это ограничение имеет важное практическое значение: для РЗА модель должна быть
не только чувствительной к ОЗЗ, но и устойчивой к событиям, которые похожи на ОЗЗ
по отдельным признакам, но физически ими не являются.

### 18.2 Предлагаемый путь: дообучение на размеченных реальных отрицательных примерах

Перспективный следующий шаг — вручную разметить реальные осциллограммы на уровне
`есть ОЗЗ / нет ОЗЗ` и использовать их для дообучения. При этом не требуется
ручная разметка типа дуги: тип дуги остаётся исследовательской задачей, а ручная
разметка может задавать только бинарный факт наличия ОЗЗ.

Предлагаемая схема:

1. Сформировать набор реальных COMTRADE-осциллограмм с бинарной разметкой:
  - `real_any_ozz = 1` — экспертно подтверждено наличие ОЗЗ;
  - `real_any_ozz = 0` — экспертно подтверждено отсутствие ОЗЗ.
2. Для `real_any_ozz = 0` задавать target `[0,0,0,0]` для всех зон.
3. Для `real_any_ozz = 1` использовать только auxiliary binary loss `any_ozz`,
  не навязывая тип дуги.
4. Сохранить SimOZZ как источник типовой multi-label разметки по 4 классам.
5. Дообучать модель в multi-task режиме:
  - synthetic loss: 4-классовая multi-label классификация типов ОЗЗ;
  - real loss: бинарная классификация `any_ozz` для реальных данных;
  - hard-negative loss: усиленный вес для реальных отрицательных примеров,
    где текущая модель даёт ложные срабатывания.

Такой подход позволит бороться именно с примерами, которые модель ранее не видела,
не разрушая знания о типах ОЗЗ, полученные на RTDS-симуляциях.

### 18.3 Active learning сценарий

После дообучения полезно прогонять модель по большому массиву реальных данных и
сортировать осциллограммы по максимальной вероятности ОЗЗ и по неопределённости.
На ручную проверку в первую очередь стоит отправлять:

- false_detection с высоким `max P(OZZ)`;
- confirmed OZZ с низким `max P(OZZ)`;
- файлы с предсказанием нескольких типов ОЗЗ одновременно;
- файлы, где разные модели baseline дают противоречивые ответы.

Это позволит итеративно расширять реальный датасет наиболее информативными
примерами, а не размечать случайную выборку.

### 18.4 Задача для кода: inference и экспорт разметки для конкретного COMTRADE

Нужна отдельная функция/CLI-команда для обработки одной конкретной осциллограммы,
чтобы можно было быстро перерисовать график, проверить отдельный случай или
сформировать CSV для ручного анализа.

Реализованный интерфейс:

```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/real_ozz/export_single_comtrade_marking.py \
  --checkpoint CHECKPOINT \
  --cfg path/to/file.cfg \
  --bus 1 \
  --output reports/phase4/single_marking/file_bus1.csv \
  --plot
```

Дополнительно поддерживается `--bus auto`: скрипт сначала пробует найти секции
через отчёт real_OZZ, а если файл новый и не входит в отчёт — перебирает типовые
секции `1..5`. Для нескольких секций можно указать `--bus 1,2`.

Желаемый CSV-выход:

| Колонка | Содержание |
|---------|------------|
| `time_s` | время отсчёта |
| `IA`, `IB`, `IC`, `IN` | нормированные/масштабированные токи для графика |
| `UA`, `UB`, `UC`, `UN` | нормированные/масштабированные напряжения для графика |
| `P_Stable_SPGF` | вероятность Stable SPGF |
| `P_Petersen_AIGF` | вероятность Petersen AIGF |
| `P_PetersSlepyan_AIGF` | вероятность Peters-Slepyan AIGF |
| `P_Belyakov_AIGF` | вероятность Belyakov AIGF |
| `coverage` | число окон модели, покрывающих точку |
| `pred_any_ozz` | бинарная разметка по выбранному порогу |

Также сохраняется sidecar JSON с метаданными: путь чекпоинта, путь COMTRADE,
секция, частота дискретизации, найденные каналы, максимальная вероятность и факт
срабатывания. Скрипт переиспользует `load_comtrade_raw()`, `mark_real_oscillogram()`
и `plot_real_ozz_marking()` из `inference_real_ozz.py`, поэтому не дублирует
pipeline обработки COMTRADE.

### 18.5 Baseline-модели для статьи

Для статьи необходимо сравнить PhysicalKANTransformer не только с «любой» простой
моделью, а с архитектурами сопоставимой постановки и сложности.

План baseline-экспериментов:

1. **Spectral BaselineTransformer**
  - вход: те же 220 спектральных каналов × 72 зоны;
  - архитектура: обычный Transformer без KAN и физических модификаций;
  - цель: понять вклад физически ориентированных блоков и KAN при одинаковом
    представлении входа.
2. **Raw Instantaneous Transformer**
  - вход: мгновенные значения до FFT, например 8 каналов × окно 10 периодов;
  - цель: проверить, насколько полезна спектральная физическая
    предобработка по сравнению с end-to-end обучением на сигнале.
3. **PhysicsTransformer без KAN**
  - вход: 220 спектральных каналов × 72 зоны;
  - сохранить физические блоки: угловое управление, комплексно-подобное
    восприятие, операции взаимодействия признаков;
  - заменить KAN-слои на MLP/Linear/GEGLU блоки сопоставимой размерности;
  - цель: отделить вклад KAN-нелинейностей от вклада физической архитектуры.

Для первых двух spectral-вариантов создан единый runner:

```bash
# Обычный spectral Transformer baseline
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py \
    --exp spectral_baseline --complexity light --epochs 50

# Физическая архитектура без KAN-блоков
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py \
    --exp physical_mlp --complexity light --epochs 50

# Запустить оба последовательно
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py \
    --exp all --complexity light --epochs 50
```

Вариант `physical_mlp` использует новый `PhysicalMLPTransformer`: сохраняет
физический Stem, complex attention и полярную структуру признаков, но заменяет
KAN в Stem на линейный fallback, FFN на MLP и классификационную голову на MLP.

Raw instantaneous baseline пока не включён в runner: для него требуется отдельный
датасет с ресэмплингом 10 периодов сырого сигнала к фиксированной длине и
согласованием с real/no-OZZ примерами. Это отдельная задача, чтобы не ломать
текущий проверенный spectral pipeline.

### 18.6 Как корректно сравнивать модели по сложности

Чтобы сравнение было честным, в таблицу статьи нужно добавить не только F1/ROC-AUC,
но и метрики сложности (+ модели хорошо бы настроить так, чтобы их сложность была соизмерима по итогу, а то вдруг будет слишком большое различие):

| Метрика | Зачем нужна |
|---------|-------------|
| Trainable params | базовая оценка размера модели |
| Model size MB | практический размер чекпоинта |
| FLOPs или MACs на одно окно | вычислительная сложность inference |
| Latency batch=1 на GPU | пригодность для online/РЗА сценария |
| Throughput batch=N | эффективность пакетной обработки архива |
| Peak VRAM | требование к оборудованию |
| `d_model`, layers, heads, FFN/KAN width | архитектурная сопоставимость |
| Число входных каналов и тип входа | spectral/raw/symmetric-only |
| Training budget | эпохи, batch, число seen samples, seed |

Практическое правило: baseline-модели желательно подбирать так, чтобы число
параметров и FLOPs отличались от PhysicalKANTransformer не более чем примерно на
±20-30%. Если модель существенно меньше, её проигрыш может быть связан не с
архитектурной идеей, а с недостаточной ёмкостью. Если существенно больше — её
выигрыш может быть следствием большего числа параметров.

Для автоматического сбора части этих метрик добавлен скрипт:

```bash
C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:\ProgramData\anaconda3 --no-capture-output \
  python scripts/phase4_experiments/evaluation/model_complexity_report.py \
    --checkpoint CHECKPOINT --batch-size 256 --repeats 50
```

Результат сохраняется в `reports/phase4/model_complexity/` и содержит число
параметров, размер чекпоинта, latency batch=1, throughput batch=N и peak VRAM.

### 18.7 Рекомендуемые следующие задачи по коду

1. ✅ Реализован `export_single_comtrade_marking.py` для обработки одного COMTRADE и
  сохранения CSV + PNG.
2. ✅ Добавлен `model_complexity_report.py`, который для любого чекпоинта
  считает параметры, размер, latency, throughput и peak VRAM.
3. 🔄 Подготовлены spectral baseline-конфиги из §18.5 без дублирования
  тренировочного pipeline.
4. ⏳ Добавить raw instantaneous baseline с отдельным датасетом сырого сигнала.
5. ⏳ Добавить режим hard-negative fine-tuning на реальных `no OZZ` осциллограммах.
6. ⏳ Сформировать таблицу результатов для статьи:
  - SimOZZ F1/ROC-AUC;
  - real confirmed detection rate;
  - real false positive rate;
  - latency/params/FLOPs;
  - ключевые признаки интерпретируемости.

### 18.8 Требования к дальнейшим изменениям

При добавлении baseline-моделей и новых режимов обучения важно не ломать текущий
запуск PhysicalKANTransformer. Новые архитектурные варианты должны подключаться
через конфигурацию (`model_type`, параметры входа, тип блока), а не через
копирование всего тренировочного скрипта. Для новых моделей минимально нужны:

- smoke-test forward pass на малом батче;
- проверка формы выхода `(B, 72, 4)`;
- проверка совместимости с текущим DataLoader;
- тест/скрипт загрузки чекпоинта и inference;
- ручной режим запуска, аналогичный текущим скриптам этапа 4.5.

---

## 19. Инструкция по ручному запуску экспериментов (последовательность для статьи)

Все скрипты ниже поддерживают два способа запуска:
1) CLI (python ... --флаг ...);
2) **ручной режим** — запуск файла без аргументов; параметры редактируются прямо в блоке
   # === РУЧНОЙ РЕЖИМ === в конце файла. Ниже описан именно ручной режим.

Все пути относительны корня репозитория `Scientific_research_osc_ML`.

### 19.1 Шаг 1 — § 18.5: baseline-модели (приоритет № 1)

Сначала обучаем базовые модели для сравнения с PhysicalKANTransformer:

- `spectral_baseline` — обычный Transformer на 220 спектральных каналах (без KAN, без physical-stem);
- `physical_mlp` — physical-stem + комплексное внимание, но без KAN-блоков (для оценки вклада KAN).

Файл: `scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py`.

Параметры в блоке `РУЧНОЙ РЕЖИМ`:

```python
EXP = 'all'                  # 'spectral_baseline' | 'physical_mlp' | 'all'
COMPLEXITY = 'light'         # 'light' | 'medium' | 'heavy'  (light ≈ конфиг текущего PhysicalKAN)
EPOCHS = 50
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
MAX_FILES = None             # None = все SimOZZ файлы; для smoke поставьте 100
TRAIN_BATCHES_PER_EPOCH = 256
VAL_BATCHES_PER_EPOCH = 16
RESUME = None                # путь к чекпоинту, если хотите дообучать
RESET_OPTIMIZER = False
```
Запуск (в VS Code — кнопкой Run/F5 или из задачи):

`powershell
C:/ProgramData/anaconda3/python.exe scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py
`

После каждого эксперимента появляется папка `experiments/phase4/sim_ozz_finetune_<Model>_<timestamp>/`
с `best_model.pt` и `latest_checkpoint.pt`. Их используем во всех последующих шагах.

### 19.2 Шаг 2 — § 18.4: экспорт одной COMTRADE для статьи (приоритет № 2)

Для подготовки рисунков статьи. Работает с **любой** обученной моделью этапа 4.5
(`PhysicalKANTransformer`, `PhysicalMLPTransformer`, `BaselineTransformer`):
тип берётся из чекпоинта.

Файл: `scripts/phase4_experiments/real_ozz/export_single_comtrade_marking.py`.

Параметры в блоке `РУЧНОЙ РЕЖИМ`:

```python
CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_.../latest_checkpoint.pt'
CFG = 'data/real_OZZ/osc_comtrade/<имя_файла>.cfg'
BUS = 'auto'           # '1' | '1,2' | 'auto' (берёт bus из overvoltage_report)
OUTPUT = None          # путь CSV; None = автоимя в OUTPUT_DIR
OUTPUT_DIR = None      # None = reports/phase4/single_marking
THRESHOLD = 0.5
MASK_NEUTRAL = False   # True = обнулять IN/UN перед inference
PLOT = True            # сохранять PNG
NO_CROP = False        # True = не кропить PNG
FAST_MODE = False      # True = ускоренный inference (шаг = размер окна)
```
Запуск:

`powershell
C:/ProgramData/anaconda3/python.exe scripts/phase4_experiments/real_ozz/export_single_comtrade_marking.py
`

На выходе: `<OUTPUT_DIR>/<file>_bus<N>_marking.csv`, `.png` (если `PLOT=True`)
и `.json` с метаданными (модель, fs, max_probability, detected).

### 19.3 Шаг 3 — Авторазметка реальной базы и расширение датасета (приоритет № 3)

Файл: `scripts/phase4_experiments/real_ozz/inference_real_ozz.py` — теперь поддерживает
четыре подмножества:

| `SUBSET` | Описание | Куда сохраняется |
|---|---|---|
| `confirmed` | Файлы из отчёта с `Проверка == '+'` | `confirmed_ozz/` (PNG) |
| `false_detection` | Файлы из отчёта с `Проверка == '-'` | `false_detection/` (PNG) |
| `unknown` | Файлы из `data/real_OZZ/osc_comtrade/`, **не** упомянутые в отчёте | лог CSV + опционально PNG/CSV-сегменты |
| `all` | Объединение confirmed + false_detection | `confirmed_ozz/` + `false_detection/` |

Параметры в блоке `РУЧНОЙ РЕЖИМ`:

```python
CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_.../latest_checkpoint.pt'
SUBSET = 'unknown'             # для расширения базы no-OZZ берём 'unknown'
MAX_FILES = None               # None = все
THRESHOLD = 0.5
AUTO_CROP = True
MASK_NEUTRAL = False

# --- Ускорение и расширение базы ---
FAST_MODE = True               # шаг = размер окна (~10× быстрее, грубее по времени)
SAVE_UNKNOWN_PLOTS = False     # PNG для unknown с детектом — выкл по умолчанию
SAVE_PSEUDO_CSV = True         # сохранять CSV-сегменты вокруг детекта (для дообучения)
PSEUDO_MARGIN_PERIODS = 10     # запас периодов вокруг каждой зоны интереса
PSEUDO_MERGE_GAP_PERIODS = 5   # объединять детекты, если зазор < N периодов
PSEUDO_CSV_DIR = None          # None = <output>/pseudo_csv
```
Что сохраняется в `reports/phase4/real_ozz_inference/`:

- `unknown_pseudo_marking.csv` — лог по каждому unknown-файлу
  (`filename, bus, fs, n_samples, max_probability, detected_ozz`).
  Используется для последующей ручной перепроверки.
- `unknown_detections/<file>.png` — PNG только для unknown с детектом и только
  если `SAVE_UNKNOWN_PLOTS=True`.
- `pseudo_csv/<file>_bus<N>_seg<NN>.csv` — CSV-сегменты сырых сигналов
  (IA..UN) + вероятностей (P_class0..3, P_max, pred_class, pred_any_ozz, coverage)
  вокруг каждого детекта; границы определяются превышением `THRESHOLD`
  с запасом `PSEUDO_MARGIN_PERIODS` и склейкой по `PSEUDO_MERGE_GAP_PERIODS`.
- `pseudo_csv/pseudo_segments_index.csv` — реестр всех сегментов
  (`file, bus, segment, start_sample, end_sample, duration_s, max_probability, csv`).

Запуск:

`powershell
C:/ProgramData/anaconda3/python.exe scripts/phase4_experiments/real_ozz/inference_real_ozz.py
`

#### Замечания по шагу 3 (что есть и чего пока нет)

1. **`FAST_MODE` теперь поддержан** в `mark_real_oscillogram` и пробрасывается через
   `inference_real_ozz.py` и `export_single_comtrade_marking.py`. Для
   `collect_real_ozz_statistics.py` он пока не подключён — добавим, если потребуется
   ускорить и сводную статистику.
2. **CSV-сегменты сохраняются в "сыром" формате** (8 каналов IA..UN + вероятности).
   Это удобно для ручной перепроверки и подойдёт для бинарной задачи "any_ozz/no_ozz",
   но **формат не идентичен** тренировочным CSV (`ComtradeParser.create_csv`,
   `osc_tools/io/comtrade_parser.py`). Если для дообучения потребуется тот же набор
   признаков (split_buses, structure_columns, simple_csv) — нужно будет добавить
   отдельный конвертер `pseudo_segments → training_csv`. **Дайте знать формат
   обучающих CSV, который вы используете на сайте**, и я подключу его напрямую.
3. **Имена файлов с сайта (с суффиксом)**. Если вы будете подавать готовые CSV из вашего
   архива (где имена уже содержат суффикс bus), нужен короткий адаптер: считывание
   таких CSV в общую схему `IA..UN` + `time_s` + `fs`. Пришлите, пожалуйста,
   пример (1–2 файла) — добавлю загрузчик в `inference_real_ozz` параллельно с COMTRADE.
4. **`unknown`-режим использует пробные секции `['1', '2']`** для файлов вне отчёта
   (так как в отчёте нет записи о bus). Если у файла есть только Bus-3/4 — он будет
   пропущен. Решается переходом на парсинг каналов из самого `.cfg` — могу добавить,
   если в реальной базе встречаются такие случаи.

### 19.4 Краткая шпаргалка по последовательности

`
1) baseline-обучение  → run_phase4_5_baselines.py        (manual, EXP='all')
2) экспорт одной OSC → export_single_comtrade_marking.py (manual, CFG=...)
3) авторазметка базы → inference_real_ozz.py             (manual, SUBSET='unknown',
                                                         FAST_MODE=True,
                                                         SAVE_PSEUDO_CSV=True)
`

Чекпоинты для шагов 2–3 берутся из папок `experiments/phase4/sim_ozz_finetune_*`,
сформированных на шаге 1 (или из ранее обученной `PhysicalKANTransformer`).
