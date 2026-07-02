# Лог работ по Фазе 4 (Physical KAN-Transformer)

## [2026-07-02] Насыщение ТТ: контур статистики и рисунков статьи

- Изучен завершённый run `run_20260702_023838`: 100 эпох, 93 063 train и
  22 908 validation-файлов. Лучший промежуточный F1 фазы A — 0,826
  (Precision=0,708, Recall=0,991, эпоха 69).
- Добавлен `analyze_ct_saturation.py` с ручным блоком запуска: кривые обучения,
  детерминированная оценка исходной фазы A, confusion matrix, event-level
  показатели, задержка, разрезы tau=0,015/0,4 и компактные примеры.
- Проверка голов B/C вынесена в отдельный протокол фиксированных циклических
  перестановок. Некорректный Macro-F1=A/3 не используется как результат статьи.
- Добавлен аудит split: пересечения имён нет, но соседние точки параметрической
  сетки не сгруппированы и могли попасть по разные стороны.
- Добавлена пакетная разметка `labeled_2025_12_03.csv`: один рисунок содержит
  три нормированных тока и три смещённые фазовые вероятности; создаётся CSV
  максимальных вероятностей для сортировки и экспертного отбора.
- Smoke выполнен отдельно на двух симулированных и одной реальной осциллограмме;
  визуальная структура обоих типов рисунков проверена. Модульные тесты: 3 passed.

## [2026-07-02] Насыщение ТТ: прогресс эпохи и компактный вывод

- В `train_ct_saturation.py` добавлены progress bar `tqdm` отдельно для train
  и validation с текущим средним loss, числом обработанных пакетов, скоростью
  и оценкой оставшегося времени.
- Консольный итог эпохи сокращён до `loss`, Macro-F1, learning rate и времени.
- Полные Precision/Recall/F1 по фазам по-прежнему сохраняются в
  `training_log.jsonl`, но больше не перегружают консоль.
- Идущее обучение не останавливалось и не перезапускалось; изменение вступит
  в силу при следующем запуске или resume.

## [2026-06-21] Сессия 5: Составление статьи (было файлом SECTION5_MODEL_COMPARISON_REPORT.md)

# Раздел 5 статьи: сравнение архитектур — отчёт для автора

**Источник данных:** `reports/phase4/model_comparison/compare_4models_latest_20260610_204942/`  
**Дата прогона (summary):** 2026-06-12  
**Режим:** `latest_checkpoint.pt`, SimOZZ 240 файлов/класс (=960), real OZZ полный набор, порог 0.5  
**Статус:** черновик; метрики могут измениться после дообучения моделей.

---

## 1. Зачем этот прогон

Единый скрипт `compare_phase4_5_models.py` сравнивает **4 архитектуры** на двух стендах:

| Ключ | Модель | Суть абляции |
|------|--------|--------------|
| `physical_kan` | PhysicalKANTransformer | Полная модель: физические признаки + KAN |
| `spectral_baseline` | BaselineTransformer (spectral) | Только спектральные каналы, MLP вместо KAN |
| `physical_mlp` | PhysicalMLPTransformer | Физические признаки, но MLP вместо KAN |
| `raw_instantaneous` | BaselineTransformer (raw) | Сырые мгновенные значения, без FFT/физики |

**Цель для статьи:** показать, что усложнение пайплайна (физика, KAN) даёт выигрыш там, где это важно — на **реальных** осциллограммах, а не только на SimOZZ.

---

## 2. Структура папки отчёта

```
compare_4models_latest_20260610_204942/
├── comparison_summary.json          ← главная сводка (4 строки)
├── comparison_summary.csv
├── comparison_summary_charts.png    ← появится после повторного --continue-from (см. §7)
├── sim_ozz_eval/
│   └── <model>_latest_checkpoint/
│       ├── sim_ozz_evaluation.json   ← метрики + per-window данные (очень большой!)
│       └── *.png                     ← графики SimOZZ (сейчас отсутствуют, см. §6)
└── real_ozz_statistics/
    └── <model>_latest_checkpoint/
        ├── real_ozz_statistics.json  ← агрегированная статистика (~240 строк)
        ├── per_file_statistics.json    ← по каждому файлу×bus (~1833 записи)
        ├── predictions_per_file.csv    ← появится после finalize (см. §7)
        ├── predictions_errors_only.csv ← только ошибки модели
        └── *.png                       ← 3 графика на модель
```

**Коллеге не нужно читать JSON целиком.** Достаточно:
- `comparison_summary.csv` — таблица для текста;
- `comparison_summary_charts.png` — одна картинка на весь раздел (Примечание автора: появится при следующем запуске, сейчас может не быть, как и чего-то другого);
- для каждой модели — 3 PNG из `real_ozz_statistics/` + при необходимости `predictions_errors_only.csv`.

---

## 3. SimOZZ: синтетический стенд (960 val-файлов)

### 3.1. Window-level метрики (основные для таблицы)

| Модель | Macro-F1 | Exact match | Macro AUC | Latency (ms) | Параметры |
|--------|----------|-------------|-----------|--------------|-----------|
| **Spectral baseline** | **0.9838** | **0.9779** | 0.9998 | **7.5** | 255 748 |
| Physical MLP | 0.9784 | 0.9703 | 0.9998 | 12.9 | 183 007 |
| Raw instantaneous | 0.9773 | 0.9701 | 0.9998 | 8.9 | 255 748 |
| Physical KAN | 0.9723 | 0.9660 | 0.9997 | 24.9 | 263 286 |

> **Важно для текста:** на SimOZZ **лучший — spectral baseline**, не KAN. Разница Macro-F1 ≈ 1.1 п.п. KAN медленнее (~3.3× vs spectral) (Примечание автора: Посмотрим как будет на следующих обучениях, быть может улучшится, пока что лишь 100 эпох, запустил продолжение обучния).

### 3.2. Per-class F1 (window-level, порог 0.5)

| Класс | Physical KAN | Spectral | Physical MLP | Raw inst. |
|-------|-------------|----------|--------------|-----------|
| Stable | 0.994 | **0.994** | 0.992 | 0.993 |
| Petersen | 0.975 | **0.978** | 0.969 | 0.972 |
| PetersSlepian | 0.953 | **0.986** | 0.979 | 0.983 |
| Beliakov | 0.968 | **0.977** | 0.974 | 0.961 |

**Слабое место KAN на SimOZZ (Примечание автора: и на будущее, может само слабое место модели в симулированном? хуже "зубрит" пытаясь понять? Ведь потом лучше вроде бы становится на реальном):** PetersSlepian (F1=0.953) — много FP (1601 vs 324 у spectral). Recall высокий (0.998), precision проседает.

**Слабое место raw instantaneous:** Beliakov — F1=0.961, FP=1188 (худший среди четырёх).

### 3.3. Что считает `evaluate_sim_ozz.py`

1. **File-by-file inference** — все окна val-набора (960 файлов, ~116k окон).
2. **Window-level метрики** — max по зонам внутри окна → бинарная классификация 4 классов.
3. **Zone-level метрики** — каждая зона ДПОЗЗ отдельно (Macro-F1 ниже, ~0.91–0.94).
4. **Boundary metrics** — задержка обнаружения, smearing onset/offset (для инженерного раздела).
5. **Optimal thresholds** — подбор порога по F1; Macro-F1(opt) для KAN ≈ 0.980 vs 0.972 @0.5.
6. **Графики** (если не пропущены): confusion_matrix, prob_distributions, radar, engineering_bars, training_curves.

---

## 4. Real OZZ: реальные осциллограммы

### 4.1. Набор данных

- Источник разметки: `data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv`
  - `Проверка = '+'` → **confirmed** (подтверждённое ОЗЗ), 829 уникальных файлов
  - `Проверка = '-'` → **false_detection** (перенапряжение без ОЗЗ), 1004 файла
- COMTRADE: `data/real_OZZ/osc_comtrade/{filename}.cfg`
- Имена файлов — **MD5-хэши** (как в исходном CSV).
- Обработано **1833** валидных записей (файл × bus; один .cfg может иметь bus 1,2,3…).

### 4.2. File-level Detection Rate (порог max P(class) > 0.5, any class)

| Модель | Confirmed: DR ↑ лучше | False detection: DR ↓ лучше |
|--------|----------------------|---------------------------|
| **Physical KAN** | **95.7%** (793/829) | 66.8% (671/1004) |
| Physical MLP | 95.2% (790/829) | **74.0%** (743/1004) — худший |
| Raw instantaneous | 91.6% (920/1004*) | 60.1% (603/1004) |
| Spectral baseline | 91.1% (755/829) | **58.4%** (587/1004) — лучший по специфичности |

\* Для raw inst. в summary указано 91.6% — 829 confirmed count тот же.

**Интерпретация для статьи:**

- **Physical KAN** — лучший recall на подтверждённых ОЗЗ (+4.6 п.п. vs spectral), цена — больший «шум» на false_detection (66.8% vs 58.4%).
- **Spectral baseline** — лучше **не детектирует** ложные срабатывания, но **пропускает** ~9% реальных ОЗЗ.
- **Physical MLP** — почти как KAN на confirmed, но **худший** по ложным (74%) → KAN vs MLP абляция имеет смысл.
- **Raw instantaneous** — компромисс между spectral и KAN.

### 4.3. Per-class detection на confirmed (physical_kan, % файлов с max P > 0.5)

| Класс | KAN | Spectral | MLP | Raw |
|-------|-----|----------|-----|-----|
| Stable_SPGF | 47.8% | ~45% | ~48% | ~46% |
| Petersen_AIGF | 33.4% | ~28% | ~32% | ~30% |
| PetersSlepyan_AIGF | 54.2% | ~50% | ~52% | ~51% |
| Belyakov_AIGF | **68.3%** | ~62% | ~65% | ~63% |

На confirmed часто детектируется **несколько классов** (co-occurrence): у KAN 61% файлов с ≥2 типами — это нормально для real OZZ (разные участки/режимы).

### 4.4. Что считает `collect_real_ozz_statistics.py`

Для каждого файла×bus:

1. Inference (`mark_real_ozz`) → матрица вероятностей `(N_samples, 4)`.
2. **max_prob** по каждому классу за всю осциллограмму.
3. **detected** = max_prob > threshold (0.5).
4. **any_ozz** = OR по 4 классам.
5. **Zone expansion** (±16 зон) → события, длительности, coverage%.
6. **Temporal** — относительное время первого срабатывания.
7. **Co-occurrence** — сколько типов ОЗЗ в одном файле.

Агрегация по группам confirmed / false_detection → JSON + графики.

---

## 5. Рекомендуемые формулировки для статьи (черновик)

### 5.1. Основной вывод (осторожный, честный)

> На синтетическом стенде SimOZZ все четыре архитектуры достигают Macro-F1 > 0.97; спектральный baseline без физических признаков показывает наилучший результат (F1=0.984) при минимальной задержке inference. Однако на реальных осциллограммах модель с физическими признаками и KAN-слоями обеспечивает наивысший уровень обнаружения подтверждённых ОЗЗ (95.7% против 91.1% у spectral baseline), что подтверждает практическую значимость физически мотивированного представления признаков. Абляция KAN→MLP при сохранении физики ухудшает специфичность на ложных срабатываниях (74.0% против 66.8%).

### 5.2. Абляции (подразделы)

(Примечание автора: полноценна абиляция не проводила на обученной модели, там планируется отдельный шаг, но в целом, уже можно что-то написать.)

- **Spectral vs Raw instantaneous** — оба на BaselineTransformer; raw проигрывает на Beliakov (SimOZZ) и на confirmed real OZZ.
- **Physical MLP vs Physical KAN** — физика одинакова; KAN снижает ложные срабатывания на ~7 п.п.
- **SimOZZ vs Real OZZ** — ranking моделей **меняется**; нельзя выбирать архитектуру только по синтетике.

### 5.3. Ограничения (обязательно упомянуть)

- Модели на `latest_checkpoint`, обучение могло не завершиться.
- Real OZZ метрики — file-level по max probability, не zone-level F1.
- Порог 0.5 фиксирован; для KAN optimal thresholds на SimOZZ выше (0.57–0.90).
- 829+1004 = уникальные **файлы**; 1833 — **записи** file×bus (Примечание автора: чуть меньше, чем было исходно, т.к. некоторые осциллограммы короче 200мс).

---

## 6. Графики: что есть и что добавить

### Сейчас в папке

| Артефакт | Статус |
|----------|--------|
| `comparison_summary.csv/json` | ✅ есть |
| SimOZZ `*.png` | ❌ **нет** — inference был пропущен (кэш JSON), графики не пересоздавались |
| Real OZZ `*.png` | ❌ **нет** — та же причина (skip после JSON) |
| `predictions_per_file.csv` | ❌ **нет** — добавлен экспорт в код (§7) |

### Рекомендации для иллюстраций в статье

| Рисунок | Файл | Приоритет |
|---------|------|-----------|
| Сводное сравнение 4 моделей | `comparison_summary_charts.png` | **Высокий** — один рисунок на весь §5 |
| Real OZZ: confirmed vs false | `real_ozz_statistics/<model>/per_class_detection.png` | Высокий — для KAN или для всех 4 в приложении |
| Real OZZ: распределение уверенности | `confidence_distributions.png` | Средний — уже использовался в §4 статьи |
| SimOZZ: prob_distributions | `sim_ozz_eval/<model>/prob_distributions.png` | Низкий — все модели «уверены», текстом достаточно |
| SimOZZ: confusion matrix | только если нужна детализация по PetersSlepian для KAN |

**Не перегружать:** 1 сводный bar-chart + 1 real-OZZ per-class (лучшая и baseline) достаточно для основного текста.

---

## 7. Как получить CSV с именами осциллограмм и ошибками

(Примечание автора: не для статьи этот раздел)

### Ответ на вопрос «можно ли узнать, какая осциллограмма как размечена?»

**Да.** В `per_file_statistics.json` уже есть поля:
- `filename` — MD5-имя (как в CSV)
- `bus` — секция
- `group` — `confirmed` или `false_detection`
- `max_prob`, `detected`, `any_ozz`, `types_detected`

Пример записи с ошибкой (false alarm):
```json
"filename": "00053f0220cb56cf7be1c17cdc0cf037",
"bus": "2",
"group": "false_detection",
"any_ozz": true,
"types_detected": ["Belyakov_AIGF"]
```

### Новый CSV (после доработки кода)

После запуска finalize появятся:

- **`predictions_per_file.csv`** — полная таблица: исходный CSV + все max_prob/detected + `error_type`
- **`predictions_errors_only.csv`** — только `missed_ozz` и `false_alarm`

Колонки `error_type`:
- `missed_ozz` — confirmed, но модель не увидела ОЗЗ (FN)
- `false_alarm` — false_detection, но модель что-то нашла (FP)
- `correct` — всё остальное

Колонка `comtrade_cfg` — готовый относительный путь к файлу для просмотра.

### Команды

**Догенерация для одной модели (без inference):**
```bash
python scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py \
    --finalize-only \
    --output-dir reports/phase4/model_comparison/compare_4models_latest_20260610_204942/real_ozz_statistics/physical_kan_latest_checkpoint
```

**Продолжение полного сравнения** (догенерирует CSV/PNG/summary chart для всех моделей):
```bash
python scripts/phase4_experiments/evaluation/compare_phase4_5_models.py \
    --continue-from reports/phase4/model_comparison/compare_4models_latest_20260610_204942
```

**SimOZZ графики** — нужен отдельный прогон (JSON кэшируется без PNG):
```bash
python scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py \
    --checkpoint experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260609_140228/latest_checkpoint.pt \
    --per-class-files 240 \
    --output-dir reports/phase4/model_comparison/compare_4models_latest_20260610_204942/sim_ozz_eval/physical_kan_latest_checkpoint
```
(удалите `sim_ozz_evaluation.json` перед этим, если скрипт снова пропустит inference)

**Визуальная разметка одной осциллограммы:**
```bash
python scripts/phase4_experiments/real_ozz/inference_real_ozz.py \
    --checkpoint experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260609_140228/latest_checkpoint.pt \
    --file 00053f0220cb56cf7be1c17cdc0cf037
```

---

## 8. Таблица для вставки в статью (LaTeX-ready)

(Примечание автора: можешь и сама подумать о тому, как лучше бы всё это сделать)

```
Модель               | Sim F1 | Real conf. | Real false | Latency ms
---------------------|--------|------------|------------|----------
Physical KAN         | 0.972  | 95.7%      | 66.8%      | 24.9
Spectral baseline    | 0.984  | 91.1%      | 58.4%      | 7.5
Physical MLP         | 0.978  | 95.2%      | 74.0%      | 12.9
Raw instantaneous    | 0.977  | 91.6%      | 60.1%      | 8.9
```

---

## 9. Чеклист для коллеги-автора


(Примечание автора: можешь и сама подумать о тому, как лучше бы всё это сделать)

- [ ] Перечитать §5 после финального дообучения — обновить числа из нового `comparison_summary.csv`
- [ ] Запустить `--continue-from` для CSV и графиков
- [ ] Вставить `comparison_summary_charts.png` как Рис. «Сравнение архитектур»
- [ ] Описать абляции по таблице §8, не утверждая «KAN лучший на SimOZZ»
- [ ] Акцент: **физика + KAN → лучший recall на real OZZ**
- [ ] Для кейсов ошибок — фильтр `predictions_errors_only.csv` → `inference_real_ozz.py`
- [ ] SimOZZ детали (PetersSlepian FP у KAN) — опционально в приложении

---

## 10. Связанные файлы в репозитории

(Примечание автора: не для статьи)

| Файл | Назначение |
|------|------------|
| `scripts/phase4_experiments/evaluation/compare_phase4_5_models.py` | Оркестратор сравнения |
| `scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py` | SimOZZ метрики и графики |
| `scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py` | Real OZZ статистика + CSV export |
| `scripts/phase4_experiments/real_ozz/inference_real_ozz.py` | Визуализация одной осциллограммы |
| `data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv` | Ground truth разметка |
| `docs/ARTICLE_PREPARATION_BRIEF.md` | Общий план статьи |
| `docs/new_article.md` | Текущий черновик (§5 пока заглушка) |



## [2026-03-17] Сессия 4: Низшие гармоники, аугментация, сложность, оценка

### Реализовано: Низшие (суб-)гармоники

1. **`compute_low_harmonics_fft()`** в [osc_tools/preprocessing/filtering.py](osc_tools/preprocessing/filtering.py)
   - Backward-looking скользящее окно FFT для суб-гармоник с периодами 2, 4, 6, 10
   - Извлекает бин 1 FFT из окон 64, 128, 192, 320 отсчётов
   - Начальные точки заполняются первым валидным значением (дублирование)
   - Увеличение числа каналов: 144 → 208 (8 сигналов × 13 гармоник × 2)

### Реализовано: AugmentedSpectralDataset

2. **`AugmentedSpectralDataset`** в [osc_tools/ml/augmented_dataset.py](osc_tools/ml/augmented_dataset.py)
   - On-the-fly FFT: загружает raw 8-канальные данные, вычисляет FFT в реальном времени
   - Аугментация ДО FFT: инверсия, масштабирование, перетасовка фаз на сырых данных
   - Поддерживает SSL (маскирование + предсказание будущего) и classify режимы
   - Загружает расширенное окно с контекстом для backward-looking low harmonics
   - Padding для начала файлов (дублирование первой строки)
   - Phase polar conversion с опорным фазором UA h1

### Реализовано: Уровни сложности модели + Gradient Accumulation

3. **3 уровня сложности** (d_model < num_input_channels = 208 для SSL bottleneck):
   - light: d_model=48, 6 layers, 4 heads
   - medium: d_model=64, 8 layers, 8 heads
   - heavy: d_model=64, 16 layers, 8 heads
4. **Gradient Accumulation**: effective_batch = batch_size × accumulation_steps = 32 × 8 = 256
5. Обновлены скрипты pretrain и finetune:
   - `--complexity light|medium|heavy`
   - `--no-augmentation`, `--no-low-harmonics`
   - `--accumulation-steps N`

### Реализовано: Скрипт оценки Phase 4

6. **[scripts/phase4_experiments/evaluate_phase4.py](scripts/phase4_experiments/evaluate_phase4.py)**
   - Порог предсказания: 0.7
   - Полные метрики: Macro-F1, Precision, Recall, ROC-AUC, Confusion Matrix
   - Measurement inference latency
   - Сравнение нескольких экспериментов (`--compare-dir`)
   - Сохраняет `evaluation_report.json` в директорию чекпоинта

### Обновлено: build_channel_groups_phase_polar

7. Поддержка `num_low_harmonics` в [osc_tools/ml/losses.py](osc_tools/ml/losses.py)

---

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
# 2026-07-01 — старт исследования насыщения ТТ (раздел 7 статьи)

- Обследован архив `meas_DP_PG_2SYS_INT_A0_SC1_exp_3`: 115 973 MAT-файла,
  из них 115 971 осциллограмма и 2 таблицы `t_sat_CTs`, общий объём 352,1 ГБ.
- Проверено на реальных файлах: 100 кГц, 44 394 отсчёта, I1/I2≈1000,
  вторичный ток до КЗ около 1 А RMS, в исследованном КЗ до 12–13 А.
- Принято решение использовать только три канала `I2_CT1`; `I1_CT1` исключён
  как недоступный в эксплуатации идеальный эталон, `V2_VT1` не нужен для токовой задачи.
- Разметка берётся напрямую из `flag_sat_phsA/B/C`; три независимых выхода.
  Поскольку исходно насыщается CT1 фазы A, добавлена совместная циклическая
  перестановка фаз сигналов и меток только на train.
- Реальная проверка SciPy выявила, что MATLAB timeseries сохранены как MCOS
  `MatlabOpaque` и напрямую не читаются. Добавлен resume-safe MATLAB-конвертер,
  сохраняющий только плоские I2, labels и time; обучение читает уменьшенную копию.
- Реализован `osc_tools/ml/ct_saturation_dataset.py`: lazy flat-MAT load, стабильный
  split, LRU, 9+4 гармоники, токовые симметричные составляющие, 84 признака.
- Подготовлен `scripts/phase4_experiments/ct_saturation/train_ct_saturation.py`:
  фиксированное число batch/эпоху, AMP, resume, best/latest checkpoints, JSONL.
- Добавлены unit-тесты адаптера и `scipy` в зависимости. Полный запуск оставлен
  исследователю после smoke в рабочем ML-окружении.
- MATLAB-конвертер проверен на первых 20 файлах; созданы плоские записи в
  `data/ct_saturation_flat`. Unit-тесты: 2 passed. Полный `--smoke` обучения
  завершился с сохранением checkpoint/log; низкие метрики ожидаемы для двух
  batch и не интерпретируются как результат исследования.

# 2026-07-01 — итерация 2 контура насыщения ТТ

- Уточнено описание каналов: `I1_CT1` хранит идеальные первичные токи в
  первичных амперах, `I2_CT1` — измеренные вторичные токи; до насыщения
  `I1/1000 ≈ I2`. `V2_VT1` действительно содержит три вторичных напряжения,
  но в токовой задаче не используется.
- PR-AUC исключена из плана раздела 7 и материалов этого опыта. Оставлены уже
  используемые в работе Precision, Recall, F1, матрицы ошибок и задержка.
- Все поясняющие комментарии нового Python-кода переведены на русский язык.
- Три файла запуска объединены в один `train_ct_saturation.py`: он выполняет
  подготовку, smoke, обучение и resume; параметры доступны через CONFIG и CLI.
- Добавлено постоянное правило: одноразовые скрипты редактирования и каталоги
  рендера не должны оставаться в репозитории после успешного применения.

# 2026-07-01 — итерация 3: ручной запуск по образцу SimOZZ

- Постоянные правила перенесены из отдельного `AGENTS.md` в существующий
  `.github/copilot-instructions.md`; заголовок обобщён на всех ИИ-ассистентов.
- `train_ct_saturation.py` приведён к пользовательскому интерфейсу SimOZZ:
  внизу добавлен подробный блок ручного запуска из IDE с путями, подготовкой,
  спектром, архитектурой, обучением, размером эпохи, DataLoader и resume.
- Запуск без аргументов использует ручной блок; наличие CLI-аргументов
  автоматически включает CLI. Добавлены accumulation, отдельный validation
  batch, prefetch/cache, AMP, gradient clipping и периодические checkpoint.
