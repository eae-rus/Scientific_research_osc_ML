# Журнал работ Phase 5

## 16.07.2026 — GPU smoke/resume, A/B и robust loss

- Подключён пользовательский PyTorch user-site: Python 3.13.5, torch 2.7.1+cu118,
  CUDA, NVIDIA RTX 3060 Ti. Torch model suite реально выполнен: **82 passed**.
- Resume Version B продолжил epoch 2 из `latest_checkpoint.pt`: val `0.11574`,
  Open_EE `0.12273`, French `0.11498`. Сформированы JSON/PNG curves и проверен
  checkpoint passport.
- Средний 512-window A/B ComplexMSE probe: B `0.01664`, A `0.01704`, однако
  train loss обеих версий сильно прыгал из-за редких выбросов.
- Добавлен `RobustComplexLoss` (SmoothL1 в Re/Im, beta=0.1). Huber probe дал
  устойчивое снижение train loss. B combined/Open_EE/French =
  `0.06077/0.06374/0.05657`; A = `0.06223/0.07166/0.05224`.
- Создан `reports/phase5/pretrain_smoke_comparison.md`. Huber зафиксирован как
  Phase 5 default; B — предварительный основной контракт до PDR probe.
- Профиль целевой B-модели (`d_model=64`, 4 слоя, batch 16): 67.4 train
  samples/s, peak allocated CUDA 25.99 MiB. Full F5 блок документирован как
  20k окон × 50 эпох, ориентировочно 5–6 часов; добавлены progress и ETA.

## 16.07.2026 — Исправлен fully-missing sample в Version B

- Воспроизведён сбой validation index 24: Open_EE record 44635, SPP=12,
  внутри выбранного окна полностью отсутствуют IB и UA/UB/UC. Version B состоит
  из симметричных составляющих, поэтому без полной тройки фаз не имела ни одной
  физически допустимой SSL target-группы.
- `MaskedSpectralDataset` теперь детерминированно пробует другую индексную
  выборку (до 16 попыток), сохраняет requested/sampled index и номер retry в
  metadata. Значения не подменяются нулями, а неполная запись не используется
  как фиктивный zero-loss sample. На исходном smoke validation index 24 был
  корректно заменён index 7 с одной попытки.

## 15.07.2026 — Подготовлен research-strict SSL smoke

- Создан и реально сгенерирован `data/phase5/research_strict_splits.json` с
  SHA-256 `c8f0089f8cb09427df59f9fea10b26f5a5026b4b375ab65ced472f15aa4d88ff`.
  Open_EE разделён целыми исходными CSV: train=38473, validation=1823,
  never-seen holdout=4477. French разделён по записям: 9643/1205/1205.
- Добавлены group-aware masked modeling и immutable checkpoint passport с
  feature names/order, temporal mode, cyclic/provenance flags и schema hash.
- `run_phase5_pretrain.py` поддерживает F5, `--smoke`, `--resume`,
  `--reset-optimizer`, latest/best checkpoints и проверку контракта до загрузки
  state_dict. Реальный запуск в bundled runtime дошёл до ожидаемой понятной
  ошибки об отсутствии PyTorch; синтаксис и вся numpy/data цепочка проверены.
- В VS Code Anaconda найден pytest, но нет torch/polars. С workspace basetemp и
  без общего polars-conftest весь набор `test_phase5_*.py` прошёл: **35 passed**.
  Попутно исправлен устаревший тест extraction: он теперь проверяет требуемый
  idempotent reuse совместимого NPY и отказ для несовместимого файла. Временный
  pytest-каталог после проверки удалён.

## 15.07.2026 — Feature v2 подключён к реальным lazy sources

- `SpectralMultiSourceDataset` связал Open_EE/French readers с версиями A/B и
  режимами `snapshot_2`, `snapshot_5`, `sequence_1_8`. Low-period FFT теперь
  causal: позиции означают конец окна, а 10 периодов предыстории загружаются
  отдельно от 10-периодного модельного интервала.
- Provenance `measured/derived/missing` проходит от shard/French adapter до
  каждого спектрального признака; line-voltage ветвь вычисляет доступные U1/U2
  и маскирует физически невосстановимый U0.
- Реальный smoke выполнен для 12 сочетаний source × A/B × temporal mode.
  Формы: A=220, B=156; snapshot=2/5; sequence на проверенных SPP даёт 81 токен,
  а при SPP=20 и округлённом шаге 3 — 67 токенов. В Open_EE случайно попала одна
  из записей без напряжений, и masks/provenance корректно сохранили отсутствие.

## 15.07.2026 — Циклическая Phase 5-ветка модели

- `PhysicalStem` и `ComplexMultiheadAttention` получили явный
  `cyclic_angle_encoding`: перед обучаемыми angle-проекциями используются
  `sin/cos`, а `torch.polar` и directional gate продолжают получать сырой угол.
- Default оставлен `False`, поэтому формы legacy Phase 4 state_dict не меняются.
  `PhysicalKANTransformer(..., cyclic_angle_encoding=True)` включает новый путь.
- Добавлены torch-тесты инвариантности к `angle + 2*pi` и проверки legacy/new
  shapes. В текущем bundled runtime нет torch/pytest: выполнены `py_compile` и
  numpy real-data smoke; torch-набор подготовлен для основного окружения.

## 15.07.2026 — Начат feature contract v2

- Добавлен `osc_tools/ml/spectral_features.py` с изолированным от legacy Phase 4
  `SpectralFeatureBuilder`, `SpectralFeatureConfig` и `FeatureSchema`.
  Version A возвращает 220 внешних `magnitude+angle` признаков, Version B — 156
  самостоятельных symmetric-polar признаков.
- Builder сохраняет NaN/mask для missing signals и harmonics выше Найквиста:
  при SPP=12 доступны h1–h6, недоступные h7–h9 маскируются.
- Smoke на аналитической трёхфазной синусоиде выполнен. В первом запуске
  обнаружена лишняя symmetric ветвь A для h2–h9 (364 вместо 220 признаков),
  исправлена до интеграции: symmetric в A остаётся только для h1.

## 15.07.2026 — Завершена подготовка Open_EE shards

- Ручной full build успешно завершён: 44773 осциллограммы в 448 uncompressed
  shards по 100 записей (последний неполный). Manifest содержит SPP/metadata;
  44755 записей имеют phase voltage basis, 18 — missing basis.
- `datasets_registry.json` подключён к фактически подготовленному
  `data/phase5/open_ee_shards`.

## 15.07.2026 — Lazy multi-dataset и подготовка полного Open_EE build

- Добавлен `osc_tools/ml/lazy_multi_dataset.py`: deterministic epoch/index
  sampling источников, выбор целой осциллограммы и 10-period raw window.
  Spectral builder намеренно остаётся отдельным следующим слоем.
- Реальный smoke на Open_EE/French readers вернул `(8, 200)` окно с SPP=20 и
  metadata source/record/window. French использует утверждённый per-unit profile.
- `prepare_open_ee_shards.py` получил F5 mode для полного uncompressed build в
  `data/phase5/open_ee_shards`, shard=100, `MAX_RECORDS=None` и resume.
  Полный build ожидает ручного длительного запуска.

## 15.07.2026 — Зафиксирована French/RTE нормировка

- Исследователь подтвердил engineering contract French/RTE:
  `current_nominal_a=300 A`, `current_reserve=20`, `voltage_nominal_v=90000 V`,
  `voltage_reserve=3`. Registry помечает French как нормированный источник.
- `FrenchRTESource` применяет `I_pu=I_phys/(300*20)` и
  `U_pu=U_phys/(90000*3)`; медианный RMS i1 после нормировки равен около 0.0519.

## 15.07.2026 — Dataset registry и базовые lazy sources

- Созданы `osc_tools/ml/dataset_registry.py`, расширенный
  `osc_tools/ml/phase5_sources.py` и `data/phase5/datasets_registry.json`.
  Registry различает archive и prepared French source, а также не разрешает
  случайно использовать French в SSL как нормированный без `current_nominal_a`.
- `OpenEEShardedSource` реализует ограниченный LRU-кэш открытых shards;
  `FrenchRTESource` читает `.npy` через mmap и заполняет `IN/UN` как NaN.
- Реально проверено: Open_EE prototype source вернул `(8, 7001)`, French source
  — `(8, 21000)` float32 в physical units.
- Удалены временные 10-record prototype shards и smoke JSON. Сохранены 1000-record
  benchmark shards как компактный реальный fixture следующего lazy-dataset этапа.

## 15.07.2026 — Подтверждён формат Open_EE shard и resume

- На 1000 одинаковых Open_EE записей (10 shards по 100) проведён benchmark:
  uncompressed — 10320.4 records/s и 133.22 MiB; compressed — 2587.5 records/s
  и 52.47 MiB. Для основной подготовки выбран uncompressed `.npz` layout.
- Sandbox оборвал создание compressed prototype после 700 записей. В ответ
  `prepare_open_ee_shards.py` дополнен resume: готовые shards читаются и
  валидно переиспользуются, а CSV перечитывается только для продолжения
  детерминированного порядка. Реальный restart достроил оставшиеся 300 записей.
- Результат сохранён в `reports/phase5/storage_benchmark_1000.json`.

## 12.07.2026 — Open_EE shard prototype и I/O benchmark

- По полному `open_ee_scan.json` проверено покрытие: 44755 из 44773 записей
  имеют полный фазный `UA/UB/UC`; только 18 не имеют полного фазного напряжения,
  и записей с одними полноценными линейными напряжениями нет. Базовая schema не
  требует line-voltage fallback, но adapter сохраняет `voltage_basis`.
- Добавлены `osc_tools/ml/phase5_sources.py` и
  `scripts/phase5_experiments/prepare_open_ee_shards.py`: flat float32 signals
  + offsets вместо padding, 8 logical channels, NaN для missing, provenance,
  `source_columns`, SPP и voltage basis.
- Реальный 10-record prototype успешно собран в два shard-а. Ошибка попытки
  сериализовать provenance в JSON была обнаружена и исправлена до дальнейшей
  конвертации.
- Добавлен `benchmark_phase5_storage.py`; 3-повторный benchmark того же
  prototype: uncompressed 5045.2 records/s и 1.31 MiB, compressed 1848.5
  records/s и 0.54 MiB. Результат сохранён в
  `reports/phase5/storage_benchmark.json`.

## 12.07.2026 — Завершён French RMS scan и добавлен прогресс длительных задач

- Полный mmap-проход French `DATA_S.npy` выполнен агентом: 12053 записи,
  164 полных периода на запись. Результаты сохранены в
  `data/digital-fault-recording-database/french_scan.json` и
  `reports/phase5/french_normalization_notes.md`.
- Добавлен dependency-free индикатор `scripts/phase5_experiments/progress.py`.
  French показывает прогресс по записям, Open_EE и real_OZZ exclusion — по
  байтам при полном проходе и по числу файлов в smoke-режиме, extraction — по
  распакованным байтам. Отображаются процент и ETA.
- Реально проверены French full scan с прогрессом и Open_EE smoke с корректным
  file-level прогрессом. Номинал French current по-прежнему не выбран
  автоматически.

## 12.07.2026 — Исправление ручных запусков Stage 1

- `prepare_french_rte_npy.py` сделан идемпотентным: существующий NPY сверяется
  по shape/dtype с архивом и при совпадении возвращается как
  `reused_existing=true`; ошибкой остаётся только несовместимый файл.
- Все новые Stage 1 сценарии используют `PROJECT_ROOT`, вычисленный от
  `__file__`; direct F5 больше не требует вручную задавать cwd или `PYTHONPATH`.
- Реально выполнен direct-script smoke из `scripts/phase5_experiments`:
  повторный French extraction, French mmap scan на одной записи и Open_EE smoke
  на всех 18 CSV. Все три команды завершились с кодом 0.

## 12.07.2026 — Ручной запуск Stage 1 через F5

- Во все длительные сценарии Stage 1 добавлены явные блоки ручного запуска
  через F5: `scan_french_dataset.py`, `scan_open_ee_dataset.py` и
  `build_real_ozz_exclusion.py`. CLI остаётся доступным при передаче аргументов.
- Для French в `run_manual()` уже указан извлечённый
  `data/phase5/french_rte/DATA_S.npy`; `MAX_RECORDS = None` запускает полный
  RMS-проход, а значение `100` позволяет сначала проверить окружение.
- Реально проверено: CLI French scan с `--max-records 1` открыл извлечённый NPY
  через mmap и сформировал корректный отчёт без материализации всего массива.

## 12.07.2026 — Подготовка French random-access и real_OZZ exclusion

- Исследователь разрешил разовое извлечение French `DATA_S.npy` (~12,15 ГБ).
  Реализован `scripts/phase5_experiments/prepare_french_rte_npy.py`: он сохраняет
  архив неизменным, проверяет свободное место, извлекает member во временный файл,
  валидирует shape/dtype через mmap и записывает `preparation_manifest.json`.
- Полный extraction не завершён в sandbox: каждый terminal-вызов принудительно
  ограничен 60 секундами, а распаковка занимает больше. Сценарий готов для
  ручного запуска; исходный архив не был изменён.
- Реализован `scripts/phase5_experiments/build_real_ozz_exclusion.py` и unit-test:
  файл `real_ozz_exclusion.json` фиксирует exact/soft/ambiguous совпадения, а
  `open_ee_real_no_ozz_index.json` исключает только подтверждённые совпадения.
- `osc_tools/ml/__init__.py` переведён на ленивые legacy exports. Это устраняет
  неявное требование PyTorch при CPU-only сканировании/подготовке Phase 5.
  Реальный smoke `scan_open_ee_dataset.py --smoke` снова прочитал все 18 CSV.
- В актуальный план внесены статусы French extraction, real_OZZ exclusion и
  первой стадии French storage benchmark.

## 12.07.2026 — Синхронизация плана и handoff-документов с фактическим прогрессом

- `PHASE_5_PLAN.md`, `PHASE_5_START_PROMPT.md` и исследовательская выжимка
  синхронизированы с уже реализованными контрактами/сканерами и результатами
  от 08.07.2026; исправлен путь summary после переноса в `docs/article/`.
- В план внесён подтверждённый формат French `DATA_S.npz`: deflate-сжатый
  `float64`, около 2,046 ГБ в архиве и 12,149 ГБ после extraction; прямой mmap
  сжатого member невозможен. Разделены source archive и prepared training data.
- Для научной оценки разделены протоколы `research_strict` (never-seen file-level
  holdout до SSL) и `full_archive` (финальный рабочий backbone без претензии на
  независимый transfer test).
- Код не изменялся; выполнена структурная проверка Markdown и `git diff --check`.

## 08.07.2026 — Потоковые сканеры Open_EE и French/RTE

### Выполнено

- Добавлен `scripts/phase5_experiments/scan_open_ee_dataset.py`: последовательное
  чтение CSV стандартным `csv` без загрузки файла целиком, точные агрегаты
  count/min/max/mean/std, детерминированный reservoir для приближённых квантилей,
  длины записей по `file_name`, metadata SPP/stride и доступных гармоник.
- Smoke-прогон прочитал первые 1000 строк каждого из 18 реальных Open_EE CSV.
  Подтверждено: при 600/50 Гц доступны h1–h6, при 800/50 Гц — h1–h8; повторная
  нормировка явно запрещена в машинном отчёте.
- Добавлен `scripts/phase5_experiments/scan_french_dataset.py`. Инспекция ZIP/NPY
  выполняется без распаковки массива; RMS-режим принимает отдельно извлечённый
  `.npy` и читает его через mmap пакетами записей.
- Реальный `DATA_S.npz` подтверждён как deflate-сжатый массив float64 формы
  `(12053, 6, 21000)`: 2,046 ГБ в архиве и 12,149 ГБ после распаковки.
  `np.load(..., mmap_mode='r')` для него не является настоящим mmap, поэтому
  обычное обращение к `DATA_S` пытается распаковать весь массив.
- Добавлены `test_phase5_open_ee_scan.py` и `test_phase5_french_scan.py`.
  Совместный результат новых тестов: `16 passed in 1.48s`.

### Открытая практическая развилка

- Для полной RMS-статистики French требуется один раз извлечь `DATA_S.npy`
  размером около 12,15 ГБ либо выбрать иной подготовленный формат. До проверки
  свободного места и согласования этого дискового расхода extraction не выполнен.

## 08.07.2026 — Старт Phase 5 и базовый временной контракт

### Выполнено

- Прочитаны `PHASE_5_START_PROMPT.md`, `PHASE_5_PLAN.md`, краткий контекст
  исследований и постоянные инструкции проекта.
- Выполнен первый Phase 4 safety pass: 90 тестов модели, аугментаций и задачи
  насыщения ТТ прошли; один тест первоначально не запустился из-за запрета
  sandbox на системный `%TEMP%`, после переноса `basetemp` внутрь workspace он
  прошёл. Два legacy smoke-модуля не были собраны из-за отсутствия `sklearn` в
  bundled runtime; это ограничение среды, а не подтверждённая регрессия кода.
- Добавлен независимый от PyTorch модуль
  `osc_tools/ml/phase5_contracts.py`: единый порядок восьми каналов,
  `measured/derived/missing` provenance, детерминированное округление SPP,
  окна и шага, ограничение гармоник по Найквисту, snapshot-индексы и
  сериализуемые metadata временной сетки.
- Добавлены тесты `tests/unit/test_phase5_contracts.py` для групп 50/60 Гц,
  SPP 12/18/32/128, шага 1/8 периода, доступных гармоник и режимов
  `snapshot_2`/`snapshot_5`.

### Реально запущенные проверки

```text
pytest test_transformer_model.py test_augmentation.py test_ct_saturation_dataset.py
Результат: 90 passed; 1 setup error из-за недоступного системного TEMP.

pytest test_ct_saturation_dataset.py::test_analysis_restores_same_64_token_model_as_training
Результат: 1 passed после задания workspace basetemp.

pytest test_phase5_contracts.py test_transformer_model.py
Результат: 88 passed in 2.02s.
```

### Следующий шаг

- Завершить инвентаризацию существующего spectral/checkpoint API Phase 4.
- Реализовать потоковые scan-сценарии Open_EE и French/RTE без выбора
  нормировки французского тока до получения RMS-статистики и решения
  исследователя.
