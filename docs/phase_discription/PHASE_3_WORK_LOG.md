# Лог работ по Фазе 3 (Новые архитектуры и библиотеки KAN)

## [2026-03-01] Инкремент 8: авто-сводка результатов Phase 3 (train + benchmark)

### Выполненные работы

1. Добавлен скрипт сводки
   - Создан [scripts/phase3_experiments/summarize_phase3_results.py](scripts/phase3_experiments/summarize_phase3_results.py).
   - Скрипт собирает:
     - training summary по папкам `experiments/phase3/*`;
     - latest benchmark summary по `reports/phase3/phase3_benchmark_*.json`;
     - ranking backend-ов по `best_val_f1`.

2. Добавлен автозапуск сводки из единого раннера
   - В [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py) подключён вызов `run_phase3_summary(...)` в конце запуска.
   - Добавлен флаг `--no-summary` для отключения автосводки при необходимости.

3. Форматы итогов
   - Сводка сохраняется в `reports/phase3` в CSV/JSON:
     - `phase3_training_summary_*.csv`
     - `phase3_benchmark_latest_*.csv`
     - `phase3_backend_ranking_*.csv`
     - `phase3_summary_*.json`

## [2026-03-01] Инкремент 7: диагностика benchmark-ошибок и пакетный train backend-ов

### Выполненные работы

1. Разобрана причина падения backend `wav`
   - Причина: в [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py) функция `_create_wav_kan_linear` могла возвращать `None` (из-за отсутствия return-блока после `layer_cls is not None`).
   - Симптом в benchmark: `TypeError: 'NoneType' object is not callable` внутри `nn.Sequential` головы `PhysicsKANConditional`.
   - Фикс: восстановлен блок `try/return layer_cls(...)` + fallback на baseline.

2. Повторная проверка backend-ов после фикса
   - На локальном диагностическом прогоне `cheby`, `wav`, `torch_wavelet` проходят forward/train/inference без падений.

3. Добавлен режим пакетного обучения для итогового сравнения точности
   - В [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py) добавлен аргумент `--train-backends`.
   - Позволяет запускать последовательное обучение нескольких backend-ов (например, на 60 эпох) одной командой.

## [2026-03-01] Инкремент 6: ускорение валидации и backend torch_wavelet

### Выполненные работы

1. Ускорение валидации через шаг окон
   - В [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py) добавлен параметр `--val-index-stride` (по умолчанию `16`).
   - В [scripts/phase3_experiments/run_phase3_benchmark.py](scripts/phase3_experiments/run_phase3_benchmark.py) добавлен аналогичный параметр `--val-index-stride`.
   - Это уменьшает количество валидационных окон (и шагов), ускоряя `val` при сохранении downsampling признаков.

2. Подключение `torch-wavelet-kan`
   - В [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py) добавлен backend `torch_wavelet`.
   - Источник: локальная копия `torch-conv-kan-main/kans/layers.py` (`WavKANLayer`).
   - Добавлены CLI choices `torch_wavelet` для train и benchmark в `run_phase3_libraries.py`.

3. Расширен список backend-ов по умолчанию
   - Benchmark теперь по умолчанию использует `baseline,efficient,fast,cheby,wav,torch_wavelet`.

## [2026-03-01] Инкремент 5: единая точка запуска и интеграция backend-ов fast/cheby/wav

### Выполненные работы

1. Объединён сценарий запуска Фазы 3
   - Основной entrypoint: [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py).
   - Добавлен параметр `--mode {train,benchmark,both}`:
     - `train` — обучение одного backend;
     - `benchmark` — сравнение backend-ов по скорости/памяти/стабильности;
     - `both` — последовательный запуск train + benchmark.

2. Расширены backend-ы KAN в обёртке
   - Обновлён [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py).
   - Добавлена поддержка backend-ов: `baseline`, `efficient`, `fast`, `cheby`, `wav`.
   - Источник библиотек: локальные копии в `osc_tools/ml/*-main` и `osc_tools/ml/*-master`.
   - Для несовместимых сигнатур сохранён безопасный fallback на baseline.

3. Переиспользование benchmark-кода
   - В [scripts/phase3_experiments/run_phase3_benchmark.py](scripts/phase3_experiments/run_phase3_benchmark.py) добавлена публичная функция `run_phase3_benchmark(...)`.
   - `run_phase3_libraries.py` использует её напрямую, чтобы не дублировать логику сравнения.

4. Расширены параметры CLI Фазы 3
   - `--kan-backend` теперь принимает: `baseline|efficient|fast|cheby|wav`.
   - Добавлены benchmark-параметры (`--benchmark-backends`, `--benchmark-train-steps`, `--benchmark-val-steps`, батчи и устройство).

## [2026-03-01] Инкремент 4: подготовка benchmark-контура для сравнений backend-ов

### Выполненные работы

1. Добавлен отдельный benchmark-скрипт Фазы 3
   - Создан [scripts/phase3_experiments/run_phase3_benchmark.py](scripts/phase3_experiments/run_phase3_benchmark.py).
   - Сценарий сравнивает backend-ы (`baseline`, `efficient`) на фиксированной модели `PhysicsKANConditional`.

2. Реализованы метрики сравнения для короткого прогона
   - Время train-step (среднее/STD).
   - Стабильность лосса на train (`train_loss_std`) и средний val loss.
   - Инференс-латентность/FPS через `InferenceBenchmark`.
   - Пиковая VRAM (CUDA) и RSS процесса (если доступен `psutil`).

3. Реализовано сохранение отчётов
   - Результаты benchmark сохраняются в `reports/phase3` в форматах JSON и CSV.
   - Имена файлов включают `run-name` и timestamp для удобного сравнения серий запусков.

4. Сохранена изоляция Фазы 3
   - Используются только `PhysicsKANConditional`, `phase_polar_h1_angle`, `stride`, `num_harmonics=9`.
   - Для валидации используется отдельный precomputed-файл Фазы 3 (`test_precomputed_phase3.csv`).

## [2026-03-01] Инкремент 3: корректировка гармоник для Фазы 3 (амплитуды всех, угол только h1)

### Выполненные работы

1. Исправлена стратегия гармоник в соответствии с требованиями Фазы 3
   - Убрана логика «по умолчанию 1 гармоника» для Фазы 3.
   - Теперь по умолчанию используется `num_harmonics=9`.
   - Амплитуды сохраняются для всех гармоник, углы сохраняются только для 1-й гармоники.

2. Добавлен отдельный режим признаков `phase_polar_h1_angle`
   - Обновлён [osc_tools/ml/dataset.py](osc_tools/ml/dataset.py): добавлен on-the-fly режим `phase_polar_h1_angle`.
   - Обновлён [osc_tools/ml/precomputed_dataset.py](osc_tools/ml/precomputed_dataset.py): поддержка колонок нового режима.
   - Обновлён [osc_tools/data_management/dataset_manager.py](osc_tools/data_management/dataset_manager.py):
     - генерация колонок `phase_polar_h1_angle`;
     - параметр `phase_polar_h1_angle_only` в `create_precomputed_test_csv`;
     - поддержка отдельного имени precomputed-файла через `output_filename`/`precomputed_filename`.

3. Изоляция Фазы 3 от старых этапов
   - В [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py):
     - добавлен отдельный precomputed-файл `test_precomputed_phase3.csv`;
     - feature mode переключён на `phase_polar_h1_angle`;
     - дефолт `--num-harmonics` изменён на `9`.
   - Старые пайплайны Фазы 2.x продолжают использовать прежние режимы и старый `test_precomputed.csv`.

## [2026-03-01] Инкремент 2: минимальная backend-абстракция для PhysicsKANConditional

### Выполненные работы

1. Добавлен backend-адаптер KAN для Фазы 3
   - Создан [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py).
   - Реализована фабрика `build_kan_linear(backend=...)` с поддержкой:
     - `baseline` (текущий KANLinear проекта);
     - `efficient` (попытка использовать efficient-kan с безопасным fallback на baseline).

2. Подключён backend в `PhysicsKANConditional`
   - Обновлён [osc_tools/ml/models/kan.py](osc_tools/ml/models/kan.py):
     - добавлен параметр `kan_backend` (по умолчанию `baseline`);
     - KAN-головы (`head_normal`, `head_ml1`, `head_ml2`, `head_ml3`) теперь создаются через `build_kan_linear`.
   - Важно: экстрактор признаков на `KANConv1d` оставлен без изменений для минимально рискованного старта.

3. Расширен раннер Фазы 3
   - В [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py) добавлен CLI аргумент `--kan-backend` (`baseline`/`efficient`).
   - Выбранный backend прокидывается в модель через `run_single_experiment(..., model_param_overrides={'kan_backend': ...})`.

4. Минимальное расширение общего helper-а запусков
   - В [scripts/phase2_experiments/run_phase2_6.py](scripts/phase2_experiments/run_phase2_6.py) `run_single_experiment` получил опциональный параметр `model_param_overrides`.
   - Изменение backward-compatible: старые вызовы не затронуты.

### Следующий шаг

- Добавить smoke-тесты на инициализацию/forward для `PhysicsKANConditional` с `kan_backend=baseline|efficient` и фиксированными параметрами Фазы 3.

## [2026-03-01] Старт Фазы 3: выделение отдельного раннера и фиксация минимального контура

### Выполненные работы

1. Выделен отдельный скрипт запуска для Фазы 3
   - Создан [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py).
   - Скрипт переиспользует тренировочный контур из Фазы 2.6 (`run_single_experiment`) без изменений старого пайплайна.

2. Зафиксированы ограничения первого инкремента Фазы 3
   - Модель: только `PhysicsKANConditional`.
   - Сложность: только `heavy`.
   - Режим признаков: только `phase_polar`.
   - Sampling: только `stride` (`downsampling_stride=16`).
   - Целевой уровень: `base_sequential`.

3. Добавлено принудительное отключение высших гармоник по умолчанию
   - Параметр `--num-harmonics` добавлен в CLI раннера.
   - Значение по умолчанию: `1` (используется только основная гармоника).
   - Цель: уменьшение размерности признаков и упрощение пространства экспериментов.

4. Обновлён план Фазы 3
   - В [docs/phase_discription/PHASE_3_PLAN.md](docs/phase_discription/PHASE_3_PLAN.md) отмечен старт реализации и добавлен этап минимального старта.

### Примечание

На данном шаге фокус — инфраструктурный: безопасно отделить Фазу 3 от Фазы 2.6 и подготовить базу для сравнений библиотек (EfficientKAN/FastKAN/ChebyKAN) в следующих инкрементах.
