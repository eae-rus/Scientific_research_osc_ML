# Лог работ по Фазе 3 (Новые архитектуры и библиотеки KAN)

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
