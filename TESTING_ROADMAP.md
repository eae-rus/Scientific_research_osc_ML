# Testing Architecture & Roadmap

## ✅ Сделано (Iteration 1)

### Структура тестов
```
tests/
├── conftest.py                    # Глобальные pytest fixtures (балансированные/нулевые фазы, сигналы)
├── pytest_runner.ipynb            # Ноутбук для запуска тестов из Jupyter
├── test_data/
│   ├── __init__.py
│   └── fixtures.py                # Factory-функции для синтезирования тестовых данных
├── unit/
│   ├── test_constants.py          # 18 тестов для TYPE_OSC, Features, PDRFeatures
│   └── test_pdr_calculator.py     # 6 тестов для symmetrical_components
└── integration/                   # Пока пусто (для будущих интеграционных тестов)
```

### Тесты (24 тест ✅ все пройдены)
- **test_constants.py (18 тестов)**:
  - TYPE_OSC enum: 3 теста
  - Features класс: 5 тестов
  - PDRFeatures класс: 6 тестов
  - Интеграционная проверка: 1 тест (между Features и PDRFeatures)
  - Дополнительные проверки: 3 теста

- **test_pdr_calculator.py (6 тестов)**:
  - Сбалансированная система: 1 тест
  - Нулевая последовательность: 1 тест
  - Обратная последовательность: 1 тест
  - Многоточечный ввод: 1 тест
  - Комплексный вывод: 1 тест
  - Полнота разложения: 1 тест

### Инфраструктура
1. **conftest.py** с fixtures:
   - `test_data_dir` — путь к папке test_data
   - `balanced_three_phase` — сбалансированная ФФ система
   - `zero_sequence_three_phase` — нулевая последовательность
   - `negative_sequence_three_phase` — обратная последовательность
   - `multi_window_signal` — многоточечный синусоидальный сигнал

2. **test_data/fixtures.py** с factory-функциями:
   - `create_sinusoidal_signal()` — генерация синусов
   - `create_three_phase_balanced_signal()` — сбалансированная ФФ
   - `create_harmonics_signal()` — сигналы с гармониками
   - `create_phasor_balanced_system()` — фазоры
   - `create_phasor_zero_sequence()` — нулевая последовательность фазоров
   - `create_noise()` — гауссовский шум

3. **pytest_runner.ipynb** (находится в tests/):
   - Импорты и глобальные переменные
   - Функция `run_pytests()` — запуск pytest через subprocess
   - Функция `summarize_report()` — очистка ANSI-кодов и красивый вывод
   - Функции-обёртки:
     - `run_all_tests()` — все тесты
     - `run_test_file(path)` — тесты из файла
     - `run_test_node(nodeid)` — конкретный тест
     - `run_by_keyword(keyword)` — тесты по маркеру/ключевому слову
     - `run_and_raise(args)` — для CI (вызывает SystemExit при падении)
   - Логирование: тексты сохраняются в `reports/pytest_log_YYYYMMDD_HHMMSS.txt`

4. **.gitignore обновлён**:
   - `reports/` — артефакты тестов
   - `pytest_*.json`, `pytest_log_*.txt`
   - `.pytest_cache/`, `.ipynb_checkpoints/`
   - `*.pyc`, `.vscode/`, `.idea/` и т.д.

---

## 🎯 Планируемая структура (итоговая)

```
tests/
├── conftest.py
├── pytest_runner.ipynb
├── pytest.ini
├── test_data/
│   ├── __init__.py
│   ├── fixtures.py
│   ├── signals/                   # Синтезированные сигналы (если большие)
│   └── comtrade/                  # Минимальные .cfg/.dat файлы
├── unit/
│   ├── test_constants.py          # ✅ DONE
│   ├── test_pdr_calculator.py     # ✅ DONE
│   ├── test_filtering.py          # ⚠️ needs refactor
│   ├── test_normalization.py      # ⚠️ needs refactor
│   └── test_io.py                 # ⚠️ needs refactor
└── integration/
    ├── test_comtrade_io.py        # Чтение/запись COMTRADE
    └── test_processing_pipeline.py # E2E обработка данных
```

---

## 📋 Следующие шажочки (Iteration 2 & 3)

### Шаг 1: ✅ Переместить и переделать старые тесты
- [x] Старые unit-тесты удалены (правильно, начали заново)
- [x] Текущая структура tests/ чистая и готова к новым тестам
- [x] Базовые unit-тесты (24 теста) работают без проблем

### Шаг 2: ✅ Переместить comtrade_processing.py
- [x] Переместить `dataflow/comtrade_processing.py` → `osc_tools/data_management/comtrade_processing.py`
- [x] Обновить импорты в 5 файлах:
  - `osc_tools/preprocessing/filtering.py`
  - `osc_tools/io/comtrade_parser.py`
  - `osc_tools/features/normalization.py`
  - `osc_tools/analysis/overvoltage.py`
  - `osc_tools/analysis/detect_motor_starts.py`
- [x] Импорты обновлены с `from dataflow.comtrade_processing` → `from osc_tools.data_management.comtrade_processing`
- [x] Написать интеграционный тест для `comtrade_processing.py` в `tests/integration/`
  - Созданы базовые тесты для `ReadComtrade` класса (успешное чтение и обработка ошибок)

### Шаг 3: Добавить тесты для preprocessing модулей
- [x] **tests/unit/test_preprocessing_filtering.py** — тесты для `osc_tools/preprocessing/filtering.py` (11 тестов)
  - ✅ Базовая синусоида (test_sliding_window_fft_basic_sine_wave)
  - ✅ Короткий сигнал (test_sliding_window_fft_signal_too_short)
  - ✅ Несколько гармоник (test_sliding_window_fft_multiple_harmonics)
  - ✅ Нулевой сигнал (test_sliding_window_fft_zero_signal)
  - ✅ Форма и dtype (test_sliding_window_fft_output_shape_and_dtype)
  - ✅ Границы гармоник (test_sliding_window_fft_harmonic_index_bounds)
  - ✅ Постоянный сигнал (test_sliding_window_fft_constant_signal)
  - ✅ Точный размер окна (test_sliding_window_fft_exactly_window_size)
  - ✅ С шумом (test_sliding_window_fft_with_noise)
  - ✅ Отрицательные значения (test_sliding_window_fft_negative_values)
  - ✅ Вспомогательная функция is_complex_nan для комплексных NaN

- [ ] **tests/unit/test_preprocessing_segmentation.py** — тесты для `osc_tools/preprocessing/segmentation.py`
  - Базовая сегментация
  - Краевые случаи
  - Сигналы различной длины

### Шаг 4: ✅ Тесты для edge cases в pdr_calculator
- [x] **tests/unit/test_pdr_calculator_edge_cases.py** — расширенные тесты (11 тестов)
  - ✅ Edge cases для symmetrical_components
  - ✅ Edge cases для sliding_window_fft
  - ✅ Тесты численной стабильности

### Шаг 5: ⏳ Тесты для segmentation/normalization
- [ ] **tests/unit/test_preprocessing_segmentation.py** — тесты для segmentation.py
  - Вспомогательные функции
  - OscillogramEventSegmenter edge cases
  
- [ ] **tests/unit/test_features_normalization.py** — тесты для normalization.py
  - CreateNormOsc edge cases
  - NormOsc методы

---

## 🔧 Техинформация

### Запуск тестов из ноутбука (tests/pytest_runner.ipynb):
```python
# Все тесты
run_all_tests()

# Только unit-тесты
run_by_keyword('unit')

# Конкретный файл
run_test_file('tests/unit/test_constants.py')

# Конкретный тест
run_test_node('tests/unit/test_constants.py::TestTypeOSCEnum::test_type_osc_members_exist')

# Для CI (вызывает SystemExit если упало)
run_and_raise(['tests/unit'])
```

### Структура маркеров pytest:
```python
@pytest.mark.unit          # Быстрые unit-тесты (обязательны для каждого коммита)
@pytest.mark.integration   # Интеграционные тесты (E2E, медленнее)
@pytest.mark.slow          # Медленные тесты (>5 сек)
```

### Логирование:
- Все логи сохраняются в `reports/pytest_log_YYYYMMDD_HHMMSS.txt`
- ANSI-коды удаляются для читаемости
- Каждый запуск создаёт новый файл логов

---

## 📊 Текущее состояние метрик

| Метрика | Значение |
|---------|---------|
| Всего тестов | 46+ ✅ |
| Unit-тесты | 24 (constants + pdr_calculator) + 11 (filtering) + 11 (pdr_edge_cases) = 46 |
| Integration-тесты | ~3 (ReadComtrade) |
| Coverage (approx) | ~28% (constants + pdr_calculator + filtering + pdr_edge_cases + comtrade_io) |
| Pass rate | 100% ✅ |

### Новое (Iteration 3):
- ✅ **test_pdr_calculator_edge_cases.py** — 11 тестов для edge cases
  - ✅ Нулевые входы (test_symmetrical_components_all_zeros)
  - ✅ Одна точка (test_symmetrical_components_single_point)
  - ✅ Очень маленькие значения (test_symmetrical_components_very_small_values)
  - ✅ Несовпадающие длины (test_symmetrical_components_mismatched_length)
  - ✅ Отрицательные фазы (test_symmetrical_components_negative_phase)
  - ✅ Длинные сигналы (test_sliding_window_fft_very_long_signal)
  - ✅ Окно больше сигнала (test_sliding_window_fft_window_larger_than_signal)
  - ✅ Одна гармоника (test_sliding_window_fft_single_harmonic_request)
  - ✅ Высокочастотный сигнал (test_sliding_window_fft_high_frequency_signal)
  - ✅ DC компонента (test_sliding_window_fft_dc_component)
  - ✅ Численная стабильность (test_repeated_calculations_consistency, test_symmetrical_components_numerical_error)

---

## 🚀 Рекомендуемый порядок работ (UPDATED)

✅ **Завершено (Iteration 1-3)**:
- Базовая структура тестов
- Перемещение comtrade_processing.py
- Unit-тесты для filtering.py (sliding_window_fft) — 11 тестов
- Unit-тесты для edge cases pdr_calculator.py — 11 тестов
- Интеграционные тесты для ReadComtrade — 3 теста
- **Всего: 46 тестов, 100% pass rate**

🔜 **Следующие шаги (Iteration 4+)**:
1. **Шаг 5**: Тесты для segmentation.py (~8-10 тестов)
2. **Шаг 6**: Тесты для normalization.py (~10-12 тестов)
3. **Шаг 7**: Тесты для io/comtrade_parser.py (~8 тестов)
4. **Шаг 8**: E2E интеграционные тесты (~5-7 тестов)
5. **Шаг 9**: Coverage и маркеры для slow тестов

**Целевой план**: 75-85 тестов с coverage >65%

---

## 📝 Примечания

- Все тесты выполняются **независимо** (не должны полагаться на порядок выполнения)
- Используются **factories** из `test_data/fixtures.py` для синтезирования данных (нет привязки к реальным файлам)
- Ноутбук `pytest_runner.ipynb` находится в `tests/` и автоматически определяет `PROJECT_ROOT`
- При добавлении новых тестов: добавляйте их в соответствующие файлы и используйте существующие fixtures
