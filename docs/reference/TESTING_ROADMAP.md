# Testing Architecture & Roadmap

> **Последнее обновление:** 4 января 2026  
> **Контекст:** Проект достиг 100% прохождения тестов (320+ тестов). Добавлены тесты для новых ML компонентов и оптимизаций.

---

## 📋 Оглавление

1. [Философия тестирования](#-философия-тестирования)
2. [Текущее состояние](#-текущее-состояние-259-тестов)
3. [Архитектура проекта и приоритеты](#-архитектура-проекта-и-приоритеты-тестирования)
4. [Целевая структура тестов](#-целевая-структура-тестов)
5. [Roadmap по итерациям](#-roadmap-по-итерациям)
6. [ML-тестирование (специальный раздел)](#-ml-тестирование-специальный-раздел)
7. [Техническая информация](#-техническая-информация)
8. [Инфраструктура](#-инфраструктура)

---

## 🧠 Философия тестирования

### Принцип "Стабильное ядро" 

```
┌─────────────────────────────────────────────────────────────┐
│                    🧪 ЭКСПЕРИМЕНТАЛЬНАЯ ЗОНА                │
│         ML_model/, osc_tools/ml/ — активные эксперименты    │
│              ↓ Тесты: минимальные, smoke-тесты ↓            │
├─────────────────────────────────────────────────────────────┤
│                   🔄 АДАПТИВНЫЙ СЛОЙ                        │
│     osc_tools/features/ — может меняться с новыми данными   │
│            ↓ Тесты: средний приоритет, API-контракты ↓      │
├─────────────────────────────────────────────────────────────┤
│                   🏛️ СТАБИЛЬНОЕ ЯДРО                        │
│   osc_tools/core/, io/, preprocessing/, analysis/           │
│          ↓ Тесты: ВЫСОКИЙ приоритет, полное покрытие ↓      │
└─────────────────────────────────────────────────────────────┘
```

### Почему именно так?

| Слой | Что там | Частота изменений | Стратегия тестирования |
|------|---------|-------------------|------------------------|
| **Ядро** | constants, comtrade_parser, filtering, analysis | Редко | Полное покрытие, edge cases |
| **Адаптивный** | normalization, pdr_calculator, features | Средне | API-контракты, regression |
| **Экспериментальный** | models.py (1479 строк!), train_PDR.py | Часто | Smoke-тесты, проверка shapes |

---

## 📊 Текущее состояние (320+ тестов)

### Структура тестов
```
tests/
├── conftest.py                          # ✅ Глобальные fixtures
├── pytest_runner.ipynb                  # ✅ Запуск из Jupyter
├── test_data/
│   ├── __init__.py
│   └── fixtures.py                      # ✅ Factory-функции
├── unit/
│   ├── test_core_comtrade_custom.py     # ✅ 18 тестов (Layer 1: Core)
│   ├── test_constants.py                # ✅ 18 тестов
│   ├── test_pdr_calculator.py           # ✅ 6 тестов  
│   ├── test_pdr_calculator_edge_cases.py# ✅ 11 тестов
│   ├── test_features_pdr_calculator.py  # ✅ 7 тестов (Layer 2: Math helpers)
│   ├── test_preprocessing_filtering.py  # ✅ 11 тестов
│   ├── test_preprocessing_segmentation.py # ✅ 23 теста
│   ├── test_comtrade_parser.py          # ✅ 11 тестов
│   ├── test_analysis_overvoltage.py     # ✅ 16 тестов
│   ├── test_analysis_motor_and_spef.py  # ✅ 16 тестов
│   ├── test_features_normalization.py   # ✅ 18 тестов
│   ├── test_data_management_renaming.py # ✅ 16 тестов
│   ├── test_data_management_complete.py # ✅ 15 тестов
│   ├── test_ml_infrastructure.py        # ✅ 4 теста (Dataset, Samplers)
│   ├── test_ml_models.py                # ✅ 2 теста (Layer 3: Smoke tests)
│   ├── test_benchmark.py                # ✅ 5 тестов (InferenceBenchmark)
│   ├── test_dataset_features.py         # ✅ 8 тестов (Feature modes)
│   ├── test_fft_speed.py                # ✅ 3 теста (FFT optimizations)
│   ├── test_ml_kan.py                   # ✅ 4 теста (KAN layers)
│   ├── test_ml_models_baseline.py       # ✅ 6 тестов (MLP/CNN baselines)
│   ├── test_ml_models_basic.py          # ✅ 2 теста (Basic model tests)
│   ├── test_ml_models_kan.py            # ✅ 5 тестов (ConvKAN specifics)
│   ├── test_ml_models_structure.py      # ✅ 7 тестов (Model architectures)
│   ├── test_ml_runner.py                # ✅ 3 теста (ExperimentRunner)
│   ├── test_ml_training_smoke.py        # ✅ 2 теста (Training smoke tests)
│   ├── test_scripts_infrastructure.py   # ✅ 4 теста (Script utilities)
│   ├── test_features_pdr_complete.py    # ✅ 12 тестов (PDR edge cases)
│   ├── test_io_comtrade_parser_complete.py # ✅ 9 тестов (Parser robustness)
│   ├── test_analysis_overvoltage_fixed.py # ✅ 14 тестов (Overvoltage fixes)
│   └── ...
└── integration/
    └── test_comtrade_io.py              # ✅ 24 теста
```

### Метрики

| Метрика | Значение |
|---------|----------|
| Всего тестов | **320+** (было 276, добавлено 44+) |
| Pass rate | **100%** ✅ |
| Покрытие (approx) | ~70% |
| Время выполнения | ~25 сек |

**Примечание:** Добавлены тесты для KAN слоев, benchmark'инга, новых feature modes и инфраструктуры экспериментов.


---

## 🎯 Архитектура проекта и приоритеты тестирования

### Карта модулей с приоритетами

```
osc_tools/
├── core/                           ← ПРИОРИТЕТ 1 (стабильное)
│   ├── constants.py                   ✅ 18 тестов (DONE)
│   └── comtrade_custom.py             ✅ 18 тестов (DONE: Read/Write/Save)
│
├── io/                             ← ПРИОРИТЕТ 1
│   └── comtrade_parser.py             ✅ 11 тестов (DONE)
│
├── preprocessing/                  ← ПРИОРИТЕТ 1
│   ├── filtering.py                   ✅ 11 тестов (DONE)
│   └── segmentation.py                ✅ 23 теста (DONE)
│
├── analysis/                       ← ПРИОРИТЕТ 1
│   ├── overvoltage.py                 ✅ 16 тестов (DONE)
│   ├── detect_motor_starts.py         ✅ 16 тестов (DONE)
│   └── spef_finder.py                 ✅ 16 тестов (DONE)
│
├── features/                       ← ПРИОРИТЕТ 2 (адаптивный слой)
│   ├── pdr_calculator.py              ✅ 24 теста (DONE)
│   └── normalization.py               ✅ 18 тестов (DONE)
│
├── data_management/                ← ПРИОРИТЕТ 2
│   ├── comtrade_processing.py         ✅ ~24 integration теста
│   ├── processing.py                  📅 Нужны тесты
│   └── search.py, renaming.py         📅 Низкий приоритет
│
└── ml/                             ← ПРИОРИТЕТ 3 (экспериментальный)
    ├── models.py (1479 строк!)        ⚠️ Smoke-тесты только
    └── kan_conv/                      ⚠️ После стабилизации

ML_model/                           ← ПРИОРИТЕТ 3
├── train_PDR.py                       ✅ Тесты CustomDataset, Sampler
└── train.py                           ✅ Тесты CustomDataset
```

---

## 🏗️ Целевая структура тестов

```
tests/
├── conftest.py
├── pytest_runner.ipynb
├── pytest.ini
│
├── test_data/
│   ├── __init__.py
│   ├── fixtures.py                    # Сигналы, фазоры
│   ├── ml_fixtures.py                 # 🆕 Тензоры, батчи для ML
│   └── comtrade/                      # Минимальные .cfg/.dat
│
├── unit/                              # Быстрые изолированные тесты
│   ├── core/
│   │   └── test_constants.py          ✅
│   ├── io/
│   │   └── test_comtrade_parser.py    ✅
│   ├── preprocessing/
│   │   ├── test_filtering.py          ✅
│   │   └── test_segmentation.py       🔜
│   ├── features/
│   │   ├── test_pdr_calculator.py     ✅
│   │   └── test_normalization.py      📅
│   └── analysis/
│       ├── test_overvoltage.py        📅
│       ├── test_motor_starts.py       📅
│       └── test_spef_finder.py        📅
│
├── integration/                       # E2E тесты
│   ├── test_comtrade_io.py            ✅
│   └── test_data_pipeline.py          📅 
│
└── ml/                                # 🆕 Отдельная категория для ML
    ├── test_model_shapes.py           # Проверка размерностей
    ├── test_model_components.py       # cLeakyReLU, Conv_3 и др.
    ├── test_dataset_sampler.py        # CustomDataset, Sampler
    └── test_training_smoke.py         # Forward pass, backward pass
```

---

## 🚀 Roadmap по итерациям

### ✅ Завершено (Iteration 1-4, ноябрь 2025)

- [x] Базовая инфраструктура (conftest, fixtures, pytest_runner)
- [x] `test_constants.py` — 18 тестов
- [x] `test_pdr_calculator.py` — 6 тестов + 11 edge cases
- [x] `test_preprocessing_filtering.py` — 11 тестов
- [x] `test_comtrade_parser.py` — 11 тестов
- [x] `test_comtrade_io.py` — 3 integration теста

### 🔜 Iteration 5: Preprocessing & Analysis (ТЕКУЩИЙ)

**Цель:** Покрыть стабильное ядро обработки сигналов

| Задача | Тесты | Статус |
|--------|-------|--------|
| `segmentation.py` | ~10 | 🔜 NEXT |
| `overvoltage.py` | ~8 | 📅 |
| `detect_motor_starts.py` | ~5 | 📅 |
| `spef_finder.py` | ~5 | 📅 |

**Ожидаемый результат:** ~85 тестов

### 📅 Iteration 6: Адаптивный слой

**Цель:** Тесты для компонентов, которые могут меняться

| Задача | Тесты | Заметки |
|--------|-------|---------|
| `normalization.py` | ~10 | API-контракт тесты |
| `comtrade_processing.py` (расширить) | ~5 | Integration |
| `processing.py` | ~5 | Если стабилизируется |

**Ожидаемый результат:** ~105 тестов

### 📅 Iteration 7: ML Smoke-тесты

**Цель:** Базовые проверки для ML-кода (после стабилизации архитектуры моделей)

| Задача | Тесты | Стратегия |
|--------|-------|-----------|
| Model shapes | ~5 | Проверка input/output размерностей |
| Forward pass | ~3 | Smoke-тест без обучения |
| CustomDataset | ~5 | Корректность батчей |
| MultiFileRandomPointSampler | ~3 | Логика сэмплирования |

**Триггер:** Начинать после фиксации базовой архитектуры моделей

---

## 🤖 ML-тестирование (специальный раздел)

### Почему ML тестируется иначе?

1. **Модели экспериментальные** — `models.py` содержит 1479 строк с TODO и FIXME
2. **Архитектура будет меняться** — планируется много новых моделей
3. **Тесты должны быть устойчивы** — не ломаться при рефакторинге

### Стратегия ML-тестирования

```python
# ❌ ПЛОХО: Тест привязан к конкретной архитектуре
def test_pdr_mlp_v1_exact_output():
    model = PDR_MLP_v1(frame_size=64)
    output = model(torch.randn(1, 4, 1))
    assert output.shape == (1, 1)
    assert output[0, 0] == 0.5  # Слишком жёсткое условие!

# ✅ ХОРОШО: Тест проверяет контракт
def test_pdr_mlp_v1_output_contract():
    """Модель должна принимать (batch, channels, features) и возвращать (batch, 1)"""
    model = PDR_MLP_v1(frame_size=64)
    batch_size = 8
    output = model(torch.randn(batch_size, 4, 1))
    
    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.float32
    assert torch.all((output >= 0) & (output <= 1))  # После sigmoid
```

### Что тестировать в ML

| Компонент | Что тестировать | Пример |
|-----------|----------------|--------|
| Слои (cLeakyReLU, cMaxPool1d) | Контракт вход/выход | Shape, dtype сохраняются |
| Модели | Forward pass не падает | Любой валидный input |
| Dataset | __getitem__ возвращает корректный формат | (x, target) tuples |
| Sampler | Генерирует валидные индексы | В пределах данных |

### Когда НЕ тестировать ML

- ❌ Точные значения весов после обучения
- ❌ Конкретные значения loss на определённой эпохе
- ❌ Архитектурные детали, которые будут меняться

---

## 🔧 Техническая информация

### Запуск тестов

```bash
# Все тесты
python -m pytest tests/ -v

# Только unit
python -m pytest tests/unit/ -v

# Конкретный файл
python -m pytest tests/unit/test_constants.py -v

# По маркеру
python -m pytest -m "unit and not slow" -v

# С покрытием
python -m pytest tests/ --cov=osc_tools --cov-report=html
```

### Из Jupyter (tests/pytest_runner.ipynb)

```python
run_all_tests()                    # Все тесты
run_by_keyword('unit')             # По маркеру
run_test_file('tests/unit/test_constants.py')  # Файл
```

### Маркеры pytest

```python
@pytest.mark.unit          # Быстрые unit-тесты (<1 сек)
@pytest.mark.integration   # Тесты с файлами, E2E
@pytest.mark.slow          # Медленные тесты (>5 сек)
@pytest.mark.ml            # ML-специфичные тесты
```

### Паттерны тестирования

**Mock-based (для файловых операций):**
```python
@patch('os.path.exists', return_value=True)
@patch('json.load')
def test_something(self, mock_json, mock_exists):
    mock_json.return_value = {'key': 'value'}
    # test code
```

**Factory-based (для сигналов):**
```python
from test_data.fixtures import create_sinusoidal_signal

def test_filtering():
    signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600)
    result = sliding_window_fft(signal, window_size=32)
    assert result.shape[0] == len(signal) - 32 + 1
```

### Windows-специфичные проблемы

- ❌ НЕ использовать `tmp_path` fixture (permission errors)
- ✅ Использовать `patch('os.path.exists')` и mock
- ✅ Абсолютные пути для всех файловых операций

---

## 🛠️ Инфраструктура

### conftest.py fixtures

| Fixture | Описание | Когда использовать |
|---------|----------|-------------------|
| `test_data_dir` | Путь к tests/test_data/ | Доступ к тестовым данным |
| `sample_normalized_dataframe` | 1600 samples, 3-фазные U и I, 2 шины | Layer 1/2 тесты обработки |
| `sample_signal_with_event` | DF с событием во второй половине | Тесты обнаружения событий |
| `balanced_three_phase` | Сбалансированная ФФ система (phasor) | Расчёты симметричных составляющих |
| `zero_sequence_three_phase` | Нулевая последовательность | Тесты защиты от ОЗЗ |
| `negative_sequence_three_phase` | Обратная последовательность | Тесты несимметрии |
| `multi_window_signal` | 50 Гц синусоида @ 1600 Гц | Тесты FFT, фильтрации |
| `fixture_comtrade_dir` | Временная директория с mock COMTRADE (.cfg/.dat) | Интеграционные тесты парсера |

### test_data/fixtures.py factories

| Factory | Описание |
|---------|----------|
| `create_sinusoidal_signal()` | Синусоида с параметрами |
| `create_three_phase_balanced_signal()` | Три фазы со сдвигом 120° |
| `create_harmonics_signal()` | Сигнал с гармониками |
| `create_phasor_balanced_system()` | Комплексные фазоры |
| `create_noise()` | Гауссовский шум |

### 🆕 Планируемые ML fixtures (test_data/ml_fixtures.py)

```python
def create_batch_tensor(batch_size=8, channels=4, seq_len=64):
    """Создаёт тензор для тестирования моделей."""
    return torch.randn(batch_size, channels, seq_len)

def create_complex_batch(batch_size=8, channels=4, seq_len=64):
    """Создаёт комплексный тензор."""
    return torch.randn(batch_size, channels, seq_len, dtype=torch.cfloat)

def create_mock_dataframe_for_dataset(rows=1000):
    """DataFrame с колонками из PDRFeatures.ALL_MODEL_1."""
    ...
```

---

## 📈 Целевые метрики

| Метрика | Сейчас | Цель (Iter 5) | Цель (Iter 7) |
|---------|--------|---------------|---------------|
| Тесты | 58 | 85 | 120+ |
| Coverage | ~32% | ~50% | ~65% |
| Pass rate | 100% | 100% | 100% |
| Время | <5 сек | <10 сек | <30 сек |

---

## 📝 История изменений

### 23 декабря 2025 (Системный анализ)
- Переработана структура документа
- Добавлена философия тестирования с учётом ML-разработки
- Выделен специальный раздел ML-тестирования
- Обновлены приоритеты с учётом планов по нейросетям
- Добавлена карта модулей с приоритетами

### 21 ноября 2025 (Iteration 4)
- 58 тестов, 100% pass rate
- Добавлены тесты comtrade_parser.py
- Исправлены проблемы с JSON mock структурой

---

## 💡 Рекомендации

1. **Не тратьте время на ML-тесты сейчас** — модели активно меняются
2. **Фокус на preprocessing и analysis** — это стабильное ядро
3. **Создайте ml_fixtures.py** когда начнёте ML-тесты
4. **Используйте smoke-тесты для ML** — проверяйте контракты, не значения
5. **Обновляйте этот документ** после каждой итерации
