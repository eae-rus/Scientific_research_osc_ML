# Инструкции GitHub Copilot для проекта Scientific Research OSC ML

## Обзор проекта

**Scientific Research OSC ML** — это Python проект для анализа осциллограмм (COMTRADE формат) и применения машинного обучения для классификации аварийных режимов в энергосистемах. Проект сосредоточен на:
- Анализе осциллографов реальных повреждений в сетях
- Расчёте вторичных признаков (симметричные составляющие, PDR, и др.)
- Обучении ML моделей для распознавания типов повреждений
- Обработке высокоразмерных временных рядов
- **НОВОЕ:** Исследовании KAN (Kolmogorov-Arnold Networks) для интерпретируемых органов РЗА

## Ключевые документы

- **RESEARCH_PLAN_KAN_RZA.md** — Подробный план исследований по KAN для РЗА
- **TESTING_ROADMAP.md** — Стратегия тестирования и приоритеты
- Этот файл (`copilot-instructions.md`) — Инструкции для разработки

## Структура проекта

### Основные модули

```
osc_tools/                     # Основной пакет для обработки осциллограмм
├── core/                      # Ядро функционала
│   ├── comtrade_custom.py    # Парсинг COMTRADE файлов
│   ├── constants.py          # Константы (номиналы, индексы сигналов)
│
├── io/                        # Ввод-вывод данных
│   └── comtrade_parser.py    # Парсер COMTRADE → DataFrame
│
├── preprocessing/             # Предварительная обработка
│   ├── filtering.py          # Фильтрация сигналов
│   ├── segmentation.py       # Сегментация событий
│   └── ...
│
├── features/                  # Расчёт признаков
│   ├── normalization.py      # Нормализация сигналов
│   ├── pdr_calculator.py     # Расчёт PDR (расстояние до места повреждения)
│   └── ...
│
├── analysis/                  # Анализ повреждений
│   ├── overvoltage.py        # Анализ перенапряжений (ОЗЗ)
│   ├── detect_motor_starts.py # Определение пусков двигателей
│   ├── spef_finder.py        # Поиск ОЗЗ событий
│   └── ...
│
├── ml/                        # ML компоненты
│   ├── models.py             # Архитектуры нейросетей
│   └── kan_conv/             # KAN конволюционные слои
│
└── standalone_tools/          # Утилиты
```

### ML компоненты

- **ML_model/train.py** - Обучение моделей классификации
- **ML_model/train_PDR.py** - Обучение моделей для PDR
- **trained_models/** - Сохранённые веса моделей

### Тесты

```
tests/
├── unit/                      # Unit тесты для модулей
├── integration/               # Интеграционные тесты
├── test_data/                 # Фабрики для создания тестовых данных
├── fixtures/                  # Тестовые фикстуры
└── conftest.py               # Конфигурация pytest
```

## Технический стек

- **Python 3.13.5** - Основной язык
- **PyTorch** - Deep Learning фреймворк
- **pandas** - Обработка временных рядов и таблиц
- **numpy** - Численные вычисления
- **pytest** - Testing framework
- **.NET Framework 4.8** - Для чтения COMTRADE через `ReadComtrade` (Windows)

## Правила кодирования

### ⚠️ КРИТИЧНО: Оптимизация использования контекста

#### Файлы НЕ для полного чтения

Эти файлы **очень большие** (~5000+ строк):

- **Никогда** не читай полностью: `trained_models/*.pt` (бинарные файлы весов)
- **Будь осторожен**: ML модели в `ML_model/train*.py` (сложная логика, много кода)

#### ✅ Эффективный подход

**1. Для понимания структуры:**
- Используй `grep_search` для поиска нужных функций
- Читай файлы небольшими блоками (50-100 строк)
- Используй `list_code_usages` для нахождения мест вызова функций

**2. Для изменений в ML:**
- Пользователь **сам активно переписывает модели** - не привязывайся к деталям
- Фокусируйся на **контрактах функций** (что на входе, что на выходе)
- Обновляй тесты для новых сигнатур, не беспокойся о внутренней реализации

### Стиль кода и комментарии

```python
# Комментарии на РУССКОМ языке для сложной логики
def calculate_symmetrical_components(phasors: np.ndarray) -> dict:
    """
    Расчёт симметричных составляющих (прямая, обратная, нулевая).
    
    Args:
        phasors: Комплексные фазоры напряжений/токов (3 фазы)
    
    Returns:
        dict с ключами 'positive', 'negative', 'zero'
    """
    # Матрица преобразования для трёхфазной системы
    A = np.array([
        [1, 1, 1],
        [1, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)],
        [1, np.exp(4j*np.pi/3), np.exp(2j*np.pi/3)]
    ])
    
    components = (1/3) * A @ phasors
    return {
        'positive': components[0],
        'negative': components[1],
        'zero': components[2]
    }
```

### Типизация

Используй **type hints** для всех функций:

```python
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

def process_oscillogram(
    data: pd.DataFrame,
    sampling_rate: int = 1600,
    filter_cutoff: float = 50.0
) -> Dict[str, np.ndarray]:
    """Обработка осциллограммы с фильтрацией."""
    ...
    return {'filtered': filtered_signal, 'frequency': freq}
```

### Работа с конфигурациями

Конфиги хранятся в **JSON или YAML**, используй pathlib:

```python
from pathlib import Path
import json

config_path = Path(__file__).parent / 'config.json'
with open(config_path) as f:
    config = json.load(f)
```

## Тестирование

**ВАЖНО:** Любой новый функционал должен быть покрыт тестами. Это предотвращает регрессии.

### Стратегия тестирования (Three-Layer Model)

#### Layer 1: Stable Core (Full Coverage)
Полное тестирование для модулей которые **не меняются**:
- `osc_tools/core/` - Константы, утилиты
- `osc_tools/io/comtrade_parser.py` - Парсинг файлов
- `osc_tools/preprocessing/` - Фильтрация, сегментация
- `osc_tools/analysis/` - Анализ (overvoltage, motor detection)

**Что тестировать:**
- Корректность расчётов на граничных значениях
- Обработка ошибок и пустых данных
- Integration с внешними функциями

#### Layer 2: Adaptive Layer (API Contracts)
Тесты контрактов для модулей с **частыми изменениями**:
- `osc_tools/features/normalization.py` - Может меняться формат нормализации
- `osc_tools/features/pdr_calculator.py` - Могут изменяться формулы расчёта

**Что тестировать:**
- Сигнатуры функций (входы/выходы)
- Типы возвращаемых значений
- Базовые сценарии использования

#### Layer 3: Experimental Zone (Smoke Tests Only)
**Минимальные** тесты для ML компонентов:
- `osc_tools/ml/models.py` - Сложные архитектуры нейросетей
- `ML_model/train*.py` - Обучение (пользователь это переписывает)

**Что тестировать:**
- Forward pass работает без ошибок
- Форма выходного тензора правильная
- **НЕ тестируем**: веса моделей, точность на датасете, логику обучения

### Выполнение тестов

**ОБЯЗАТЕЛЬНО:** После внесения изменений запускай тесты, чтобы убедиться, что ничего не сломалось.
Тесты запускаются через ноутбук `tests/pytest_runner.ipynb`.
1. Открой ноутбук.
2. Выполни ячейки по очереди.
3. Дойди до ячейки "Запуск всех тестов" и выполни её.
4. Проверь отчет в папке `reports/`.

- **Не делаем**: Не запускай тесты напрямую из командной строки, чтобы избежать проблем с окружением.
```

### Fixtures и Mock

Используй `pytest` fixtures для переиспользуемых данных:

```python
@pytest.fixture
def sample_normalized_dataframe():
    """DataFrame с нормализованными сигналами."""
    fs = 1600
    f = 50
    t = np.arange(0, 1, 1/fs)
    
    return pd.DataFrame({
        'ia': 0.5 * np.sin(2*np.pi*f*t),
        'ib': 0.5 * np.sin(2*np.pi*f*t - 2*np.pi/3),
        'ic': 0.5 * np.sin(2*np.pi*f*t + 2*np.pi/3),
    })
```

Для файловых операций используй **mocks**. 
**Важно:** При тестировании `Comtrade.load` необходимо мокать не только `open`, но и `os.path.exists`, а также внутренние методы `_load_inf` и `_load_hdr`, чтобы избежать попыток чтения несуществующих файлов метаданных.

```python
from unittest.mock import patch, MagicMock

@patch('osc_tools.core.comtrade_custom.Comtrade._load_inf')
@patch('osc_tools.core.comtrade_custom.Comtrade._load_hdr')
@patch('os.path.exists', return_value=True)
def test_comtrade_load(mock_exists, mock_hdr, mock_inf):
    # тест без обращения к реальной ФС
```

## Типовые задачи

### Добавление нового анализатора

1. Создай файл в `osc_tools/analysis/my_analyzer.py`
2. Реализуй класс с методом `run(dataframe) -> dict`
3. Добавь type hints и docstrings
4. Напиши тесты в `tests/unit/test_analysis_my_analyzer.py`

### Добавление нового признака (Feature)

1. Добавь расчёт в `osc_tools/features/new_feature.py`
2. Функция должна принимать `pd.DataFrame`, возвращать `np.ndarray` или `dict`
3. Обнови `osc_tools/core/constants.py` если нужны новые индексы
4. Напиши тесты (Layer 1: полные)

### Работа с ML моделями

**Когда пользователь переписывает модель:**
1. Обнови type hints в `forward()` методе
2. Обнови тесты для проверки формы выходного тензора
3. **НЕ тестируй** точность обучения или веса
4. Обнови docstring с новой информацией об архитектуре

## Частые ошибки

❌ **ПЛОХО:**
```python
def analyze(data):
    # Без type hints
    return some_result
```

✅ **ХОРОШО:**
```python
def analyze(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Анализ данных. Returns: результаты анализа"""
    return {'metric1': values}
```

---

❌ **ПЛОХО:**
```python
# Тестирование деталей ML модели
assert model.layer1.weight[0,0] == 0.123
```

✅ **ХОРОШО:**
```python
# Тестирование контракта
output = model(input_tensor)
assert output.shape == (batch_size, num_classes)
```

---

❌ **ПЛОХО:**
```python
# Игнорирование edge cases
def get_dominant_frequency(signal):
    fft = np.fft.fft(signal)
    return np.argmax(fft)  # Ошибка если signal пустой
```

✅ **ХОРОШО:**
```python
def get_dominant_frequency(signal: np.ndarray) -> Optional[float]:
    """Returns dominant frequency or None if signal is empty/invalid."""
    if signal is None or len(signal) < 2:
        return None
    fft = np.fft.fft(signal)
    return float(np.argmax(np.abs(fft)))
```

❌ **ПЛОХО:**
```python
# Check if model saved
```

✅ **ХОРОШО:**
```python
# Проверьте, сохранена ли модель
```

## Полезные инструменты

- `pandas.DataFrame.describe()` - Статистика данных
- `numpy.fft` - Спектральный анализ
- `scipy.signal` - Фильтрация и обработка сигналов
- `matplotlib` - Визуализация (для debug)
- `torch.autograd` - Градиенты при необходимости

## Вопросы?

Когда в сомнениях:
1. Следуй существующим паттернам в похожих модулях
2. Читай docstrings и type hints
3. Используй `grep_search` для нахождения примеров
4. Фокусируйся на стабильном ядре (Layer 1), будь готов к изменениям в ML (Layer 3)
5. Пиши комментарии на русском для сложной логики.