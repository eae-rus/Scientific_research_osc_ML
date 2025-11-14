
# Scientific_research_osc_ML

## Описание проекта
Репозиторий предназначен для хранения и развития кода по предобработке, анализу и исследованию осциллограмм электроэнергетических объектов с применением методов машинного обучения и глубокого обучения.

## Требуемые библиотеки
### Основные библиотеки для работы проекта:
- numpy
- pandas
- tqdm
- pytest
- os, sys, shutil, re, csv, json, datetime, pathlib, hashlib (стандартная библиотека Python)

### Для работы с архивами и файлами:
- py7zr (работа с 7z-архивами)
- zipfile (стандартная библиотека, работа с zip)
- aspose.zip (опционально, для расширенной работы с архивами)

### Для визуализации и анализа (используются в ноутбуке 003_analisis.ipynb):
- matplotlib
- seaborn

### Для работы с ноутбуками:
- jupyter
- ipywidgets (опционально, для интерактивности)

### Для машинного обучения (опционально, если потребуется):
- scikit-learn
- torch, tensorflow, keras и др. (в зависимости от задач)

## Зависимости по блокам/ноутбукам
#### 001_run_search_oscillograms_guide.ipynb
- numpy, pandas, tqdm, os, sys, shutil, json, csv, datetime, pathlib
- py7zr, zipfile, aspose.zip (для работы с архивами)

#### 002_run_processing_oscillograms_guide.ipynb
- numpy, pandas, tqdm, os, sys, shutil, json, csv, datetime
- osc_tools.data_management.processing, renaming (модули проекта)

#### 003_analisis.ipynb
- numpy, pandas
- matplotlib, seaborn

## Установка зависимостей
Рекомендуется использовать виртуальное окружение (venv или conda).

1. Создайте и активируйте виртуальное окружение:
	```bash
	python -m venv venv
	.\venv\Scripts\activate
	```
2. Установите основные зависимости:
	```bash
	pip install -r requirements.txt
	```
3. Для работы с архивами и визуализацией (если требуется):
	```bash
	pip install py7zr matplotlib seaborn jupyter
	```
4. Для расширенного ML (опционально):
	```bash
	pip install scikit-learn torch tensorflow keras
	```

## Структура проекта
- `osc_tools/` — основной модуль с инструментами для обработки осциллограмм
- `dataflow/` — обработка COMTRADE и вспомогательные скрипты
- `ML_model/` — обучение и хранение моделей
- `notebooks/` — jupyter-ноутбуки с примерами и анализом
- `tests/` — тесты
- `raw_data/` — исходные данные

---
Если при запуске возникнут ошибки, убедитесь, что установлены все необходимые библиотеки и активировано виртуальное окружение.
