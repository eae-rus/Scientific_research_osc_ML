# План Фазы 3: Новые архитектуры и библиотеки KAN (Modern KAN Architectures)

-> **Статус:** В работе  
> **Цель:** Исследование альтернативных реализаций KAN (EfficientKAN, FastKAN, ResKAN) для повышения стабильности обучения и точности без расширения набора данных. Фокус на архитектурных улучшениях.

---

## 🎯 Основные цели фазы

1.  **Библиотечный апгрейд (Library Hunt):** Переход с базовой реализации `pykan` (которая часто медленная и нестабильная) на современные оптимизированные библиотеки (`EfficientKAN`, `FastKAN`, `WavKAN`, `ChebyKAN`).
2.  **Архитектурная эволюция (ResKAN):** Проверка гипотезы пользователя: "попробовать как у ResNet проходные уровни для передачи градиента". Реализация Residual Connections внутри KAN-блоков.
3.  **Минимализм в данных:** Использование строго фиксированного лучшего пайплайна из Фазы 2.6 (`phase_polar` + `stride`) с модификацией для Фазы 3: угол только для 1-й гармоники. Отказ от широких экспериментов с данными.
4.  **Комплексность (CVKAN):** Поиск возможностей для работы с комплексными числами, но *только в рамках новых эффективных библиотек* (не самоцель, а приятный бонус).

---

## 📦 Фиксированные условия ("Константы эксперимента")

Чтобы результаты были сравнимы с Фазой 2.6:
- **Feature Mode:** `phase_polar_h1_angle` (модуль для всех гармоник, угол только для 1-й гармоники).
- **Sampling:** `stride` (Окно 200мс, уменьшенное до 20 точек).
- **Target:** `base_labels` (4 класса) — для скорости и четкости выводов.
- **Baseline:** Лучшая модель из Фазы 2.6 (вероятно, `ConvKAN` или `PhysicsKAN` на этих же настройках).

---

## 📅 Детальный план работ

### Этап 1: Разведка и интеграция библиотек (Library Integration)

**Задача:** Найти "Production-ready" реализации KAN.

**Срок:** ~1 неделя на анализ + 1 неделя на интеграцию.

1.  **Анализ репозиториев (с приоритетами):**
    *   [x] 🥇 `Blealtan/efficient-kan` — https://github.com/Blealtan/efficient-kan (B-Splines, оптимизировано по памяти, ~2.5k stars).
        - Подключён backend `efficient` в обёртке (см. [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py)).
    *   [x] 🥈 `ZiyaoLi/fast-kan` — https://github.com/ZiyaoLi/fast-kan (RBF kernels, заявлено 3.3x быстрее).
        - Подключён backend `fast` в обёртке (см. [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py)).
    *   [x] 🥉 `SynodicMonth/ChebyKAN` — https://github.com/SynodicMonth/ChebyKAN (Полиномы Чебышева, нет grid → нет проблем с экстраполяцией).
        - Подключён backend `cheby` в обёртке (см. [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py)).
    *   [x] `zavareh1/Wav-KAN` — https://github.com/zavareh1/Wav-KAN (Wavelet KAN, потенциально хорош для временных рядов).
        - Подключён backend `wav` в обёртке (см. [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py)).
    *   [x] `torch-wavelet-kan` — Может и копия верхнего... Но легко устанавливается как библиотека - установил вот.
        - Учтён как дополнительный источник wavelet-реализаций при дальнейших сравнениях.
    *   [ ] (опционально) `IvanDrokin/torch-conv-kan` — свёрточные KAN слои.
    
2.  **Создание оберток (Wrappers):**
    *   [x] Создан единый интерфейс в `osc_tools/ml/kan_conv/modern_wrappers.py` (см. [osc_tools/ml/kan_conv/modern_wrappers.py](osc_tools/ml/kan_conv/modern_wrappers.py)).
        - Реализован backend `baseline` и первичная интеграция `efficient` с безопасным fallback на baseline при несовместимости/отсутствии библиотеки.
3.  **Smoke Tests:**
    *   [x] Подготовлен benchmark smoke-контур для `PhysicsKANConditional` c backend `baseline|efficient` (см. [scripts/phase3_experiments/run_phase3_benchmark.py](scripts/phase3_experiments/run_phase3_benchmark.py)).
        - Скрипт измеряет время train-step, инференс-латентность, VRAM/RSS и стабильность loss на коротком прогоне.

4.  **Скрипт запуска:**
    *   [x] Создать `scripts/phase3_experiments/run_phase3_libraries.py` по аналогии с `run_phase2_6.py` (см. [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py)).
    *   [x] Использовать тот же `ExperimentRunner`, но с новыми моделями (пока минимальный сценарий через `run_single_experiment` для `PhysicsKANConditional`).
    *   [x] Добавить параметр backend (`--kan-backend`) для таргетного запуска на `PhysicsKANConditional` (см. [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py)).
    *   [x] Добавить отдельный benchmark-скрипт для сравнения backend-ов (см. [scripts/phase3_experiments/run_phase3_benchmark.py](scripts/phase3_experiments/run_phase3_benchmark.py)).
    *   [x] Объединить запуск train и benchmark в единую точку входа через `--mode` (см. [scripts/phase3_experiments/run_phase3_libraries.py](scripts/phase3_experiments/run_phase3_libraries.py)).

### Этап 0: Минимальный старт (фиксированная база)

Чтобы не повторять масштаб Фазы 2.5/2.6, старт Фазы 3 делаем с жёсткими ограничениями:
- Только модель `PhysicsKANConditional`.
- Только сложность `heavy`.
- Только режим признаков `phase_polar_h1_angle`.
- Только `stride` sampling.
- По умолчанию `num_harmonics=9` (амплитуды всех гармоник сохраняются), при этом угол оставляем только для 1-й гармоники.

**Таблица сравнения библиотек (заполнить по результатам):**
| Библиотека | Stars | Скорость (эпоха) | Память | Стабильность | PyTorch совместимость |
|------------|-------|------------------|--------|--------------|----------------------|
| pykan (текущая) | ~14k | baseline | baseline | ⚠️ скачки | ✅ |
| efficient-kan | ~2.5k | ? | ? | ? | ? |
| fast-kan | ~0.5k | ? | ? | ? | ? |
| ChebyKAN | ~0.6k | ? | ? | ? | ? |

### Этап 2: Архитектура ResKAN (Residual KAN)

**Гипотеза пользователя:** Добавление skip-connection (как в ResNet) поможет передавать градиент и позволит строить **глубокие** KAN сети (Deep KAN), которые раньше не учились.

1.  **Разработка `ResKANBlock`:**
    *   **Identity Mapping:** Реализовать прямой проброс сигнала: $y = Activation(KAN(x)) + x$.
    *   **Projection Shortcut:** Если размерность меняется (`in_channels != out_channels`), использовать линейную проекцию (1x1 Conv) в skip-connection.
    *   **Pre-Activation:** Проверить вариант $y = KAN(\text{Activation}(x)) + x$, который часто стабильнее.

2.  **Разработка `DeepResKAN`:**
    *   Модель из 5-7-10 блоков ResKAN. Проверка, не затухает ли градиент (gradient flow check).

### Этап 3: Эксперименты (Focused Experiments)

**Exp 3.1: Битва библиотек (Library Benchmark)**
*   **Цель:** Сравнить `SimpleKAN` (baseline) с `EfficientKAN` и `FastKAN`.
*   **Сравнение:** Время эпохи, потребление памяти, стабильность Loss (меньше ли "скачков").

**Exp 3.2: ResKAN vs PlainKAN (Depth Test)**
*   **Цель:** Проверить, дает ли ResNet-подобная структура выигрыш.
*   **Конфигурации:**
    *   `PlainKAN_Deep` (6 слоев, без skip).
    *   `ResKAN_Deep` (6 слоев, со skip).
*   **Ожидание:** `PlainKAN` скорее всего не сойдется или переобучится, `ResKAN` должен работать.

**Exp 3.3: (Опционально) CVKAN / Complex Architectures**
*   Если выбранная библиотека поддерживает комплексные веса (или легко адаптируется) — проверить это. Если нет — пропускаем, фокусируемся на вещественных моделях с поляными признаками.

---

### Этап 4: Вывод и выбор "Чемпиона"

1.  Выбрать одну лучшую реализацию (библиотека + архитектура) для Фазы 4.
2.  Обновить `osc_tools`, сделав эту реализацию дефолтной.

---

## ✅ Критерии завершения

- [ ] В коде есть поддержка минимум 2 альтернативных библиотек KAN.
- [ ] Реализован класс `ResKAN` (или аналог с skip-connections).
- [ ] Получен ответ: работают ли глубокие KAN лучше широких.
- [ ] Выбрана финальная архитектура для сегментации (Фаза 4).
- [ ] Обновлён `osc_tools/ml/models/kan.py` с новыми реализациями.

---

## 📌 Примечания и риски

- **Риск совместимости:** Некоторые библиотеки могут не поддерживать нужные версии PyTorch. Проверить перед глубокой интеграцией.
- **Риск переусложнения:** Не увлекаться количеством библиотек. Достаточно 2-3 рабочих вариантов.
- **PDR/РНМ:** По-прежнему отложено — нет датасета.
- **Связь с Фазой 4:** Выбранная архитектура должна легко адаптироваться под U-Net/TCN структуру (encoder-decoder).

