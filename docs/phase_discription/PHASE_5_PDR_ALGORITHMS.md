# Спецификация и архитектура алгоритмов РНМ (PDR) в Phase 5

Дата создания: **24.07.2026**  
Модуль: `osc_tools/pdr/`

## 1. Назначение и научные рамки

Реле направления мощности (РНМ, англ. Power Directional Relay — PDR) выступает основным прикладным измерительным органом для исследования возможностей Physical KAN-Transformer на реальных осциллограммах.

Основная задача нейросетевого органа: по 200 мс (10 периодов сети) спектральной предыстории токов и напряжений принимать детерминированное физически обоснованное решение о направлении мощности в текущей конечной точке.

---

## 2. Изоляция и структура алгоритмов

Для сохранения коммерческой тайны и поддержания открытости публичной части репозитория алгоритмы РНМ строго разделены по папкам вместе со своей математической документацией:

```
Scientific_research_osc_ML/
├── osc_tools/pdr/                             # Публичное ядро РНМ (в Git)
│   ├── base.py                                # Контракты PDRInputData, PDROutput, PDRDirection
│   ├── signal_analysis.py                     # Проверка полноты сигналов и расчёт I_B = -(I_A + I_C)
│   ├── registry.py                            # Реестр PDRRegistry и подгрузка плагинов
│   ├── placeholder.py                         # Безопасная заглушка PlaceholderPDRAlgorithm
│   ├── labeler.py                             # Генератор псевдоразметки по 10-периодному окну
│   ├── pdr_dataset.py                         # PyTorch Task Dataset
│   ├── pdr_trainer.py                         # Контур fine-tuning
│   │
│   └── public_algorithms/                     # Папка публичных открытых алгоритмов
│       ├── __init__.py                        # Реэкспорт открытых классов
│       ├── basic_phase.py                     # Фазный РНМ (PhasePDRAlgorithm)
│       ├── basic_pos_seq.py                   # РНМ прямой последовательности (PositiveSequencePDRAlgorithm)
│       ├── stubs.py                           # Точки расширения для алгоритмов сторонних БАВР
│       └── PUBLIC_PDR_ALGORITHMS_DESCRIPTION.md # Подробная математика открытых алгоритмов
│
└── private/                                   # ИСКЛЮЧЕНО ИЗ GIT (.gitignore)
    └── pdr_algorithms/                        # Закрытые адаптивные алгоритмы БАВР (mir/MIRAPS)
        ├── __init__.py
        ├── adaptive_pdr.py                    # Python-реализация адаптивного РНМ
        ├── test_adaptive_pdr.py               # Unit-тесты закрытого модуля
        └── ADAPTIVE_PDR_ALGORITHM_DESCRIPTION.md # Подробное описание адаптивного РНМ
```

---

## 3. Контракты данных и кодирование направлений

### Перечисление `PDRDirection`:
- `UNLABELED (-999)`: Неразмеченная область (первые 10 периодов разогрева / недостаток предыстории). Позволяет однозначно отличать отсутствие разметки от нулевых решений РНМ.
- `REVERSE (0)`: Обратное направление мощности / малый сигнал / блокировка (КЗ за спиной / запрет БАВР).
- `FORWARD (1)`: Прямое направление мощности (КЗ в зоне действия / разрешение БАВР).

---

## 4. Публичные открытые алгоритмы

Подробное математическое описание каждого публичного алгоритма с формулами и уставками приведено прямо в папке алгоритмов: [PUBLIC_PDR_ALGORITHMS_DESCRIPTION.md](file:///c:/Users/User/Desktop/Projects/Scientific_research_osc_ML/osc_tools/pdr/public_algorithms/PUBLIC_PDR_ALGORITHMS_DESCRIPTION.md).

### Краткий обзор:
1. **Базовый фазный алгоритм (`phase_pdr_basic`)**:
   - Реализован в `osc_tools/pdr/public_algorithms/basic_phase.py`.
   - Пофазный расчёт углов $\varphi_k = \arg(U_k) - \arg(I_k)$.
   - Логика «И» (AND): `FORWARD` выставляется только если все 3 фазы показывают прямое направление.
2. **Базовый алгоритм прямой последовательности (`pos_seq_pdr_basic`)**:
   - Реализован в `osc_tools/pdr/public_algorithms/basic_pos_seq.py`.
   - Вычисление симметричных составляющих $U_1, I_1$ и угла $\varphi_1 = \arg(U_1) - \arg(I_1)$.
3. **Заглушки алгоритмов сторонних БАВР**:
   - Реализованы в `osc_tools/pdr/public_algorithms/stubs.py`.

---

## 5. Закрытый адаптивный алгоритм (mir/MIRAPS)

Подробное математическое описание адаптивного алгоритма БАВР находится в закрытом файле [ADAPTIVE_PDR_ALGORITHM_DESCRIPTION.md](file:///c:/Users/User/Desktop/Projects/Scientific_research_osc_ML/private/pdr_algorithms/ADAPTIVE_PDR_ALGORITHM_DESCRIPTION.md).

### Краткий обзор:
- Реализован в `private/pdr_algorithms/adaptive_pdr.py`.
- Работает по умолчанию на прямой последовательности (`rnm_type = "positive"`).
- Содержит 3 вида адаптивностей: по углу нагрузки предыстории ($\varphi_{mch\_eff}$), по сужению зоны ($\alpha_{pos}, \alpha_{neg}$) и по модулю тока предыстории ($I_{abs\_thresh}$).
