# План Фазы 5: предобучение на реальных осциллограммах и дообучение под задачи

-> **Статус:** план к реализации  
**Цель:** продолжить Фазу 4/4.5 и обучить модернизированный Physical KAN-Transformer на большом наборе реальных осциллограмм. Сначала модель должна получить общее представление о формах токов/напряжений без разметки, затем использоваться как backbone для задач с разметкой: ДПОЗЗ/ОЗЗ, будущие РНМ-задачи и другие аварийные режимы.

---

## 1. Ключевая идея Фазы 5

Фаза 4 доказала рабочий контур:

- спектральное представление из сырых осциллограмм;
- 8-канальный физический контракт;
- SSL pretrain с реконструкцией/прогнозом;
- fine-tuning на ОЗЗ/ДПОЗЗ;
- lazy-загрузчики для тяжёлых CSV;
- ручной запуск через константы в `main` + CLI overrides;
- checkpoint/resume, логи, split-файлы и отчёты.

Фаза 5 должна не переписать это с нуля, а масштабировать:

1. **Старый `data/ml_datasets` не использовать в Phase 5 pretrain.** Он остаётся для фаз 2-4 и regression-проверок.
2. **Основной pretrain делать на реальных неразмеченных данных:**
   - `data/Open_EE_Dataset_v1_3_osc_CSV/` — российская сеть 6-35 кВ, уже первично нормализованная;
   - `data/digital-fault-recording-database/DATA_S.npz` — французские записи RTE, цельные осциллограммы аварий/событий.
3. **Fine-tuning делать task-specific:** текущий ДПОЗЗ/ОЗЗ, затем новые задачи. Simulated_OZZ остаётся размеченным источником для fine-tuning, а не главным источником SSL pretrain.
4. **Инфраструктура должна поддерживать новые датасеты:** не зашивать Open_EE/French как единственные варианты.
5. **Данные читать лениво:** 64 ГБ RAM много, но не достаточно для наивной загрузки всех CSV/NPZ/SimOZZ.

---

## 2. Технические инварианты

### 2.1. 8-канальный контракт

Все реальные и симулированные источники должны приводиться к единому виду:

1. `IA`
2. `IB`
3. `IC`
4. `IN`
5. `UA`
6. `UB`
7. `UC`
8. `UN`

Правила:

- если сигнал физически отсутствует, записывать `NaN`, а не `0`;
- `NaN` дальше обрабатывается существующей логикой `DataSanitizer`/missing mask;
- для Open_EE фазные напряжения брать с приоритетом `UA/UB/UC BB`, затем `UA/UB/UC CL`;
- если фазных напряжений нет, но есть линейные `UAB/UBC/UCA`, временно класть их в позиции `UA/UB/UC` как `AB/BC/CA` без восстановления фазных, с явным TODO;
- для French: 6 сигналов трактовать как 3 фазных напряжения и 3 фазных тока, `IN` и `UN` заполнять `NaN`;
- исходные датасеты не удалять и не портить, все преобразования писать рядом или в отдельную папку подготовленных данных.

### 2.2. Окна, stride и SPP

`SPP` = samples per period = число отсчётов на один период сети.

Примеры:

- Open_EE `unlabeled_50_1200.csv`: `SPP = 1200 / 50 = 24`;
- Open_EE `unlabeled_50_1600.csv`: `SPP = 32`;
- French `DATA_S.npz`: `SPP = 6400 / 50 = 128`;
- Simulated_OZZ: `SPP` около 769.

Инварианты:

- окно модели = 10 периодов;
- stride по умолчанию = 1/8 периода, потому что это ближе к текущей логике ОЗЗ;
- если `SPP / 8` не целое, сначала использовать `round(SPP / 8)` с логированием фактического шага, затем отдельно исследовать более аккуратные варианты;
- частота сети и частота дискретизации всегда должны быть частью метаданных окна.

### 2.3. Спектральные признаки

Сохранять идею Фазы 4:

- стандартные гармоники;
- низшие/длинные гармоники;
- симметричные составляющие там, где они физически корректны;
- missing mask для отсутствующих каналов;
- расчёт on-the-fly для ленивых датасетов.

Для низких частот дискретизации нужно отдельно проверить, какие гармоники реально доступны. Если гармоника выше допустимой по дискретизации или окно недостаточно информативно, лучше явно писать `NaN`/mask, а не выдумывать значение.

---

## 3. Источники данных

### 3.1. Open_EE_Dataset_v1_3_osc_CSV

Путь: `data/Open_EE_Dataset_v1_3_osc_CSV/`

Факты из текущей структуры:

- 18 CSV-файлов вида `unlabeled_{f_network}_{f_adc}.csv`;
- есть 50 Гц и 60 Гц;
- частоты дискретизации: 600, 800, 916, 1000, 1100, 1200, 1600, 1800, 2000, 2100, 2400, 4000, 4800, 6400, 8000, 1200/1920/3840 для 60 Гц;
- самые большие файлы: `unlabeled_50_1200.csv` и `unlabeled_50_1600.csv`;
- пример колонок: `sample,file_name,IA,IB,IC,IN,UA BB,UB BB,UC BB,UN BB,UAB BB,UBC BB,UCA BB`;
- данные уже поделены на номиналы ТТ/ТН с коэффициентами запаса. Повторно применять `norm_coef_all_v1.4.csv` нельзя.

Что важно проверить:

- все варианты колонок по 18 файлам;
- число уникальных `file_name` в каждом CSV;
- длины осциллограмм по `file_name`;
- наличие/отсутствие `IN`, `UN`, фазных и линейных напряжений;
- диапазоны значений как sanity-check нормировки;
- совпадение имён с `data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv` для исключения реальных ОЗЗ из `real_no_OZZ`.

### 3.2. digital-fault-recording-database

Путь: `data/digital-fault-recording-database/`

Главный файл для Phase 5: `DATA_S.npz`.

Факты из README:

- `DATA_S` содержит 12053 записи;
- форма `(12053, 6, 21000)`;
- сигналы: `v1, v2, v3, i1, i2, i3`;
- частота сети 50 Гц;
- частота дискретизации 6400 Гц;
- длительность 21000 / 6400 = 3.28125 с;
- номинальное напряжение 90 кВ;
- шаг квантования напряжения 18.310 В;
- шаг квантования тока 4.314 А;
- данные выбраны из реальных fault records по форме волны;
- разметки под наши задачи нет.

`DATA_u.npz` и `DATA_i.npz` пока не являются основным источником: это уже отобранные однопериодные транзиенты, полезные для диагностики/сравнения, но не для обучения на цельной осциллограмме.

Открытый вопрос: нормировка токов. Номинал ТТ неизвестен, поэтому в Phase 5 сначала собираем статистику, а затем исследователь выбирает рабочий делитель.

### 3.3. Simulated_OZZ_v1

Путь: `data/Simulated_OZZ_v1/`

Использование:

- не основной SSL pretrain;
- основной размеченный датасет для fine-tuning ДПОЗЗ/ОЗЗ;
- текущий lazy-загрузчик `SimOZZLazyDataset` и контур `scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py` считать базой для Phase 5 task-пайплайна.

### 3.4. real_OZZ и real_no_OZZ

Путь: `data/real_OZZ/`

Использование:

- `overvoltage_report_T1_with_com_v1.7.csv` задаёт список известных реальных ОЗЗ/подозрительных событий;
- для Phase 5 `real_no_OZZ` означает реальные осциллограммы Open_EE, за исключением записей, имя которых совпадает или содержится в списке из `overvoltage_report_T1_with_com_v1.7.csv`;
- текущие COMTRADE-файлы real_OZZ остаются для валидации/инференса/сравнения, как в Фазе 4.5.

---

## 4. Этап 0: защитить Фазу 4 перед изменениями

**Цель:** перед масштабированием убедиться, что Phase 4/4.5 не сломается.

Задачи:

- запустить доступный набор unit/integration тестов или хотя бы быстрый поднабор по ML/датасетам;
- отдельно проверить тесты вокруг:
  - `osc_tools/ml/augmented_dataset.py`;
  - `osc_tools/ml/ssl_dataset.py`;
  - `osc_tools/ml/losses.py`;
  - `osc_tools/ml/models/transformer.py`;
  - `osc_tools/ml/simulated_ozz_dataset.py`;
  - `scripts/phase4_experiments/run_phase4_pretrain.py`;
  - `scripts/phase4_experiments/run_phase4_finetune.py`;
  - `scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py`;
- добавить smoke-тесты, если их не хватает:
  - SSL forward/backward на синтетическом 8-канальном окне;
  - `compute_spectral_from_raw` при разных `SPP`;
  - `SpectralReconstructionLoss` с `NaN`/mask;
  - resume checkpoint для pretrain и fine-tune;
  - `SimOZZLazyDataset` на малом числе файлов;
- зафиксировать набор команд для ручной проверки перед крупными изменениями.

**Примечание:** Если не получится запустить агенту - то стоит сказать об этом, я запущу отдельно тесты в ноутбуке для тестирования. Стоит прописать инструкцию для запуска на всякий случай.

Выход:

- список реально покрытых компонентов;
- список пробелов;
- минимальный regression-набор, который запускается быстро и защищает Phase 4.

---

## 5. Этап 1: инвентаризация и статистика датасетов

**Цель:** собрать факты до проектирования загрузчиков и нормировки.

### 5.1. Open_EE scan

Скрипт: `scripts/phase5_experiments/scan_open_ee_dataset.py`

Функции:

- пройти по всем `unlabeled_*.csv`;
- считать заголовки без загрузки всего файла;
- определить `f_network`, `f_adc`, `SPP` из имени;
- посчитать размер файла, число строк, число уникальных `file_name`;
- посчитать длины осциллограмм по `file_name`;
- собрать список колонок и долю непустых значений по каждой колонке;
- собрать min/max/mean/std и квантили по каналам на сэмпле или чанками;
- проверить, что значения похожи на уже нормализованные;
- сохранить отчёт:
  - `data/Open_EE_Dataset_v1_3_osc_CSV/open_ee_scan.json`;
  - `reports/phase5/open_ee_scan.md`.

Важно: этот этап не должен читать гигантские CSV целиком в RAM.

### 5.2. French scan

Скрипт: `scripts/phase5_experiments/scan_french_dataset.py`

Функции:

- открыть `DATA_S.npz` через `np.load(..., mmap_mode='r')`, если это возможно для текущего NPZ;
- подтвердить ключ, shape, dtype;
- проверить несколько случайных записей;
- перевести значения в физические единицы через шаги квантования;
- посчитать статистику напряжений и токов по периодным RMS, а не по мгновенным максимумам.

Для RMS:

- окно = 1 период = 128 отсчётов;
- считать RMS по каждой фазе и записи;
- желательно собирать квантили 0.1, 0.2, ..., 0.9, 0.95, 0.99, max;
- отдельно по участкам до аварии, если можно эвристически выделить “спокойное начало”, иначе по всем периодам с пометкой ограничения;
- собрать статистику 1-й гармоники как дополнительный вариант, но базово ориентироваться на RMS за период.

Выход:

- `data/digital-fault-recording-database/french_scan.json`;
- `reports/phase5/french_normalization_notes.md`;
- рекомендации по делителю тока: не выбирать автоматически без просмотра статистики.

### 5.3. Исключение известных ОЗЗ из Open_EE

Скрипт: `scripts/phase5_experiments/build_real_ozz_exclusion.py`

Функции:

- прочитать `data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv`;
- извлечь базовые имена COMTRADE/осциллограмм;
- сопоставить с `file_name` из Open_EE;
- сформировать:
  - `data/phase5/real_ozz_exclusion.json`;
  - `data/phase5/open_ee_real_no_ozz_index.json`.

Нужна мягкая логика matching:

- точное совпадение;
- имя из отчёта является префиксом/частью `file_name` (Такой вариант наиболее вероятен, так как местами может указываться ещё номер секции, поэтому имена будут лишь частью. Т.е. имя comtrade файла скорее всего будет частью имени в сборнике `overvoltage_report_T1_with_com_v1.7.csv`);
- отчёт о неоднозначных совпадениях вручную просматривается.

---

## 6. Этап 2: формат хранения и индекс осциллограмм

**Цель:** сделать данные быстрыми для случайного чтения и не завязаться на один монолитный формат.

### 6.1. Не конвертировать всё сразу в один большой NPZ

Решение по умолчанию:

- Open_EE оставить исходными CSV как источник истины;
- подготовить шардированное представление для обучения;
- French оставить `DATA_S.npz` как источник истины;
- унификацию делать через adapters.

Причина: один гигантский NPZ может стать неудобным для перезаписи, кэширования, частичного пересчёта и восстановления после сбоя. Более практичны shards.

### 6.2. Open_EE shards

Скрипт: `scripts/phase5_experiments/prepare_open_ee_shards.py`

Варианты, которые нужно протестировать на малом поднаборе:

- один файл на одну осциллограмму;
- один shard на 100 осциллограмм;
- один shard на 500/1000 осциллограмм;
- `npz_compressed`;
- обычный `.npy`/`.npz` без сильной компрессии, если чтение окажется важнее размера;
- возможно zarr/parquet только если дадут явное преимущество и не усложнят проект.

Предпочтение на старте: shards по 100 осциллограмм. Это компромисс между миллионами мелких файлов и огромными монолитами. Но стоит корректно учесть это в файлах подгрузки рандомных примеров, чтобы не осуществлялась загрузка этого shard-а ради одного примера, и так кучу файлов открывать. Стоит сделать загрузку по несколько файлов из данного shard, но учесть частотность по примера (чтобы чаще брались из 1200Гц и 1600Гц примеры). В общем обдумать глубоко, как лучше бы сделать равновероятное взятие примеров по их частотности.

Каждый shard должен хранить:

- `signals`: массивы 8 каналов, по возможности `float32`;
- `file_names`;
- `lengths`;
- `f_network`;
- `f_adc`;
- `spp`;
- `channel_order`;
- `source_csv`;
- `normalized=true`;
- `normalization_note="Open_EE already divided by CT/PT nominal values and reserve factors"`.

Если длины осциллограмм разные, не паддить всё до максимума без необходимости. Лучше хранить список массивов или плоский массив + offsets.

### 6.3. French adapter

Файл: `osc_tools/ml/phase5_sources.py` или отдельный `osc_tools/ml/real_datasets.py`

Нужны классы:

- `FrenchRTEIndex`;
- `FrenchRTEDatasetSource`;
- функция нормировки в per-unit с параметрами:
  - `voltage_nominal_v=90000`;
  - `voltage_reserve=3`;
  - `current_nominal_a` задаётся пользователем после анализа;
  - `current_reserve=20`;
  - `quant_voltage=18.310`;
  - `quant_current=4.314`.

До выбора `current_nominal_a` источник может работать в режиме `physical_units` для статистики, но не должен незаметно попадать в общий pretrain как “нормализованный”.

### 6.4. Общий индекс

Файл: `data/phase5/datasets_registry.json`

Минимальная структура:

```json
{
  "version": 1,
  "sources": {
    "open_ee": {
      "kind": "open_ee_sharded",
      "path": "data/phase5/open_ee_shards",
      "default_weight": 0.6667,
      "normalized": true
    },
    "french_rte": {
      "kind": "french_rte_npz",
      "path": "data/digital-fault-recording-database/DATA_S.npz",
      "default_weight": 0.3333,
      "normalized": false
    }
  }
}
```

Дополнительно нужен индекс осциллограмм:

- `source`;
- `record_id`;
- `file_name`;
- `shard_path`;
- `offset` или `local_index`;
- `n_samples`;
- `f_network`;
- `f_adc`;
- `spp`;
- `channels_available`;
- `is_known_real_ozz`;
- `split_group`;
- `normalization_profile`.

---

## 7. Этап 3: унифицированный lazy multi-dataset

**Цель:** один PyTorch Dataset для SSL pretrain на нескольких источниках.

Файлы:

- `osc_tools/ml/dataset_registry.py`;
- `osc_tools/ml/phase5_sources.py`;
- `osc_tools/ml/lazy_multi_dataset.py`;
- тесты `tests/unit/test_phase5_dataset_registry.py`;
- тесты `tests/unit/test_phase5_lazy_multi_dataset.py`.

### 7.1. DatasetSource API

Нужен общий интерфейс:

```python
class DatasetSource:
    name: str
    channel_order: tuple[str, ...]

    def __len__(self) -> int: ...
    def get_metadata(self, idx: int) -> dict: ...
    def load_signal(self, idx: int) -> np.ndarray:
        """Return float32 array with shape (8, n_samples). Missing channels are NaN."""
```

Реализации:

- `OpenEEShardedSource`;
- `FrenchRTESource`;
- позже `ComtradeSource`, `SimOZZSource`, другие.

### 7.2. Выборка

Алгоритм `__getitem__` для SSL:

1. выбрать источник по глобальным весам (`open_ee=2/3`, `french=1/3` как стартовая настройка);
2. внутри источника выбрать группу `f_network/f_adc/SPP` пропорционально числу осциллограмм или по пользовательским весам;
3. выбрать осциллограмму;
4. выбрать случайный фрагмент 10 периодов + future-зоны, если они нужны;
5. рассчитать спектр on-the-fly;
6. применить маскирование SSL;
7. вернуть `X`, `target`, `mask`, `metadata`.

Важно: внутри одного batch желательно группировать одинаковый `SPP` или одинаковую итоговую длину спектральной последовательности. Иначе collate будет сложным и может резко увеличить VRAM. 

**Примечание:** И ещё комментарий: обработку стоит обдумать так, чтобы не открывать shard-ы ради одного "среза", ибо загрузка данных файлов может быть узким горлышком. Вероятно лучше будет сделать по 1 файлы хранение файлов, или обработку сразу нескольких разных примеров. Плюс надо бы задать параллельную обработку с обучением, чтобы не тормозить процессы по возможности. В общем, нужно будет обдумать этот момент для осмысления ускорения обработку без порчи случайно выборки.

### 7.3. Sampling weights

Нужны уровни весов:

- глобально по источникам: например Open_EE 2/3, French 1/3;
- внутри Open_EE по файлам `unlabeled_*`: пропорционально числу осциллограмм, не строк;
- внутри конкретного файла/группы: равномерно по осциллограммам или пропорционально числу доступных окон;
- опционально ограничение `max_windows_per_epoch`, чтобы эпоха не длилась бесконечно (это как раз важный факт, и эпоху предлагаю ограничивать количество примеров).

Стратегии:

- `custom`: пользователь задаёт веса (пока что планируется только на сами датасеты, т.е. речь про Open_EE и French, а на дообучении - ещё ДПОЗЗ и потом насыщение ТТ - этот датасет появится позже);
- `proportional_records`: по числу осциллограмм;
- `proportional_windows`: по числу окон;
- `balanced_sources`: одинаковый вес источников;
- `debug_small`: фиксированный малый поднабор.

### 7.4. Кэширование

Минимум:

- LRU-кэш на загруженные осциллограммы;
- ограничение по числу записей и по памяти;
- отключаемый кэш для тестов;
- логирование hit/miss на debug.

---

## 8. Этап 4: Phase 5 SSL pretrain

**Цель:** обучать модель без учителя на реальных данных.

Новый скрипт:

- `scripts/phase5_experiments/run_phase5_pretrain.py`

Он должен наследовать стиль Phase 4:

- ручной блок констант внизу файла;
- CLI overrides;
- `--smoke`;
- `--resume`;
- `--reset-optimizer`;
- `latest_checkpoint.pt`;
- `best_model.pt`;
- `config.json`;
- `training_log.jsonl`;
- экспорт кривых.

### 8.1. Pretext tasks

Базовый набор:

- masked spectral-temporal modeling: скрываем часть временных зон и восстанавливаем спектральные признаки;
- реконструкция текущего окна;
- прогноз будущих зон, если текущая архитектура стабильно это держит;
- опционально contrastive/consistency-задача между разными augmentations одной осциллограммы, но только после базовой стабильности.

Не надо сразу превращать Phase 5 в архитектурный зоопарк. Первое качество фазы — надёжная инфраструктура и воспроизводимое обучение.

### 8.2. Loss

База:

- оставить `SpectralReconstructionLoss`/`ComplexMSELoss` как старт;
- проверить работу при `NaN` каналах;
- логировать компоненты loss отдельно: current, future, masked, per-channel group.

Исследовать улучшения:

- Huber/SmoothL1 поверх комплексной ошибки для устойчивости к выбросам;
- амплитудно-фазовый loss с раздельными весами;
- относительная ошибка с нижним порогом шума, как в Phase 4;
- channel-balanced loss, чтобы отсутствующие/малые каналы не ломали масштаб;
- curriculum: сначала h1 и базовые гармоники, затем расширенный спектр.

### 8.3. LR scheduler

Проблема не в “косинусной функции потерь”, а в управлении скоростью обучения. В Phase 5 нужен выбор scheduler, который реагирует на процесс.

Поддержать:

- `cosine_warmup` как legacy baseline;
- `cosine_warm_restarts`;
- `reduce_on_plateau` по `val_loss`;
- `onecycle` для коротких прогонов;
- `warmup_plateau_decay`: warmup -> plateau monitoring -> decay;
- логирование фактического LR каждую эпоху.

Стартовая рекомендация:

- для длинного SSL: warmup + `ReduceLROnPlateau` или warmup + cosine restarts;
- для коротких smoke/ablation: `OneCycleLR`;
- early stopping по умолчанию выключить, потому что запуск ручной и долгий, но предупреждения о стагнации логировать.

### 8.4. Validation

Validation-набор должен быть стабильным:

- фиксированный seed;
- фиксированный список осциллограмм/окон;
- отдельно Open_EE-val и French-val;
- общая val-метрика и per-source метрики;
- сохранение нескольких реконструкций в отчёт/PNG для визуального контроля.

---

## 9. Этап 5: fine-tuning под задачи

**Цель:** использовать Phase 5 checkpoint как старт для задач.

### 9.1. Общий task-контур

Файлы:

- `scripts/phase5_experiments/run_phase5_finetune.py`;
- возможно `osc_tools/ml/task_registry.py`;
- task-specific подпапки в `scripts/phase5_experiments/tasks/`.

Поддержать:

- загрузку Phase 5 SSL checkpoint;
- замену SSL head на classification/regression head;
- разные LR для backbone и head;
- freeze/unfreeze режимы;
- resume;
- сохранение split;
- отчёты как в Phase 4/4.5.

### 9.2. Задача ДПОЗЗ/ОЗЗ

Стартовая реализация:

- взять `SimOZZLazyDataset` и текущий `run_phase4_finetune_sim_ozz.py` как основу;
- добавить загрузку Phase 5 pretrain checkpoint;
- сохранить current random-init режим как baseline;
- добавить `real_no_OZZ` из Open_EE через новый индекс исключений;
- не использовать `ml_datasets` как источник новых реальных no-OZZ для Phase 5.

Оценка:

- SimOZZ validation;
- real_OZZ confirmed/false_detection;
- unknown real records из текущего inference-контура;
- сравнение random init vs Phase 5 pretrain.

### 9.3. Будущие задачи

Заложить структуру, но не реализовывать преждевременно:

- Насыщение ТТ (если уже появится в разделе 4)
- РНМ-разметка из Этапа 8 Phase 4 будет отдельной задачей позже (тут может быть много доработок, поэтому приступим когда решу);
- другие аварии;
- кластеризация/поиск похожих событий;
- полуавтоматическая разметка French/Open_EE после появления устойчивых моделей.

---

## 10. Этап 6: оценка, отчёты и статистика моделей

**Цель:** не просто обучать, а сравнивать модели и понимать, что улучшилось.

Нужно:

- единый `experiments/phase5/...` layout;
- `model_registry.json` или расширение текущего генератора registry;
- сводка по каждому запуску:
  - датасеты и веса;
  - нормировка;
  - checkpoint source;
  - модель/размер;
  - scheduler;
  - train/val loss;
  - task metrics;
  - время эпохи;
  - peak memory, если доступно;
- графики:
  - loss/LR;
  - reconstruction examples;
  - per-source validation;
  - confusion/ROC/PR для fine-tuning;
  - сравнение Phase 4.5 baseline vs Phase 5 pretrained.

Отдельно полезно сохранить “паспорт” каждого SSL checkpoint:

- какие источники использованы;
- какие веса источников;
- какая нормировка French current;
- какие Open_EE shards;
- commit/hash/дата, если доступно;
- список конфигов.

---

## 11. Этап 7: документация и исследовательские заметки

Документы:

- обновлять этот план по мере выполнения;
- вести `docs/phase_discription/PHASE_5_WORK_LOG.md`;
- сохранять краткие отчёты:
  - `reports/phase5/open_ee_scan.md`;
  - `reports/phase5/french_normalization_notes.md`;
  - `reports/phase5/pretrain_report_*.md`;
  - `reports/phase5/finetune_ozz_report_*.md`.

В рабочем логе фиксировать:

- что было запущено;
- сколько заняло;
- какие файлы созданы;
- какие параметры оказались плохими;
- какие решения требуют ручного выбора исследователя.

---

## 12. Предлагаемая последовательность реализации

1. **Phase 4 safety pass**
   - прогнать/добавить минимальные тесты;
   - убедиться, что текущий SimOZZ и pretrain smoke живы.

2. **Dataset scans**
   - Open_EE scan;
   - French scan;
   - отчёт по нормировке French RMS;
   - исключения real_OZZ из Open_EE.

3. **Small prototype**
   - сделать Open_EE shards на малом поднаборе;
   - сделать French adapter;
   - проверить 8-канальный контракт;
   - проверить `compute_spectral_from_raw` на Open_EE и French.

4. **Registry + LazyMultiDataset**
   - общий индекс;
   - weighted sampling;
   - batch grouping по SPP;
   - тесты.

5. **Phase 5 pretrain smoke**
   - 1-2 источника;
   - маленький `max_windows_per_epoch`;
   - проверка resume;
   - визуализация реконструкции.

6. **Полный pretrain**
   - стартовые веса Open_EE/French = 2/3 и 1/3;
   - per-source validation;
   - сохранение checkpoint-паспорта.

7. **Fine-tuning ДПОЗЗ/ОЗЗ**
   - random init baseline;
   - Phase 5 pretrained backbone;
   - сравнение на SimOZZ и real_OZZ.

8. **Расширение на новые задачи**
   - только после устойчивого результата по ДПОЗЗ/ОЗЗ.

---

## 13. Ожидаемые новые файлы

| Файл | Назначение |
|------|------------|
| `docs/phase_discription/PHASE_5_PLAN.md` | текущий план |
| `docs/phase_discription/PHASE_5_WORK_LOG.md` | журнал выполнения |
| `scripts/phase5_experiments/scan_open_ee_dataset.py` | статистика Open_EE |
| `scripts/phase5_experiments/scan_french_dataset.py` | статистика French/RTE |
| `scripts/phase5_experiments/build_real_ozz_exclusion.py` | исключение известных ОЗЗ из real_no_OZZ |
| `scripts/phase5_experiments/prepare_open_ee_shards.py` | подготовка Open_EE shards |
| `scripts/phase5_experiments/run_phase5_pretrain.py` | SSL pretrain |
| `scripts/phase5_experiments/run_phase5_finetune.py` | общий fine-tuning |
| `osc_tools/ml/dataset_registry.py` | реестр источников |
| `osc_tools/ml/phase5_sources.py` | adapters Open_EE/French |
| `osc_tools/ml/lazy_multi_dataset.py` | общий lazy SSL dataset |
| `data/phase5/datasets_registry.json` | машинный реестр |
| `reports/phase5/*.md` | отчёты |
| `tests/unit/test_phase5_*.py` | unit-тесты Phase 5 |
| `tests/integration/test_phase5_pretrain_smoke.py` | smoke pretrain |

---

## 14. Главные риски

1. **Open_EE уже нормализован.** Повторная нормировка через `norm_coef_all_v1.4.csv` испортит масштаб.
2. **French current nominal неизвестен.** Нельзя молча выбрать делитель по максимуму КЗ. Сначала RMS/квантили, затем ручное решение.
3. **Разные SPP.** Нужно batch grouping или padding/mask, иначе обучение станет нестабильным и тяжёлым по VRAM.
4. **Огромные CSV.** Нельзя использовать pandas/polars так, чтобы файл целиком попадал в RAM.
5. **Слишком ранняя конвертация всего архива.** Сначала прототип shards и benchmark чтения.
6. **Смешение pretrain и fine-tuning источников.** SimOZZ не должен незаметно попасть в SSL как “реальные данные”.
7. **Неправильный real_no_OZZ.** Из Open_EE надо исключить известные ОЗЗ по отчёту, иначе normal-класс будет загрязнён.
8. **Слом Phase 4.** Все новые адаптеры должны добавляться рядом, без разрушения текущих скриптов.

---

## 15. Стартовые решения по умолчанию

- Open_EE/French weights для первого полного pretrain: `0.667 / 0.333`.
- Окно: 10 периодов.
- Stride: 1/8 периода.
- Open_EE storage: shards по 100 осциллограмм после benchmark (но это стоит обдумать).
- French storage: оставить `DATA_S.npz`, использовать adapter.
- French voltage normalization: `value * 18.310 / (90000 * 3)` (проверить).
- French current normalization: после RMS-отчёта, параметр задаётся явно.
- Missing channels: `NaN`.
- Scheduler для первого длинного SSL: warmup + `ReduceLROnPlateau` или warmup + cosine restarts; выбор зафиксировать после smoke.
- Early stopping: выключен, но логировать стагнацию.
- Main style: ручные константы + CLI как в Phase 4.