# Журнал работ Phase 5

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
