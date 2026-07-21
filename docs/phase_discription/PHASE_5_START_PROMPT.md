# Стартовая выжимка для продолжения Phase 5

Дата актуализации: **21.07.2026**. Этот документ — короткая точка входа для
следующего агента, а не замена плану и журналу.

## Что прочитать в самом начале

В указанном порядке прочитай:

1. `.github/agents-instructions.md` — постоянные правила проекта: чистота
   репозитория, ручные сценарии, тесты, запрет чтения `private/` без прямого
   запроса исследователя, необходимость вести план и журнал.
2. `docs/phase_discription/PHASE_5_PLAN.md` — последовательность этапов и
   критерии готовности.
3. `docs/phase_discription/PHASE_5_WORK_LOG.md` — актуальная история решений;
   верхняя запись важнее старых разделов «следующий шаг».
4. `docs/article/RESEARCH_CONTEXT_SUMMARY.md` — научный контекст фаз 2–4.

Углубляться в DOCX из `docs/article/` нужно только для конкретного вопроса.
Поздний код и versioned-контракты имеют приоритет над ранними архитектурными
гипотезами статей.

## Научная цель и границы

Phase 5 строит общий Physical KAN-Transformer backbone на реальных
осциллограммах без учителя. Главная последующая прикладная задача — РНМ/PDR;
ОЗЗ/ДПОЗЗ остаётся подключаемой задачей. Не смешивать общий SSL и будущую
PDR-разметку.

Для PDR сначала нужны аналитические органы, их статистика и решение
исследователя о teacher. Не придумывать формулы или уставки самостоятельно.
Закрытые алгоритмы в `private/` не читать и не раскрывать без прямой задачи.
До PDR по плану идёт общий task-контур (этап 6), затем PDR (этап 7).

## Неподвижные инварианты

- Единый порядок: `IA, IB, IC, IN, UA, UB, UC, UN`.
- Физически отсутствующий канал — `NaN`, не ноль; provenance различает
  `missing/measured/derived`.
- Open_EE уже первично нормирован: повторно применять
  `norm_coef_all_v1.4.csv` нельзя.
- Линейные напряжения всегда несут `voltage_basis='line'`; их нельзя молча
  считать фазными.
- Окна задаются в периодах сети; SPP, шаг и доступные гармоники versioned и
  записываются в metadata/checkpoint.
- Legacy Phase 4 `feature_contract_v1` и raw-angle checkpoint path должны
  сохранять совместимость. Новое поведение — только за явным version/flag.
- Не оставлять одноразовые скрипты, временные каталоги, `__pycache__` и пробные
  артефакты без исследовательской ценности.

## Реализованная инфраструктура Phase 5

- `osc_tools/ml/phase5_contracts.py`: 8-канальный и временной контракт,
  гармоники, snapshot/sequence positions.
- French/RTE: `DATA_S.npy` подготовлен для mmap из исходного NPZ;
  `(12053, 6, 21000)`, float64. Подтверждённая нормировка:
  `I / (300 A × 20)`, `U / (90 kV × 3)`.
- Open_EE: 44 773 осциллограммы в 448 uncompressed shards по 100 записей,
  `data/phase5/open_ee_shards/`; adapter/lazy LRU reader готовы.
- `dataset_registry.py`, `phase5_sources.py`, `lazy_multi_dataset.py`:
  Open_EE и French приводятся к общему contract; French заполняет `IN/UN` как
  `NaN`.
- `spectral_features.py`: feature contract v2.
  - **A**: 220 признаков, `phase-polar h1–h9 + low + h1 symmetric`.
  - **B**: 156 признаков, `symmetric-polar h1–h9 + low`.
  - causal FFT, `snapshot_2`, `snapshot_5`, `sequence_1_8`, Nyquist/missing
    mask, provenance, line-voltage branch.
- В `PhysicalStem` и `ComplexMultiheadAttention` добавлен
  `cyclic_angle_encoding=True`; legacy default — `False`.
- `use_provenance_embedding=True` добавляет модели token-level контекст
  measured/derived/missing; legacy default — `False`.
- `phase5_ssl.py`, `checkpoint_contracts.py`, `phase5_splits.py`: group-aware
  masked modeling, retry fully-missing Version B samples, passport признаков и
  immutable `research_strict` split.

## Split и фактическое покрытие данных

`data/phase5/research_strict_splits.json`, SHA-256:
`c8f0089f8cb09427df59f9fea10b26f5a5026b4b375ab65ced472f15aa4d88ff`.

| Источник | Train | Validation | Never-seen holdout |
|---|---:|---:|---:|
| Open_EE | 38 473 | 1 823 | 4 477 |
| French/RTE | 9 643 | 1 205 | 1 205 |

Open_EE split не дробит исходные CSV; поэтому validation не обязан быть ровно
10% по числу записей. Holdout не должен участвовать в SSL/scheduler selection
и сохраняется для честной downstream/transfer проверки.

## Что уже проверено

- CPU Phase 5 tests: 36 passed.
- Torch model tests в основной среде: 82 passed.
- Среда обучения: Python 3.13.5, torch 2.7.1+cu118, RTX 3060 Ti.
- A/B smoke и resume прошли; отчёт:
  `reports/phase5/pretrain_smoke_comparison.md`.
- Squared ComplexMSE давал выбросы train loss. `RobustComplexLoss` (complex
  Huber, `beta=0.1`) стал default для Phase 5.
- Version B временно выбрана как основной SSL contract: немного лучше общей и
  Open_EE validation, меньше размерность. A чуть лучше на French; окончательный
  выбор A/B возможен только после PDR probe, не по SSL loss.

## Завершённый основной SSL pretrain

Артефакты: `experiments/phase5/pretrain_b/`.

- Модель: Version B, `snapshot_5`, `d_model=64`, `num_heads=4`, `num_layers=4`,
  `d_ff=256`, cyclic angles и provenance embedding включены.
- Loss: complex Huber, `beta=0.1`; batch 32; 20 000 train windows/epoch;
  2 000 mixed validation windows/epoch; веса Open_EE/French = 2/3 и 1/3.
- Выполнены две детерминированно одинаковые fresh-волны по 50 эпох: в каждом
  запуске 1 000 000 случайно выбранных train окон. Это не строгий один проход
  по каждому record, но при таком объёме почти все train осциллограммы видятся
  многократно. Holdout не использовался.
- Лучший combined validation loss: **0.000466846** на эпохе 11.
  Лучший Open_EE: **0.000436304** на эпохе 11.
  Лучший French: **0.000507804** на эпохе 32.
- К эпохе 49 combined validation стабилизировался около 0.000569; LR был снижен
  до `5.859375e-07`. Для использования брать `best_model.pt`, а не
  `latest_checkpoint.pt`.
- Доступны `best_model.pt`, `latest_checkpoint.pt`, `config.json`, JSONL log,
  JSON/PNG curves. Loss — только SSL reconstruction proxy, не метрика РНМ.

Важный технический долг: fresh запуск с уже существующей output-папкой дописал
вторую волну в `training_log.jsonl` вместо явного отказа/архивации. До следующей
дорогой волны нужно исправить runner: при `resume=False` непустой output-dir
должен требовать нового пути либо явного archive/overwrite режима. Не удалять
существующие артефакты автоматически.

## Ближайший порядок действий

1. Проанализировать full pretrain: подтвердить passport, выделить лучший
   checkpoint, построить reconstruction примеры и per-feature-group метрики.
2. Исправить защиту output-dir/logs и добавить явный run ID до новых запусков.
3. Подготовить controlled scale-up: сначала smoke целевой более тяжёлой модели
   в новой папке, затем фиксированный budget-ablation против текущего baseline.
   Не запускать «очень большую» модель без этого сопоставления.
4. После выбора масштаба запустить следующую long-run волну на
   `research_strict`; `full_archive` разрешён только после сравнительного
   исследования и маркируется как transductive.
5. Реализовать общий task API (этап 6). Затем запросить у исследователя
   математику/уставки открытых PDR органов и переходить к PDR (этап 7).

Для тяжёлых запусков использовать
`scripts/phase5_experiments/run_phase5_pretrain.py`: ручной блок параметров,
`--smoke`, `--resume`, паспорта checkpoint, progress/ETA и per-source validation
уже реализованы. Длительные GPU-прогоны запускает исследователь или агент только
после явного согласования ресурсов; после прерывания используется `RESUME=True`.

## Манера работы следующего агента

- Сначала inspect/test, затем локальная правка, затем пропорциональная проверка.
- Обновлять `PHASE_5_PLAN.md` и `PHASE_5_WORK_LOG.md` после значимых этапов.
- Не менять порядок/смысл признаков, checkpoint и split молча.
- Отделять факт эксперимента от интерпретации и гипотезы.
- Не останавливаться ради уточнения, если безопасный следующий шаг очевиден;
  спрашивать исследователя только при выборе teacher/математики PDR, крупных
  ресурсных затратах или изменении научного протокола.
