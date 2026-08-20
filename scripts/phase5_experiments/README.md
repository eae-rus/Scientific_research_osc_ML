# Сценарии Phase 5

Файлы остаются в одной папке: разнесение по подпапкам сломало бы
сложившиеся IDE/командные запуски и Python-импорты. Ниже — карта файлов.

## PDR: ручной pipeline

1. `run_pdr_dataset_study.py` — единственный длительный прогон пяти РНМ,
   shards, resume, progress/rate/ETA. Снача `SMOKE=True`, затем `False`.
2. `analyze_pdr_dataset_study.py` — основная статистика, signals, exact
   agreement, кластеры и PNG/CSV-диагностика. Обычно `MODE="all"`.
3. `review_pdr_analysis_results.py` — компактный воспроизводимый научный
   слой над готовыми CSV: current-bin/split/source profiles, bootstrap,
   duplicate sensitivity, source×SPP, сравнение record/time/point weighting,
   пояснение метрик, ECDF/heatmap/state-pattern figures и кандидаты устойчивых
   расхождений.

`review_pdr_analysis_results.py` не временный: он нужен для повтора
одинаковых научных таблиц после каждой версии разметки. Одноразовый
`build_pdr_unlabeled_overlay.py` удалён: он был нужен только для миграции v3.

Полная инструкция: `docs/phase_discription/PHASE_5_PDR_PIPELINE_GUIDE.md`.

## Подготовка данных

- `scan_open_ee_dataset.py`, `scan_french_dataset.py` — первичный аудит источников.
- `prepare_open_ee_shards.py`, `prepare_french_rte_npy.py` — подготовленные
  ленивые форматы.
- `build_phase5_splits.py` — strict split по хэшам целых осциллограмм.
- `build_real_ozz_exclusion.py` — слой исключений реальных ОЗЗ.

## SSL/pretrain и служебные сценарии

- `run_phase5_pretrain.py`, `eval_phase5_pretrain.py` — обучение и оценка backbone.
- `benchmark_phase5_storage.py` — оценка RAM/диска/скорости.
- `compare_phase5_smokes.py` — сравнение smoke-прогонов pretrain.
- `progress.py` — общий индикатор прогресса, не ручной entrypoint.

Старый `audit_pdr_signals.py` удалён: он проверял только provenance Open_EE
и дублировал более полные coverage/signal-проверки двух актуальных
PDR-сценариев.

Новый файл следует добавлять только если он воспроизводим и не дублирует один
из трёх PDR-entrypoint. Одноразовые migration/render/debug-сценарии удаляются
после применения.
