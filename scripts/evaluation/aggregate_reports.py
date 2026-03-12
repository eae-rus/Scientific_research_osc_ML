"""
Агрегация отчётов экспериментов.

Тонкий оркестратор: вся логика вынесена в подмодули ``_core/``.
Этот файл содержит только:
- реэкспорт публичных символов (обратная совместимость),
- функцию ``aggregate_reports()`` (основной цикл),
- ``run_physics_baseline_if_enabled()``,
- CLI-блок ``__main__``.
"""
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional, Tuple
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import re
import logging

# Добавляем корень проекта в путь импорта
ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR_PROJECT))

# ============================================================
# Реэкспорт из _core для обратной совместимости
# ============================================================
from scripts.evaluation._core.constants import (
    ENGINEERING_CLASS_MAP,
    ENGINEERING_CLASS_MAP_OZZ,
    OZZ_MULTILABEL_TRUE_COLS,
    OZZ_MULTILABEL_PRED_COLS,
    OZZ_MULTILABEL_ALL_COLS,
    get_engineering_class_map,
    PLOT_SWITCHES_DEFAULT,
    ENGINEERING_PLOTS_SUBDIR_DEFAULT,
    MAX_MODELS_FOR_COMBINED_DEFAULT,
    PREDICTION_FILE_PATTERNS,
    FULL_EVAL_SPLIT_DEFAULT,
)

from scripts.evaluation._core.cache_and_data import (
    load_existing_summary,
    needs_full_eval_recalc,
    needs_benchmark_recalc,
    get_existing_row,
    load_metrics,
    load_history_as_metrics,
    load_config,
)

from scripts.evaluation._core.config_resolvers import (
    parse_experiment_info,
    extract_base_exp_id,
    _coerce_per_class_f1_values,
)

from scripts.evaluation._core.model_utils import (
    _create_model_from_config,
    _normalize_state_dict_keys,
    _load_state_dict_safe,
    _get_eval_batch_size,
    benchmark_model_cpu,
    set_eval_logger,
    get_eval_logger,
)

from scripts.evaluation._core.predictions import (
    _is_multilabel_pred_df,
    _ensure_multilabel_ozz,
    _normalize_prediction_columns,
    _load_prediction_file_safe,
    load_best_final_predictions,
    _compute_f1_for_pred_df,
    select_best_final_by_test_f1,
    add_best_final_selection_columns,
    _save_predictions_csv,
)

from scripts.evaluation._core.full_evaluation import (
    _test_data_cache,
    _get_test_data_cached,
    _evaluate_with_batch_fallback,
    _evaluate_hierarchical_with_batch_fallback,
    evaluate_model_full,
    evaluate_full_test_dataset,
)

from scripts.evaluation._core.engineering_stats import (
    build_engineering_stats,
    build_engineering_stats_multilabel,
    choose_selected_predictions_per_experiment,
    collapse_selected_to_model_level,
)

from scripts.evaluation._core.engineering_plots import (
    plot_engineering_bidirectional_bars,
    build_error_severity_matrix,
    plot_custom_confusion_matrix,
    plot_multilabel_confusion_matrices,
    plot_engineering_bars_combined,
    generate_engineering_plot_pack,
)

from scripts.evaluation._core.report_visualizer import (
    plot_learning_curves,
    plot_comparison,
    combine_training_histories,
    ReportVisualizer,
)

# Импорт расширенного визуализатора
try:
    from scripts.visualization.advanced_plots import AdvancedVisualizer
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    print("[Warning] advanced_plots.py не найден - расширенная визуализация недоступна")

# Глобальный логгер (обёрнут через set_eval_logger / get_eval_logger в _core.model_utils)
_eval_logger: Optional[logging.Logger] = None

def aggregate_reports(
    root_dir: str, 
    output_dir: str = None, 
    plot: bool = False, 
    benchmark: bool = False, 
    full_eval: bool = False, 
    data_dir: str = None, 
    lang: str = 'ru',
    advanced_plots: bool = False,
    plot_switches: Optional[Dict[str, bool]] = None,
    engineering_subdir: str = ENGINEERING_PLOTS_SUBDIR_DEFAULT,
    max_models_for_combined: int = MAX_MODELS_FOR_COMBINED_DEFAULT,
    extra_experiment_roots: Optional[List[str]] = None,
    full_eval_split: str = FULL_EVAL_SPLIT_DEFAULT,
):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_dir: Опциональный путь к папке для сохранения всех отчетов и графиков.
        plot: Генерировать ли графики обучения для каждого эксперимента.
        benchmark: Выполнять ли глубокий бенчмарк на CPU.
        full_eval: Выполнять ли полную оценку на тестовом датасете (GPU, все точки).
        data_dir: Путь к директории с датасетами (для full_eval).
        lang: Язык графиков ('ru' или 'en').
        advanced_plots: Генерировать ли расширенные графики (Парето, heatmaps и др.).
        plot_switches: Словарь включения/выключения инженерных графиков.
        engineering_subdir: Подпапка в report для инженерных графиков.
        max_models_for_combined: Порог количества моделей для единого общего графика.
        full_eval_split: На каком split выполнять full_eval ('test' или 'train').
    """
    root_path = Path(root_dir)
    full_eval_split = str(full_eval_split or FULL_EVAL_SPLIT_DEFAULT).strip().lower()
    if full_eval_split not in ('test', 'train'):
        raise ValueError(f"Неподдерживаемый full_eval_split: {full_eval_split}. Ожидается 'test' или 'train'.")

    experiments = []
    all_histories = {}
    effective_plot_switches = PLOT_SWITCHES_DEFAULT.copy()
    if plot_switches:
        effective_plot_switches.update(plot_switches)

    # Здесь храним найденные предсказания для выбора Best/Final по тесту.
    exp_predictions: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
    
    # --- Подготовка путей вывода ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        figures_path = out_path / "figures"
        csv_path = out_path / "summary_report.csv"
        history_path = out_path / "combined_metrics_history.txt"
        error_log_path = out_path / "processing_errors.log"
        eval_log_path = out_path / "eval_log.txt"
        for p in (error_log_path, eval_log_path):
            if p.exists():
                try:
                    p.unlink()
                except PermissionError:
                    print(f"[!] Не удалось очистить {p} (файл занят). Продолжаем без очистки.")
    else:
        out_path = None
        figures_path = root_path / "figures"
        csv_path = None
        history_path = None
        error_log_path = None
        eval_log_path = None

    # Инициализация логгера для оценки
    global _eval_logger
    if eval_log_path:
        _eval_logger = logging.getLogger('eval_logger')
        _eval_logger.setLevel(logging.DEBUG)
        _eval_logger.handlers.clear()
        fh = logging.FileHandler(eval_log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        _eval_logger.addHandler(fh)
        print(f"[Log] Оценка будет логироваться в: {eval_log_path}")
    else:
        _eval_logger = None
    set_eval_logger(_eval_logger)

    # Путь к датасетам по умолчанию
    if data_dir is None:
        data_dir = str(ROOT_DIR_PROJECT / 'data' / 'ml_datasets')

    # Загружаем существующий отчёт для пропуска уже посчитанных моделей
    existing_df = load_existing_summary(csv_path) if csv_path else None
    
    # Счётчики для статистики пропусков
    skip_stats = {
        'full_eval_skipped': 0, 'full_eval_recalc': 0,
        'benchmark_skipped': 0, 'benchmark_recalc': 0,
        'plots_skipped': 0, 'plots_generated': 0,
    }

    # --- Сбор корневых директорий ---
    all_roots: List[Path] = [root_path]
    if extra_experiment_roots:
        for extra_root in extra_experiment_roots:
            if not extra_root:
                continue
            extra_path = Path(extra_root)
            if extra_path.exists():
                all_roots.append(extra_path)
            else:
                print(f"[!] Дополнительная директория не найдена и будет пропущена: {extra_path}")
    all_roots = list(dict.fromkeys(all_roots))
    roots_str = ", ".join(str(p) for p in all_roots)
    print(f"Сканирование: {roots_str} (Язык: {lang}, full_eval_split: {full_eval_split})...")
    
    # Собираем все эксперименты
    all_exp_dirs_set = set()
    for scan_root in all_roots:
        for metrics_file in scan_root.rglob("metrics.jsonl"):
            all_exp_dirs_set.add(metrics_file.parent)
        for history_file in scan_root.rglob("history.json"):
            if (history_file.parent / "config.json").exists():
                all_exp_dirs_set.add(history_file.parent)
    all_exp_dirs = sorted(all_exp_dirs_set)
    print(f"Найдено {len(all_exp_dirs)} экспериментов")

    # --- Основной цикл по экспериментам ---
    for exp_idx, exp_dir in enumerate(tqdm(all_exp_dirs, desc="Сбор логов")):
        try:
            metrics_file = exp_dir / "metrics.jsonl"
            history_file = exp_dir / "history.json"
            config_file = exp_dir / "config.json"
            
            metrics = load_metrics(metrics_file)
            if not metrics:
                metrics = load_history_as_metrics(history_file)
            config = load_config(config_file) if config_file.exists() else {}

            exp_predictions[exp_dir.name] = load_best_final_predictions(exp_dir, eval_split=full_eval_split)
            
            if not metrics:
                continue
                
            all_histories[exp_dir.name] = metrics
            info = parse_experiment_info(exp_dir.name)
            existing_row = get_existing_row(existing_df, exp_dir.name)

            current_target_level = info.get("target_level", "base")
            if existing_row is not None and 'TargetLevel' in existing_row.index:
                current_target_level = existing_row.get('TargetLevel', current_target_level)

            # Генерация графиков (пропускаем если файл уже существует)
            if plot:
                plot_path = exp_dir / "learning_curves.png"
                if plot_path.exists():
                    skip_stats['plots_skipped'] += 1
                else:
                    plot_learning_curves(metrics, plot_path)
                    skip_stats['plots_generated'] += 1

            # Лучшая эпоха (min val_loss)
            if any('val_loss' in m for m in metrics):
                best_epoch = min(metrics, key=lambda x: x.get('val_loss', 999.0))
            else:
                best_epoch = metrics[-1]

            num_params = config.get('model_info', {}).get('num_params', best_epoch.get('num_params', 0))
            cpu_inf = best_epoch.get('cpu_inf_time_ms', 0.0)
            
            # Benchmark
            if benchmark:
                if needs_benchmark_recalc(existing_row):
                    cpu_inf = benchmark_model_cpu(exp_dir, config, iterations=1000)
                    skip_stats['benchmark_recalc'] += 1
                else:
                    cpu_inf = existing_row['CPU Inf (ms)']
                    skip_stats['benchmark_skipped'] += 1

            # Формируем данные (приоритет кэшированной строке)
            base_fields = {
                "ExpID": info["exp_id"], "Experiment": exp_dir.name,
                "Model": info["model_family"], "Complexity": info["complexity"],
                "Features": info["feature_mode"], "Sampling": info["sampling"],
                "TargetLevel": info["target_level"], "Balancing": info["balancing"],
                "Aug": info["is_aug"], "arch_type": info["arch_type"],
            }
            if existing_row is not None:
                for k in ("ExpID", "Model", "Complexity", "Features", "Sampling",
                          "TargetLevel", "Balancing", "Aug", "arch_type"):
                    base_fields[k] = existing_row.get(k, base_fields[k])

            exp_data = {
                **base_fields,
                "Val F1": best_epoch.get('val_f1', 0.0),
                "Val Loss": best_epoch.get('val_loss', 0.0),
                "Val Acc": best_epoch.get('val_acc', 0.0),
                "CPU Inf (ms)": cpu_inf,
                "Params": num_params,
                "Epochs": len(metrics),
                "Best Epoch": best_epoch.get('epoch'),
                "Path": exp_dir.name,
            }
            
            if 'data' in config:
                exp_data['Window'] = config['data'].get('window_size')
                exp_data['Batch'] = config['data'].get('batch_size')

            # Per-class F1 из history
            per_class_vals = _coerce_per_class_f1_values(best_epoch.get('val_f1_per_class', {}))
            for idx_cls, cls_f1 in enumerate(per_class_vals):
                exp_data[f'Class_{idx_cls}_F1'] = cls_f1

            # Полная оценка на тестовом датасете (GPU)
            if full_eval:
                require_hier = current_target_level in ('full', 'full_by_levels')
                model_name_cfg = str(config.get('model', {}).get('name', ''))
                is_physics_baseline = (model_name_cfg == 'PhysicsBaseline')

                if is_physics_baseline:
                    # Для формульной модели: чекпоинтов нет, метрики уже записаны
                    row_split = str(existing_row.get('Full Eval Split', 'test')).strip().lower() if existing_row is not None else ''
                    if existing_row is not None and row_split == full_eval_split:
                        for k in ('Full Best Acc', 'Full Best F1', 'Full Final Acc',
                                  'Full Final F1', 'Full Best ROC-AUC', 'Full Final ROC-AUC',
                                  'Full Eval Time (s)'):
                            exp_data[k] = existing_row.get(k, 0.0)
                        exp_data['Full Eval Samples'] = existing_row.get('Full Eval Samples', 0)
                        exp_data['Num Classes'] = existing_row.get(
                            'Num Classes',
                            config.get('model', {}).get('params', {}).get('num_classes', 0)
                        )
                    else:
                        exp_data['Full Best Acc'] = float(best_epoch.get('val_acc', 0.0))
                        exp_data['Full Best F1'] = float(best_epoch.get('val_f1', 0.0))
                        exp_data['Full Final Acc'] = float(best_epoch.get('val_acc', 0.0))
                        exp_data['Full Final F1'] = float(best_epoch.get('val_f1', 0.0))
                        exp_data['Full Best ROC-AUC'] = 0.0
                        exp_data['Full Final ROC-AUC'] = 0.0
                        exp_data['Full Eval Time (s)'] = 0.0
                        exp_data['Full Eval Samples'] = 0
                        exp_data['Num Classes'] = config.get('model', {}).get('params', {}).get('num_classes', 0)
                    exp_data['Full Eval Split'] = full_eval_split
                    skip_stats['full_eval_skipped'] += 1

                elif needs_full_eval_recalc(existing_row, require_hierarchical=require_hier, eval_split=full_eval_split):
                    full_metrics = evaluate_full_test_dataset(exp_dir, config, data_dir, eval_split=full_eval_split)
                    exp_data['Full Best Acc'] = full_metrics['full_best_acc']
                    exp_data['Full Best F1'] = full_metrics['full_best_f1']
                    exp_data['Full Final Acc'] = full_metrics['full_final_acc']
                    exp_data['Full Final F1'] = full_metrics['full_final_f1']
                    exp_data['Full Best ROC-AUC'] = full_metrics.get('full_best_roc_auc', 0.0)
                    exp_data['Full Final ROC-AUC'] = full_metrics.get('full_final_roc_auc', 0.0)
                    exp_data['Full Eval Time (s)'] = full_metrics['full_eval_time_s']
                    exp_data['Full Eval Samples'] = full_metrics['full_eval_samples']
                    exp_data['Full Eval Split'] = full_eval_split
                    exp_data['Num Classes'] = full_metrics.get('num_classes', 0)
                    exp_data['Full Per Class F1 Count'] = full_metrics.get('full_per_class_f1_count', 0)
                    exp_data['Full Per Class Support Count'] = full_metrics.get('full_per_class_support_count', 0)

                    if require_hier:
                        for suffix in ('Best', 'Final'):
                            for metric in ('F1', 'Acc', 'Exact'):
                                key = f'Hier Base {metric} ({suffix})'
                                exp_data[key] = full_metrics.get(
                                    f'hier_base_{metric.lower()}_{suffix.lower()}', 0.0
                                )
                        for map_key in ('full_best_per_class_f1_map', 'full_final_per_class_f1_map',
                                        'full_best_per_class_support_map', 'full_final_per_class_support_map'):
                            prefix_map = {
                                'full_best_per_class_f1_map': 'Full F1 {} (Best)',
                                'full_final_per_class_f1_map': 'Full F1 {} (Final)',
                                'full_best_per_class_support_map': 'Full Support {} (Best)',
                                'full_final_per_class_support_map': 'Full Support {} (Final)',
                            }
                            for cls_name, val in full_metrics.get(map_key, {}).items():
                                exp_data[prefix_map[map_key].format(cls_name)] = val

                    # Per-class F1 для radar charts
                    for idx_cls, val in enumerate(full_metrics.get('full_best_per_class_f1', [])):
                        exp_data[f'Class_{idx_cls}_F1'] = val
                    for idx_cls, val in enumerate(full_metrics.get('full_best_per_class_roc_auc', [])):
                        exp_data[f'Class_{idx_cls}_ROC-AUC'] = val
                    
                    skip_stats['full_eval_recalc'] += 1

                else:
                    # Используем существующие данные из кэша
                    for k in ('Full Best Acc', 'Full Best F1', 'Full Final Acc',
                              'Full Final F1', 'Full Best ROC-AUC', 'Full Final ROC-AUC',
                              'Full Eval Time (s)'):
                        exp_data[k] = existing_row.get(k, 0.0)
                    exp_data['Full Eval Samples'] = existing_row.get('Full Eval Samples', 0)
                    exp_data['Full Eval Split'] = full_eval_split
                    exp_data['Num Classes'] = existing_row.get('Num Classes', 0)
                    exp_data['Full Per Class F1 Count'] = existing_row.get('Full Per Class F1 Count', 0)
                    exp_data['Full Per Class Support Count'] = existing_row.get('Full Per Class Support Count', 0)

                    if require_hier:
                        for suffix in ('Best', 'Final'):
                            for metric in ('F1', 'Acc', 'Exact'):
                                key = f'Hier Base {metric} ({suffix})'
                                exp_data[key] = existing_row.get(key, 0.0)
                        for col_name in existing_row.index:
                            if col_name.startswith('Full F1 ') or col_name.startswith('Full Support '):
                                exp_data[col_name] = existing_row.get(col_name, 0)
                    
                    for col_name in existing_row.index:
                        if re.match(r'^Class_\d+_F1$', str(col_name)) or re.match(r'^Class_\d+_ROC-AUC$', str(col_name)):
                            exp_data[col_name] = existing_row.get(col_name, None)
                    
                    skip_stats['full_eval_skipped'] += 1

            experiments.append(exp_data)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке опыта {exp_dir.name}: {str(e)}"
            print(f"\n[!] {error_msg}")
            if output_dir:
                try:
                    error_log = Path(output_dir) / "processing_errors.log"
                    with open(error_log, 'a', encoding='utf-8') as f:
                        f.write(f"=== {exp_dir.name} ===\n{error_msg}\n")
                        import traceback
                        f.write(traceback.format_exc())
                        f.write("-" * 50 + "\n\n")
                except Exception:
                    pass
            continue
    
    if not experiments:
        print("Эксперименты не найдены.")
        return

    df = pd.DataFrame(experiments).fillna(0)
    df = add_best_final_selection_columns(df)

    # Перезагружаем предсказания (CSV могли появиться после full_eval)
    for exp_dir in all_exp_dirs:
        fresh = load_best_final_predictions(exp_dir, eval_split=full_eval_split)
        if fresh.get('best') is not None or fresh.get('final') is not None:
            exp_predictions[exp_dir.name] = fresh

    # Выбор Best/Final по фактическому F1
    prediction_selection_df, selected_by_experiment = choose_selected_predictions_per_experiment(exp_predictions)
    if not prediction_selection_df.empty:
        df = df.merge(prediction_selection_df, how='left', on='Experiment', suffixes=('', '_pred'))
        if 'Selected Weights_pred' in df.columns:
            df['Selected Weights'] = df['Selected Weights_pred'].where(
                df['Selected Weights_pred'].notna(), df.get('Selected Weights', 'N/A'))
            df = df.drop(columns=['Selected Weights_pred'])
        if 'Selected Test F1_pred' in df.columns:
            df['Selected Test F1'] = df['Selected Test F1_pred'].where(
                df['Selected Test F1_pred'].notna(), df.get('Selected Test F1', np.nan))
            df = df.drop(columns=['Selected Test F1_pred'])

    # Генерация инженерных графиков
    if out_path and selected_by_experiment:
        selected_by_model = collapse_selected_to_model_level(
            selected_by_experiment=selected_by_experiment,
            selection_df=prediction_selection_df,
            summary_df=df,
        )
        if selected_by_model:
            eng_dir = out_path / engineering_subdir
            print(f"\n=== Генерация инженерных графиков ({len(selected_by_model)} моделей) -> {eng_dir} ===")
            generate_engineering_plot_pack(
                selected_by_model=selected_by_model,
                output_dir=eng_dir,
                plot_switches=effective_plot_switches,
                max_models_for_combined=max_models_for_combined,
            )
        else:
            print("[!] Не удалось сопоставить предсказания с моделями для инженерных графиков.")
    elif out_path:
        print(f"[!] CSV-файлы предсказаний ({full_eval_split}_predictions_best/final.csv) не найдены.")
        print("    Запустите с --full-eval чтобы пересчитать и создать файлы предсказаний.")

    # --- ВИЗУАЛИЗАЦИЯ (Report Engine 2.0) ---
    if plot:
        viz = ReportVisualizer(figures_path, lang=lang)
        
        group_cols = [
            'Complexity', 'Features', 'Sampling', 'TargetLevel',
            'Balancing', 'Aug', 'arch_type'
        ]
        grouped = df.groupby(group_cols, dropna=False)
        for idx, (group_keys, group_df) in enumerate(tqdm(grouped, desc=viz.t['groups_desc'])):
            base_ids = sorted({extract_base_exp_id(str(x)) for x in group_df['Experiment'].tolist()})
            base_ids = [x for x in base_ids if x != "Unknown"]

            if len(base_ids) == 1:
                group_label = base_ids[0]
                file_prefix = f"exp_{base_ids[0]}"
            elif len(base_ids) > 1:
                group_label = ", ".join(base_ids)
                file_prefix = f"exp_multi_{idx:03d}"
            else:
                group_label = f"Group {idx:03d}"
                file_prefix = f"group_{idx:03d}"

            params_suffix = "_".join([str(k) for k in group_keys])
            params_suffix = re.sub(r'[^a-zA-Z0-9._-]+', '_', params_suffix)[:80]
            group_file_id = f"{file_prefix}_{params_suffix}" if params_suffix else file_prefix

            group_plot_path = figures_path / "groups" / f"{group_file_id}_comparison_{lang}.png"
            if group_plot_path.exists():
                skip_stats['plots_skipped'] += 1
                continue

            exp_folders = group_df['Path'].tolist()
            group_histories = {f: all_histories[f] for f in exp_folders if f in all_histories}
            viz.plot_group_curves(group_label, group_df, group_histories, group_file_id)
            skip_stats['plots_generated'] += 1
            
        # Парето (ВСЕГДА обновляем)
        viz.plot_pareto(df)

    # Сортировка
    if 'Selected Test F1' in df.columns:
        df = df.sort_values(['ExpID', 'Selected Test F1'], ascending=[True, False])
    elif full_eval and 'Full Best F1' in df.columns:
        df = df.sort_values(['ExpID', 'Full Best F1'], ascending=[True, False])
    else:
        df = df.sort_values(['ExpID', 'Val F1'], ascending=[True, False])

    # Отобразить топ-результаты
    print("\nАгрегированный отчет (Топ-20):")
    cols_to_show = ['ExpID', 'Model', 'Complexity', 'Val F1', 'CPU Inf (ms)']
    if 'Selected Weights' in df.columns:
        cols_to_show.extend(['Selected Weights', 'Selected Test F1'])
    if full_eval:
        cols_to_show.extend(['Full Best F1', 'Full Best ROC-AUC'])
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].head(20).to_markdown(index=False, floatfmt=".4f"))

    # Сохранение файлов
    if out_path:
        df.to_csv(csv_path, index=False)
        print(f"\nCSV-отчет сохранен: {csv_path}")

        if not prediction_selection_df.empty:
            pred_select_path = out_path / 'selected_best_final_by_predictions.csv'
            prediction_selection_df.to_csv(pred_select_path, index=False)
            print(f"Таблица выбора Best/Final по предсказаниям сохранена: {pred_select_path}")
        
        combine_training_histories(root_path, history_path)
        print(f"История обучения объединена: {history_path}")

    if plot:
        plot_comparison(all_histories, out_path if out_path else root_path)
    
    # --- РАСШИРЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    if advanced_plots and ADVANCED_VIZ_AVAILABLE and out_path:
        print("\n=== Генерация расширенных графиков ===")
        advanced_figures_path = out_path / "figures_advanced"
        adv_viz = AdvancedVisualizer(advanced_figures_path, lang=lang)
        adv_viz.generate_all_plots(df, histories=all_histories)
    elif advanced_plots and not ADVANCED_VIZ_AVAILABLE:
        print("[!] Расширенная визуализация недоступна (модуль не найден)")
    
    # Статистика оптимизации
    print(f"\n=== Статистика оптимизации ===")
    if full_eval:
        print(f"  Full Eval: пересчитано {skip_stats['full_eval_recalc']}, пропущено {skip_stats['full_eval_skipped']}")
    if benchmark:
        print(f"  Benchmark: пересчитано {skip_stats['benchmark_recalc']}, пропущено {skip_stats['benchmark_skipped']}")
    if plot:
        print(f"  Графики: сгенерировано {skip_stats['plots_generated']}, пропущено {skip_stats['plots_skipped']}")


def run_physics_baseline_if_enabled(enabled: bool, eval_split: str = FULL_EVAL_SPLIT_DEFAULT) -> None:
    """Опционально запускает deterministic PhysicsBaseline_OZZ перед агрегацией."""
    if not enabled:
        return
    try:
        print(f"[PhysicsBaseline] Запуск детерминированной baseline-модели (split='{eval_split}')...")
        from scripts.evaluation.evaluate_physics_baseline import main as physics_baseline_main
        physics_baseline_main(eval_split=eval_split)
        print("[PhysicsBaseline] Готово: результаты baseline обновлены.")
    except Exception as exc:
        print(f"[!] Ошибка запуска PhysicsBaseline: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
    parser.add_argument("--root", type=str, default=None, help="Корневая директория экспериментов")
    parser.add_argument("--out-dir", type=str, default=None, help="Путь к папке для сохранения всех отчетов")
    parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
    parser.add_argument("--benchmark", action="store_true", help="Выполнить глубокий бенчмарк на CPU")
    parser.add_argument("--full-eval", action="store_true", help="Выполнить полную оценку на тестовом датасете (GPU)")
    parser.add_argument("--advanced-plots", action="store_true", help="Генерировать расширенные графики (Парето, heatmaps)")
    parser.add_argument("--data-dir", type=str, default=None, help="Путь к директории с датасетами")
    parser.add_argument("--extra-root", action="append", default=None, help="Дополнительная директория с экспериментами")
    parser.add_argument("--run-physics-baseline", action="store_true", help="Перед агрегацией запустить PhysicsBaseline_OZZ")
    parser.add_argument("--lang", type=str, default="ru", choices=["ru", "en"], help="Язык графиков (ru/en)")
    
    args, unknown = parser.parse_known_args()

    MANUAL_RUN = True
    
    if MANUAL_RUN or len(sys.argv) <= 1 or args.root is None:
        # Корневая директория, где ищем эксперименты для агрегации.
        ROOT_DIR = "experiments/Для_запуска_стат"
        # Папка, куда сохраняются сводный CSV, графики и служебные логи.
        OUTPUT_DIR = "reports/Exp_2_5_and_start_Exp_2_6"
        # Генерировать базовые графики обучения и сравнительные фигуры.
        GENERATE_PLOTS = True
        # Пересчитывать CPU benchmark инференса (если не закэширован).
        RUN_BENCHMARK = True
        # Запускать полную оценку (best/final) на выбранном сплите.
        RUN_FULL_EVAL = True
        # Выбранный сплит для full_eval: 'test' (по умолчанию) или 'train'.
        FULL_EVAL_SPLIT = "test"
        # Строить расширенные графики (heatmap, Pareto и т.д.).
        ADVANCED_PLOTS = True
        # Язык подписей на графиках и в визуализациях.
        LANG = "ru"
        # Подпапка инженерных графиков внутри OUTPUT_DIR.
        ENGINEERING_SUBDIR = ENGINEERING_PLOTS_SUBDIR_DEFAULT
        # Ограничение на число моделей в объединённых инженерных графиках.
        MAX_MODELS_FOR_COMBINED = MAX_MODELS_FOR_COMBINED_DEFAULT
        # Перед агрегацией запустить формульный PhysicsBaseline.
        ENABLE_PHYSICS_BASELINE = True
        # Дополнительные папки экспериментов, которые надо включить в сканирование.
        ADDITIONAL_EXPERIMENT_ROOTS = [
            "experiments/phase2_6/PhysicsBaseline_OZZ"
        ]
        # Тонкая настройка инженерных графиков (можно отключать отдельные виды).
        PLOT_SWITCHES = {
            'engineering_bars_per_model_absolute': True,
            'engineering_bars_per_model_relative': True,
            'engineering_bars_combined_absolute': True,
            'engineering_bars_combined_relative': True,
            'custom_cm_per_model_absolute': True,
            'custom_cm_per_model_relative': True,
        }
        print(f"[!] Запуск в ручном режиме (MANUAL_RUN)")
    else:
        ROOT_DIR = args.root
        OUTPUT_DIR = args.out_dir
        GENERATE_PLOTS = args.plot
        RUN_BENCHMARK = args.benchmark
        RUN_FULL_EVAL = args.full_eval
        FULL_EVAL_SPLIT = 'test'
        ADVANCED_PLOTS = args.advanced_plots
        LANG = args.lang
        ENGINEERING_SUBDIR = ENGINEERING_PLOTS_SUBDIR_DEFAULT
        MAX_MODELS_FOR_COMBINED = MAX_MODELS_FOR_COMBINED_DEFAULT
        ENABLE_PHYSICS_BASELINE = bool(args.run_physics_baseline)
        ADDITIONAL_EXPERIMENT_ROOTS = args.extra_root or []
        PLOT_SWITCHES = PLOT_SWITCHES_DEFAULT.copy()

    if ROOT_DIR:
        run_physics_baseline_if_enabled(ENABLE_PHYSICS_BASELINE, eval_split=FULL_EVAL_SPLIT)
        aggregate_reports(
            ROOT_DIR, 
            OUTPUT_DIR, 
            GENERATE_PLOTS, 
            RUN_BENCHMARK,
            full_eval=RUN_FULL_EVAL,
            lang=LANG,
            data_dir=args.data_dir,
            advanced_plots=ADVANCED_PLOTS,
            plot_switches=PLOT_SWITCHES,
            engineering_subdir=ENGINEERING_SUBDIR,
            max_models_for_combined=MAX_MODELS_FOR_COMBINED,
            extra_experiment_roots=ADDITIONAL_EXPERIMENT_ROOTS,
            full_eval_split=FULL_EVAL_SPLIT,
        )
