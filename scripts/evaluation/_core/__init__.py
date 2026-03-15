"""
Пакет _core: вынесенные подмодули из aggregate_reports.py.

Реэкспортирует все публичные символы для обратной совместимости.
Внешние потребители могут продолжать использовать:
    from scripts.evaluation.aggregate_reports import <symbol>
"""

# --- constants ---
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
)

# --- cache_and_data ---
from scripts.evaluation._core.cache_and_data import (
    load_existing_summary,
    needs_full_eval_recalc,
    needs_benchmark_recalc,
    get_existing_row,
    load_metrics,
    load_history_as_metrics,
    load_config,
)

# --- config_resolvers ---
from scripts.evaluation._core.config_resolvers import (
    parse_experiment_info,
    extract_base_exp_id,
    _coerce_per_class_f1_values,
)

# --- model_utils ---
from scripts.evaluation._core.model_utils import (
    _create_model_from_config,
    _normalize_state_dict_keys,
    _load_state_dict_safe,
    _get_eval_batch_size,
    benchmark_model_cpu,
    set_eval_logger,
    get_eval_logger,
)

# --- predictions ---
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

# --- full_evaluation ---
from scripts.evaluation._core.full_evaluation import (
    _test_data_cache,
    _get_test_data_cached,
    _evaluate_with_batch_fallback,
    _evaluate_hierarchical_with_batch_fallback,
    evaluate_model_full,
    evaluate_full_test_dataset,
)

# --- engineering_stats ---
from scripts.evaluation._core.engineering_stats import (
    build_engineering_stats,
    build_engineering_stats_multilabel,
    choose_selected_predictions_per_experiment,
    collapse_selected_to_model_level,
)

# --- engineering_plots ---
from scripts.evaluation._core.engineering_plots import (
    plot_engineering_bidirectional_bars,
    build_error_severity_matrix,
    plot_custom_confusion_matrix,
    plot_multilabel_confusion_matrices,
    plot_engineering_bars_combined,
    generate_engineering_plot_pack,
)

# --- report_visualizer ---
from scripts.evaluation._core.report_visualizer import (
    plot_learning_curves,
    plot_comparison,
    combine_training_histories,
    ReportVisualizer,
)
