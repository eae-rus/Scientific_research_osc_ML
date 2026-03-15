"""
Инженерные графики: столбцы TP/FP/FN, матрицы ошибок, пакетная генерация.

Все функции визуализации для инженерных метрик.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from scripts.evaluation._core.constants import (
    OZZ_MULTILABEL_TRUE_COLS,
    OZZ_MULTILABEL_PRED_COLS,
    get_engineering_class_map,
)
from scripts.evaluation._core.predictions import _is_multilabel_pred_df
from scripts.evaluation._core.engineering_stats import (
    build_engineering_stats,
    build_engineering_stats_multilabel,
)


def plot_engineering_bidirectional_bars(
    stats_df: pd.DataFrame,
    title: str,
    save_path: Path,
    relative: bool = False,
    show_fp_fn_split: bool = True
) -> None:
    """
    Двунаправленные инженерные столбцы:
    вверх TP (зелёный), вниз ошибки (красный, опционально split FP/FN),
    плюс контурный столбец GT.
    """
    plot_df = stats_df.copy()
    if relative:
        gt_safe = plot_df['gt'].replace(0, np.nan)
        plot_df['tp_v'] = (plot_df['tp'] / gt_safe * 100.0).fillna(0.0)
        plot_df['fp_v'] = (plot_df['fp'] / gt_safe * 100.0).fillna(0.0)
        plot_df['fn_v'] = (plot_df['fn'] / gt_safe * 100.0).fillna(0.0)
        plot_df['err_v'] = (plot_df['errors'] / gt_safe * 100.0).fillna(0.0)
        plot_df['gt_v'] = 100.0
        y_label = 'Доля от GT класса, %'
    else:
        plot_df['tp_v'] = plot_df['tp']
        plot_df['fp_v'] = plot_df['fp']
        plot_df['fn_v'] = plot_df['fn']
        plot_df['err_v'] = plot_df['errors']
        plot_df['gt_v'] = plot_df['gt']
        y_label = 'Количество окон'

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x,
        plot_df['gt_v'].to_numpy(),
        width=0.72,
        fill=False,
        edgecolor='black',
        linewidth=1.3,
        linestyle='--',
        alpha=0.7,
        label='Ground Truth'
    )

    ax.bar(x, plot_df['tp_v'].to_numpy(), width=0.55, color='#2ca02c', alpha=0.85, label='TP')

    if show_fp_fn_split:
        ax.bar(x, -plot_df['fp_v'].to_numpy(), width=0.55, color='#ff7f0e', alpha=0.85, label='FP')
        ax.bar(
            x,
            -plot_df['fn_v'].to_numpy(),
            width=0.55,
            bottom=-plot_df['fp_v'].to_numpy(),
            color='#d62728',
            alpha=0.85,
            label='FN'
        )
    else:
        ax.bar(x, -plot_df['err_v'].to_numpy(), width=0.55, color='#d62728', alpha=0.85, label='Ошибки (FP+FN)')

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['class_name'].tolist(), rotation=0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='upper right')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def build_error_severity_matrix(class_ids: List[int]) -> np.ndarray:
    """
    Матрица важности ошибок для РЗА.
    Повышаем вес фатальной путаницы Коммутация->Авария (1->3).
    """
    n = len(class_ids)
    severity = np.ones((n, n), dtype=float)
    np.fill_diagonal(severity, 0.0)

    id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    if 1 in id_to_idx and 3 in id_to_idx:
        severity[id_to_idx[1], id_to_idx[3]] = 1.8
    if 3 in id_to_idx and 2 in id_to_idx:
        severity[id_to_idx[3], id_to_idx[2]] = 1.2

    return severity


def plot_custom_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_map: Dict[int, str],
    title: str,
    save_path: Path,
    relative: bool = False,
    severity_matrix: Optional[np.ndarray] = None
) -> None:
    """
    Кастомная матрица ошибок:
    - диагональ: зелёный слой (правильные ответы),
    - вне диагонали: красно-жёлтый слой (ошибки) с возможным усилением severity.
    """
    class_ids = list(class_map.keys())
    labels = [class_map[i] for i in class_ids]
    cm = confusion_matrix(y_true, y_pred, labels=class_ids).astype(float)

    if relative:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0) * 100.0

    if severity_matrix is None:
        severity_matrix = build_error_severity_matrix(class_ids)

    cm_diag = np.zeros_like(cm)
    cm_off = cm.copy()
    for i in range(cm.shape[0]):
        cm_diag[i, i] = cm[i, i]
        cm_off[i, i] = 0.0

    cm_off_weighted = cm_off * severity_matrix

    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    sns.heatmap(
        cm_off_weighted,
        cmap='YlOrRd',
        annot=False,
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.7,
        linecolor='white',
        ax=ax
    )

    sns.heatmap(
        cm_diag,
        cmap='Greens',
        annot=False,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        mask=(cm_diag == 0),
        linewidths=0.7,
        linecolor='white',
        ax=ax
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.1f}%" if relative else f"{int(val)}"
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', color='black', fontsize=9)

    ax.set_title(title)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_multilabel_confusion_matrices(
    pred_df: pd.DataFrame,
    class_map: Dict[int, str],
    title_prefix: str,
    save_path: Path,
    relative: bool = False
) -> None:
    """
    Строит 3 отдельные бинарные матрицы ошибок (2x2) для OZZ multilabel.
    """
    n_targets = len(class_map)
    fig, axes = plt.subplots(1, n_targets, figsize=(5.5 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for idx, (class_id, class_name) in enumerate(class_map.items()):
        ax = axes[idx]
        tc = OZZ_MULTILABEL_TRUE_COLS[class_id]
        pc = OZZ_MULTILABEL_PRED_COLS[class_id]
        y_t = pred_df[tc].to_numpy()
        y_p = pred_df[pc].to_numpy()

        cm = confusion_matrix(y_t, y_p, labels=[0, 1]).astype(float)
        if relative:
            row_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0) * 100.0

        labels = ['Нет', 'Да']
        sns.heatmap(
            cm, annot=True,
            fmt='.1f' if relative else '.0f',
            cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.7, linecolor='white',
            ax=ax
        )
        ax.set_title(class_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle(title_prefix, fontsize=13, y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_engineering_bars_combined(
    selected_by_model: Dict[str, pd.DataFrame],
    class_map: Dict[int, str],
    save_path: Path,
    relative: bool = False
) -> None:
    """
    Обобщённый график по нескольким моделям: для каждого класса
    показываем TP вверх и ошибки вниз по цветам моделей.
    """
    if not selected_by_model:
        return

    model_names = list(selected_by_model.keys())
    class_ids = list(class_map.keys())
    class_labels = [class_map[c] for c in class_ids]

    per_model_stats: Dict[str, pd.DataFrame] = {}
    for model_name, pred_df in selected_by_model.items():
        if _is_multilabel_pred_df(pred_df):
            per_model_stats[model_name] = build_engineering_stats_multilabel(pred_df, class_map)
        else:
            per_model_stats[model_name] = build_engineering_stats(
                pred_df['y_true'].to_numpy(),
                pred_df['y_pred'].to_numpy(),
                class_map
            )

    n_models = len(model_names)
    x = np.arange(len(class_ids))
    width = 0.78 / max(n_models, 1)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(max(11, 1.8 * len(class_ids) + n_models), 6.6))

    for idx, model_name in enumerate(model_names):
        st = per_model_stats[model_name].copy()
        if relative:
            gt_safe = st['gt'].replace(0, np.nan)
            tp_vals = (st['tp'] / gt_safe * 100.0).fillna(0.0).to_numpy()
            err_vals = (st['errors'] / gt_safe * 100.0).fillna(0.0).to_numpy()
        else:
            tp_vals = st['tp'].to_numpy()
            err_vals = st['errors'].to_numpy()

        offset = (idx - (n_models - 1) / 2.0) * width
        color = cmap(idx % 10)

        ax.bar(x + offset, tp_vals, width=width * 0.95, color=color, alpha=0.82)
        ax.bar(x + offset, -err_vals, width=width * 0.95, color=color, alpha=0.35)

    first_stats = next(iter(per_model_stats.values())).copy()
    if relative:
        gt_reference = np.full(len(class_ids), 100.0)
    else:
        gt_reference = first_stats['gt'].to_numpy(dtype=float)

    cluster_half = (n_models * width) / 2.0
    for i, gt_val in enumerate(gt_reference):
        x_left = x[i] - cluster_half
        x_right = x[i] + cluster_half
        ax.hlines(
            y=gt_val,
            xmin=x_left,
            xmax=x_right,
            colors='black',
            linestyles='--',
            linewidth=2.3,
            alpha=0.9,
            zorder=6
        )
        ax.scatter(
            x[i],
            gt_val,
            s=36,
            color='black',
            alpha=0.95,
            zorder=7
        )

    legend_handles = [
        plt.Line2D([0], [0], color='black', lw=1.3, linestyle='--', label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2ca02c', alpha=0.8, label='TP (вверх)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#d62728', alpha=0.35, label='Ошибки (вниз)')
    ]
    for idx, model_name in enumerate(model_names):
        legend_handles.append(
            plt.Line2D([0], [0], color=cmap(idx % 10), lw=5, label=model_name)
        )

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_title('Обобщённые инженерные столбцы по выбранным моделям')
    ax.set_ylabel('Доля от GT, %' if relative else 'Количество окон')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=190)
    plt.close(fig)


def generate_engineering_plot_pack(
    selected_by_model: Dict[str, Dict[str, any]],
    output_dir: Path,
    plot_switches: Dict[str, bool],
    max_models_for_combined: int
) -> None:
    """
    Генерирует пакет инженерных графиков по выбранным моделям.
    Для OZZ multilabel: каждый таргет оценивается как бинарный классификатор.
    """
    if not selected_by_model:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_info in selected_by_model.items():
        pred_df = model_info['pred_df']
        target_level = model_info.get('target_level', 'base')
        class_map = get_engineering_class_map(target_level)
        is_ml = _is_multilabel_pred_df(pred_df)

        if is_ml:
            stats_df = build_engineering_stats_multilabel(pred_df, class_map)
        else:
            stats_df = build_engineering_stats(
                pred_df['y_true'].to_numpy(),
                pred_df['y_pred'].to_numpy(),
                class_map
            )

        model_slug = re.sub(r'[^a-zA-Z0-9._-]+', '_', model_name)

        if plot_switches.get('engineering_bars_per_model_absolute', True):
            plot_engineering_bidirectional_bars(
                stats_df,
                title=f"{model_name}: инженерные столбцы (абсолютные)",
                save_path=output_dir / f"bars_{model_slug}_abs.png",
                relative=False,
                show_fp_fn_split=True
            )

        if plot_switches.get('engineering_bars_per_model_relative', True):
            plot_engineering_bidirectional_bars(
                stats_df,
                title=f"{model_name}: инженерные столбцы (относительные)",
                save_path=output_dir / f"bars_{model_slug}_rel.png",
                relative=True,
                show_fp_fn_split=True
            )

        if is_ml:
            if plot_switches.get('custom_cm_per_model_absolute', True):
                plot_multilabel_confusion_matrices(
                    pred_df, class_map,
                    title_prefix=f"{model_name}: матрицы ошибок (абсолютные)",
                    save_path=output_dir / f"cm_{model_slug}_abs.png",
                    relative=False
                )
            if plot_switches.get('custom_cm_per_model_relative', True):
                plot_multilabel_confusion_matrices(
                    pred_df, class_map,
                    title_prefix=f"{model_name}: матрицы ошибок (% по строкам)",
                    save_path=output_dir / f"cm_{model_slug}_rel.png",
                    relative=True
                )
        else:
            if plot_switches.get('custom_cm_per_model_absolute', True):
                plot_custom_confusion_matrix(
                    pred_df['y_true'].to_numpy(),
                    pred_df['y_pred'].to_numpy(),
                    class_map=class_map,
                    title=f"{model_name}: кастомная матрица ошибок (абсолютные)",
                    save_path=output_dir / f"cm_{model_slug}_abs.png",
                    relative=False
                )
            if plot_switches.get('custom_cm_per_model_relative', True):
                plot_custom_confusion_matrix(
                    pred_df['y_true'].to_numpy(),
                    pred_df['y_pred'].to_numpy(),
                    class_map=class_map,
                    title=f"{model_name}: кастомная матрица ошибок (% по строкам)",
                    save_path=output_dir / f"cm_{model_slug}_rel.png",
                    relative=True
                )

    if len(selected_by_model) <= max_models_for_combined:
        by_target_level: Dict[str, Dict[str, pd.DataFrame]] = {}
        for model_name, model_info in selected_by_model.items():
            t_level = str(model_info.get('target_level', 'base')).lower()
            by_target_level.setdefault(t_level, {})[model_name] = model_info['pred_df']

        for t_level, model_group in by_target_level.items():
            class_map = get_engineering_class_map(t_level)
            suffix = '' if len(by_target_level) == 1 else f'_{t_level}'

            if plot_switches.get('engineering_bars_combined_absolute', True):
                plot_engineering_bars_combined(
                    selected_by_model=model_group,
                    class_map=class_map,
                    save_path=output_dir / f'bars_combined_abs{suffix}.png',
                    relative=False
                )
            if plot_switches.get('engineering_bars_combined_relative', True):
                plot_engineering_bars_combined(
                    selected_by_model=model_group,
                    class_map=class_map,
                    save_path=output_dir / f'bars_combined_rel{suffix}.png',
                    relative=True
                )
