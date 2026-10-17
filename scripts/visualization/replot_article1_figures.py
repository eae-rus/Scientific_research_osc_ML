"""Перерисовка отдельных рисунков для первой статьи.

Скрипт не запускает обучение и не пересчитывает метрики моделей. Он берёт уже
сохранённые таблицы/предсказания и строит версии графиков под печать:
  - рисунок 7: диаграммы Парето;
  - рисунок 8: radar-график F1 по классам;
  - рисунок 9: инженерные столбцы TP/ошибок по выбранным моделям.

Общая идея оформления:
  - цвет оставлен для PDF-версии;
  - форма/тип линии/номера помогают отличать модели в ЧБ;
  - подписи и легенды увеличены для двухколоночной печати.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ORDER = [
    "MLP",
    "CNN",
    "ResNet",
    "SimpleKAN",
    "ConvKAN",
    "PhysicsKAN",
    "cPhysicsKAN",
    "rPhysicsKAN",
]

MODEL_COLORS = {
    "MLP": "#7f8c8d",
    "CNN": "#1f77b4",
    "ResNet": "#2ca02c",
    "SimpleKAN": "#ff7f0e",
    "ConvKAN": "#d62728",
    "PhysicsKAN": "#9467bd",
    "cPhysicsKAN": "#f1c40f",
    "rPhysicsKAN": "#b7950b",
}

MODEL_MARKERS = {
    "MLP": "o",
    "CNN": "s",
    "ResNet": "^",
    "SimpleKAN": "v",
    "ConvKAN": "D",
    "PhysicsKAN": "P",
    "cPhysicsKAN": "X",
    "rPhysicsKAN": "*",
}

MODEL_LINESTYLES = {
    "MLP": "-",
    "CNN": "--",
    "ResNet": "-.",
    "SimpleKAN": ":",
    "ConvKAN": (0, (5, 2)),
    "PhysicsKAN": (0, (3, 1, 1, 1)),
    "cPhysicsKAN": (0, (1, 1)),
    "rPhysicsKAN": (0, (6, 2, 1, 2)),
}

EXPERIMENT_MODEL_ALIASES = {
    "MLP": ["SimpleMLP", "MLP"],
    "CNN": ["SimpleCNN", "CNN"],
    "ResNet": ["ResNet1D", "ResNet"],
    "SimpleKAN": ["SimpleKAN"],
    "ConvKAN": ["ConvKAN"],
    "PhysicsKAN": ["PhysicsKAN"],
    "cPhysicsKAN": ["cPhysicsKAN"],
    "rPhysicsKAN": ["rPhysicsKAN"],
}

FIG9_REQUIRED_COMPLEXITY = {
    "MLP": "medium",
    "CNN": "medium",
    "ResNet": "medium",
    "SimpleKAN": "light",
    "ConvKAN": "heavy",
    "PhysicsKAN": "heavy",
    "cPhysicsKAN": "heavy",
    "rPhysicsKAN": "heavy",
}

ENGINEERING_CLASS_MAP = {
    0: "Норма",
    1: "Коммутация",
    2: "Аномалия",
    3: "Авария",
}

PREDICTION_FILE_PATTERNS = {
    "best": [
        "test_predictions_best.csv",
        "predictions_best.csv",
        "best_predictions.csv",
        "best_test_predictions.csv",
    ],
    "final": [
        "test_predictions_final.csv",
        "predictions_final.csv",
        "final_predictions.csv",
        "final_test_predictions.csv",
    ],
}

COMPLEXITY_ORDER = ["Light", "Medium", "Heavy"]
COMPLEXITY_SHORT = {"Light": "L", "Medium": "M", "Heavy": "H"}
COMPLEXITY_FACE = {
    "Light": "white",
    "Medium": "#bdbdbd",
    "Heavy": "#3f3f3f",
}
COMPLEXITY_SIZE = {
    "Light": 105,
    "Medium": 135,
    "Heavy": 175,
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _best_full_f1(row: pd.Series) -> float:
    return max(
        float(row.get("Full Best F1", 0.0) or 0.0),
        float(row.get("Full Final F1", 0.0) or 0.0),
    )


def _pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Возвращает недоминируемые точки: меньше X и больше Y лучше."""
    points = df[[x_col, y_col]].to_numpy()
    keep: list[int] = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if (x_j <= x_i and y_j >= y_i) and (x_j < x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(x_col)


def _load_article_fig7_data(summary_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(summary_csv))
    required = {"Model", "Complexity", "Params", "CPU Inf (ms)", "Full Best F1", "Full Final F1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"В таблице нет обязательных колонок: {missing}")

    df = df.copy()
    df["Best Full F1"] = df.apply(_best_full_f1, axis=1)
    df = df[
        df["Model"].isin(MODEL_ORDER)
        & df["Complexity"].isin(COMPLEXITY_ORDER)
        & (df["Best Full F1"] > 0)
        & (df["Params"] > 0)
        & (df["CPU Inf (ms)"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("После фильтрации не осталось точек для рисунка 7.")

    df["Model"] = pd.Categorical(df["Model"], MODEL_ORDER, ordered=True)
    df["Complexity"] = pd.Categorical(df["Complexity"], COMPLEXITY_ORDER, ordered=True)
    return df.sort_values(["Model", "Complexity"])


def _legend_model_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for model in MODEL_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=MODEL_COLORS[model],
                markeredgewidth=2.0,
                markersize=9,
                label=model,
            )
        )
    return handles


def _legend_complexity_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for complexity in COMPLEXITY_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=COMPLEXITY_FACE[complexity],
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=8 + 1.5 * COMPLEXITY_ORDER.index(complexity),
                label=f"{COMPLEXITY_SHORT[complexity]} - {complexity}",
            )
        )
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Фронт Парето")
    )
    return handles


def _legend_spacer(label: str = "") -> Line2D:
    """Пустой элемент легенды для визуального разделения групп."""
    return Line2D([0], [0], linestyle="None", marker="", color="none", label=label)


def _plot_one_pareto(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    out_path: Path,
    figure_width: float,
    figure_height: float,
    legend_mode: str,
    annotate_pareto: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    for _, row in df.iterrows():
        model = str(row["Model"])
        complexity = str(row["Complexity"])
        ax.scatter(
            row[x_col],
            row["Best Full F1"],
            s=COMPLEXITY_SIZE[complexity],
            marker=MODEL_MARKERS[model],
            facecolor=COMPLEXITY_FACE[complexity],
            edgecolor=MODEL_COLORS[model],
            linewidth=2.0,
            alpha=0.98,
            zorder=3,
        )

    pareto = _pareto_front(df, x_col, "Best Full F1")
    ax.plot(
        pareto[x_col],
        pareto["Best Full F1"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.7,
        zorder=2,
    )

    if annotate_pareto:
        for _, row in pareto.iterrows():
            ax.annotate(
                f"{row['Model']}\n{COMPLEXITY_SHORT[str(row['Complexity'])]}",
                xy=(row[x_col], row["Best Full F1"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
            )

    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel("Full Test F1-Macro (лучшая)", fontsize=13)
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, which="major", color="#808080", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", color="#a0a0a0", alpha=0.18, linewidth=0.6)

    # Пустой элемент после моделей сдвигает сложности в отдельный визуальный блок.
    legend_handles = _legend_model_handles() + [_legend_spacer()] + _legend_complexity_handles()
    legend_mode = legend_mode.lower().strip()
    if legend_mode == "bottom":
        ax.legend(
            handles=legend_handles,
            title="Обозначения: модель - форма/цвет; сложность - заливка/размер",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=5,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            columnspacing=1.45,
            handletextpad=0.7,
        )
        fig.tight_layout(rect=(0, 0.30, 1, 1))
    elif legend_mode == "right":
        ax.legend(
            handles=legend_handles,
            title="Обозначения",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            handletextpad=0.7,
        )
        fig.tight_layout(rect=(0, 0, 0.80, 1))
    elif legend_mode == "none":
        fig.tight_layout()
    else:
        raise ValueError("legend_mode должен быть 'bottom', 'right' или 'none'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def replot_article1_fig7_pareto(
    summary_csv: str | Path,
    output_dir: str | Path,
    figure_width: float = 8.2,
    figure_height: float = 7.2,
    show_panel_titles: bool = False,
    legend_mode: str = "right",
    annotate_pareto: bool = True,
) -> dict[str, Path]:
    """Строит две версии рисунка 7 из готового summary_aggregated.csv."""
    df = _load_article_fig7_data(summary_csv)
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Сохраняем ровно тот набор точек, который ушёл в рисунок.
    used_csv = output_path / "fig7_pareto_points_used.csv"
    df.to_csv(used_csv, index=False)

    cpu_path = output_path / "fig7a_pareto_full_f1_cpu_article.png"
    params_path = output_path / "fig7b_pareto_full_f1_params_article.png"

    _plot_one_pareto(
        df,
        x_col="CPU Inf (ms)",
        x_label="Время инференса CPU, мс (лог. шкала)",
        title="а) Парето: качество и скорость расчёта" if show_panel_titles else "",
        out_path=cpu_path,
        figure_width=figure_width,
        figure_height=figure_height,
        legend_mode=legend_mode,
        annotate_pareto=annotate_pareto,
    )

    _plot_one_pareto(
        df,
        x_col="Params",
        x_label="Количество параметров (лог. шкала)",
        title="б) Парето: качество и размер модели" if show_panel_titles else "",
        out_path=params_path,
        figure_width=figure_width,
        figure_height=figure_height,
        legend_mode=legend_mode,
        annotate_pareto=annotate_pareto,
    )

    return {
        "cpu_png": cpu_path,
        "cpu_svg": cpu_path.with_suffix(".svg"),
        "params_png": params_path,
        "params_svg": params_path.with_suffix(".svg"),
        "points_csv": used_csv,
    }


def _class_f1_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("Class_") and c.endswith("_F1")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def _numbered_model_label(model: str) -> str:
    return f"{MODEL_ORDER.index(model) + 1}. {model}"


def _normalize_prediction_columns(df_pred: pd.DataFrame) -> pd.DataFrame:
    result = df_pred.copy()
    true_col = next((c for c in ["y_true", "true", "target", "label_true", "gt"] if c in result.columns), None)
    pred_col = next((c for c in ["y_pred", "pred", "prediction", "label_pred"] if c in result.columns), None)
    if true_col is None or pred_col is None:
        raise ValueError("В таблице предсказаний не найдены колонки y_true/y_pred")
    result = result.rename(columns={true_col: "y_true", pred_col: "y_pred"})
    result = result.dropna(subset=["y_true", "y_pred"]).copy()
    result["y_true"] = pd.to_numeric(result["y_true"], errors="coerce")
    result["y_pred"] = pd.to_numeric(result["y_pred"], errors="coerce")
    result = result.dropna(subset=["y_true", "y_pred"]).copy()
    result["y_true"] = result["y_true"].astype(int)
    result["y_pred"] = result["y_pred"].astype(int)
    return result[["y_true", "y_pred"]]


def _load_best_final_predictions(exp_dir: Path) -> dict[str, pd.DataFrame | None]:
    loaded: dict[str, pd.DataFrame | None] = {"best": None, "final": None}
    for version, patterns in PREDICTION_FILE_PATTERNS.items():
        for pattern in patterns:
            candidate = exp_dir / pattern
            if not candidate.exists():
                continue
            try:
                loaded[version] = _normalize_prediction_columns(pd.read_csv(candidate))
                break
            except Exception:
                continue
    return loaded


def _macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    labels = sorted(set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist()))
    if not labels:
        return float("nan")
    f1_values: list[float] = []
    yt = y_true.to_numpy()
    yp = y_pred.to_numpy()
    for label in labels:
        tp = float(((yt == label) & (yp == label)).sum())
        fp = float(((yt != label) & (yp == label)).sum())
        fn = float(((yt == label) & (yp != label)).sum())
        denom = 2 * tp + fp + fn
        f1_values.append(0.0 if denom == 0 else (2 * tp / denom))
    return float(np.mean(f1_values))


def _select_best_final_by_f1(versions: dict[str, pd.DataFrame | None]) -> dict[str, object]:
    best_df = versions.get("best")
    final_df = versions.get("final")
    best_f1 = _macro_f1(best_df["y_true"], best_df["y_pred"]) if best_df is not None else np.nan
    final_f1 = _macro_f1(final_df["y_true"], final_df["y_pred"]) if final_df is not None else np.nan
    if np.isnan(best_f1) and np.isnan(final_f1):
        return {"selected_f1": np.nan, "selected_df": None}
    if np.isnan(final_f1) or (not np.isnan(best_f1) and best_f1 >= final_f1):
        return {"selected_f1": best_f1, "selected_df": best_df}
    return {"selected_f1": final_f1, "selected_df": final_df}


def _per_class_f1(y_true: pd.Series, y_pred: pd.Series, class_ids: Iterable[int]) -> dict[int, float]:
    yt = y_true.to_numpy()
    yp = y_pred.to_numpy()
    result: dict[int, float] = {}
    for class_id in class_ids:
        tp = float(((yt == class_id) & (yp == class_id)).sum())
        fp = float(((yt != class_id) & (yp == class_id)).sum())
        fn = float(((yt == class_id) & (yp != class_id)).sum())
        denom = 2 * tp + fp + fn
        result[int(class_id)] = 0.0 if denom == 0 else float(2 * tp / denom)
    return result


def _load_article_fig8_data(summary_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(summary_csv))
    class_cols = _class_f1_columns(df)
    if not class_cols:
        raise ValueError("В таблице нет колонок Class_<n>_F1 для рисунка 8.")

    df = df[df["Model"].isin(MODEL_ORDER)].copy()
    df[class_cols] = df[class_cols].apply(pd.to_numeric, errors="coerce")
    grouped = df.groupby("Model", observed=True)[class_cols].mean()
    grouped = grouped.reindex([m for m in MODEL_ORDER if m in grouped.index]).dropna(how="all")
    if grouped.empty:
        raise ValueError("После группировки не осталось данных для рисунка 8.")
    grouped.index.name = "Model"
    return grouped.reset_index()


def _load_article_fig8_fixed_complexity_summary(summary_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(summary_csv))
    class_cols = _class_f1_columns(df)
    if not class_cols:
        raise ValueError("В таблице нет колонок Class_<n>_F1 для рисунка 8.")
    rows: list[pd.Series] = []
    missing: list[str] = []
    for model in MODEL_ORDER:
        required = FIG9_REQUIRED_COMPLEXITY[model].capitalize()
        match = df[
            (df["Model"].astype(str) == model)
            & (df["Complexity"].astype(str).str.lower() == required.lower())
        ].copy()
        if match.empty:
            missing.append(f"{model} ({required})")
            continue
        score_col = "Full Best F1" if "Full Best F1" in match.columns else class_cols[0]
        match[score_col] = pd.to_numeric(match[score_col], errors="coerce")
        rows.append(match.sort_values(score_col, ascending=False).iloc[0])
    if missing:
        raise ValueError("В summary не найдены требуемые строки для рисунка 8: " + ", ".join(missing))
    result = pd.DataFrame(rows)
    keep_cols = ["Model", "Complexity", "ExpID"] + class_cols
    keep_cols = [c for c in keep_cols if c in result.columns]
    return result[keep_cols].reset_index(drop=True)


def _load_article_fig8_exported_data(article_data_dir: str | Path, fallback_summary_csv: str | Path) -> pd.DataFrame:
    article_path = _resolve(article_data_dir)
    selected_path = article_path / "radar_selected_models_class_f1.csv"
    mean_path = article_path / "radar_model_type_mean_class_f1.csv"
    if selected_path.exists():
        return _load_article_fig8_data(selected_path)
    if mean_path.exists():
        return _load_article_fig8_data(mean_path)
    return _load_article_fig8_data(fallback_summary_csv)


def _load_article_fig8_fixed_complexity_predictions(
    selection_csv: str | Path,
    experiment_roots: Iterable[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_model, sources = _load_selected_predictions_by_model(selection_csv, experiment_roots)
    rows: list[dict[str, object]] = []
    class_ids = sorted(ENGINEERING_CLASS_MAP)
    for model in MODEL_ORDER:
        pred_df = selected_by_model.get(model)
        if pred_df is None:
            continue
        class_f1 = _per_class_f1(pred_df["y_true"], pred_df["y_pred"], class_ids)
        row: dict[str, object] = {"Model": model}
        for class_id in class_ids:
            row[f"Class_{class_id}_F1"] = class_f1[class_id]
        rows.append(row)
    if not rows:
        raise ValueError("Не удалось собрать F1 по классам из prediction CSV для рисунка 8.")
    return pd.DataFrame(rows), sources


def replot_article1_fig8_radar(
    summary_csv: str | Path,
    output_dir: str | Path,
    figure_width: float = 8.2,
    figure_height: float = 6.8,
    legend_mode: str = "right",
    annotate_line_numbers: bool = True,
    article_data_dir: str | Path | None = None,
    selection_csv: str | Path | None = None,
    experiment_roots: Iterable[str | Path] | None = None,
    use_fixed_complexity_predictions: bool = False,
    use_fixed_complexity_summary: bool = False,
) -> dict[str, Path]:
    """Строит radar-график F1 по классам с ЧБ-различимыми линиями."""
    sources = pd.DataFrame()
    if use_fixed_complexity_predictions:
        if selection_csv is None or experiment_roots is None:
            raise ValueError(
                "Для use_fixed_complexity_predictions нужны selection_csv и experiment_roots."
            )
        df, sources = _load_article_fig8_fixed_complexity_predictions(selection_csv, experiment_roots)
    elif use_fixed_complexity_summary:
        df = _load_article_fig8_fixed_complexity_summary(summary_csv)
    elif article_data_dir is not None:
        df = _load_article_fig8_exported_data(article_data_dir, summary_csv)
    else:
        df = _load_article_fig8_data(summary_csv)
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    used_csv = output_path / "fig8_radar_points_used.csv"
    sources_csv = output_path / "fig8_radar_sources_used.csv"
    df.to_csv(used_csv, index=False)
    if not sources.empty:
        sources.to_csv(sources_csv, index=False)

    class_cols = _class_f1_columns(df)
    class_labels = ["Норма", "Коммутации", "Аномалии", "Аварии"][: len(class_cols)]
    angles = np.linspace(0, 2 * np.pi, len(class_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(0)
    ax.set_theta_direction(1)

    for _, row in df.iterrows():
        model = str(row["Model"])
        values = [float(row[c]) for c in class_cols]
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            marker=MODEL_MARKERS[model],
            linewidth=2.1,
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=_numbered_model_label(model),
            zorder=3,
        )
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.035, zorder=1)

        if annotate_line_numbers:
            ax.annotate(
                str(MODEL_ORDER.index(model) + 1),
                xy=(angles[0], values[0]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8.5,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=MODEL_COLORS[model], alpha=0.85),
                zorder=5,
            )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, color="#808080", alpha=0.35, linewidth=0.8)
    ax.spines["polar"].set_color("#555555")

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[m],
            linestyle=MODEL_LINESTYLES[m],
            marker=MODEL_MARKERS[m],
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.1,
            label=_numbered_model_label(m),
        )
        for m in MODEL_ORDER
        if m in set(df["Model"])
    ]
    legend_mode = legend_mode.lower().strip()
    if legend_mode == "right":
        ax.legend(
            handles=handles,
            title="Модели",
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            fontsize=9.5,
            title_fontsize=10,
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 0.82, 1))
    elif legend_mode == "bottom":
        ax.legend(
            handles=handles,
            title="Модели",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=4,
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            columnspacing=1.4,
            handletextpad=0.7,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1))
    elif legend_mode == "none":
        fig.tight_layout()
    else:
        raise ValueError("legend_mode должен быть 'bottom', 'right' или 'none'")

    out_path = output_path / "fig8_radar_by_model_type_article.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    result = {"radar_png": out_path, "radar_svg": out_path.with_suffix(".svg"), "points_csv": used_csv}
    if not sources.empty:
        result["sources_csv"] = sources_csv
    return result


def _find_experiment_dir(exp_name: str, experiment_roots: Iterable[str | Path]) -> Path | None:
    for root in experiment_roots:
        root_path = _resolve(root)
        direct = root_path / exp_name
        if direct.exists():
            return direct
        matches = list(root_path.rglob(exp_name))
        if matches:
            return matches[0]
    return None


def _canonical_model_from_experiment(exp_name: str) -> str | None:
    for model, aliases in EXPERIMENT_MODEL_ALIASES.items():
        if any(f"_{alias}_" in exp_name for alias in aliases):
            return model
    return None


def _complexity_from_experiment(exp_name: str) -> str | None:
    low = exp_name.lower()
    for complexity in ("light", "medium", "heavy"):
        if f"_{complexity}_" in low:
            return complexity
    return None


def _load_selected_predictions_by_model(
    selection_csv: str | Path,
    experiment_roots: Iterable[str | Path],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    selection = pd.read_csv(_resolve(selection_csv))
    candidates: list[dict[str, object]] = []
    seen_experiments: set[str] = set()

    def add_candidate(exp_name: str, exp_dir: Path) -> None:
        if exp_name in seen_experiments:
            return
        seen_experiments.add(exp_name)
        model = _canonical_model_from_experiment(exp_name)
        if model is None or model not in MODEL_ORDER:
            return
        complexity = _complexity_from_experiment(exp_name)
        if complexity != FIG9_REQUIRED_COMPLEXITY.get(model):
            return
        versions = _load_best_final_predictions(exp_dir)
        picked = _select_best_final_by_f1(versions)
        if picked["selected_df"] is None:
            return
        candidates.append(
            {
                "model": model,
                "experiment": exp_name,
                "complexity": complexity,
                "source_dir": str(exp_dir),
                "selected_f1": picked["selected_f1"],
                "selected_df": picked["selected_df"],
            }
        )

    for _, row in selection.iterrows():
        exp_name = str(row["Experiment"])
        exp_dir = _find_experiment_dir(exp_name, experiment_roots)
        if exp_dir is not None:
            add_candidate(exp_name, exp_dir)

    # В старых отчётах часть сохранённых предсказаний лежит в отдельной папке
    # "Для_запуска_стат" и может иметь соседний номер опыта (например 2.6.13).
    # Поэтому добираем кандидатов по наличию prediction CSV без повторного расчёта.
    for root in experiment_roots:
        root_path = _resolve(root)
        if not root_path.exists():
            continue
        prediction_files = list(root_path.rglob("test_predictions_best.csv")) + list(
            root_path.rglob("predictions_best.csv")
        )
        for pred_file in prediction_files:
            exp_dir = pred_file.parent
            exp_name = exp_dir.name
            if "phase_polar_stride_base_weights_aug" not in exp_name:
                continue
            if not (exp_name.startswith("Exp_2.6.12_") or exp_name.startswith("Exp_2.6.13_")):
                continue
            add_candidate(exp_name, exp_dir)

    selected: dict[str, pd.DataFrame] = {}
    source_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for model in MODEL_ORDER:
        required = FIG9_REQUIRED_COMPLEXITY[model]
        model_candidates = [c for c in candidates if c["model"] == model]
        if not model_candidates:
            missing.append(f"{model} ({required})")
            continue
        best = sorted(model_candidates, key=lambda c: float(c["selected_f1"]), reverse=True)[0]
        selected[model] = best["selected_df"]  # type: ignore[assignment]
        source_rows.append(
            {
                "model": model,
                "required_complexity": required,
                "experiment": best["experiment"],
                "source_dir": best["source_dir"],
                "selected_f1_from_predictions": best["selected_f1"],
            }
        )
    if missing:
        raise FileNotFoundError(
            "Не найдены сохранённые prediction CSV для требуемых моделей рисунка 9: "
            + ", ".join(missing)
        )
    return selected, pd.DataFrame(source_rows)


def _engineering_stats_by_model(selected_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, pred_df in selected_by_model.items():
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        for class_id, class_name in ENGINEERING_CLASS_MAP.items():
            true_mask = y_true == class_id
            pred_mask = y_pred == class_id
            tp = int(np.sum(true_mask & pred_mask))
            fp = int(np.sum(~true_mask & pred_mask))
            fn = int(np.sum(true_mask & ~pred_mask))
            gt = int(np.sum(true_mask))
            rows.append(
                {
                    "model": model,
                    "class_id": class_id,
                    "class_name": class_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "errors": fp + fn,
                    "gt": gt,
                }
            )
    return pd.DataFrame(rows)


def replot_article1_fig9_engineering_bars(
    selection_csv: str | Path,
    experiment_roots: Iterable[str | Path],
    output_dir: str | Path,
    figure_width: float = 10.5,
    figure_height: float = 6.0,
    legend_mode: str = "right",
    annotate_bar_numbers: bool = True,
    article_data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Строит инженерные столбцы по сохранённым CSV предсказаний."""
    sources = pd.DataFrame()
    stats = pd.DataFrame()
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    used_csv = output_path / "fig9_engineering_bars_stats_used.csv"
    sources_csv = output_path / "fig9_engineering_bars_sources_used.csv"

    if article_data_dir is not None:
        article_path = _resolve(article_data_dir)
        exported_stats = article_path / "engineering_stats_selected_models.csv"
        exported_sources = article_path / "selected_models_manifest.csv"
        if exported_stats.exists():
            stats = pd.read_csv(exported_stats)
            if exported_sources.exists():
                sources = pd.read_csv(exported_sources)

    if stats.empty:
        selected_by_model, sources = _load_selected_predictions_by_model(selection_csv, experiment_roots)
        if not selected_by_model:
            raise ValueError("Не удалось найти CSV предсказаний для рисунка 9.")
        stats = _engineering_stats_by_model(selected_by_model)

    stats.to_csv(used_csv, index=False)
    if not sources.empty:
        sources.to_csv(sources_csv, index=False)

    model_col = "Model" if "Model" in stats.columns else "model"
    models = [m for m in MODEL_ORDER if m in set(stats[model_col].astype(str))]
    class_ids = sorted(stats["class_id"].unique())
    class_labels = [str(stats[stats["class_id"] == c]["class_name"].iloc[0]) for c in class_ids]

    def plot_one(relative: bool, out_path: Path) -> None:
        x = np.arange(len(class_ids))
        width = 0.78 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))

        for idx, model in enumerate(models):
            st = stats[stats[model_col].astype(str) == model].sort_values("class_id")
            offset = (idx - (len(models) - 1) / 2.0) * width
            if relative:
                gt_safe = st["gt"].replace(0, np.nan)
                tp_vals = (st["tp"] / gt_safe * 100.0).fillna(0.0).to_numpy(dtype=float)
                err_vals = (st["errors"] / gt_safe * 100.0).fillna(0.0).to_numpy(dtype=float)
            else:
                tp_vals = st["tp"].to_numpy(dtype=float)
                err_vals = st["errors"].to_numpy(dtype=float)
            bars_up = ax.bar(
                x + offset,
                tp_vals,
                width=width * 0.94,
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.86,
                zorder=3,
            )
            ax.bar(
                x + offset,
                -err_vals,
                width=width * 0.94,
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.28,
                zorder=3,
            )
            if annotate_bar_numbers:
                y_pad = max(float(np.nanmax(tp_vals)), 1.0) * 0.025
                for bar, val in zip(bars_up, tp_vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + y_pad,
                        str(MODEL_ORDER.index(model) + 1),
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.4),
                        zorder=6,
                    )

        first = stats[stats[model_col].astype(str) == models[0]].sort_values("class_id")
        gt_vals = np.full(len(class_ids), 100.0) if relative else first["gt"].to_numpy(dtype=float)
        cluster_half = (len(models) * width) / 2.0
        for i, gt_val in enumerate(gt_vals):
            ax.hlines(
                y=gt_val,
                xmin=x[i] - cluster_half,
                xmax=x[i] + cluster_half,
                colors="black",
                linestyles="--",
                linewidth=2.0,
                alpha=0.9,
                zorder=5,
            )
            ax.scatter(x[i], gt_val, s=28, color="black", zorder=6)

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, fontsize=12)
        ax.set_ylabel("Доля от GT, %" if relative else "Количество окон", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)

        handles: list[object] = [
            Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Ground Truth"),
            Patch(facecolor="#777777", edgecolor="black", alpha=0.86, label="TP (вверх)"),
            Patch(facecolor="#777777", edgecolor="black", alpha=0.28, label="Ошибки (вниз)"),
            _legend_spacer(),
        ]
        handles.extend(
            Patch(
                facecolor=MODEL_COLORS[m],
                edgecolor="black",
                alpha=0.86,
                label=_numbered_model_label(m),
            )
            for m in models
        )

        legend_mode_local = legend_mode.lower().strip()
        if legend_mode_local == "right":
            ax.legend(
                handles=handles,
                title="Обозначения",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9.5,
                title_fontsize=10,
                frameon=True,
            )
            fig.tight_layout(rect=(0, 0, 0.82, 1))
        elif legend_mode_local == "bottom":
            ax.legend(
                handles=handles,
                title="Обозначения",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=4,
                fontsize=9,
                title_fontsize=10,
                frameon=True,
            )
            fig.tight_layout(rect=(0, 0.18, 1, 1))
        elif legend_mode_local == "none":
            fig.tight_layout()
        else:
            raise ValueError("legend_mode должен быть 'bottom', 'right' или 'none'")

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

    abs_path = output_path / "fig9a_engineering_bars_combined_abs_article.png"
    rel_path = output_path / "fig9b_engineering_bars_combined_rel_article.png"
    plot_one(relative=False, out_path=abs_path)
    plot_one(relative=True, out_path=rel_path)
    return {
        "bars_abs_png": abs_path,
        "bars_abs_svg": abs_path.with_suffix(".svg"),
        "bars_rel_png": rel_path,
        "bars_rel_svg": rel_path.with_suffix(".svg"),
        "stats_csv": used_csv,
        "sources_csv": sources_csv,
    }


def _resolve_marking_files_by_hash(
    data_dir: str | Path,
    split: str,
    old_exp_name: str,
    mark_hashes: Iterable[str],
) -> dict[str, str]:
    """Восстанавливает file_name по hash из mark_<hash>.png старого разметчика."""
    data_path = _resolve(data_dir)
    csv_name = "train.csv" if split.lower() == "train" else "test.csv"
    csv_path = data_path / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден датасет для поиска осциллограмм: {csv_path}")

    needed = {h.lower().replace("mark_", "").replace(".png", "") for h in mark_hashes}
    files = pd.read_csv(csv_path, usecols=["file_name"])["file_name"].dropna().astype(str).unique()

    resolved: dict[str, str] = {}
    for file_name in files:
        digest = hashlib.md5(f"{file_name}|{old_exp_name}".encode("utf-8")).hexdigest()[:12]
        if digest in needed:
            resolved[digest] = file_name

    missing = sorted(needed - set(resolved))
    if missing:
        raise FileNotFoundError(
            "Не удалось восстановить file_name для mark-hash: " + ", ".join(missing)
        )
    return resolved


def replot_article1_fig10_marking(
    old_exp_name: str,
    new_exp_name: str,
    data_dir: str | Path,
    output_dir: str | Path,
    mark_hashes: Iterable[str],
    split: str = "train",
    plot_mode: str = "discrete",
    threshold: float = 0.5,
    figure_width: float = 12.0,
    figure_height: float = 7.4,
    show_title: bool = False,
    file_time_ranges_ms: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Path]:
    """Строит две размеченные осциллограммы для рисунка 10 обновлённой моделью."""
    from scripts.evaluation.plot_model_marking import generate_marking_plots_for_model

    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_marking_files_by_hash(data_dir, split, old_exp_name, mark_hashes)
    selected_files = [resolved[h.lower().replace("mark_", "").replace(".png", "")] for h in mark_hashes]

    generated_root = output_path / "_generated"
    generate_marking_plots_for_model(
        exp_name=new_exp_name,
        output_dir=generated_root,
        data_dir=_resolve(data_dir),
        include_zero_current=True,
        include_zero_voltage=True,
        split=split,
        plot_mode=plot_mode,
        threshold=threshold,
        inference_backend="auto",
        selected_files=selected_files,
        file_time_ranges_ms=file_time_ranges_ms,
        figure_size=(figure_width, figure_height),
        dpi=300,
        signal_linewidth=1.55,
        label_fontsize=13,
        tick_fontsize=11,
        legend_fontsize=10,
        title_fontsize=13,
        marker_size=24,
        show_title=show_title,
    )

    generated_dir = generated_root / "marking_plots" / f"{new_exp_name}_{split}"
    manifest_rows: list[dict[str, str]] = []
    saved: dict[str, Path] = {}
    for idx, old_hash in enumerate(mark_hashes, start=1):
        normalized_hash = old_hash.lower().replace("mark_", "").replace(".png", "")
        file_name = resolved[normalized_hash]
        new_hash = hashlib.md5(f"{file_name}|{new_exp_name}".encode("utf-8")).hexdigest()[:12]
        src = generated_dir / f"mark_{new_hash}.png"
        if not src.exists():
            raise FileNotFoundError(f"Ожидался построенный PNG, но он не найден: {src}")

        suffix = "a" if idx == 1 else "b"
        dst = output_path / f"fig10{suffix}_marking_{normalized_hash}_article.png"
        shutil.copy2(src, dst)
        saved[f"fig10{suffix}_png"] = dst
        manifest_rows.append(
            {
                "panel": suffix,
                "old_hash": normalized_hash,
                "new_hash": new_hash,
                "file_name": file_name,
                "old_exp_name": old_exp_name,
                "new_exp_name": new_exp_name,
                "source_png": str(src),
                "article_png": str(dst),
            }
        )

    manifest_path = output_path / "fig10_marking_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    saved["manifest_csv"] = manifest_path
    return saved


def _find_experiment_dir_by_name(exp_name: str, experiment_roots: Iterable[str | Path]) -> Path:
    for root in experiment_roots:
        root_path = _resolve(root)
        direct = root_path / exp_name
        if direct.exists():
            return direct
        matches = [p for p in root_path.rglob(exp_name) if p.is_dir()] if root_path.exists() else []
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Эксперимент не найден: {exp_name}")


def _load_torch_model_for_article(exp_name: str, experiment_roots: Iterable[str | Path], weights: str = "best"):
    import importlib.util
    import torch

    model_utils_path = Path(__file__).resolve().parents[1] / "evaluation" / "_core" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("_article_model_utils", model_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить model_utils.py: {model_utils_path}")
    model_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_utils)
    _create_model_from_config = model_utils._create_model_from_config
    _load_state_dict_safe = model_utils._load_state_dict_safe

    exp_dir = _find_experiment_dir_by_name(exp_name, experiment_roots)
    with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model = _create_model_from_config(config)
    if model is None:
        raise ValueError(f"Не удалось создать модель из config.json: {exp_dir / 'config.json'}")

    candidates = [weights]
    candidates += ["best", "final"] if weights != "best" else ["final"]
    ckpt_path = None
    for name in candidates:
        path = exp_dir / f"{name}_model.pt"
        if path.exists():
            ckpt_path = path
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Не найден best/final checkpoint в {exp_dir}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_state_dict_safe(model, checkpoint, exp_dir.name, f"article_{weights}")
    model.eval()
    return model, exp_dir, ckpt_path


def _kan_linear_layers(model) -> list[tuple[str, object]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "base_weight")
        and hasattr(module, "spline_weight")
        and hasattr(module, "b_splines")
        and hasattr(module, "grid")
    ]


def _edge_score_matrix(layer) -> np.ndarray:
    import torch

    with torch.no_grad():
        spline = layer.scaled_spline_weight if hasattr(layer, "scaled_spline_weight") else layer.spline_weight
        score = torch.abs(layer.base_weight).detach().cpu().float()
        score = score + torch.mean(torch.abs(spline).detach().cpu().float(), dim=-1)
    return score.numpy()


def _select_kan_layer(
    layers: list[tuple[str, object]],
    layer_index: int = 0,
    layer_name_contains: str | None = None,
) -> tuple[str, object]:
    if layer_name_contains:
        for layer_name, layer in layers:
            if layer_name_contains in layer_name:
                return layer_name, layer
        available = ", ".join(name for name, _ in layers)
        raise ValueError(
            f"Не найден KAN-слой по фрагменту имени '{layer_name_contains}'. "
            f"Доступные слои: {available}"
        )
    return layers[min(layer_index, len(layers) - 1)]


def _edge_function_components(
    layer,
    out_idx: int,
    in_idx: int,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    x_tensor = torch.zeros((len(x), layer.in_features), dtype=torch.float32)
    x_tensor[:, in_idx] = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        base = layer.base_activation(x_tensor[:, in_idx : in_idx + 1]).squeeze(1)
        base = base * layer.base_weight[out_idx, in_idx].detach().cpu()
        bases = layer.b_splines(x_tensor)[:, in_idx, :].detach().cpu()
        spline_weight = layer.scaled_spline_weight if hasattr(layer, "scaled_spline_weight") else layer.spline_weight
        spline = bases @ spline_weight[out_idx, in_idx, :].detach().cpu()
        y = base + spline
    return y.numpy(), base.numpy(), spline.numpy()


def _edge_function(layer, out_idx: int, in_idx: int, x: np.ndarray) -> np.ndarray:
    y, _, _ = _edge_function_components(layer, out_idx, in_idx, x)
    return y


def _spline_shape_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | str]:
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    amp = float(np.max(y) - np.min(y)) if y.size else 0.0
    eps = 1e-9
    if amp < eps:
        return {
            "label": "нулевая",
            "amplitude": amp,
            "corr": 0.0,
            "nonlinearity": 0.0,
            "step_score": 0.0,
            "saturation": 0.0,
            "turns": 0.0,
            "crosses_zero": 0.0,
            "center_near_zero": 0.0,
            "balance": 0.0,
        }

    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > eps else 0.0
    if not np.isfinite(corr):
        corr = 0.0

    fit = np.polyval(np.polyfit(x, y, deg=1), x)
    nonlinearity = float(np.sqrt(np.mean((y - fit) ** 2)) / (amp + eps))
    max_abs = float(np.max(np.abs(y))) + eps
    crosses_zero = bool(np.min(y) <= 0.0 <= np.max(y))
    center_idx = int(np.argmin(np.abs(x)))
    center_near_zero = float(1.0 - min(1.0, abs(float(y[center_idx])) / max_abs))
    balance = float(1.0 - min(1.0, abs(float(np.mean(y))) / max_abs))

    dy = np.diff(y)
    abs_dy = np.abs(dy)
    step_score = float(np.max(abs_dy) / (np.sum(abs_dy) + eps)) if abs_dy.size else 0.0
    strong = np.sign(dy[np.abs(dy) > 0.05 * (np.max(abs_dy) + eps)])
    turns = float(np.sum(strong[1:] * strong[:-1] < 0)) if strong.size > 1 else 0.0

    n = max(3, len(dy) // 5)
    edge_slope = float((np.mean(abs_dy[:n]) + np.mean(abs_dy[-n:])) / 2.0) if abs_dy.size else 0.0
    mid = len(dy) // 2
    mid_slope = float(np.mean(abs_dy[max(0, mid - n // 2) : min(len(dy), mid + n // 2)])) if abs_dy.size else 0.0
    saturation = float(max(0.0, mid_slope - edge_slope) / (mid_slope + eps))

    if turns >= 3 and nonlinearity > 0.10:
        label = "волнообразная"
    elif saturation > 0.45 and abs(corr) > 0.35:
        label = "S/порог"
    elif step_score > 0.16 and nonlinearity > 0.08:
        label = "пороговая"
    elif nonlinearity < 0.10 and abs(corr) > 0.85:
        label = "линейная"
    elif abs(corr) > 0.65:
        label = "монотонная"
    else:
        label = "нелинейная"

    return {
        "label": label,
        "amplitude": amp,
        "corr": corr,
        "nonlinearity": nonlinearity,
        "step_score": step_score,
        "saturation": saturation,
        "turns": turns,
        "crosses_zero": float(crosses_zero),
        "center_near_zero": center_near_zero,
        "balance": balance,
    }


def _select_diverse_spline_edges(
    layers: list[tuple[str, object]],
    x: np.ndarray,
    top_n: int,
    candidate_pool_per_layer: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for layer_idx, (layer_name, layer) in enumerate(layers):
        scores = _edge_score_matrix(layer)
        flat_order = np.argsort(scores.reshape(-1))[::-1]
        if candidate_pool_per_layer and candidate_pool_per_layer > 0:
            flat_order = flat_order[:candidate_pool_per_layer]
        for flat_idx in flat_order:
            out_idx, in_idx = np.unravel_index(int(flat_idx), scores.shape)
            y = _edge_function(layer, int(out_idx), int(in_idx), x)
            metrics = _spline_shape_metrics(x, y)
            candidates.append(
                {
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "layer": layer,
                    "out_idx": int(out_idx),
                    "in_idx": int(in_idx),
                    "edge_score": float(scores[out_idx, in_idx]),
                    "y": y,
                    **metrics,
                }
            )

    if not candidates:
        return []

    chosen: list[dict[str, object]] = []
    used_edges: set[tuple[int, int, int]] = set()
    chosen_shapes: list[np.ndarray] = []

    def display_curve(candidate: dict[str, object]) -> np.ndarray:
        y = np.asarray(candidate["y"], dtype=float)
        y = y - float(np.mean(y))
        max_abs = float(np.max(np.abs(y)))
        return y / max_abs if max_abs > 1e-9 else y

    def shape_distance(candidate: dict[str, object]) -> float:
        y = display_curve(candidate)
        if not chosen_shapes:
            return 1.0
        distances = []
        for prev in chosen_shapes:
            if np.std(y) < 1e-9 or np.std(prev) < 1e-9:
                distances.append(0.0)
                continue
            corr = float(np.corrcoef(y, prev)[0, 1])
            if not np.isfinite(corr):
                corr = 1.0
            distances.append(1.0 - abs(corr))
        return float(min(distances))

    def visual_score(candidate: dict[str, object]) -> float:
        return (
            1.00 * float(candidate["nonlinearity"])
            + 0.35 * min(3.0, float(candidate["turns"]))
            + 0.45 * float(candidate["crosses_zero"])
            + 0.35 * float(candidate["center_near_zero"])
            + 0.25 * float(candidate["balance"])
            + 0.08 * np.log1p(max(0.0, float(candidate["edge_score"])))
        )

    def add_candidate(candidate: dict[str, object], min_shape_distance: float = 0.08) -> bool:
        key = (int(candidate["layer_idx"]), int(candidate["out_idx"]), int(candidate["in_idx"]))
        if key in used_edges or len(chosen) >= top_n:
            return False
        if chosen_shapes and shape_distance(candidate) < min_shape_distance:
            return False
        chosen.append(candidate)
        chosen_shapes.append(display_curve(candidate))
        used_edges.add(key)
        return True

    # Сначала набираем разные формы, а не только самые крупные коэффициенты.
    desired_labels = ["S/порог", "пороговая", "волнообразная", "линейная", "монотонная", "нелинейная"]
    for label in desired_labels:
        same_label = [c for c in candidates if c["label"] == label]
        if same_label:
            same_label.sort(key=visual_score, reverse=True)
            for candidate in same_label:
                if add_candidate(candidate):
                    break

    # Добираем оставшиеся наиболее выразительные кривые, штрафуя почти одинаковые формы.
    remaining = sorted(candidates, key=visual_score, reverse=True)
    for candidate in remaining:
        add_candidate(candidate)
        if len(chosen) >= top_n:
            break

    if len(chosen) < top_n:
        for candidate in remaining:
            add_candidate(candidate, min_shape_distance=0.0)
            if len(chosen) >= top_n:
                break

    return chosen


def replot_article1_fig11_kan_splines(
    exp_name: str,
    experiment_roots: Iterable[str | Path],
    output_dir: str | Path,
    weights: str = "best",
    layer_index: int = 0,
    layer_name_contains: str | None = "processing_net.features.4.kan_layer",
    top_n: int = 9,
    figure_width: float = 9.6,
    figure_height: float = 7.0,
    plot_components: bool = False,
    normalize_curves: bool = True,
    input_indices: Iterable[int] = (0, 1, 2),
    output_indices: Iterable[int] = (0, 1, 2),
    scan_all_layers: bool = True,
    candidate_pool_per_layer: int = 0,
) -> dict[str, Path]:
    """Рисунок 11: примеры KAN-функций на рёбрах.

    По умолчанию строится обзор разнообразных нелинейных функций с подписью
    формы; component-режим оставлен как вспомогательный диагностический.
    """
    model, exp_dir, ckpt_path = _load_torch_model_for_article(exp_name, experiment_roots, weights=weights)
    layers = _kan_linear_layers(model)
    if not layers:
        raise ValueError(f"В модели не найдены KANLinear-слои: {exp_name}")

    x = np.linspace(-1.15, 1.15, 240)

    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / "fig11_kan_splines_article.png"
    manifest = output_path / "fig11_kan_splines_manifest.csv"

    if plot_components:
        layer_name, layer = _select_kan_layer(layers, layer_index=layer_index, layer_name_contains=layer_name_contains)
        max_out, max_in = layer.base_weight.shape
        in_ids = [int(i) for i in input_indices if 0 <= int(i) < int(max_in)]
        out_ids = [int(i) for i in output_indices if 0 <= int(i) < int(max_out)]
        if not in_ids or not out_ids:
            raise ValueError(
                f"Для слоя {layer_name} недопустимые индексы: "
                f"input_indices={list(input_indices)}, output_indices={list(output_indices)}, "
                f"shape=({int(max_out)}, {int(max_in)})"
            )

        fig, axes = plt.subplots(
            len(out_ids),
            len(in_ids),
            figsize=(figure_width, figure_height),
            sharex=True,
        )
        axes_grid = np.array(axes, dtype=object).reshape(len(out_ids), len(in_ids))
        manifest_rows: list[dict[str, object]] = []
        edge_scores = _edge_score_matrix(layer)

        for row_idx, out_idx in enumerate(out_ids):
            for col_idx, in_idx in enumerate(in_ids):
                ax = axes_grid[row_idx, col_idx]
                y, base, spline = _edge_function_components(layer, out_idx, in_idx, x)
                ax.plot(x, y, color="#1f77b4", linewidth=2.0, label="Сумма")
                ax.plot(x, base, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Базовая (SiLU)")
                ax.plot(x, spline, color="#2ca02c", linewidth=1.4, linestyle=":", label="Сплайн")
                ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
                ax.grid(True, alpha=0.28, linestyle=":")
                ax.set_title(f"Вход {in_idx} -> Выход {out_idx}", fontsize=10.5)
                ax.tick_params(axis="both", labelsize=9.5)
                if col_idx == 0:
                    ax.set_ylabel(f"Выход {out_idx}", fontsize=10)
                if row_idx == len(out_ids) - 1:
                    ax.set_xlabel(f"Вход {in_idx}", fontsize=10)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=8.5, frameon=True, loc="best")

                manifest_rows.append(
                    {
                        "rank": len(manifest_rows) + 1,
                        "layer": layer_name,
                        "out_idx": out_idx,
                        "in_idx": in_idx,
                        "mode": "components_sum_base_spline",
                        "edge_score": float(edge_scores[out_idx, in_idx]),
                        "experiment": exp_name,
                        "checkpoint": str(ckpt_path),
                        "experiment_dir": str(exp_dir),
                    }
                )

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        return {"fig11_png": out_path, "fig11_svg": out_path.with_suffix(".svg"), "fig11_manifest_csv": manifest}

    selected_layers = layers if scan_all_layers else [layers[min(layer_index, len(layers) - 1)]]
    selected = _select_diverse_spline_edges(
        selected_layers,
        x=x,
        top_n=top_n,
        candidate_pool_per_layer=candidate_pool_per_layer,
    )
    if not selected:
        raise ValueError(f"Не удалось выбрать KAN-рёбра для визуализации: {exp_name}")

    ncols = 3
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, figure_height), sharex=True)
    axes_flat = np.array(axes).reshape(-1)
    linestyles = ["-", "--", "-.", ":"]

    for plot_idx, (ax, item) in enumerate(zip(axes_flat, selected), start=1):
        y = np.asarray(item["y"], dtype=float)
        if normalize_curves:
            y = y - float(np.mean(y))
            max_abs = float(np.max(np.abs(y)))
            if max_abs > 1e-9:
                y = y / max_abs
        ax.plot(
            x,
            y,
            color=MODEL_COLORS["PhysicsKAN"],
            linestyle=linestyles[(plot_idx - 1) % len(linestyles)],
            linewidth=2.0,
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.axvline(0, color="black", linewidth=0.6, alpha=0.25)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_title(
            f"{plot_idx}. {item['label']} | o={item['out_idx']}, i={item['in_idx']}",
            fontsize=9.5,
        )
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes_flat[len(selected) :]:
        ax.axis("off")

    fig.supxlabel("Вход ребра", fontsize=13)
    fig.supylabel("Значение функции", fontsize=13)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "rank": i + 1,
                "layer": str(item["layer_name"]),
                "out_idx": int(item["out_idx"]),
                "in_idx": int(item["in_idx"]),
                "shape_label": str(item["label"]),
                "edge_score": float(item["edge_score"]),
                "amplitude": float(item["amplitude"]),
                "corr": float(item["corr"]),
                "nonlinearity": float(item["nonlinearity"]),
                "step_score": float(item["step_score"]),
                "saturation": float(item["saturation"]),
                "turns": float(item["turns"]),
                "crosses_zero": float(item["crosses_zero"]),
                "center_near_zero": float(item["center_near_zero"]),
                "balance": float(item["balance"]),
                "display_normalized": bool(normalize_curves),
                "experiment": exp_name,
                "checkpoint": str(ckpt_path),
                "experiment_dir": str(exp_dir),
            }
            for i, item in enumerate(selected)
        ]
    ).to_csv(manifest, index=False)
    return {"fig11_png": out_path, "fig11_svg": out_path.with_suffix(".svg"), "fig11_manifest_csv": manifest}


def replot_article1_fig12_kan_heatmap(
    exp_name: str,
    experiment_roots: Iterable[str | Path],
    output_dir: str | Path,
    weights: str = "best",
    layer_index: int = 0,
    layer_name_contains: str | None = "processing_net.features.4.kan_layer",
    figure_width: float = 8.8,
    figure_height: float = 6.2,
    show_weak_share: bool = False,
    use_pruning_importance: bool = False,
    reference_pruning_report_json: str | Path | None = None,
    rescale_to_reference_max: bool = True,
    data_dir: str | Path = "data/ml_datasets",
    max_importance_windows: int = -1,
    eval_stride: int = 1,
    importance_batch_size: int = 64,
    importance_max_batches: int = 5,
    importance_max_samples: int = 1000,
) -> dict[str, Path]:
    """Рисунок 12: тепловая карта активности рёбер выбранного KAN-слоя."""
    model, exp_dir, ckpt_path = _load_torch_model_for_article(exp_name, experiment_roots, weights=weights)
    layers = _kan_linear_layers(model)
    if not layers:
        raise ValueError(f"В модели не найдены KANLinear-слои: {exp_name}")

    layer_name, layer = _select_kan_layer(layers, layer_index=layer_index, layer_name_contains=layer_name_contains)
    scores_source = "edge_weight_activity"
    scores = _edge_score_matrix(layer)
    if use_pruning_importance:
        try:
            import torch
            from torch.utils.data import DataLoader

            from osc_tools.ml.kan_pruning import calculate_kan_importance, collect_kan_inputs
            from scripts.phase2_experiments.run_phase2_6_kan_pruning import _build_dataset

            with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            device = torch.device("cpu")
            model = model.to(device)
            ds_importance, _, _ = _build_dataset(
                config=config,
                exp_name=exp_name,
                data_dir=_resolve(data_dir),
                max_windows=max_importance_windows,
                eval_stride=eval_stride,
            )
            loader = DataLoader(
                ds_importance,
                batch_size=importance_batch_size,
                shuffle=False,
                num_workers=0,
            )
            inputs = collect_kan_inputs(
                model,
                loader,
                device,
                max_batches=importance_max_batches,
                max_samples=importance_max_samples,
            )
            importances = calculate_kan_importance(model, inputs, device)
            if layer_name in importances:
                scores = importances[layer_name].detach().cpu().numpy()
                scores_source = "mean_abs_phi_on_dataset"
        except Exception as exc:
            print(f"[!] Рисунок 12: не удалось посчитать pruning-importance, fallback на веса: {exc}")

    if reference_pruning_report_json and rescale_to_reference_max and scores.size and np.max(scores) > 0:
        report_path = _resolve(reference_pruning_report_json)
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            ref_stats = report.get("importance_stats", {}).get(layer_name)
            if ref_stats and float(ref_stats.get("max", 0.0)) > 0:
                ref_max = float(ref_stats["max"])
                scores = scores * (ref_max / float(np.max(scores)))
                scores_source = f"{scores_source}_scaled_to_reference_max"
    weak_threshold = float(np.max(scores) * 0.05) if scores.size else 0.0
    weak_share = float(np.mean(scores <= weak_threshold)) if scores.size else 0.0

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    im = ax.imshow(scores, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Важность |phi(x)|", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Вход", fontsize=13)
    ax.set_ylabel("Выход", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    if show_weak_share:
        ax.text(
            0.02,
            0.98,
            f"Слабые связи (<5% max): {weak_share:.0%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8, alpha=0.85),
        )
    fig.tight_layout()

    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / "fig12_kan_edge_activity_heatmap_article.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    csv_path = output_path / "fig12_kan_edge_activity_matrix.csv"
    pd.DataFrame(scores).to_csv(csv_path, index=False)
    manifest = output_path / "fig12_kan_heatmap_manifest.csv"
    pd.DataFrame(
        [
            {
                "layer": layer_name,
                "experiment": exp_name,
                "checkpoint": str(ckpt_path),
                "experiment_dir": str(exp_dir),
                "rows": scores.shape[0],
                "cols": scores.shape[1],
                "scores_source": scores_source,
                "weak_threshold_5pct_max": weak_threshold,
                "weak_share_5pct_max": weak_share,
            }
        ]
    ).to_csv(manifest, index=False)
    return {
        "fig12_png": out_path,
        "fig12_svg": out_path.with_suffix(".svg"),
        "fig12_matrix_csv": csv_path,
        "fig12_manifest_csv": manifest,
    }


def replot_article1_fig13_ablation(
    pruning_report_json: str | Path,
    output_dir: str | Path,
    figure_width: float = 7.6,
    figure_height: float = 3.5,
) -> dict[str, Path]:
    """Рисунок 13: влияние отключения физических слоёв на F1-Macro."""
    report_path = _resolve(pruning_report_json)
    if not report_path.exists():
        raise FileNotFoundError(f"Не найден pruning/ablation report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    ablations = report.get("ablation_results", {})
    if not ablations:
        raise ValueError(f"В отчёте нет ablation_results: {report_path}")

    label_map = {
        "baseline": "Базовая",
        "no_mult": "Без S=U*I",
        "no_div": "Без Y=I/U",
        "no_arith": "Без S,Y",
        "only_arith": "Только S,Y",
    }
    order = [k for k in ["baseline", "no_mult", "no_div", "no_arith", "only_arith"] if k in ablations]
    values = [float(ablations[k].get("f1", 0.0)) for k in order]
    labels = [label_map.get(k, k) for k in order]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    x = np.arange(len(order))
    hatches = ["", "//", "\\\\", "xx", ".."]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    bars = ax.bar(
        x,
        values,
        color=colors[: len(order)],
        edgecolor="black",
        linewidth=1.1,
        width=0.68,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for idx, (bar, value) in enumerate(zip(bars, values), start=1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{idx}\n{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}. {label}" for i, label in enumerate(labels, start=1)], fontsize=11)
    ax.set_ylabel("F1-Macro", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, min(1.0, max(values) + 0.12))
    ax.grid(True, axis="y", alpha=0.28, linestyle=":", zorder=0)
    fig.tight_layout()

    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / "fig13_physics_ablation_article.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    csv_path = output_path / "fig13_physics_ablation_values.csv"
    pd.DataFrame(
        [
            {"variant": key, "label": label, "f1_macro": value, "source_report": str(report_path)}
            for key, label, value in zip(order, labels, values)
        ]
    ).to_csv(csv_path, index=False)
    return {"fig13_png": out_path, "fig13_svg": out_path.with_suffix(".svg"), "fig13_values_csv": csv_path}


if __name__ == "__main__":
    # =====================================================================
    # РУЧНОЙ ЗАПУСК ОТДЕЛЬНЫХ РИСУНКОВ ПЕРВОЙ СТАТЬИ
    # =====================================================================
    REPORT_ROOT = Path("reports/Exp_2_5_and_start_Exp_2_6")
    CANONICAL_ARTICLE1_REPORT_ROOT = REPORT_ROOT / "_Память" / "Опыт 2.6.12 (первая попытка)"
    SOURCE_SUMMARY = CANONICAL_ARTICLE1_REPORT_ROOT / "summary_aggregated.csv"
    SOURCE_SELECTION = CANONICAL_ARTICLE1_REPORT_ROOT / "selected_best_final_by_predictions.csv"
    ARTICLE1_OUTPUT_ROOT = REPORT_ROOT / "figures_article1"
    ARTICLE_DATA_DIR = REPORT_ROOT / "article_plot_data"

    # Для рисунка 9 основной источник - ARTICLE_DATA_DIR/engineering_stats_selected_models.csv.
    # Эти корни сканируются только как резерв, если готовой таблицы нет.
    EXPERIMENT_ROOTS = [
        "experiments/Для_запуска_стат",
        "experiments/phase2_6",
    ]

    # Рисунок 7 уже доведён. Чтобы перестроить его заново, поставьте True.
    RUN_FIG7_PARETO = False
    RUN_FIG8_RADAR = False
    # Текущие prediction CSV были перезаписаны и не совпадают со старым отчётом
    # 2.6.12; не включайте, пока не восстановлены исходные prediction CSV.
    RUN_FIG9_ENGINEERING_BARS = False
    RUN_FIG10_MARKING = False
    RUN_FIG11_KAN_SPLINES = True
    RUN_FIG12_KAN_HEATMAP = True
    RUN_FIG13_PHYSICS_ABLATION = True

    PHYSICSKAN_INTERPRET_EXP = "Exp_2.6.1_PhysicsKAN_medium_phase_polar_stride_base_weights_aug"
    PHYSICSKAN_PRUNING_REPORT = (
        Path("reports/phase2_6/exp_2_6_5")
        / f"{PHYSICSKAN_INTERPRET_EXP}_pruning_report.json"
    )

    saved: dict[str, Path] = {}

    if RUN_FIG7_PARETO:
        # Размер фигуры в дюймах. Увеличивайте figure_height, если график
        # после вставки в две колонки выглядит слишком плоским.
        saved.update(
            replot_article1_fig7_pareto(
                summary_csv=SOURCE_SUMMARY,
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig7_pareto",
                figure_width=9.6,
                figure_height=5.5,
                show_panel_titles=False,
                legend_mode="right",
                annotate_pareto=False,
            )
        )

    if RUN_FIG8_RADAR:
        saved.update(
            replot_article1_fig8_radar(
                summary_csv=SOURCE_SUMMARY,
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig8_radar",
                figure_width=8.4,
                figure_height=7.2,
                legend_mode="bottom",
                annotate_line_numbers=False,
                use_fixed_complexity_summary=True,
            )
        )

    if RUN_FIG9_ENGINEERING_BARS:
        try:
            saved.update(
                replot_article1_fig9_engineering_bars(
                    selection_csv=SOURCE_SELECTION,
                    experiment_roots=EXPERIMENT_ROOTS,
                    output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig9_engineering_bars",
                    figure_width=10.8,
                    figure_height=4.5,
                    legend_mode="right",
                    annotate_bar_numbers=True,
                    article_data_dir=None,
                )
            )
        except FileNotFoundError as exc:
            print(f"[!] Рисунок 9 не перестроен: {exc}")
            print(
                "    Нужны test_predictions_best.csv/test_predictions_final.csv "
                "для точного набора лучших моделей класса."
            )

    if RUN_FIG10_MARKING:
        saved.update(
            replot_article1_fig10_marking(
                old_exp_name="Exp_2.6.9_cPhysicsKAN_heavy_phase_polar_stride_base_weights_aug",
                new_exp_name="Exp_2.6.12_rPhysicsKAN_heavy_phase_polar_stride_base_weights_aug",
                data_dir="data/ml_datasets",
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig10_marking",
                mark_hashes=[
                    "0d50f79a162f",
                    "7a8bd50023cd",
                ],
                split="train",
                plot_mode="discrete",
                threshold=0.7,
                figure_width=12.0,
                figure_height=10,
                show_title=False,
            )
        )

    if RUN_FIG11_KAN_SPLINES:
        saved.update(
            replot_article1_fig11_kan_splines(
                exp_name=PHYSICSKAN_INTERPRET_EXP,
                experiment_roots=EXPERIMENT_ROOTS + ["experiments/phase2_5"],
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig11_kan_splines",
                weights="best",
                layer_index=0,
                layer_name_contains="processing_net.features.4.kan_layer",
                top_n=9,
                figure_width=8.8,
                figure_height=7.0,
                plot_components=False,
                normalize_curves=True,
                input_indices=(0, 1, 2),
                output_indices=(0, 1, 2),
                scan_all_layers=True,
                candidate_pool_per_layer=0,
            )
        )

    if RUN_FIG12_KAN_HEATMAP:
        saved.update(
            replot_article1_fig12_kan_heatmap(
                exp_name=PHYSICSKAN_INTERPRET_EXP,
                experiment_roots=EXPERIMENT_ROOTS + ["experiments/phase2_5"],
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig12_kan_heatmap",
                weights="best",
                layer_index=0,
                layer_name_contains="processing_net.features.4.kan_layer",
                figure_width=8.8,
                figure_height=6.2,
                use_pruning_importance=False,
                reference_pruning_report_json=PHYSICSKAN_PRUNING_REPORT,
                rescale_to_reference_max=True,
            )
        )

    if RUN_FIG13_PHYSICS_ABLATION:
        saved.update(
            replot_article1_fig13_ablation(
                pruning_report_json=PHYSICSKAN_PRUNING_REPORT,
                output_dir=Path(ARTICLE1_OUTPUT_ROOT) / "fig13_physics_ablation",
                figure_width=7.6,
                figure_height=4.8,
            )
        )

    print("Готово. Сохранены файлы:")
    for key, path in saved.items():
        print(f"  {key}: {path}")

    # Если нужно сразу экспортировать в папку статьи, раскомментируйте и
    # укажите путь. По умолчанию не включено, чтобы не писать вне проекта.
    #
    # import shutil
    # ARTICLE_FIGURES_DIR = Path(
    #     r"D:/Учебное/Аспирантура/4) Научные исследования/2) Публикация статей/"
    #     r"2026-2027/Широкие опыты по нейронкам/Рисунки, статья 1"
    # )
    # ARTICLE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # if "cpu_png" in saved:
    #     shutil.copy2(saved["cpu_png"], ARTICLE_FIGURES_DIR / "Рис.7.а_pareto_full_f1_cpu_article.png")
    # if "params_png" in saved:
    #     shutil.copy2(saved["params_png"], ARTICLE_FIGURES_DIR / "Рис.7.б_pareto_full_f1_params_article.png")
    # if "radar_png" in saved:
    #     shutil.copy2(saved["radar_png"], ARTICLE_FIGURES_DIR / "Рис.8_radar_by_model_type_article.png")
    # if "bars_abs_png" in saved:
    #     shutil.copy2(saved["bars_abs_png"], ARTICLE_FIGURES_DIR / "Рис.9.а_engineering_bars_abs_article.png")
    # if "bars_rel_png" in saved:
    #     shutil.copy2(saved["bars_rel_png"], ARTICLE_FIGURES_DIR / "Рис.9.б_engineering_bars_rel_article.png")
    # if "fig11_png" in saved:
    #     shutil.copy2(saved["fig11_png"], ARTICLE_FIGURES_DIR / "Рис.11_kan_splines_article.png")
    # if "fig12_png" in saved:
    #     shutil.copy2(saved["fig12_png"], ARTICLE_FIGURES_DIR / "Рис.12_kan_heatmap_article.png")
    # if "fig13_png" in saved:
    #     shutil.copy2(saved["fig13_png"], ARTICLE_FIGURES_DIR / "Рис.13_physics_ablation_article.png")
