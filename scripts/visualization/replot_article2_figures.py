from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Iterable
import hashlib

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib_cache"))
(ROOT_DIR / ".matplotlib_cache").mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODEL_ORDER = [
    "CNN",
    "ConvKAN",
    "MLP",
    "PhysicsKAN",
    "ResNet",
    "SimpleKAN",
    "cPhysicsKAN",
    "rPhysicsKAN",
]

FEATURE_ORDER = [
    "AB",
    "PhasePolar",
    "PhaseRect",
    "Power",
    "Raw",
    "Symmetric",
]

FEATURE_COLORS = {
    "PhasePolar": "#1f77b4",
    "Symmetric": "#9467bd",
    "AB": "#d99a24",
    "PhaseRect": "#2a9d8f",
    "Raw": "#3fb56f",
    "Power": "#d95a4e",
}

FEATURE_HATCHES = {
    "PhasePolar": "",
    "Symmetric": "//",
    "AB": "\\\\",
    "PhaseRect": "xx",
    "Raw": "..",
    "Power": "--",
}

OZZ_MODEL_ORDER = [
    "ConvKAN",
    "PhysicsKAN",
    "cPhysicsKAN",
    "PhysicsBaseline",
    "CNN",
    "SimpleKAN",
    "MLP",
    "ResNet",
]

OZZ_MODEL_COLORS = {
    "ConvKAN": "#1f77b4",
    "PhysicsKAN": "#ff7f0e",
    "cPhysicsKAN": "#2ca02c",
    "PhysicsBaseline": "#d62728",
    "CNN": "#9467bd",
    "SimpleKAN": "#8c564b",
    "MLP": "#e377c2",
    "ResNet": "#7f7f7f",
}

OZZ_CLASS_ORDER = [
    "ОЗЗ (обнаружение)",
    "Затухающее ОЗЗ",
    "ДПОЗЗ",
]

ARTICLE2_FIG3_PANELS = [
    {
        "panel": "a",
        "file_name": "b700c5c625f7c4d3c78faf197b7542a4_Bus 1 _event N1",
        "description": "ОЗЗ с пробоем и последующее затухающее ОЗЗ",
    },
    {
        "panel": "b",
        "file_name": "524e846b63557e667fc44b3e8bcb5c8e_Bus 2 _event N2",
        "description": "ДПОЗЗ",
    },
]

ARTICLE2_FIG3_CONVKAN_EXP = "Exp_2.6.11_ConvKAN_heavy_phase_polar_stride_ozz_win_any_weights_aug"
ARTICLE2_FIG3_EXPERIMENT_ZIP = Path("experiments/phase2_6 (после выпуска статей 1 и 2 - удалить).zip")

# В текущем дереве для опыта 2.6.11 не сохранены prediction-CSV,
# поэтому статистика восстановлена из старых cm_*_abs.png.
LEGACY_OZZ_CONFUSION_STATS = [
    ("ConvKAN", "ОЗЗ (обнаружение)", 202220, 1290, 964, 5305),
    ("ConvKAN", "Затухающее ОЗЗ", 202024, 5553, 751, 1451),
    ("ConvKAN", "ДПОЗЗ", 203833, 4471, 4, 1471),
    ("PhysicsKAN", "ОЗЗ (обнаружение)", 202177, 1333, 1171, 5098),
    ("PhysicsKAN", "Затухающее ОЗЗ", 203131, 4446, 1102, 1100),
    ("PhysicsKAN", "ДПОЗЗ", 204223, 4081, 7, 1468),
    ("cPhysicsKAN", "ОЗЗ (обнаружение)", 202295, 1215, 1263, 5006),
    ("cPhysicsKAN", "Затухающее ОЗЗ", 202278, 5299, 1115, 1087),
    ("cPhysicsKAN", "ДПОЗЗ", 204635, 3669, 128, 1347),
    ("PhysicsBaseline", "ОЗЗ (обнаружение)", 199178, 2418, 842, 7341),
    ("PhysicsBaseline", "Затухающее ОЗЗ", 204525, 2414, 2840, 0),
    ("PhysicsBaseline", "ДПОЗЗ", 203118, 4548, 35, 2078),
    ("CNN", "ОЗЗ (обнаружение)", 197593, 5917, 458, 5811),
    ("CNN", "Затухающее ОЗЗ", 196957, 10620, 362, 1840),
    ("CNN", "ДПОЗЗ", 201369, 6935, 0, 1475),
    ("SimpleKAN", "ОЗЗ (обнаружение)", 197627, 5883, 1149, 5120),
    ("SimpleKAN", "Затухающее ОЗЗ", 197039, 10538, 1050, 1152),
    ("SimpleKAN", "ДПОЗЗ", 202292, 6012, 31, 1444),
    ("MLP", "ОЗЗ (обнаружение)", 193522, 9988, 1203, 5066),
    ("MLP", "Затухающее ОЗЗ", 190010, 17567, 896, 1306),
    ("MLP", "ДПОЗЗ", 198924, 9380, 27, 1448),
    ("ResNet", "ОЗЗ (обнаружение)", 188022, 15488, 645, 5624),
    ("ResNet", "Затухающее ОЗЗ", 180539, 27038, 443, 1759),
    ("ResNet", "ДПОЗЗ", 192563, 15741, 15, 1460),
]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _best_full_f1(row: pd.Series) -> float:
    full_candidates = [
        row.get("Selected Test F1", np.nan),
        row.get("Full Best F1", np.nan),
        row.get("Full Final F1", np.nan),
    ]
    full_vals = pd.to_numeric(pd.Series(full_candidates), errors="coerce").dropna()
    if not full_vals.empty:
        return float(full_vals.max())

    val = pd.to_numeric(pd.Series([row.get("Val F1", np.nan)]), errors="coerce").dropna()
    return float(val.iloc[0]) if not val.empty else np.nan


def _prepare_summary(summary_csv: str | Path) -> pd.DataFrame:
    path = _resolve(summary_csv)
    if not path.exists():
        raise FileNotFoundError(f"Не найден summary CSV: {path}")

    df = pd.read_csv(path)
    required = {"Model", "Features"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} нет обязательных колонок: {sorted(missing)}")

    df = df.copy()
    df["Article F1"] = df.apply(_best_full_f1, axis=1)
    df = df.dropna(subset=["Article F1", "Model", "Features"])
    df = df[df["Features"].astype(str).str.lower() != "unknown"]
    return df


def _ordered_present(values: Iterable[str], preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(str(v) for v in values))
    ordered = [v for v in preferred if v in present]
    ordered.extend(sorted(v for v in present if v not in ordered))
    return ordered


def _heatmap_text_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "black"
    normalized = (value - vmin) / (vmax - vmin)
    return "white" if normalized > 0.68 else "black"


def replot_article2_fig1_feature_comparison(
    summary_csv: str | Path,
    output_dir: str | Path,
    figure_width_heatmap: float = 9.2,
    figure_height_heatmap: float = 5.6,
    figure_width_boxplot: float = 9.2,
    figure_height_boxplot: float = 5.2,
    show_titles: bool = False,
) -> dict[str, Path]:
    """Рисунок 1 статьи 2: сравнение F1-Macro по типам входных данных."""
    df = _prepare_summary(summary_csv)
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = _ordered_present(df["Model"].unique(), MODEL_ORDER)
    features = _ordered_present(df["Features"].unique(), FEATURE_ORDER)

    heat = (
        df.pivot_table(index="Model", columns="Features", values="Article F1", aggfunc="max")
        .reindex(index=models, columns=features)
    )

    vmin = max(0.0, float(np.nanmin(heat.to_numpy())) - 0.025)
    vmax = min(1.0, float(np.nanmax(heat.to_numpy())) + 0.025)

    fig, ax = plt.subplots(figsize=(figure_width_heatmap, figure_height_heatmap))
    im = ax.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("F1-Macro (лучшее)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(features, fontsize=11)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Тип данных", fontsize=13)
    ax.set_ylabel("Модель", fontsize=13)
    if show_titles:
        ax.set_title("а) Лучшие значения F1-Macro по типам данных", fontsize=14, pad=10)

    ax.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    values = heat.to_numpy(dtype=float)
    row_max = np.nanmax(values, axis=1)
    global_max = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                ax.text(j, i, "н/д", ha="center", va="center", fontsize=10, color="black")
                continue
            is_row_best = np.isclose(value, row_max[i])
            is_global_best = np.isclose(value, global_max)
            label = f"{value:.3f}" + ("*" if is_row_best else "")
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold" if is_row_best else "normal",
                color=_heatmap_text_color(float(value), vmin, vmax),
            )
            if is_global_best:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                    )
                )

    ax.text(
        0.0,
        -0.13,
        "* лучший тип данных для данной модели; рамка - максимум на всей карте",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
    )
    fig.tight_layout()

    heat_png = output_path / "fig1a_feature_heatmap_article2.png"
    fig.savefig(heat_png, dpi=300, bbox_inches="tight")
    fig.savefig(heat_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    heat_csv = output_path / "fig1a_feature_heatmap_values.csv"
    heat.to_csv(heat_csv)

    medians = df.groupby("Features")["Article F1"].median().sort_values(ascending=False)
    box_features = [f for f in medians.index if f in features]
    data = [df.loc[df["Features"] == feature, "Article F1"].to_numpy(dtype=float) for feature in box_features]

    fig, ax = plt.subplots(figsize=(figure_width_boxplot, figure_height_boxplot))
    positions = np.arange(1, len(box_features) + 1)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": "black", "linewidth": 1.5},
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5.5,
        },
        boxprops={"edgecolor": "black", "linewidth": 1.15},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
            "alpha": 0.85,
        },
    )

    for patch, feature in zip(box["boxes"], box_features):
        patch.set_facecolor(FEATURE_COLORS.get(feature, "#bdbdbd"))
        patch.set_alpha(0.82)
        patch.set_hatch(FEATURE_HATCHES.get(feature, ""))

    rng = np.random.default_rng(42)
    for pos, feature, values_feature in zip(positions, box_features, data):
        jitter = rng.normal(0.0, 0.045, size=len(values_feature))
        ax.scatter(
            np.full(len(values_feature), pos) + jitter,
            values_feature,
            s=16,
            color="black",
            alpha=0.42,
            linewidths=0,
            zorder=3,
        )
        median = float(np.median(values_feature))
        ax.text(pos, median + 0.012, f"{median:.3f}", ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(box_features, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("F1-Macro (лучшее)", fontsize=13)
    ax.set_xlabel("Тип данных", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.28, linestyle=":")
    if show_titles:
        ax.set_title("б) Разброс F1-Macro по типам данных", fontsize=14, pad=10)

    y_min = max(0.0, float(np.nanmin(df["Article F1"])) - 0.04)
    y_max = min(1.0, float(np.nanmax(df["Article F1"])) + 0.04)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    box_png = output_path / "fig1b_feature_boxplot_article2.png"
    fig.savefig(box_png, dpi=300, bbox_inches="tight")
    fig.savefig(box_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    box_csv = output_path / "fig1b_feature_boxplot_values.csv"
    df[["Model", "Complexity", "Features", "Sampling", "Article F1"]].to_csv(box_csv, index=False)

    manifest = output_path / "fig1_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_csv": str(_resolve(summary_csv)),
                "rows": len(df),
                "features": ", ".join(features),
                "models": ", ".join(models),
                "metric": "max(Selected Test F1, Full Best F1, Full Final F1); Val F1 only if full metrics are absent",
                "fig1a_png": str(heat_png),
                "fig1b_png": str(box_png),
            }
        ]
    ).to_csv(manifest, index=False)

    return {
        "fig1a_png": heat_png,
        "fig1a_svg": heat_png.with_suffix(".svg"),
        "fig1a_values_csv": heat_csv,
        "fig1b_png": box_png,
        "fig1b_svg": box_png.with_suffix(".svg"),
        "fig1b_values_csv": box_csv,
        "manifest_csv": manifest,
    }


def _legacy_ozz_stats_df() -> pd.DataFrame:
    rows = []
    model_no = {model: idx + 1 for idx, model in enumerate(OZZ_MODEL_ORDER)}
    for model, class_name, tn, fp, fn, tp in LEGACY_OZZ_CONFUSION_STATS:
        gt = fn + tp
        errors = fp + fn
        rows.append(
            {
                "model_no": model_no[model],
                "model": model,
                "class_name": class_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "gt": int(gt),
                "errors": int(errors),
                "tp_percent_of_gt": 100.0 * tp / gt if gt else np.nan,
                "errors_percent_of_gt": 100.0 * errors / gt if gt else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=OZZ_MODEL_ORDER, ordered=True)
    df["class_name"] = pd.Categorical(df["class_name"], categories=OZZ_CLASS_ORDER, ordered=True)
    return df.sort_values(["class_name", "model"]).reset_index(drop=True)


def _plot_ozz_bidirectional_bars(
    stats: pd.DataFrame,
    output_png: Path,
    relative: bool,
    figure_width: float,
    figure_height: float,
    show_title: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    classes = OZZ_CLASS_ORDER
    models = OZZ_MODEL_ORDER
    x = np.arange(len(classes), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_width

    for model_idx, model in enumerate(models):
        model_df = stats[stats["model"] == model].set_index("class_name")
        if relative:
            tp_vals = [float(model_df.loc[class_name, "tp_percent_of_gt"]) for class_name in classes]
            err_vals = [-float(model_df.loc[class_name, "errors_percent_of_gt"]) for class_name in classes]
        else:
            tp_vals = [float(model_df.loc[class_name, "tp"]) for class_name in classes]
            err_vals = [-float(model_df.loc[class_name, "errors"]) for class_name in classes]

        xpos = x + offsets[model_idx]
        color = OZZ_MODEL_COLORS[model]
        ax.bar(
            xpos,
            tp_vals,
            width=bar_width * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            zorder=3,
        )
        ax.bar(
            xpos,
            err_vals,
            width=bar_width * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.30,
            zorder=3,
        )

        for xi, down in zip(xpos, err_vals):
            ax.text(
                xi,
                down - (11 if relative else 720),
                str(model_idx + 1),
                ha="center",
                va="top",
                fontsize=8.4,
                color="black",
                alpha=0.72,
            )

    gt_values = []
    reference_model = models[0]
    reference_df = stats[stats["model"] == reference_model].set_index("class_name")
    for class_name in classes:
        gt_values.append(100.0 if relative else float(reference_df.loc[class_name, "gt"]))

    cluster_half = group_width / 2.0
    for class_x, gt_value in zip(x, gt_values):
        ax.hlines(
            y=gt_value,
            xmin=class_x - cluster_half,
            xmax=class_x + cluster_half,
            colors="black",
            linestyles=(0, (4, 2)),
            linewidth=1.9,
            zorder=4,
        )
        ax.scatter(
            class_x,
            gt_value,
            s=24,
            color="black",
            zorder=5,
        )

    ax.axhline(0, color="black", linewidth=1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.36, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["ОЗЗ", "Затухающее\nОЗЗ", "ДПОЗЗ"], fontsize=11)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylabel("Доля от GT, %" if relative else "Количество окон", fontsize=12.5)
    ax.set_xlabel("Тип события ОЗЗ", fontsize=12.5)
    if show_title:
        ax.set_title(
            "Двунаправленная гистограмма предсказаний ОЗЗ"
            + (" (относительные величины)" if relative else " (абсолютные значения)"),
            fontsize=13.5,
            pad=9,
        )

    all_up = np.array(gt_values + [stats["tp_percent_of_gt"].max() if relative else stats["tp"].max()], dtype=float)
    all_down = np.array(
        [
            -stats["errors_percent_of_gt"].max() if relative else -stats["errors"].max(),
        ],
        dtype=float,
    )
    ax.set_ylim(float(all_down.min()) * 1.13, float(all_up.max()) * (1.20 if relative else 1.16))

    mode_handles = [
        Patch(facecolor="#595959", edgecolor="black", alpha=0.92, label="TP (вверх)"),
        Patch(facecolor="#595959", edgecolor="black", alpha=0.30, label="ошибки FP+FN (вниз)"),
        Line2D([0], [0], color="black", linestyle=(0, (4, 2)), marker="o", linewidth=1.55, label="GT"),
    ]
    model_handles = [
        Patch(
            facecolor=OZZ_MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.45,
            label=f"{idx + 1}. {model}",
        )
        for idx, model in enumerate(models)
    ]

    first_legend = ax.legend(
        handles=mode_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.24),
        ncol=3,
        fontsize=9.4,
        frameon=True,
        title="Столбцы:",
        title_fontsize=9.6,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=model_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.24),
        ncol=4,
        fontsize=8.8,
        frameon=True,
        title="Номера моделей:",
        title_fontsize=9.6,
        columnspacing=0.9,
        handletextpad=0.45,
    )

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def replot_article2_fig2_ozz_bidirectional_bars(
    output_dir: str | Path,
    figure_width: float = 10.8,
    figure_height: float = 6.2,
    show_titles: bool = False,
) -> dict[str, Path]:
    """Рисунок 2 статьи 2: двунаправленные столбцы TP и ошибок для ОЗЗ."""
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = _legacy_ozz_stats_df()
    stats_csv = output_path / "fig2_ozz_bidirectional_stats_used.csv"
    stats.to_csv(stats_csv, index=False)

    abs_png = output_path / "fig2a_ozz_bidirectional_abs_article2.png"
    rel_png = output_path / "fig2b_ozz_bidirectional_rel_article2.png"
    _plot_ozz_bidirectional_bars(stats, abs_png, False, figure_width, figure_height, show_titles)
    _plot_ozz_bidirectional_bars(stats, rel_png, True, figure_width, figure_height, show_titles)

    manifest = output_path / "fig2_manifest.csv"
    pd.DataFrame(
        [
            {
                "source": "reconstructed from reports/.../_Память/Опыт 2.6.11/engineering_plots/cm_*_abs.png",
                "models": ", ".join(OZZ_MODEL_ORDER),
                "classes": ", ".join(OZZ_CLASS_ORDER),
                "positive_bars": "TP",
                "negative_bars": "FP + FN",
                "relative_formula": "value / GT * 100%, where GT = TP + FN",
                "gt_line": f"reference GT from first model in order: {OZZ_MODEL_ORDER[0]}",
                "fig2a_png": str(abs_png),
                "fig2b_png": str(rel_png),
                "stats_csv": str(stats_csv),
            }
        ]
    ).to_csv(manifest, index=False)

    return {
        "fig2a_png": abs_png,
        "fig2a_svg": abs_png.with_suffix(".svg"),
        "fig2b_png": rel_png,
        "fig2b_svg": rel_png.with_suffix(".svg"),
        "fig2_stats_csv": stats_csv,
        "fig2_manifest_csv": manifest,
    }


def _find_experiment_dir(exp_name: str) -> Path | None:
    for root in [ROOT_DIR / "experiments"]:
        if not root.exists():
            continue
        matches = [p for p in root.rglob(exp_name) if p.is_dir()]
        if matches:
            return matches[0]
    return None


def _ensure_experiment_from_zip(exp_name: str, archive_path: str | Path = ARTICLE2_FIG3_EXPERIMENT_ZIP) -> Path:
    existing = _find_experiment_dir(exp_name)
    if existing is not None:
        return existing

    archive = _resolve(archive_path)
    if not archive.exists():
        raise FileNotFoundError(
            f"Не найден эксперимент {exp_name} и нет архива для восстановления: {archive}"
        )

    target_prefix = f"phase2_6/Exp_2.6.11/{exp_name}/"
    alt_prefix = f"phase2_6/{exp_name}/"
    with zipfile.ZipFile(archive) as zf:
        members = [
            info
            for info in zf.infolist()
            if info.filename.startswith(target_prefix) or info.filename.startswith(alt_prefix)
        ]
        if not members:
            raise FileNotFoundError(f"В архиве {archive} не найден эксперимент {exp_name}")
        zf.extractall(ROOT_DIR / "experiments", members)

    restored = _find_experiment_dir(exp_name)
    if restored is None:
        raise FileNotFoundError(f"Эксперимент {exp_name} извлечён из архива, но папка не найдена")
    return restored


def _resolve_article2_fig3_checkpoint(
    exp_name: str = ARTICLE2_FIG3_CONVKAN_EXP,
    weights: str = "final",
    restore_from_zip: bool = False,
) -> Path:
    exp_dir = _find_experiment_dir(exp_name)
    if exp_dir is None and restore_from_zip:
        exp_dir = _ensure_experiment_from_zip(exp_name)
    if exp_dir is None:
        raise FileNotFoundError(
            f"Не найдена папка эксперимента {exp_name}. "
            f"Её можно восстановить из {ARTICLE2_FIG3_EXPERIMENT_ZIP}, запустив с restore_experiment_from_zip=True."
        )

    candidates = ["final_model.pt", "best_model.pt"] if weights.lower() == "final" else ["best_model.pt", "final_model.pt"]
    for name in candidates:
        path = exp_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"В {exp_dir} не найден final_model.pt/best_model.pt")


def replot_article2_fig3_marking(
    output_dir: str | Path,
    data_dir: str | Path = "data/ml_datasets",
    split: str = "train",
    panels: Iterable[dict[str, object]] = ARTICLE2_FIG3_PANELS,
    exp_name: str = ARTICLE2_FIG3_CONVKAN_EXP,
    restore_experiment_from_zip: bool = False,
    plot_mode: str = "confidence",
    threshold: float = 0.5,
    figure_width: float = 13.2,
    figure_height: float = 9.2,
    prediction_display_shift_samples: int = 320,
    physical_normalization: bool = False,
) -> dict[str, Path]:
    """Рисунок 3 статьи 2: разметка выбранных осциллограмм ConvKAN из фазы 2.6."""
    output_path = _resolve(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    panel_list = list(panels)
    exp_dir = _find_experiment_dir(exp_name)
    if exp_dir is None and restore_experiment_from_zip:
        exp_dir = _ensure_experiment_from_zip(exp_name)
    if exp_dir is None:
        raise FileNotFoundError(
            f"Не найдена папка эксперимента {exp_name}. "
            f"Её можно восстановить из {ARTICLE2_FIG3_EXPERIMENT_ZIP}, "
            "запустив с restore_experiment_from_zip=True."
        )

    saved: dict[str, Path] = {}
    manifest_rows: list[dict[str, str]] = []

    from scripts.evaluation.plot_model_marking import generate_marking_plots_for_model

    selected_files = [str(panel["file_name"]) for panel in panel_list]
    generated_root = output_path / "_generated"
    generate_marking_plots_for_model(
        exp_name=exp_name,
        output_dir=generated_root,
        data_dir=_resolve(data_dir),
        include_zero_current=True,
        include_zero_voltage=True,
        split=split,
        plot_mode=plot_mode,
        threshold=threshold,
        inference_backend="auto",
        selected_files=selected_files,
        figure_size=(figure_width, figure_height),
        dpi=220,
        signal_linewidth=1.35,
        label_fontsize=12,
        tick_fontsize=10,
        legend_fontsize=9.5,
        title_fontsize=13,
        marker_size=18,
        show_title=False,
        prediction_display_shift_samples=prediction_display_shift_samples,
        physical_normalization=physical_normalization,
    )

    generated_dir = generated_root / "marking_plots" / f"{exp_name}_{split}"
    if not generated_dir.exists():
        selected_text = ", ".join(selected_files)
        raise FileNotFoundError(
            f"Генератор не создал папку с PNG: {generated_dir}. "
            f"Проверьте, что файлы есть в split='{split}': {selected_text}"
        )

    for panel in panel_list:
        panel_letter = str(panel["panel"])
        file_name = str(panel["file_name"])
        mark_hash = hashlib.md5(f"{file_name}|{exp_name}".encode("utf-8")).hexdigest()[:12]
        src_png = generated_dir / f"mark_{mark_hash}.png"
        if not src_png.exists():
            raise FileNotFoundError(f"Ожидался построенный PNG, но он не найден: {src_png}")

        dst = output_path / f"fig3{panel_letter}_marking_{mark_hash}_article2.png"
        shutil.copy2(src_png, dst)

        saved[f"fig3{panel_letter}_png"] = dst
        manifest_rows.append(
            {
                "panel": panel_letter,
                "description": str(panel.get("description", "")),
                "file_name": file_name,
                "mark_hash": mark_hash,
                "source_png": str(src_png),
                "output_png": str(dst),
                "exp_name": exp_name,
                "exp_dir": str(exp_dir),
                "split": split,
                "plot_mode": plot_mode,
                "threshold": str(threshold),
            }
        )

    manifest = output_path / "fig3_marking_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    saved["fig3_manifest_csv"] = manifest
    return saved


if __name__ == "__main__":
    REPORT_ROOT = Path("reports/Exp_2_5_and_start_Exp_2_6")
    ARTICLE2_OUTPUT_ROOT = REPORT_ROOT / "figures_article2"

    # Для рисунка 1 нужен широкий набор экспериментов по типам данных.
    SOURCE_SUMMARY = REPORT_ROOT / "_Память" / "Общие опыты 2.5 и 2.6.1" / "summary_report.csv"

    RUN_FIG1_FEATURE_COMPARISON = False
    RUN_FIG2_OZZ_BIDIRECTIONAL_BARS = False
    # Включайте, когда нужно заново разметить COMTRADE-файлы для рисунка 3.
    # Если папка эксперимента 2.6.11 не распакована, поставьте restore_experiment_from_zip=True.
    RUN_FIG3_MARKING = True

    saved: dict[str, Path] = {}
    if RUN_FIG1_FEATURE_COMPARISON:
        saved.update(
            replot_article2_fig1_feature_comparison(
                summary_csv=SOURCE_SUMMARY,
                output_dir=ARTICLE2_OUTPUT_ROOT / "fig1_feature_comparison",
                figure_width_heatmap=9.2,
                figure_height_heatmap=5.6,
                figure_width_boxplot=9.2,
                figure_height_boxplot=5.2,
                show_titles=False,
            )
        )
    if RUN_FIG2_OZZ_BIDIRECTIONAL_BARS:
        saved.update(
            replot_article2_fig2_ozz_bidirectional_bars(
                output_dir=ARTICLE2_OUTPUT_ROOT / "fig2_ozz_bidirectional_bars",
                figure_width=10.8,
                figure_height=8.0,
                show_titles=False,
            )
        )
    if RUN_FIG3_MARKING:
        saved.update(
            replot_article2_fig3_marking(
                output_dir=ARTICLE2_OUTPUT_ROOT / "fig3_marking",
                data_dir="data/ml_datasets",
                split="train",
                exp_name=ARTICLE2_FIG3_CONVKAN_EXP,
                restore_experiment_from_zip=False,
                plot_mode="confidence",
                threshold=0.5,
                figure_width=13.2,
                figure_height=9.2,
                prediction_display_shift_samples=320,
                physical_normalization=False,
            )
        )

    print("Готово. Сохранены файлы:")
    for key, path in saved.items():
        print(f"  {key}: {path}")
