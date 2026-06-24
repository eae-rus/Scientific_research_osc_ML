from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT_DIR = Path(__file__).resolve().parents[2]

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


if __name__ == "__main__":
    REPORT_ROOT = Path("reports/Exp_2_5_and_start_Exp_2_6")
    ARTICLE2_OUTPUT_ROOT = REPORT_ROOT / "figures_article2"

    # Для рисунка 1 нужен широкий набор экспериментов по типам данных.
    SOURCE_SUMMARY = REPORT_ROOT / "_Память" / "Общие опыты 2.5 и 2.6.1" / "summary_report.csv"

    RUN_FIG1_FEATURE_COMPARISON = True

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

    print("Готово. Сохранены файлы:")
    for key, path in saved.items():
        print(f"  {key}: {path}")
