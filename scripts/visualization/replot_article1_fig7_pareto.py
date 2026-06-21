"""Перерисовка рисунка 7 для первой статьи.

Строит две диаграммы Парето из уже готовой таблицы эксперимента 2.6.12:
  а) Full Test F1 vs CPU inference time
  б) Full Test F1 vs number of parameters

Эксперименты заново не запускаются: используются сохранённые CSV-данные.

Главная идея оформления:
  - цвет оставлен для PDF-версии;
  - форма маркера кодирует тип модели и различима в ЧБ;
  - заливка/размер маркера кодирует сложность Light/Medium/Heavy;
  - легенда разделена на "Модель" и "Сложность", без 24 строк.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == "__main__":
    # =====================================================================
    # РУЧНОЙ ЗАПУСК ДЛЯ РИСУНКА 7 ПЕРВОЙ СТАТЬИ
    # =====================================================================
    SOURCE_SUMMARY = (
        "reports/Exp_2_5_and_start_Exp_2_6/_Память/"
        "Опыт 2.6.12 (первая попытка)/summary_aggregated.csv"
    )
    OUTPUT_DIR = (
        "reports/Exp_2_5_and_start_Exp_2_6/_Память/"
        "Опыт 2.6.12 (первая попытка)/figures_article1/fig7_pareto"
    )

    # Размер фигуры в дюймах. Увеличивайте FIGURE_HEIGHT, если график
    # после вставки в две колонки выглядит слишком плоским.
    FIGURE_WIDTH = 9.6
    FIGURE_HEIGHT = 5.5
    SHOW_PANEL_TITLES = False
    # "right" лучше сохраняет высоту поля графика; "bottom" похож на прошлую версию;
    # "none" полезен, если легенда будет вынесена в подпись или отдельную панель.
    LEGEND_MODE = "right"

    paths = replot_article1_fig7_pareto(
        summary_csv=SOURCE_SUMMARY,
        output_dir=OUTPUT_DIR,
        figure_width=FIGURE_WIDTH,
        figure_height=FIGURE_HEIGHT,
        show_panel_titles=SHOW_PANEL_TITLES,
        legend_mode=LEGEND_MODE,
        annotate_pareto=False,
    )

    print("Готово. Сохранены файлы:")
    for key, path in paths.items():
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
    # shutil.copy2(paths["cpu_png"], ARTICLE_FIGURES_DIR / "Рис.7.а_pareto_full_f1_cpu_article.png")
    # shutil.copy2(paths["params_png"], ARTICLE_FIGURES_DIR / "Рис.7.б_pareto_full_f1_params_article.png")
