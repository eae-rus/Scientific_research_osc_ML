"""Воспроизводимая подготовка рисунков раздела 7 статьи 4.

Сценарий не переоценивает весь архив: он берёт готовые итоги
симуляции и повторно размечает только два зафиксированных реальных примера.
Все важные пути и имена задаются в ручном блоке внизу.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_experiments.ct_saturation.analyze_ct_saturation import (
    infer_oscillogram,
    load_model,
    load_normalization_lookup,
    parse_real_file_identity,
    real_normalization_profile,
)

PHASE_COLORS = ("#D49A00", "#278C2D", "#E04416")
PHASE_NAMES = ("A", "B", "C")


def _load_real_case(csv_path: Path, file_name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Загрузить три тока и лучший доступный комплект напряжений."""
    columns = [
        "file_name", "IA", "IB", "IC",
        "UA BB", "UB BB", "UC BB", "UA CL", "UB CL", "UC CL",
    ]
    frame = pl.read_csv(
        csv_path, columns=columns, schema_overrides={"file_name": pl.String},
    ).filter(pl.col("file_name") == file_name)
    if frame.is_empty():
        raise ValueError(f"В CSV не найдена запись {file_name!r}")
    numeric = [column for column in columns if column != "file_name"]
    frame = frame.with_columns([pl.col(column).cast(pl.Float32, strict=False) for column in numeric])
    currents = frame.select(["IA", "IB", "IC"]).to_numpy().astype(np.float32)
    bb = frame.select(["UA BB", "UB BB", "UC BB"]).to_numpy().astype(np.float32)
    cl = frame.select(["UA CL", "UB CL", "UC CL"]).to_numpy().astype(np.float32)
    bb_count, cl_count = int(np.isfinite(bb).sum()), int(np.isfinite(cl).sum())
    if bb_count > 0 and bb_count >= cl_count:
        voltages, source = bb, "BB"
    elif cl_count > 0:
        voltages, source = cl, "CL"
    else:
        voltages, source = np.zeros_like(currents), "NONE"
    return (
        np.nan_to_num(currents, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(voltages, nan=0.0, posinf=0.0, neginf=0.0),
        source,
    )


def _plot_real_case(
    output: Path,
    currents: np.ndarray,
    probabilities: np.ndarray,
    positions: np.ndarray,
    *,
    fs_hz: float,
    title: str,
    threshold: float,
    xlim_ms: tuple[float, float] | None = None,
) -> None:
    """Построить двухпанельный рисунок токов и фазных вероятностей."""
    time_ms = np.arange(len(currents)) / fs_hz * 1000.0
    probability_time = positions / fs_hz * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.7), sharex=True, height_ratios=(1.45, 1.0))
    for phase, color, name in zip(currents.T, PHASE_COLORS, PHASE_NAMES):
        axes[0].plot(time_ms, phase, color=color, linewidth=1.05, label=f"I{name}")
    axes[0].set_ylabel("Вторичный ток, А")
    axes[0].set_title(title)
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].grid(alpha=0.23, linestyle=":")

    for index, (color, name) in enumerate(zip(PHASE_COLORS, PHASE_NAMES)):
        axes[1].plot(
            probability_time, probabilities[:, index], color=color,
            linewidth=1.25, label=f"P({name})",
        )
    axes[1].axhline(threshold, color="#555555", linestyle="--", linewidth=1.0, label="Порог 0,5")
    axes[1].set(xlabel="Время, мс", ylabel="Вероятность насыщения", ylim=(-0.03, 1.03))
    axes[1].legend(ncol=4, loc="upper right")
    axes[1].grid(alpha=0.23, linestyle=":")
    if xlim_ms is not None:
        axes[1].set_xlim(*xlim_ms)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(metrics_path: Path, output: Path) -> None:
    """Построить матрицу ошибок с абсолютными и построчными долями."""
    zone = json.loads(metrics_path.read_text(encoding="utf-8"))["zone_phase_A"]
    matrix = np.array([[zone["tn"], zone["fp"]], [zone["fn"], zone["tp"]]], dtype=np.int64)
    row_fraction = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    image = ax.imshow(row_fraction, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(2):
        for column in range(2):
            color = "white" if row_fraction[row, column] > 0.55 else "black"
            ax.text(
                column, row, f"{matrix[row, column]:,}\n{row_fraction[row, column] * 100:.2f}%",
                ha="center", va="center", fontsize=12, color=color,
            )
    labels = ["Нет насыщения", "Насыщение"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set(xlabel="Предсказание", ylabel="Истинная метка", title="Матрица ошибок, фаза A")
    fig.colorbar(image, ax=ax, label="Доля внутри истинного класса")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_article_figures(
    *,
    run_dir: Path,
    real_csv: Path,
    norm_csv: Path,
    output_dir: Path,
    switching_name: str,
    saturation_name: str,
    saturation_zoom_ms: tuple[float, float],
    real_fs_hz: float = 1600.0,
    threshold: float = 0.5,
) -> None:
    """Собрать все четыре рисунка и манифест воспроизведения."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / "article_report"
    _plot_confusion_matrix(
        report / "simulated" / "metrics.json", output_dir / "fig11_confusion_matrix_phase_A.png",
    )
    shutil.copy2(
        report / "simulated" / "examples" / "sim_tau_0.015_1436.png",
        output_dir / "fig12_sim_tau_0.015_1436.png",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Основной анализ данного опыта выполнен по latest_checkpoint.pt; рисунки
    # обязаны использовать тот же checkpoint, иначе вероятности не совпадут.
    checkpoint_path = run_dir / "latest_checkpoint.pt"
    model, model_config = load_model(checkpoint_path, device)
    lookup = load_normalization_lookup(norm_csv)
    cases = [
        (switching_name, "fig13_real_switching_without_saturation.png", None,
         "Коммутационный бросок тока"),
        (saturation_name, "fig14_real_saturation_zoom.png", saturation_zoom_ms,
         "Фрагмент с признаками насыщения ТТ"),
    ]
    manifest_cases = []
    for file_name, output_name, zoom, title in cases:
        identity = parse_real_file_identity(file_name)
        if identity is None:
            raise ValueError(f"Не удалось разобрать {file_name!r}")
        currents, voltages, voltage_source = _load_real_case(real_csv, file_name)
        current_divisor, voltage_divisor = real_normalization_profile(
            file_name, lookup, voltage_source=voltage_source,
        )
        probabilities, positions, _ = infer_oscillogram(
            model, currents / current_divisor, voltages / voltage_divisor,
            real_fs_hz, model_config, device,
        )
        _plot_real_case(
            output_dir / output_name, currents, probabilities, positions,
            fs_hz=real_fs_hz, title=title, threshold=threshold, xlim_ms=zoom,
        )
        manifest_cases.append({
            "file_name": file_name,
            "article_figure": output_name,
            "zoom_ms": zoom,
            "current_divisor": current_divisor,
            "voltage_divisor": voltage_divisor,
            "voltage_source": voltage_source,
            "max_probability": probabilities.max(axis=0).tolist(),
        })
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "real_csv": str(real_csv),
        "norm_csv": str(norm_csv),
        "simulation_source": "sim_tau_0.015_1436.png",
        "cases": manifest_cases,
    }
    (output_dir / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    # ================================================================
    # РУЧНОЙ ЗАПУСК ДЛЯ СТАТЬИ 4
    # ================================================================
    RUN_DIR = PROJECT_ROOT / "experiments" / "phase4" / "ct_saturation_v2" / "spectral_run_20260703_145033"
    REAL_CSV = PROJECT_ROOT / "data" / "ml_datasets" / "labeled_2025_12_03.csv"
    NORM_CSV = PROJECT_ROOT / "data" / "norm_coef_all_v1.4.csv"
    OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4" / "article4_ct_saturation"

    SWITCHING_NAME = "0a3cba1c0d3a410500489925e5282827_Bus 1 _event N1"
    SATURATION_NAME = "0ca97e0055731b99ece0d3e6ee96fa70_Bus 2 _event N1"
    SATURATION_ZOOM_MS = (180.0, 285.0)

    build_article_figures(
        run_dir=RUN_DIR,
        real_csv=REAL_CSV,
        norm_csv=NORM_CSV,
        output_dir=OUTPUT_DIR,
        switching_name=SWITCHING_NAME,
        saturation_name=SATURATION_NAME,
        saturation_zoom_ms=SATURATION_ZOOM_MS,
    )
