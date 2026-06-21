"""Полный сбор отчётов Phase 4 для одной KAN-модели.

Скрипт-оркестратор для ручного запуска. Он последовательно вызывает уже
существующие инструменты оценки, визуализации и интерпретируемости, чтобы
собрать набор артефактов, аналогичный reports/phase4/От весны, но для одной
актуальной модели PhysicalKANTransformer.

Запуск:
    python scripts/phase4_experiments/run_full_kan_report_collection.py

Важно: запускать нужно из ML-окружения, где доступны torch, matplotlib, polars.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: str | Path) -> Path:
    """Возвращает абсолютный путь относительно корня проекта."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _run_step(name: str, args: list[str], cwd: Path = PROJECT_ROOT) -> None:
    """Запускает один шаг и останавливает весь прогон при ошибке."""
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print("CMD:")
    print(" ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def run_full_kan_report_collection(
    checkpoint: str | Path,
    output_root: str | Path = "reports/phase4/kan_20260616_072711",
    sim_per_class_files: int = 240,
    sim_batch_size: int = 128,
    sim_num_workers: int = 4,
    marking_per_class: int = 240,
    real_threshold: float = 0.5,
    channel_dropout_max_files: int = 240 * 4,
    channel_dropout_batch_size: int = 256,
    gradient_max_files: int = 240 * 4,
    gradient_batch_size: int = 64,
    interpretability_num_workers: int = 4,
    complexity_batch_size: int = 256,
    complexity_warmup: int = 10,
    complexity_repeats: int = 50,
    run_sim_eval: bool = True,
    run_sim_markings: bool = True,
    run_real_statistics: bool = True,
    run_real_inference: bool = True,
    run_channel_dropout: bool = True,
    run_gradient_attribution: bool = True,
    run_model_complexity: bool = True,
) -> None:
    """Полный прогон отчётов только для PhysicalKANTransformer.

    По умолчанию используется максимальный для статьи режим SimOZZ:
    240 файлов на каждый из 4 классов дуги.
    """
    checkpoint_path = _project_path(checkpoint)
    output_path = _project_path(output_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт: {checkpoint_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.executable}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output root: {output_path}")

    py = sys.executable
    ckpt = str(checkpoint_path)

    sim_eval_dir = output_path / "sim_ozz_eval"
    sim_marking_dir = sim_eval_dir / "marking_plots"
    real_stats_dir = output_path / "real_ozz_statistics"
    real_inference_dir = output_path / "real_ozz_inference"
    interpretability_dir = output_path / "interpretability"
    complexity_dir = output_path / "model_complexity"

    if run_sim_eval:
        _run_step(
            "1/7 SimOZZ evaluation: метрики + графики",
            [
                py,
                "scripts/phase4_experiments/sim_ozz/evaluate_sim_ozz.py",
                "--checkpoint",
                ckpt,
                "--per-class-files",
                str(sim_per_class_files),
                "--batch-size",
                str(sim_batch_size),
                "--num-workers",
                str(sim_num_workers),
                "--roc",
                "--output-dir",
                str(sim_eval_dir),
            ],
        )

    if run_sim_markings:
        _run_step(
            "2/7 SimOZZ marking plots: размеченные синтетические осциллограммы",
            [
                py,
                "scripts/phase4_experiments/sim_ozz/plot_sim_ozz_marking.py",
                "--checkpoint",
                ckpt,
                "--per-class",
                str(marking_per_class),
                "--threshold",
                str(real_threshold),
                "--output-dir",
                str(sim_marking_dir),
            ],
        )

    if run_real_statistics:
        _run_step(
            "3/7 Real OZZ statistics: подтверждённые и ложные реальные случаи",
            [
                py,
                "scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py",
                "--checkpoint",
                ckpt,
                "--threshold",
                str(real_threshold),
                "--output-dir",
                str(real_stats_dir),
            ],
        )

    if run_real_inference:
        _run_step(
            "4/7 Real OZZ inference plots: PNG-разметка COMTRADE",
            [
                py,
                "scripts/phase4_experiments/real_ozz/inference_real_ozz.py",
                "--checkpoint",
                ckpt,
                "--subset",
                "all",
                "--threshold",
                str(real_threshold),
                "--output-dir",
                str(real_inference_dir),
            ],
        )

    if run_channel_dropout:
        _run_step(
            "5/7 Interpretability: channel dropout probing",
            [
                py,
                "scripts/phase4_experiments/interpretability/channel_dropout_probing.py",
                "--checkpoint",
                ckpt,
                "--max-files",
                str(channel_dropout_max_files),
                "--batch-size",
                str(channel_dropout_batch_size),
                "--num-workers",
                str(interpretability_num_workers),
                "--threshold",
                str(real_threshold),
                "--output-dir",
                str(interpretability_dir),
            ],
        )

    if run_gradient_attribution:
        _run_step(
            "6/7 Interpretability: gradient attribution",
            [
                py,
                "scripts/phase4_experiments/interpretability/gradient_attribution.py",
                "--checkpoint",
                ckpt,
                "--max-files",
                str(gradient_max_files),
                "--batch-size",
                str(gradient_batch_size),
                "--num-workers",
                str(interpretability_num_workers),
                "--output-dir",
                str(interpretability_dir),
            ],
        )

    if run_model_complexity:
        _run_step(
            "7/7 Model complexity: параметры и latency",
            [
                py,
                "scripts/phase4_experiments/evaluation/model_complexity_report.py",
                "--checkpoint",
                ckpt,
                "--batch-size",
                str(complexity_batch_size),
                "--warmup",
                str(complexity_warmup),
                "--repeats",
                str(complexity_repeats),
                "--output",
                str(complexity_dir / "physical_kan_20260616_latest_complexity.json"),
            ],
        )

    print("\n" + "=" * 90)
    print("Полный сбор отчётов завершён.")
    print(f"Артефакты: {output_path}")
    print("=" * 90)


if __name__ == "__main__":
    # =====================================================================
    # РУЧНОЙ РЕЖИМ
    # =====================================================================
    CHECKPOINT = (
        "experiments/phase4/"
        "sim_ozz_finetune_PhysicalKANTransformer_20260616_072711/"
        "latest_checkpoint.pt"
    )
    OUTPUT_ROOT = "reports/phase4/kan_20260616_072711"

    # Полный режим для статьи: 240 файлов на каждый из 4 классов SimOZZ.
    SIM_PER_CLASS_FILES = 240
    MARKING_PER_CLASS = 240

    # Интерпретируемость. Для ускорения можно временно уменьшить до 200,
    # но полный режим оставлен как 240*4.
    CHANNEL_DROPOUT_MAX_FILES = 240 * 4
    GRADIENT_MAX_FILES = 240 * 4

    run_full_kan_report_collection(
        checkpoint=CHECKPOINT,
        output_root=OUTPUT_ROOT,
        sim_per_class_files=SIM_PER_CLASS_FILES,
        marking_per_class=MARKING_PER_CLASS,
        channel_dropout_max_files=CHANNEL_DROPOUT_MAX_FILES,
        gradient_max_files=GRADIENT_MAX_FILES,
    )
