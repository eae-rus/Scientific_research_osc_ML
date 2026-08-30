"""Единая оценка weak/expert checkpoint на экспертных validation и holdout.

Сценарий не обучает модель и не изменяет checkpoint. Он нужен для сравнения
моделей на одном и том же экспертном эталоне, чего нельзя получить прямым
сопоставлением обычных training_history двух этапов.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_experiments.progress import ProgressReporter
from scripts.phase5_experiments.evaluate_pdr_full_weak_validation import (
    _combine_source_metrics,
)
from scripts.phase5_experiments.run_phase5_pdr_training import (
    PDRTrainingConfig,
    _build_model,
    _collate,
    _expert_groups,
    _imports,
)


def _config_from_json(path: Path) -> PDRTrainingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PDRTrainingConfig)}
    return PDRTrainingConfig(**{key: value for key, value in payload.items() if key in allowed})


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    *,
    splits: tuple[str, ...] = ("validation",),
    max_samples_per_record: int = 256,
    num_workers: int = 0,
) -> dict[str, object]:
    (
        torch, _, DataLoader, _, _, _, evaluate_pdr_metrics, _, _,
    ) = _imports()
    checkpoint_path = checkpoint_path.resolve()
    config_path = config_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")
    cfg = _config_from_json(config_path)
    cfg.stage = "expert"
    cfg.validation_max_samples_per_record = int(max_samples_per_record)
    cfg.expert_validation_max_samples_per_record = int(max_samples_per_record)
    cfg.num_workers = int(num_workers)
    cfg.augmentation_probability = 0.0
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else "cpu" if cfg.device == "auto" else cfg.device
    )
    model, head, initialization = _build_model(cfg, None, checkpoint_path)
    model, head = model.to(device), head.to(device)

    split_results: dict[str, object] = {}
    for split in splits:
        if split not in ("validation", "holdout"):
            raise ValueError("Оценивать разрешено только validation или holdout")
        groups = [(name, dataset) for name, dataset in _expert_groups(cfg, split, False) if len(dataset)]
        if not groups:
            raise RuntimeError(f"Нет экспертных данных для split={split}")

        by_source: dict[str, object] = {}
        for name, dataset in groups:
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=_collate,
                persistent_workers=cfg.num_workers > 0,
            )
            source_name = name.removeprefix("expert_")
            progress = ProgressReporter(
                f"Экспертная оценка {split}/{source_name}", len(loader), unit="batch"
            )
            by_source[source_name] = evaluate_pdr_metrics(
                model, head, loader, str(device), progress_callback=progress.update
            )
            progress.finish()
        overall = _combine_source_metrics(by_source)
        split_results[split] = {
            "overall": overall,
            "by_source": by_source,
            "indexed_targets": {name: len(dataset) for name, dataset in groups},
        }

    result: dict[str, object] = {
        "evaluated_at": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "training_config": str(config_path),
        "initialization": initialization,
        "max_samples_per_record": max_samples_per_record,
        "label_stride_samples": cfg.label_stride_samples,
        "temporal_mode": cfg.temporal_mode,
        "model_preset": cfg.model_preset,
        "device": str(device),
        "splits": split_results,
        "interpretation": (
            "Holdout не используется для выбора checkpoint; сравнивать weak и expert "
            "следует только при одинаковых split и параметрах отбора точек."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"[Готово] {output_path}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", action="append", choices=("validation", "holdout"), default=None
    )
    parser.add_argument("--max-samples-per-record", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.config,
        args.output,
        splits=tuple(args.split or ("validation",)),
        max_samples_per_record=args.max_samples_per_record,
        num_workers=args.num_workers,
    )
    return 0


def run_manual() -> None:
    """Ручная проверка одинакового expert-эталона для weak и expert моделей."""

    MAX_SAMPLES_PER_RECORD = 256
    NUM_WORKERS = 0
    # Holdout добавить только после фиксации архитектуры/порогов:
    # SPLITS = ("validation", "holdout")
    SPLITS = ("validation",)
    RUNS = (
        (
            "snapshot_2_weak_best",
            "experiments/phase5/pdr_weak_snapshot_2_stride5/best_model.pt",
            "experiments/phase5/pdr_weak_snapshot_2_stride5/config.json",
        ),
        (
            "snapshot_2_expert_best",
            "experiments/phase5/pdr_expert_snapshot_2_stride5/best_model.pt",
            "experiments/phase5/pdr_expert_snapshot_2_stride5/config.json",
        ),
        (
            "snapshot_5_weak_best",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/best_model.pt",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/config.json",
        ),
        (
            "snapshot_5_expert_best",
            "experiments/phase5/pdr_expert_snapshot_5_stride5/"
            "archive_20260830_131809/best_model.pt",
            "experiments/phase5/pdr_expert_snapshot_5_stride5/"
            "archive_20260830_131809/config.json",
        ),
        (
            "sequence_1_8_weak_best",
            "experiments/phase5/pdr_weak_sequence_1_8_stride5/best_model.pt",
            "experiments/phase5/pdr_weak_sequence_1_8_stride5/config.json",
        ),
        (
            "sequence_1_8_expert_best",
            "experiments/phase5/pdr_expert_sequence_1_8_stride5/best_model.pt",
            "experiments/phase5/pdr_expert_sequence_1_8_stride5/config.json",
        ),
    )
    output_root = PROJECT_ROOT / "experiments/phase5/pdr_expert_evaluation_v2"
    for name, checkpoint, config in RUNS:
        evaluate_checkpoint(
            PROJECT_ROOT / checkpoint,
            PROJECT_ROOT / config,
            output_root / f"{name}.json",
            splits=SPLITS,
            max_samples_per_record=MAX_SAMPLES_PER_RECORD,
            num_workers=NUM_WORKERS,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
