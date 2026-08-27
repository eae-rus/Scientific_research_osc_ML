"""Итоговая оценка PDR checkpoint на всём weak validation-split.

Сценарий не обучает модель, не изменяет checkpoint и не обращается к holdout.
В отличие от быстрой проверки во время эпох он использует все validation-записи
Open_EE и French/RTE. Число точек одной осциллограммы ограничивается отдельно,
поэтому длинные записи не получают неограниченный вес.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_experiments.progress import ProgressReporter
from scripts.phase5_experiments.run_phase5_pdr_training import (
    PDRTrainingConfig,
    _build_model,
    _collate,
    _imports,
    _weak_groups,
)


def _binary_from_counts(tp: float, tn: float, fp: float, fn: float) -> dict[str, float]:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_f1 = (
        2 * negative_precision * specificity / (negative_precision + specificity)
        if negative_precision + specificity else 0.0
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "negative_f1_score": negative_f1,
        "macro_f1_score": 0.5 * (f1 + negative_f1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
    }


def _combine_source_metrics(by_source: dict[str, dict[str, float]]) -> dict[str, float]:
    """Объединить source-метрики без второго чтения всех осциллограмм."""

    result: dict[str, float] = {}
    for prefix in ("", "applicability_"):
        tp = sum(float(item[prefix + "tp"]) for item in by_source.values())
        tn = sum(float(item[prefix + "tn"]) for item in by_source.values())
        fp = sum(float(item[prefix + "fp"]) for item in by_source.values())
        fn = sum(float(item[prefix + "fn"]) for item in by_source.values())
        for key, value in _binary_from_counts(tp, tn, fp, fn).items():
            result[prefix + key] = value
        result[prefix + "tp"] = tp
        result[prefix + "tn"] = tn
        result[prefix + "fp"] = fp
        result[prefix + "fn"] = fn

    result["forward_support"] = result["tp"] + result["fn"]
    result["reverse_support"] = result["tn"] + result["fp"]
    result["n_samples"] = sum(float(item["n_samples"]) for item in by_source.values())
    result["n_applicability_samples"] = sum(
        float(item["n_applicability_samples"]) for item in by_source.values()
    )
    result["n_margin_samples"] = sum(
        float(item.get("n_margin_samples", 0.0)) for item in by_source.values()
    )
    result["mae_margin"] = (
        sum(
            float(item["mae_margin"]) * float(item.get("n_margin_samples", 0.0))
            for item in by_source.values()
        ) / result["n_margin_samples"]
        if result["n_margin_samples"] else 0.0
    )
    for prefix in ("", "applicability_"):
        count_key = "n_direction_records" if not prefix else "n_applicability_records"
        count = sum(float(item[count_key]) for item in by_source.values())
        result[count_key] = count
        for metric in (
            "record_macro_accuracy",
            "record_macro_f1_score",
            "record_present_class_f1_score",
        ):
            key = prefix + metric
            result[key] = (
                sum(float(item.get(key, 0.0)) * float(item[count_key]) for item in by_source.values())
                / count if count else 0.0
            )
    return result


def _config_from_json(path: Path) -> PDRTrainingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PDRTrainingConfig)}
    return PDRTrainingConfig(
        **{key: value for key, value in payload.items() if key in allowed}
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    *,
    max_samples_per_record: int = 32,
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
    cfg.stage = "weak"
    cfg.validation_max_samples_per_record = int(max_samples_per_record)
    cfg.num_workers = int(num_workers)
    cfg.augmentation_probability = 0.0
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else "cpu" if cfg.device == "auto" else cfg.device
    )
    model, head, initialization = _build_model(cfg, None, checkpoint_path)
    model, head = model.to(device), head.to(device)

    groups = [
        (name, dataset)
        for name, dataset in _weak_groups(
            cfg, "validation", False, all_records=True
        )
        if len(dataset)
    ]
    if not groups:
        raise RuntimeError("Полный weak validation оказался пуст")
    print(
        "[Оценка объёма] "
        + ", ".join(
            f"{name}: {len(dataset.indices):,} записей / {len(dataset):,} точек"
            for name, dataset in groups
        ),
        flush=True,
    )

    by_source: dict[str, dict[str, float]] = {}
    for name, dataset in groups:
        source_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            persistent_workers=cfg.num_workers > 0,
        )
        progress = ProgressReporter(
            f"Полная weak-validation/{name}", len(source_loader), unit="batch"
        )
        by_source[name] = evaluate_pdr_metrics(
            model,
            head,
            source_loader,
            str(device),
            progress_callback=progress.update,
        )
        progress.finish()
    overall = _combine_source_metrics(by_source)

    result: dict[str, object] = {
        "evaluated_at": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "training_config": str(config_path),
        "initialization": initialization,
        "scope": "all_research_strict_validation_records",
        "holdout_used": False,
        "max_samples_per_record": max_samples_per_record,
        "label_stride_samples": cfg.label_stride_samples,
        "temporal_mode": cfg.temporal_mode,
        "model_preset": cfg.model_preset,
        "device": str(device),
        "indexed_targets": {name: len(dataset) for name, dataset in groups},
        "indexed_records": {name: len(dataset.indices) for name, dataset in groups},
        "overall": overall,
        "by_source": by_source,
        "interpretation": (
            "Это teacher-validation по всем validation-записям, а не экспертный "
            "эталон и не holdout. Сопоставлять checkpoint можно только при "
            "одинаковом max_samples_per_record и контракте признаков."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_path)
    print(f"[Готово] {output_path}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-record", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.config,
        args.output,
        max_samples_per_record=args.max_samples_per_record,
        num_workers=args.num_workers,
    )
    return 0


def run_manual() -> None:
    """Ручной F5-запуск итоговой проверки best/latest без holdout."""

    MAX_SAMPLES_PER_RECORD = 32
    NUM_WORKERS = 0
    RUNS = (
        (
            "weak_best",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/best_model.pt",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/config.json",
        ),
        (
            "weak_latest",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/latest_checkpoint.pt",
            "experiments/phase5/pdr_weak_snapshot_5_stride5/config.json",
        ),
    )
    output_root = PROJECT_ROOT / "experiments/phase5/pdr_full_weak_validation_v1"
    for name, checkpoint, config in RUNS:
        evaluate_checkpoint(
            PROJECT_ROOT / checkpoint,
            PROJECT_ROOT / config,
            output_root / f"{name}.json",
            max_samples_per_record=MAX_SAMPLES_PER_RECORD,
            num_workers=NUM_WORKERS,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
