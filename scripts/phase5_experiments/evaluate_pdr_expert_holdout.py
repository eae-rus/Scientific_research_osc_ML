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
import hashlib
from pathlib import Path
import sys

import numpy as np

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
from osc_tools.pdr.base import PDRDirection
from osc_tools.pdr.study import PDRStudyLabelStore


def _sha256(path: Path) -> str:
    """Отпечаток содержимого, независимый от копирования файла между ПК."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _grid_signature(groups) -> dict[str, object]:
    """Связать результат с фактическими точками и экспертными целями."""
    result = {}
    for name, dataset in groups:
        digest = hashlib.sha256()
        selected = {}
        for record_id, window_index in dataset.samples:
            selected.setdefault(int(record_id), []).append(int(window_index))
        for record_id, indices in sorted(selected.items()):
            labels = dataset.label_store.get_record(record_id)
            digest.update(np.asarray([record_id, len(indices)], dtype="<i8").tobytes())
            for key in ("samples", "directions"):
                digest.update(np.asarray(labels[key][indices], dtype="<i8").tobytes())
        result[name.removeprefix("expert_")] = {
            "sha256": digest.hexdigest(), "records": len(selected), "points": len(dataset),
        }
    return result


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
        grid = _grid_signature(groups)
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
            collected = {key: [] for key in ("record_id", "window_idx", "target", "prediction", "applicable")}

            def collect(batch, outputs):
                for key in ("record_id", "window_idx"):
                    collected[key].extend(torch.as_tensor(batch[key]).cpu().tolist())
                collected["target"].extend(batch["pdr_direction"].cpu().tolist())
                collected["prediction"].extend(outputs["logits"].argmax(-1).cpu().tolist())
                collected["applicable"].extend((outputs["applicability_logit"] >= 0).cpu().tolist())

            by_source[source_name] = evaluate_pdr_metrics(
                model, head, loader, str(device), progress_callback=progress.update,
                prediction_callback=collect,
            )
            arrays = {key: np.asarray(value) for key, value in collected.items()}
            predicted_state = np.where(arrays["applicable"], arrays["prediction"], -999)
            by_source[source_name]["joint_state_accuracy"] = float(np.mean(predicted_state == arrays["target"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path.with_name(f"{output_path.stem}__{split}__{source_name}.npz"), **arrays,
            )
            progress.finish()
        overall = _combine_source_metrics(by_source)
        overall["joint_state_accuracy"] = sum(
            item["joint_state_accuracy"] * item["n_applicability_samples"] for item in by_source.values()
        ) / overall["n_applicability_samples"]
        split_results[split] = {
            "overall": overall,
            "by_source": by_source,
            "indexed_targets": {name: len(dataset) for name, dataset in groups},
            "grid_signature": grid,
        }

    result: dict[str, object] = {
        "evaluated_at": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_config_sha256": _sha256(config_path),
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


def _direction_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """Единые метрики для аналитических органов на той же сетке, что и модели."""

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    accuracy = ratio(tp + tn, tp + tn + fp + fn)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1_forward = ratio(2.0 * precision * recall, precision + recall)
    precision_reverse = ratio(tn, tn + fn)
    f1_reverse = ratio(
        2.0 * precision_reverse * specificity,
        precision_reverse + specificity,
    )
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_forward,
        "negative_f1_score": f1_reverse,
        "macro_f1_score": 0.5 * (f1_forward + f1_reverse),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": ratio(tp * tn - fp * fn, denominator),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "n_samples": float(tp + tn + fp + fn),
    }


def evaluate_analytical_algorithms_on_neural_grid(
    config_path: Path,
    labels_root: Path,
    output_path: Path,
    *,
    split: str = "validation",
    max_samples_per_record: int = 256,
) -> dict[str, object]:
    """Сравнить все РНМ v6 с экспертом на точной validation-сетке нейросетей."""

    if split not in ("validation", "holdout"):
        raise ValueError("Аналитический benchmark разрешён только на validation/holdout")
    config_path = config_path.resolve()
    labels_root = labels_root.resolve()
    if not labels_root.exists():
        raise FileNotFoundError(f"Не найдена разметка РНМ: {labels_root}")

    cfg = _config_from_json(config_path)
    cfg.stage = "expert"
    cfg.expert_validation_max_samples_per_record = int(max_samples_per_record)
    cfg.augmentation_probability = 0.0
    groups = [(name, dataset) for name, dataset in _expert_groups(cfg, split, False) if len(dataset)]
    if not groups:
        raise RuntimeError(f"Нет экспертных данных для split={split}")

    manifest = json.loads((labels_root / "open_ee" / "manifest.json").read_text(encoding="utf-8"))
    algorithm_ids = tuple(str(value) for value in manifest["algorithm_ids"])
    aggregate_counts = {algorithm_id: np.zeros(4, dtype=np.int64) for algorithm_id in algorithm_ids}
    aggregate_app_counts = {
        algorithm_id: np.zeros(4, dtype=np.int64) for algorithm_id in algorithm_ids
    }
    by_source: dict[str, dict[str, object]] = {}

    for name, dataset in groups:
        source = name.removeprefix("expert_")
        selected_by_record: dict[int, list[int]] = {}
        for record_id, window_index in dataset.samples:
            selected_by_record.setdefault(int(record_id), []).append(int(window_index))
        source_rows: dict[str, object] = {}
        for algorithm_id in algorithm_ids:
            store = PDRStudyLabelStore(labels_root / source, algorithm_id=algorithm_id)
            tp = tn = fp = fn = 0
            app_tp = app_tn = app_fp = app_fn = 0
            try:
                for record_id, window_indices_list in selected_by_record.items():
                    if not store.has_record(record_id):
                        continue
                    expert = dataset.label_store.get_record(record_id)
                    automatic = store.get_record(record_id)
                    window_indices = np.asarray(window_indices_list, dtype=np.int64)
                    expert_samples = np.asarray(expert["samples"])[window_indices]
                    automatic_samples = np.asarray(automatic["samples"])
                    positions = np.searchsorted(automatic_samples, expert_samples)
                    matched = positions < automatic_samples.size
                    matched_indices = np.flatnonzero(matched)
                    if matched_indices.size:
                        matched[matched_indices] &= (
                            automatic_samples[positions[matched_indices]] == expert_samples[matched_indices]
                        )

                    expert_values = np.asarray(expert["directions"])[window_indices]
                    automatic_values = np.full(
                        expert_values.shape,
                        int(PDRDirection.UNLABELED),
                        dtype=np.int16,
                    )
                    automatic_values[matched] = np.asarray(automatic["directions"])[positions[matched]]
                    expert_applicable = expert_values != int(PDRDirection.UNLABELED)
                    automatic_applicable = automatic_values != int(PDRDirection.UNLABELED)
                    app_tp += int(np.count_nonzero(expert_applicable & automatic_applicable))
                    app_tn += int(np.count_nonzero(~expert_applicable & ~automatic_applicable))
                    app_fp += int(np.count_nonzero(~expert_applicable & automatic_applicable))
                    app_fn += int(np.count_nonzero(expert_applicable & ~automatic_applicable))

                    common = expert_applicable & automatic_applicable
                    target = expert_values[common]
                    prediction = automatic_values[common]
                    tp += int(np.count_nonzero((target == 1) & (prediction == 1)))
                    tn += int(np.count_nonzero((target == 0) & (prediction == 0)))
                    fp += int(np.count_nonzero((target == 0) & (prediction == 1)))
                    fn += int(np.count_nonzero((target == 1) & (prediction == 0)))
            finally:
                store.close()

            metrics = _direction_metrics_from_counts(tp, tn, fp, fn)
            app_count = app_tp + app_tn + app_fp + app_fn
            app_metrics = _direction_metrics_from_counts(app_tp, app_tn, app_fp, app_fn)
            metrics.update({
                "joint_state_accuracy": (tp + tn + app_tn) / app_count if app_count else 0.0,
                "applicability_accuracy": (app_tp + app_tn) / app_count if app_count else 0.0,
                "applicability_macro_f1_score": app_metrics["macro_f1_score"],
                "applicability_balanced_accuracy": app_metrics["balanced_accuracy"],
                "applicability_mcc": app_metrics["mcc"],
                "applicability_tp": float(app_tp),
                "applicability_tn": float(app_tn),
                "applicability_fp": float(app_fp),
                "applicability_fn": float(app_fn),
                "n_applicability_samples": float(app_count),
            })
            source_rows[algorithm_id] = metrics
            aggregate_counts[algorithm_id] += np.asarray((tp, tn, fp, fn), dtype=np.int64)
            aggregate_app_counts[algorithm_id] += np.asarray(
                (app_tp, app_tn, app_fp, app_fn), dtype=np.int64
            )
        by_source[source] = source_rows

    overall: dict[str, dict[str, float]] = {}
    for algorithm_id, counts in aggregate_counts.items():
        metrics = _direction_metrics_from_counts(*map(int, counts))
        app_metrics = _direction_metrics_from_counts(
            *map(int, aggregate_app_counts[algorithm_id])
        )
        metrics.update({
            "joint_state_accuracy": (int(counts[0] + counts[1]) + int(aggregate_app_counts[algorithm_id][1]))
                / app_metrics["n_samples"] if app_metrics["n_samples"] else 0.0,
            "applicability_accuracy": app_metrics["accuracy"],
            "applicability_macro_f1_score": app_metrics["macro_f1_score"],
            "applicability_balanced_accuracy": app_metrics["balanced_accuracy"],
            "applicability_mcc": app_metrics["mcc"],
            "n_applicability_samples": app_metrics["n_samples"],
        })
        overall[algorithm_id] = metrics
    result: dict[str, object] = {
        "evaluated_at": datetime.now().isoformat(),
        "kind": "pdr_analytical_expert_comparison",
        "labels_root": str(labels_root),
        "reference_config": str(config_path),
        "split": split,
        "max_samples_per_record": max_samples_per_record,
        "label_stride_samples": cfg.label_stride_samples,
        "algorithm_ids": list(algorithm_ids),
        "overall": overall,
        "by_source": by_source,
        "grid_signature": _grid_signature(groups),
        "interpretation": (
            "Органы и нейросети сравниваются при одинаковых split, stride, лимите "
            "точек на файл и каузальной маске. Direction-метрики аналитического "
            "органа условны по совместной применимости эксперта и органа; число таких "
            "точек сохранено как n_samples. Применимость оценивается отдельно на всей "
            "сетке. Этот файл, а не full-expert aggregate, является источником рисунка."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"[Готово] {output_path}", flush=True)
    return result


def _split_overlap_audit(rows: list[dict]) -> dict:
    """Проверить дубли входа между обучением и обеими контрольными частями."""
    training = {}
    for row in rows:
        if row["split"] == "train" and row.get("input_sha256"):
            training.setdefault((row["source"], row["input_sha256"]), []).append(int(row["record_id"]))
    overlap = []
    for row in rows:
        key = (row["source"], row.get("input_sha256"))
        if row["split"] in ("validation", "holdout") and key in training:
            overlap.append({"source": row["source"], "record_id": int(row["record_id"]),
                            "split": row["split"], "input_sha256": row["input_sha256"],
                            "training_record_ids": training[key]})
    return {"training_overlap": overlap, "passed": not overlap}


def _cached_prediction_metrics(arrays: dict) -> dict:
    """Описательная оценка сохранённых ответов после исключения дублей."""
    y, p = arrays["target"], arrays["prediction"]
    valid, predicted_valid = y != -999, arrays["applicable"].astype(bool)
    def counts(target, prediction):
        return [int(np.sum(target & prediction)), int(np.sum(~target & ~prediction)),
                int(np.sum(~target & prediction)), int(np.sum(target & ~prediction))]
    direction = _direction_metrics_from_counts(*counts(y[valid] == 1, p[valid] == 1))
    applicability = _direction_metrics_from_counts(*counts(valid, predicted_valid))
    return {"direction": direction, "applicability": applicability,
            "joint_state_accuracy": float(np.mean(np.where(predicted_valid, p, -999) == y)) if y.size else None,
            "points": int(y.size), "records": int(np.unique(arrays["record_id"]).size)}


def summarize_current_evaluations(output_root: Path, bootstrap_repetitions: int = 2000) -> dict:
    """Парные интервалы по группам исходных файлов, не по зависимым отсчётам.

    Интервалы описывают выборку validation при фиксированных весах; они не
    учитывают оптимизм выбора лучшей эпохи и разброс между обучениями.
    """
    import csv
    with (PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1/records.csv").open(encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
        groups = {(r["source"], int(r["record_id"])): r["input_sha256"] for r in records}
    result = {"models": {}, "bootstrap_repetitions": bootstrap_repetitions, "seed": 20260910}
    result["split_integrity"] = _split_overlap_audit(records)
    excluded = {(r["source"], r["record_id"]) for r in result["split_integrity"]["training_overlap"]}
    result["split_integrity"]["interpretation"] = (
        "Дубли проверены по хешам входов экспертного архива. Исключение их после обучения "
        "не отменяет выбора эпохи на исходной validation и не делает оценку независимым тестом."
    )
    if excluded:
        print(f"[ВНИМАНИЕ] Найдены контрольные копии train: {sorted(excluded)}", flush=True)
    expert_bootstraps = {}
    common_grid = None
    for mode in ("snapshot_2", "snapshot_5", "sequence_1_8"):
        comparison = {}
        grouped = {}
        reference_grid = None
        for stage in ("weak_initial", "expert_best"):
            path = output_root / f"{mode}_{stage}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            validation = payload["splits"]["validation"]
            if reference_grid is not None and reference_grid != validation["grid_signature"]:
                raise ValueError("Нельзя сравнивать разные экспертные сетки")
            reference_grid = validation["grid_signature"]
            counts, correct, total = {}, 0, 0
            source_joint = {}
            clean_sources, clean_arrays = {}, []
            for source in ("open_ee", "french_rte"):
                with np.load(output_root / f"{mode}_{stage}__validation__{source}.npz") as archive:
                    arrays = {key: archive[key] for key in archive.files}
                keep = ~np.isin(arrays["record_id"], [rid for src, rid in excluded if src == source])
                cleaned = {key: value[keep] for key, value in arrays.items()}
                clean_sources[source] = _cached_prediction_metrics(cleaned)
                # ID разных источников могут совпадать: для общего счётчика — уникальное смещение.
                cleaned["record_id"] = cleaned["record_id"] + (0 if source == "open_ee" else 1_000_000)
                clean_arrays.append(cleaned)
                target, prediction = arrays["target"], arrays["prediction"]
                state = np.where(arrays["applicable"], prediction, -999)
                source_joint[source] = float(np.mean(state == target))
                correct += int(np.count_nonzero(state == target))
                total += len(target)
                for record_id in np.unique(arrays["record_id"]):
                    mask = (arrays["record_id"] == record_id) & (target != -999)
                    y, p = target[mask], prediction[mask]
                    key = (source, groups[(source, int(record_id))])
                    counts.setdefault(key, np.zeros(4, dtype=np.int64))
                    counts[key] += np.array([np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0)),
                                             np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p == 0))])
            grouped[stage] = counts
            comparison[stage] = {
                "metrics": validation["overall"], "by_source": validation["by_source"],
                "joint_state_accuracy": correct / total, "joint_state_accuracy_by_source": source_joint,
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "excluding_training_duplicates": {
                    "overall": _cached_prediction_metrics({key: np.concatenate([a[key] for a in clean_arrays])
                                                           for key in clean_arrays[0]}),
                    "by_source": clean_sources,
                },
            }
        keys = sorted(grouped["expert_best"])
        if set(keys) != set(grouped["weak_initial"]):
            raise ValueError("Несовпадающие группы файлов")
        matrices = {stage: np.asarray([values[key] for key in keys]) for stage, values in grouped.items()}
        strata = [[i for i, key in enumerate(keys) if key[0] == source] for source in ("open_ee", "french_rte")]
        rng = np.random.default_rng(20260910)
        boot = []
        for _ in range(bootstrap_repetitions):
            indices = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata if indices])
            scores = [_direction_metrics_from_counts(*map(int, matrices[stage][indices].sum(axis=0)))["macro_f1_score"]
                      for stage in ("weak_initial", "expert_best")]
            boot.append([scores[0], scores[1], scores[1] - scores[0]])
        comparison["bootstrap_95_percentile"] = dict(zip(
            ("weak_initial", "expert_best", "paired_improvement"), np.quantile(boot, [.025, .975], axis=0).T.tolist()))
        comparison["independent_file_groups"] = len(keys)
        comparison["grid_signature"] = reference_grid
        if common_grid is not None and common_grid != reference_grid:
            raise ValueError("Архитектуры оценены на разных сетках")
        common_grid = reference_grid
        expert_bootstraps[mode] = np.asarray(boot)[:, 1]
        result["models"][mode] = comparison
    result["architecture_differences_95_percentile"] = {
        f"snapshot_5_minus_{mode}": np.quantile(
            expert_bootstraps["snapshot_5"] - expert_bootstraps[mode], [.025, .975]
        ).tolist() for mode in ("snapshot_2", "sequence_1_8")
    }
    path = output_root / "paired_comparison_summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Готово] Парное сравнение: {path}", flush=True)
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
    RUN_NEURAL_EVALUATION = True  # Повторить после завершения нового expert-обучения.
    RUN_ANALYTICAL_V6 = True       # Запускать только после complete обоих pdr_labels_v6.
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
            "experiments/phase5/pdr_expert_snapshot_5_stride5/best_model.pt",
            "experiments/phase5/pdr_expert_snapshot_5_stride5/config.json",
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
    output_root = PROJECT_ROOT / "experiments/phase5/pdr_expert_evaluation_v3"
    if RUN_NEURAL_EVALUATION:
        for name, checkpoint, config in RUNS:
            evaluate_checkpoint(
                PROJECT_ROOT / checkpoint,
                PROJECT_ROOT / config,
                output_root / f"{name}.json",
                splits=SPLITS,
                max_samples_per_record=MAX_SAMPLES_PER_RECORD,
                num_workers=NUM_WORKERS,
            )
        # Expert запускался от latest, а не от weak-best: парный прирост
        # следует считать именно относительно фактической инициализации.
        for mode in ("snapshot_2", "snapshot_5", "sequence_1_8"):
            config_path = PROJECT_ROOT / f"experiments/phase5/pdr_expert_{mode}_stride5/config.json"
            initialization = Path(json.loads(config_path.read_text(encoding="utf-8"))["initialization"])
            if not initialization.exists():
                initialization = PROJECT_ROOT / f"experiments/phase5/pdr_weak_{mode}_stride5" / initialization.name
            evaluate_checkpoint(
                initialization, config_path, output_root / f"{mode}_weak_initial.json",
                splits=SPLITS, max_samples_per_record=MAX_SAMPLES_PER_RECORD,
                num_workers=NUM_WORKERS,
            )
    if RUN_ANALYTICAL_V6:
        evaluate_analytical_algorithms_on_neural_grid(
            PROJECT_ROOT / RUNS[3][2],
            PROJECT_ROOT / "data/phase5/pdr_labels_v6",
            output_root / "analytical_validation_v6.json",
            split="validation",
            max_samples_per_record=MAX_SAMPLES_PER_RECORD,
        )
    if RUN_NEURAL_EVALUATION:
        summarize_current_evaluations(output_root)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
