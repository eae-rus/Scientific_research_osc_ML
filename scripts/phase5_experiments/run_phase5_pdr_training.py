"""Двухэтапное обучение нейросетевого РНМ: weak teacher -> экспертный слой.

Первый этап обучается по адаптивному РНМ v5. Второй начинает с checkpoint
первого этапа, добавляет экспертные метки и небольшой replay автоматической
разметки, чтобы не забыть обычные режимы. Каждая осциллограмма остаётся целиком
в исходной части research-strict split.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.dataset_registry import create_source
from osc_tools.ml.phase5_contracts import TimebaseContract
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
from osc_tools.pdr.pdr_dataset import PDRTaskDataset
from scripts.phase5_experiments.progress import ProgressReporter


@dataclass
class PDRTrainingConfig:
    stage: str = "weak"  # weak | expert
    temporal_mode: str = "snapshot_5"
    label_stride_samples: int = 5
    feature_version: str = "B"
    model_preset: str = "small"
    epochs: int = 40
    samples_per_epoch: int = 20_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_samples_per_record: int = 64
    weak_open_records: int = 4_000
    weak_french_records: int = 1_000
    weak_validation_records_per_source: int = 400
    expert_replay_open_records: int = 300
    expert_fraction: float = 0.75
    open_ee_fraction_within_expert: float = 0.75
    augmentation_probability: float = 0.5
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    use_ssl_initialization: bool = True


MODEL_PRESETS = {
    "small": {"d_model": 64, "num_heads": 4, "num_layers": 4, "d_ff": 256},
    "medium": {"d_model": 128, "num_heads": 8, "num_layers": 6, "d_ff": 512},
    "heavy": {"d_model": 256, "num_heads": 8, "num_layers": 8, "d_ff": 1024},
}


def _imports():
    try:
        import torch
        from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
        from osc_tools.ml.models.transformer import PhysicalKANTransformer
        from osc_tools.pdr.pdr_trainer import (
            PDRTaskHead,
            evaluate_pdr_metrics,
            extract_backbone_features,
            pdr_combined_loss,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("Нужно основное окружение проекта с PyTorch") from exc
    return (
        torch, ConcatDataset, DataLoader, WeightedRandomSampler,
        PhysicalKANTransformer, PDRTaskHead, evaluate_pdr_metrics,
        extract_backbone_features, pdr_combined_loss,
    )


def _load_splits() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "data/phase5/research_strict_splits.json").read_text(encoding="utf-8"))


def _stable_subset(values: Sequence[int], count: int, seed: int, namespace: str) -> list[int]:
    def key(value: int) -> bytes:
        return hashlib.sha256(f"{namespace}:{seed}:{int(value)}".encode()).digest()
    ordered = sorted((int(value) for value in values), key=key)
    return ordered[: min(count, len(ordered))]


def _dataset(
    cfg: PDRTrainingConfig,
    source_name: str,
    indices: Sequence[int],
    labels_path: Path,
    *,
    augment: bool,
    max_samples_per_record: int | None = None,
) -> PDRTaskDataset:
    source = create_source(PROJECT_ROOT / "data/phase5/datasets_registry.json", source_name)
    # Реальные f_adc/f_network читаются отдельно для каждой записи; этот
    # контракт используется только как проверенный fallback.
    fallback = TimebaseContract.create(1000.0, 50.0)
    return PDRTaskDataset(
        source,
        list(map(int, indices)),
        labels_path,
        fallback,
        temporal_mode=cfg.temporal_mode,
        feature_version=cfg.feature_version,
        teacher_algorithm_id="adaptive_pdr_mir" if cfg.stage == "weak" else None,
        sample_subset="train",
        index_stride_samples=cfg.label_stride_samples,
        max_samples_per_record=max_samples_per_record or cfg.max_samples_per_record,
        augmentation_seed=cfg.seed if augment else None,
        augmentation_probability=cfg.augmentation_probability if augment else 0.0,
    )


def _expert_indices(source: str, split: str) -> list[int]:
    import csv
    path = PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1/records.csv"
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source"] == source and row["split"] == split:
                result.append(int(row["record_id"]))
    return result


def _weak_groups(cfg: PDRTrainingConfig, split: str, augment: bool):
    splits = _load_splits()["sources"]
    labels_root = PROJECT_ROOT / "data/phase5/pdr_labels_v5"
    if split == "train":
        limits = {"open_ee": cfg.weak_open_records, "french_rte": cfg.weak_french_records}
    else:
        limits = {name: cfg.weak_validation_records_per_source for name in ("open_ee", "french_rte")}
    groups = []
    for offset, source in enumerate(("open_ee", "french_rte")):
        indices = _stable_subset(
            splits[source]["splits"][split], limits[source], cfg.seed + offset, f"weak:{split}:{source}"
        )
        groups.append((source, _dataset(cfg, source, indices, labels_root / source, augment=augment)))
    return groups


def _expert_groups(cfg: PDRTrainingConfig, split: str, augment: bool):
    expert_root = PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1"
    groups = []
    for source in ("open_ee", "french_rte"):
        indices = _expert_indices(source, split)
        if indices:
            groups.append((f"expert_{source}", _dataset(
                cfg, source, indices, expert_root / source, augment=augment,
                max_samples_per_record=max(cfg.max_samples_per_record, 256),
            )))
    if split == "train" and cfg.expert_replay_open_records > 0:
        splits = _load_splits()["sources"]
        replay_ids = _stable_subset(
            splits["open_ee"]["splits"]["train"],
            cfg.expert_replay_open_records,
            cfg.seed + 700,
            "expert:replay:open_ee",
        )
        # Вторая стадия читает автоматический архив явно через временную копию
        # конфигурации stage, иначе teacher_algorithm_id не был бы выбран.
        original_stage = cfg.stage
        cfg.stage = "weak"
        try:
            replay = _dataset(
                cfg, "open_ee", replay_ids,
                PROJECT_ROOT / "data/phase5/pdr_labels_v5/open_ee",
                augment=augment,
            )
        finally:
            cfg.stage = original_stage
        groups.append(("replay_open_ee", replay))
    return groups


def _collate(samples):
    torch, *_ = _imports()
    max_time = max(sample["features"].shape[0] for sample in samples)
    channels = samples[0]["features"].shape[1]
    batch = len(samples)
    features = torch.full((batch, max_time, channels), float("nan"), dtype=torch.float32)
    provenance = torch.zeros((batch, max_time, channels), dtype=torch.long)
    result = {"features": features, "provenance": provenance}
    scalar_keys = (
        "target_class", "target_applicable", "pdr_direction", "pdr_margin",
        "pdr_confidence", "warmup_mask", "expert_train_mask",
        "expert_transition_eval_mask", "record_id", "window_idx",
    )
    for row, sample in enumerate(samples):
        n_time = sample["features"].shape[0]
        features[row, :n_time] = sample["features"]
        provenance[row, :n_time] = sample["provenance"]
    for key in scalar_keys:
        values = [sample[key] for sample in samples]
        result[key] = torch.stack(values) if hasattr(values[0], "shape") else torch.tensor(values)
    return result


def _group_weights(cfg: PDRTrainingConfig, groups, stage: str) -> list[float]:
    if stage == "weak":
        shares = {"open_ee": 0.8, "french_rte": 0.2}
    else:
        expert_share = cfg.expert_fraction
        shares = {
            "expert_open_ee": expert_share * cfg.open_ee_fraction_within_expert,
            "expert_french_rte": expert_share * (1.0 - cfg.open_ee_fraction_within_expert),
            "replay_open_ee": 1.0 - expert_share,
        }
    weights: list[float] = []
    present_total = sum(shares.get(name, 0.0) for name, dataset in groups if len(dataset))
    for name, dataset in groups:
        share = shares.get(name, 0.0) / max(present_total, 1e-12)
        weights.extend([share / max(1, len(dataset))] * len(dataset))
    return weights


def _build_model(cfg: PDRTrainingConfig, ssl_checkpoint: Path | None, weak_checkpoint: Path | None):
    (
        torch, _, _, _, PhysicalKANTransformer, PDRTaskHead, *_
    ) = _imports()
    preset = MODEL_PRESETS[cfg.model_preset]
    builder = SpectralFeatureBuilder(SpectralFeatureConfig(cfg.feature_version))
    model = PhysicalKANTransformer(
        num_input_channels=len(builder.schema.names),
        ssl_output_channels=len(builder.schema.names),
        **preset,
        dropout=0.1,
        max_seq_len=256,
        cyclic_angle_encoding=True,
        use_provenance_embedding=True,
    )
    head = PDRTaskHead(preset["d_model"], use_margin_head=True)
    initialization = "random"
    if weak_checkpoint is not None:
        checkpoint = torch.load(weak_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["backbone_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        initialization = str(weak_checkpoint)
    elif cfg.use_ssl_initialization and ssl_checkpoint is not None:
        checkpoint = torch.load(ssl_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        initialization = str(ssl_checkpoint)
    return model, head, initialization


def run(
    cfg: PDRTrainingConfig,
    output_dir: Path,
    *,
    ssl_checkpoint: Path | None = None,
    weak_checkpoint: Path | None = None,
) -> None:
    (
        torch, ConcatDataset, DataLoader, WeightedRandomSampler, _, _,
        evaluate_pdr_metrics, extract_backbone_features, pdr_combined_loss,
    ) = _imports()
    if cfg.stage not in ("weak", "expert"):
        raise ValueError("stage должен быть weak или expert")
    if cfg.stage == "expert" and weak_checkpoint is None:
        raise ValueError("Для expert-stage обязателен checkpoint weak-stage")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else "cpu" if cfg.device == "auto" else cfg.device
    )
    train_groups = (
        _weak_groups(cfg, "train", True) if cfg.stage == "weak"
        else _expert_groups(cfg, "train", True)
    )
    validation_groups = (
        _weak_groups(cfg, "validation", False) if cfg.stage == "weak"
        else _expert_groups(cfg, "validation", False)
    )
    train_groups = [(name, dataset) for name, dataset in train_groups if len(dataset)]
    validation_groups = [(name, dataset) for name, dataset in validation_groups if len(dataset)]
    if not train_groups or not validation_groups:
        raise RuntimeError("Пустой train/validation после применения контрактов")
    train_dataset = ConcatDataset([dataset for _, dataset in train_groups])
    validation_dataset = ConcatDataset([dataset for _, dataset in validation_groups])
    weights = torch.tensor(_group_weights(cfg, train_groups, cfg.stage), dtype=torch.double)
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=cfg.samples_per_epoch, replacement=True, generator=generator
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )
    model, head, initialization = _build_model(cfg, ssl_checkpoint, weak_checkpoint)
    model, head = model.to(device), head.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=1e-6
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(cfg) | {
        "initialization": initialization,
        "train_groups": {name: len(dataset) for name, dataset in train_groups},
        "validation_groups": {name: len(dataset) for name, dataset in validation_groups},
        "transition_exclusion_ms": 5.0,
        "direction_classes": {"REVERSE": 0, "FORWARD": 1},
        "unlabeled_is_applicability_target": True,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "training_log.jsonl"
    best_score = -1.0
    for epoch in range(cfg.epochs):
        started = time.time()
        for _, dataset in train_groups:
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(epoch)
        model.train(); head.train()
        total_loss = 0.0
        progress = ProgressReporter(f"PDR {cfg.stage} epoch {epoch + 1}/{cfg.epochs}", len(train_loader))
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            latent = extract_backbone_features(model, batch, str(device))
            outputs = head(latent)
            targets = {
                "target_class": batch["target_class"].to(device),
                "target_applicable": batch["target_applicable"].to(device),
                "pdr_margin": batch["pdr_margin"].to(device),
                "pdr_confidence": batch["pdr_confidence"].to(device),
                "supervised_mask": batch["expert_train_mask"].to(device),
            }
            loss = pdr_combined_loss(outputs, targets, margin_loss_weight=0.5 if cfg.stage == "weak" else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            progress.update(batch_index)
        progress.finish()
        scheduler.step()
        metrics = evaluate_pdr_metrics(model, head, validation_loader, str(device))
        score = float(metrics.get("f1_score", 0.0)) + float(metrics.get("applicability_f1_score", 0.0))
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader)),
            "validation": metrics,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = {
            "epoch": epoch + 1,
            "backbone_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_payload,
            "validation": metrics,
        }
        temporary = output_dir / "latest_checkpoint.tmp"
        torch.save(state, temporary); temporary.replace(output_dir / "latest_checkpoint.pt")
        if score > best_score:
            best_score = score
            torch.save(state, output_dir / "best_model.pt")
        print(json.dumps(record, ensure_ascii=False))
    (output_dir / "run_summary.json").write_text(json.dumps({
        "finished_at": datetime.now().isoformat(),
        "best_selection_score": best_score,
        "epochs": cfg.epochs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_output(cfg: PDRTrainingConfig, smoke: bool) -> Path:
    name = f"pdr_{cfg.stage}_{cfg.temporal_mode}_stride{cfg.label_stride_samples}"
    if smoke:
        name = "smoke_" + name
    return PROJECT_ROOT / "experiments/phase5" / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("weak", "expert"), default="weak")
    parser.add_argument("--temporal-mode", choices=("snapshot_2", "snapshot_5", "sequence_1_8"), default="snapshot_5")
    parser.add_argument("--label-stride-samples", type=int, choices=(1, 2, 5), default=5)
    parser.add_argument("--model-preset", choices=tuple(MODEL_PRESETS), default="small")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ssl-checkpoint", type=Path, default=None)
    parser.add_argument("--weak-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    cfg = PDRTrainingConfig(
        stage=args.stage,
        temporal_mode=args.temporal_mode,
        label_stride_samples=args.label_stride_samples,
        model_preset=args.model_preset,
    )
    if args.smoke:
        cfg.epochs = 1
        cfg.samples_per_epoch = 32
        cfg.batch_size = 8
        cfg.weak_open_records = 8
        cfg.weak_french_records = 4
        cfg.weak_validation_records_per_source = 4
        cfg.expert_replay_open_records = 4
        cfg.max_samples_per_record = 4
    if args.epochs is not None:
        cfg.epochs = args.epochs
    ssl_checkpoint = args.ssl_checkpoint or (
        PROJECT_ROOT / f"experiments/phase5/pretrain_b_{args.model_preset}/best_model.pt"
    )
    weak_checkpoint = args.weak_checkpoint
    if cfg.stage == "expert" and weak_checkpoint is None:
        weak_checkpoint = PROJECT_ROOT / (
            f"experiments/phase5/pdr_weak_{cfg.temporal_mode}_stride{cfg.label_stride_samples}/best_model.pt"
        )
    run(cfg, args.output_dir or _default_output(cfg, args.smoke),
        ssl_checkpoint=ssl_checkpoint, weak_checkpoint=weak_checkpoint)
    return 0


def run_manual() -> None:
    """Параметры ручного запуска; начать со SMOKE=True и STAGE='weak'."""
    STAGE = "weak"                 # weak, затем expert
    TEMPORAL_MODE = "snapshot_5"   # затем snapshot_2 и sequence_1_8
    LABEL_STRIDE_SAMPLES = 5        # основной; абляция 2 и 1
    MODEL_PRESET = "small"
    SMOKE = True

    cfg = PDRTrainingConfig(
        stage=STAGE,
        temporal_mode=TEMPORAL_MODE,
        label_stride_samples=LABEL_STRIDE_SAMPLES,
        model_preset=MODEL_PRESET,
    )
    if SMOKE:
        cfg.epochs = 1; cfg.samples_per_epoch = 32; cfg.batch_size = 8
        cfg.weak_open_records = 8; cfg.weak_french_records = 4
        cfg.weak_validation_records_per_source = 4
        cfg.expert_replay_open_records = 4; cfg.max_samples_per_record = 4
    ssl_checkpoint = PROJECT_ROOT / f"experiments/phase5/pretrain_b_{MODEL_PRESET}/best_model.pt"
    weak_checkpoint = None
    if STAGE == "expert":
        weak_checkpoint = PROJECT_ROOT / (
            f"experiments/phase5/pdr_weak_{TEMPORAL_MODE}_stride{LABEL_STRIDE_SAMPLES}/best_model.pt"
        )
    run(cfg, _default_output(cfg, SMOKE),
        ssl_checkpoint=ssl_checkpoint, weak_checkpoint=weak_checkpoint)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
