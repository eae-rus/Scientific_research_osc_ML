"""Двухэтапное обучение нейросетевого РНМ: weak teacher -> экспертный слой.

Первый этап обучается по адаптивному РНМ v5. Второй начинает с checkpoint
первого этапа, добавляет экспертные метки и небольшой replay автоматической
разметки, чтобы не забыть обычные режимы. Каждая осциллограмма остаётся целиком
в исходной части research-strict split.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
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
    validation_max_samples_per_record: int = 32
    expert_validation_max_samples_per_record: int = 256
    weak_open_records: int = 4_000
    weak_french_records: int = 1_000
    rotate_weak_records_each_epoch: bool = True
    min_samples_per_record_per_epoch: int = 4
    transition_sampling_fraction: float = 0.5
    transition_sampling_radius_ms: float = 20.0
    weak_validation_records_per_source: int = 400
    expert_replay_open_records: int = 1_000
    rotate_expert_samples_each_epoch: bool = True
    rotate_expert_replay_records_each_epoch: bool = True
    expert_fraction: float = 0.75
    open_ee_fraction_within_expert: float = 0.75
    expert_labels_root: str = "data/phase5/pdr_expert_labels_v1"
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


def _checkpoint_filename(kind: str) -> str:
    if kind == "best":
        return "best_model.pt"
    if kind == "latest":
        return "latest_checkpoint.pt"
    raise ValueError("Вид checkpoint должен быть 'best' или 'latest'")


def _restart_checkpoint(output_dir: Path, kind: str) -> Path:
    """Найти checkpoint текущего либо последнего архивированного запуска."""

    filename = _checkpoint_filename(kind)
    current = output_dir / filename
    if current.exists():
        return current
    archived = sorted(
        (path / filename for path in output_dir.glob("archive_*") if path.is_dir()),
        reverse=True,
    )
    candidate = next((path for path in archived if path.exists()), None)
    if candidate is not None:
        print(
            f"[Инициализация] В корне checkpoint отсутствует; используется "
            f"последний архив: {candidate}",
            flush=True,
        )
        return candidate
    return current


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


def _epoch_record_pool(
    values: Sequence[int],
    max_records: int,
    seed: int,
    namespace: str,
    epoch: int,
) -> tuple[list[int], int, int]:
    """Вернуть очередной непересекающийся пул записей стабильного цикла.

    Полный train-split один раз детерминированно перемешивается хешем. Затем он
    делится на последовательные части не более ``max_records``. Эпохи циклически
    обходят эти части, поэтому ограничение RAM/I/O не превращается в постоянное
    исключение большей части архива.
    """

    if max_records <= 0:
        raise ValueError("max_records должен быть положительным")
    # PDR-разметка хранится блоками по 64 записи. Полностью случайный порядок
    # заставил бы каждый пул открывать почти все shards. Перемешиваем сами блоки,
    # сохраняя последовательное чтение внутри них.
    blocks: dict[int, list[int]] = {}
    for value in values:
        blocks.setdefault(int(value) // 64, []).append(int(value))

    def block_key(block_id: int) -> bytes:
        return hashlib.sha256(
            f"{namespace}:block:{seed}:{block_id}".encode()
        ).digest()

    ordered = [
        value
        for block_id in sorted(blocks, key=block_key)
        for value in sorted(blocks[block_id])
    ]
    if not ordered:
        return [], 0, 0
    n_pools = (len(ordered) + max_records - 1) // max_records
    pool_index = int(epoch) % n_pools
    # Делим максимально равномерно: последняя эпоха цикла не должна содержать
    # заметно меньше записей и тем самым давать каждой из них больший вес.
    base_size, remainder = divmod(len(ordered), n_pools)
    start = pool_index * base_size + min(pool_index, remainder)
    size = base_size + (1 if pool_index < remainder else 0)
    return ordered[start : start + size], pool_index, n_pools


def _scheduled_unique_records(
    split_records: int, max_records: int, completed_epochs: int
) -> int:
    """Число уникальных записей, запланированных с начала weak-цикла."""

    if split_records <= 0 or completed_epochs <= 0:
        return 0
    pool_count = (split_records + max_records - 1) // max_records
    pools_seen = min(completed_epochs, pool_count)
    base_size, remainder = divmod(split_records, pool_count)
    return pools_seen * base_size + min(pools_seen, remainder)


def _weak_record_indices(
    cfg: PDRTrainingConfig,
    source: str,
    split: str,
    *,
    epoch: int | None = None,
    all_records: bool = False,
) -> tuple[list[int], dict[str, int | str | bool]]:
    split_values = _load_splits()["sources"][source]["splits"][split]
    if all_records:
        indices = sorted(map(int, split_values))
        return indices, {
            "policy": "all_records",
            "pool_index": 0,
            "pool_count": 1,
            "split_records": len(split_values),
        }

    if split == "train":
        limit = cfg.weak_open_records if source == "open_ee" else cfg.weak_french_records
        if cfg.rotate_weak_records_each_epoch:
            if epoch is None:
                raise ValueError("Для ротации weak train-пула требуется номер эпохи")
            indices, pool_index, pool_count = _epoch_record_pool(
                split_values,
                limit,
                cfg.seed + (0 if source == "open_ee" else 1),
                f"weak:{split}:{source}",
                epoch,
            )
            return indices, {
                "policy": "rotating_epoch_pool",
                "pool_index": pool_index,
                "pool_count": pool_count,
                "split_records": len(split_values),
            }
    else:
        limit = cfg.weak_validation_records_per_source

    indices = _stable_subset(
        split_values,
        limit,
        cfg.seed + (0 if source == "open_ee" else 1),
        f"weak:{split}:{source}",
    )
    return indices, {
        "policy": "fixed_subset",
        "pool_index": 0,
        "pool_count": 1,
        "split_records": len(split_values),
    }


def _dataset(
    cfg: PDRTrainingConfig,
    source_name: str,
    indices: Sequence[int],
    labels_path: Path,
    *,
    augment: bool,
    max_samples_per_record: int | None = None,
    index_selection_seed: int | None = None,
) -> PDRTaskDataset:
    ordered_indices = sorted(map(int, indices))
    label = (
        f"Индекс PDR {cfg.stage}/{source_name}"
        f" ({'train' if augment else 'validation'})"
    )
    reporter = ProgressReporter(label, max(1, len(ordered_indices)), unit="зап.")

    def report_index(completed: int, total: int) -> None:
        if total:
            reporter.update(completed)

    source = create_source(PROJECT_ROOT / "data/phase5/datasets_registry.json", source_name)
    # Реальные f_adc/f_network читаются отдельно для каждой записи; этот
    # контракт используется только как проверенный fallback.
    fallback = TimebaseContract.create(1000.0, 50.0)
    dataset = PDRTaskDataset(
        source,
        ordered_indices,
        labels_path,
        fallback,
        temporal_mode=cfg.temporal_mode,
        feature_version=cfg.feature_version,
        teacher_algorithm_id="adaptive_pdr_mir" if cfg.stage == "weak" else None,
        sample_subset="train",
        index_stride_samples=cfg.label_stride_samples,
        max_samples_per_record=max_samples_per_record or cfg.max_samples_per_record,
        index_selection_seed=index_selection_seed,
        transition_sampling_fraction=cfg.transition_sampling_fraction,
        transition_sampling_radius_ms=cfg.transition_sampling_radius_ms,
        augmentation_seed=cfg.seed if augment else None,
        augmentation_probability=cfg.augmentation_probability if augment else 0.0,
        index_progress_callback=report_index,
    )
    reporter.finish()
    print(
        f"[Готово] {label}: {len(ordered_indices):,} записей, "
        f"{len(dataset):,} индексированных целевых точек.",
        flush=True,
    )
    return dataset


def _expert_root(cfg: PDRTrainingConfig) -> Path:
    root = Path(cfg.expert_labels_root)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _expert_indices(expert_root: Path, source: str, split: str) -> list[int]:
    import csv
    path = expert_root / "records.csv"
    if not path.exists():
        raise FileNotFoundError(f"Реестр ручной разметки не найден: {path}")
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source"] == source and row["split"] == split:
                result.append(int(row["record_id"]))
    return result


def _weak_groups(
    cfg: PDRTrainingConfig,
    split: str,
    augment: bool,
    *,
    epoch: int | None = None,
    all_records: bool = False,
):
    labels_root = PROJECT_ROOT / "data/phase5/pdr_labels_v5"
    groups = []
    for source_offset, source in enumerate(("open_ee", "french_rte")):
        indices, pool = _weak_record_indices(
            cfg, source, split, epoch=epoch, all_records=all_records
        )
        print(
            f"[Пул] {source}/{split}: {len(indices):,} из "
            f"{int(pool['split_records']):,}; политика={pool['policy']}; "
            f"часть={int(pool['pool_index']) + 1}/{int(pool['pool_count'])}",
            flush=True,
        )
        groups.append((source, _dataset(
            cfg,
            source,
            indices,
            labels_root / source,
            augment=augment,
            max_samples_per_record=(
                cfg.max_samples_per_record
                if split == "train" else cfg.validation_max_samples_per_record
            ),
            index_selection_seed=(
                cfg.seed + int(epoch) * 10_000_019 + source_offset * 97_409
                if split == "train" and cfg.rotate_weak_records_each_epoch
                else None
            ),
        )))
    return groups


def _weak_pool_summary(cfg: PDRTrainingConfig, epoch: int) -> dict[str, dict[str, int | str | bool]]:
    result: dict[str, dict[str, int | str | bool]] = {}
    for source in ("open_ee", "french_rte"):
        indices, pool = _weak_record_indices(cfg, source, "train", epoch=epoch)
        result[source] = dict(pool) | {"scheduled_records": len(indices)}
    return result


def _expert_groups(
    cfg: PDRTrainingConfig,
    split: str,
    augment: bool,
    *,
    epoch: int | None = None,
):
    expert_root = _expert_root(cfg)
    if not expert_root.exists():
        raise FileNotFoundError(f"Папка ручной разметки не найдена: {expert_root}")
    groups = []
    if split == "train" and cfg.rotate_expert_samples_each_epoch and epoch is None:
        raise ValueError("Для ротации expert train-точек требуется номер эпохи")
    for source in ("open_ee", "french_rte"):
        indices = _expert_indices(expert_root, source, split)
        if indices:
            per_record_limit = (
                max(cfg.max_samples_per_record, 256)
                if split == "train" else cfg.expert_validation_max_samples_per_record
            )
            groups.append((f"expert_{source}", _dataset(
                cfg, source, indices, expert_root / source, augment=augment,
                max_samples_per_record=per_record_limit,
                index_selection_seed=(
                    cfg.seed
                    + int(epoch) * 10_000_019
                    + (0 if source == "open_ee" else 97_409)
                    if split == "train" and cfg.rotate_expert_samples_each_epoch
                    else None
                ),
            )))
    if split == "train" and cfg.expert_replay_open_records > 0:
        splits = _load_splits()["sources"]
        expert_open_train = set(_expert_indices(expert_root, "open_ee", "train"))
        replay_train = [
            int(record_id)
            for record_id in splits["open_ee"]["splits"]["train"]
            if int(record_id) not in expert_open_train
        ]
        if cfg.rotate_expert_replay_records_each_epoch:
            if epoch is None:
                raise ValueError("Для ротации expert replay-пула требуется номер эпохи")
            replay_ids, replay_pool_index, replay_pool_count = _epoch_record_pool(
                replay_train,
                cfg.expert_replay_open_records,
                cfg.seed + 700,
                "expert:replay:open_ee",
                epoch,
            )
            replay_policy = "rotating_epoch_pool"
        else:
            replay_ids = _stable_subset(
                replay_train,
                cfg.expert_replay_open_records,
                cfg.seed + 700,
                "expert:replay:open_ee",
            )
            replay_pool_index, replay_pool_count = 0, 1
            replay_policy = "fixed_subset"
        print(
            f"[Пул] expert replay open_ee/train: {len(replay_ids):,} из "
            f"{len(replay_train):,}; политика={replay_policy}; "
            f"часть={replay_pool_index + 1}/{replay_pool_count}; "
            f"исключено expert train={len(expert_open_train):,}",
            flush=True,
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
                index_selection_seed=(
                    cfg.seed + int(epoch) * 10_000_019 + 700
                    if cfg.rotate_expert_samples_each_epoch
                    else None
                ),
            )
        finally:
            cfg.stage = original_stage
        groups.append(("replay_open_ee", replay))
    return groups


def _expert_replay_pool_summary(
    cfg: PDRTrainingConfig, epoch: int
) -> dict[str, int | float | str] | None:
    """План и накопленный охват непересекающегося с expert Open_EE replay."""

    if cfg.expert_replay_open_records <= 0:
        return None
    expert_root = _expert_root(cfg)
    excluded = set(_expert_indices(expert_root, "open_ee", "train"))
    split_values = _load_splits()["sources"]["open_ee"]["splits"]["train"]
    population = [int(value) for value in split_values if int(value) not in excluded]
    if cfg.rotate_expert_replay_records_each_epoch:
        pool, pool_index, pool_count = _epoch_record_pool(
            population,
            cfg.expert_replay_open_records,
            cfg.seed + 700,
            "expert:replay:open_ee",
            epoch,
        )
        covered = _scheduled_unique_records(
            len(population), cfg.expert_replay_open_records, epoch + 1
        )
        policy = "rotating_epoch_pool"
    else:
        pool = _stable_subset(
            population,
            cfg.expert_replay_open_records,
            cfg.seed + 700,
            "expert:replay:open_ee",
        )
        pool_index, pool_count = 0, 1
        covered = len(pool)
        policy = "fixed_subset"
    return {
        "policy": policy,
        "population_records": len(population),
        "excluded_expert_train_records": len(excluded),
        "scheduled_records": len(pool),
        "pool_index": pool_index,
        "pool_count": pool_count,
        "scheduled_unique_records_through_epoch": covered,
        "scheduled_coverage_fraction": covered / max(1, len(population)),
    }


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
        "expert_transition_eval_mask", "source_id", "record_id", "window_idx",
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


def _mandatory_record_sample_indices(
    groups, seed: int, samples_per_record: int
) -> list[int]:
    """Выбрать несколько индексированных точек каждой пригодной записи пула."""

    if samples_per_record <= 0:
        raise ValueError("samples_per_record должен быть положительным")

    rng = np.random.default_rng(seed)
    result: list[int] = []
    offset = 0
    for _, dataset in groups:
        positions_by_record: dict[int, list[int]] = {}
        for local_index, (record_id, _) in enumerate(dataset.samples):
            positions_by_record.setdefault(int(record_id), []).append(local_index)
        for record_id in sorted(positions_by_record):
            positions = positions_by_record[record_id]
            count = min(samples_per_record, len(positions))
            chosen = rng.choice(positions, size=count, replace=False)
            result.extend(offset + int(position) for position in chosen)
        offset += len(dataset)
    return result


def _training_sampler(
    torch,
    cfg: PDRTrainingConfig,
    groups,
    epoch: int,
    weighted_sampler_class,
):
    weights = torch.tensor(_group_weights(cfg, groups, cfg.stage), dtype=torch.double)
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    use_mandatory_coverage = (
        (cfg.stage == "weak" and cfg.rotate_weak_records_each_epoch)
        or (
            cfg.stage == "expert"
            and (
                cfg.rotate_expert_samples_each_epoch
                or cfg.rotate_expert_replay_records_each_epoch
            )
        )
    )
    if not use_mandatory_coverage:
        return weighted_sampler_class(
            weights,
            num_samples=cfg.samples_per_epoch,
            replacement=True,
            generator=generator,
        )

    mandatory = _mandatory_record_sample_indices(
        groups,
        cfg.seed + epoch * 1_000_003,
        cfg.min_samples_per_record_per_epoch,
    )
    if len(mandatory) > cfg.samples_per_epoch:
        raise ValueError(
            "SAMPLES_PER_EPOCH меньше обязательной квоты текущего train-пула: "
            f"{cfg.samples_per_epoch:,} < {len(mandatory):,}. Увеличьте объём "
            "эпохи, уменьшите MIN_SAMPLES_PER_RECORD_PER_EPOCH либо "
            "уменьшите размер индексируемого пула."
        )
    remaining = cfg.samples_per_epoch - len(mandatory)
    sampled = list(mandatory)
    if remaining:
        extra = weighted_sampler_class(
            weights,
            num_samples=remaining,
            replacement=True,
            generator=generator,
        )
        sampled.extend(int(index) for index in extra)
    # Обязательные точки не должны образовывать отдельную первую часть эпохи.
    permutation = torch.randperm(len(sampled), generator=generator).tolist()
    return [sampled[index] for index in permutation]


def _planned_group_coverage(groups, sampler) -> dict[str, dict[str, int | float]]:
    """Описать гарантированный охват записей для явного epoch-sampler."""

    if not isinstance(sampler, list):
        return {}
    ends: list[int] = []
    total = 0
    for _, dataset in groups:
        total += len(dataset)
        ends.append(total)
    sampled: list[dict[int, int]] = [dict() for _ in groups]
    for global_index in sampler:
        group_index = bisect.bisect_right(ends, int(global_index))
        start = 0 if group_index == 0 else ends[group_index - 1]
        record_id, _ = groups[group_index][1].samples[int(global_index) - start]
        counts = sampled[group_index]
        counts[int(record_id)] = counts.get(int(record_id), 0) + 1
    result: dict[str, dict[str, int | float]] = {}
    for group_index, (name, dataset) in enumerate(groups):
        indexed_records = len({int(record_id) for record_id, _ in dataset.samples})
        counts = list(sampled[group_index].values())
        result[name] = {
            "indexed_records": indexed_records,
            "indexed_target_points": len(dataset),
            "sampled_records": len(counts),
            "record_coverage_fraction": len(counts) / max(1, indexed_records),
            "sampled_points_per_record_min": min(counts) if counts else 0,
            "sampled_points_per_record_median": float(np.median(counts)) if counts else 0.0,
            "sampled_points_per_record_max": max(counts) if counts else 0,
        }
    return result


def _export_training_history(log_path: Path, output_dir: Path) -> None:
    """Сохранить JSON и графики истории после каждой завершённой эпохи."""

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (output_dir / "training_history.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    epochs = [record["epoch"] for record in records]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(epochs, [record["train_loss"] for record in records], marker="o", label="train loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for key, label in (
        ("f1_score", "direction F1"),
        ("applicability_f1_score", "applicability F1"),
        ("accuracy", "direction accuracy"),
        ("applicability_accuracy", "applicability accuracy"),
    ):
        if all(key in record.get("validation", {}) for record in records):
            axes[1].plot(
                epochs,
                [record["validation"][key] for record in records],
                marker="o",
                label=label,
            )
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Метрика")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(figure)


def _archive_existing_output(output_dir: Path) -> Path | None:
    """Перед свежим запуском сохранить прежнюю историю и checkpoints."""

    log_path = output_dir / "training_log.jsonl"
    if not log_path.exists():
        return None
    archive_dir = output_dir / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for filename in (
        "training_log.jsonl",
        "training_history.json",
        "training_history.png",
        "config.json",
        "latest_checkpoint.pt",
        "best_model.pt",
        "run_summary.json",
    ):
        source = output_dir / filename
        if source.exists():
            shutil.move(str(source), str(archive_dir / filename))
    print(f"[Архив] Предыдущий запуск сохранён: {archive_dir}", flush=True)
    return archive_dir


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
    resume: bool = False,
) -> None:
    (
        torch, ConcatDataset, DataLoader, WeightedRandomSampler, _, _,
        evaluate_pdr_metrics, extract_backbone_features, pdr_combined_loss,
    ) = _imports()
    if cfg.stage not in ("weak", "expert"):
        raise ValueError("stage должен быть weak или expert")
    if cfg.stage == "expert" and weak_checkpoint is None and not resume:
        raise ValueError("Для expert-stage обязателен checkpoint weak-stage")
    positive_fields = (
        "epochs", "samples_per_epoch", "batch_size", "max_samples_per_record",
        "validation_max_samples_per_record", "expert_validation_max_samples_per_record",
        "weak_open_records",
        "weak_french_records", "weak_validation_records_per_source",
        "min_samples_per_record_per_epoch",
    )
    invalid = [name for name in positive_fields if int(getattr(cfg, name)) <= 0]
    if invalid:
        raise ValueError("Параметры должны быть положительными: " + ", ".join(invalid))
    if cfg.num_workers < 0:
        raise ValueError("num_workers не может быть отрицательным")
    if not 0.0 <= cfg.transition_sampling_fraction <= 1.0:
        raise ValueError("transition_sampling_fraction должен быть в диапазоне [0, 1]")
    if cfg.transition_sampling_radius_ms < 0:
        raise ValueError("transition_sampling_radius_ms не может быть отрицательным")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else "cpu" if cfg.device == "auto" else cfg.device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = output_dir / "latest_checkpoint.pt"
    if resume and not latest_checkpoint.exists():
        raise FileNotFoundError(
            f"Запрошено продолжение, но checkpoint отсутствует: {latest_checkpoint}"
        )
    resume_checkpoint = (
        torch.load(latest_checkpoint, map_location=device, weights_only=False)
        if resume else None
    )
    resume_epoch_hint = int(resume_checkpoint["epoch"]) if resume_checkpoint else 0
    if not resume:
        archive_dir = _archive_existing_output(output_dir)
        if (
            archive_dir is not None
            and weak_checkpoint is not None
            and weak_checkpoint.parent.resolve() == output_dir.resolve()
        ):
            archived_checkpoint = archive_dir / weak_checkpoint.name
            if archived_checkpoint.exists():
                weak_checkpoint = archived_checkpoint
                print(
                    f"[Инициализация] Checkpoint текущего каталога перенесён "
                    f"в архив и будет загружен оттуда: {weak_checkpoint}",
                    flush=True,
                )
    print(json.dumps({
        "этап": cfg.stage,
        "режим_времени": cfg.temporal_mode,
        "шаг_целевых_точек": cfg.label_stride_samples,
        "модель": cfg.model_preset,
        "эпох": cfg.epochs,
        "обучающих_примеров_на_эпоху": cfg.samples_per_epoch,
        "размер_batch": cfg.batch_size,
        "максимум_точек_на_запись_train": cfg.max_samples_per_record,
        "максимум_точек_на_запись_validation": cfg.validation_max_samples_per_record,
        "максимум_точек_на_экспертную_запись_validation": (
            cfg.expert_validation_max_samples_per_record if cfg.stage == "expert" else None
        ),
        "максимум_train_записей_Open_EE_на_эпоху": (
            cfg.weak_open_records if cfg.stage == "weak" else None
        ),
        "максимум_train_записей_French_RTE_на_эпоху": (
            cfg.weak_french_records if cfg.stage == "weak" else None
        ),
        "максимум_expert_replay_Open_EE_на_эпоху": (
            cfg.expert_replay_open_records if cfg.stage == "expert" else None
        ),
        "ротация_weak_train_между_эпохами": (
            cfg.rotate_weak_records_each_epoch if cfg.stage == "weak" else None
        ),
        "ротация_expert_точек_между_эпохами": (
            cfg.rotate_expert_samples_each_epoch if cfg.stage == "expert" else None
        ),
        "ротация_expert_replay_записей_между_эпохами": (
            cfg.rotate_expert_replay_records_each_epoch if cfg.stage == "expert" else None
        ),
        "минимум_точек_каждой_train_записи_на_эпоху": (
            cfg.min_samples_per_record_per_epoch
        ),
        "доля_кандидатов_около_переключений": (
            cfg.transition_sampling_fraction
        ),
        "радиус_зоны_переключения_мс": (
            cfg.transition_sampling_radius_ms
        ),
        "validation_записей_на_источник": cfg.weak_validation_records_per_source,
        "num_workers": cfg.num_workers,
        "устройство": str(device),
        "CUDA": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "продолжение": resume,
        "инициализация_SSL": str(ssl_checkpoint) if ssl_checkpoint is not None else None,
        "инициализация_PDR": str(weak_checkpoint) if weak_checkpoint is not None else None,
        "выход": str(output_dir),
    }, ensure_ascii=False, indent=2), flush=True)
    print(
        "[Подготовка] Строится компактный индекс меток. Сырые осциллограммы "
        "будут загружаться лениво по batch во время обучения.",
        flush=True,
    )
    initial_train_pool_started = time.time()
    train_groups = (
        _weak_groups(cfg, "train", True, epoch=resume_epoch_hint) if cfg.stage == "weak"
        else _expert_groups(cfg, "train", True, epoch=resume_epoch_hint)
    )
    initial_train_pool_prepare_seconds = time.time() - initial_train_pool_started
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
    sampler = _training_sampler(
        torch,
        cfg,
        train_groups,
        resume_epoch_hint,
        WeightedRandomSampler,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=_collate,
        persistent_workers=(
            cfg.num_workers > 0
            and not (cfg.stage == "weak" and cfg.rotate_weak_records_each_epoch)
        ),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
        persistent_workers=cfg.num_workers > 0,
    )
    model, head, initialization = _build_model(
        cfg,
        None if resume else ssl_checkpoint,
        None if resume else weak_checkpoint,
    )
    if resume:
        initialization = f"resume:{latest_checkpoint}"
    model, head = model.to(device), head.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=1e-6
    )
    config_payload = asdict(cfg) | {
        "initialization": initialization,
        "train_groups": {name: len(dataset) for name, dataset in train_groups},
        "validation_groups": {name: len(dataset) for name, dataset in validation_groups},
        "weak_record_pool_plan": (
            _weak_pool_summary(cfg, resume_epoch_hint) if cfg.stage == "weak" else None
        ),
        "expert_replay_pool_plan": (
            _expert_replay_pool_summary(cfg, resume_epoch_hint)
            if cfg.stage == "expert" else None
        ),
        "transition_exclusion_ms": 5.0,
        "direction_classes": {"REVERSE": 0, "FORWARD": 1},
        "unlabeled_is_applicability_target": True,
    }
    start_epoch = 0
    best_score = -1.0
    if resume:
        checkpoint = resume_checkpoint
        assert checkpoint is not None
        stored_config = checkpoint.get("config", {})
        contract_keys = (
            "stage", "temporal_mode", "label_stride_samples", "feature_version",
            "model_preset", "seed",
        )
        if cfg.stage == "expert":
            contract_keys += (
                "expert_labels_root",
                "expert_validation_max_samples_per_record",
                "expert_replay_open_records",
                "rotate_expert_samples_each_epoch",
                "rotate_expert_replay_records_each_epoch",
                "min_samples_per_record_per_epoch",
                "transition_sampling_fraction",
                "transition_sampling_radius_ms",
            )
        else:
            contract_keys += (
                "validation_max_samples_per_record",
                "weak_validation_records_per_source",
                "weak_open_records",
                "weak_french_records",
                "rotate_weak_records_each_epoch",
            )
            if cfg.rotate_weak_records_each_epoch:
                contract_keys += (
                    "min_samples_per_record_per_epoch",
                    "transition_sampling_fraction",
                    "transition_sampling_radius_ms",
                )
        stored_for_compare = dict(stored_config)
        # Старые weak-checkpoint использовали один неизменный пул во всех эпохах.
        stored_for_compare.setdefault("rotate_weak_records_each_epoch", False)
        # Checkpoint первого пилота создан до появления отдельного expert-лимита.
        # Его старое значение однозначно равно общему validation-лимиту.
        if (
            cfg.stage == "expert"
            and "expert_validation_max_samples_per_record" not in stored_for_compare
        ):
            stored_for_compare["expert_validation_max_samples_per_record"] = (
                stored_for_compare.get("validation_max_samples_per_record")
            )
        mismatches = [
            key for key in contract_keys
            if stored_for_compare.get(key) != config_payload.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Нельзя продолжить запуск с изменённым контрактом: "
                + ", ".join(mismatches)
            )
        model.load_state_dict(checkpoint["backbone_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_selection_score", -1.0))
        print(f"[Продолжение] Следующая эпоха: {start_epoch + 1}", flush=True)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "training_log.jsonl"
    print(
        f"[Старт] train batches={len(train_loader):,}; "
        f"validation batches={len(validation_loader):,}; "
        f"параметров модели={sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()):,}",
        flush=True,
    )
    for epoch in range(start_epoch, cfg.epochs):
        pool_prepare_seconds = (
            initial_train_pool_prepare_seconds if epoch == resume_epoch_hint else 0.0
        )
        rotate_train_pool = (
            (cfg.stage == "weak" and cfg.rotate_weak_records_each_epoch)
            or (cfg.stage == "expert" and cfg.rotate_expert_samples_each_epoch)
            or (cfg.stage == "expert" and cfg.rotate_expert_replay_records_each_epoch)
        )
        if rotate_train_pool and epoch != resume_epoch_hint:
            # Индекс остаётся ограниченным по размеру, но набор записей меняется
            # между эпохами. DataLoader пересоздаётся, чтобы Windows workers не
            # удерживали dataset предыдущего пула.
            pool_prepare_started = time.time()
            rebuilt_groups = (
                _weak_groups(cfg, "train", True, epoch=epoch)
                if cfg.stage == "weak"
                else _expert_groups(cfg, "train", True, epoch=epoch)
            )
            train_groups = [(name, dataset) for name, dataset in rebuilt_groups if len(dataset)]
            if not train_groups:
                raise RuntimeError(f"Пустой {cfg.stage} train-пул на эпохе {epoch + 1}")
            train_dataset = ConcatDataset([dataset for _, dataset in train_groups])
            sampler = _training_sampler(
                torch,
                cfg,
                train_groups,
                epoch,
                WeightedRandomSampler,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=cfg.num_workers,
                collate_fn=_collate,
                persistent_workers=False,
            )
            pool_prepare_seconds = time.time() - pool_prepare_started
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _, dataset in train_groups:
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(epoch)
        model.train(); head.train()
        total_loss = 0.0
        sampled_records: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
        sampled_record_counts: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}}
        progress = ProgressReporter(f"PDR {cfg.stage} epoch {epoch + 1}/{cfg.epochs}", len(train_loader))
        for batch_index, batch in enumerate(train_loader, start=1):
            for source_id, record_id in zip(
                batch["source_id"].tolist(), batch["record_id"].tolist()
            ):
                source_id = int(source_id)
                record_id = int(record_id)
                sampled_records[source_id].add(record_id)
                sampled_record_counts[source_id][record_id] = (
                    sampled_record_counts[source_id].get(record_id, 0) + 1
                )
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
        train_seconds = time.time() - started
        scheduler.step()
        validation_progress = ProgressReporter(
            f"PDR {cfg.stage} validation {epoch + 1}/{cfg.epochs}",
            len(validation_loader),
            unit="batch",
        )
        metrics = evaluate_pdr_metrics(
            model,
            head,
            validation_loader,
            str(device),
            progress_callback=validation_progress.update,
        )
        validation_progress.finish()
        score = float(metrics.get("f1_score", 0.0)) + float(metrics.get("applicability_f1_score", 0.0))
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader)),
            "validation": metrics,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "train_seconds": train_seconds,
            "train_pool_prepare_seconds": pool_prepare_seconds,
            "total_seconds_with_pool_prepare": (
                time.time() - started + pool_prepare_seconds
            ),
            "train_samples_per_second": cfg.samples_per_epoch / max(train_seconds, 1e-9),
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0
            ),
        }
        record["train_group_coverage"] = _planned_group_coverage(train_groups, sampler)
        if cfg.stage == "expert":
            record["expert_replay_pool"] = _expert_replay_pool_summary(cfg, epoch)
        if cfg.stage == "weak":
            pool_summary = _weak_pool_summary(cfg, epoch)
            source_ids = {"open_ee": 0, "french_rte": 1}
            datasets_by_source = {name: dataset for name, dataset in train_groups}
            for source, summary in pool_summary.items():
                limit = cfg.weak_open_records if source == "open_ee" else cfg.weak_french_records
                dataset = datasets_by_source.get(source)
                summary["indexed_eligible_records"] = (
                    len({int(record_id) for record_id, _ in dataset.samples})
                    if dataset is not None else 0
                )
                summary["indexed_target_points"] = len(dataset) if dataset is not None else 0
                summary["sampled_records"] = len(sampled_records[source_ids[source]])
                counts = list(sampled_record_counts[source_ids[source]].values())
                summary["sampled_points_per_record_min"] = min(counts) if counts else 0
                summary["sampled_points_per_record_median"] = (
                    float(np.median(counts)) if counts else 0.0
                )
                summary["sampled_points_per_record_max"] = max(counts) if counts else 0
                summary["index_selection"] = (
                    dict(dataset.index_selection_stats) if dataset is not None else {}
                )
                summary["scheduled_unique_records_through_epoch"] = _scheduled_unique_records(
                    int(summary["split_records"]), limit, epoch + 1
                )
                summary["scheduled_coverage_fraction"] = (
                    int(summary["scheduled_unique_records_through_epoch"])
                    / max(1, int(summary["split_records"]))
                )
            record["weak_record_pool"] = pool_summary
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        _export_training_history(log_path, output_dir)
        state = {
            "epoch": epoch + 1,
            "backbone_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_selection_score": max(best_score, score),
            "config": config_payload,
            "validation": metrics,
        }
        temporary = output_dir / "latest_checkpoint.tmp"
        torch.save(state, temporary); temporary.replace(latest_checkpoint)
        if score > best_score:
            best_score = score
            torch.save(state, output_dir / "best_model.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples-per-record", type=int, default=None)
    parser.add_argument("--validation-max-samples-per-record", type=int, default=None)
    parser.add_argument("--expert-validation-max-samples-per-record", type=int, default=None)
    parser.add_argument("--weak-open-records", type=int, default=None)
    parser.add_argument("--weak-french-records", type=int, default=None)
    parser.add_argument("--expert-replay-open-records", type=int, default=None)
    parser.add_argument(
        "--min-samples-per-record-per-epoch",
        type=int,
        default=None,
        help="Минимум различных train-точек каждой пригодной weak-записи в эпохе",
    )
    parser.add_argument(
        "--transition-sampling-fraction",
        type=float,
        default=None,
        help="Доля ограниченного train-индекса из зон около смены teacher-метки",
    )
    parser.add_argument(
        "--transition-sampling-radius-ms",
        type=float,
        default=None,
        help="Полуширина зоны около переключения teacher в миллисекундах",
    )
    parser.add_argument(
        "--fixed-weak-record-pool",
        action="store_true",
        help=(
            "Не ротировать ограниченный weak train-пул между эпохами. "
            "Нужно только для точного воспроизведения старого пилота."
        ),
    )
    parser.add_argument(
        "--fixed-expert-sampling",
        action="store_true",
        help="Не менять ограниченный индекс expert-точек между эпохами",
    )
    parser.add_argument(
        "--fixed-expert-replay-pool",
        action="store_true",
        help="Не ротировать weak replay-пул на expert-stage",
    )
    parser.add_argument("--validation-records-per-source", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ssl-checkpoint", type=Path, default=None)
    parser.add_argument("--weak-checkpoint", type=Path, default=None)
    parser.add_argument("--ssl-checkpoint-kind", choices=("best", "latest"), default="best")
    parser.add_argument("--weak-checkpoint-kind", choices=("best", "latest"), default="best")
    parser.add_argument(
        "--expert-labels-root", type=Path, default=None,
        help="Папка pdr_expert_labels_v1 (по умолчанию data/phase5/pdr_expert_labels_v1)",
    )
    parser.add_argument(
        "--restart-weak-from",
        choices=("best", "latest"),
        default=None,
        help="Начать новый weak-цикл с весов предыдущего best/latest, сбросив optimizer/scheduler",
    )
    args = parser.parse_args()
    cfg = PDRTrainingConfig(
        stage=args.stage,
        temporal_mode=args.temporal_mode,
        label_stride_samples=args.label_stride_samples,
        model_preset=args.model_preset,
    )
    if args.smoke:
        cfg.epochs = 1
        cfg.samples_per_epoch = 64
        cfg.batch_size = 8
        cfg.weak_open_records = 8
        cfg.weak_french_records = 4
        cfg.weak_validation_records_per_source = 4
        cfg.expert_replay_open_records = 4
        cfg.max_samples_per_record = 4
        # Smoke проверяет тракт на 64 примерах, а не полноту покрытия сотен
        # expert-записей; обязательная пофайловая квота действует в full-run.
        cfg.rotate_expert_samples_each_epoch = False
        cfg.rotate_expert_replay_records_each_epoch = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    for argument, attribute in (
        (args.samples_per_epoch, "samples_per_epoch"),
        (args.batch_size, "batch_size"),
        (args.max_samples_per_record, "max_samples_per_record"),
        (args.validation_max_samples_per_record, "validation_max_samples_per_record"),
        (
            args.expert_validation_max_samples_per_record,
            "expert_validation_max_samples_per_record",
        ),
        (args.weak_open_records, "weak_open_records"),
        (args.weak_french_records, "weak_french_records"),
        (args.expert_replay_open_records, "expert_replay_open_records"),
        (args.min_samples_per_record_per_epoch, "min_samples_per_record_per_epoch"),
        (args.validation_records_per_source, "weak_validation_records_per_source"),
        (args.num_workers, "num_workers"),
    ):
        if argument is not None:
            setattr(cfg, attribute, argument)
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.transition_sampling_fraction is not None:
        cfg.transition_sampling_fraction = args.transition_sampling_fraction
    if args.transition_sampling_radius_ms is not None:
        cfg.transition_sampling_radius_ms = args.transition_sampling_radius_ms
    if args.fixed_weak_record_pool:
        cfg.rotate_weak_records_each_epoch = False
    if args.fixed_expert_sampling:
        cfg.rotate_expert_samples_each_epoch = False
    if args.fixed_expert_replay_pool:
        cfg.rotate_expert_replay_records_each_epoch = False
    if args.expert_labels_root is not None:
        cfg.expert_labels_root = str(args.expert_labels_root)
    ssl_checkpoint = args.ssl_checkpoint or (
        PROJECT_ROOT
        / f"experiments/phase5/pretrain_b_{args.model_preset}"
        / _checkpoint_filename(args.ssl_checkpoint_kind)
    )
    output_dir = args.output_dir or _default_output(cfg, args.smoke)
    weak_checkpoint = args.weak_checkpoint
    if cfg.stage == "weak" and args.restart_weak_from is not None:
        if args.resume:
            raise ValueError("--restart-weak-from несовместим с --resume")
        if weak_checkpoint is not None:
            raise ValueError(
                "Используйте либо --restart-weak-from, либо --weak-checkpoint"
            )
        weak_checkpoint = _restart_checkpoint(output_dir, args.restart_weak_from)
    if cfg.stage == "expert" and weak_checkpoint is None:
        weak_checkpoint = PROJECT_ROOT / (
            f"experiments/phase5/pdr_weak_{cfg.temporal_mode}_stride{cfg.label_stride_samples}"
        ) / _checkpoint_filename(args.weak_checkpoint_kind)
    run(
        cfg,
        output_dir,
        ssl_checkpoint=ssl_checkpoint,
        weak_checkpoint=weak_checkpoint,
        resume=args.resume,
    )
    return 0


def run_manual() -> None:
    """Ручной запуск F5 с полным управлением объёмом и историей.

    Порядок первого эксперимента:
    1. STAGE="weak", SMOKE=True — проверка тракта.
    2. STAGE="weak", SMOKE=False — полный teacher-stage.
    3. После появления best_model.pt/latest_checkpoint.pt: STAGE="expert",
       сначала smoke, затем full; checkpoint выбирается отдельным параметром.

    Разметка v5 остаётся доступной в каждой точке. LABEL_STRIDE_SAMPLES лишь
    задаёт шаг отбора целевых точек в обучающий индекс. Сырые осциллограммы
    загружаются лениво по batch; при старте в RAM строится только ограниченный
    индекс меток (не более заданного числа точек от записи). На weak-stage
    ограниченный пул записей меняется между эпохами до покрытия research-strict
    train-split. На expert-stage используются все экспертные train-записи в
    каждой эпохе, точки внутри них динамически переотбираются, а replay-пул
    последовательно обходит weak Open_EE.

    После каждой эпохи создаются training_log.jsonl, training_history.json,
    training_history.png, latest_checkpoint.pt и best_model.pt. Для продолжения
    прерванного запуска используется RESUME=True. Свежий запуск с RESUME=False
    автоматически переносит прежние результаты в archive_YYYYMMDD_HHMMSS.
    RESTART_WEAK_FROM="best"/"latest" начинает новый weak-цикл с готовых весов,
    но с нулевого внутреннего индекса эпохи и новым optimizer/scheduler.
    """
    # Основной контракт опыта.
    STAGE = "expert"                  # weak, затем expert
    TEMPORAL_MODE = "snapshot_2"   #  snapshot_2, snapshot_5 и sequence_1_8
    LABEL_STRIDE_SAMPLES = 5        # основной; абляция 2 и 1
    MODEL_PRESET = "small"          # small, medium, heavy
    SMOKE = False                   # True: короткая проверка перед full
    RESUME = False                   # True: продолжить latest_checkpoint.pt
    SSL_CHECKPOINT_KIND = "latest"    # best или latest для инициализации weak
    WEAK_CHECKPOINT_KIND = "latest"   # best рекомендуется; latest остаётся доступен
    WEAK_CHECKPOINT_PATH = None      # None = путь автоматически совпадает с TEMPORAL_MODE
    EXPERT_LABELS_ROOT = "data/phase5/pdr_expert_labels_v1"
    RESTART_WEAK_FROM = "latest"        # None, best или latest: новый weak-цикл с готовых весов

    # Объём и длительность обучения.
    EPOCHS = 100
    SAMPLES_PER_EPOCH = 20_000      # случайных целевых точек с возвращением
    BATCH_SIZE = 32                 # число примеров в одном шаге оптимизатора
    MAX_SAMPLES_PER_RECORD = 64     # ограничение train-индекса на осциллограмму
    VALIDATION_MAX_SAMPLES_PER_RECORD = 32 # максимум validation-точек одной осциллограммы
    EXPERT_VALIDATION_MAX_SAMPLES_PER_RECORD = 256 # более плотная expert-validation; число файлов всё равно важнее числа точек

    # Сколько целых осциллограмм индексировать на weak-stage.
    WEAK_OPEN_RECORDS = 4_000      # верхний размер ротационного пула Open_EE одной эпохи
    WEAK_FRENCH_RECORDS = 1_000    # верхний размер ротационного пула French/RTE одной эпохи
    ROTATE_WEAK_RECORDS_EACH_EPOCH = True # обходить весь train-split последовательными пулами
    MIN_SAMPLES_PER_RECORD_PER_EPOCH = 4  # гарантированный минимум точек каждой записи на weak и expert
    TRANSITION_SAMPLING_FRACTION = 0.5    # целевая доля индексных кандидатов из зон переключений
    TRANSITION_SAMPLING_RADIUS_MS = 50.0  # полуширина переходной зоны вокруг смены метки в мс
    VALIDATION_RECORDS_PER_SOURCE = 400   # validation-осциллограмм каждого источника

    # Оптимизация и загрузка.
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4                 # Windows: начать с 0; затем можно проверить 2
    DEVICE = "auto"                # auto, cuda или cpu
    AUGMENTATION_PROBABILITY = 0.5

    # Expert-stage: 75% экспертных примеров + 25% replay обычных Open_EE.
    # При 1000 записях/эпоху Open_EE train без expert-файлов обходится за
    # 39 эпох; точное число частей печатается при запуске.
    EXPERT_REPLAY_OPEN_RECORDS = 1_000
    ROTATE_EXPERT_SAMPLES_EACH_EPOCH = True  # переотбирать точки всех expert-записей
    ROTATE_EXPERT_REPLAY_RECORDS_EACH_EPOCH = True # обходить весь weak replay-архив
    EXPERT_FRACTION = 0.75
    OPEN_EE_FRACTION_WITHIN_EXPERT = 0.75

    cfg = PDRTrainingConfig(
        stage=STAGE,
        temporal_mode=TEMPORAL_MODE,
        label_stride_samples=LABEL_STRIDE_SAMPLES,
        model_preset=MODEL_PRESET,
        epochs=EPOCHS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        max_samples_per_record=MAX_SAMPLES_PER_RECORD,
        validation_max_samples_per_record=VALIDATION_MAX_SAMPLES_PER_RECORD,
        expert_validation_max_samples_per_record=EXPERT_VALIDATION_MAX_SAMPLES_PER_RECORD,
        weak_open_records=WEAK_OPEN_RECORDS,
        weak_french_records=WEAK_FRENCH_RECORDS,
        rotate_weak_records_each_epoch=ROTATE_WEAK_RECORDS_EACH_EPOCH,
        min_samples_per_record_per_epoch=MIN_SAMPLES_PER_RECORD_PER_EPOCH,
        transition_sampling_fraction=TRANSITION_SAMPLING_FRACTION,
        transition_sampling_radius_ms=TRANSITION_SAMPLING_RADIUS_MS,
        weak_validation_records_per_source=VALIDATION_RECORDS_PER_SOURCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        augmentation_probability=AUGMENTATION_PROBABILITY,
        expert_replay_open_records=EXPERT_REPLAY_OPEN_RECORDS,
        rotate_expert_samples_each_epoch=ROTATE_EXPERT_SAMPLES_EACH_EPOCH,
        rotate_expert_replay_records_each_epoch=ROTATE_EXPERT_REPLAY_RECORDS_EACH_EPOCH,
        expert_fraction=EXPERT_FRACTION,
        open_ee_fraction_within_expert=OPEN_EE_FRACTION_WITHIN_EXPERT,
        expert_labels_root=EXPERT_LABELS_ROOT,
    )
    if SMOKE:
        cfg.epochs = 1; cfg.samples_per_epoch = 64; cfg.batch_size = 8
        cfg.weak_open_records = 8; cfg.weak_french_records = 4
        cfg.weak_validation_records_per_source = 4
        cfg.expert_replay_open_records = 4; cfg.max_samples_per_record = 4
        cfg.rotate_expert_samples_each_epoch = False
        cfg.rotate_expert_replay_records_each_epoch = False
    ssl_checkpoint = (
        PROJECT_ROOT
        / f"experiments/phase5/pretrain_b_{MODEL_PRESET}"
        / _checkpoint_filename(SSL_CHECKPOINT_KIND)
    )
    weak_checkpoint = None
    output_dir = _default_output(cfg, SMOKE)
    if STAGE == "weak" and RESTART_WEAK_FROM is not None:
        if RESUME:
            raise ValueError("RESTART_WEAK_FROM несовместим с RESUME=True")
        weak_checkpoint = _restart_checkpoint(output_dir, RESTART_WEAK_FROM)
    elif STAGE == "expert":
        if WEAK_CHECKPOINT_PATH:
            weak_checkpoint = Path(WEAK_CHECKPOINT_PATH)
            if not weak_checkpoint.is_absolute():
                weak_checkpoint = PROJECT_ROOT / weak_checkpoint
        else:
            weak_checkpoint = PROJECT_ROOT / (
                f"experiments/phase5/pdr_weak_{TEMPORAL_MODE}_stride{LABEL_STRIDE_SAMPLES}"
            ) / _checkpoint_filename(WEAK_CHECKPOINT_KIND)
    run(
        cfg,
        output_dir,
        ssl_checkpoint=ssl_checkpoint,
        weak_checkpoint=weak_checkpoint,
        resume=RESUME,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
