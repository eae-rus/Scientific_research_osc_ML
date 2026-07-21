"""Phase 5 evaluation script: детальная оценка ошибки реконструкции по подгруппам признаков."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.checkpoint_contracts import FeatureContractPassport
from osc_tools.ml.dataset_registry import create_source
from osc_tools.ml.lazy_multi_dataset import LazyMultiSourceDataset, SpectralMultiSourceDataset
from osc_tools.ml.phase5_splits import IndexedDatasetSource, split_manifest_hash
from osc_tools.ml.phase5_ssl import MaskedSpectralDataset, SpectralMaskingConfig
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


def _imports():
    import torch
    from torch.utils.data import DataLoader
    from osc_tools.ml.losses import RobustComplexLoss
    from osc_tools.ml.models.transformer import PhysicalKANTransformer
    return torch, DataLoader, RobustComplexLoss, PhysicalKANTransformer


def _load_splits(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != split_manifest_hash(manifest):
        raise ValueError("research_strict split manifest повреждён или изменён")
    return manifest


def _collate(samples: list[dict[str, object]]):
    torch, *_ = _imports()
    max_time = max(np.asarray(sample["features"]).shape[0] for sample in samples)
    channels = np.asarray(samples[0]["features"]).shape[1]
    batch = len(samples)
    features = np.full((batch, channels, max_time), np.nan, dtype=np.float32)
    targets = np.full_like(features, np.nan)
    provenance = np.zeros((batch, channels, max_time), dtype=np.int64)
    reconstruction = np.zeros((batch, channels, max_time), dtype=bool)
    missing = np.ones((batch, channels, max_time), dtype=bool)
    metadata = []
    for row, sample in enumerate(samples):
        n_time = np.asarray(sample["features"]).shape[0]
        features[row, :, :n_time] = np.asarray(sample["features"]).T
        targets[row, :, :n_time] = np.asarray(sample["target"]).T
        provenance[row, :, :n_time] = np.asarray(sample["provenance"]).T
        reconstruction[row, :, :n_time] = np.asarray(sample["reconstruction_mask"]).T
        missing[row, :, :n_time] = np.asarray(sample["missing_mask"]).T
        metadata.append(sample["metadata"])
    return {
        "features": torch.from_numpy(features),
        "target": torch.from_numpy(targets),
        "provenance": torch.from_numpy(provenance),
        "reconstruction_mask": torch.from_numpy(reconstruction),
        "missing_mask": torch.from_numpy(missing),
        "metadata": metadata,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_samples: int = 1000,
    batch_size: int = 32,
    device_str: str = "auto",
) -> dict[str, object]:
    torch, DataLoader, RobustComplexLoss, PhysicalKANTransformer = _imports()

    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Загрузка чекпоинта: {checkpoint_path} (Устройство: {device})")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    passport_dict = checkpoint.get("feature_passport", {})
    feature_names = passport_dict.get("feature_names", [])

    split_path = PROJECT_ROOT / "data/phase5/research_strict_splits.json"
    splits = _load_splits(split_path)

    registry = PROJECT_ROOT / "data/phase5/datasets_registry.json"
    sources = {}
    for name in ("open_ee", "french_rte"):
        base = create_source(registry, name)
        indices = splits["sources"][name]["splits"]["validation"]
        sources[name] = IndexedDatasetSource(base, indices, "validation")

    weights = {"open_ee": 0.6667, "french_rte": 0.3333}
    raw = LazyMultiSourceDataset(
        sources,
        weights,
        samples_per_epoch=validation_samples,
        window_periods=10.0,
        history_periods=10.0,
        seed=42,
    )

    feature_version = config.get("feature_version", "B")
    temporal_mode = config.get("temporal_mode", "snapshot_5")
    builder = SpectralFeatureBuilder(SpectralFeatureConfig(feature_version))
    spectral = SpectralMultiSourceDataset(raw, builder, temporal_mode)
    dataset = MaskedSpectralDataset(spectral, SpectralMaskingConfig(0.25), seed=10042)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    d_model = config.get("d_model", 64)
    num_heads = config.get("num_heads", 4)
    num_layers = config.get("num_layers", 4)
    d_ff = config.get("d_ff", 256)
    cyclic_angle_encoding = config.get("cyclic_angle_encoding", True)
    use_provenance_embedding = config.get("use_provenance_embedding", True)

    model = PhysicalKANTransformer(
        num_input_channels=len(feature_names),
        ssl_output_channels=len(feature_names),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=256,
        cyclic_angle_encoding=cyclic_angle_encoding,
        use_provenance_embedding=use_provenance_embedding,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_fn = RobustComplexLoss(beta=0.1)

    # Категории признаков
    groups = {
        "h1_fundamental": [i for i, name in enumerate(feature_names) if "_h1_" in name],
        "higher_harmonics": [i for i, name in enumerate(feature_names) if any(f"_h{h}_" in name for h in range(2, 10))],
        "low_frequency": [i for i, name in enumerate(feature_names) if "_lp" in name],
        "currents": [i for i, name in enumerate(feature_names) if name.startswith("I")],
        "voltages": [i for i, name in enumerate(feature_names) if name.startswith("U")],
        "seq_pos_1": [i for i, name in enumerate(feature_names) if name.startswith("I1") or name.startswith("U1")],
        "seq_neg_2": [i for i, name in enumerate(feature_names) if name.startswith("I2") or name.startswith("U2")],
        "seq_zero_0": [i for i, name in enumerate(feature_names) if name.startswith("I0") or name.startswith("U0")],
    }

    group_losses: dict[str, list[float]] = {g: [] for g in groups}
    total_losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            target = batch["target"].to(device)
            provenance = batch["provenance"].to(device)
            rec_mask = batch["reconstruction_mask"].to(device)
            miss_mask = batch["missing_mask"].to(device)

            output = model(x, mode="ssl", provenance=provenance)["ssl"]

            pair_selected = rec_mask[:, 0::2, :] & rec_mask[:, 1::2, :]
            pair_missing = miss_mask[:, 0::2, :] | miss_mask[:, 1::2, :]
            ignore = ~pair_selected | pair_missing

            loss = loss_fn(
                output[:, 0::2, :],
                output[:, 1::2, :],
                target[:, 0::2, :],
                target[:, 1::2, :],
                mask=ignore,
            )
            total_losses.append(float(loss))

            # Детализация по группам
            for g_name, indices in groups.items():
                if not indices:
                    continue
                # Выбор только парных индексов группы (комплексная пара: magnitude i, angle i+1)
                comp_indices = [idx // 2 for idx in indices if idx % 2 == 0]
                if not comp_indices:
                    continue

                sub_pred_re = output[:, 0::2, :][:, comp_indices, :]
                sub_pred_im = output[:, 1::2, :][:, comp_indices, :]
                sub_targ_re = target[:, 0::2, :][:, comp_indices, :]
                sub_targ_im = target[:, 1::2, :][:, comp_indices, :]
                sub_ignore = ignore[:, comp_indices, :]

                sub_loss = loss_fn(sub_pred_re, sub_pred_im, sub_targ_re, sub_targ_im, mask=sub_ignore)
                group_losses[g_name].append(float(sub_loss))

    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "overall_val_loss": float(np.mean(total_losses)),
        "group_losses": {g: float(np.mean(vals)) for g, vals in group_losses.items() if vals},
    }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "experiments/phase5/pretrain_b/best_model.pt")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    results = evaluate_checkpoint(args.checkpoint, validation_samples=args.samples)
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ДЕТАЛЬНОЙ ОЦЕНКИ ОШИБКИ РЕКОНСТРУКЦИИ")
    print("=" * 50)
    print(f"Чекпоинт: {results['checkpoint']}")
    print(f"Эпоха чекпоинта: {results['epoch']}")
    print(f"Общая ошибка (Robust Loss): {results['overall_val_loss']:.7f}")
    print("-" * 50)
    print("Ошибки по подгруппам признаков:")
    for group, loss_val in results["group_losses"].items():
        print(f"  - {group:20s}: {loss_val:.7f}")

    reports_dir = PROJECT_ROOT / "reports/phase5"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "eval_pretrain_b.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Отчёт детальной оценки реконструкции Phase 5",
        "",
        f"- **Чекпоинт**: `{results['checkpoint']}`",
        f"- **Эпоха**: `{results['epoch']}`",
        f"- **Общий Robust Complex Loss**: `{results['overall_val_loss']:.7f}`",
        "",
        "## Ошибки по группам признаков",
        "",
        "| Группа признаков | Robust Complex Loss | Описание |",
        "|---|---:|---|",
        f"| **Фундаментальная h1** | `{results['group_losses'].get('h1_fundamental', 0):.7f}` | Первая гармоника 50 Гц |",
        f"| **Высшие гармоники h2-h9** | `{results['group_losses'].get('higher_harmonics', 0):.7f}` | Гармоники h2–h9 |",
        f"| **Низшие компоненты lp2-lp10** | `{results['group_losses'].get('low_frequency', 0):.7f}` | Субгармоники за 2–10 периодов |",
        f"| **Токи (I)** | `{results['group_losses'].get('currents', 0):.7f}` | Все токовые каналы |",
        f"| **Напряжения (U)** | `{results['group_losses'].get('voltages', 0):.7f}` | Все каналы напряжений |",
        f"| **Прямая посл. (1)** | `{results['group_losses'].get('seq_pos_1', 0):.7f}` | Составляющая прямой последовательности |",
        f"| **Обратная посл. (2)** | `{results['group_losses'].get('seq_neg_2', 0):.7f}` | Составляющая обратной последовательности |",
        f"| **Нулевая посл. (0)** | `{results['group_losses'].get('seq_zero_0', 0):.7f}` | Составляющая нулевой последовательности |",
    ]
    md_path = reports_dir / "eval_pretrain_b.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nОтчёты сохранены в:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
