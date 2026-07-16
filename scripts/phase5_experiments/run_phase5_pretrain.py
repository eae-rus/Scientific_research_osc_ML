"""Phase 5 masked spectral pretrain: ручной F5 и CLI smoke/resume."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

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


@dataclass
class PretrainConfig:
    feature_version: str = "B"
    temporal_mode: str = "snapshot_5"
    samples_per_epoch: int = 20000
    validation_samples: int = 2000
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mask_ratio: float = 0.25
    seed: int = 42
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    num_workers: int = 0
    device: str = "auto"
    source_weights_open_ee: float = 2.0 / 3.0
    source_weights_french_rte: float = 1.0 / 3.0
    cyclic_angle_encoding: bool = True
    use_provenance_embedding: bool = True
    protocol: str = "research_strict"


def _imports():
    try:
        import torch
        from torch.utils.data import DataLoader
        from osc_tools.ml.losses import ComplexMSELoss
        from osc_tools.ml.models.transformer import PhysicalKANTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pretrain требует основное Python-окружение проекта с PyTorch; "
            "bundled runtime Codex предназначен только для CPU/numpy smoke"
        ) from exc
    return torch, DataLoader, ComplexMSELoss, PhysicalKANTransformer


def _load_splits(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != split_manifest_hash(manifest):
        raise ValueError("research_strict split manifest повреждён или изменён")
    return manifest


def _spectral_dataset(
    cfg: PretrainConfig,
    split_manifest: dict[str, object],
    split: str,
    samples: int,
    seed_offset: int,
) -> MaskedSpectralDataset:
    registry = PROJECT_ROOT / "data/phase5/datasets_registry.json"
    sources = {}
    for name in ("open_ee", "french_rte"):
        base = create_source(registry, name)
        indices = split_manifest["sources"][name]["splits"][split]
        sources[name] = IndexedDatasetSource(base, indices, split)
    weights = {
        "open_ee": cfg.source_weights_open_ee,
        "french_rte": cfg.source_weights_french_rte,
    }
    raw = LazyMultiSourceDataset(
        sources,
        weights,
        samples_per_epoch=samples,
        window_periods=10.0,
        history_periods=10.0,
        seed=cfg.seed + seed_offset,
    )
    builder = SpectralFeatureBuilder(SpectralFeatureConfig(cfg.feature_version))
    spectral = SpectralMultiSourceDataset(raw, builder, cfg.temporal_mode)
    return MaskedSpectralDataset(
        spectral,
        SpectralMaskingConfig(cfg.mask_ratio),
        seed=cfg.seed + 10_000 + seed_offset,
    )


def _collate(samples: list[dict[str, object]]):
    """Padding variable-SPP sequences; snapshot modes обычно padding не требуют."""

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


def _masked_complex_loss(loss_fn, prediction, target, reconstruction_mask, missing_mask):
    pair_selected = reconstruction_mask[:, 0::2, :] & reconstruction_mask[:, 1::2, :]
    pair_missing = missing_mask[:, 0::2, :] | missing_mask[:, 1::2, :]
    ignore = ~pair_selected | pair_missing
    return loss_fn(
        prediction[:, 0::2, :], prediction[:, 1::2, :],
        target[:, 0::2, :], target[:, 1::2, :], mask=ignore,
    )


def run(cfg: PretrainConfig, output_dir: Path, resume: bool, reset_optimizer: bool) -> None:
    torch, DataLoader, ComplexMSELoss, PhysicalKANTransformer = _imports()
    print(json.dumps({
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, ensure_ascii=False, indent=2))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = PROJECT_ROOT / "data/phase5/research_strict_splits.json"
    splits = _load_splits(split_path)
    train_dataset = _spectral_dataset(cfg, splits, "train", cfg.samples_per_epoch, 0)
    val_dataset = _spectral_dataset(cfg, splits, "validation", cfg.validation_samples, 1)
    builder = train_dataset.dataset.feature_builder
    passport = FeatureContractPassport.create(
        builder.schema, cfg.temporal_mode, cfg.cyclic_angle_encoding,
        cfg.use_provenance_embedding,
    )
    config_payload = asdict(cfg) | {
        "split_manifest": str(split_path.relative_to(PROJECT_ROOT)),
        "split_sha256": splits["sha256"],
        "feature_passport": passport.to_dict(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else "cpu" if cfg.device == "auto" else cfg.device)
    model = PhysicalKANTransformer(
        num_input_channels=len(builder.schema.names),
        ssl_output_channels=len(builder.schema.names),
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=256,
        cyclic_angle_encoding=cfg.cyclic_angle_encoding,
        use_provenance_embedding=cfg.use_provenance_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    loss_fn = ComplexMSELoss()
    start_epoch = 0
    best_val = float("inf")
    latest = output_dir / "latest_checkpoint.pt"
    if resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        passport.assert_compatible(checkpoint["feature_passport"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate)
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(start_epoch, cfg.epochs):
        started = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        train_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["features"].to(device)
            target = batch["target"].to(device)
            provenance = batch["provenance"].to(device)
            output = model(x, mode="ssl", provenance=provenance)["ssl"]
            loss = _masked_complex_loss(loss_fn, output, target, batch["reconstruction_mask"].to(device), batch["missing_mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach())

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch["features"].to(device), mode="ssl", provenance=batch["provenance"].to(device))["ssl"]
                loss = _masked_complex_loss(loss_fn, output, batch["target"].to(device), batch["reconstruction_mask"].to(device), batch["missing_mask"].to(device))
                val_total += float(loss)
        train_loss = train_total / max(1, len(train_loader))
        val_loss = val_total / max(1, len(val_loader))
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": min(best_val, val_loss), "config": config_payload, "feature_passport": passport.to_dict()}
        temporary = output_dir / "latest_checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(latest)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, output_dir / "best_model.pt")
        print(json.dumps(record, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--feature-version", choices=("A", "B"), default="B")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/phase5/pretrain_b")
    args = parser.parse_args()
    cfg = PretrainConfig(feature_version=args.feature_version)
    if args.smoke:
        cfg.samples_per_epoch = 64
        cfg.validation_samples = 32
        cfg.epochs = 2
        cfg.batch_size = 8
        cfg.d_model = 32
        cfg.num_layers = 2
        cfg.d_ff = 128
        args.output_dir = PROJECT_ROOT / f"experiments/phase5/smoke_{args.feature_version.lower()}"
    run(cfg, args.output_dir, args.resume, args.reset_optimizer)
    return 0


def run_manual() -> None:
    # ================================================================
    # РУЧНОЙ ЗАПУСК F5: сначала SMOKE=True; затем False для полного run.
    # RESUME=True продолжает latest_checkpoint.pt.
    # ================================================================
    FEATURE_VERSION = "B"
    SMOKE = True
    RESUME = False
    RESET_OPTIMIZER = False
    OUTPUT_DIR = PROJECT_ROOT / f"experiments/phase5/{'smoke' if SMOKE else 'pretrain'}_{FEATURE_VERSION.lower()}"
    cfg = PretrainConfig(feature_version=FEATURE_VERSION)
    if SMOKE:
        cfg.samples_per_epoch = 64
        cfg.validation_samples = 32
        cfg.epochs = 2
        cfg.batch_size = 8
        cfg.d_model = 32
        cfg.num_layers = 2
        cfg.d_ff = 128
    run(cfg, OUTPUT_DIR, RESUME, RESET_OPTIMIZER)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
