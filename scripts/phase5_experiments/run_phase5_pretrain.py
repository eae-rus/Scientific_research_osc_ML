"""Phase 5 masked spectral pretrain: ручной F5 и CLI smoke/resume с пресетами моделей."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
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

from osc_tools.ml.checkpoint_contracts import FeatureContractPassport
from osc_tools.ml.dataset_registry import create_source
from osc_tools.ml.lazy_multi_dataset import LazyMultiSourceDataset, SpectralMultiSourceDataset
from osc_tools.ml.phase5_splits import IndexedDatasetSource, split_manifest_hash
from osc_tools.ml.phase5_ssl import MaskedSpectralDataset, SpectralMaskingConfig
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
from scripts.phase5_experiments.progress import ProgressReporter


# Пресеты сложностей моделей для 8 ГБ VRAM (RTX 3060 Ti)
MODEL_PRESETS = {
    "small": {"d_model": 64, "num_heads": 4, "num_layers": 4, "d_ff": 256},      # ~300 тыс. параметров (Лёгкая)
    "medium": {"d_model": 128, "num_heads": 8, "num_layers": 6, "d_ff": 512},    # ~1.2 млн параметров (Средняя)
    "heavy": {"d_model": 256, "num_heads": 8, "num_layers": 8, "d_ff": 1024},    # ~4.8 млн параметров (Тяжёлая)
}


@dataclass
class PretrainConfig:
    model_preset: str = "small"
    feature_version: str = "B"
    temporal_mode: str = "snapshot_5"
    samples_per_epoch: int = 20000
    validation_samples: int = 2000
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_type: str = "cosine_restarts"
    restart_epochs: int = 50
    mask_ratio: float = 0.25
    loss_type: str = "complex_huber"
    huber_beta: float = 0.1
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

    def apply_preset(self) -> None:
        """Применить параметры выбранного пресета сложности."""
        if self.model_preset in MODEL_PRESETS:
            preset = MODEL_PRESETS[self.model_preset]
            self.d_model = preset["d_model"]
            self.num_heads = preset["num_heads"]
            self.num_layers = preset["num_layers"]
            self.d_ff = preset["d_ff"]


def _imports():
    try:
        import torch
        from torch.utils.data import DataLoader
        from osc_tools.ml.losses import ComplexMSELoss, RobustComplexLoss
        from osc_tools.ml.models.transformer import PhysicalKANTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pretrain требует основное Python-окружение проекта с PyTorch; "
            "bundled runtime Codex предназначен только для CPU/numpy smoke"
        ) from exc
    return torch, DataLoader, ComplexMSELoss, RobustComplexLoss, PhysicalKANTransformer


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
    source_names: Sequence[str] = ("open_ee", "french_rte"),
) -> MaskedSpectralDataset:
    registry = PROJECT_ROOT / "data/phase5/datasets_registry.json"
    sources = {}
    for name in source_names:
        base = create_source(registry, name)
        indices = split_manifest["sources"][name]["splits"][split]
        sources[name] = IndexedDatasetSource(base, indices, split)
    weights = {
        "open_ee": cfg.source_weights_open_ee,
        "french_rte": cfg.source_weights_french_rte,
    }
    weights = {name: weights[name] for name in source_names}
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
        prediction[:, 0::2, :],
        prediction[:, 1::2, :],
        target[:, 0::2, :],
        target[:, 1::2, :],
        mask=ignore,
    )


def _evaluate(model, loader, loss_fn, device, label: str) -> float:
    model.eval()
    total = 0.0
    progress = ProgressReporter(label, len(loader))
    with __import__("torch").no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            output = model(
                batch["features"].to(device),
                mode="ssl",
                provenance=batch["provenance"].to(device),
            )["ssl"]
            loss = _masked_complex_loss(
                loss_fn,
                output,
                batch["target"].to(device),
                batch["reconstruction_mask"].to(device),
                batch["missing_mask"].to(device),
            )
            total += float(loss)
            progress.update(batch_index)
    progress.finish()
    return total / max(1, len(loader))


def _export_training_curves(log_path: Path, output_dir: Path) -> None:
    """Экспортировать компактные данные кривых и PNG, если доступен matplotlib."""

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    (output_dir / "training_curves.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    epochs = [record["epoch"] for record in records]
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, label in (
        ("train_loss", "train"),
        ("val_loss", "validation combined"),
        ("val_open_ee_loss", "validation Open_EE"),
        ("val_french_rte_loss", "validation French/RTE"),
    ):
        if all(key in record for record in records):
            axis.plot(epochs, [record[key] for record in records], marker="o", label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Complex masked reconstruction loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def archive_existing_output(output_dir: Path) -> Path | None:
    """Автоматически архивировать существующие артефакты при свежем запуске (resume=False)."""

    log_file = output_dir / "training_log.jsonl"
    if not log_file.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / f"archive_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    files_to_move = [
        "training_log.jsonl",
        "training_curves.json",
        "training_curves.png",
        "config.json",
        "latest_checkpoint.pt",
        "best_model.pt",
    ]
    for filename in files_to_move:
        filepath = output_dir / filename
        if filepath.exists():
            shutil.move(str(filepath), str(archive_dir / filename))

    print(f"[Уведомление] Предыдущие результаты сохранены в архив: {archive_dir}")
    return archive_dir


def run(cfg: PretrainConfig, output_dir: Path, resume: bool, reset_optimizer: bool) -> None:
    torch, DataLoader, ComplexMSELoss, RobustComplexLoss, PhysicalKANTransformer = _imports()
    cfg.apply_preset()

    print(json.dumps({
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_preset": cfg.model_preset,
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "d_ff": cfg.d_ff,
        "scheduler_type": cfg.scheduler_type,
        "epochs": cfg.epochs,
    }, ensure_ascii=False, indent=2))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        archive_existing_output(output_dir)

    split_path = PROJECT_ROOT / "data/phase5/research_strict_splits.json"
    splits = _load_splits(split_path)
    train_dataset = _spectral_dataset(cfg, splits, "train", cfg.samples_per_epoch, 0)
    val_dataset = _spectral_dataset(cfg, splits, "validation", cfg.validation_samples, 1)
    per_source_samples = max(16, cfg.validation_samples // 2)
    val_open_ee_dataset = _spectral_dataset(
        cfg, splits, "validation", per_source_samples, 101, ("open_ee",)
    )
    val_french_dataset = _spectral_dataset(
        cfg, splits, "validation", per_source_samples, 102, ("french_rte",)
    )
    builder = train_dataset.dataset.feature_builder
    passport = FeatureContractPassport.create(
        builder.schema,
        cfg.temporal_mode,
        cfg.cyclic_angle_encoding,
        cfg.use_provenance_embedding,
    )
    config_payload = asdict(cfg) | {
        "split_manifest": str(split_path.relative_to(PROJECT_ROOT)),
        "split_sha256": splits["sha256"],
        "feature_passport": passport.to_dict(),
    }
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available() else "cpu" if cfg.device == "auto" else cfg.device
    )
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

    if cfg.scheduler_type == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.restart_epochs, T_mult=1, eta_min=1e-6
        )
    elif cfg.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    elif cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Неизвестный scheduler_type: {cfg.scheduler_type!r}")

    if cfg.loss_type == "complex_mse":
        loss_fn = ComplexMSELoss()
    elif cfg.loss_type == "complex_huber":
        loss_fn = RobustComplexLoss(beta=cfg.huber_beta)
    else:
        raise ValueError(f"Неизвестный loss_type: {cfg.loss_type!r}")

    start_epoch = 0
    best_val = float("inf")
    latest = output_dir / "latest_checkpoint.pt"
    if resume and not latest.exists():
        raise FileNotFoundError(f"Запрошен resume, но checkpoint отсутствует: {latest}")
    if resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        passport.assert_compatible(checkpoint["feature_passport"])
        stored_config = checkpoint.get("config", {})
        checkpoint_loss = stored_config.get("loss_type", "complex_mse")
        if checkpoint_loss != cfg.loss_type:
            raise ValueError(
                f"Нельзя resume с другим loss: checkpoint={checkpoint_loss}, current={cfg.loss_type}"
            )
        data_contract_keys = (
            "split_sha256",
            "feature_version",
            "temporal_mode",
            "mask_ratio",
            "seed",
            "source_weights_open_ee",
            "source_weights_french_rte",
            "huber_beta",
        )
        mismatches = [
            key for key in data_contract_keys
            if stored_config.get(key) != config_payload.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Нельзя resume с изменённым data/SSL contract: "
                + ", ".join(mismatches)
            )
        if not reset_optimizer:
            optimizer_contract_keys = ("scheduler_type", "restart_epochs")
            optimizer_mismatches = [
                key for key in optimizer_contract_keys
                if stored_config.get(key) != config_payload.get(key)
            ]
            if optimizer_mismatches:
                raise ValueError(
                    "Для смены scheduler при resume требуется --reset-optimizer: "
                    + ", ".join(optimizer_mismatches)
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))

    # Конфиг обновляется только после успешной проверки resume, чтобы ошибочный
    # запуск не затёр паспорт предыдущего эксперимента.
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate
    )
    val_open_ee_loader = DataLoader(
        val_open_ee_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate
    )
    val_french_loader = DataLoader(
        val_french_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=_collate
    )
    log_path = output_dir / "training_log.jsonl"

    for epoch in range(start_epoch, cfg.epochs):
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_dataset.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_batch_losses: list[float] = []
        train_progress = ProgressReporter(f"Epoch {epoch} train", len(train_loader))
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            x = batch["features"].to(device)
            target = batch["target"].to(device)
            provenance = batch["provenance"].to(device)
            output = model(x, mode="ssl", provenance=provenance)["ssl"]
            loss = _masked_complex_loss(
                loss_fn,
                output,
                target,
                batch["reconstruction_mask"].to(device),
                batch["missing_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss = float(loss.detach())
            train_total += batch_loss
            train_batch_losses.append(batch_loss)
            train_progress.update(batch_index)
        train_progress.finish()

        train_seconds = time.time() - started
        train_loss = train_total / max(1, len(train_loader))
        val_loss = _evaluate(model, val_loader, loss_fn, device, f"Epoch {epoch} validation combined")
        val_open_ee_loss = _evaluate(model, val_open_ee_loader, loss_fn, device, f"Epoch {epoch} validation Open_EE")
        val_french_rte_loss = _evaluate(model, val_french_loader, loss_fn, device, f"Epoch {epoch} validation French/RTE")

        if cfg.scheduler_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_batch_loss_median": float(np.median(train_batch_losses)),
            "train_batch_loss_p95": float(np.quantile(train_batch_losses, 0.95)),
            "train_batch_loss_max": max(train_batch_losses),
            "val_loss": val_loss,
            "val_open_ee_loss": val_open_ee_loss,
            "val_french_rte_loss": val_french_rte_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "train_seconds": train_seconds,
            "train_samples_per_second": cfg.samples_per_epoch / max(train_seconds, 1e-9),
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0
            ),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        _export_training_curves(log_path, output_dir)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": min(best_val, val_loss),
            "config": config_payload,
            "feature_passport": passport.to_dict(),
        }
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
    parser.add_argument("--model-preset", choices=("small", "medium", "heavy"), default="small")
    parser.add_argument("--feature-version", choices=("A", "B"), default="B")
    parser.add_argument("--temporal-mode", choices=("snapshot_2", "snapshot_5", "sequence_1_8"), default="snapshot_5")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scheduler-type", choices=("cosine_restarts", "plateau", "cosine"), default="cosine_restarts")
    parser.add_argument("--restart-epochs", type=int, default=50)
    parser.add_argument("--loss-type", choices=("complex_mse", "complex_huber"), default="complex_huber")
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = PretrainConfig(
        model_preset=args.model_preset,
        feature_version=args.feature_version,
        temporal_mode=args.temporal_mode,
        scheduler_type=args.scheduler_type,
        restart_epochs=args.restart_epochs,
        loss_type=args.loss_type,
        huber_beta=args.huber_beta,
    )
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / f"experiments/phase5/pretrain_{args.feature_version.lower()}_{args.model_preset}"
    )
    if args.smoke:
        cfg.samples_per_epoch = 64
        cfg.validation_samples = 32
        cfg.epochs = 2
        cfg.batch_size = 8
        output_dir = (
            args.output_dir
            or PROJECT_ROOT / f"experiments/phase5/smoke_{args.feature_version.lower()}_{args.model_preset}"
        )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.samples_per_epoch is not None:
        cfg.samples_per_epoch = args.samples_per_epoch
    if args.validation_samples is not None:
        cfg.validation_samples = args.validation_samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    run(cfg, output_dir, args.resume, args.reset_optimizer)
    return 0


def run_manual() -> None:
    # ================================================================
    # ЕДИНСТВЕННАЯ НАСТРОЙКА СЛОЖНОСТИ МОДЕЛИ:
    # "small"  - Лёгкая   (d_model=64,  4 слоя, ~300 тыс. параметров)
    # "medium" - Средняя  (d_model=128, 6 слоёв, ~1.2 млн параметров)
    # "heavy"  - Тяжёлая  (d_model=256, 8 слоёв, ~4.8 млн параметров)
    # ================================================================
    MODEL_PRESET = "small"

    # Параметры запуска
    SMOKE = False                  # Сначала поставь True для быстрого 2-эпохового теста
    RESUME = False                 # True если нужно продолжить с latest_checkpoint.pt
    RESET_OPTIMIZER = False        # True если при resume нужно сбросить оптимизатор
    EPOCHS = 200                   # Всего эпох
    SCHEDULER_TYPE = "cosine_restarts" # Косинусный шедулер с циклическим ускорением
    RESTART_EPOCHS = 50            # Период рестарта скорости обучения (каждые 50 эпох)
    SAMPLES_PER_EPOCH = 20_000     # Число случайно выбираемых окон на эпоху
    VALIDATION_SAMPLES = 2_000     # Число окон на валидацию
    BATCH_SIZE = 32
    FEATURE_VERSION = "B"
    TEMPORAL_MODE = "snapshot_5"
    LOSS_TYPE = "complex_huber"
    HUBER_BETA = 0.1

    OUTPUT_DIR = (
        PROJECT_ROOT
        / f"experiments/phase5/{'smoke' if SMOKE else 'pretrain'}_{FEATURE_VERSION.lower()}_{MODEL_PRESET}"
    )

    cfg = PretrainConfig(
        model_preset=MODEL_PRESET,
        feature_version=FEATURE_VERSION,
        temporal_mode=TEMPORAL_MODE,
        loss_type=LOSS_TYPE,
        huber_beta=HUBER_BETA,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_samples=VALIDATION_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        scheduler_type=SCHEDULER_TYPE,
        restart_epochs=RESTART_EPOCHS,
    )
    if SMOKE:
        cfg.samples_per_epoch = 64
        cfg.validation_samples = 32
        cfg.epochs = 2
        cfg.batch_size = 8

    run(cfg, OUTPUT_DIR, RESUME, RESET_OPTIMIZER)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
