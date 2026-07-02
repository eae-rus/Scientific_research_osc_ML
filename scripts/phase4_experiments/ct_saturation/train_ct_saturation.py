"""Единый ручной запуск подготовки данных и обучения детектора насыщения ТТ.

Размер эпохи намеренно фиксирован и не зависит от полного объёма архива::

    python scripts/phase4_experiments/ct_saturation/train_ct_saturation.py
    python scripts/phase4_experiments/ct_saturation/train_ct_saturation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.ct_saturation_dataset import (
    CTSaturationLazyDataset,
    current_feature_count,
    deterministic_split,
    scan_ct_saturation_files,
)
from osc_tools.ml.models.transformer import PhysicalKANTransformer


CONFIG = {
    # Пути. Исходные MAT неизменяемы; плоская копия содержит только I2 и метки.
    "source_data_dir": str(PROJECT_ROOT / "data" / "meas_DP_PG_2SYS_INT_A0_SC1_exp_3"),
    "data_dir": str(PROJECT_ROOT / "data" / "ct_saturation_flat"),
    "output_root": str(PROJECT_ROOT / "experiments" / "phase4" / "ct_saturation"),

    # Воспроизводимость и разбиение.
    "seed": 42,
    "val_fraction": 0.2,

    # Длительность обучения. Эпоха содержит заданное число пакетов, а не весь архив.
    "epochs": 40,
    "train_batches_per_epoch": 96,
    "val_batches": 32,
    "batch_size": 16,
    "val_batch_size": 32,
    "num_workers": 4,
    "prefetch_factor": 2,
    "cache_size": 2,

    # Оптимизация.
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "scheduler_eta_min": 1e-7,
    "use_amp": True,
    "grad_clip": 1.0,
    "accumulation_steps": 1,

    # Спектральная обработка.
    "num_harmonics": 9,
    "sub_periods": [2, 4, 6, 10],
    "num_periods": 10,
    "stride_fraction": 8,

    # Архитектура Physical KAN-Transformer.
    "d_model": 48,
    "num_heads": 4,
    "num_layers": 4,
    "dropout": 0.1,

    # Выбор окон, нормализация и фазовая аугментация.
    "event_window_probability": 0.8,
    "nominal_secondary_a": 5.0,
    "reserve_factor": 20.0,
    "phase_permutation": True,

    # Порог решения и компенсация редких положительных зон.
    "threshold": 0.5,
    # ~35% файлов содержат насыщение, обычно лишь 6–8 из 72 зон.
    "pos_weight": 30.0,

    # Сохранение промежуточных контрольных точек.
    "checkpoint_frequency": 5,
}


def _matlab_path(path: str | Path) -> str:
    """Экранировать путь для строкового литерала MATLAB."""
    return str(Path(path).resolve()).replace("'", "''")


def prepare_flat_data(cfg: dict, max_files: int | None = None) -> None:
    """Потоково распаковать MATLAB MCOS-timeseries в плоские MAT-файлы.

    MATLAB запускается один раз. Каждый исходный файл открывается отдельно,
    поэтому оперативная память не зависит от размера архива. Уже готовые файлы
    пропускаются, а запись через временное имя защищает от оборванных результатов.
    """
    matlab = shutil.which("matlab")
    if matlab is None:
        raise RuntimeError("MATLAB не найден в PATH; он необходим для первичной распаковки MCOS")
    source = Path(cfg["source_data_dir"])
    output = Path(cfg["data_dir"])
    output.mkdir(parents=True, exist_ok=True)
    limit = "inf" if max_files is None else str(int(max_files))
    expression = (
        f"source_dir='{_matlab_path(source)}';"
        f"output_dir='{_matlab_path(output)}';"
        "files=dir(fullfile(source_dir,'A0_INT_SAT_CT1_regime_1_*_exp1.mat'));"
        f"n=min(numel(files),{limit});"
        "fprintf('Подготовка %d из %d файлов\\n',n,numel(files));"
        "for k=1:n,"
        "src=fullfile(files(k).folder,files(k).name);dst=fullfile(output_dir,files(k).name);"
        "if ~exist(dst,'file'),"
        "s=load(src,'I2_CT1','flag_sat_phsA','flag_sat_phsB','flag_sat_phsC');"
        "secondary_a=single(s.I2_CT1.Data);time_s=double(s.I2_CT1.Time(:));"
        "labels=logical([s.flag_sat_phsA(:),s.flag_sat_phsB(:),s.flag_sat_phsC(:)]);"
        "tmp=[dst '.tmp.mat'];save(tmp,'secondary_a','time_s','labels','-v7');movefile(tmp,dst,'f');"
        "end;"
        "if mod(k,100)==0||k==n,fprintf('%d/%d (%.1f%%)\\n',k,n,100*k/n);end;"
        "end;"
    )
    subprocess.run([matlab, "-batch", expression], check=True)


def seed_everything(seed: int) -> None:
    """Зафиксировать генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(files, cfg, *, train: bool):
    """Собрать обучающую или проверочную версию ленивого датасета."""
    return CTSaturationLazyDataset(
        files,
        num_periods=cfg["num_periods"],
        stride_fraction=cfg["stride_fraction"],
        num_harmonics=cfg["num_harmonics"],
        sub_periods=cfg["sub_periods"],
        event_window_probability=cfg["event_window_probability"] if train else 1.0,
        nominal_secondary_a=cfg["nominal_secondary_a"],
        reserve_factor=cfg["reserve_factor"],
        phase_permutation=cfg["phase_permutation"] if train else False,
        cache_size=cfg["cache_size"],
        seed=cfg["seed"] + (0 if train else 10_000),
    )


def make_loader(dataset, batches: int, cfg: dict, *, train: bool) -> DataLoader:
    """Создать загрузчик с фиксированным количеством примеров на эпоху."""
    batch_size = cfg["batch_size"] if train else cfg["val_batch_size"]
    count = batches * batch_size
    generator = torch.Generator().manual_seed(cfg["seed"] + (0 if train else 1))
    sampler = RandomSampler(dataset, replacement=True, num_samples=count, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=cfg["prefetch_factor"] if cfg["num_workers"] > 0 else None,
    )


def metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    """Рассчитать Precision, Recall и F1 по каждой фазе и Macro-F1."""
    pred = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    tp = (pred & truth).sum((0, 1)).float()
    fp = (pred & ~truth).sum((0, 1)).float()
    fn = (~pred & truth).sum((0, 1)).float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "macro_f1": float(f1.mean()),
        "phase_f1": [float(x) for x in f1],
        "phase_precision": [float(x) for x in precision],
        "phase_recall": [float(x) for x in recall],
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    cfg: dict,
    *,
    epoch_index: int,
    total_epochs: int,
    optimizer=None,
    scaler=None,
):
    """Выполнить одну эпоху обучения или проверки."""
    train = optimizer is not None
    model.train(train)
    losses, all_logits, all_targets = [], [], []
    accumulation_steps = max(1, int(cfg["accumulation_steps"]))
    if train:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch_index + 1}/{total_epochs}" if train else f"Val   {epoch_index + 1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_index, (x, y) in enumerate(progress):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(cfg["use_amp"] and device.type == "cuda"),
        ):
            logits = model(x, mode="classify")["classify"]
            loss = criterion(logits, y)
        if train:
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
        progress.set_postfix(loss=f"{np.mean(losses):.4f}")
    progress.close()
    return float(np.mean(losses)), torch.cat(all_logits), torch.cat(all_targets)


def run_experiment(
    cfg: dict,
    *,
    prepare_data: bool,
    prepare_only: bool = False,
    max_prepare_files: int | None = None,
    resume_path: str | Path | None = None,
    reset_optimizer: bool = False,
) -> Path | None:
    """Выполнить подготовку данных и/или полный цикл обучения."""
    resume = Path(resume_path).resolve() if resume_path else None
    if prepare_data:
        prepare_flat_data(cfg, max_files=max_prepare_files)
    if prepare_only:
        return None

    seed_everything(cfg["seed"])
    files = scan_ct_saturation_files(cfg["data_dir"])
    if not files:
        raise FileNotFoundError(
            f"В {cfg['data_dir']} нет подготовленных файлов. "
            "Включите PREPARE_DATA или запустите без --skip-prepare."
        )
    train_files, val_files = deterministic_split(files, cfg["val_fraction"], cfg["seed"])
    if not train_files or not val_files:
        raise RuntimeError("После разбиения обучающая или проверочная выборка оказалась пустой")

    out = resume.parent if resume else (
        Path(cfg["output_root"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split.json").write_text(json.dumps({
        "train": [x.path.name for x in train_files],
        "val": [x.path.name for x in val_files],
    }, ensure_ascii=False), encoding="utf-8")

    train_loader = make_loader(
        make_dataset(train_files, cfg, train=True),
        cfg["train_batches_per_epoch"], cfg, train=True,
    )
    val_loader = make_loader(
        make_dataset(val_files, cfg, train=False),
        cfg["val_batches"], cfg, train=False,
    )
    n_features = current_feature_count(cfg["num_harmonics"], cfg["sub_periods"])
    model = PhysicalKANTransformer(
        num_input_channels=n_features,
        num_current_pairs=n_features // 2,
        num_classes=3,
        zone_size=1,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        cls_head_type="kan",
        max_seq_len=(cfg["num_periods"] - 1) * cfg["stride_fraction"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["scheduler_eta_min"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["use_amp"] and device.type == "cuda")
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.full((3,), cfg["pos_weight"], device=device)
    )
    start_epoch, best_f1 = 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_f1 = checkpoint.get("best_f1", -1.0)
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1

    print(f"Устройство: {device}; файлов train/val: {len(train_files)}/{len(val_files)}")
    print(f"Каталог результатов: {out}")
    log_path = out / "training_log.jsonl"
    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_started = time.perf_counter()
        train_loss, train_logits, train_y = run_epoch(
            model, train_loader, criterion, device, cfg,
            epoch_index=epoch, total_epochs=cfg["epochs"],
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_logits, val_y = run_epoch(
            model, val_loader, criterion, device, cfg,
            epoch_index=epoch, total_epochs=cfg["epochs"],
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.perf_counter() - epoch_started,
            "train": metrics(train_logits, train_y, cfg["threshold"]),
            "val": metrics(val_logits, val_y, cfg["threshold"]),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
            "config": cfg,
        }
        is_best = row["val"]["macro_f1"] > best_f1
        if is_best:
            best_f1 = row["val"]["macro_f1"]
            state["best_f1"] = best_f1
            torch.save(state, out / "best_model.pt")
        torch.save(state, out / "latest_checkpoint.pt")
        if (epoch + 1) % cfg["checkpoint_frequency"] == 0:
            torch.save(state, out / f"checkpoint_epoch_{epoch + 1:04d}.pt")
        marker = " ★" if is_best else ""
        print(
            f"Epoch {epoch + 1:3d}/{cfg['epochs']} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"F1={row['train']['macro_f1']:.4f}/{row['val']['macro_f1']:.4f}{marker} | "
            f"lr={row['lr']:.2e} | time={row['time_sec']:.1f}s"
        )
    return out


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Подготовка данных и обучение Physical KAN-Transformer для насыщения ТТ"
    )
    parser.add_argument("--smoke", action="store_true", help="Короткая проверка всего контура")
    parser.add_argument("--resume", type=Path, help="Продолжить обучение из checkpoint")
    parser.add_argument("--skip-prepare", action="store_true", help="Не запускать распаковку исходных MAT")
    parser.add_argument("--prepare-only", action="store_true", help="Только распаковать данные и завершить работу")
    parser.add_argument("--max-prepare-files", type=int, help="Ограничить число распаковываемых файлов")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-batches", type=int)
    parser.add_argument("--val-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--cache-size", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--source-data-dir", type=str)
    parser.add_argument("--prepared-data-dir", type=str)
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--reset-optimizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Запуск с аргументами командной строки."""
    args = parse_args()
    cfg = dict(CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.train_batches is not None:
        cfg["train_batches_per_epoch"] = args.train_batches
    if args.val_batches is not None:
        cfg["val_batches"] = args.val_batches
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.val_batch_size is not None:
        cfg["val_batch_size"] = args.val_batch_size
    if args.workers is not None:
        cfg["num_workers"] = args.workers
    for argument, key in (
        (args.learning_rate, "learning_rate"),
        (args.weight_decay, "weight_decay"),
        (args.d_model, "d_model"),
        (args.heads, "num_heads"),
        (args.layers, "num_layers"),
        (args.dropout, "dropout"),
        (args.pos_weight, "pos_weight"),
        (args.threshold, "threshold"),
        (args.accumulation_steps, "accumulation_steps"),
        (args.cache_size, "cache_size"),
        (args.prefetch_factor, "prefetch_factor"),
        (args.grad_clip, "grad_clip"),
        (args.source_data_dir, "source_data_dir"),
        (args.prepared_data_dir, "data_dir"),
        (args.output_root, "output_root"),
    ):
        if argument is not None:
            cfg[key] = argument
    if args.smoke:
        cfg.update(
            epochs=1, train_batches_per_epoch=2, val_batches=2,
            batch_size=2, val_batch_size=2, num_workers=0,
        )
    prepare_limit = args.max_prepare_files
    if args.smoke and prepare_limit is None:
        prepare_limit = 20
    run_experiment(
        cfg,
        prepare_data=not args.skip_prepare,
        prepare_only=args.prepare_only,
        max_prepare_files=prepare_limit,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
    )


if __name__ == "__main__":
    # =================================================================
    # РУЧНОЙ ЗАПУСК ИЗ IDE (VS Code, PyCharm и т.д.)
    # =================================================================
    # Если переданы аргументы командной строки, автоматически используется CLI.
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    # 1. Пути к данным и результатам
    SOURCE_DATA_DIR = PROJECT_ROOT / "data" / "meas_DP_PG_2SYS_INT_A0_SC1_exp_3"
    PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "ct_saturation_flat"
    OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "phase4" / "ct_saturation"

    # 2. Подготовка исходных MATLAB timeseries
    PREPARE_DATA = True          # True: подготовить отсутствующие плоские MAT
    PREPARE_ONLY = False         # True: только подготовка, без обучения
    MAX_PREPARE_FILES = None     # None: весь архив; 20/100/...: отладочный поднабор

    # 3. Разбиение и воспроизводимость
    SEED = 42
    VAL_FRACTION = 0.20

    # 4. Спектральная обработка
    NUM_HARMONICS = 9
    SUB_PERIODS = [2, 4, 6, 10]
    NUM_PERIODS = 10             # Длина окна модели в периодах 50 Гц
    STRIDE_FRACTION = 8          # Один токен каждые 1/8 периода
    NOMINAL_SECONDARY_A = 5.0    # Номинальный вторичный ток ТТ
    RESERVE_FACTOR = 20.0        # Делитель нормализации = 5 А × 20

    # 5. Архитектура Physical KAN-Transformer
    D_MODEL = 48
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.10

    # 6. Основные параметры обучения
    EPOCHS = 100
    BATCH_SIZE = 16              # Уменьшить при нехватке VRAM
    VAL_BATCH_SIZE = 32
    ACCUMULATION_STEPS = 1       # Эффективный batch = BATCH_SIZE × это значение
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    SCHEDULER_ETA_MIN = 1e-7
    USE_AMP = True               # Смешанная точность на CUDA
    GRAD_CLIP = 1.0

    # 7. Размер фиксированной эпохи для огромного lazy-датасета
    TRAIN_BATCHES_PER_EPOCH = 96
    VAL_BATCHES_PER_EPOCH = 32

    # 8. Загрузка данных и кэширование
    NUM_WORKERS = 4              # Подбирать по SSD и числу ядер CPU
    PREFETCH_FACTOR = 2
    CACHE_SIZE = 2               # Число MAT-файлов в LRU-кэше каждого worker

    # 9. Разметка, баланс и решение
    EVENT_WINDOW_PROBABILITY = 0.80
    PHASE_PERMUTATION = True
    POS_WEIGHT = 30.0
    THRESHOLD = 0.50

    # 10. Сохранение и продолжение обучения
    CHECKPOINT_FREQUENCY = 5
    RESUME_PATH = None
    # Пример:
    # RESUME_PATH = PROJECT_ROOT / "experiments/phase4/ct_saturation/run_.../latest_checkpoint.pt"
    RESUME_PATH = PROJECT_ROOT / "experiments/phase4/ct_saturation/run_20260702_023838/latest_checkpoint.pt"
    RESET_OPTIMIZER = True      # True: загрузить веса, но начать оптимизацию с эпохи 0

    # =================================================================
    config = dict(CONFIG)
    config.update({
        "source_data_dir": str(SOURCE_DATA_DIR),
        "data_dir": str(PREPARED_DATA_DIR),
        "output_root": str(OUTPUT_ROOT),
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "num_harmonics": NUM_HARMONICS,
        "sub_periods": SUB_PERIODS,
        "num_periods": NUM_PERIODS,
        "stride_fraction": STRIDE_FRACTION,
        "nominal_secondary_a": NOMINAL_SECONDARY_A,
        "reserve_factor": RESERVE_FACTOR,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_eta_min": SCHEDULER_ETA_MIN,
        "use_amp": USE_AMP,
        "grad_clip": GRAD_CLIP,
        "train_batches_per_epoch": TRAIN_BATCHES_PER_EPOCH,
        "val_batches": VAL_BATCHES_PER_EPOCH,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "cache_size": CACHE_SIZE,
        "event_window_probability": EVENT_WINDOW_PROBABILITY,
        "phase_permutation": PHASE_PERMUTATION,
        "pos_weight": POS_WEIGHT,
        "threshold": THRESHOLD,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY,
    })
    run_experiment(
        config,
        prepare_data=PREPARE_DATA,
        prepare_only=PREPARE_ONLY,
        max_prepare_files=MAX_PREPARE_FILES,
        resume_path=RESUME_PATH,
        reset_optimizer=RESET_OPTIMIZER,
    )
