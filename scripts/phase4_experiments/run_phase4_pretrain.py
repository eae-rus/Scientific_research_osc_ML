"""
Скрипт запуска экспериментов Фазы 4: Physical KAN-Transformer.

Режимы:
  --mode pretrain  : Self-supervised pre-training (реконструкция спектров)
  --mode finetune  : Supervised fine-tuning (классификация)
  --mode smoke     : Быстрый smoke-test (1-2 эпохи, проверка OOM)

Примеры:
  python scripts/phase4_experiments/run_phase4_pretrain.py --mode smoke
  python scripts/phase4_experiments/run_phase4_pretrain.py --mode pretrain --epochs 50
  python scripts/phase4_experiments/run_phase4_pretrain.py --mode pretrain --resume experiments/phase4/latest_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

# Добавляем корень проекта в PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.losses import (
    SpectralReconstructionLoss,
    build_channel_groups_phase_polar,
)
from osc_tools.ml.ssl_dataset import SSLSpectralDataset
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.augmented_dataset import (
    AugmentedSpectralDataset, compute_num_channels, compute_stride, SAMPLES_PER_PERIOD,
    standardize_voltage_columns,
)
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.labels import clean_labels, add_base_labels


# ---------------------------------------------------------------------------
# Уровни сложности модели
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS = {
    'light':  {'d_model': 48,  'num_layers': 6,  'num_heads': 4},
    'medium': {'d_model': 64,  'num_layers': 8,  'num_heads': 8},
    'heavy':  {'d_model': 64,  'num_layers': 16, 'num_heads': 8},
}


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

def get_default_config(mode: str = 'smoke') -> dict:
    """Конфигурация по умолчанию для разных режимов."""

    base = {
        # Данные
        'window_size': 320,
        'downsampling_stride': 16,     # Вычисляется автоматически из stride_fraction
        'stride_fraction': 2,             # Доля периода: 2 = полпериода (16), 4 = четверть (8)
        'feature_mode': 'phase_polar',
        'num_harmonics': 9,
        'sub_periods': [2, 4, 6, 10],   # Низшие гармоники (периоды в ед. промышленной частоты)
        'use_augmentation': True,         # Аугментация на сырых данных до FFT
        'use_low_harmonics': True,        # Добавить низшие гармоники
        'include_symmetric': True,        # Симметричные составляющие (I1,I2,I0,U1,U2,U0)
        'future_zones': 4,
        'mask_ratio': 0.25,
        'train_batches_per_epoch': 64,  # Сколько batch-ов случайно брать за эпоху
        'val_stride_multiplier': 4,      # Во сколько раз реже брать окна на валидации
        'batch_size': 32,
        'val_batch_size': 64,
        'num_workers': 8,  # параллельная предподготовка (FFT) в фоне
        'val_split': 0.2,  # 20% файлов на валидацию

        # Модель (значения по умолчанию = light, переопределяются через --complexity)
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 220,  # 8 × (9+4) × 2 = 208 (phase_polar) + 6 × 2 = 12 (symmetric) = 220
        'd_model': 48,
        'num_heads': 4,
        'num_layers': 6,
        'kan_grid_size': 5,
        'dropout': 0.1,
        'use_angle_gate': True,           # DirectionalRelayGate (направленный орган)
        'use_mixed_layer_norm': False,    # False = AmpOnlyLayerNorm (углы не норм.)

        # Обучение
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lr_scheduler': 'cosine',   # 'cosine' или 'plateau'
        'warmup_epochs': 3,         # Линейный warmup перед cosine
        'use_amp': True,            # Mixed precision для 8 ГБ VRAM
        'grad_clip': 1.0,           # Gradient clipping
        'accumulation_steps': 8,    # Gradient accumulation: effective_batch = 32 × 8 = 256

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,  # Сохранять каждые N эпох
        'seed': 42,

        # Данные (путь)
        'data_dir': str(PROJECT_ROOT / 'data' / 'ml_datasets'),
        'precomputed_file': 'labeled_2025_12_03.csv',
    }

    if mode == 'smoke':
        base.update({
            'epochs': 2,
            'batch_size': 4,
            'val_batch_size': 4,
            'd_model': 32,
            'num_layers': 2,
            'num_heads': 2,
            'use_amp': False,
            'warmup_epochs': 0,
            'checkpoint_frequency': 1,
            'use_augmentation': False,
            'use_low_harmonics': False,
            'include_symmetric': False,
            'num_input_channels': 144,  # 8 × 9 × 2 = 144 (без low harmonics и symmetric)
            'accumulation_steps': 1,
        })
    elif mode == 'pretrain':
        # Пересчитываем каналы для pretrain
        num_lh = len(base.get('sub_periods', []))
        base['num_input_channels'] = compute_num_channels(
            base['num_harmonics'], num_lh, base.get('include_symmetric', True),
        )
    elif mode == 'finetune':
        base.update({
            'epochs': 30,
            'learning_rate': 5e-5,
            'warmup_epochs': 1,
        })

    return base


# ---------------------------------------------------------------------------
# Создание модели
# ---------------------------------------------------------------------------

def create_model(config: dict) -> torch.nn.Module:
    """Создаёт модель по конфигурации."""
    model_type = config['model_type']

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            kan_grid_size=config['kan_grid_size'],
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            dropout=config['dropout'],
            max_seq_len=64,
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_len=64,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    return model


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

def _split_files_train_val(
    df: pl.DataFrame, val_split: float, seed: int
) -> tuple[list[str], list[str]]:
    """Разделяет уникальные file_name на train/val (детерминированно)."""
    files = sorted(df['file_name'].unique().to_list())
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]
    return train_files, list(val_files)


def prepare_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, list[list[int]] | None]:
    """Создаёт train/val DataLoader-ы на основе precomputed CSV.

    Если use_augmentation=True и use_low_harmonics=True — используется AugmentedSpectralDataset
    (on-the-fly FFT с низшими гармониками и аугментацией на сырых данных).
    Иначе — SSLSpectralDataset + PrecomputedDataset (legacy).

    Returns:
        (train_loader, val_loader, channel_groups)
    """
    data_path = Path(config['data_dir']) / config['precomputed_file']    
    print(f"Загрузка данных: {data_path}")
    df = pl.read_csv(str(data_path), infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
    print(f"  Строк: {df.height:,}, Колонок: {df.width}")

    # Стандартизация колонок напряжений (raw CSV: 'UA BB' → 'UA')
    df = standardize_voltage_columns(df)

    # Подготовка меток: если raw CSV без Target_* — создаём их
    if 'Target_Normal' not in df.columns:
        df = clean_labels(df)
        df = add_base_labels(df)
        print("  Метки подготовлены из ML_* колонок")

    # Разделяем файлы на train / val
    train_files, val_files = _split_files_train_val(
        df, config['val_split'], config['seed']
    )
    print(f"  Файлов: train={len(train_files)}, val={len(val_files)}")

    # Фильтруем DF
    train_df = df.filter(pl.col('file_name').is_in(train_files))
    val_df = df.filter(pl.col('file_name').is_in(val_files))
    print(f"  Строк: train={train_df.height:,}, val={val_df.height:,}")

    use_augmented = config.get('use_augmentation', False) or config.get('use_low_harmonics', False)

    if use_augmented:
        return _prepare_augmented_dataloaders(config, train_df, val_df)
    else:
        return _prepare_legacy_dataloaders(config, train_df, val_df)


def _prepare_augmented_dataloaders(
    config: dict,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[DataLoader, DataLoader, list[list[int]] | None]:
    """DataLoader-ы через AugmentedSpectralDataset (on-the-fly FFT)."""

    sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if config.get('use_low_harmonics') else []
    include_symmetric = config.get('include_symmetric', True)

    # Stride из доли периода
    stride_frac = config.get('stride_fraction', 2)
    stride = compute_stride(SAMPLES_PER_PERIOD, stride_frac)
    config['downsampling_stride'] = stride

    # Аугментатор для обучения
    augmenter = TimeSeriesAugmenter() if config.get('use_augmentation') else None

    # Пересчёт num_input_channels (чтобы совпадало с реальным dataset)
    num_lh = len(sub_periods)
    actual_channels = compute_num_channels(config['num_harmonics'], num_lh, include_symmetric)
    if config.get('num_input_channels') != actual_channels:
        print(f"  [!] Корректировка num_input_channels: {config['num_input_channels']} -> {actual_channels}")
        config['num_input_channels'] = actual_channels

    # Вычисляем границы файлов
    train_boundaries = AugmentedSpectralDataset.compute_file_boundaries(train_df)
    val_boundaries = AugmentedSpectralDataset.compute_file_boundaries(val_df)

    # Размер окна с будущими периодами
    full_window = config['window_size'] + config.get('future_zones', 0) * config['downsampling_stride']

    val_stride = max(
        config['downsampling_stride'],
        config['downsampling_stride'] * config.get('val_stride_multiplier', 4),
    )

    # Индексы (скользящее окно)
    train_indices = PrecomputedDataset.create_indices(
        train_df, window_size=full_window, mode='val', stride=config['downsampling_stride'],
    )
    val_indices = PrecomputedDataset.create_indices(
        val_df, window_size=full_window, mode='val', stride=val_stride,
    )
    print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,} (val_stride={val_stride})")

    # Datasets
    train_ds = AugmentedSpectralDataset(
        dataframe=train_df,
        file_boundaries=train_boundaries,
        indices=train_indices,
        window_size=config['window_size'],
        num_harmonics=config['num_harmonics'],
        sub_periods=sub_periods if sub_periods else None,
        include_symmetric=include_symmetric,
        downsampling_stride=config['downsampling_stride'],
        future_zones=config.get('future_zones', 0),
        mask_ratio=config['mask_ratio'],
        augmenter=augmenter,
        mode='ssl',
    )
    val_ds = AugmentedSpectralDataset(
        dataframe=val_df,
        file_boundaries=val_boundaries,
        indices=val_indices,
        window_size=config['window_size'],
        num_harmonics=config['num_harmonics'],
        sub_periods=sub_periods if sub_periods else None,
        include_symmetric=include_symmetric,
        downsampling_stride=config['downsampling_stride'],
        future_zones=config.get('future_zones', 0),
        mask_ratio=0.0,  # Валидация без маскирования
        augmenter=None,   # Валидация без аугментации
        mode='ssl',
    )

    channel_groups = train_ds.get_channel_groups(separated=True)

    train_loader = _build_train_epoch_loader(train_ds, config, epoch=0)
    val_loader = DataLoader(
        val_ds,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        **_extra_loader_kwargs(config),
    )

    return train_loader, val_loader, channel_groups


def _prepare_legacy_dataloaders(
    config: dict,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[DataLoader, DataLoader, list[list[int]] | None]:
    """DataLoader-ы через SSLSpectralDataset (legacy, precomputed features)."""

    # Размер окна с будущими зонами (для SSLSpectralDataset)
    future_zones = config.get('future_zones', 0)
    stride = config['downsampling_stride']
    full_window = config['window_size'] + future_zones * stride

    val_stride = max(
        stride,
        stride * config.get('val_stride_multiplier', 4),
    )

    # Создаём индексы (скользящее окно)
    train_indices = PrecomputedDataset.create_indices(
        train_df, window_size=full_window, mode='val', stride=stride
    )
    val_indices = PrecomputedDataset.create_indices(
        val_df, window_size=full_window, mode='val', stride=val_stride
    )
    print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,} (val_stride={val_stride})")

    # SSL Datasets (legacy: передаём future_zones вместо future_periods)
    train_ds = SSLSpectralDataset(
        dataframe=train_df,
        indices=train_indices,
        window_size=config['window_size'],
        feature_mode=config['feature_mode'],
        num_harmonics=config['num_harmonics'],
        downsampling_stride=stride,
        future_zones=future_zones,
        mask_ratio=config['mask_ratio'],
    )
    val_ds = SSLSpectralDataset(
        dataframe=val_df,
        indices=val_indices,
        window_size=config['window_size'],
        feature_mode=config['feature_mode'],
        num_harmonics=config['num_harmonics'],
        downsampling_stride=stride,
        future_zones=future_zones,
        mask_ratio=0.0,  # Валидация без маскирования
    )

    channel_groups = train_ds.get_channel_groups()

    train_loader = _build_train_epoch_loader(train_ds, config, epoch=0)
    val_loader = DataLoader(
        val_ds,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        **_extra_loader_kwargs(config),
    )

    return train_loader, val_loader, channel_groups


# ---------------------------------------------------------------------------
# Цикл обучения и валидации
# ---------------------------------------------------------------------------

def _extra_loader_kwargs(config: dict) -> dict:
    """Доп. kwargs для DataLoader: persistent_workers + prefetch_factor при num_workers > 0."""
    nw = config.get('num_workers', 0)
    if nw > 0:
        return {'persistent_workers': True, 'prefetch_factor': 2}
    return {}


def _build_train_epoch_loader(
    train_ds: torch.utils.data.Dataset,
    config: dict,
    epoch: int,
) -> DataLoader:
    """Строит train DataLoader с фиксированной случайной подвыборкой на эпоху."""
    extra = _extra_loader_kwargs(config)
    batches_per_epoch = config.get('train_batches_per_epoch')
    if batches_per_epoch is None:
        return DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            **extra,
        )

    target_samples = int(batches_per_epoch) * config['batch_size']
    total_samples = len(train_ds)
    if target_samples <= 0 or target_samples >= total_samples:
        return DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            **extra,
        )

    rng = np.random.default_rng(config.get('seed', 42) + epoch)
    sampled_indices = rng.choice(total_samples, size=target_samples, replace=False).tolist()
    sampler = SubsetRandomSampler(sampled_indices)
    return DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        sampler=sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        **extra,
    )

def _split_amp_angle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Разделяет чередующийся тензор [amp, angle, amp, angle, ...] на два.

    Args:
        x: (B, C, T) где C = num_pairs * 2

    Returns:
        amp: (B, C//2, T), angle: (B, C//2, T)
    """
    amp = x[:, 0::2, :]
    angle = x[:, 1::2, :]
    return amp, angle


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: SpectralReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    accumulation_steps: int = 1,
    epoch_idx: int = 0,
    total_epochs: int = 1,
) -> dict[str, float]:
    """Одна эпоха обучения с gradient accumulation.

    Args:
        accumulation_steps: число шагов накопления градиентов.
            effective_batch = batch_size × accumulation_steps.

    Returns:
        dict с 'loss', 'time_sec'
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch_idx + 1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(progress):
        x_input = batch['input'].to(device)       # (B, C, T_current)
        x_target = batch['target'].to(device)      # (B, C, T_full)
        mask_loss = batch['mask_loss'].to(device)   # (B, C, T_full)
        current_len = batch['current_len'][0].item()

        # Forward pass через модель (SSL: реконструкция)
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x_input, mode='ssl')
            pred = out['ssl']  # (B, C, T_current)

            # Модель реконструирует только текущее окно (T_current шагов).
            T_pred = pred.shape[2]
            target_trimmed = x_target[:, :, :T_pred]
            mask_trimmed = mask_loss[:, :, :T_pred]

            # Разделяем на амплитуды и углы
            pred_amp, pred_angle = _split_amp_angle(pred)
            true_amp, true_angle = _split_amp_angle(target_trimmed)
            mask_amp = mask_trimmed[:, 0::2, :]  # Маска для амплитуд (same для углов)

            loss = loss_fn(
                pred_amp, pred_angle, true_amp, true_angle,
                mask=mask_amp, current_len=current_len,
            )

            # Нормализация loss для gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step каждые accumulation_steps (или на последнем батче)
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Восстанавливаем реальное значение loss для логирования
        total_loss += loss.item() * (accumulation_steps if accumulation_steps > 1 else 1)
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

    progress.close()

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)
    return {'loss': avg_loss, 'time_sec': elapsed}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: SpectralReconstructionLoss,
    device: torch.device,
    epoch_idx: int = 0,
    total_epochs: int = 1,
) -> dict[str, float]:
    """Валидация.

    Returns:
        dict с 'loss', 'time_sec'
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    t0 = time.perf_counter()

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Val   {epoch_idx + 1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        x_input = batch['input'].to(device)
        x_target = batch['target'].to(device)
        mask_loss = batch['mask_loss'].to(device)
        current_len = batch['current_len'][0].item()

        out = model(x_input, mode='ssl')
        pred = out['ssl']

        T_pred = pred.shape[2]
        target_trimmed = x_target[:, :, :T_pred]
        mask_trimmed = mask_loss[:, :, :T_pred]

        pred_amp, pred_angle = _split_amp_angle(pred)
        true_amp, true_angle = _split_amp_angle(target_trimmed)
        mask_amp = mask_trimmed[:, 0::2, :]

        loss = loss_fn(
            pred_amp, pred_angle, true_amp, true_angle,
            mask=mask_amp, current_len=current_len,
        )
        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

    progress.close()

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)
    return {'loss': avg_loss, 'time_sec': elapsed}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    """Сохраняет checkpoint (модель, оптимизатор, scheduler, scaler, эпоха)."""
    state = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    reset_optimizer: bool = False,
) -> dict:
    """Загружает checkpoint. Возвращает мета-информацию (epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if reset_optimizer:
        print("  [!] Optimizer, scheduler and epoch state will be reset (starting from 0).")
        return {
            'epoch': -1,
            'best_val_loss': float('inf'),
        }

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return {
        'epoch': ckpt.get('epoch', 0),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
    }


# ---------------------------------------------------------------------------
# Основной цикл pre-training
# ---------------------------------------------------------------------------

def pretrain(config: dict, resume_path: str | None = None, reset_optimizer: bool = False) -> None:
    """Self-supervised pre-training на спектральных данных.

    1. Загружает test_precomputed.csv, делит по file_name на train/val
    2. Создаёт SSLSpectralDataset + DataLoader
    3. Обучает модель реконструировать спектральные признаки
    4. Сохраняет checkpoints и лучшую модель
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config['save_dir']) / f"pretrain_{config['model_type']}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем конфиг
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"PRE-TRAINING — {config['model_type']}")
    print(f"Сохранение: {save_dir}")
    print("=" * 60)

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")

    # Фиксация seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Данные ---
    train_loader, val_loader, channel_groups = prepare_dataloaders(config)

    # --- Модель ---
    model = create_model(config).to(device)
    n_params = model.num_parameters()
    print(f"Параметры модели: {n_params:,}")

    # --- Loss ---
    # Число гармоник с учётом низших
    num_low_harmonics = len(config.get('sub_periods', [])) if config.get('use_low_harmonics') else 0
    total_harmonics = config['num_harmonics'] + num_low_harmonics
    # Число каналов амплитуд тока: 4 сигнала (IA, IB, IC, IN) × total гармоник
    num_current_amp_channels = 4 * total_harmonics
    # channel_groups для amp-only тензора (после split amp/angle)
    separated_groups = build_channel_groups_phase_polar(
        num_signals=8, num_harmonics=config['num_harmonics'],
        separated=True, num_low_harmonics=num_low_harmonics,
    )
    loss_fn = SpectralReconstructionLoss(
        num_current_channels=num_current_amp_channels,
        channel_groups=separated_groups,
    )

    # --- Оптимизатор ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # --- LR Scheduler ---
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config['epochs']

    if config['lr_scheduler'] == 'cosine':
        # CosineAnnealing после warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=1e-6,
        )
    elif config['lr_scheduler'] == 'plateau':
        cosine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    else:
        cosine_scheduler = None

    # --- Mixed Precision ---
    scaler = None
    if config['use_amp'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision: включён (AMP)")

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_path is not None:
        resume_p = Path(resume_path)
        if resume_p.exists():
            print(f"Восстановление из чекпоинта: {resume_p}")
            meta = load_checkpoint(
                resume_p, model, optimizer, cosine_scheduler, scaler,
                reset_optimizer=reset_optimizer,
            )
            start_epoch = meta['epoch'] + 1
            best_val_loss = meta['best_val_loss']
            if not reset_optimizer:
                print(f"  Продолжаем с эпохи {start_epoch}, best_val_loss={best_val_loss:.6f}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: чекпоинт не найден: {resume_p}")

    # --- Лог обучения ---
    history: list[dict] = []
    log_path = save_dir / 'training_log.jsonl'

    print(f"\nНачало обучения: {total_epochs} эпох, batch_size={config['batch_size']}")
    print(f"LR scheduler: {config['lr_scheduler']}, warmup: {warmup_epochs} эпох")
    accum = config.get('accumulation_steps', 1)
    print(f"Gradient accumulation: {accum} шагов -> effective batch = {config['batch_size'] * accum}")
    if config.get('train_batches_per_epoch') is not None:
        print(f"Train batches per epoch: {config['train_batches_per_epoch']} (случайная подвыборка)")
    print("-" * 60)

    for epoch in range(start_epoch, total_epochs):
        train_loader = _build_train_epoch_loader(train_loader.dataset, config, epoch)

        # --- Warmup LR ---
        if epoch < warmup_epochs:
            warmup_lr = config['learning_rate'] * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, config.get('grad_clip', 1.0),
            accumulation_steps=accum,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        # --- Val ---
        val_metrics = validate(
            model, val_loader, loss_fn, device,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        # --- LR Step ---
        if epoch >= warmup_epochs and cosine_scheduler is not None:
            if config['lr_scheduler'] == 'cosine':
                cosine_scheduler.step()
            elif config['lr_scheduler'] == 'plateau':
                cosine_scheduler.step(val_metrics['loss'])

        # --- Логирование ---
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        vram_mb = 0.0
        if device.type == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'lr': current_lr,
            'train_time': train_metrics['time_sec'],
            'val_time': val_metrics['time_sec'],
            'vram_mb': vram_mb,
            'is_best': is_best,
        }
        history.append(record)

        # Запись в JSONL (побатчно, не теряется при крэше)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        marker = ' ★' if is_best else ''
        print(
            f"Epoch {epoch + 1:3d}/{total_epochs} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}{marker} | "
            f"lr={current_lr:.2e} | "
            f"time={train_metrics['time_sec'] + val_metrics['time_sec']:.1f}s"
        )

        # --- Checkpointing ---
        # Всегда сохраняем последнюю эпоху
        save_checkpoint(
            save_dir / 'latest_checkpoint.pt',
            model, optimizer, cosine_scheduler, scaler,
            epoch, best_val_loss, config,
        )

        # Лучшая модель
        if is_best:
            save_checkpoint(
                save_dir / 'best_model.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_loss, config,
            )

        # Периодический snapshot
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint(
                save_dir / f'checkpoint_epoch_{epoch:03d}.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_loss, config,
            )

    # --- Итоги ---
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  Лучшая val_loss: {best_val_loss:.6f}")
    if history:
        best_epoch = min(history, key=lambda r: r['val_loss'])['epoch']
        print(f"  Лучшая эпоха: {best_epoch}")
    print(f"  Чекпоинты: {save_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Smoke Test (синтетические данные)
# ---------------------------------------------------------------------------

def smoke_test(config: dict) -> None:
    """Быстрый smoke-test: проверяет forward pass и Memory без реальных данных."""
    print("=" * 60)
    print("SMOKE TEST — Phase 4 Physical KAN-Transformer")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")

    # Создаём модели
    for model_type in ['PhysicalKANTransformer', 'BaselineTransformer']:
        print(f"\n--- {model_type} ---")
        config['model_type'] = model_type
        model = create_model(config).to(device)
        n_params = model.num_parameters()
        print(f"  Параметры: {n_params:,}")

        # Синтетический batch
        B = config['batch_size']
        C = config['num_input_channels']
        T_steps = 18  # ~(320-32)/16 = 18 шагов после stride

        x = torch.randn(B, C, T_steps, device=device)
        # Внедрим несколько -1 для проверки Sanitizer
        x[0, :4, :3] = -1.0

        # Forward pass (SSL)
        with torch.no_grad():
            out = model(x, mode='ssl')
        print(f"  SSL output: {out['ssl'].shape}")
        print(f"  Features:   {out['features'].shape}")

        # Forward pass (classify) — если бы были num_classes
        if model_type == 'PhysicalKANTransformer':
            model_cls = PhysicalKANTransformer(
                num_input_channels=C, d_model=config['d_model'],
                num_heads=config['num_heads'], num_layers=config['num_layers'],
                num_classes=4, zone_size=16, kan_grid_size=config['kan_grid_size'],
                use_angle_gate=config.get('use_angle_gate', True),
                use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
                dropout=config['dropout'], max_seq_len=64,
            ).to(device)
        else:
            model_cls = BaselineTransformer(
                num_input_channels=C, d_model=config['d_model'],
                num_heads=config['num_heads'], num_layers=config['num_layers'],
                num_classes=4, zone_size=16, dropout=config['dropout'],
                max_seq_len=64,
            ).to(device)

        with torch.no_grad():
            out_cls = model_cls(x, mode='classify')
        print(f"  Classify:   {out_cls['classify'].shape}")

        if device.type == 'cuda':
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak VRAM:  {mem_mb:.0f} МБ")
            torch.cuda.reset_peak_memory_stats()

        del model, model_cls
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Проверяем Loss функцию
    print("\n--- SpectralReconstruction Loss ---")
    B, C_half, T = 4, 72, 18
    pred_amp = torch.randn(B, C_half, T, device=device)
    pred_phase = torch.randn(B, C_half, T, device=device)
    true_amp = torch.randn(B, C_half, T, device=device).abs()
    true_phase = torch.randn(B, C_half, T, device=device)

    spectral_loss_fn = SpectralReconstructionLoss(num_current_channels=36)
    loss_s = spectral_loss_fn(pred_amp, pred_phase, true_amp, true_phase, current_len=14)
    print(f"  SpectralRecon: {loss_s.item():.4f}")

    # Проверяем pipeline на реальных данных (если доступны)
    print("\n--- Проверка pipeline на реальных данных ---")
    data_path = Path(config['data_dir']) / config['precomputed_file']
    if data_path.exists():
        try:
            train_loader, val_loader, channel_groups = prepare_dataloaders(config)
            batch = next(iter(train_loader))
            print(f"  input:      {batch['input'].shape}")
            print(f"  target:     {batch['target'].shape}")
            print(f"  mask_input: {batch['mask_input'].shape}")
            print(f"  mask_loss:  {batch['mask_loss'].shape}")
            print(f"  current_len: {batch['current_len'][0].item()}")
            if channel_groups:
                print(f"  channel_groups: {len(channel_groups)} групп")
            print("  ✓ Pipeline данных работает")
        except Exception as e:
            print(f"  ✗ Ошибка pipeline: {e}")
    else:
        print(f"  Файл не найден: {data_path} — пропуск")

    print("\n✓ Smoke test пройден успешно!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Physical KAN-Transformer')
    parser.add_argument('--mode', choices=['smoke', 'pretrain', 'finetune'],
                        default='smoke', help='Режим запуска')
    parser.add_argument('--complexity', choices=['light', 'medium', 'heavy'],
                        default=None, help='Уровень сложности модели')
    parser.add_argument('--epochs', type=int, default=None, help='Число эпох (override)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (override)')
    parser.add_argument('--model', choices=['PhysicalKANTransformer', 'BaselineTransformer'],
                        default='PhysicalKANTransformer', help='Тип модели')
    parser.add_argument('--d-model', type=int, default=None, help='d_model (override)')
    parser.add_argument('--resume', type=str, default=None, help='Путь к чекпоинту для продолжения')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (override)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Отключить аугментацию на сырых данных')
    parser.add_argument('--no-low-harmonics', action='store_true',
                        help='Отключить низшие гармоники')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help='Шаги gradient accumulation (override)')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Сбросить оптимизатор и начать с 0 эпохи (использовать веса из resume)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config(args.mode)

    # Применяем уровень сложности
    if args.complexity and args.mode != 'smoke':
        level = COMPLEXITY_LEVELS[args.complexity]
        config.update(level)
        print(f"Сложность: {args.complexity} -> {level}")

    # Overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.d_model is not None:
        config['d_model'] = args.d_model
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.accumulation_steps is not None:
        config['accumulation_steps'] = args.accumulation_steps
    config['model_type'] = args.model

    if args.no_augmentation:
        config['use_augmentation'] = False
    if args.no_low_harmonics:
        config['use_low_harmonics'] = False
        config['include_symmetric'] = False
        config['num_input_channels'] = 144  # 8 × 9 × 2

    if args.mode == 'smoke':
        smoke_test(config)
    elif args.mode == 'pretrain':
        pretrain(config, resume_path=args.resume, reset_optimizer=args.reset_optimizer)
    elif args.mode == 'finetune':
        print("Fine-tuning будет доступен после pre-training.")
        print("Используйте --mode pretrain для запуска pre-training.")


if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Раскомментируйте нужный блок и запустите файл напрямую.
    # Для CLI: python run_phase4_pretrain.py --mode pretrain --complexity light
    # =================================================================

    # --- Режим: 'smoke' | 'pretrain' ---
    RUN_MODE = 'pretrain'

    # --- Тип модели ---
    MODEL_TYPE = 'PhysicalKANTransformer'   # 'PhysicalKANTransformer' | 'BaselineTransformer'

    # --- Сложность ---
    SELECTED_COMPLEXITY = 'light'           # 'light' | 'medium' | 'heavy'

    # --- Эпохи ---
    EPOCHS = 100

    # --- Аугментация и признаки ---
    USE_AUGMENTATION = True
    USE_LOW_HARMONICS = True
    INCLUDE_SYMMETRIC = True                # Симметричные составляющие (I1,I2,I0,U1,U2,U0)

    # --- Stride (доля периода: 2 = полпериода=16, 4 = четверть=8) ---
    STRIDE_FRACTION = 2
    VAL_STRIDE_MULTIPLIER = 10              # Валидация реже, чем обучение
    TRAIN_BATCHES_PER_EPOCH = 128           # Случайных batch-ов за эпоху

    # --- Gradient accumulation ---
    ACCUMULATION_STEPS = 8                  # effective batch = batch_size × ACCUMULATION_STEPS

    # --- Частота чекпоинтов ---
    CHECKPOINT_FREQUENCY = 5

    # --- Продолжение обучения (None или путь к чекпоинту) ---
    # RESUME_PATH = None
    RESUME_PATH = 'experiments/phase4/pretrain_PhysicalKANTransformer_20260327_122737/best_model.pt'
    RESET_OPTIMIZER = True  # True, если нужно сбросить оптимизатор и начать с 0 эпохи

    # =================================================================

    # Собираем конфиг
    config = get_default_config(RUN_MODE)
    config['model_type'] = MODEL_TYPE
    config['epochs'] = EPOCHS
    config['use_augmentation'] = USE_AUGMENTATION
    config['use_low_harmonics'] = USE_LOW_HARMONICS
    config['include_symmetric'] = INCLUDE_SYMMETRIC
    config['accumulation_steps'] = ACCUMULATION_STEPS
    config['checkpoint_frequency'] = CHECKPOINT_FREQUENCY
    config['stride_fraction'] = STRIDE_FRACTION
    config['val_stride_multiplier'] = VAL_STRIDE_MULTIPLIER
    config['train_batches_per_epoch'] = TRAIN_BATCHES_PER_EPOCH

    if RUN_MODE != 'smoke':
        level = COMPLEXITY_LEVELS[SELECTED_COMPLEXITY]
        config.update(level)
        # Пересчитываем каналы
        num_lh = len(config['sub_periods']) if USE_LOW_HARMONICS else 0
        config['num_input_channels'] = compute_num_channels(
            config['num_harmonics'], num_lh, INCLUDE_SYMMETRIC,
        )
        if not USE_LOW_HARMONICS and not INCLUDE_SYMMETRIC:
            config['num_input_channels'] = 144

    if RUN_MODE == 'smoke':
        smoke_test(config)
    else:
        pretrain(config, resume_path=RESUME_PATH, reset_optimizer=RESET_OPTIMIZER)
