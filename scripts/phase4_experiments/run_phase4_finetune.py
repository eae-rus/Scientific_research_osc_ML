"""
Fine-tuning скрипт для Фазы 4: Physical KAN-Transformer.

Инициализирует модель из SSL-чекпоинта и дообучает на задачу классификации
аварийных режимов (4 базовых класса: Normal, ML_1, ML_2, ML_3).

Зональная классификация: zone_size=1, каждый временной шаг (при stride=16
= полпериода) получает свою предсказанную метку.

Примеры:
  python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_.../best_model.pt
  python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_.../best_model.pt --model BaselineTransformer
  python scripts/phase4_experiments/run_phase4_finetune.py --smoke  # быстрый smoke-test
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
import torch.nn as nn
from torch.utils.data import DataLoader

# Добавляем корень проекта в PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.augmented_dataset import AugmentedSpectralDataset, compute_num_channels
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.labels import get_target_columns
from osc_tools.ml.class_weights import compute_pos_weight_from_loader


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

NUM_BASE_CLASSES = 4  # Target_Normal, Target_ML_1, Target_ML_2, Target_ML_3

def get_finetune_config() -> dict:
    """Конфигурация fine-tuning."""
    return {
        # Данные
        'window_size': 320,
        'downsampling_stride': 16,
        'feature_mode': 'phase_polar',
        'num_harmonics': 9,
        'sub_periods': [2, 4, 6, 10],
        'use_augmentation': True,
        'use_low_harmonics': True,
        'include_symmetric': True,         # Симметричные составляющие (I1,I2,I0,U1,U2,U0)
        'future_periods': 2,               # Будущие периоды для расширенной метки
        'batch_size': 32,
        'val_batch_size': 64,
        'num_workers': 0,
        'val_split': 0.2,
        'target_level': 'base',  # 4 класса

        # Модель (должно совпадать с pretrain; по умолчанию = light)
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 220,  # 8 × (9+4) × 2 = 208 + 6 × 2 = 12 = 220
        'd_model': 48,
        'num_heads': 4,
        'num_layers': 6,
        'kan_grid_size': 5,
        'dropout': 0.1,

        # Fine-tuning специфика
        'num_classes': NUM_BASE_CLASSES,
        'zone_size': 1,  # Каждый временной шаг = отдельная зона

        # Обучение
        'epochs': 50,
        'lr_backbone': 1e-5,     # Низкий LR для backbone (уже обучен SSL)
        'lr_head': 5e-4,          # Высокий LR для новой головы
        'weight_decay': 1e-5,
        'warmup_epochs': 2,
        'use_amp': True,
        'grad_clip': 1.0,
        'accumulation_steps': 8,  # effective batch = 32 × 8 = 256

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,
        'seed': 42,

        # Данные (путь)
        'data_dir': str(PROJECT_ROOT / 'data' / 'ml_datasets'),
        'precomputed_file': 'test_precomputed.csv',
    }


# ---------------------------------------------------------------------------
# Создание модели с классификационной головой
# ---------------------------------------------------------------------------

def create_model_for_finetune(
    config: dict,
    ssl_checkpoint_path: str | None = None,
) -> tuple[nn.Module, dict]:
    """Создаёт модель для fine-tuning, опционально загружая SSL-чекпоинт.

    Стратегия загрузки:
    1. Создаём модель с num_classes (имеет и ssl_head, и cls_head)
    2. Если есть SSL-чекпоинт — загружаем веса backbone (strict=False,
       чтобы пропустить ssl_head если размеры не совпадают)
    3. cls_head инициализируется случайно (новая задача)

    Returns:
        (model, ssl_info) — модель и информация от SSL-чекпоинта
    """
    model_type = config['model_type']

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            zone_size=config['zone_size'],
            kan_grid_size=config['kan_grid_size'],
            dropout=config['dropout'],
            max_seq_len=64,
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            zone_size=config['zone_size'],
            dropout=config['dropout'],
            max_seq_len=64,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    ssl_info = {}

    if ssl_checkpoint_path is not None:
        ckpt_path = Path(ssl_checkpoint_path)
        if ckpt_path.exists():
            print(f"Загрузка SSL-чекпоинта: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            ssl_state = ckpt['model_state_dict']

            # Фильтруем cls_head (его нет в SSL-чекпоинте) — strict=False
            missing, unexpected = model.load_state_dict(ssl_state, strict=False)
            print(f"  Пропущенные ключи (ожидаемо): {missing}")
            if unexpected:
                print(f"  Неожиданные ключи: {unexpected}")

            ssl_info = {
                'ssl_epoch': ckpt.get('epoch', -1),
                'ssl_val_loss': ckpt.get('best_val_loss', float('inf')),
                'ssl_config': ckpt.get('config', {}),
            }
            print(f"  SSL эпоха: {ssl_info['ssl_epoch']}, val_loss: {ssl_info['ssl_val_loss']:.4f}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: SSL-чекпоинт не найден: {ckpt_path}")

    return model, ssl_info


# ---------------------------------------------------------------------------
# Подготовка данных (с метками)
# ---------------------------------------------------------------------------

def _split_files_train_val(
    df: pl.DataFrame, val_split: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Разделяет файлы на train/val (детерминированно)."""
    files = sorted(df['file_name'].unique().to_list())
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]
    return train_files, list(val_files)


def prepare_finetune_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Создаёт train/val DataLoader-ы для fine-tuning (с метками).

    Если use_augmentation или use_low_harmonics — AugmentedSpectralDataset (on-the-fly FFT).
    Иначе — PrecomputedDataset (legacy).

    Returns:
        (train_loader, val_loader, target_columns)
    """
    data_path = Path(config['data_dir']) / config['precomputed_file']
    print(f"Загрузка данных: {data_path}")
    df = pl.read_csv(str(data_path))
    print(f"  Строк: {df.height:,}, Колонок: {df.width}")

    # Целевые колонки
    target_columns = get_target_columns(config['target_level'], df)
    print(f"  Целевые колонки ({config['target_level']}): {target_columns}")

    # Разделяем файлы
    train_files, val_files = _split_files_train_val(
        df, config['val_split'], config['seed'],
    )
    train_df = df.filter(pl.col('file_name').is_in(train_files))
    val_df = df.filter(pl.col('file_name').is_in(val_files))
    print(f"  Файлов: train={len(train_files)}, val={len(val_files)}")
    print(f"  Строк: train={train_df.height:,}, val={val_df.height:,}")

    # Индексы создаются ниже (в зависимости от режима)
    window_size = config['window_size']
    stride = config['downsampling_stride']

    use_augmented = config.get('use_augmentation', False) or config.get('use_low_harmonics', False)

    if use_augmented:
        sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if config.get('use_low_harmonics') else []
        include_symmetric = config.get('include_symmetric', True)
        augmenter = TimeSeriesAugmenter() if config.get('use_augmentation') else None
        future_periods = config.get('future_periods', 2)

        # Пересчёт num_input_channels
        num_lh = len(sub_periods)
        actual_channels = compute_num_channels(config.get('num_harmonics', 9), num_lh, include_symmetric)
        if config.get('num_input_channels') != actual_channels:
            print(f"  [!] Корректировка num_input_channels: {config['num_input_channels']} → {actual_channels}")
            config['num_input_channels'] = actual_channels

        # Индексы: учитываем будущие периоды
        full_window = window_size + future_periods * 32
        train_indices = PrecomputedDataset.create_indices(
            train_df, window_size=full_window, mode='val', stride=stride,
        )
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=full_window, mode='val', stride=stride,
        )
        print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,}")

        train_boundaries = AugmentedSpectralDataset.compute_file_boundaries(train_df)
        val_boundaries = AugmentedSpectralDataset.compute_file_boundaries(val_df)

        train_ds = AugmentedSpectralDataset(
            dataframe=train_df,
            file_boundaries=train_boundaries,
            indices=train_indices,
            window_size=window_size,
            num_harmonics=config['num_harmonics'],
            sub_periods=sub_periods if sub_periods else None,
            include_symmetric=include_symmetric,
            downsampling_stride=stride,
            future_periods=future_periods,
            mask_ratio=0.0,         # Нет маскирования при finetune
            augmenter=augmenter,
            target_columns=target_columns,
            mode='classify',
            target_window_mode='any_in_window',
        )
        val_ds = AugmentedSpectralDataset(
            dataframe=val_df,
            file_boundaries=val_boundaries,
            indices=val_indices,
            window_size=window_size,
            num_harmonics=config['num_harmonics'],
            sub_periods=sub_periods if sub_periods else None,
            include_symmetric=include_symmetric,
            downsampling_stride=stride,
            future_periods=future_periods,
            mask_ratio=0.0,
            augmenter=None,
            target_columns=target_columns,
            mode='classify',
            target_window_mode='any_in_window',
        )
    else:
        # Legacy: PrecomputedDataset
        train_indices = PrecomputedDataset.create_indices(
            train_df, window_size=window_size, mode='val', stride=stride,
        )
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=window_size, mode='val', stride=stride,
        )
        print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,}")

        train_ds = PrecomputedDataset(
            dataframe=train_df,
            indices=train_indices,
            window_size=window_size,
            feature_mode=config['feature_mode'],
            target_columns=target_columns,
            sampling_strategy='stride',
            downsampling_stride=stride,
            target_position=window_size - 1,
            target_window_mode='any_in_window',
            num_harmonics=config['num_harmonics'],
        )
        val_ds = PrecomputedDataset(
            dataframe=val_df,
            indices=val_indices,
            window_size=window_size,
            feature_mode=config['feature_mode'],
            target_columns=target_columns,
            sampling_strategy='stride',
            downsampling_stride=stride,
            target_position=window_size - 1,
            target_window_mode='any_in_window',
            num_harmonics=config['num_harmonics'],
        )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['val_batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader, target_columns


# ---------------------------------------------------------------------------
# Обучение и валидация
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    """Одна эпоха fine-tuning с gradient accumulation.

    Модель выдаёт (B, num_zones, num_classes). Для loss используем
    последнюю зону (соответствует метке последней точки окна).
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        x = x.to(device)       # (B, C, T)
        y = y.to(device)       # (B, num_classes)

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x, mode='classify')
            logits = out['classify']  # (B, num_zones, num_classes)
            # Берём последнюю зону для сравнения с window-level label
            logits_last = logits[:, -1, :]  # (B, num_classes)

        # Loss вычисляем в float32: BCEWithLogitsLoss c pos_weight
        # может переполниться в float16 (log(1+exp(x)) при x > 11)
        loss = loss_fn(logits_last.float(), y)

        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

        loss_val = loss.item() * (accumulation_steps if accumulation_steps > 1 else 1)
        if not np.isnan(loss_val):
            total_loss += loss_val
            num_batches += 1

        # Сохраняем предсказания для метрик
        with torch.no_grad():
            probs = torch.sigmoid(logits_last.float()).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)

    # Вычисляем метрики
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    metrics['time_sec'] = elapsed

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Валидация."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    t0 = time.perf_counter()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x, mode='classify')
        logits = out['classify']
        logits_last = logits[:, -1, :]
        loss = loss_fn(logits_last, y)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits_last).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(y.cpu().numpy())

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    metrics['time_sec'] = elapsed

    return metrics


def _compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.7,
) -> dict[str, float]:
    """Вычисляет Macro-F1, ROC-AUC, Accuracy.

    Args:
        preds: (N, C) вероятности после sigmoid
        targets: (N, C) бинарные метки
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    preds_bin = (preds >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    metrics = {}

    # Macro-F1
    metrics['macro_f1'] = float(f1_score(
        targets_int, preds_bin, average='macro', zero_division=0,
    ))

    # Per-class F1
    per_cls_f1 = f1_score(targets_int, preds_bin, average=None, zero_division=0)
    for i, f in enumerate(per_cls_f1):
        metrics[f'f1_class_{i}'] = float(f)

    # ROC-AUC (macro, только для классов с обоими значениями 0 и 1)
    try:
        valid_cols = []
        for c in range(targets.shape[1]):
            if len(np.unique(targets_int[:, c])) > 1:
                valid_cols.append(c)
        if valid_cols:
            metrics['roc_auc'] = float(roc_auc_score(
                targets_int[:, valid_cols], preds[:, valid_cols],
                average='macro',
            ))
        else:
            metrics['roc_auc'] = 0.0
    except ValueError:
        metrics['roc_auc'] = 0.0

    # Exact match
    metrics['exact_match'] = float((preds_bin == targets_int).all(axis=1).mean())

    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_f1: float,
    config: dict,
) -> None:
    """Сохраняет fine-tuning checkpoint."""
    state = {
        'epoch': epoch,
        'best_val_f1': best_val_f1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Основной цикл fine-tuning
# ---------------------------------------------------------------------------

def finetune(config: dict, ssl_checkpoint: str | None = None) -> Path:
    """Supervised fine-tuning на задачу классификации.

    1. Загружает данные с метками
    2. Создаёт модель, инициализирует из SSL-чекпоинта
    3. Обучает с раздельным LR (backbone vs head)
    4. Сохраняет лучшую модель по Macro-F1

    Returns:
        Путь к директории с результатами
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config['save_dir']) / f"finetune_{config['model_type']}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем конфиг
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"FINE-TUNING — {config['model_type']}")
    print(f"SSL чекпоинт: {ssl_checkpoint or 'НЕТ (random init)'}")
    print(f"Сохранение: {save_dir}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Данные ---
    train_loader, val_loader, target_columns = prepare_finetune_dataloaders(config)

    # Статистика классов
    _print_class_distribution(train_loader, target_columns)

    # --- Модель ---
    model, ssl_info = create_model_for_finetune(config, ssl_checkpoint)
    model = model.to(device)
    n_params = model.num_parameters()
    print(f"Параметры модели: {n_params:,}")

    # --- Loss: BCEWithLogitsLoss с pos_weight для балансировки ---
    pos_weight = compute_pos_weight_from_loader(train_loader, device=device)
    print(f"pos_weight: {pos_weight.cpu().numpy()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Оптимизатор: раздельный LR ---
    # Backbone (всё кроме cls_head) — низкий LR
    # cls_head — высокий LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'cls_head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr_backbone']},
        {'params': head_params, 'lr': config['lr_head']},
    ], weight_decay=config['weight_decay'])

    print(f"Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={config['lr_backbone']})")
    print(f"Head params: {sum(p.numel() for p in head_params):,} (lr={config['lr_head']})")

    # --- LR Scheduler ---
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config['epochs']
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-7,
    )

    # --- AMP ---
    scaler = None
    if config['use_amp'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision: включён (AMP)")

    # --- Лог ---
    history = []
    log_path = save_dir / 'training_log.jsonl'
    best_val_f1 = 0.0

    print(f"\nНачало fine-tuning: {total_epochs} эпох, batch_size={config['batch_size']}")
    print(f"Целевые классы: {target_columns}")
    accum = config.get('accumulation_steps', 1)
    print(f"Gradient accumulation: {accum} шагов → effective batch = {config['batch_size'] * accum}")
    print("-" * 70)

    for epoch in range(total_epochs):
        # Warmup LR (линейный от 0 до target)
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]['lr'] = config['lr_backbone'] * frac
            optimizer.param_groups[1]['lr'] = config['lr_head'] * frac

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, config.get('grad_clip', 1.0),
            accumulation_steps=accum,
        )

        # Val
        val_metrics = validate(model, val_loader, loss_fn, device)

        # LR step
        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        # Логирование
        is_best = val_metrics['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['macro_f1']

        vram_mb = 0.0
        if device.type == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['macro_f1'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['macro_f1'],
            'val_roc_auc': val_metrics.get('roc_auc', 0.0),
            'val_exact_match': val_metrics.get('exact_match', 0.0),
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'train_time': train_metrics['time_sec'],
            'val_time': val_metrics['time_sec'],
            'vram_mb': vram_mb,
            'is_best': is_best,
        }
        # Per-class F1 на валидации
        for i, col in enumerate(target_columns):
            record[f'val_f1_{col}'] = val_metrics.get(f'f1_class_{i}', 0.0)

        history.append(record)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        marker = ' ★' if is_best else ''
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"F1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}{marker} | "
            f"AUC={val_metrics.get('roc_auc', 0):.4f} | "
            f"time={train_metrics['time_sec'] + val_metrics['time_sec']:.1f}s"
        )

        # Checkpointing
        save_checkpoint(
            save_dir / 'latest_checkpoint.pt',
            model, optimizer, cosine_scheduler, scaler,
            epoch, best_val_f1, config,
        )
        if is_best:
            save_checkpoint(
                save_dir / 'best_model.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config,
            )
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint(
                save_dir / f'checkpoint_epoch_{epoch:03d}.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config,
            )

    # Итоги
    print("\n" + "=" * 60)
    print("FINE-TUNING ЗАВЕРШЁН")
    print(f"  Лучший val Macro-F1: {best_val_f1:.4f}")
    if history:
        best_epoch = max(history, key=lambda r: r['val_f1'])['epoch']
        print(f"  Лучшая эпоха: {best_epoch}")
        best_rec = history[best_epoch]
        print(f"  ROC-AUC: {best_rec.get('val_roc_auc', 0):.4f}")
        print(f"  Per-class F1:")
        for col in target_columns:
            f1_val = best_rec.get(f'val_f1_{col}', 0)
            print(f"    {col}: {f1_val:.4f}")
    print(f"  Результаты: {save_dir}")
    print("=" * 60)

    # Сохраняем финальную сводку
    summary = {
        'model_type': config['model_type'],
        'ssl_checkpoint': ssl_checkpoint,
        'best_epoch': best_epoch if history else -1,
        'best_val_f1': best_val_f1,
        'best_val_roc_auc': history[best_epoch].get('val_roc_auc', 0) if history else 0,
        'total_epochs': total_epochs,
        'num_params': n_params,
        'target_columns': target_columns,
        'ssl_info': ssl_info,
    }
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return save_dir


def _print_class_distribution(loader: DataLoader, target_columns: list[str]) -> None:
    """Печатает распределение классов в данных."""
    total = 0
    pos_counts = None
    for _, y in loader:
        if pos_counts is None:
            pos_counts = torch.zeros(y.shape[1], dtype=torch.float64)
        pos_counts += y.float().sum(dim=0).double()
        total += y.shape[0]

    if pos_counts is None:
        return

    print(f"\n  Распределение классов (N={total}):")
    for i, col in enumerate(target_columns):
        n_pos = int(pos_counts[i].item())
        pct = 100.0 * n_pos / total if total > 0 else 0
        print(f"    {col}: {n_pos} ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к SSL-чекпоинту для инициализации')
    parser.add_argument('--model', choices=['PhysicalKANTransformer', 'BaselineTransformer'],
                        default='PhysicalKANTransformer', help='Тип модели')
    parser.add_argument('--complexity', choices=['light', 'medium', 'heavy'],
                        default=None, help='Уровень сложности модели')
    parser.add_argument('--epochs', type=int, default=None, help='Число эпох')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--smoke', action='store_true', help='Smoke-test (2 эпохи)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Отключить аугментацию на сырых данных')
    parser.add_argument('--no-low-harmonics', action='store_true',
                        help='Отключить низшие гармоники')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help='Шаги gradient accumulation (override)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_finetune_config()

    config['model_type'] = args.model
    if args.complexity:
        level = COMPLEXITY_LEVELS[args.complexity]
        config.update(level)
        print(f"Сложность: {args.complexity} → {level}")
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.accumulation_steps is not None:
        config['accumulation_steps'] = args.accumulation_steps
    if args.no_augmentation:
        config['use_augmentation'] = False
    if args.no_low_harmonics:
        config['use_low_harmonics'] = False
        config['include_symmetric'] = False
        config['num_input_channels'] = 144
    if args.smoke:
        config['epochs'] = 2
        config['batch_size'] = 4
        config['val_batch_size'] = 4
        config['use_augmentation'] = False
        config['use_low_harmonics'] = False
        config['include_symmetric'] = False
        config['num_input_channels'] = 144
        config['accumulation_steps'] = 1
        config['future_periods'] = 0

    finetune(config, ssl_checkpoint=args.checkpoint)


if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Раскомментируйте нужный блок и запустите файл напрямую.
    # Для CLI: python run_phase4_finetune.py --checkpoint path/to/best_model.pt
    # =================================================================

    # --- Тип модели (должен совпадать с pretrain) ---
    MODEL_TYPE = 'PhysicalKANTransformer'   # 'PhysicalKANTransformer' | 'BaselineTransformer'

    # --- Сложность (должна совпадать с pretrain) ---
    SELECTED_COMPLEXITY = 'light'           # 'light' | 'medium' | 'heavy'

    # --- Путь к SSL-чекпоинту (None = random init) ---
    SSL_CHECKPOINT = None
    # SSL_CHECKPOINT = 'experiments/phase4/pretrain_.../best_model.pt'

    # --- Эпохи ---
    EPOCHS = 50

    # --- Аугментация и признаки ---
    USE_AUGMENTATION = True
    USE_LOW_HARMONICS = True
    INCLUDE_SYMMETRIC = True                # Симметричные составляющие
    FUTURE_PERIODS = 2                      # Будущие периоды (расширяет метки)

    # --- Gradient accumulation ---
    ACCUMULATION_STEPS = 8

    # --- Частота чекпоинтов ---
    CHECKPOINT_FREQUENCY = 5

    # =================================================================

    config = get_finetune_config()
    config['model_type'] = MODEL_TYPE
    config['epochs'] = EPOCHS
    config['use_augmentation'] = USE_AUGMENTATION
    config['use_low_harmonics'] = USE_LOW_HARMONICS
    config['include_symmetric'] = INCLUDE_SYMMETRIC
    config['future_periods'] = FUTURE_PERIODS
    config['accumulation_steps'] = ACCUMULATION_STEPS
    config['checkpoint_frequency'] = CHECKPOINT_FREQUENCY

    level = COMPLEXITY_LEVELS[SELECTED_COMPLEXITY]
    config.update(level)

    # Пересчёт каналов
    num_lh = len(config['sub_periods']) if USE_LOW_HARMONICS else 0
    config['num_input_channels'] = compute_num_channels(
        config['num_harmonics'], num_lh, INCLUDE_SYMMETRIC,
    )
    if not USE_LOW_HARMONICS and not INCLUDE_SYMMETRIC:
        config['num_input_channels'] = 144

    finetune(config, ssl_checkpoint=SSL_CHECKPOINT)
