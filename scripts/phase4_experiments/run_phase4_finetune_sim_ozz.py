"""
Fine-tuning скрипт для Этапа 4.5: Simulated_OZZ_v1.

Обучает Physical KAN-Transformer / BaselineTransformer на задачу
классификации типов ОЗЗ/ДПОЗЗ (4 класса + implicit Normal) на
симулированных осциллограммах RTDS.

Ключевые отличия от ``run_phase4_finetune.py``:
- **Lazy dataset:** файлы грузятся по требованию (62 ГБ ≫ RAM).
- **Без ресэмплирования:** FFT напрямую по сырым данным (Fs ~ 38.4..40 кГц, per-file).
- **3I0 → IN, 3U0 → UN:** каналы нулевой последовательности.
- **Stride = 1/8 периода** (96..100 отсчётов в зависимости от Fs).
- **Split 80/20:** стратификация по типу дуги X (1..4).

Примеры::

    python scripts/phase4_experiments/run_phase4_finetune_sim_ozz.py --smoke
    python scripts/phase4_experiments/run_phase4_finetune_sim_ozz.py --max-files 100
    python scripts/phase4_experiments/run_phase4_finetune_sim_ozz.py --resume experiments/phase4/sim_ozz_.../latest_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем общие функции из оригинального finetune
from scripts.phase4_experiments.run_phase4_finetune import (
    COMPLEXITY_LEVELS,
    create_model_for_finetune,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    _reduce_logits_targets,
    _compute_supervision_pos_weight,
    _save_confusion_matrix,
    _print_class_distribution,
)

from osc_tools.ml.augmented_dataset import compute_num_channels
from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex,
    SimOZZLazyDataset,
    ARC_TYPES,
    TARGET_COLUMNS,
)
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from osc_tools.ml.labels import get_target_columns


# ---------------------------------------------------------------------------
# Конфигурация для Simulated_OZZ_v1
# ---------------------------------------------------------------------------

NUM_SIM_OZZ_CLASSES = 4   # Stable, Petersen, PetersSlepian, Beliakov


def get_sim_ozz_config() -> dict:
    """Конфигурация fine-tuning для Simulated_OZZ_v1."""
    return {
        # Данные
        'data_dir': str(PROJECT_ROOT / 'data' / 'Simulated_OZZ_v1'),
        'target_level': 'sim_ozz',
        'num_harmonics': 9,
        'sub_periods': [2, 4, 6, 10],
        'include_symmetric': True,
        'stride_fraction': 8,              # 1/8 периода (96..100 отсчётов, per-file)
        'num_periods_window': 10,           # 10 периодов ≈ 7690 отсчётов
        'use_augmentation': False,          # Пока без аугментации (для MVP)
        'zone_target_aggregation': 'max',
        'val_split': 0.2,                  # 80/20
        'train_batches_per_epoch': 64,
        'batch_size': 32,
        'val_batch_size': 64,
        'num_workers': 0,                  # lazy loading + LRU → 0 workers безопаснее
        'cache_size': 500,                 # LRU-кэш файлов (~500 МБ)

        # Модель
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 220,          # 8 × (9+4) × 2 + 6 × 2
        'd_model': 48,
        'num_heads': 4,
        'num_layers': 6,
        'kan_grid_size': 5,
        'dropout': 0.1,
        'cls_head_type': 'kan',
        'use_angle_gate': True,
        'use_mixed_layer_norm': False,

        # Fine-tuning
        'num_classes': NUM_SIM_OZZ_CLASSES,
        'zone_size': 1,
        'supervision_mode': 'zone',
        'num_future_zones': 0,

        # Обучение
        'epochs': 50,
        'lr_backbone': 1e-3,               # Нет SSL → единый LR
        'lr_head': 1e-3,
        'weight_decay': 1e-5,
        'warmup_epochs': 2,
        'scheduler': 'cosine_warm_restarts',
        'scheduler_T0': 50,
        'scheduler_T_mult': 2,
        'scheduler_eta_min': 1e-7,
        'use_amp': True,
        'grad_clip': 1.0,
        'accumulation_steps': 8,

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,
        'seed': 42,
    }


# ---------------------------------------------------------------------------
# Подготовка данных (lazy)
# ---------------------------------------------------------------------------

def prepare_sim_ozz_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Создаёт train/val DataLoader-ы для Simulated_OZZ_v1 (lazy loading).

    Returns:
        (train_loader, val_loader, target_columns)
    """
    data_dir = Path(config['data_dir'])
    max_files = config.get('max_files')

    print(f"Сканирование файлов: {data_dir}")
    file_index = SimOZZFileIndex.from_directory(
        data_dir, max_files=max_files, verbose=True,
    )

    # Стратифицированный split
    all_names = [fi.path.name for fi in file_index.files]
    train_names, val_names, _, stats = stratified_sim_ozz_split(
        all_names,
        val_split=config['val_split'],
        seed=config['seed'],
    )
    print(f"  Split: train={len(train_names)}, val={len(val_names)}")
    for split_name, split_stats in stats.items():
        if any(v > 0 for v in split_stats.values()):
            print(f"    {split_name}: {split_stats}")

    # Сохраняем split
    config['_split_data'] = {
        'train_files': train_names,
        'val_files': val_names,
    }

    train_paths = [data_dir / n for n in train_names]
    val_paths = [data_dir / n for n in val_names]

    target_columns = get_target_columns(config['target_level'])
    config['num_classes'] = len(target_columns)

    # Пересчёт num_input_channels
    num_lh = len(config.get('sub_periods', [2, 4, 6, 10]))
    include_sym = config.get('include_symmetric', True)
    actual_ch = compute_num_channels(config['num_harmonics'], num_lh, include_sym)
    if config.get('num_input_channels') != actual_ch:
        print(f"  [!] num_input_channels: {config['num_input_channels']} -> {actual_ch}")
        config['num_input_channels'] = actual_ch

    # Общие параметры dataset
    ds_kwargs = dict(
        file_index=file_index,
        num_periods_window=config['num_periods_window'],
        stride_fraction=config['stride_fraction'],
        num_harmonics=config['num_harmonics'],
        sub_periods=config.get('sub_periods'),
        include_symmetric=include_sym,
        target_level=config['target_level'],
        mode='classify',
        zone_target_aggregation=config['zone_target_aggregation'],
        cache_size=config.get('cache_size', 500),
    )

    train_ds = SimOZZLazyDataset(
        file_paths=train_paths,
        augmenter=None,             # TODO: добавить аугментацию
        verbose=True,
        **ds_kwargs,
    )

    val_ds = SimOZZLazyDataset(
        file_paths=val_paths,
        augmenter=None,
        verbose=True,
        **ds_kwargs,
    )

    # supervision_mode — устанавливаем на dataset
    for ds in (train_ds, val_ds):
        ds.supervision_mode = config.get('supervision_mode', 'zone')

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader, target_columns


# ---------------------------------------------------------------------------
# Основной цикл fine-tuning (адаптирован из run_phase4_finetune)
# ---------------------------------------------------------------------------

def finetune_sim_ozz(
    config: dict,
    resume_path: str | None = None,
    reset_optimizer: bool = False,
) -> Path:
    """Fine-tuning на Simulated_OZZ_v1."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = config.get('run_name')
    if run_name:
        dir_name = run_name
    else:
        dir_name = f"sim_ozz_finetune_{config['model_type']}_{timestamp}"
    save_dir = Path(config['save_dir']) / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SIM_OZZ FINE-TUNING — {config['model_type']}")
    if resume_path:
        print(f"ПРОДОЛЖЕНИЕ: {resume_path}")
    else:
        print("Random init (без SSL)")
    print(f"Сохранение: {save_dir}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Данные ---
    train_loader, val_loader, target_columns = prepare_sim_ozz_dataloaders(config)

    # Сохраняем конфиг и split
    split_data = config.pop('_split_data', None)
    # Убираем non-serializable перед сохранением
    config_save = {k: v for k, v in config.items() if not k.startswith('_')}
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    if split_data:
        with open(save_dir / 'split.json', 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Split: {save_dir / 'split.json'}")

    # Статистика классов
    _print_class_distribution(train_loader, target_columns)

    # --- Модель (random init) ---
    model, _ = create_model_for_finetune(config, ssl_checkpoint_path=None)
    model = model.to(device)
    n_params = model.num_parameters()
    print(f"Параметры: {n_params:,}")

    supervision_mode = config.get('supervision_mode', 'zone')

    # --- Loss ---
    pos_weight = _compute_supervision_pos_weight(
        train_loader, supervision_mode=supervision_mode, device=device,
    )
    print(f"pos_weight ({supervision_mode}): {pos_weight.cpu().numpy()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer (единый LR, нет SSL) ---
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

    print(f"Backbone: {sum(p.numel() for p in backbone_params):,} params "
          f"(lr={config['lr_backbone']})")
    print(f"Head: {sum(p.numel() for p in head_params):,} params "
          f"(lr={config['lr_head']})")

    # --- LR scheduler ---
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config['epochs']
    eta_min = config.get('scheduler_eta_min', 1e-7)
    T_0 = config.get('scheduler_T0', 50)
    T_mult = config.get('scheduler_T_mult', 2)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, T_0), T_mult=T_mult, eta_min=eta_min,
    )
    print(f"Scheduler: CosineWarmRestarts T_0={T_0}, T_mult={T_mult}")

    # --- AMP ---
    scaler = None
    if config['use_amp'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("AMP: ON")

    # --- Resume ---
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float('inf')

    if resume_path is not None:
        rp = Path(resume_path)
        if rp.exists():
            print(f"Восстановление: {rp}")
            meta = load_checkpoint(
                rp, model, optimizer, cosine_scheduler, scaler,
                reset_optimizer=reset_optimizer,
            )
            start_epoch = meta['epoch'] + 1
            best_val_f1 = meta['best_val_f1']
            best_val_loss = meta.get('best_val_loss', float('inf'))
            if not reset_optimizer:
                print(f"  epoch={start_epoch}, best_f1={best_val_f1:.4f}")
        else:
            print(f"  [!] Чекпоинт не найден: {rp}")

    # --- Training loop ---
    history = []
    log_path = save_dir / 'training_log.jsonl'
    accum = config.get('accumulation_steps', 1)

    print(f"\nОбучение: {total_epochs} эпох, batch={config['batch_size']}, "
          f"accum={accum} (eff={config['batch_size'] * accum})")
    print(f"Классы: {target_columns}")
    print("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        # Warmup
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]['lr'] = config['lr_backbone'] * frac
            optimizer.param_groups[1]['lr'] = config['lr_head'] * frac

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, config.get('grad_clip', 1.0),
            accumulation_steps=accum,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        val_metrics = validate(
            model, val_loader, loss_fn, device,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        is_best = val_metrics['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['macro_f1']

        is_best_loss = val_metrics['loss'] < best_val_loss
        if is_best_loss:
            best_val_loss = val_metrics['loss']

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
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'train_time': train_metrics['time_sec'],
            'val_time': val_metrics['time_sec'],
            'vram_mb': vram_mb,
            'is_best': is_best,
            'is_best_loss': is_best_loss,
        }
        for i, col in enumerate(target_columns):
            record[f'val_f1_{col}'] = val_metrics.get(f'f1_class_{i}', 0.0)

        history.append(record)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        marker = ' ★' if is_best else (' ★L' if is_best_loss else '')
        lr_bb = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"F1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}{marker} | "
            f"AUC={val_metrics.get('roc_auc', 0):.4f} | "
            f"lr={lr_bb:.1e} | "
            f"time={train_metrics['time_sec'] + val_metrics['time_sec']:.1f}s"
        )
        cls_f1s = [f"{val_metrics.get(f'f1_class_{i}', 0):.3f}"
                   for i in range(len(target_columns))]
        print(f"         Per-class F1: {cls_f1s}")

        # Checkpointing
        save_checkpoint(
            save_dir / 'latest_checkpoint.pt',
            model, optimizer, cosine_scheduler, scaler,
            epoch, best_val_f1, config_save, best_val_loss,
        )
        if is_best:
            save_checkpoint(
                save_dir / 'best_model.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )
        if is_best_loss:
            save_checkpoint(
                save_dir / 'best_model_loss.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint(
                save_dir / f'checkpoint_{epoch:03d}.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )

    # --- Итоги ---
    print("\n" + "=" * 60)
    print("SIM_OZZ FINE-TUNING ЗАВЕРШЁН")
    print(f"  Best val F1: {best_val_f1:.4f}")
    if history:
        best_rec = max(history, key=lambda r: r['val_f1'])
        best_ep = best_rec['epoch']
        print(f"  Best epoch: {best_ep}")
        for col in target_columns:
            print(f"    {col}: {best_rec.get(f'val_f1_{col}', 0):.4f}")
    print(f"  Результаты: {save_dir}")
    print("=" * 60)

    # Confusion matrix
    _save_confusion_matrix(
        model, val_loader, loss_fn, device, target_columns,
        save_dir / 'confusion_matrix.txt',
    )

    # Summary
    summary = {
        'model_type': config['model_type'],
        'best_epoch': best_ep if history else -1,
        'best_val_f1': best_val_f1,
        'best_val_roc_auc': best_rec.get('val_roc_auc', 0) if history else 0,
        'total_epochs': total_epochs,
        'num_params': n_params,
        'target_columns': target_columns,
        'dataset': 'Simulated_OZZ_v1',
    }
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    if history:
        import csv
        csv_path = save_dir / 'training_curves.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"  Кривые: {csv_path}")

    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Phase 4.5: Simulated_OZZ Fine-tuning')
    p.add_argument('--model', choices=['PhysicalKANTransformer', 'BaselineTransformer'],
                   default='PhysicalKANTransformer')
    p.add_argument('--complexity', choices=['light', 'medium', 'heavy'], default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--max-files', type=int, default=None,
                   help='Ограничить число файлов (для отладки)')
    p.add_argument('--accumulation-steps', type=int, default=None)
    p.add_argument('--smoke', action='store_true', help='Smoke-test (2 эпохи, 20 файлов)')
    p.add_argument('--resume', type=str, default=None,
                   help='Путь к чекпоинту для продолжения')
    p.add_argument('--reset-optimizer', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = get_sim_ozz_config()

    config['model_type'] = args.model
    if args.complexity:
        config.update(COMPLEXITY_LEVELS[args.complexity])
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.accumulation_steps is not None:
        config['accumulation_steps'] = args.accumulation_steps
    if args.max_files is not None:
        config['max_files'] = args.max_files

    if args.smoke:
        config['epochs'] = 2
        config['batch_size'] = 4
        config['val_batch_size'] = 4
        config['accumulation_steps'] = 1
        config['max_files'] = 20
        config['cache_size'] = 50
        config['include_symmetric'] = False
        num_lh = len(config['sub_periods'])
        config['num_input_channels'] = compute_num_channels(
            config['num_harmonics'], num_lh, False,
        )

    finetune_sim_ozz(
        config,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
    )


if __name__ == '__main__':
    # =================================================================
    # РУЧНОЙ ЗАПУСК (раскомментируйте нужное)
    # =================================================================

    MODEL_TYPE = 'PhysicalKANTransformer'
    SELECTED_COMPLEXITY = 'light'
    RESUME_PATH = None
    RESET_OPTIMIZER = False
    EPOCHS = 50
    MAX_FILES = None            # None = все файлы

    # =================================================================

    config = get_sim_ozz_config()
    config['model_type'] = MODEL_TYPE
    config['epochs'] = EPOCHS
    if MAX_FILES is not None:
        config['max_files'] = MAX_FILES

    level = COMPLEXITY_LEVELS[SELECTED_COMPLEXITY]
    config.update(level)

    # Каналы
    num_lh = len(config['sub_periods'])
    config['num_input_channels'] = compute_num_channels(
        config['num_harmonics'], num_lh, config['include_symmetric'],
    )

    finetune_sim_ozz(
        config,
        resume_path=RESUME_PATH,
        reset_optimizer=RESET_OPTIMIZER,
    )
