"""
Скрипт запуска экспериментов Фазы 4: Physical KAN-Transformer.

Режимы:
  --mode pretrain  : Self-supervised pre-training (реконструкция спектров)
  --mode finetune  : Supervised fine-tuning (классификация)
  --mode smoke     : Быстрый smoke-test (1-2 эпохи, проверка OOM)

Примеры:
  python scripts/phase4_experiments/run_phase4_pretrain.py --mode smoke
  python scripts/phase4_experiments/run_phase4_pretrain.py --mode pretrain --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Добавляем корень проекта в PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.losses import ComplexMSELoss, SpectralReconstructionLoss


def get_default_config(mode: str = 'smoke') -> dict:
    """Конфигурация по умолчанию для разных режимов."""

    base = {
        # Данные
        'window_size': 320,
        'downsampling_stride': 16,
        'feature_mode': 'phase_polar', # TODO: тут он более хитрый так-то...И их больше, см. примечания в "ssl_dataset.py" и исходную инструкцию (надеюсь оттуда не удалилось)
        'num_harmonics': 9,
        'future_periods': 2,
        'mask_ratio': 0.25,
        'batch_size': 32,

        # Модель
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 144,  # 8 каналов * 9 гармоник * 2 (mag+angle) = 144
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 4,
        'kan_grid_size': 5,
        'dropout': 0.1,

        # Обучение
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lr_scheduler': 'cosine',
        'use_amp': True,  # Mixed precision для 8 ГБ VRAM

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,
        'seed': 42,
    }

    if mode == 'smoke':
        base.update({
            'epochs': 2,
            'batch_size': 4,
            'd_model': 32,
            'num_layers': 2,
            'num_heads': 2,
            'use_amp': False,
        })
    elif mode == 'pretrain':
        pass  # Базовые параметры
    elif mode == 'finetune':
        base.update({
            'epochs': 30,
            'learning_rate': 5e-5,
        })

    return base


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
            dropout=config['dropout'],
            max_seq_len=64,  # Достаточно для stride=16, window=320+64
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
        config_cls = {**config, 'model_type': model_type}
        if model_type == 'PhysicalKANTransformer':
            model_cls = PhysicalKANTransformer(
                num_input_channels=C, d_model=config['d_model'],
                num_heads=config['num_heads'], num_layers=config['num_layers'],
                num_classes=4, zone_size=16, kan_grid_size=config['kan_grid_size'],
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
    print("\n--- ComplexMSE Loss ---")
    B, C_half, T = 4, 72, 18  # C_half = num_pairs * num_harmonics
    pred_amp = torch.randn(B, C_half, T, device=device)
    pred_phase = torch.randn(B, C_half, T, device=device)
    true_amp = torch.randn(B, C_half, T, device=device).abs()
    true_phase = torch.randn(B, C_half, T, device=device)

    loss_fn = ComplexMSELoss()
    loss = loss_fn(pred_amp, pred_phase, true_amp, true_phase)
    print(f"  ComplexMSE: {loss.item():.4f}")

    spectral_loss_fn = SpectralReconstructionLoss(num_current_channels=36)
    loss_s = spectral_loss_fn(pred_amp, pred_phase, true_amp, true_phase, current_len=14)
    print(f"  SpectralRecon: {loss_s.item():.4f}")

    print("\n✓ Smoke test пройден успешно!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Physical KAN-Transformer')
    parser.add_argument('--mode', choices=['smoke', 'pretrain', 'finetune'],
                        default='smoke', help='Режим запуска')
    parser.add_argument('--epochs', type=int, default=None, help='Число эпох (override)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (override)')
    parser.add_argument('--model', choices=['PhysicalKANTransformer', 'BaselineTransformer'],
                        default='PhysicalKANTransformer', help='Тип модели')
    parser.add_argument('--d-model', type=int, default=None, help='d_model (override)')
    parser.add_argument('--resume', type=str, default=None, help='Путь к чекпоинту для продолжения')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config(args.mode)

    # Overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.d_model is not None:
        config['d_model'] = args.d_model
    config['model_type'] = args.model

    if args.mode == 'smoke':
        smoke_test(config)
    elif args.mode == 'pretrain':
        print("Pre-training пока не реализован — требуется подготовка датасета (Этап 1).")
        print("Используйте --mode smoke для проверки архитектуры.")
    elif args.mode == 'finetune':
        print("Fine-tuning будет доступен после pre-training.")


if __name__ == '__main__':
    main()
