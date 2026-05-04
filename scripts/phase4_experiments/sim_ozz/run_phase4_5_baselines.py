"""Запуск baseline-экспериментов для статьи по этапу 4.5.

Реализованные варианты:
1) spectral_baseline — обычный BaselineTransformer на тех же 220 спектральных
   каналах, без KAN и без физических stem-блоков.
2) physical_mlp — физически ориентированный Transformer без KAN-блоков:
   PhysicalStem с линейным fallback вместо KAN, ComplexMHA, MLP-FFN, MLP-head.
3) raw_instantaneous — BaselineTransformer на сырых мгновенных значениях (8 каналов
   × 32 отсчёта/период), ресэмплированных к 32 spp линейной интерполяцией.
   Спектральное представление НЕ вычисляется — модель получает сырой сигнал.
   Используется для оценки вклада именно spectral pipeline.

Ресэмплинг к 32 spp реализован через resample_to_spp (линейная интерполяция)
в osc_tools/ml/augmented_dataset.py и вызывается внутри Dataset «на лету».

Примеры:
    python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py --exp spectral_baseline
    python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py --exp physical_mlp
    python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py --exp raw_instantaneous
    python scripts/phase4_experiments/sim_ozz/run_phase4_5_baselines.py --exp all --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_experiments.run_phase4_finetune import COMPLEXITY_LEVELS
from scripts.phase4_experiments.sim_ozz.run_phase4_finetune_sim_ozz import (
    compute_num_channels,
    finetune_sim_ozz,
    get_sim_ozz_config,
)

EXPERIMENTS = {
    'spectral_baseline': {
        'model_type': 'BaselineTransformer',
        'cls_head_type': 'linear',
        'd_model': 56,  # Выровнено по params с PhysicalKAN (≈255K vs 263K)
        'description': 'Обычный Transformer на 220 spectral features, без KAN/physical stem.\n'
                       'd_model=56 для выравнивания числа параметров с PhysicalKAN.',
    },
    'physical_mlp': {
        'model_type': 'PhysicalMLPTransformer',
        'cls_head_type': 'mlp',
        # d_model НЕ переопределяется — используем те же гиперпараметры, что у KAN.
        # Разница в params (~183K vs 263K) = overhead KAN-блоков (ablation study).
        'description': 'Physical stem/complex attention, без KAN-блоков.\n'
                       'Те же d_model/num_layers — ablation вклада KAN.',
    },
    'raw_instantaneous': {
        'model_type': 'BaselineTransformer',
        'cls_head_type': 'linear',
        'd_model': 56,  # Выровнено по params (~232K vs 263K, ≈0.88 ratio — вход 8ch вместо 220)
        'use_raw_input': True,
        'description': 'BaselineTransformer на 8 сырых каналах (IA..UN), ресэмплированных к 32 spp.\n'
                       'Спектральный pipeline отключён — модель работает на мгновенных значениях.\n'
                       'num_input_channels=8, seq_len=320 (10 периодов × 32 spp), d_model=56.',
    },
}


def build_config(
    exp_name: str,
    complexity: str,
    epochs: int | None,
    batch_size: int | None,
    accumulation_steps: int | None,
    max_files: int | None,
    train_batches_per_epoch: int | None,
    val_batches_per_epoch: int | None,
) -> dict:
    """Формирует конфиг baseline-эксперимента без дублирования pipeline."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Неизвестный эксперимент: {exp_name}")

    config = get_sim_ozz_config()
    config.update(COMPLEXITY_LEVELS[complexity])
    config.update(EXPERIMENTS[exp_name])

    if epochs is not None:
        config['epochs'] = epochs
    if batch_size is not None:
        config['batch_size'] = batch_size
    if accumulation_steps is not None:
        config['accumulation_steps'] = accumulation_steps
    if max_files is not None:
        config['max_files'] = max_files
    if train_batches_per_epoch is not None:
        config['train_batches_per_epoch'] = train_batches_per_epoch
    if val_batches_per_epoch is not None:
        config['val_batches_per_epoch'] = val_batches_per_epoch

    num_lh = len(config['sub_periods'])
    if config.get('use_raw_input'):
        # Raw instantaneous: 8 каналов мгновенных значений, seq_len = window_size (320)
        config['num_input_channels'] = 8
        config['use_raw_input'] = True
    else:
        config['num_input_channels'] = compute_num_channels(
            config['num_harmonics'], num_lh, config['include_symmetric'],
        )
    return config


def run_experiment(exp_name: str, args: argparse.Namespace) -> Path:
    """Запускает один baseline-эксперимент."""
    config = build_config(
        exp_name=exp_name,
        complexity=args.complexity,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        max_files=args.max_files,
        train_batches_per_epoch=args.train_batches_per_epoch,
        val_batches_per_epoch=args.val_batches_per_epoch,
    )

    print("\n" + "=" * 80)
    print(f"Baseline experiment: {exp_name}")
    print(EXPERIMENTS[exp_name]['description'])
    print(f"model_type={config['model_type']}, cls_head_type={config['cls_head_type']}")
    print(f"complexity={args.complexity}, epochs={config['epochs']}")
    print("=" * 80)

    return finetune_sim_ozz(
        config,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4.5 baseline experiments')
    parser.add_argument('--exp', choices=['spectral_baseline', 'physical_mlp', 'raw_instantaneous', 'all'],
                        default='all')
    parser.add_argument('--complexity', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--max-files', type=int, default=None,
                        help='Ограничить число SimOZZ файлов для smoke/отладки')
    parser.add_argument('--train-batches-per-epoch', type=int, default=256)
    parser.add_argument('--val-batches-per-epoch', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset-optimizer', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(EXPERIMENTS.keys()) if args.exp == 'all' else [args.exp]

    results = []
    for name in names:
        save_dir = run_experiment(name, args)
        results.append((name, save_dir))

    print("\nЗапущенные/завершённые baseline-эксперименты:")
    for name, save_dir in results:
        print(f"  {name}: {save_dir}")


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================
        # РУЧНОЙ РЕЖИМ — отредактируйте константы ниже
        # =====================================================
        # Перечень baseline-экспериментов (см. EXPERIMENTS):
        #   'spectral_baseline'   — обычный Transformer на 220 spectral каналах
        #   'physical_mlp'        — physical stem + complex attention, без KAN
        #   'raw_instantaneous'   — BaselineTransformer на 8 сырых каналах (32 spp)
        #   'all'                 — все три последовательно
        EXP = 'all'
        COMPLEXITY = 'light'           # 'light' | 'medium' | 'heavy'
        EPOCHS = 50
        BATCH_SIZE = 32
        ACCUMULATION_STEPS = 8
        MAX_FILES = None               # None = все SimOZZ файлы; для smoke поставьте 100
        TRAIN_BATCHES_PER_EPOCH = 256
        VAL_BATCHES_PER_EPOCH = 16
        RESUME = None                  # путь к checkpoint для дообучения
        RESET_OPTIMIZER = False

        names = list(EXPERIMENTS.keys()) if EXP == 'all' else [EXP]

        class _Args:
            pass
        _args = _Args()
        _args.complexity = COMPLEXITY
        _args.epochs = EPOCHS
        _args.batch_size = BATCH_SIZE
        _args.accumulation_steps = ACCUMULATION_STEPS
        _args.max_files = MAX_FILES
        _args.train_batches_per_epoch = TRAIN_BATCHES_PER_EPOCH
        _args.val_batches_per_epoch = VAL_BATCHES_PER_EPOCH
        _args.resume = RESUME
        _args.reset_optimizer = RESET_OPTIMIZER

        results = []
        for _name in names:
            _save_dir = run_experiment(_name, _args)
            results.append((_name, _save_dir))

        print("\nЗапущенные/завершённые baseline-эксперименты:")
        for _name, _save_dir in results:
            print(f"  {_name}: {_save_dir}")
