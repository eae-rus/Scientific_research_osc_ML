"""Channel Dropout Probing — оценка важности групп входных каналов.

Зануляет группы из 220 спектральных каналов и измеряет
падение per-class F1 на val-выборке SimOZZ.

Сценарий: модель уже обучена, запускаем post-hoc анализ.

Запуск:
    python scripts/phase4_experiments/interpretability/channel_dropout_probing.py \
        --checkpoint experiments/phase4/.../best_model.pt \
        [--max-files 200] [--batch-size 64]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Пути проекта ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import (
    PhysicalKANTransformer, PhysicalMLPTransformer, BaselineTransformer,
)
from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex,
    SimOZZLazyDataset,
    TARGET_COLUMNS as SIM_TARGET_COLUMNS,
    ARC_TYPES,
)
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split

# ── Константы имён каналов (220 ch) ──
SIGNAL_NAMES = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
HARMONIC_LABELS = [f'h{i}' for i in range(1, 10)]  # h1..h9
SUB_PERIOD_LABELS = ['lh2', 'lh4', 'lh6', 'lh10']
ALL_HARMONICS = HARMONIC_LABELS + SUB_PERIOD_LABELS  # 13 гармонических компонент
SYMMETRIC_NAMES = ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']

CLASS_NAMES = ['Stable_SPGF', 'Petersen_AIGF', 'PetersSlepyan_AIGF', 'Belyakov_AIGF']


def build_feature_names() -> List[str]:
    """Формирует список имён 220 каналов."""
    names = []
    for sig in SIGNAL_NAMES:
        for harm in ALL_HARMONICS:
            names.append(f'{sig}_{harm}_mag')
            names.append(f'{sig}_{harm}_angle')
    for comp in SYMMETRIC_NAMES:
        names.append(f'{comp}_mag')
        names.append(f'{comp}_angle')
    return names


def build_channel_groups() -> Dict[str, List[int]]:
    """Группы каналов для зануления.

    Returns:
        dict[group_name → list[channel_indices]]
    """
    names = build_feature_names()
    n_phase_polar = len(SIGNAL_NAMES) * len(ALL_HARMONICS) * 2  # 208
    groups: Dict[str, List[int]] = {}

    # ── По сигналам ──
    for sig in SIGNAL_NAMES:
        idx = [i for i, n in enumerate(names) if n.startswith(f'{sig}_')]
        groups[f'signal_{sig}'] = idx

    # ── Нулевые составляющие (IN + UN) ──
    groups['all_neutral'] = groups['signal_IN'] + groups['signal_UN']

    # ── Фазные токи ──
    groups['phase_currents'] = (
        groups['signal_IA'] + groups['signal_IB'] + groups['signal_IC']
    )
    # ── Фазные напряжения ──
    groups['phase_voltages'] = (
        groups['signal_UA'] + groups['signal_UB'] + groups['signal_UC']
    )

    # ── По гармоникам ──
    for harm in ALL_HARMONICS:
        idx = [i for i, n in enumerate(names) if f'_{harm}_' in n]
        groups[f'harmonic_{harm}'] = idx

    # ── Только h1 (удаляем всё кроме h1) ──
    h1_idx = groups['harmonic_h1']
    groups['only_h1__drop_rest'] = [i for i in range(n_phase_polar) if i not in h1_idx]

    # ── Sub-periods ──
    sp_idx = []
    for lh in SUB_PERIOD_LABELS:
        sp_idx.extend(groups[f'harmonic_{lh}'])
    groups['all_sub_periods'] = sp_idx

    # ── Высшие гармоники h2..h9 ──
    higher = []
    for h in HARMONIC_LABELS[1:]:  # h2..h9
        higher.extend(groups[f'harmonic_{h}'])
    groups['higher_harmonics_h2_h9'] = higher

    # ── Только симметричные (удаляем phase_polar) ──
    groups['only_symmetric__drop_phase_polar'] = list(range(n_phase_polar))

    # ── Симметричные компоненты ──
    sym_idx = list(range(n_phase_polar, 220))
    groups['symmetric_components'] = sym_idx

    # ── Mag / angle ──
    groups['all_magnitudes'] = [i for i, n in enumerate(names) if '_mag' in n]
    groups['all_angles'] = [i for i, n in enumerate(names) if '_angle' in n]

    return groups


# ── Загрузка модели ──
def load_model(ckpt_path: Path, device: torch.device):
    """Загрузка модели из чекпоинта."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model_type = config.get('model_type', 'PhysicalKANTransformer')
    num_classes = config.get('num_classes', 4)
    zone_size = config.get('zone_size', 1)

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            kan_grid_size=config.get('kan_grid_size', 5),
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            cls_head_type=config.get('cls_head_type', 'kan'),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', 128),
        )
    elif model_type == 'PhysicalMLPTransformer':
        model = PhysicalMLPTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            kan_grid_size=config.get('kan_grid_size', 5),
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            cls_head_type=config.get('cls_head_type', 'mlp'),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', 128),
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            cls_head_type=config.get('cls_head_type', 'linear'),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', 128),
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


# ── Val dataset ──
def prepare_val_dataset(
    config: dict, max_files: int | None = None, seed: int = 42,
) -> SimOZZLazyDataset:
    """Создаёт val-датасет SimOZZ."""
    import random as _random

    data_dir = Path(config['data_dir'])
    file_index = SimOZZFileIndex.from_directory(data_dir, use_cache=True)

    file_names = [fi.path.name for fi in file_index.files]
    _, val_names, _, _ = stratified_sim_ozz_split(
        file_names,
        val_split=config.get('val_split', 0.2),
        seed=config.get('seed', 42),
    )
    val_name_set = set(val_names)
    val_files = [fi for fi in file_index.files if fi.path.name in val_name_set]

    if max_files and len(val_files) > max_files:
        by_class: Dict[int, list] = defaultdict(list)
        for fi in val_files:
            by_class[fi.meta['x']].append(fi)
        rng = _random.Random(seed)
        per_class = max(max_files // len(by_class), 1)
        selected = []
        for x_type in sorted(by_class.keys()):
            pool = by_class[x_type]
            selected.extend(rng.sample(pool, min(per_class, len(pool))))
        val_files = selected

    val_file_index = SimOZZFileIndex(val_files)
    val_ds = SimOZZLazyDataset(
        file_paths=[fi.path for fi in val_file_index.files],
        file_index=val_file_index,
        num_harmonics=config.get('num_harmonics', 9),
        sub_periods=config.get('sub_periods', [2, 4, 6, 10]),
        include_symmetric=config.get('include_symmetric', True),
        stride_fraction=config.get('stride_fraction', 8),
        num_periods_window=config.get('num_periods_window', 10),
        target_columns=SIM_TARGET_COLUMNS,
        zone_target_aggregation=config.get('zone_target_aggregation', 'max'),
        augmenter=None,
        cache_size=config.get('cache_size', 500),
    )
    return val_ds


# ── Метрики ──
def compute_per_class_f1(
    preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Per-class F1 + macro-F1.

    Args:
        preds: (N, C) вероятности
        targets: (N, C) бинарные метки

    Returns:
        (per_class_f1 array, macro_f1)
    """
    n_classes = preds.shape[1]
    f1s = np.zeros(n_classes)
    for c in range(n_classes):
        p = (preds[:, c] > threshold).astype(float)
        t = targets[:, c]
        tp = float((p * t).sum())
        fp = float((p * (1 - t)).sum())
        fn = float(((1 - p) * t).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1s, float(f1s.mean())


# ── Основной inference ──
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_channels: List[int] | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Прогоняет val данные через модель, опционально зануляя каналы.

    Args:
        mask_channels: список индексов каналов для зануления (из 220).
                       None = без маскирования.
        verbose: показывать прогресс по батчам.

    Returns:
        preds (N_zones, C), targets (N_zones, C)
    """
    all_preds, all_targets = [], []
    n_batches = len(loader)
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        # batch_x: (B, C=220, T=72), batch_y: (B, T=72, n_classes)
        batch_x = batch_x.to(device)
        if mask_channels:
            batch_x[:, mask_channels, :] = 0.0

        out = model(batch_x, mode='classify')
        logits = out['classify']  # (B, num_zones, num_classes)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        # Flatten zone dim
        B, T, C = probs.shape
        all_preds.append(probs.reshape(-1, C))
        all_targets.append(batch_y.numpy().reshape(-1, C))

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx + 1}/{n_batches}", end='\r')

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ── Главная логика ──
def run_probing(
    checkpoint_path: str,
    max_files: int | None = None,
    batch_size: int = 256,
    threshold: float = 0.5,
    num_workers: int = 8,
    output_dir: str | None = None,
) -> dict:
    """Запуск Channel Dropout Probing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1) Загрузка модели
    model, config = load_model(Path(checkpoint_path), device)
    print(f"Модель: {config.get('model_type')}, d_model={config.get('d_model')}")

    # 2) Val dataset
    val_ds = prepare_val_dataset(config, max_files=max_files)
    print(f"Val dataset: {len(val_ds)} осциллограмм")
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )

    # 3) Baseline (без маскирования)
    print("\n[Baseline] Inference без маскирования...")
    preds_base, targets = run_inference(model, loader, device,
                                        mask_channels=None, verbose=True)
    f1_base, macro_base = compute_per_class_f1(preds_base, targets, threshold)
    print(f"  Baseline Macro-F1: {macro_base:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: F1={f1_base[i]:.4f}")

    # 4) Прогон по группам
    groups = build_channel_groups()
    n_groups = len(groups)
    results = {
        'baseline': {
            'macro_f1': macro_base,
            'per_class_f1': {CLASS_NAMES[i]: float(f1_base[i]) for i in range(4)},
        },
        'groups': {},
    }

    print(f"\nПрогон {n_groups} групп каналов...")
    t_start = time.time()
    for g_idx, (group_name, ch_indices) in enumerate(sorted(groups.items())):
        t0 = time.time()
        preds_masked, _ = run_inference(model, loader, device, mask_channels=ch_indices)
        f1_masked, macro_masked = compute_per_class_f1(preds_masked, targets, threshold)
        dt = time.time() - t0

        delta_macro = macro_base - macro_masked
        delta_per_class = {
            CLASS_NAMES[i]: float(f1_base[i] - f1_masked[i]) for i in range(4)
        }

        results['groups'][group_name] = {
            'num_channels_masked': len(ch_indices),
            'macro_f1': float(macro_masked),
            'delta_macro_f1': float(delta_macro),
            'per_class_f1': {CLASS_NAMES[i]: float(f1_masked[i]) for i in range(4)},
            'delta_per_class_f1': delta_per_class,
        }

        elapsed = time.time() - t_start
        done = g_idx + 1
        eta = elapsed / done * (n_groups - done) if done > 0 else 0
        print(f"  [{done}/{n_groups}] {group_name:40s} ({len(ch_indices):3d} ch) "
              f"Macro-F1={macro_masked:.4f} (Δ={delta_macro:+.4f}) "
              f"[{dt:.1f}s] ETA {eta:.0f}s")

    # 5) Сортируем по важности (delta macro F1)
    ranked = sorted(
        results['groups'].items(),
        key=lambda x: x[1]['delta_macro_f1'],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("Ранжирование групп по важности (delta Macro-F1):")
    print("=" * 70)
    for rank, (name, info) in enumerate(ranked, 1):
        print(f"  {rank:2d}. {name:40s} Δ={info['delta_macro_f1']:+.4f} "
              f"({info['num_channels_masked']:3d} ch)")
    results['ranked'] = [name for name, _ in ranked]

    # 6) Сохранение
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / 'reports' / 'phase4' / 'interpretability')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / 'channel_dropout_probing.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены: {json_path}")

    # 7) Визуализация (если matplotlib доступен)
    try:
        _plot_results(results, out_path)
    except ImportError:
        print("matplotlib не найден — визуализация пропущена")

    return results


def _plot_results(results: dict, out_path: Path) -> None:
    """Строит bar-chart важности групп."""
    import matplotlib.pyplot as plt

    groups = results['groups']
    ranked_names = results['ranked']

    # ── Bar chart: delta macro F1 ──
    fig, ax = plt.subplots(figsize=(14, max(6, len(ranked_names) * 0.35)))
    deltas = [groups[n]['delta_macro_f1'] for n in ranked_names]
    colors = ['#d32f2f' if d > 0.01 else '#ff9800' if d > 0.005 else '#4caf50'
              for d in deltas]
    y_pos = range(len(ranked_names))
    ax.barh(y_pos, deltas, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked_names, fontsize=8)
    ax.set_xlabel('Δ Macro-F1 (больше = важнее)')
    ax.set_title('Channel Dropout Probing: важность групп каналов')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path / 'channel_dropout_importance.png', dpi=150)
    plt.close(fig)
    print(f"  График: {out_path / 'channel_dropout_importance.png'}")

    # ── Heatmap: per-class ──
    fig, ax = plt.subplots(figsize=(10, max(6, len(ranked_names) * 0.3)))
    matrix = np.array([
        [groups[n]['delta_per_class_f1'][c] for c in CLASS_NAMES]
        for n in ranked_names
    ])
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                   interpolation='nearest')
    ax.set_yticks(range(len(ranked_names)))
    ax.set_yticklabels(ranked_names, fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=30, ha='right')
    ax.set_title('Δ F1 per-class при зануление группы каналов')
    plt.colorbar(im, ax=ax, label='Δ F1')
    plt.tight_layout()
    fig.savefig(out_path / 'channel_dropout_heatmap.png', dpi=150)
    plt.close(fig)
    print(f"  Heatmap: {out_path / 'channel_dropout_heatmap.png'}")


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description='Channel Dropout Probing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту модели (.pt)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Макс. число val-файлов (для быстрого теста)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    run_probing(
        checkpoint_path=args.checkpoint,
        max_files=args.max_files,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================
        # РУЧНОЙ РЕЖИМ — отредактируйте константы ниже
        # =====================================================
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260428_174217/latest_checkpoint.pt'
        # Пример: CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_.../best_model.pt'

        MAX_FILES = 200         # 200 файлов, сбалансировано по классам
        BATCH_SIZE = 256
        THRESHOLD = 0.5

        if CHECKPOINT is None:
            # Автопоиск последнего sim_ozz эксперимента
            exp_root = PROJECT_ROOT / 'experiments' / 'phase4'
            sim_dirs = sorted(
                [d for d in exp_root.iterdir()
                 if d.is_dir() and 'sim_ozz' in d.name],
                key=lambda d: d.stat().st_mtime,
            ) if exp_root.exists() else []

            if sim_dirs:
                latest = sim_dirs[-1]
                best = latest / 'best_model.pt'
                if not best.exists():
                    best = latest / 'latest_checkpoint.pt'
                if best.exists():
                    CHECKPOINT = str(best)
                    print(f"Автоматически найден чекпоинт: {CHECKPOINT}")

        if CHECKPOINT is None:
            print("Нет чекпоинта. Укажите CHECKPOINT или передайте --checkpoint.")
        else:
            _ckpt = (str(PROJECT_ROOT / CHECKPOINT)
                     if not Path(CHECKPOINT).is_absolute() else CHECKPOINT)
            run_probing(
                checkpoint_path=_ckpt,
                max_files=MAX_FILES,
                batch_size=BATCH_SIZE,
                threshold=THRESHOLD,
            )
