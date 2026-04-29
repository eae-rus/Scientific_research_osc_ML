"""Gradient-based Attribution — saliency maps для каждого класса ОЗЗ.

Вычисляет Gradient×Input для каждого класса:
    attribution_c = ∇_x ŷ_c · x

Для каждого примера: какие из 220 каналов в каких зонах сильнее
всего влияют на предсказание класса c.

Запуск:
    python scripts/phase4_experiments/interpretability/gradient_attribution.py \
        --checkpoint experiments/phase4/.../best_model.pt \
        [--max-files 100] [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Пути проекта ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_experiments.interpretability.channel_dropout_probing import (
    load_model,
    prepare_val_dataset,
    build_feature_names,
    SIGNAL_NAMES,
    ALL_HARMONICS,
    SYMMETRIC_NAMES,
    CLASS_NAMES,
    SIM_TARGET_COLUMNS,
)


def compute_gradient_attribution(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> Dict[str, np.ndarray]:
    """Вычисляет Gradient×Input attribution per-class.

    Returns:
        dict с ключами:
          'attribution_per_class': (num_classes, 220) — усреднённый |grad×input| по OZZ-зонам
          'attribution_temporal': (num_classes, 220, 72) — полная temporal карта
          'counts_per_class': (num_classes,) — число зон с этим классом
    """
    C_in = 220
    T_zones = 72

    # Аккумуляторы: суммируем |grad × input| по зонам, где target[c]=1
    sum_attr = np.zeros((num_classes, C_in), dtype=np.float64)
    sum_attr_temporal = np.zeros((num_classes, C_in, T_zones), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    model.eval()

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        # batch_x: (B, C=220, T=72), batch_y: (B, T=72, num_classes)
        batch_x = batch_x.to(device).requires_grad_(True)

        out = model(batch_x, mode='classify')
        logits = out['classify']  # (B, T, num_classes)

        B, T, _ = logits.shape

        for c in range(num_classes):
            # Считаем градиент суммарного logit класса c по всем зонам
            model.zero_grad()
            if batch_x.grad is not None:
                batch_x.grad.zero_()

            score = logits[:, :, c].sum()
            score.backward(retain_graph=(c < num_classes - 1))

            grad = batch_x.grad.detach().cpu().numpy()  # (B, 220, 72)
            x_val = batch_x.detach().cpu().numpy()       # (B, 220, 72)
            attr = np.abs(grad * x_val)                   # |grad × input|

            y_cls = batch_y[:, :, c].numpy()  # (B, T)

            # Per-sample, per-zone — суммируем только где target=1
            for b in range(B):
                mask = y_cls[b] > 0.5  # (T,)
                n_pos = mask.sum()
                if n_pos == 0:
                    continue
                # По каналам (усреднение по зонам)
                sum_attr[c] += attr[b, :, mask].sum(axis=1)  # (220,)
                # Temporal — усредняем per-zone
                sum_attr_temporal[c, :, mask] += attr[b, :, mask]
                counts[c] += n_pos

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    # Нормализация
    for c in range(num_classes):
        if counts[c] > 0:
            sum_attr[c] /= counts[c]
            sum_attr_temporal[c] /= np.maximum(counts[c] / T_zones, 1)

    return {
        'attribution_per_class': sum_attr,          # (4, 220)
        'attribution_temporal': sum_attr_temporal,    # (4, 220, 72)
        'counts_per_class': counts,                   # (4,)
    }


def aggregate_by_signal(
    attr: np.ndarray,
) -> np.ndarray:
    """Группирует 220 каналов по 8 сигналам → (num_classes, 8).

    Суммирует attribution всех гармоник каждого сигнала.
    """
    n_classes = attr.shape[0]
    result = np.zeros((n_classes, 8), dtype=np.float64)
    ch_per_signal = len(ALL_HARMONICS) * 2  # 26

    for s_idx in range(8):
        start = s_idx * ch_per_signal
        end = start + ch_per_signal
        result[:, s_idx] = attr[:, start:end].sum(axis=1)

    return result


def run_gradient_attribution(
    checkpoint_path: str,
    max_files: int | None = None,
    batch_size: int = 32,
    output_dir: str | None = None,
) -> dict:
    """Запуск Gradient Attribution."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1) Модель
    model, config = load_model(Path(checkpoint_path), device)
    print(f"Модель: {config.get('model_type')}, d_model={config.get('d_model')}")

    # 2) Val dataset
    val_ds = prepare_val_dataset(config, max_files=max_files)
    print(f"Val dataset: {len(val_ds)} осциллограмм")
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # 3) Gradient Attribution
    print("\nВычисление Gradient×Input Attribution...")
    ga = compute_gradient_attribution(model, loader, device)

    attr = ga['attribution_per_class']  # (4, 220)
    attr_t = ga['attribution_temporal']  # (4, 220, 72)
    counts = ga['counts_per_class']

    print(f"\nЗон per-class: {dict(zip(CLASS_NAMES, counts.tolist()))}")

    # 4) ТОП-20 каналов per-class
    feature_names = build_feature_names()
    top_k = 20

    results_json = {
        'counts': {CLASS_NAMES[c]: int(counts[c]) for c in range(4)},
        'top_channels': {},
        'signal_importance': {},
    }

    print(f"\n{'='*70}")
    print(f"ТОП-{top_k} каналов per-class (по средней |grad × input|):")
    print(f"{'='*70}")
    for c in range(4):
        ranked = np.argsort(attr[c])[::-1][:top_k]
        print(f"\n  [{CLASS_NAMES[c]}]:")
        top_list = []
        for rank, ch_idx in enumerate(ranked, 1):
            name = feature_names[ch_idx] if ch_idx < len(feature_names) else f'ch_{ch_idx}'
            val = float(attr[c, ch_idx])
            print(f"    {rank:2d}. {name:30s} = {val:.6f}")
            top_list.append({'channel': name, 'index': int(ch_idx), 'value': val})
        results_json['top_channels'][CLASS_NAMES[c]] = top_list

    # 5) Группировка по сигналам
    sig_attr = aggregate_by_signal(attr)  # (4, 8)
    print(f"\n{'='*70}")
    print("Важность по сигналам (сумма attribution):")
    print(f"{'='*70}")
    for c in range(4):
        ranked_sig = np.argsort(sig_attr[c])[::-1]
        sig_str = ', '.join(
            f"{SIGNAL_NAMES[s]}={sig_attr[c, s]:.4f}" for s in ranked_sig
        )
        print(f"  [{CLASS_NAMES[c]}]: {sig_str}")
        results_json['signal_importance'][CLASS_NAMES[c]] = {
            SIGNAL_NAMES[s]: float(sig_attr[c, s]) for s in range(8)
        }

    # 6) Сохранение
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / 'reports' / 'phase4' / 'interpretability')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / 'gradient_attribution.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    # Numpy arrays
    np.savez_compressed(
        out_path / 'gradient_attribution_arrays.npz',
        attribution_per_class=attr.astype(np.float32),
        attribution_temporal=attr_t.astype(np.float32),
        counts_per_class=counts,
    )
    print(f"NPZ: {out_path / 'gradient_attribution_arrays.npz'}")

    # 7) Визуализация
    try:
        _plot_results(attr, sig_attr, attr_t, out_path)
    except ImportError:
        print("matplotlib не найден — визуализация пропущена")

    return results_json


def _plot_results(
    attr: np.ndarray,        # (4, 220)
    sig_attr: np.ndarray,    # (4, 8)
    attr_t: np.ndarray,      # (4, 220, 72)
    out_path: Path,
) -> None:
    """Визуализация результатов."""
    import matplotlib.pyplot as plt

    # ── 1) Heatmap по сигналам ──
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(sig_attr, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xticks(range(8))
    ax.set_xticklabels(SIGNAL_NAMES, fontsize=10)
    ax.set_title('Gradient Attribution: важность сигналов per-class')
    plt.colorbar(im, ax=ax, label='Mean |grad × input|')
    for i in range(4):
        for j in range(8):
            ax.text(j, i, f'{sig_attr[i, j]:.3f}', ha='center', va='center',
                    fontsize=7, color='white' if sig_attr[i, j] > sig_attr.max() * 0.6 else 'black')
    plt.tight_layout()
    fig.savefig(out_path / 'gradient_signal_heatmap.png', dpi=150)
    plt.close(fig)
    print(f"  Heatmap сигналов: {out_path / 'gradient_signal_heatmap.png'}")

    # ── 2) Temporal heatmap per-class (TOP-30 каналов) ──
    feature_names = build_feature_names()
    top_n = 30
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for c, ax in enumerate(axes.flat):
        top_idx = np.argsort(attr[c])[::-1][:top_n]
        data = attr_t[c, top_idx, :]  # (top_n, 72)
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_title(CLASS_NAMES[c], fontsize=11)
        ax.set_xlabel('Зона (0-71)')
        labels = [feature_names[i] if i < len(feature_names) else f'{i}'
                  for i in top_idx]
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels, fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Temporal Attribution: ТОП-30 каналов per-class', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path / 'gradient_temporal_heatmap.png', dpi=150)
    plt.close(fig)
    print(f"  Temporal heatmap: {out_path / 'gradient_temporal_heatmap.png'}")


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description='Gradient Attribution')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    run_gradient_attribution(
        checkpoint_path=args.checkpoint,
        max_files=args.max_files,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
