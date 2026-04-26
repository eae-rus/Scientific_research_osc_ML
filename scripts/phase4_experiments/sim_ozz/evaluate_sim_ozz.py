"""
Оценка моделей Этапа 4.5 на симулированных данных SimOZZ.

File-by-file подход: для каждого val-файла загружаются ВСЕ окна,
прогоняется inference большими батчами (без градиентов -> больше влезает),
результаты накапливаются и метрики считаются по всему набору в конце.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex, SimOZZLazyDataset, TARGET_COLUMNS as SIM_TARGET_COLUMNS,
)
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from scripts.phase4_experiments.evaluate_phase4 import (
    compute_full_metrics,
    find_optimal_thresholds,
    compute_boundary_metrics,
    measure_inference_latency,
    print_report,
    load_model_from_checkpoint,
    PREDICTION_THRESHOLD,
)


@torch.no_grad()
def inference_file_by_file(
    model: nn.Module,
    dataset: SimOZZLazyDataset,
    device: torch.device,
    batch_size: int = 256,
    max_files: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """File-by-file inference: для каждого файла все окна сразу."""
    model.eval()

    file_to_indices: Dict[str, List[int]] = defaultdict(list)
    for global_idx, (fp, _ws) in enumerate(dataset._indices):
        file_to_indices[fp.name].append(global_idx)

    file_names = sorted(file_to_indices.keys())
    if max_files:
        file_names = file_names[:max_files]

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    per_file_stats: Dict[str, dict] = {}
    total_files = len(file_names)
    running_count = 0
    t0 = time.perf_counter()

    for fi_idx, fname in enumerate(file_names):
        indices = file_to_indices[fname]
        n_windows = len(indices)
        file_preds, file_targets = [], []

        for batch_start in range(0, n_windows, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            xs, ys = [], []
            for idx in batch_indices:
                x, y = dataset[idx]
                xs.append(x)
                ys.append(y)

            X = torch.stack(xs).to(device)
            Y = torch.stack(ys)
            out = model(X, mode='classify')
            logits = out['classify']
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            file_preds.append(probs)
            file_targets.append(Y.numpy())

        file_preds_arr = np.concatenate(file_preds, axis=0)
        file_targets_arr = np.concatenate(file_targets, axis=0)
        per_file_stats[fname] = {
            'n_windows': n_windows,
            'idx_start': running_count,
            'idx_end': running_count + n_windows,
        }
        all_preds.append(file_preds_arr)
        all_targets.append(file_targets_arr)
        running_count += n_windows

        if (fi_idx + 1) % 20 == 0 or fi_idx == 0 or fi_idx == total_files - 1:
            elapsed_so_far = time.perf_counter() - t0
            print(f"  [{fi_idx+1}/{total_files}] {fname}: "
                  f"{n_windows} окон, всего={running_count}, "
                  f"time={elapsed_so_far:.1f}s")

    elapsed = time.perf_counter() - t0
    return (np.concatenate(all_preds, axis=0),
            np.concatenate(all_targets, axis=0),
            elapsed, per_file_stats)


def prepare_val_dataset(
    config: dict, max_files: int | None = None,
) -> Tuple[SimOZZLazyDataset, List[str]]:
    """Создаёт val SimOZZLazyDataset (все окна, без random sampling)."""
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
    val_file_index = SimOZZFileIndex(val_files)

    if max_files:
        val_files = val_file_index.files[:max_files]
        val_file_index = SimOZZFileIndex(val_files)

    print(f"Val файлов: {len(val_file_index)}")

    target_columns = SIM_TARGET_COLUMNS
    val_file_paths = [fi.path for fi in val_file_index.files]
    val_ds = SimOZZLazyDataset(
        file_paths=val_file_paths,
        file_index=val_file_index,
        num_harmonics=config.get('num_harmonics', 9),
        sub_periods=config.get('sub_periods', [2, 4, 6, 10]),
        include_symmetric=config.get('include_symmetric', True),
        stride_fraction=config.get('stride_fraction', 8),
        num_periods_window=config.get('num_periods_window', 10),
        target_columns=target_columns,
        zone_target_aggregation=config.get('zone_target_aggregation', 'max'),
        augmenter=None,
        cache_size=config.get('cache_size', 500),
    )
    return val_ds, target_columns


# --- Визуализация ---

def plot_confusion_matrices(preds_bin, targets, target_columns, save_path):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    n_classes = len(target_columns)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5))
    if n_classes == 1:
        axes = [axes]
    for i, col in enumerate(target_columns):
        cm = confusion_matrix(targets[:, i], preds_bin[:, i], labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Нет', 'Есть'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(col.replace('Target_OZZ_', ''), fontsize=11)
        axes[i].set_xlabel('Предсказание')
        axes[i].set_ylabel('Истина' if i == 0 else '')
    fig.suptitle('Confusion Matrix (per-class)', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Confusion matrix: {save_path}")


def plot_roc_curves(preds, targets, target_columns, save_path):
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E74C3C', '#F39C12', '#8E44AD', '#2980B9']
    for i, col in enumerate(target_columns):
        if len(np.unique(targets[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(targets[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{col.replace("Target_OZZ_", "")} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC-кривые'); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  ROC-кривые: {save_path}")


def plot_training_curves(exp_dir, save_path):
    log_path = exp_dir / 'training_log.jsonl'
    if not log_path.exists():
        csv_path = exp_dir / 'training_curves.csv'
        if not csv_path.exists():
            print("  Кривые обучения: файл не найден"); return
        import pandas as pd
        df = pd.read_csv(csv_path)
    else:
        import pandas as pd
        records = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return
        df = pd.DataFrame(records)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if 'train_loss' in df.columns:
        ax.plot(df['epoch'], df['train_loss'], label='Train', color='#2980B9', lw=2)
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Val', color='#E74C3C', lw=2)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('Loss'); ax.set_title('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    for col in [c for c in df.columns if 'f1' in c.lower()]:
        ax.plot(df['epoch'], df[col], label=col, lw=2)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('F1'); ax.set_title('F1-score')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Кривые обучения - {exp_dir.name}', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Кривые обучения: {save_path}")


def plot_probability_distributions(preds, targets, target_columns, save_path):
    n_classes = len(target_columns)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    for i, col in enumerate(target_columns):
        ax = axes[i]
        pos = targets[:, i] == 1; neg = ~pos
        if pos.sum() > 0:
            ax.hist(preds[pos, i], bins=50, alpha=0.6,
                    color='#E74C3C', label=f'Есть ({pos.sum()})', density=True)
        if neg.sum() > 0:
            ax.hist(preds[neg, i], bins=50, alpha=0.6,
                    color='#2980B9', label=f'Нет ({neg.sum()})', density=True)
        ax.axvline(0.5, color='black', ls='--', lw=1, alpha=0.7)
        ax.set_title(col.replace('Target_OZZ_', ''))
        ax.set_xlabel('P(class)'); ax.set_ylabel('Плотность')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Распределение вероятностей', fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Распределение P: {save_path}")


# --- Основная функция ---

def evaluate_sim_ozz(
    checkpoint_path: str,
    max_files: int | None = None,
    batch_size: int = 256,
    save_plots: bool = True,
) -> dict:
    """Полная оценка SimOZZ-модели: file-by-file inference + метрики."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")
    exp_dir = ckpt_path.parent

    model, config = load_model_from_checkpoint(ckpt_path, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: {config.get('model_type')}, параметров: {num_params:,}")

    val_ds, target_columns = prepare_val_dataset(config, max_files=max_files)
    print(f"Всего val окон: {len(val_ds):,}")

    print(f"\nFile-by-file inference (batch_size={batch_size})...")
    preds_3d, targets_3d, elapsed, per_file_stats = inference_file_by_file(
        model, val_ds, device, batch_size=batch_size, max_files=max_files,
    )
    print(f"\n  Итого: {preds_3d.shape[0]:,} окон за {elapsed:.1f}с "
          f"({preds_3d.shape[0]/max(elapsed,0.01):.0f} окон/с)")

    preds_window = preds_3d.mean(axis=1)
    targets_window = targets_3d.max(axis=1)
    B, T, C = preds_3d.shape
    preds_zone = preds_3d.reshape(-1, C)
    targets_zone = targets_3d.reshape(-1, C)

    print("\n=== WINDOW-LEVEL МЕТРИКИ ===")
    metrics = compute_full_metrics(preds_window, targets_window, target_columns)
    opt = find_optimal_thresholds(preds_window, targets_window, target_columns)
    metrics['optimal_thresholds'] = opt

    model_info = {
        'checkpoint': str(ckpt_path),
        'model_type': config.get('model_type', 'N/A'),
        'num_params': num_params,
        'd_model': config.get('d_model'),
        'num_layers': config.get('num_layers'),
        'num_input_channels': config.get('num_input_channels'),
        'inference_time': elapsed,
        'num_windows': int(preds_window.shape[0]),
        'num_files': len(per_file_stats),
    }
    print_report(metrics, target_columns, model_info)

    print("\n--- Оптимальные пороги ---")
    print(f"  {'Класс':<30s} {'Порог':>8s} {'F1_opt':>8s} {'F1_0.5':>8s}")
    print("  " + "-" * 60)
    for col in target_columns:
        o = opt['per_class'].get(col, {})
        s = metrics['per_class'].get(col, {})
        print(f"  {col:<30s} {o.get('threshold', 0.5):8.3f} "
              f"{o.get('f1', 0):8.4f} {s.get('f1', 0):8.4f}")
    print(f"\n  Macro-F1 (0.5):  {metrics['macro_f1']:.4f}")
    print(f"  Macro-F1 (opt):  {opt['macro_f1_optimal']:.4f}")

    print("\n=== ZONE-LEVEL МЕТРИКИ ===")
    zone_metrics = compute_full_metrics(preds_zone, targets_zone, target_columns)
    print(f"  Zone Macro-F1: {zone_metrics['macro_f1']:.4f}")

    boundary = compute_boundary_metrics(
        preds_zone, targets_zone, target_columns,
        threshold=PREDICTION_THRESHOLD,
        stride_samples=config.get('stride_fraction', 8),
    )
    metrics['boundary'] = boundary
    metrics['zone_level'] = {
        'macro_f1': zone_metrics['macro_f1'],
        'exact_match': zone_metrics['exact_match'],
    }
    print(f"\n  {'Класс':<30s} {'Events':>7s} {'Delay':>7s} "
          f"{'Smear_on':>9s} {'Smear_off':>10s} {'FP':>6s}")
    print("  " + "-" * 75)
    for col in target_columns:
        bc = boundary['per_class'].get(col, {})
        print(f"  {col:<30s} {bc.get('num_events', 0):7d} "
              f"{bc.get('mean_delay_zones', 0):7.2f} "
              f"{bc.get('mean_smearing_onset', 0):9.2f} "
              f"{bc.get('mean_smearing_offset', 0):10.2f} "
              f"{bc.get('false_alarm_zones', 0):6d}")

    per_file_correct = 0
    per_file_total = len(per_file_stats)
    for fstats in per_file_stats.values():
        s, e = fstats['idx_start'], fstats['idx_end']
        pred_any = (preds_window[s:e].mean(axis=0) >= PREDICTION_THRESHOLD).astype(int)
        target_any = (targets_window[s:e].max(axis=0) >= 0.5).astype(int)
        if np.array_equal(pred_any, target_any):
            per_file_correct += 1
    file_accuracy = per_file_correct / per_file_total if per_file_total else 0.0
    print(f"\n  File-level accuracy: {per_file_correct}/{per_file_total} = {file_accuracy:.3f}")
    metrics['file_level_accuracy'] = file_accuracy

    n_zones = (config.get('num_periods_window', 10) - 1) * config.get('stride_fraction', 8)
    latency_ms = measure_inference_latency(
        model, config['num_input_channels'], n_zones, device)
    print(f"  Latency: {latency_ms:.2f} мс/sample")
    metrics['latency_ms'] = latency_ms

    if save_plots:
        plot_dir = exp_dir / 'eval_plots'
        plot_dir.mkdir(exist_ok=True)
        print(f"\nГрафики -> {plot_dir}")
        preds_bin = (preds_window >= PREDICTION_THRESHOLD).astype(np.int32)
        targets_int = targets_window.astype(np.int32)
        plot_confusion_matrices(preds_bin, targets_int, target_columns,
                                plot_dir / 'confusion_matrix.png')
        plot_roc_curves(preds_window, targets_int, target_columns,
                        plot_dir / 'roc_curves.png')
        plot_probability_distributions(preds_window, targets_int, target_columns,
                                       plot_dir / 'prob_distributions.png')
        plot_training_curves(exp_dir, plot_dir / 'training_curves.png')

    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'metrics_window_level': metrics,
        'metrics_zone_level': {'macro_f1': zone_metrics['macro_f1'],
                               'exact_match': zone_metrics['exact_match']},
        'boundary': boundary,
        'optimal_thresholds': opt,
        'per_file_summary': {'total': per_file_total, 'accuracy': file_accuracy},
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
    }
    report_path = exp_dir / 'sim_ozz_evaluation.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nОтчёт: {report_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Оценка SimOZZ (file-by-file)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    evaluate_sim_ozz(
        checkpoint_path=args.checkpoint,
        max_files=args.max_files,
        batch_size=args.batch_size,
        save_plots=not args.no_plots,
    )


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) > 1:
        main()
    else:
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260425_225849/latest_checkpoint.pt'
        if CHECKPOINT is None:
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
        if CHECKPOINT is None:
            print("Нет чекпоинта.")
        else:
            _ckpt = (str(PROJECT_ROOT / CHECKPOINT)
                     if not Path(CHECKPOINT).is_absolute() else CHECKPOINT)
            evaluate_sim_ozz(
                checkpoint_path=_ckpt,
                max_files=None,
                batch_size=4048,
                save_plots=True,
            )
