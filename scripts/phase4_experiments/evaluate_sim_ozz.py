"""
Оценка моделей Этапа 4.5 на симулированных данных SimOZZ.

Загружает fine-tuned чекпоинт, прогоняет inference на val-сплите SimOZZ,
вычисляет метрики и строит диаграммы (confusion matrix, ROC, кривые обучения).

Переиспользует функции метрик из evaluate_phase4.py.

Примеры:
  python scripts/phase4_experiments/evaluate_sim_ozz.py --checkpoint experiments/phase4/sim_ozz_.../best_model.pt
  python scripts/phase4_experiments/evaluate_sim_ozz.py --checkpoint experiments/phase4/sim_ozz_.../best_model.pt --plot-marking 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex, SimOZZLazyDataset, TARGET_COLUMNS as SIM_TARGET_COLUMNS,
)
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split

# Переиспользуем метрики из evaluate_phase4
from scripts.phase4_experiments.evaluate_phase4 import (
    compute_full_metrics,
    find_optimal_thresholds,
    compute_boundary_metrics,
    measure_inference_latency,
    print_report,
    load_model_from_checkpoint,
    PREDICTION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Inference на SimOZZ (lazy dataset)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sim_ozz_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inference на SimOZZ DataLoader.

    Т.к. val lazy DataLoader может быть огромным, поддерживаем ограничение
    max_batches для быстрой оценки.

    Returns:
        (preds, targets, elapsed_sec)
        preds: (N, num_zones, num_classes) — zone-level вероятности
        targets: (N, num_zones, num_classes) — zone-level метки
    """
    model.eval()
    all_preds, all_targets = [], []
    t0 = time.perf_counter()
    total_batches = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        out = model(x, mode='classify')
        logits = out['classify']  # (B, num_zones, num_classes)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_preds.append(probs)

        y_np = y.numpy()
        all_targets.append(y_np)

        # Прогресс
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            elapsed_so_far = time.perf_counter() - t0
            samples_done = sum(p.shape[0] for p in all_preds)
            print(f"  batch {batch_idx+1}/{total_batches}, "
                  f"samples={samples_done}, time={elapsed_so_far:.1f}s")

    elapsed = time.perf_counter() - t0
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets, elapsed


# ---------------------------------------------------------------------------
# Подготовка DataLoader для валидации
# ---------------------------------------------------------------------------

def prepare_sim_ozz_val_loader(
    config: dict,
    max_files: int | None = None,
    val_batch_size: int = 64,
    val_max_samples: int = 5000,
) -> tuple[DataLoader, list[str]]:
    """Создаёт val DataLoader для SimOZZ (lazy).

    Args:
        config: конфиг из чекпоинта
        max_files: ограничение файлов (для отладки)
        val_batch_size: размер батча
        val_max_samples: максимум элементов для оценки (lazy → нельзя всё)

    Returns:
        (val_loader, target_columns)
    """
    data_dir = Path(config['data_dir'])

    # Загрузка/кеширование индекса файлов
    file_index = SimOZZFileIndex.from_directory(data_dir, use_cache=True)

    if max_files:
        file_index = SimOZZFileIndex(file_index.files[:max_files])

    # Стратифицированный split (тот же seed → те же файлы)
    file_names = [fi.path.name for fi in file_index.files]
    train_names, val_names, _, _ = stratified_sim_ozz_split(
        file_names,
        val_split=config.get('val_split', 0.2),
        seed=config.get('seed', 42),
    )
    val_name_set = set(val_names)
    val_files = [fi for fi in file_index.files if fi.path.name in val_name_set]
    val_file_index = SimOZZFileIndex(val_files)

    print(f"Val файлов: {len(val_file_index)}")

    target_columns = SIM_TARGET_COLUMNS

    # Создаём lazy dataset
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
        augmenter=None,  # Без аугментации для eval
        cache_size=config.get('cache_size', 500),
    )

    # Ограничиваем выборку (lazy = нельзя прогнать всё)
    n_total = len(val_ds)
    n_samples = min(n_total, val_max_samples)
    rng = np.random.RandomState(config.get('seed', 42))
    indices = rng.choice(n_total, size=n_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Val элементов: {n_samples} из {n_total}")

    return val_loader, target_columns


# ---------------------------------------------------------------------------
# Визуализация: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    preds_bin: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    save_path: Path,
) -> None:
    """Строит confusion matrix для каждого класса (2×2) и общую multi-label."""
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
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {save_path}")


# ---------------------------------------------------------------------------
# Визуализация: ROC-кривые
# ---------------------------------------------------------------------------

def plot_roc_curves(
    preds: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    save_path: Path,
) -> None:
    """Строит ROC-кривые для каждого класса."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E74C3C', '#F39C12', '#8E44AD', '#2980B9']

    for i, col in enumerate(target_columns):
        if len(np.unique(targets[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(targets[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        label = col.replace('Target_OZZ_', '')
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривые (per-class)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ROC-кривые: {save_path}")


# ---------------------------------------------------------------------------
# Визуализация: кривые обучения из training_log.jsonl
# ---------------------------------------------------------------------------

def plot_training_curves(exp_dir: Path, save_path: Path) -> None:
    """Строит кривые обучения из training_log.jsonl (loss, F1).

    Если файл отсутствует — пропускает.
    """
    log_path = exp_dir / 'training_log.jsonl'
    if not log_path.exists():
        # Попробуем training_curves.csv
        csv_path = exp_dir / 'training_curves.csv'
        if not csv_path.exists():
            print(f"  Кривые обучения: файл не найден ({log_path})")
            return
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
            print(f"  Кривые обучения: пустой лог")
            return
        df = pd.DataFrame(records)

    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    if 'train_loss' in df.columns:
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#2980B9', lw=2)
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#E74C3C', lw=2)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- F1 ---
    ax = axes[1]
    f1_cols = [c for c in df.columns if 'f1' in c.lower()]
    for col in f1_cols:
        ax.plot(df['epoch'], df[col], label=col, lw=2)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('F1')
    ax.set_title('F1-score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Кривые обучения — {exp_dir.name}', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Кривые обучения: {save_path}")


# ---------------------------------------------------------------------------
# Визуализация: распределение вероятностей
# ---------------------------------------------------------------------------

def plot_probability_distributions(
    preds: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    save_path: Path,
) -> None:
    """Гистограммы предсказанных вероятностей (раздельно для pos/neg)."""
    n_classes = len(target_columns)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    colors_pos = '#E74C3C'
    colors_neg = '#2980B9'

    for i, col in enumerate(target_columns):
        ax = axes[i]
        pos_mask = targets[:, i] == 1
        neg_mask = ~pos_mask

        if pos_mask.sum() > 0:
            ax.hist(preds[pos_mask, i], bins=50, alpha=0.6,
                    color=colors_pos, label=f'Есть ({pos_mask.sum()})', density=True)
        if neg_mask.sum() > 0:
            ax.hist(preds[neg_mask, i], bins=50, alpha=0.6,
                    color=colors_neg, label=f'Нет ({neg_mask.sum()})', density=True)

        ax.axvline(0.5, color='black', linestyle='--', lw=1, alpha=0.7)
        ax.set_title(col.replace('Target_OZZ_', ''))
        ax.set_xlabel('P(class)')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Распределение вероятностей (positive vs negative)', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Распределение P: {save_path}")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def evaluate_sim_ozz(
    checkpoint_path: str,
    max_files: int | None = None,
    val_max_samples: int = 5000,
    save_plots: bool = True,
) -> dict:
    """Полная оценка SimOZZ-модели: метрики + визуализация."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    exp_dir = ckpt_path.parent

    # --- Загрузка модели ---
    model, config = load_model_from_checkpoint(ckpt_path, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: {config.get('model_type')}, параметров: {num_params:,}")

    # --- Данные ---
    val_loader, target_columns = prepare_sim_ozz_val_loader(
        config,
        max_files=max_files,
        val_batch_size=config.get('val_batch_size', 64),
        val_max_samples=val_max_samples,
    )

    # --- Inference (zone-level) ---
    print("\nInference...")
    preds_3d, targets_3d, elapsed = run_sim_ozz_inference(model, val_loader, device)
    # preds_3d: (N, num_zones, C), targets_3d: (N, num_zones, C)
    print(f"  Обработано: {preds_3d.shape[0]} элементов за {elapsed:.2f} с")

    # Window-level: среднее по зонам / max по зонам
    preds_window = preds_3d.mean(axis=1)   # (N, C) — window-level вероятности
    targets_window = targets_3d.max(axis=1)  # (N, C) — window-level метки

    # Zone-level: flatten
    B, T, C = preds_3d.shape
    preds_zone = preds_3d.reshape(-1, C)
    targets_zone = targets_3d.reshape(-1, C)

    # --- Метрики (window-level) ---
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
        'num_samples': preds_window.shape[0],
    }
    print_report(metrics, target_columns, model_info)

    # Оптимальные пороги
    print("\n--- Оптимальные пороги per-class ---")
    print(f"  {'Класс':<30s} {'Порог':>8s} {'F1_opt':>8s} {'F1_0.5':>8s}")
    print("  " + "-" * 60)
    for col in target_columns:
        o = opt['per_class'].get(col, {})
        s = metrics['per_class'].get(col, {})
        print(
            f"  {col:<30s} "
            f"{o.get('threshold', 0.5):8.3f} "
            f"{o.get('f1', 0):8.4f} "
            f"{s.get('f1', 0):8.4f}"
        )
    print(f"\n  Macro-F1 (0.5):  {metrics['macro_f1']:.4f}")
    print(f"  Macro-F1 (opt):  {opt['macro_f1_optimal']:.4f}")

    # --- Zone-level: Boundary метрики ---
    print("\n=== ZONE-LEVEL МЕТРИКИ ===")
    zone_metrics = compute_full_metrics(preds_zone, targets_zone, target_columns)
    print(f"  Zone Macro-F1: {zone_metrics['macro_f1']:.4f}")

    # Boundary (smearing/delay) — на file-level (используем 3D predictions)
    print("\n--- Метрики границ (per-file zone-level) ---")
    stride_samples = config.get('stride_fraction', 8)  # зоны в единицах stride
    boundary = compute_boundary_metrics(
        preds_zone, targets_zone, target_columns,
        threshold=PREDICTION_THRESHOLD, stride_samples=stride_samples,
    )
    metrics['boundary'] = boundary
    metrics['zone_level'] = {
        'macro_f1': zone_metrics['macro_f1'],
        'exact_match': zone_metrics['exact_match'],
    }

    print(f"  {'Класс':<30s} {'Events':>7s} {'Delay':>7s} {'Smear_on':>9s} {'Smear_off':>10s} {'FP':>6s}")
    print("  " + "-" * 75)
    for col in target_columns:
        bc = boundary['per_class'].get(col, {})
        print(
            f"  {col:<30s} "
            f"{bc.get('num_events', 0):7d} "
            f"{bc.get('mean_delay_zones', 0):7.2f} "
            f"{bc.get('mean_smearing_onset', 0):9.2f} "
            f"{bc.get('mean_smearing_offset', 0):10.2f} "
            f"{bc.get('false_alarm_zones', 0):6d}"
        )

    # --- Latency ---
    num_channels = config['num_input_channels']
    # Число зон для одного окна SimOZZ
    n_zones_per_window = (config.get('num_periods_window', 10) - 1) * config.get('stride_fraction', 8)
    latency_ms = measure_inference_latency(model, num_channels, n_zones_per_window, device)
    print(f"\n  Latency: {latency_ms:.2f} мс/sample")
    metrics['latency_ms'] = latency_ms

    # --- Визуализация ---
    if save_plots:
        plot_dir = exp_dir / 'eval_plots'
        plot_dir.mkdir(exist_ok=True)
        print(f"\nГрафики → {plot_dir}")

        # Confusion matrices
        preds_bin = (preds_window >= PREDICTION_THRESHOLD).astype(np.int32)
        targets_int = targets_window.astype(np.int32)
        plot_confusion_matrices(preds_bin, targets_int, target_columns,
                                plot_dir / 'confusion_matrix.png')

        # ROC-кривые
        plot_roc_curves(preds_window, targets_int, target_columns,
                        plot_dir / 'roc_curves.png')

        # Распределение вероятностей
        plot_probability_distributions(preds_window, targets_int, target_columns,
                                       plot_dir / 'prob_distributions.png')

        # Кривые обучения (из лога)
        plot_training_curves(exp_dir, plot_dir / 'training_curves.png')

    # --- Сохранение отчёта ---
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'metrics_window_level': metrics,
        'metrics_zone_level': {
            'macro_f1': zone_metrics['macro_f1'],
            'exact_match': zone_metrics['exact_match'],
        },
        'boundary': boundary,
        'optimal_thresholds': opt,
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
    }
    report_path = exp_dir / 'sim_ozz_evaluation.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nОтчёт сохранён: {report_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Оценка SimOZZ модели (Этап 4.5)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту (best_model.pt)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Ограничить число файлов (для отладки)')
    parser.add_argument('--val-max-samples', type=int, default=5000,
                        help='Максимум элементов для оценки')
    parser.add_argument('--no-plots', action='store_true',
                        help='Не строить графики')
    args = parser.parse_args()

    evaluate_sim_ozz(
        checkpoint_path=args.checkpoint,
        max_files=args.max_files,
        val_max_samples=args.val_max_samples,
        save_plots=not args.no_plots,
    )


if __name__ == '__main__':
    # =====================================================================
    # РУЧНОЙ ЗАПУСК — отредактируйте константы ниже
    # =====================================================================
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # Ручной режим
        # Укажите путь к чекпоинту после обучения модели на SimOZZ (best_model.pt).
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260425_225849/latest_checkpoint.pt' 
        # Пример: CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_.../best_model.pt'

        if CHECKPOINT is None:
            # Автопоиск последнего эксперимента
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
            print("Нет чекпоинта. Укажите CHECKPOINT или используйте --checkpoint")
        else:
            evaluate_sim_ozz(
                checkpoint_path=str(PROJECT_ROOT / CHECKPOINT) if not Path(CHECKPOINT).is_absolute() else CHECKPOINT,
                max_files=None,
                val_max_samples=50000,
                save_plots=True,
            )
