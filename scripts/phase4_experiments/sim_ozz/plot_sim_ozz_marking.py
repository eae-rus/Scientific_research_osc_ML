"""
Построение графиков разметки SimOZZ осциллограмм.

Выбирает N рандомных файлов для каждого класса ОЗЗ,
прогоняет скользящее окно inference с усреднением перекрытий,
строит графики: сигналы + вероятности + ground truth.

Примеры:
  python scripts/phase4_experiments/sim_ozz/plot_sim_ozz_marking.py --checkpoint .../best_model.pt
  python scripts/phase4_experiments/sim_ozz/plot_sim_ozz_marking.py --checkpoint .../best_model.pt --per-class 10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex, FileInfo, load_raw_csv, SIM_NOMINAL,
    TARGET_COLUMNS as SIM_TARGET_COLUMNS, ARC_TYPES,
)
from osc_tools.ml.augmented_dataset import compute_spectral_from_raw
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from scripts.phase4_experiments.evaluate_phase4 import (
    load_model_from_checkpoint, PREDICTION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Цвета
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    'IA': '#FFD700', 'IB': '#228B22', 'IC': '#FF4500', 'IN': '#1E90FF',
    'UA': '#FFD700', 'UB': '#228B22', 'UC': '#FF4500', 'UN': '#1E90FF',
}
CLASS_COLORS = ['#E74C3C', '#F39C12', '#8E44AD', '#2980B9']
RAW_CHANNELS = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']


# ---------------------------------------------------------------------------
# Sliding-window inference с усреднением перекрытий
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_sim_ozz_oscillogram(
    model: torch.nn.Module,
    raw_data: np.ndarray,
    config: dict,
    device: torch.device,
    fs: float,
    f_network: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Разметка SimOZZ файла скользящим окном с усреднением.

    Returns:
        probs: (T, num_classes) — пообразцовые вероятности (NaN где нет покрытия)
        coverage: (T,) — число окон, покрыших каждый отсчёт
    """
    model.eval()
    N = raw_data.shape[0]

    spp = round(fs / f_network)
    num_harmonics = config.get('num_harmonics', 9)
    sub_periods = config.get('sub_periods', [2, 4, 6, 10])
    include_symmetric = config.get('include_symmetric', True)
    stride_fraction = config.get('stride_fraction', 8)
    num_periods_window = config.get('num_periods_window', 10)
    num_classes = config.get('num_classes', 4)

    stride = max(1, spp // stride_fraction)
    fft_window = spp
    window_size = spp * num_periods_window
    step = spp  # шаг скользящего окна = 1 период

    if N < window_size + spp:
        return np.full((N, num_classes), np.nan, dtype=np.float32), np.zeros(N, dtype=np.int32)

    prob_sum = np.zeros((N, num_classes), dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int32)

    starts = list(range(0, max(1, N - window_size + 1), step))

    for win_start in starts:
        win_end = min(win_start + window_size, N)
        raw_win = raw_data[win_start:win_end].copy()

        if len(raw_win) < window_size:
            pad = np.zeros((window_size - len(raw_win), 8), dtype=np.float32)
            raw_win = np.concatenate([raw_win, pad], axis=0)

        spectral = compute_spectral_from_raw(
            raw_win, num_harmonics,
            sub_periods if sub_periods else None,
            include_symmetric,
            stride=stride, warmup=fft_window,
            fft_window=fft_window, samples_per_period=spp,
        )

        X = torch.from_numpy(spectral.T.copy()).unsqueeze(0).to(device)
        out = model(X, mode='classify')
        logits = out['classify']
        probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

        n_zones = probs.shape[0]
        for z in range(n_zones):
            raw_pos = win_start + fft_window + z * stride
            z_end = min(raw_pos + stride, N)
            if raw_pos >= N:
                break
            prob_sum[raw_pos:z_end] += probs[z]
            coverage[raw_pos:z_end] += 1

    result = np.full((N, num_classes), np.nan, dtype=np.float32)
    mask = coverage > 0
    result[mask] = (prob_sum[mask] / coverage[mask, np.newaxis]).astype(np.float32)

    return result, coverage


# ---------------------------------------------------------------------------
# Визуализация одной осциллограммы
# ---------------------------------------------------------------------------

def plot_sim_ozz_marking(
    out_path: Path,
    raw_data: np.ndarray,
    probs: np.ndarray,
    coverage: np.ndarray,
    targets: np.ndarray,
    fs: float,
    title: str,
    threshold: float = 0.5,
) -> None:
    """Строит график разметки для одной SimOZZ осциллограммы."""
    N = raw_data.shape[0]
    num_classes = probs.shape[1]
    time_ms = np.arange(N) * 1000.0 / fs

    # 2 строки (токи, напряжения) + num_classes строк вероятностей + 1 строка coverage
    height_ratios = [1.2, 1.2] + [0.6] * num_classes + [0.3]
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(18, 8 + 1.2 * num_classes))
    gs = fig.add_gridspec(nrows=n_rows, ncols=1, height_ratios=height_ratios)

    # --- Токи (×1000 → А) ---
    ax_i = fig.add_subplot(gs[0, 0])
    for idx, name in enumerate(['IA', 'IB', 'IC', 'IN']):
        ax_i.plot(time_ms, raw_data[:, idx] * 1000, label=name,
                  color=PHASE_COLORS[name], linewidth=0.8, alpha=0.9)
    ax_i.set_ylabel('Токи, А')
    ax_i.legend(loc='upper right', fontsize=8)
    ax_i.grid(True, alpha=0.3, linestyle=':')

    # --- Напряжения (кВ) ---
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_i)
    for idx_off, name in enumerate(['UA', 'UB', 'UC', 'UN']):
        ax_u.plot(time_ms, raw_data[:, 4 + idx_off], label=name,
                  color=PHASE_COLORS[name], linewidth=0.8, alpha=0.9)
    ax_u.set_ylabel('Напряжения, кВ')
    ax_u.legend(loc='upper right', fontsize=8)
    ax_u.grid(True, alpha=0.3, linestyle=':')

    # --- Вероятности + Ground Truth ---
    for ci in range(num_classes):
        ax = fig.add_subplot(gs[2 + ci, 0], sharex=ax_i)
        p = probs[:, ci]
        gt = targets[:, ci]
        color = CLASS_COLORS[ci % len(CLASS_COLORS)]
        label = ARC_TYPES[ci] if ci < len(ARC_TYPES) else f'Class {ci}'

        # Ground truth (полоса)
        gt_mask = gt >= 0.5
        if np.any(gt_mask):
            ax.fill_between(time_ms, 0, 1, where=gt_mask,
                            alpha=0.15, color='green', label='GT')

        # Предсказание
        valid = ~np.isnan(p)
        ax.plot(time_ms[valid], p[valid], color=color, linewidth=1.2, alpha=0.8)

        # Зона выше порога
        high = (p >= threshold) & valid
        if np.any(high):
            ax.fill_between(time_ms, 0, p, where=high,
                            alpha=0.2, color=color)

        ax.axhline(threshold, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle=':')

    # --- Coverage ---
    ax_cov = fig.add_subplot(gs[-1, 0], sharex=ax_i)
    ax_cov.fill_between(time_ms, 0, coverage, color='steelblue', alpha=0.3)
    ax_cov.plot(time_ms, coverage, color='steelblue', linewidth=0.6)
    ax_cov.set_ylabel('Coverage', fontsize=8)
    ax_cov.set_xlabel('Время, мс')
    ax_cov.grid(True, alpha=0.3, linestyle=':')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Выбор файлов по классам
# ---------------------------------------------------------------------------

def select_files_per_class(
    file_index: SimOZZFileIndex,
    per_class: int = 25,
    seed: int = 42,
) -> Dict[str, List[FileInfo]]:
    """Выбирает per_class рандомных файлов для каждого типа ОЗЗ (X=1..4).

    В SimOZZ: X (meta['x']) = 1..4 → {Stable, Petersen, PetersSlepian, Beliakov}.
    """
    rng = np.random.RandomState(seed)
    by_class: Dict[str, List[FileInfo]] = defaultdict(list)

    for fi in file_index.files:
        x = fi.meta.get('x', 0)
        if 1 <= x <= 4:
            arc_name = ARC_TYPES[x - 1]
            by_class[arc_name].append(fi)

    selected: Dict[str, List[FileInfo]] = {}
    for cls_name, files in by_class.items():
        n = min(per_class, len(files))
        chosen_indices = rng.choice(len(files), size=n, replace=False)
        selected[cls_name] = [files[i] for i in chosen_indices]
        print(f"  {cls_name}: {n} файлов (всего доступно: {len(files)})")

    return selected


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def generate_sim_ozz_markings(
    checkpoint_path: str,
    output_dir: str | None = None,
    per_class: int = 25,
    max_files: int | None = None,  # используется как max_files для val split
    threshold: float = 0.5,
    seed: int = 42,
) -> None:
    """Генерация графиков разметки SimOZZ: per_class файлов из каждого класса."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    model, config = load_model_from_checkpoint(ckpt_path, device)
    print(f"Модель: {config.get('model_type')}")

    # Папка вывода
    if output_dir is None:
        exp_dir = ckpt_path.parent
        out_base = exp_dir / 'marking_plots'
    else:
        out_base = Path(output_dir)

    # Val split
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
    print(f"Val файлов: {len(val_file_index)}")

    # Выбор файлов по классам
    print(f"\nВыбор {per_class} файлов на класс...")
    selected = select_files_per_class(val_file_index, per_class=per_class, seed=seed)

    total_plots = sum(len(v) for v in selected.values())
    plot_count = 0

    for cls_name, files in selected.items():
        cls_dir = out_base / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for fi in files:
            plot_count += 1

            # Загрузка через официальный парсер (корректный маппинг колонок)
            data_raw = load_raw_csv(fi.path, normalize=False)
            data_norm = load_raw_csv(fi.path, normalize=True)
            if data_raw is None or data_norm is None:
                print(f"  Пропуск: {fi.path.name} (не удалось прочитать)")
                continue
            raw_display = data_raw['raw']      # для графиков (исходные единицы)
            raw_model = data_norm['raw']        # для модели (нормализованные)
            targets = data_raw['targets']
            fs = fi.fs

            # Inference на нормализованных данных (как при обучении)
            probs, coverage = mark_sim_ozz_oscillogram(
                model, raw_model, config, device, fs=fs,
            )

            # График на сырых данных (читаемые единицы)
            title = f'{cls_name} | {fi.path.stem} | Fs={fs:.0f} Гц | T={raw_display.shape[0]}'
            plot_path = cls_dir / f'{fi.path.stem}.png'

            plot_sim_ozz_marking(
                out_path=plot_path,
                raw_data=raw_display,
                probs=probs,
                coverage=coverage,
                targets=targets,
                fs=fs,
                title=title,
                threshold=threshold,
            )

            if plot_count % 5 == 0 or plot_count == total_plots:
                print(f"  [{plot_count}/{total_plots}] {cls_name}/{fi.path.stem}")

    print(f"\nГотово! {plot_count} графиков → {out_base}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Графики разметки SimOZZ')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--per-class', type=int, default=25,
                        help='Число графиков на класс')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_sim_ozz_markings(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        per_class=args.per_class,
        threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================================
        # РУЧНОЙ ЗАПУСК
        # =====================================================================
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260426_143604/latest_checkpoint.pt'

        if CHECKPOINT is None:
            print("Укажите CHECKPOINT.")
        else:
            _ckpt = str(PROJECT_ROOT / CHECKPOINT) if not Path(CHECKPOINT).is_absolute() else CHECKPOINT
            generate_sim_ozz_markings(
                checkpoint_path=_ckpt,
                per_class=25,
                threshold=0.5,
            )
