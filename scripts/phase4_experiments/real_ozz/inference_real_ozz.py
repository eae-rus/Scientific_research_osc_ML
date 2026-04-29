"""
Inference на реальных COMTRADE осциллограммах моделью SimOZZ (Этап 4.5 → 10.3).

Для каждого файла из data/real_OZZ/osc_comtrade/:
1. Парсит COMTRADE (поддержка мульти-секций / разных Fs)
2. Нормализует через NormOsc (norm_coef)
3. Прогоняет inference (скользящим окном)
4. Строит графики: токи + напряжения + вероятности ОЗЗ-классов
5. Автокроп вокруг зоны ОЗЗ (если модель обнаружила ОЗЗ)

Разделяет файлы на confirmed_ozz и false_detection по overvoltage_report.

Примеры:
  python scripts/phase4_experiments/inference_real_ozz.py --checkpoint .../best_model.pt
  python scripts/phase4_experiments/inference_real_ozz.py --checkpoint .../best_model.pt --subset confirmed --max-files 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.data_management.comtrade_processing import ReadComtrade
from osc_tools.features.normalization import NormOsc
from osc_tools.ml.augmented_dataset import compute_spectral_from_raw
from osc_tools.data_management.real_ozz_split import (
    load_real_ozz_report, split_by_verification, get_bus_for_file,
)
from scripts.phase4_experiments.evaluate_phase4 import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

COMTRADE_DIR = PROJECT_ROOT / 'data' / 'real_OZZ' / 'osc_comtrade'
NORM_COEF_PATH = PROJECT_ROOT / 'raw_data' / 'norm_coef_all_v1.4.csv'

# Цвета фаз (стандарт РЗА)
PHASE_COLORS = {
    'IA': '#FFD700', 'IB': '#228B22', 'IC': '#FF4500', 'IN': '#1E90FF',
    'UA': '#FFD700', 'UB': '#228B22', 'UC': '#FF4500', 'UN': '#1E90FF',
}
CLASS_COLORS = ['#E74C3C', '#F39C12', '#8E44AD', '#2980B9']
CLASS_LABELS = ['Stable SPGF', 'Petersen AIGF', 'Peters-Slepyan AIGF', 'Belyakov AIGF']


# ---------------------------------------------------------------------------
# Парсинг COMTRADE → сырые сигналы для inference
# ---------------------------------------------------------------------------

def load_comtrade_raw(
    cfg_path: Path,
    bus: str,
    norm_osc: NormOsc | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float, np.ndarray | None, dict]:
    """Загружает COMTRADE и извлекает 8 каналов для указанной секции (bus).

    Args:
        cfg_path: путь к .cfg файлу
        bus: номер секции ('1', '2', ...)
        norm_osc: объект нормализации (если None — без нормализации)

    Returns:
        (raw_8ch_display, raw_8ch_model, fs, time_arr, signal_info)
        - raw_8ch_display: (N, 8) оригинальные значения для графика
        - raw_8ch_model: (N, 8) нормализованные значения для inference
        - raw_8ch: (N, 8) numpy array [IA, IB, IC, IN, UA, UB, UC, UN] или None
        - fs: частота дискретизации (Гц)
        - time_arr: (N,) массив времени (с)
        - signal_info: dict с информацией о найденных сигналах
    """
    reader = ReadComtrade()
    rec, raw_df = reader.read_comtrade(str(cfg_path))

    if rec is None or raw_df is None or raw_df.is_empty():
        return None, None, 0.0, None, {}

    # Частота дискретизации (первая секция COMTRADE)
    fs = rec.cfg.sample_rates[0][0]
    frequency = rec.cfg.frequency  # промышленная частота (50/60 Гц)

    # Время
    time_arr = np.array(rec.time, dtype=np.float64)

    # Ищем каналы для указанной секции (bus)
    cols = set(raw_df.columns)

    # Маппинг длинных имён -> короткие
    channel_map = {
        'IA': [f'I | Bus-{bus} | phase: A'],
        'IB': [f'I | Bus-{bus} | phase: B'],
        'IC': [f'I | Bus-{bus} | phase: C'],
        'IN': [f'I | Bus-{bus} | phase: N'],
        'UA': [f'U | BusBar-{bus} | phase: A', f'U | CableLine-{bus} | phase: A'],
        'UB': [f'U | BusBar-{bus} | phase: B', f'U | CableLine-{bus} | phase: B'],
        'UC': [f'U | BusBar-{bus} | phase: C', f'U | CableLine-{bus} | phase: C'],
        'UN': [f'U | BusBar-{bus} | phase: N', f'U | CableLine-{bus} | phase: N'],
    }

    resolved = {}
    signal_info = {}
    for short_name, candidates in channel_map.items():
        found = False
        for cand in candidates:
            if cand in cols:
                resolved[short_name] = cand
                signal_info[short_name] = cand
                found = True
                break
        if not found:
            resolved[short_name] = None

    # Проверяем минимум каналов (хотя бы 3 тока + 3 напряжения)
    available = [k for k, v in resolved.items() if v is not None]
    required_currents = {'IA', 'IB', 'IC'}
    required_voltages = {'UA', 'UB', 'UC'}
    if not required_currents.issubset(set(available)):
        return None, None, fs, time_arr, signal_info
    if not required_voltages.issubset(set(available)):
        return None, None, fs, time_arr, signal_info

    # Собираем 8 каналов ОРИГИНАЛЬНЫХ (для графика)
    order = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
    N = raw_df.height
    raw_8ch_display = np.zeros((N, 8), dtype=np.float32)
    for i, ch in enumerate(order):
        col_name = resolved.get(ch)
        if col_name is not None and col_name in raw_df.columns:
            raw_8ch_display[:, i] = raw_df[col_name].to_numpy().astype(np.float32)

    # Нормализация для модели
    file_name = cfg_path.stem  # MD5-хэш
    raw_8ch_model = raw_8ch_display.copy()  # по умолчанию — без нормализации
    if norm_osc is not None:
        raw_df_norm = norm_osc.normalize_bus_signals(raw_df, file_name)
        if raw_df_norm is not None:
            # Собираем нормализованные каналы для inference
            raw_8ch_model = np.zeros((N, 8), dtype=np.float32)
            for i, ch in enumerate(order):
                col_name = resolved.get(ch)
                if col_name is not None and col_name in raw_df_norm.columns:
                    raw_8ch_model[:, i] = raw_df_norm[col_name].to_numpy().astype(np.float32)

    return raw_8ch_display, raw_8ch_model, fs, time_arr, signal_info


# ---------------------------------------------------------------------------
# Inference скользящим окном (адаптировано для разных Fs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_real_oscillogram(
    model: torch.nn.Module,
    raw_data: np.ndarray,
    config: dict,
    device: torch.device,
    fs: float,
    f_network: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Разметка реальной осциллограммы скользящим окном (адаптация Fs).

    Args:
        model: модель в eval mode
        raw_data: (N, 8) сырые отсчёты
        config: конфиг модели
        device: torch device
        fs: частота дискретизации файла (Гц)
        f_network: промышленная частота (50 Гц)

    Returns:
        (probs, coverage): пообразцовые вероятности и покрытие
    """
    model.eval()
    N = raw_data.shape[0]

    spp = round(fs / f_network)  # samples per period
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
        print(f"    Слишком короткая осциллограмма: {N} < {window_size}")
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
            stride=stride,
            warmup=fft_window,
            fft_window=fft_window,
            samples_per_period=spp,
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

    mask_covered = coverage > 0
    result = np.full((N, num_classes), np.nan, dtype=np.float32)
    result[mask_covered] = (prob_sum[mask_covered] / coverage[mask_covered, np.newaxis]).astype(np.float32)

    return result, coverage


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def _auto_crop_range(
    probs: np.ndarray,
    fs: float,
    f_network: float = 50.0,
    detection_threshold: float = 0.3,
    margin_periods: int = 10,
) -> tuple[int, int] | None:
    """Определяет диапазон автокропа вокруг зоны ОЗЗ.

    Returns:
        (start_sample, end_sample) или None если ОЗЗ не обнаружено
    """
    spp = round(fs / f_network)
    max_prob = np.nanmax(probs, axis=1)  # max по классам
    detected = max_prob >= detection_threshold

    if not np.any(detected):
        return None

    first = np.argmax(detected)
    last = len(detected) - 1 - np.argmax(detected[::-1])

    margin = margin_periods * spp
    start = max(0, first - margin)
    end = min(len(probs), last + margin)
    return int(start), int(end)


def plot_real_ozz_marking(
    out_path: Path,
    raw_data: np.ndarray,
    probs: np.ndarray,
    coverage: np.ndarray,
    fs: float,
    title: str,
    threshold: float = 0.5,
    auto_crop: bool = True,
    f_network: float = 50.0,
) -> None:
    """Строит график разметки для реальной COMTRADE осциллограммы."""
    N = raw_data.shape[0]
    num_classes = probs.shape[1]
    time_ms = np.arange(N) * 1000.0 / fs

    # Автокроп
    crop_range = None
    if auto_crop:
        crop_range = _auto_crop_range(probs, fs, f_network, detection_threshold=0.3, margin_periods=10)

    if crop_range is not None:
        s, e = crop_range
        raw_data = raw_data[s:e]
        probs = probs[s:e]
        coverage = coverage[s:e]
        time_ms = time_ms[s:e]
        N = e - s

    channel_names = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']

    height_ratios = [1.2, 1.2] + [0.6] * num_classes + [0.3]
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(18, 8 + 1.2 * num_classes))
    gs = fig.add_gridspec(nrows=n_rows, ncols=1, height_ratios=height_ratios)

    # --- Токи ---
    ax_i = fig.add_subplot(gs[0, 0])
    for idx, name in enumerate(['IA', 'IB', 'IC', 'IN']):
        ax_i.plot(time_ms, raw_data[:, idx], label=name,
                  color=PHASE_COLORS[name], linewidth=0.8, alpha=0.9)
    ax_i.set_ylabel('Токи')
    ax_i.legend(loc='upper right', fontsize=8)
    ax_i.grid(True, alpha=0.3, linestyle=':')

    # --- Напряжения ---
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_i)
    for idx_off, name in enumerate(['UA', 'UB', 'UC', 'UN']):
        ax_u.plot(time_ms, raw_data[:, 4 + idx_off], label=name,
                  color=PHASE_COLORS[name], linewidth=0.8, alpha=0.9)
    ax_u.set_ylabel('Напряжения')
    ax_u.legend(loc='upper right', fontsize=8)
    ax_u.grid(True, alpha=0.3, linestyle=':')

    # --- Вероятности ОЗЗ (по классам) ---
    for ci in range(num_classes):
        ax = fig.add_subplot(gs[2 + ci, 0], sharex=ax_i)
        p = probs[:, ci]
        color = CLASS_COLORS[ci % len(CLASS_COLORS)]
        label = CLASS_LABELS[ci] if ci < len(CLASS_LABELS) else f'Class {ci}'

        ax.plot(time_ms, p, color=color, linewidth=1.2, alpha=0.7)
        mask_high = p >= threshold
        if np.any(mask_high & ~np.isnan(p)):
            ax.fill_between(time_ms, 0, p, where=mask_high & ~np.isnan(p),
                            alpha=0.2, color=color)

        ax.axhline(threshold, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':')

    # --- Coverage ---
    ax_cov = fig.add_subplot(gs[2 + num_classes, 0], sharex=ax_i)
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
# Основной пайплайн
# ---------------------------------------------------------------------------

def run_inference_on_real_ozz(
    checkpoint_path: str,
    output_dir: str | None = None,
    subset: str = 'all',
    max_files: int | None = None,
    threshold: float = 0.5,
    auto_crop: bool = True,
    buses_filter: list[str] | None = None,
    mask_neutral: bool = True,
) -> dict:
    """Inference на реальных COMTRADE файлах.

    Args:
        checkpoint_path: путь к чекпоинту
        output_dir: папка для графиков
        subset: 'all' | 'confirmed' | 'false_detection'
        max_files: ограничение числа файлов
        threshold: порог бин. классификации
        auto_crop: автокроп вокруг обнаруженного ОЗЗ
        buses_filter: ограничить секции (например ['1'])
        mask_neutral: маскировать IN/UN (без нулевых последовательностей)
    Returns:
        dict со статистикой
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    model, config = load_model_from_checkpoint(ckpt_path, device)
    print(f"Модель: {config.get('model_type')}")
    if mask_neutral:
        print("Режим: IN/UN маскированы (=0) — без нулевых последовательностей")

    # Папка вывода
    if output_dir is None:
        out_base = PROJECT_ROOT / 'reports' / 'phase4' / 'real_ozz_inference'
    else:
        out_base = Path(output_dir)

    # Подпапки для confirmed / false_detection
    out_confirmed = out_base / 'confirmed_ozz'
    out_false = out_base / 'false_detection'
    out_confirmed.mkdir(parents=True, exist_ok=True)
    out_false.mkdir(parents=True, exist_ok=True)

    # Загрузка отчёта
    report = load_real_ozz_report()
    confirmed_df, false_df = split_by_verification(report)

    print(f"Отчёт: {len(confirmed_df)} confirmed, {len(false_df)} false_detection")

    # Нормализация
    norm_osc = None
    if NORM_COEF_PATH.exists():
        norm_osc = NormOsc(norm_coef_file_path=str(NORM_COEF_PATH))
        print(f"Нормализация: {NORM_COEF_PATH.name}")
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: norm_coef не найден ({NORM_COEF_PATH}), без нормализации")

    # Формируем список файлов для обработки
    if subset == 'confirmed':
        files_df = confirmed_df
    elif subset == 'false_detection':
        files_df = false_df
    else:
        files_df = pl.concat([confirmed_df, false_df])

    # Уникальные файлы
    unique_files = files_df['filename'].unique().sort().to_list()
    if max_files:
        unique_files = unique_files[:max_files]

    print(f"Файлов для обработки: {len(unique_files)} ({subset})")

    stats = {
        'total_files': len(unique_files),
        'processed': 0,
        'skipped': 0,
        'detected_ozz': 0,
        'no_ozz_detected': 0,
    }

    for file_idx, fname in enumerate(unique_files):
        cfg_path = COMTRADE_DIR / f'{fname}.cfg'
        if not cfg_path.exists():
            stats['skipped'] += 1
            continue

        # Определяем секции (bus) для файла
        file_buses = get_bus_for_file(report, fname)
        if not file_buses:
            file_buses = ['1']

        if buses_filter:
            file_buses = [b for b in file_buses if b in buses_filter]

        # Определяем, confirmed или false
        is_confirmed = fname in confirmed_df['filename'].to_list()
        out_dir_file = out_confirmed if is_confirmed else out_false
        label = 'confirmed' if is_confirmed else 'false_det'

        for bus in file_buses:
            raw_display, raw_model, fs, time_arr, sig_info = load_comtrade_raw(
                cfg_path, bus=bus, norm_osc=norm_osc,
            )

            if raw_display is None:
                continue

            bus_suffix = f'_bus{bus}' if len(file_buses) > 1 else ''
            print(f"  [{file_idx+1}/{len(unique_files)}] {fname}{bus_suffix} "
                  f"(N={raw_display.shape[0]}, Fs={fs:.0f}, {label})")

            # Inference — на НОРМАЛИЗОВАННЫХ данных
            raw_for_model = raw_model
            if mask_neutral:
                # Маскируем IN (idx=3) и UN (idx=7) — в реальных осциллограммах
                # эти каналы часто не записываются или содержат мусор
                raw_for_model = raw_model.copy()
                raw_for_model[:, 3] = 0.0   # IN
                raw_for_model[:, 7] = 0.0   # UN
            probs, cov = mark_real_oscillogram(
                model, raw_for_model, config, device, fs=fs,
            )

            # Определяем, есть ли ОЗЗ
            max_prob_per_sample = np.nanmax(probs, axis=1)
            has_ozz = np.nanmax(max_prob_per_sample) >= threshold

            if has_ozz:
                stats['detected_ozz'] += 1
            else:
                stats['no_ozz_detected'] += 1

            # Построение графика — ОРИГИНАЛЬНЫЕ значения
            plot_name = f'{fname}{bus_suffix}.png'
            plot_path = out_dir_file / plot_name
            title = f'{label.upper()} | {fname} | bus={bus} | Fs={fs:.0f} Гц'

            plot_real_ozz_marking(
                out_path=plot_path,
                raw_data=raw_display,
                probs=probs,
                coverage=cov,
                fs=fs,
                title=title,
                threshold=threshold,
                auto_crop=auto_crop and has_ozz,
            )

            stats['processed'] += 1

    # Итог
    print(f"\n{'='*60}")
    print(f"ИТОГО:")
    print(f"  Обработано: {stats['processed']}")
    print(f"  Пропущено: {stats['skipped']}")
    print(f"  ОЗЗ обнаружено: {stats['detected_ozz']}")
    print(f"  ОЗЗ не обнаружено: {stats['no_ozz_detected']}")
    print(f"  Графики: {out_base}")

    # Сохраняем статистику
    stats_path = out_base / 'inference_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Inference на реальных COMTRADE (Этап 4.5 → 10.3)',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту модели')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--subset', choices=['all', 'confirmed', 'false_detection'],
                        default='all',
                        help='Подмножество файлов')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-crop', action='store_true',
                        help='Не кропить графики вокруг ОЗЗ')
    parser.add_argument('--no-mask-neutral', action='store_true',
                        help='Не маскировать IN/UN (по умолчанию обнуляются)')
    parser.add_argument('--bus', type=str, default=None,
                        help='Фильтр секций (например "1,2")')
    args = parser.parse_args()

    buses = args.bus.split(',') if args.bus else None

    run_inference_on_real_ozz(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        subset=args.subset,
        max_files=args.max_files,
        threshold=args.threshold,
        auto_crop=not args.no_crop,
        buses_filter=buses,
        mask_neutral=not args.no_mask_neutral,
    )


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================
        # РУЧНОЙ РЕЖИМ — отредактируйте константы ниже
        # =====================================================
        # Укажите путь после обучения
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260428_174217/latest_checkpoint.pt' 
        # Пример: CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_.../best_model.pt'

        SUBSET = 'all'         # 'all' | 'confirmed' | 'false_detection'
        MAX_FILES = None       # None = все файлы
        THRESHOLD = 0.5
        AUTO_CROP = True
        MASK_NEUTRAL = False    # True = обнулять IN/UN перед inference

        if CHECKPOINT is None:
            # Автопоиск
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
            print("Нет чекпоинта. Укажите CHECKPOINT.")
        else:
            _ckpt = str(PROJECT_ROOT / CHECKPOINT) if not Path(CHECKPOINT).is_absolute() else CHECKPOINT
            run_inference_on_real_ozz(
                checkpoint_path=_ckpt,
                subset=SUBSET,
                max_files=MAX_FILES,
                threshold=THRESHOLD,
                auto_crop=AUTO_CROP,
                mask_neutral=MASK_NEUTRAL,
            )
