"""Сбор развёрнутой статистики по реальным ОЗЗ (Этап 10.4).

На основе inference_real_ozz.py разметки — запускает inference по всем файлам и
собирает:
  1) Файл-уровневая статистика (detection rate, FPR, per-class dist.)
  2) Зон-уровневая с тёмпоральной коррекцией (zone expansion)
  3) Распределение уверенности модели
  4) Тёмпоральные паттерны
  5) Multi-class co-occurrence

Запуск:
    python scripts/phase4_experiments/real_ozz/collect_real_ozz_statistics.py \
        --checkpoint experiments/phase4/.../best_model.pt \
        [--max-files 100] [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import torch

from osc_tools.data_management.real_ozz_split import (
    load_real_ozz_report, split_by_verification, get_bus_for_file,
)
from osc_tools.features.normalization import NormOsc
from scripts.phase4_experiments.evaluate_phase4 import load_model_from_checkpoint
from scripts.phase4_experiments.real_ozz.inference_real_ozz import (
    load_comtrade_raw, mark_real_oscillogram,
    COMTRADE_DIR, NORM_COEF_PATH,
)

CLASS_NAMES = ['Stable_SPGF', 'Petersen_AIGF', 'PetersSlepyan_AIGF', 'Belyakov_AIGF']

# ---------------------------------------------------------------------------
# Zone expansion (ДПОЗЗ → агрегированные события)
# ---------------------------------------------------------------------------

def expand_zones(
    binary_mask: np.ndarray,
    expansion_zones: int = 16,
) -> np.ndarray:
    """Расширение бинарной маски зон на ±expansion_zones.

    Это слияние коротких всплесков ДПОЗЗ в единые «события»
    (морфологическая дилатация).

    Args:
        binary_mask: (N,) bool или 0/1
        expansion_zones: число зон расширения (~1 период при stride_fraction=8)

    Returns:
        expanded: (N,) bool
    """
    if binary_mask.sum() == 0:
        return binary_mask.astype(bool)

    expanded = binary_mask.astype(bool).copy()
    indices = np.where(binary_mask)[0]
    for idx in indices:
        lo = max(0, idx - expansion_zones)
        hi = min(len(expanded), idx + expansion_zones + 1)
        expanded[lo:hi] = True
    return expanded


def find_events(
    binary_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """Находит connected components в бинарной маске.

    Returns:
        list of (start_idx, end_idx) — включительно
    """
    events = []
    in_event = False
    start = 0
    for i in range(len(binary_mask)):
        if binary_mask[i] and not in_event:
            start = i
            in_event = True
        elif not binary_mask[i] and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(binary_mask) - 1))
    return events


# ---------------------------------------------------------------------------
# Статистика одного файла
# ---------------------------------------------------------------------------

def compute_file_statistics(
    probs: np.ndarray,           # (N, 4) sample-level probabilities
    fs: float,
    threshold: float = 0.5,
    expansion_zones: int = 16,   # ~1 период при stride_fraction=8
) -> Dict:
    """Сбор статистики по одному файлу.

    Args:
        probs: (N_samples, 4) — вероятности classов (может содержать NaN)
        fs: частота дискретизации
        threshold: порог бинаризации
        expansion_zones: зон расширения для ДПОЗЗ

    Returns:
        dict с метриками файла
    """
    n_samples, n_classes = probs.shape
    # Убираем NaN (первые/последние сэмплы)
    valid_mask = ~np.isnan(probs[:, 0])
    probs_valid = probs[valid_mask]

    if len(probs_valid) == 0:
        return {'valid': False}

    duration_s = n_samples / fs
    result: Dict = {
        'valid': True,
        'n_samples': int(n_samples),
        'n_valid': int(len(probs_valid)),
        'fs': float(fs),
        'duration_s': float(duration_s),
    }

    # ── Per-class max probability ──
    max_probs = {}
    detected = {}
    for c in range(n_classes):
        mp = float(np.nanmax(probs_valid[:, c]))
        max_probs[CLASS_NAMES[c]] = mp
        detected[CLASS_NAMES[c]] = mp > threshold
    result['max_prob'] = max_probs
    result['detected'] = detected
    result['any_ozz'] = any(detected.values())

    # ── Zone-level с expansion ──
    events_info = {}
    for c in range(n_classes):
        binary = probs_valid[:, c] > threshold
        expanded = expand_zones(binary, expansion_zones)
        events = find_events(expanded)

        total_event_samples = sum(e - s + 1 for s, e in events)
        event_durations_ms = [(e - s + 1) / fs * 1000 for s, e in events]

        events_info[CLASS_NAMES[c]] = {
            'n_events': len(events),
            'total_samples': int(total_event_samples),
            'coverage_pct': float(total_event_samples / len(probs_valid) * 100),
            'event_durations_ms': event_durations_ms,
            'mean_duration_ms': float(np.mean(event_durations_ms)) if event_durations_ms else 0.0,
        }
    result['events'] = events_info

    # ── Temporal patterns ──
    for c in range(n_classes):
        binary = probs_valid[:, c] > threshold
        if binary.any():
            first_idx = int(np.argmax(binary))
            relative_onset = first_idx / len(probs_valid)
            result.setdefault('temporal', {})[CLASS_NAMES[c]] = {
                'onset_relative': float(relative_onset),
                'onset_ms': float(first_idx / fs * 1000),
            }

    # ── Multi-class co-occurrence ──
    types_detected = [c for c in CLASS_NAMES if detected[c]]
    result['n_types_detected'] = len(types_detected)
    result['types_detected'] = types_detected

    return result


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

def aggregate_statistics(
    per_file: List[Dict],
    confirmed_files: set,
    false_files: set,
) -> Dict:
    """Агрегирует per-file статистику."""

    agg: Dict = {
        'total_files': len(per_file),
        'valid_files': sum(1 for f in per_file if f.get('valid', False)),
    }

    valid = [f for f in per_file if f.get('valid', False)]

    # ── Разделение по группам ──
    for group_name, group_set in [('confirmed', confirmed_files), ('false_detection', false_files)]:
        group = [f for f in valid if f.get('filename') in group_set]
        n = len(group)
        if n == 0:
            agg[group_name] = {'count': 0}
            continue

        n_ozz = sum(1 for f in group if f['any_ozz'])
        n_no_ozz = n - n_ozz

        # Per-class detection
        per_class_det = {}
        for c in CLASS_NAMES:
            n_det = sum(1 for f in group if f['detected'].get(c, False))
            per_class_det[c] = {'count': n_det, 'pct': n_det / n * 100}

        # Max prob distribution (для гистограмм)
        max_probs_dist = {}
        for c in CLASS_NAMES:
            vals = [f['max_prob'].get(c, 0.0) for f in group]
            max_probs_dist[c] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'p90': float(np.percentile(vals, 90)),
                'values': vals,  # для гистограммы
            }

        # Events (зон-уровень с expansion)
        events_agg = {}
        for c in CLASS_NAMES:
            n_events_list = [f['events'][c]['n_events'] for f in group]
            coverage_list = [f['events'][c]['coverage_pct'] for f in group]
            events_agg[c] = {
                'total_events': int(np.sum(n_events_list)),
                'mean_events_per_file': float(np.mean(n_events_list)),
                'mean_coverage_pct': float(np.mean(coverage_list)),
                'files_with_events': sum(1 for x in n_events_list if x > 0),
            }

        # Multi-class co-occurrence
        n_types_list = [f['n_types_detected'] for f in group]
        cooccurrence = {}
        for n_types in range(5):
            cnt = sum(1 for x in n_types_list if x == n_types)
            cooccurrence[f'{n_types}_types'] = {'count': cnt, 'pct': cnt / n * 100}

        # Mean durations per class
        duration_agg = {}
        for c in CLASS_NAMES:
            all_durations = []
            for f in group:
                all_durations.extend(f['events'][c]['event_durations_ms'])
            if all_durations:
                duration_agg[c] = {
                    'mean_ms': float(np.mean(all_durations)),
                    'median_ms': float(np.median(all_durations)),
                    'total_events': len(all_durations),
                }

        agg[group_name] = {
            'count': n,
            'ozz_detected': n_ozz,
            'ozz_not_detected': n_no_ozz,
            'detection_rate_pct': n_ozz / n * 100,
            'per_class_detection': per_class_det,
            'max_prob_distribution': {
                c: {k: v for k, v in d.items() if k != 'values'}
                for c, d in max_probs_dist.items()
            },
            'events': events_agg,
            'cooccurrence': cooccurrence,
            'duration': duration_agg,
            # Для гистограмм сохраняем отдельно (только в JSON)
            '_max_prob_values': {c: d['values'] for c, d in max_probs_dist.items()},
        }

    return agg


# ---------------------------------------------------------------------------
# Основной запуск
# ---------------------------------------------------------------------------

def collect_statistics(
    checkpoint_path: str,
    max_files: int | None = None,
    threshold: float = 0.5,
    expansion_zones: int = 16,
    mask_neutral: bool = True,
    output_dir: str | None = None,
) -> Dict:
    """Собирает статистику по реальным осциллограммам."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    model, config = load_model_from_checkpoint(ckpt_path, device)
    print(f"Модель: {config.get('model_type')}")

    # Отчёт
    report = load_real_ozz_report()
    confirmed_df, false_df = split_by_verification(report)
    confirmed_files_set = set(confirmed_df['filename'].to_list())
    false_files_set = set(false_df['filename'].to_list())

    print(f"Отчёт: {len(confirmed_files_set)} confirmed, {len(false_files_set)} false_detection")

    # Нормализация
    norm_osc = None
    if NORM_COEF_PATH.exists():
        norm_osc = NormOsc(norm_coef_file_path=str(NORM_COEF_PATH))

    # Все уникальные файлы
    all_files_df = pl.concat([confirmed_df, false_df])
    unique_files = all_files_df['filename'].unique().sort().to_list()
    if max_files:
        unique_files = unique_files[:max_files]

    print(f"Файлов: {len(unique_files)}")

    per_file_stats: List[Dict] = []
    t0 = time.time()

    for file_idx, fname in enumerate(unique_files):
        cfg_path = COMTRADE_DIR / f'{fname}.cfg'
        if not cfg_path.exists():
            continue

        file_buses = get_bus_for_file(report, fname) or ['1']

        for bus in file_buses:
            raw_display, raw_model, fs, time_arr, sig_info = load_comtrade_raw(
                cfg_path, bus=bus, norm_osc=norm_osc,
            )
            if raw_display is None:
                continue

            # Маскирование IN/UN
            raw_for_model = raw_model
            if mask_neutral:
                raw_for_model = raw_model.copy()
                raw_for_model[:, 3] = 0.0  # IN
                raw_for_model[:, 7] = 0.0  # UN

            probs, cov = mark_real_oscillogram(
                model, raw_for_model, config, device, fs=fs,
            )

            file_stat = compute_file_statistics(
                probs, fs, threshold=threshold,
                expansion_zones=expansion_zones,
            )
            file_stat['filename'] = fname
            file_stat['bus'] = bus
            file_stat['group'] = (
                'confirmed' if fname in confirmed_files_set
                else 'false_detection'
            )
            per_file_stats.append(file_stat)

            if (file_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                eta = (len(unique_files) - file_idx - 1) / rate
                print(f"  [{file_idx+1}/{len(unique_files)}] "
                      f"{rate:.1f} files/s, ETA {eta:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\nОбработано {len(per_file_stats)} файлов за {elapsed_total:.1f}s")

    # ── Агрегация ──
    agg = aggregate_statistics(per_file_stats, confirmed_files_set, false_files_set)

    # ── Сохранение ──
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / 'reports' / 'phase4' / 'real_ozz_statistics')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # JSON (полный)
    json_path = out_path / 'real_ozz_statistics.json'
    # Убираем _max_prob_values для компактности основного JSON
    agg_clean = _remove_internal_keys(agg)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(agg_clean, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")

    # Per-file (для дальнейшего анализа)
    per_file_path = out_path / 'per_file_statistics.json'
    with open(per_file_path, 'w', encoding='utf-8') as f:
        json.dump(per_file_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"Per-file: {per_file_path}")

    # ── Печать сводки ──
    _print_summary(agg)

    # ── Визуализация ──
    try:
        _plot_statistics(agg, per_file_stats, out_path)
    except ImportError:
        print("matplotlib не найден — визуализация пропущена")

    return agg


def _remove_internal_keys(d: dict) -> dict:
    """Рекурсивно убирает ключи, начинающиеся с _."""
    result = {}
    for k, v in d.items():
        if k.startswith('_'):
            continue
        if isinstance(v, dict):
            result[k] = _remove_internal_keys(v)
        else:
            result[k] = v
    return result


def _print_summary(agg: dict) -> None:
    """Печатает сводку в консоль."""
    print(f"\n{'='*70}")
    print("СВОДКА СТАТИСТИКИ")
    print(f"{'='*70}")

    for group in ['confirmed', 'false_detection']:
        g = agg.get(group, {})
        n = g.get('count', 0)
        if n == 0:
            continue

        print(f"\n{'─'*50}")
        print(f"  {group.upper()} ({n} файлов)")
        print(f"{'─'*50}")
        print(f"  Detection Rate: {g.get('detection_rate_pct', 0):.1f}%")
        print(f"  ОЗЗ обнаружено: {g.get('ozz_detected', 0)}")
        print(f"  ОЗЗ не обнаружено: {g.get('ozz_not_detected', 0)}")

        pcd = g.get('per_class_detection', {})
        print(f"\n  Per-class detection:")
        for c in CLASS_NAMES:
            info = pcd.get(c, {})
            print(f"    {c:30s}: {info.get('count', 0):4d} ({info.get('pct', 0):.1f}%)")

        mp = g.get('max_prob_distribution', {})
        print(f"\n  Max probability (median/mean±std):")
        for c in CLASS_NAMES:
            info = mp.get(c, {})
            print(f"    {c:30s}: med={info.get('median', 0):.3f}  "
                  f"mean={info.get('mean', 0):.3f}±{info.get('std', 0):.3f}")

        ev = g.get('events', {})
        print(f"\n  Events (после zone expansion):")
        for c in CLASS_NAMES:
            info = ev.get(c, {})
            print(f"    {c:30s}: {info.get('total_events', 0):4d} events "
                  f"({info.get('files_with_events', 0)} files), "
                  f"mean coverage {info.get('mean_coverage_pct', 0):.1f}%")

        co = g.get('cooccurrence', {})
        print(f"\n  Co-occurrence (типов ОЗЗ в файле):")
        for key, info in sorted(co.items()):
            print(f"    {key}: {info.get('count', 0)} ({info.get('pct', 0):.1f}%)")


def _plot_statistics(agg: dict, per_file: list, out_path: Path) -> None:
    """Визуализация статистики."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── 1) Confidence distributions per-class (confirmed vs false) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for c_idx, ax in enumerate(axes.flat):
        cls = CLASS_NAMES[c_idx]
        for group, color, label in [
            ('confirmed', '#2ecc71', 'Confirmed OZZ'),
            ('false_detection', '#e74c3c', 'False Detection'),
        ]:
            vals = agg.get(group, {}).get('_max_prob_values', {}).get(cls, [])
            if vals:
                ax.hist(vals, bins=30, alpha=0.6, color=color, label=label,
                        density=True, edgecolor='white')
        ax.set_title(cls, fontsize=11)
        ax.set_xlabel('Max P(class)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='threshold')
    fig.suptitle('Распределение уверенности модели по классам', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path / 'confidence_distributions.png', dpi=150)
    plt.close(fig)
    print(f"  График: {out_path / 'confidence_distributions.png'}")

    # ── 2) Detection pie charts ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, group in zip(axes, ['confirmed', 'false_detection']):
        g = agg.get(group, {})
        n_det = g.get('ozz_detected', 0)
        n_no = g.get('ozz_not_detected', 0)
        if n_det + n_no == 0:
            ax.set_title(f'{group}: нет данных')
            continue
        ax.pie(
            [n_det, n_no],
            labels=['ОЗЗ обнаружено', 'ОЗЗ не обнаружено'],
            colors=['#e74c3c', '#95a5a6'],
            autopct='%1.1f%%',
            startangle=90,
        )
        ax.set_title(f'{group.upper()} ({n_det + n_no} файлов)')
    plt.tight_layout()
    fig.savefig(out_path / 'detection_pie.png', dpi=150)
    plt.close(fig)
    print(f"  График: {out_path / 'detection_pie.png'}")

    # ── 3) Per-class detection bar chart ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    for i, (group, color) in enumerate([('confirmed', '#2ecc71'), ('false_detection', '#e74c3c')]):
        pcd = agg.get(group, {}).get('per_class_detection', {})
        vals = [pcd.get(c, {}).get('pct', 0) for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width, label=group, color=color, alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Per-class Detection Rate: Confirmed vs False Detection')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path / 'per_class_detection.png', dpi=150)
    plt.close(fig)
    print(f"  График: {out_path / 'per_class_detection.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Сбор статистики по реальным ОЗЗ')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту модели')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Макс. число файлов (для теста)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--expansion-zones', type=int, default=16,
                        help='Зон расширения для ДПОЗЗ (~1 период при stride_fraction=8)')
    parser.add_argument('--no-mask-neutral', action='store_true',
                        help='Не маскировать IN/UN')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    collect_statistics(
        checkpoint_path=args.checkpoint,
        max_files=args.max_files,
        threshold=args.threshold,
        expansion_zones=args.expansion_zones,
        mask_neutral=not args.no_mask_neutral,
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

        MAX_FILES = None        # None = все файлы
        THRESHOLD = 0.5
        EXPANSION_ZONES = 16    # ~1 период при stride_fraction=8
        MASK_NEUTRAL = True     # True = обнулять IN/UN перед inference

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
            collect_statistics(
                checkpoint_path=_ckpt,
                max_files=MAX_FILES,
                threshold=THRESHOLD,
                expansion_zones=EXPANSION_ZONES,
                mask_neutral=MASK_NEUTRAL,
            )
