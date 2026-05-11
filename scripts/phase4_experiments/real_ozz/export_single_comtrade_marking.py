"""Экспорт разметки одной COMTRADE-осциллограммы в CSV/PNG.

Назначение:
- быстро проверить конкретный файл без прогона всего real_OZZ набора;
- сохранить данные для последующей качественной перерисовки графиков;
- получить CSV с исходными сигналами, нормированными сигналами и вероятностями
  4 классов ОЗЗ по каждому отсчёту.

Пример:
    python scripts/phase4_experiments/real_ozz/export_single_comtrade_marking.py \
        --checkpoint experiments/phase4/.../latest_checkpoint.pt \
        --cfg data/real_OZZ/osc_comtrade/<file>.cfg \
        --bus 1 \
        --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.features.normalization import NormOsc
from osc_tools.data_management.real_ozz_split import load_real_ozz_report, get_bus_for_file
from scripts.phase4_experiments.evaluate_phase4 import load_model_from_checkpoint
from scripts.phase4_experiments.real_ozz.inference_real_ozz import (
    CLASS_LABELS,
    NORM_COEF_PATH,
    load_comtrade_raw,
    mark_real_oscillogram,
    plot_real_ozz_marking,
)

SIGNAL_NAMES = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
PROB_COLUMNS = [
    'P_Stable_SPGF',
    'P_Petersen_AIGF',
    'P_PetersSlepyan_AIGF',
    'P_Belyakov_AIGF',
]


def _resolve_checkpoint(checkpoint: str) -> Path:
    """Возвращает абсолютный путь к чекпоинту."""
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt}")
    return ckpt


def _resolve_cfg(cfg: str) -> Path:
    """Возвращает абсолютный путь к COMTRADE .cfg файлу."""
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if cfg_path.suffix.lower() != '.cfg':
        raise ValueError(f"Ожидался .cfg файл, получено: {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"COMTRADE .cfg не найден: {cfg_path}")
    return cfg_path


def _candidate_buses(cfg_path: Path, bus: str) -> list[str]:
    """Определяет список секций для обработки."""
    if bus != 'auto':
        return [b.strip() for b in bus.split(',') if b.strip()]

    try:
        report = load_real_ozz_report()
        buses = get_bus_for_file(report, cfg_path.stem)
        if buses:
            return buses
    except Exception as exc:
        print(f"Не удалось взять bus из отчёта: {exc}")

    # Если файла нет в отчёте, пробуем типовые секции.
    return ['1', '2', '3', '4', '5']


def _make_output_paths(
    cfg_path: Path,
    bus: str,
    output: str | None,
    output_dir: str | None,
) -> tuple[Path, Path]:
    """Формирует пути CSV и PNG."""
    if output is not None:
        csv_path = Path(output)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
    else:
        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / 'reports' / 'phase4' / 'single_marking'
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        csv_path = out_dir / f'{cfg_path.stem}_bus{bus}_marking.csv'

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = csv_path.with_suffix('.png')
    return csv_path, png_path


def _build_export_frame(
    time_arr: np.ndarray,
    raw_display: np.ndarray,
    raw_model: np.ndarray,
    probs: np.ndarray,
    coverage: np.ndarray,
    threshold: float,
    fs: float,
    cfg_path: Path,
    bus: str,
) -> pl.DataFrame:
    """Создаёт таблицу для CSV-экспорта."""
    data: dict[str, object] = {
        'file': [cfg_path.stem] * len(time_arr),
        'bus': [bus] * len(time_arr),
        'fs': np.full(len(time_arr), fs, dtype=np.float32),
        'time_s': time_arr.astype(np.float64),
    }

    for idx, name in enumerate(SIGNAL_NAMES):
        data[name] = raw_display[:, idx].astype(np.float32)
    for idx, name in enumerate(SIGNAL_NAMES):
        data[f'{name}_model'] = raw_model[:, idx].astype(np.float32)

    for idx, name in enumerate(PROB_COLUMNS):
        data[name] = probs[:, idx].astype(np.float32)

    max_prob = np.nanmax(probs, axis=1).astype(np.float32)
    data['P_max'] = max_prob
    data['pred_any_ozz'] = (max_prob >= threshold).astype(np.int8)
    data['coverage'] = coverage.astype(np.int32)

    for idx, label in enumerate(CLASS_LABELS):
        safe_label = label.replace(' ', '_').replace('-', '_')
        data[f'pred_{safe_label}'] = (probs[:, idx] >= threshold).astype(np.int8)

    return pl.DataFrame(data)


def export_single_comtrade_marking(
    checkpoint: str,
    cfg: str,
    bus: str = '1',
    output: str | None = None,
    output_dir: str | None = None,
    threshold: float = 0.5,
    mask_neutral: bool = True,
    plot: bool = False,
    no_crop: bool = False,
    fast_mode: bool = False,
) -> list[dict[str, str | float | int]]:
    """Обрабатывает один COMTRADE .cfg и сохраняет CSV/PNG.

    Args:
        checkpoint: путь к чекпоинту модели.
        cfg: путь к COMTRADE .cfg.
        bus: номер секции, список через запятую или 'auto'.
        output: путь CSV для одиночного bus; если bus несколько, добавляется суффикс.
        output_dir: папка вывода при автоматическом имени.
        threshold: порог бинарной разметки.
        mask_neutral: обнулять IN/UN перед inference.
        plot: сохранять PNG-график.
        no_crop: не выполнять автокроп на PNG.
        fast_mode: ускоренный inference (шаг = размер окна).

    Returns:
        список кратких словарей по обработанным секциям.
    """
    ckpt_path = _resolve_checkpoint(checkpoint)
    cfg_path = _resolve_cfg(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model, config = load_model_from_checkpoint(ckpt_path, device)
    print(f"Модель: {config.get('model_type')}")

    norm_osc = None
    if NORM_COEF_PATH.exists():
        norm_osc = NormOsc(norm_coef_file_path=str(NORM_COEF_PATH))
        print(f"Нормализация: {NORM_COEF_PATH.name}")
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: norm_coef не найден ({NORM_COEF_PATH}), без нормализации")

    results = []
    buses = _candidate_buses(cfg_path, bus)
    for bus_id in buses:
        raw_display, raw_model, fs, time_arr, sig_info = load_comtrade_raw(
            cfg_path, bus=bus_id, norm_osc=norm_osc,
        )
        if raw_display is None or raw_model is None or time_arr is None:
            print(f"bus={bus_id}: не удалось извлечь обязательные каналы")
            continue

        raw_for_model = raw_model.copy()
        if mask_neutral:
            raw_for_model[:, 3] = 0.0  # IN
            raw_for_model[:, 7] = 0.0  # UN

        print(f"bus={bus_id}: N={raw_display.shape[0]}, Fs={fs:.0f} Гц")
        probs, coverage = mark_real_oscillogram(
            model, raw_for_model, config, device, fs=fs, fast_mode=fast_mode,
        )

        csv_path, png_path = _make_output_paths(cfg_path, bus_id, output, output_dir)
        if output is not None and len(buses) > 1:
            csv_path = csv_path.with_name(f'{csv_path.stem}_bus{bus_id}{csv_path.suffix}')
            png_path = csv_path.with_suffix('.png')

        df = _build_export_frame(
            time_arr=time_arr,
            raw_display=raw_display,
            raw_model=raw_for_model,
            probs=probs,
            coverage=coverage,
            threshold=threshold,
            fs=fs,
            cfg_path=cfg_path,
            bus=bus_id,
        )
        df.write_csv(csv_path)

        max_prob = float(np.nanmax(probs))
        detected = max_prob >= threshold
        print(f"  CSV: {csv_path}")
        print(f"  max P={max_prob:.3f}, detected={detected}")

        if plot:
            title = f'{cfg_path.stem} | bus={bus_id} | Fs={fs:.0f} Гц | maxP={max_prob:.3f}'
            plot_real_ozz_marking(
                out_path=png_path,
                raw_data=raw_display,
                probs=probs,
                coverage=coverage,
                fs=fs,
                title=title,
                threshold=threshold,
                auto_crop=(not no_crop) and detected,
            )
            print(f"  PNG: {png_path}")

        meta_path = csv_path.with_suffix('.json')
        meta = {
            'checkpoint': str(ckpt_path),
            'cfg': str(cfg_path),
            'bus': bus_id,
            'fs': fs,
            'n_samples': int(raw_display.shape[0]),
            'threshold': threshold,
            'mask_neutral': mask_neutral,
            'max_probability': max_prob,
            'detected': bool(detected),
            'signal_info': sig_info,
            'csv': str(csv_path),
            'png': str(png_path) if plot else None,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        print(f"  META: {meta_path}")
        results.append(meta)

    if not results:
        raise RuntimeError("Не обработана ни одна секция. Проверьте --bus и наличие каналов.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Экспорт разметки одного COMTRADE в CSV/PNG')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту модели')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Путь к COMTRADE .cfg файлу')
    parser.add_argument('--bus', type=str, default='1',
                        help="Номер секции, список '1,2' или 'auto'")
    parser.add_argument('--output', type=str, default=None,
                        help='Путь к CSV. Для нескольких bus добавится суффикс.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Папка вывода при автоматическом имени')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-mask-neutral', action='store_true',
                        help='Не обнулять IN/UN перед inference')
    parser.add_argument('--plot', action='store_true', help='Сохранить PNG-график')
    parser.add_argument('--no-crop', action='store_true', help='Не кропить PNG вокруг ОЗЗ')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Быстрый inference (шаг = размер окна)')
    args = parser.parse_args()

    export_single_comtrade_marking(
        checkpoint=args.checkpoint,
        cfg=args.cfg,
        bus=args.bus,
        output=args.output,
        output_dir=args.output_dir,
        threshold=args.threshold,
        mask_neutral=not args.no_mask_neutral,
        plot=args.plot,
        no_crop=args.no_crop,
        fast_mode=args.fast_mode,
    )


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================
        # РУЧНОЙ РЕЖИМ — отредактируйте константы ниже
        # =====================================================
        # Пути и параметры для одиночного экспорта одной COMTRADE-осциллограммы.
        # Подходит для подготовки рисунков статьи: модель может быть почти
        # любой обученной (PhysicalKAN/PhysicalMLP/Baseline) — ключевые
        # совместимые поля берутся из чекпоинта.
        CHECKPOINT = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260428_174217/latest_checkpoint.pt'
        CFG = 'data/real_OZZ/osc_comtrade/<укажите_файл>.cfg'

        BUS = 'auto'           # '1' | '1,2' | 'auto' (взять из overvoltage_report)
        OUTPUT = None          # путь к CSV, иначе автоимя
        OUTPUT_DIR = None      # папка для авто-имени (по умолчанию reports/phase4/single_marking)
        THRESHOLD = 0.5
        MASK_NEUTRAL = False   # True = обнулять IN/UN
        PLOT = True            # сохранить PNG для статьи
        NO_CROP = False        # True = не кропить PNG
        FAST_MODE = False      # True = шаг = размер окна (быстро, грубее)

        if 'укажите_файл' in CFG:
            print("Укажите путь к CFG в константе CFG (ручной режим).")
        else:
            export_single_comtrade_marking(
                checkpoint=CHECKPOINT,
                cfg=CFG,
                bus=BUS,
                output=OUTPUT,
                output_dir=OUTPUT_DIR,
                threshold=THRESHOLD,
                mask_neutral=MASK_NEUTRAL,
                plot=PLOT,
                no_crop=NO_CROP,
                fast_mode=FAST_MODE,
            )
