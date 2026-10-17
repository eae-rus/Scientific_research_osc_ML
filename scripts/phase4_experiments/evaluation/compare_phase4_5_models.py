"""Единое сравнение моделей Этапа 4.5 на SimOZZ и real OZZ.

Скрипт запускает:
1) evaluate_sim_ozz.py (симулированные данные)
2) collect_real_ozz_statistics.py (реальные осциллограммы)

Для каждой модели поддерживается выбор чекпоинта:
- latest_checkpoint.pt
- best_model.pt
- оба варианта

Примеры:
  python scripts/phase4_experiments/evaluation/compare_phase4_5_models.py --mode latest
  python scripts/phase4_experiments/evaluation/compare_phase4_5_models.py --mode both --sim-per-class-files 240
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase4_experiments.sim_ozz.evaluate_sim_ozz import evaluate_sim_ozz
from scripts.phase4_experiments.real_ozz.collect_real_ozz_statistics import (
    CLASS_NAMES as REAL_CLASS_NAMES,
    collect_statistics,
    finalize_real_ozz_from_disk,
)
from scripts.phase4_experiments.threshold_utils import (
    resolve_threshold_config,
    threshold_label,
    threshold_metadata,
)


def _resolve_run_dir(path_str: str) -> Path:
    run_dir = Path(path_str)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f'Директория эксперимента не найдена: {run_dir}')
    return run_dir


def _resolve_checkpoints(run_dir: Path, mode: str) -> list[Path]:
    latest = run_dir / 'latest_checkpoint.pt'
    best = run_dir / 'best_model.pt'

    if mode == 'latest':
        return [latest] if latest.exists() else []
    if mode == 'best':
        return [best] if best.exists() else []
    if mode == 'both':
        out: list[Path] = []
        if latest.exists():
            out.append(latest)
        if best.exists():
            out.append(best)
        return out
    raise ValueError(f'Неизвестный mode: {mode}')


def _safe_name(name: str) -> str:
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


MODEL_LABELS = {
    'physical_kan': 'Physical KAN',
    'spectral_baseline': 'Spectral baseline',
    'physical_mlp': 'Physical MLP',
    'raw_instantaneous': 'Raw instantaneous',
}


def _plot_comparison_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Сводные bar-chart по 4 моделям для раздела статьи."""
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('matplotlib не найден — сводные графики сравнения пропущены')
        return

    labels = [MODEL_LABELS.get(r['model_key'], r['model_key']) for r in rows]
    x = np.arange(len(labels))
    colors = ['#8E44AD', '#2980B9', '#27AE60', '#E67E22']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sim_f1 = [r.get('sim_macro_f1', 0) * 100 for r in rows]
    axes[0].bar(x, sim_f1, color=colors[:len(rows)], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    axes[0].set_ylabel('Macro-F1 (%)')
    axes[0].set_title('SimOZZ (960 val-файлов)')
    axes[0].set_ylim(min(sim_f1) - 1, 100)
    axes[0].grid(True, axis='y', alpha=0.3)

    conf_dr = [r.get('real_confirmed_detection_rate_pct', 0) for r in rows]
    false_dr = [r.get('real_false_detection_rate_pct', 0) for r in rows]
    threshold_labels = sorted({str(r.get('real_threshold_label', 'fixed=0.500')) for r in rows})
    real_title = threshold_labels[0] if len(threshold_labels) == 1 else 'mixed thresholds'
    w = 0.35
    axes[1].bar(x - w / 2, conf_dr, w, label='Confirmed OZZ', color='#2ecc71', alpha=0.85)
    axes[1].bar(x + w / 2, false_dr, w, label='False detection (FPR proxy)', color='#e74c3c', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    axes[1].set_ylabel('Detection rate (%)')
    axes[1].set_title(f'Real OZZ ({real_title})')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis='y', alpha=0.3)

    latency = [r.get('sim_latency_ms', 0) for r in rows]
    axes[2].bar(x, latency, color=colors[:len(rows)], alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    axes[2].set_ylabel('Latency (ms/sample)')
    axes[2].set_title('Inference latency')
    axes[2].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Сравнение 4 архитектур (Phase 4.5)', fontsize=13)
    fig.tight_layout()
    out_path = out_dir / 'comparison_summary_charts.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Сводный график: {out_path}')


def _ensure_real_ozz_sidecars(
    real_out: Path,
    real_threshold: float | dict[str, float],
    thresholds_json: str | None = None,
) -> None:
    """CSV и PNG, если inference уже был, но sidecar-файлы не создавались."""
    per_file = real_out / 'per_file_statistics.json'
    if not per_file.exists():
        return
    need_csv = not (real_out / 'predictions_per_file.csv').exists()
    need_png = not (real_out / 'confidence_distributions.png').exists()
    if need_csv or need_png:
        print(f'[finalize] Догенерация CSV/графиков: {real_out}')
        finalize_real_ozz_from_disk(
            real_out,
            threshold=real_threshold,
            thresholds_json=thresholds_json,
        )


def _cached_thresholds_match(
    real_agg: dict[str, Any],
    requested_config: dict[str, Any],
) -> bool:
    cached_meta = real_agg.get('thresholding')
    if not isinstance(cached_meta, dict):
        return False

    cached_mode = cached_meta.get('mode', 'fixed')
    requested_mode = requested_config.get('mode', 'fixed')
    if cached_mode != requested_mode:
        return False

    if requested_mode == 'per_class':
        cached_thresholds = cached_meta.get('per_class_thresholds') or {}
        requested_thresholds = requested_config.get('per_class_thresholds') or {}
        for class_name in REAL_CLASS_NAMES:
            if abs(float(cached_thresholds.get(class_name, -1.0)) - float(requested_thresholds.get(class_name, -2.0))) > 1e-9:
                return False
        return True

    return abs(
        float(cached_meta.get('default_threshold', 0.5))
        - float(requested_config.get('default_threshold', 0.5))
    ) <= 1e-9


def compare_models(
    model_runs: dict[str, str],
    mode: str,
    sim_per_class_files: int,
    sim_batch_size: int,
    sim_num_workers: int,
    real_threshold: float,
    real_threshold_mode: str,
    real_thresholds_json: str | None,
    real_max_files: int | None,
    run_real_eval: bool,
    output_root: str | None,
    continue_from: str | None = None,
) -> dict[str, Any]:
    if continue_from is not None:
        out_dir = Path(continue_from)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        print(f'Продолжаем предыдущий запуск: {out_dir}')
    else:
        if output_root is None:
            base_out = PROJECT_ROOT / 'reports' / 'phase4' / 'model_comparison'
        else:
            base_out = Path(output_root)
            if not base_out.is_absolute():
                base_out = PROJECT_ROOT / base_out
        base_out.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = base_out / f'compare_4models_{mode}_{run_stamp}'

    sim_out_root = out_dir / 'sim_ozz_eval'
    real_out_root = out_dir / 'real_ozz_statistics'
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_out_root.mkdir(parents=True, exist_ok=True)
    if run_real_eval:
        real_out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for model_key, run_dir_str in model_runs.items():
        run_dir = _resolve_run_dir(run_dir_str)
        checkpoints = _resolve_checkpoints(run_dir, mode)
        if not checkpoints:
            print(f'[SKIP] {model_key}: нет чекпоинтов под mode={mode} в {run_dir}')
            continue

        for ckpt in checkpoints:
            ckpt_tag = ckpt.stem  # latest_checkpoint | best_model
            # Используем краткий тег, чтобы полный путь не превышал 260 символов (Windows MAX_PATH)
            item_tag = _safe_name(f'{model_key}_{ckpt_tag}')
            print('\n' + '=' * 90)
            print(f'Модель: {model_key}')
            print(f'Run dir: {run_dir}')
            print(f'Checkpoint: {ckpt.name}')
            print('=' * 90)

            sim_out = sim_out_root / item_tag
            sim_result_file = sim_out / 'sim_ozz_evaluation.json'
            if sim_result_file.exists():
                print(f'[SKIP SimOZZ] {model_key}/{ckpt_tag} — уже посчитано: {sim_result_file}')
                with open(sim_result_file, encoding='utf-8') as _f:
                    _cached = json.load(_f)
                sim_metrics = _cached.get('metrics_window_level', {})
            else:
                sim_metrics = evaluate_sim_ozz(
                    checkpoint_path=str(ckpt),
                    per_class_files=sim_per_class_files,
                    batch_size=sim_batch_size,
                    num_workers=sim_num_workers,
                    save_plots=True,
                    include_roc=False,
                    output_dir=str(sim_out),
                )

            row: dict[str, Any] = {
                'model_key': model_key,
                'run_dir': str(run_dir),
                'checkpoint_type': ckpt_tag,
                'checkpoint_path': str(ckpt),
                'sim_macro_f1': float(sim_metrics.get('macro_f1', 0.0)),
                'sim_macro_auc': float(sim_metrics.get('roc_auc_macro', 0.0)),
                'sim_exact_match': float(sim_metrics.get('exact_match', 0.0)),
                'sim_latency_ms': float(sim_metrics.get('latency_ms', 0.0)),
                'sim_report_dir': str(sim_out),
            }

            if run_real_eval:
                if real_threshold_mode == 'sim_optimal':
                    real_threshold_config = resolve_threshold_config(
                        REAL_CLASS_NAMES,
                        threshold=real_threshold,
                        per_class_thresholds=(
                            sim_metrics.get('optimal_thresholds', {}) or {}
                        ).get('thresholds'),
                        source_label='sim_optimal',
                    )
                    if real_threshold_config.get('per_class_thresholds') is None:
                        raise ValueError(
                            f'Для {model_key}/{ckpt_tag} не найдены optimal_thresholds в SimOZZ-отчёте.'
                        )
                    thresholds_json_effective = None
                elif real_threshold_mode == 'json':
                    if real_thresholds_json is None:
                        raise ValueError('Режим real_threshold_mode=json требует real_thresholds_json')
                    real_threshold_config = resolve_threshold_config(
                        REAL_CLASS_NAMES,
                        threshold=real_threshold,
                        thresholds_json=real_thresholds_json,
                    )
                    thresholds_json_effective = real_thresholds_json
                else:
                    real_threshold_config = resolve_threshold_config(
                        REAL_CLASS_NAMES,
                        threshold=real_threshold,
                        source_label='fixed',
                    )
                    thresholds_json_effective = None

                real_out = real_out_root / item_tag
                real_result_file = real_out / 'real_ozz_statistics.json'
                if real_result_file.exists():
                    with open(real_result_file, encoding='utf-8') as _f:
                        real_agg = json.load(_f)
                    if _cached_thresholds_match(real_agg, threshold_metadata(real_threshold_config)):
                        print(f'[SKIP real OZZ] {model_key}/{ckpt_tag} — уже посчитано: {real_result_file}')
                    else:
                        print(f'[RECALC real OZZ] {model_key}/{ckpt_tag} — уставки изменились, пересчёт')
                        real_agg = collect_statistics(
                            checkpoint_path=str(ckpt),
                            max_files=real_max_files,
                            threshold=real_threshold_config['threshold_spec'],
                            thresholds_json=thresholds_json_effective,
                            output_dir=str(real_out),
                        )
                else:
                    real_agg = collect_statistics(
                        checkpoint_path=str(ckpt),
                        max_files=real_max_files,
                        threshold=real_threshold_config['threshold_spec'],
                        thresholds_json=thresholds_json_effective,
                        output_dir=str(real_out),
                    )
                confirmed = real_agg.get('confirmed', {})
                false_det = real_agg.get('false_detection', {})
                row.update({
                    'real_confirmed_count': int(confirmed.get('count', 0)),
                    'real_confirmed_detection_rate_pct': float(confirmed.get('detection_rate_pct', 0.0)),
                    'real_false_count': int(false_det.get('count', 0)),
                    'real_false_detection_rate_pct': float(false_det.get('detection_rate_pct', 0.0)),
                    'real_threshold_mode': real_threshold_config['mode'],
                    'real_threshold_label': threshold_label(real_threshold_config),
                    'real_report_dir': str(real_out),
                })
                _ensure_real_ozz_sidecars(
                    real_out,
                    real_threshold_config['threshold_spec'],
                    thresholds_json=thresholds_json_effective,
                )

            rows.append(row)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'sim_per_class_files': sim_per_class_files,
        'run_real_eval': run_real_eval,
        'real_threshold': real_threshold,
        'real_threshold_mode': real_threshold_mode,
        'real_thresholds_json': real_thresholds_json,
        'real_max_files': real_max_files,
        'items': rows,
    }

    summary_json = out_dir / 'comparison_summary.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if rows:
        df = pl.DataFrame(rows)
        summary_csv = out_dir / 'comparison_summary.csv'
        df.write_csv(summary_csv)
        print(f'\nCSV: {summary_csv}')
        _plot_comparison_summary(rows, out_dir)

    print(f'JSON: {summary_json}')
    print(f'Папка сравнения: {out_dir}')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Сравнение 4 моделей Этапа 4.5 (SimOZZ + real OZZ)')
    parser.add_argument('--mode', choices=['latest', 'best', 'both'], default='latest',
                        help='Какие чекпоинты сравнивать для каждой модели')
    parser.add_argument('--sim-per-class-files', type=int, default=240,
                        help='Файлов SimOZZ на каждый из 4 классов (итого 4*N)')
    parser.add_argument('--sim-batch-size', type=int, default=128)
    parser.add_argument('--sim-num-workers', type=int, default=4)
    parser.add_argument('--real-threshold', type=float, default=0.5)
    parser.add_argument('--real-threshold-mode', choices=['sim_optimal', 'fixed', 'json'],
                        default='sim_optimal',
                        help='Источник уставок для real OZZ: optimal с SimOZZ, fixed или внешний JSON')
    parser.add_argument('--real-thresholds-json', type=str, default=None,
                        help='JSON с per-class уставками для режима --real-threshold-mode json')
    parser.add_argument('--real-max-files', type=int, default=None)
    parser.add_argument('--no-real', action='store_true',
                        help='Не запускать real OZZ статистику')
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Продолжить прерванный запуск: путь к существующей папке сравнения')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_runs = {
        # Папки экспериментов (НЕ путь к .pt). Скрипт сам возьмёт latest/best.
        'physical_kan': 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260616_072711',
        'spectral_baseline': 'experiments/phase4/sim_ozz_finetune_BaselineTransformer_20260607_104508',
        'physical_mlp': 'experiments/phase4/sim_ozz_finetune_PhysicalMLPTransformer_20260608_042419',
        'raw_instantaneous': 'experiments/phase4/sim_ozz_finetune_BaselineTransformer_20260608_202845',
    }

    compare_models(
        model_runs=model_runs,
        mode=args.mode,
        sim_per_class_files=args.sim_per_class_files,
        sim_batch_size=args.sim_batch_size,
        sim_num_workers=args.sim_num_workers,
        real_threshold=args.real_threshold,
        real_threshold_mode=args.real_threshold_mode,
        real_thresholds_json=args.real_thresholds_json,
        real_max_files=args.real_max_files,
        run_real_eval=not args.no_real,
        output_root=args.output_root,
        continue_from=args.continue_from,
    )


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) > 1:
        main()
    else:
        # =====================================================
        # РУЧНОЙ РЕЖИМ — отредактируйте константы ниже
        # =====================================================
        MODE = 'latest'  # 'latest' | 'best' | 'both'
        SIM_PER_CLASS_FILES = 240  # 240 на класс = 960 файлов (4 класса)
        SIM_BATCH_SIZE = 256
        SIM_NUM_WORKERS = 4

        RUN_REAL_EVAL = True
        REAL_THRESHOLD = 0.5
        REAL_THRESHOLD_MODE = 'sim_optimal'  # 'sim_optimal' | 'fixed' | 'json'
        REAL_THRESHOLDS_JSON = None
        REAL_MAX_FILES = None  # None = полный набор

        OUTPUT_ROOT = None
        # Укажите путь к существующей папке для продолжения прерванного запуска:
        CONTINUE_FROM = 'reports/phase4/model_comparison/compare_4models_latest_20260610_204942'
        # CONTINUE_FROM = None

        MODEL_RUN_DIRS = {
            'physical_kan': 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260616_072711',
            'spectral_baseline': 'experiments/phase4/sim_ozz_finetune_spectral_baseline_BaselineTransformer_20260612_025508',
            'physical_mlp': 'experiments/phase4/sim_ozz_finetune_physical_mlp_PhysicalMLPTransformer_20260612_204349',
            'raw_instantaneous': 'experiments/phase4/sim_ozz_finetune_raw_instantaneous_BaselineTransformer_20260613_140322',
        }

        compare_models(
            model_runs=MODEL_RUN_DIRS,
            mode=MODE,
            sim_per_class_files=SIM_PER_CLASS_FILES,
            sim_batch_size=SIM_BATCH_SIZE,
            sim_num_workers=SIM_NUM_WORKERS,
            real_threshold=REAL_THRESHOLD,
            real_threshold_mode=REAL_THRESHOLD_MODE,
            real_thresholds_json=REAL_THRESHOLDS_JSON,
            real_max_files=REAL_MAX_FILES,
            run_real_eval=RUN_REAL_EVAL,
            output_root=OUTPUT_ROOT,
            continue_from=CONTINUE_FROM,
        )
