import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).parent.parent.parent
REPORTS_DIR = ROOT_DIR / 'reports' / 'phase3'
FIGURES_DIR = REPORTS_DIR / 'figures'

SUMMARY_SCRIPT_PATH = ROOT_DIR / 'scripts' / 'phase3_experiments' / 'summarize_phase3_results.py'
_summary_spec = importlib.util.spec_from_file_location('summarize_phase3_results', SUMMARY_SCRIPT_PATH)
if _summary_spec is None or _summary_spec.loader is None:
    raise ImportError(f'Не удалось загрузить summary-модуль Фазы 3 из {SUMMARY_SCRIPT_PATH}')
_summary_module = importlib.util.module_from_spec(_summary_spec)
_summary_spec.loader.exec_module(_summary_module)
run_phase3_summary = _summary_module.run_phase3_summary


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write('')
        return

    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _combine_by_backend(
    ranking_rows: List[Dict[str, Any]],
    benchmark_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rank_map: Dict[str, Dict[str, Any]] = {}
    for row in ranking_rows:
        backend = str(row.get('backend', '')).strip()
        if backend:
            rank_map[backend] = row

    bench_map: Dict[str, Dict[str, Any]] = {}
    for row in benchmark_rows:
        backend = str(row.get('backend', '')).strip()
        if backend:
            bench_map[backend] = row

    all_backends = sorted(set(rank_map.keys()) | set(bench_map.keys()))

    combined: List[Dict[str, Any]] = []
    for backend in all_backends:
        r = rank_map.get(backend, {})
        b = bench_map.get(backend, {})
        combined.append(
            {
                'backend': backend,
                'rank_by_best_val_f1': r.get('rank_by_best_val_f1'),
                'best_experiment_name': r.get('best_experiment_name'),
                'best_val_f1': r.get('best_val_f1'),
                'final_val_f1': r.get('final_val_f1'),
                'best_val_loss': r.get('best_val_loss'),
                'final_val_loss': r.get('final_val_loss'),
                'epochs_done': r.get('epochs_done'),
                'mean_epoch_time_s': r.get('mean_epoch_time_s'),
                'total_train_time_s': r.get('total_train_time_s'),
                'benchmark_status': b.get('status'),
                'benchmark_error': b.get('error'),
                'train_step_ms_mean': b.get('train_step_ms_mean'),
                'train_step_ms_std': b.get('train_step_ms_std'),
                'val_loss_mean_bench': b.get('val_loss_mean'),
                'inference_latency_ms': b.get('inference_latency_ms'),
                'inference_fps': b.get('inference_fps'),
                'max_vram_mb': b.get('max_vram_mb'),
                'rss_mb': b.get('rss_mb'),
                'num_params': b.get('num_params'),
                'benchmark_run_name': b.get('run_name'),
                'benchmark_timestamp': b.get('timestamp'),
            }
        )

    combined.sort(
        key=lambda row: (
            _safe_float(row.get('rank_by_best_val_f1')) is None,
            _safe_float(row.get('rank_by_best_val_f1')) or 10**9,
        )
    )
    return combined


def _plot_if_possible(combined_rows: List[Dict[str, Any]], out_prefix: str) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('[Phase3 Aggregate] matplotlib не найден, графики пропущены.')
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    backends = [str(r.get('backend', 'unknown')) for r in combined_rows]
    if not backends:
        return []

    generated: List[Path] = []

    # 1) Качество (best/final F1)
    best_f1 = [_safe_float(r.get('best_val_f1')) for r in combined_rows]
    final_f1 = [_safe_float(r.get('final_val_f1')) for r in combined_rows]

    fig1 = FIGURES_DIR / f'{out_prefix}_quality_f1.png'
    plt.figure(figsize=(12, 5))
    x = list(range(len(backends)))
    width = 0.38
    plt.bar([i - width / 2 for i in x], [v if v is not None else 0.0 for v in best_f1], width=width, label='best_val_f1')
    plt.bar([i + width / 2 for i in x], [v if v is not None else 0.0 for v in final_f1], width=width, label='final_val_f1')
    plt.xticks(x, backends, rotation=30, ha='right')
    plt.ylabel('F1')
    plt.title('Фаза 3: сравнение качества backend-ов')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=140)
    plt.close()
    generated.append(fig1)

    # 2) Скорость инференса
    latency_ms = [_safe_float(r.get('inference_latency_ms')) for r in combined_rows]
    fps = [_safe_float(r.get('inference_fps')) for r in combined_rows]

    fig2 = FIGURES_DIR / f'{out_prefix}_inference_speed.png'
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.bar(x, [v if v is not None else 0.0 for v in latency_ms], alpha=0.7, label='latency_ms', color='#1f77b4')
    ax2.plot(x, [v if v is not None else 0.0 for v in fps], marker='o', label='fps', color='#d62728')

    ax1.set_xticks(x)
    ax1.set_xticklabels(backends, rotation=30, ha='right')
    ax1.set_ylabel('Latency (ms)')
    ax2.set_ylabel('FPS')
    ax1.set_title('Фаза 3: инференс (latency / fps)')
    fig.tight_layout()
    fig.savefig(fig2, dpi=140)
    plt.close(fig)
    generated.append(fig2)

    # 3) Память
    max_vram = [_safe_float(r.get('max_vram_mb')) for r in combined_rows]
    rss_mb = [_safe_float(r.get('rss_mb')) for r in combined_rows]

    fig3 = FIGURES_DIR / f'{out_prefix}_memory.png'
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], [v if v is not None else 0.0 for v in max_vram], width=width, label='max_vram_mb')
    plt.bar([i + width / 2 for i in x], [v if v is not None else 0.0 for v in rss_mb], width=width, label='rss_mb')
    plt.xticks(x, backends, rotation=30, ha='right')
    plt.ylabel('MB')
    plt.title('Фаза 3: сравнение памяти')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig3, dpi=140)
    plt.close()
    generated.append(fig3)

    # 4) Скорость обучения
    train_step = [_safe_float(r.get('train_step_ms_mean')) for r in combined_rows]
    epoch_time = [_safe_float(r.get('mean_epoch_time_s')) for r in combined_rows]

    fig4 = FIGURES_DIR / f'{out_prefix}_train_speed.png'
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(x, [v if v is not None else 0.0 for v in train_step], alpha=0.7, label='train_step_ms', color='#2ca02c')
    ax2.plot(x, [v if v is not None else 0.0 for v in epoch_time], marker='o', label='mean_epoch_time_s', color='#ff7f0e')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backends, rotation=30, ha='right')
    ax1.set_ylabel('Train step (ms)')
    ax2.set_ylabel('Epoch time (s)')
    ax1.set_title('Фаза 3: скорость обучения')
    fig.tight_layout()
    fig.savefig(fig4, dpi=140)
    plt.close(fig)
    generated.append(fig4)

    return generated


def run_phase3_aggregate(run_name_filter: Optional[str] = None) -> Dict[str, Path]:
    """Строит расширенный агрегированный отчёт Фазы 3 (CSV/JSON + графики)."""
    summary_files = run_phase3_summary(run_name_filter=run_name_filter)

    ranking_rows = _read_csv_rows(summary_files['backend_ranking_csv'])
    benchmark_rows = _read_csv_rows(summary_files['benchmark_latest_csv'])
    combined_rows = _combine_by_backend(ranking_rows=ranking_rows, benchmark_rows=benchmark_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = run_name_filter if run_name_filter else 'all'
    out_prefix = f'phase3_aggregate_{suffix}_{ts}'

    combined_csv = REPORTS_DIR / f'{out_prefix}.csv'
    combined_json = REPORTS_DIR / f'{out_prefix}.json'

    _write_csv(combined_csv, combined_rows)
    with open(combined_json, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'generated_at': ts,
                'run_name_filter': run_name_filter,
                'summary_files': {k: str(v) for k, v in summary_files.items()},
                'combined_rows_count': len(combined_rows),
                'rows': combined_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_files = _plot_if_possible(combined_rows, out_prefix=out_prefix)

    print(f'[Phase3 Aggregate] combined csv: {combined_csv}')
    print(f'[Phase3 Aggregate] combined json: {combined_json}')
    for p in plot_files:
        print(f'[Phase3 Aggregate] figure: {p}')

    return {
        'combined_csv': combined_csv,
        'combined_json': combined_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Фаза 3: расширенная агрегация отчётов и графики')
    parser.add_argument('--run-name', type=str, default='', help='Фильтр run_name (из exp-name)')
    args = parser.parse_args()

    run_name_filter = args.run_name.strip() or None
    run_phase3_aggregate(run_name_filter=run_name_filter)


if __name__ == '__main__':
    main()
