import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'phase3'
REPORTS_DIR = ROOT_DIR / 'reports' / 'phase3'


def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_backend_from_name(exp_name: str) -> str:
    match = re.search(r'_kb(.+?)_aug$', exp_name)
    if match:
        return match.group(1)
    return 'unknown'


def _read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def collect_training_summary(run_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Собирает summary по обученным экспериментам Фазы 3."""
    if not EXPERIMENTS_DIR.exists():
        return []

    results: List[Dict[str, Any]] = []

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        if run_name_filter and run_name_filter not in exp_name:
            continue

        history_path = exp_dir / f'{exp_name}_history.json'
        if not history_path.exists():
            history_path = exp_dir / 'history.json'
        metrics_path = exp_dir / 'metrics.jsonl'
        config_path = exp_dir / 'config.json'

        history = _read_json(history_path)
        metrics_rows = _read_metrics_jsonl(metrics_path)
        config = _read_json(config_path)

        if not history and not metrics_rows:
            continue

        backend = _extract_backend_from_name(exp_name)

        train_loss_list = history.get('train_loss', []) if isinstance(history, dict) else []
        val_loss_list = history.get('val_loss', []) if isinstance(history, dict) else []
        val_f1_list = history.get('val_f1', []) if isinstance(history, dict) else []

        epochs_done = len(train_loss_list) if train_loss_list else len(metrics_rows)

        best_val_f1 = None
        best_val_f1_epoch = None
        if val_f1_list:
            best_idx = max(range(len(val_f1_list)), key=lambda i: val_f1_list[i])
            best_val_f1 = _safe_float(val_f1_list[best_idx])
            best_val_f1_epoch = best_idx + 1

        best_val_loss = None
        if val_loss_list:
            best_val_loss = _safe_float(min(val_loss_list))

        final_val_f1 = _safe_float(val_f1_list[-1]) if val_f1_list else None
        final_val_loss = _safe_float(val_loss_list[-1]) if val_loss_list else None

        epoch_times = [_safe_float(r.get('epoch_time')) for r in metrics_rows]
        epoch_times = [t for t in epoch_times if t is not None]

        total_train_time_s = float(sum(epoch_times)) if epoch_times else None
        mean_epoch_time_s = (float(sum(epoch_times) / len(epoch_times))) if epoch_times else None

        configured_epochs = None
        if isinstance(config, dict):
            configured_epochs = config.get('training', {}).get('epochs')

        results.append(
            {
                'experiment_name': exp_name,
                'backend': backend,
                'epochs_done': epochs_done,
                'configured_epochs': configured_epochs,
                'best_val_f1': best_val_f1,
                'best_val_f1_epoch': best_val_f1_epoch,
                'best_val_loss': best_val_loss,
                'final_val_f1': final_val_f1,
                'final_val_loss': final_val_loss,
                'mean_epoch_time_s': mean_epoch_time_s,
                'total_train_time_s': total_train_time_s,
                'has_final_model': (exp_dir / 'final_model.pt').exists(),
                'path': str(exp_dir),
            }
        )

    return results


def collect_latest_benchmark_summary(run_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Собирает последние benchmark-результаты по каждому backend из reports/phase3."""
    if not REPORTS_DIR.exists():
        return []

    latest_by_backend: Dict[str, Dict[str, Any]] = {}
    file_pattern = re.compile(r'^phase3_benchmark_(.+?)_(\d{8}_\d{6})\.json$')

    for file_path in sorted(REPORTS_DIR.glob('phase3_benchmark_*.json')):
        match = file_pattern.match(file_path.name)
        if not match:
            continue

        run_name = match.group(1)
        timestamp = match.group(2)

        if run_name_filter and run_name_filter != run_name:
            continue

        payload = _read_json(file_path)
        if not isinstance(payload, list):
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue
            backend = str(row.get('backend', 'unknown'))
            prev = latest_by_backend.get(backend)
            if prev is None or timestamp > str(prev.get('timestamp', '')):
                enriched = dict(row)
                enriched['run_name'] = run_name
                enriched['timestamp'] = timestamp
                latest_by_backend[backend] = enriched

    return list(latest_by_backend.values())


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_training_backend_ranking(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Агрегирует training-метрики по backend (лучший эксперимент каждого backend)."""
    best_by_backend: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        backend = str(row.get('backend', 'unknown'))
        current = best_by_backend.get(backend)
        candidate_score = row.get('best_val_f1')
        if current is None:
            best_by_backend[backend] = row
            continue

        current_score = current.get('best_val_f1')
        if candidate_score is None:
            continue
        if current_score is None or float(candidate_score) > float(current_score):
            best_by_backend[backend] = row

    ranking_rows: List[Dict[str, Any]] = []
    for backend, row in best_by_backend.items():
        ranking_rows.append(
            {
                'backend': backend,
                'best_experiment_name': row.get('experiment_name'),
                'best_val_f1': row.get('best_val_f1'),
                'best_val_loss': row.get('best_val_loss'),
                'final_val_f1': row.get('final_val_f1'),
                'final_val_loss': row.get('final_val_loss'),
                'mean_epoch_time_s': row.get('mean_epoch_time_s'),
                'total_train_time_s': row.get('total_train_time_s'),
                'epochs_done': row.get('epochs_done'),
            }
        )

    ranking_rows.sort(key=lambda r: (r.get('best_val_f1') is None, -(r.get('best_val_f1') or -1.0)))
    for idx, row in enumerate(ranking_rows, start=1):
        row['rank_by_best_val_f1'] = idx

    return ranking_rows


def run_phase3_summary(run_name_filter: Optional[str] = None) -> Dict[str, Path]:
    """Запускает сводный отчёт по training и benchmark результатам Фазы 3."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = collect_training_summary(run_name_filter=run_name_filter)
    benchmark_rows = collect_latest_benchmark_summary(run_name_filter=run_name_filter)
    rank_rows = _compute_training_backend_ranking(train_rows)

    suffix = run_name_filter if run_name_filter else 'all'
    train_csv = REPORTS_DIR / f'phase3_training_summary_{suffix}_{ts}.csv'
    benchmark_csv = REPORTS_DIR / f'phase3_benchmark_latest_{suffix}_{ts}.csv'
    rank_csv = REPORTS_DIR / f'phase3_backend_ranking_{suffix}_{ts}.csv'
    summary_json = REPORTS_DIR / f'phase3_summary_{suffix}_{ts}.json'

    _write_csv(train_csv, train_rows)
    _write_csv(benchmark_csv, benchmark_rows)
    _write_csv(rank_csv, rank_rows)

    summary_payload = {
        'generated_at': ts,
        'run_name_filter': run_name_filter,
        'training_runs_count': len(train_rows),
        'benchmark_backends_count': len(benchmark_rows),
        'files': {
            'training_summary_csv': str(train_csv),
            'benchmark_latest_csv': str(benchmark_csv),
            'backend_ranking_csv': str(rank_csv),
        },
    }

    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f'[Phase3 Summary] training: {train_csv}')
    print(f'[Phase3 Summary] benchmark: {benchmark_csv}')
    print(f'[Phase3 Summary] ranking: {rank_csv}')
    print(f'[Phase3 Summary] meta: {summary_json}')

    return {
        'training_summary_csv': train_csv,
        'benchmark_latest_csv': benchmark_csv,
        'backend_ranking_csv': rank_csv,
        'summary_json': summary_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Фаза 3: сводка результатов train + benchmark')
    parser.add_argument('--run-name', type=str, default='', help='Фильтр по run-name (из имени эксперимента)')
    args = parser.parse_args()

    run_name_filter = args.run_name.strip() or None
    run_phase3_summary(run_name_filter=run_name_filter)


if __name__ == '__main__':
    main()
