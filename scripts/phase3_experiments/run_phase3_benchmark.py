import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import torch

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.data_management import DatasetManager
from osc_tools.ml.benchmark import InferenceBenchmark
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment
from osc_tools.ml.models import PhysicsKANConditional
from osc_tools.ml.precomputed_dataset import PrecomputedDataset


PHASE3_PRECOMPUTED_CSV = 'test_precomputed_phase3.csv'
PHASE3_FEATURE_MODE = 'phase_polar_h1_angle'


def _try_get_rss_mb() -> float | None:
    """Возвращает RSS памяти процесса в MB (если доступен psutil)."""
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _needs_precomputed_regen(dm: DatasetManager, data_dir: Path, num_harmonics: int) -> bool:
    """Проверяет, нужно ли пересоздать предрассчитанный датасет Фазы 3."""
    precomputed_path = data_dir / PHASE3_PRECOMPUTED_CSV
    if not precomputed_path.exists():
        return True

    try:
        header_cols = set(pl.read_csv(precomputed_path, n_rows=1, infer_schema_length=0).columns)
    except Exception:
        return True

    required_cols = set(dm.get_precomputed_feature_columns(PHASE3_FEATURE_MODE, num_harmonics=num_harmonics))
    return not required_cols.issubset(header_cols)


def _prepare_dataloaders(
    window_size: int,
    samples_per_file: int,
    val_index_stride: int,
    num_harmonics: int,
    train_batch_size: int,
    val_batch_size: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, List[str]]:
    """Готовит train/val DataLoader для бенчмарка Фазы 3."""
    data_dir = ROOT_DIR / 'data' / 'ml_datasets'
    norm_coef_path = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    dm = DatasetManager(str(data_dir), norm_coef_path=str(norm_coef_path))
    dm.ensure_train_test_split()

    force_precompute = _needs_precomputed_regen(dm, data_dir, num_harmonics)
    dm.create_precomputed_test_csv(
        force=force_precompute,
        num_harmonics=num_harmonics,
        output_filename=PHASE3_PRECOMPUTED_CSV,
        phase_polar_h1_angle_only=True,
    )

    df = dm.load_train_df().with_row_index('row_nr')
    df = prepare_labels_for_experiment(df, 'base_sequential')

    train_indices = OscillogramDataset.create_indices(
        df,
        window_size=window_size,
        mode='train',
        samples_per_file=samples_per_file,
    )

    test_df = dm.load_test_df(precomputed=True, precomputed_filename=PHASE3_PRECOMPUTED_CSV).with_row_index('row_nr')
    test_df = prepare_labels_for_experiment(test_df, 'base_sequential')

    val_indices = PrecomputedDataset.create_indices(
        test_df,
        window_size=window_size,
        mode='val',
        stride=val_index_stride,
    )

    target_cols = get_target_columns('base_sequential', df)

    train_ds = OscillogramDataset(
        dataframe=df,
        indices=train_indices,
        window_size=window_size,
        mode='classification',
        feature_mode=PHASE3_FEATURE_MODE,
        sampling_strategy='stride',
        downsampling_stride=16,
        target_columns=target_cols,
        target_level='base_sequential',
        target_window_mode='point',
        physical_normalization=True,
        norm_coef_path=str(norm_coef_path),
        augment=True,
        num_harmonics=num_harmonics,
    )

    val_ds = PrecomputedDataset(
        dataframe=test_df,
        indices=val_indices,
        window_size=window_size,
        feature_mode=PHASE3_FEATURE_MODE,
        target_columns=target_cols,
        target_level='base_sequential',
        sampling_strategy='stride',
        downsampling_stride=16,
        num_harmonics=num_harmonics,
        target_window_mode='point',
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

    sample_x, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])

    return train_loader, val_loader, in_channels, target_cols


def _run_backend_benchmark(
    backend: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    in_channels: int,
    num_classes: int,
    train_steps: int,
    val_steps: int,
    device: torch.device,
    learning_rate: float,
) -> Dict[str, Any]:
    """Выполняет короткий train+eval benchmark для одного backend."""
    result: Dict[str, Any] = {
        'backend': backend,
        'device': str(device),
        'status': 'ok',
        'error': '',
    }

    try:
        model = PhysicsKANConditional(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=[32, 64, 128],
            dropout=0.3,
            grid_size=8,
            kan_backend=backend,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        step_times: List[float] = []
        train_losses: List[float] = []

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        for step_idx, (x, y) in enumerate(train_loader):
            if step_idx >= train_steps:
                break

            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_t = time.perf_counter()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_t = time.perf_counter()

            step_times.append(end_t - start_t)
            train_losses.append(float(loss.item()))

        val_losses: List[float] = []
        model.eval()
        with torch.no_grad():
            for step_idx, (x, y) in enumerate(val_loader):
                if step_idx >= val_steps:
                    break
                x = x.to(device)
                y = y.to(device).float()
                out = model(x)
                loss = criterion(out, y)
                val_losses.append(float(loss.item()))

        inf_bench = InferenceBenchmark(device=str(device))
        input_shape = tuple(int(v) for v in next(iter(train_loader))[0][0].shape)
        inf_metrics = inf_bench.benchmark(model, input_shape=input_shape, num_runs=30, warmup=5, batch_size=1)

        max_vram_mb = None
        if device.type == 'cuda':
            max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        rss_mb = _try_get_rss_mb()

        result.update(
            {
                'train_steps_done': len(step_times),
                'train_step_ms_mean': (statistics.mean(step_times) * 1000.0) if step_times else None,
                'train_step_ms_std': (statistics.pstdev(step_times) * 1000.0) if len(step_times) > 1 else 0.0,
                'train_loss_mean': statistics.mean(train_losses) if train_losses else None,
                'train_loss_std': statistics.pstdev(train_losses) if len(train_losses) > 1 else 0.0,
                'val_steps_done': len(val_losses),
                'val_loss_mean': statistics.mean(val_losses) if val_losses else None,
                'max_vram_mb': max_vram_mb,
                'rss_mb': rss_mb,
                'num_params': inf_metrics.get('num_params') if isinstance(inf_metrics, dict) else None,
                'inference_latency_ms': inf_metrics.get('latency_ms') if isinstance(inf_metrics, dict) else None,
                'inference_fps': inf_metrics.get('fps') if isinstance(inf_metrics, dict) else None,
            }
        )

    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)

    return result


def _save_results(results: List[Dict[str, Any]], report_dir: Path, run_name: str) -> None:
    """Сохраняет результаты benchmark в JSON и CSV."""
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = report_dir / f'phase3_benchmark_{run_name}_{ts}.json'
    csv_path = report_dir / f'phase3_benchmark_{run_name}_{ts}.csv'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f'Результаты сохранены: {json_path}')
    print(f'Результаты сохранены: {csv_path}')


def run_phase3_benchmark(
    run_name: str,
    backends: List[str],
    num_harmonics: int = 9,
    samples_per_file: int = 4,
    val_index_stride: int = 16,
    window_size: int = 320,
    train_batch_size: int = 32,
    val_batch_size: int = 256,
    train_steps: int = 20,
    val_steps: int = 10,
    learning_rate: float = 0.001,
    device_name: str = 'auto',
) -> List[Dict[str, Any]]:
    """Публичная точка запуска benchmark Фазы 3 для внешнего вызова."""
    if num_harmonics < 1:
        raise ValueError('num_harmonics должен быть >= 1')

    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    if not backends:
        raise ValueError('Не переданы backend-ы для benchmark')

    print('Подготовка DataLoader-ов для benchmark...')
    train_loader, val_loader, in_channels, target_cols = _prepare_dataloaders(
        window_size=window_size,
        samples_per_file=samples_per_file,
        val_index_stride=val_index_stride,
        num_harmonics=num_harmonics,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )

    print(f'Устройство: {device}')
    print(f'Feature mode: {PHASE3_FEATURE_MODE}')
    print(f'Входных каналов: {in_channels}, классов: {len(target_cols)}')

    results: List[Dict[str, Any]] = []
    for backend in backends:
        print('-' * 80)
        print(f'Benchmark backend: {backend}')
        row = _run_backend_benchmark(
            backend=backend,
            train_loader=train_loader,
            val_loader=val_loader,
            in_channels=in_channels,
            num_classes=len(target_cols),
            train_steps=train_steps,
            val_steps=val_steps,
            device=device,
            learning_rate=learning_rate,
        )
        results.append(row)
        print(row)

    report_dir = ROOT_DIR / 'reports' / 'phase3'
    _save_results(results, report_dir=report_dir, run_name=run_name)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Фаза 3: benchmark backend-ов PhysicsKANConditional')
    parser.add_argument('--run-name', type=str, default='quick', help='Короткий идентификатор запуска benchmark')
    parser.add_argument('--backends', type=str, default='baseline,efficient,fast,cheby,wav,torch_wavelet', help='Список backend через запятую')
    parser.add_argument('--num-harmonics', type=int, default=9, help='Количество гармоник (по умолчанию 9)')
    parser.add_argument('--samples-per-file', type=int, default=4, help='Количество train окон на файл')
    parser.add_argument('--val-index-stride', type=int, default=16, help='Шаг окон для валид. индексов (больше = быстрее валидация)')
    parser.add_argument('--window-size', type=int, default=320, help='Размер окна')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Batch size для train benchmark')
    parser.add_argument('--val-batch-size', type=int, default=256, help='Batch size для val benchmark')
    parser.add_argument('--train-steps', type=int, default=20, help='Количество train шагов на backend')
    parser.add_argument('--val-steps', type=int, default=10, help='Количество val шагов на backend')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate для train benchmark')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Устройство для benchmark')
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(',') if b.strip()]
    run_phase3_benchmark(
        run_name=args.run_name,
        backends=backends,
        num_harmonics=args.num_harmonics,
        samples_per_file=args.samples_per_file,
        val_index_stride=args.val_index_stride,
        window_size=args.window_size,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        learning_rate=args.learning_rate,
        device_name=args.device,
    )


if __name__ == '__main__':
    main()
