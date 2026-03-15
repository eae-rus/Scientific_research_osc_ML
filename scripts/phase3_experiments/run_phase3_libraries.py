import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import polars as pl

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Переиспользуем устойчивую логику запуска из Фазы 2.6
PHASE2_SCRIPTS_DIR = ROOT_DIR / 'scripts' / 'phase2_experiments'
PHASE2_SCRIPT_PATH = PHASE2_SCRIPTS_DIR / 'run_phase2_6.py'

_phase2_spec = importlib.util.spec_from_file_location('run_phase2_6', PHASE2_SCRIPT_PATH)
if _phase2_spec is None or _phase2_spec.loader is None:
    raise ImportError(f'Не удалось загрузить модуль Фазы 2.6 из {PHASE2_SCRIPT_PATH}')
_phase2_module = importlib.util.module_from_spec(_phase2_spec)
_phase2_spec.loader.exec_module(_phase2_module)
run_single_experiment = _phase2_module.run_single_experiment

PHASE3_BENCH_SCRIPT_PATH = ROOT_DIR / 'scripts' / 'phase3_experiments' / 'run_phase3_benchmark.py'
_phase3_bench_spec = importlib.util.spec_from_file_location('run_phase3_benchmark', PHASE3_BENCH_SCRIPT_PATH)
if _phase3_bench_spec is None or _phase3_bench_spec.loader is None:
    raise ImportError(f'Не удалось загрузить benchmark-модуль Фазы 3 из {PHASE3_BENCH_SCRIPT_PATH}')
_phase3_bench_module = importlib.util.module_from_spec(_phase3_bench_spec)
_phase3_bench_spec.loader.exec_module(_phase3_bench_module)
run_phase3_benchmark = _phase3_bench_module.run_phase3_benchmark

PHASE3_SUMMARY_SCRIPT_PATH = ROOT_DIR / 'scripts' / 'phase3_experiments' / 'summarize_phase3_results.py'
_phase3_summary_spec = importlib.util.spec_from_file_location('summarize_phase3_results', PHASE3_SUMMARY_SCRIPT_PATH)
if _phase3_summary_spec is None or _phase3_summary_spec.loader is None:
    raise ImportError(f'Не удалось загрузить summary-модуль Фазы 3 из {PHASE3_SUMMARY_SCRIPT_PATH}')
_phase3_summary_module = importlib.util.module_from_spec(_phase3_summary_spec)
_phase3_summary_spec.loader.exec_module(_phase3_summary_module)
run_phase3_summary = _phase3_summary_module.run_phase3_summary

PHASE3_AGGREGATE_SCRIPT_PATH = ROOT_DIR / 'scripts' / 'phase3_experiments' / 'aggregate_phase3_reports.py'
_phase3_agg_spec = importlib.util.spec_from_file_location('aggregate_phase3_reports', PHASE3_AGGREGATE_SCRIPT_PATH)
if _phase3_agg_spec is None or _phase3_agg_spec.loader is None:
    raise ImportError(f'Не удалось загрузить агрегатор Фазы 3 из {PHASE3_AGGREGATE_SCRIPT_PATH}')
_phase3_agg_module = importlib.util.module_from_spec(_phase3_agg_spec)
_phase3_agg_spec.loader.exec_module(_phase3_agg_module)
run_phase3_aggregate = _phase3_agg_module.run_phase3_aggregate

from osc_tools.data_management import DatasetManager
from osc_tools.ml.config import DataConfig, TrainingConfig
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment
from osc_tools.ml.precomputed_dataset import PrecomputedDataset


PHASE3_PRECOMPUTED_CSV = 'test_precomputed_phase3.csv'
PHASE3_FEATURE_MODE = 'phase_polar_h1_angle'
ALLOWED_BACKENDS = ('baseline', 'efficient', 'fast', 'cheby', 'wav', 'torch_wavelet')

# Ручной режим (как в Фазе 2.6):
# Если скрипт запущен без CLI-аргументов, используются значения из этого блока.
PHASE3_MANUAL_CONFIG = {
    'mode': 'train',
    'exp_name': 'phase3_manual',
    'epochs': 60,
    'samples_per_file': 12,
    'val_index_stride': 16,
    'num_harmonics': 9,
    'kan_backend': 'baseline',
    'train_backends': 'baseline,efficient,fast,cheby,wav,torch_wavelet',
    'benchmark_backends': 'baseline,efficient,fast,cheby,wav,torch_wavelet',
    'benchmark_train_steps': 20,
    'benchmark_val_steps': 10,
    'benchmark_train_batch_size': 32,
    'benchmark_val_batch_size': 256,
    'benchmark_device': 'auto',
    'checkpoint_frequency': 0,
    'no_skip_existing': False,
    'no_summary': False,
    'no_aggregate': False,
}


def _manual_namespace() -> argparse.Namespace:
    """Возвращает argparse.Namespace на основе PHASE3_MANUAL_CONFIG."""
    return argparse.Namespace(
        mode=PHASE3_MANUAL_CONFIG['mode'],
        exp_name=PHASE3_MANUAL_CONFIG['exp_name'],
        epochs=PHASE3_MANUAL_CONFIG['epochs'],
        samples_per_file=PHASE3_MANUAL_CONFIG['samples_per_file'],
        val_index_stride=PHASE3_MANUAL_CONFIG['val_index_stride'],
        num_harmonics=PHASE3_MANUAL_CONFIG['num_harmonics'],
        kan_backend=PHASE3_MANUAL_CONFIG['kan_backend'],
        train_backends=PHASE3_MANUAL_CONFIG['train_backends'],
        benchmark_backends=PHASE3_MANUAL_CONFIG['benchmark_backends'],
        benchmark_train_steps=PHASE3_MANUAL_CONFIG['benchmark_train_steps'],
        benchmark_val_steps=PHASE3_MANUAL_CONFIG['benchmark_val_steps'],
        benchmark_train_batch_size=PHASE3_MANUAL_CONFIG['benchmark_train_batch_size'],
        benchmark_val_batch_size=PHASE3_MANUAL_CONFIG['benchmark_val_batch_size'],
        benchmark_device=PHASE3_MANUAL_CONFIG['benchmark_device'],
        checkpoint_frequency=PHASE3_MANUAL_CONFIG['checkpoint_frequency'],
        no_skip_existing=PHASE3_MANUAL_CONFIG['no_skip_existing'],
        no_summary=PHASE3_MANUAL_CONFIG['no_summary'],
        no_aggregate=PHASE3_MANUAL_CONFIG['no_aggregate'],
    )


def _needs_precomputed_regen(dm: DatasetManager, data_dir: Path, num_harmonics: int) -> bool:
    """Проверяет, нужно ли пересоздать предрассчитанный тестовый датасет."""
    precomputed_path = data_dir / PHASE3_PRECOMPUTED_CSV
    if not precomputed_path.exists():
        return True

    try:
        header_cols = set(pl.read_csv(precomputed_path, n_rows=1, infer_schema_length=0).columns)
    except Exception:
        return True

    required_modes = ['raw', PHASE3_FEATURE_MODE]
    required_cols: set[str] = set()
    for mode in required_modes:
        required_cols.update(dm.get_precomputed_feature_columns(mode, num_harmonics=num_harmonics))

    return not required_cols.issubset(header_cols)


def run_phase3_libraries(
    exp_name: str,
    epochs: int,
    checkpoint_frequency: int,
    samples_per_file: int,
    val_index_stride: int,
    num_harmonics: int,
    kan_backend: str,
    skip_existing: bool,
) -> None:
    """
    Запускает минимальный эксперимент Фазы 3.

    Фиксированные условия:
    - модель: PhysicsKANConditional
    - сложность: heavy
    - признаки: phase_polar
    - sampling: stride(16)
    - target_level: base_sequential
    """
    data_dir = ROOT_DIR / 'data' / 'ml_datasets'
    experiments_dir = ROOT_DIR / 'experiments' / 'phase3'
    experiments_dir.mkdir(parents=True, exist_ok=True)

    norm_coef_path = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    metadata_file = data_dir / 'train.csv'
    window_size = 320

    print('Инициализация DatasetManager...')
    dm = DatasetManager(str(data_dir), norm_coef_path=str(norm_coef_path))
    dm.ensure_train_test_split()

    force_precompute = _needs_precomputed_regen(dm, data_dir, num_harmonics)
    if force_precompute:
        print(f'[DatasetManager] Предрасчёт требует обновления — пересоздание {PHASE3_PRECOMPUTED_CSV}')
    dm.create_precomputed_test_csv(
        force=force_precompute,
        num_harmonics=num_harmonics,
        output_filename=PHASE3_PRECOMPUTED_CSV,
        phase_polar_h1_angle_only=True,
    )

    full_exp_name = (
        f"Exp_3.1.0_{exp_name}_PhysicsKANConditional_heavy_phase_polar_h1_angle_stride_"
        f"base_sequential_nh{num_harmonics}_kb{kan_backend}_aug"
    )
    checkpoint_path = experiments_dir / full_exp_name / 'final_model.pt'
    if skip_existing and checkpoint_path.exists():
        print(f'>>> Пропуск {full_exp_name} (уже обучено)')
        return

    print('Загрузка тренировочных данных...')
    df = dm.load_train_df().with_row_index('row_nr')
    df = prepare_labels_for_experiment(df, 'base_sequential')

    train_indices = OscillogramDataset.create_indices(
        df,
        window_size=window_size,
        mode='train',
        samples_per_file=samples_per_file,
    )

    print('Загрузка предрассчитанных тестовых данных...')
    test_df = dm.load_test_df(precomputed=True, precomputed_filename=PHASE3_PRECOMPUTED_CSV).with_row_index('row_nr')
    test_df = prepare_labels_for_experiment(test_df, 'base_sequential')

    val_indices = PrecomputedDataset.create_indices(
        test_df,
        window_size=window_size,
        mode='val',
        stride=val_index_stride,
    )

    target_cols = get_target_columns('base_sequential', df)
    print(f"  [Целевые колонки (base_sequential): {len(target_cols)} классов]")
    print(f"  [KAN backend: {kan_backend}]")

    learning_rate = 0.0003 if kan_backend == 'cheby' else 0.001
    if kan_backend == 'cheby':
        print(f"  [Стабилизация cheby: learning_rate={learning_rate}]")

    data_config = DataConfig(
        path=str(metadata_file),
        window_size=window_size,
        batch_size=64,
        mode='multilabel',
        norm_coef_path=str(norm_coef_path),
    )
    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        use_pos_weight=True,
        checkpoint_frequency=checkpoint_frequency,
        save_dir=str(experiments_dir),
    )

    run_single_experiment(
        exp_name=full_exp_name,
        model_name='PhysicsKANConditional',
        complexity='heavy',
        feature_mode=PHASE3_FEATURE_MODE,
        sampling_strategy='stride',
        downsampling_stride=16,
        df=df,
        train_indices=train_indices,
        val_indices=val_indices,
        target_cols=target_cols,
        data_config_base=data_config,
        train_config_base=train_config,
        norm_coef_path=norm_coef_path,
        augment=True,
        val_df=test_df,
        use_precomputed_val=True,
        balancing_mode='weights',
        balancer=None,
        num_harmonics=num_harmonics,
        target_level='base_sequential',
        target_window_mode='point',
        model_param_overrides={'kan_backend': kan_backend},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Фаза 3: единый запуск train/benchmark для KAN backend-ов')
    parser.add_argument('--mode', type=str, default=PHASE3_MANUAL_CONFIG['mode'], choices=['train', 'benchmark', 'both'], help='Режим запуска')
    parser.add_argument('--exp-name', type=str, default=PHASE3_MANUAL_CONFIG['exp_name'], help='Короткий идентификатор запуска')
    parser.add_argument('--epochs', type=int, default=PHASE3_MANUAL_CONFIG['epochs'], help='Количество эпох')
    parser.add_argument('--samples-per-file', type=int, default=PHASE3_MANUAL_CONFIG['samples_per_file'], help='Окон на файл для train')
    parser.add_argument('--val-index-stride', type=int, default=PHASE3_MANUAL_CONFIG['val_index_stride'], help='Шаг окон для валид. индексов (больше = быстрее валидация)')
    parser.add_argument(
        '--num-harmonics',
        type=int,
        default=PHASE3_MANUAL_CONFIG['num_harmonics'],
        help='Количество гармоник для Фазы 3 (по умолчанию 9; угол сохраняется только для 1-й гармоники)',
    )
    parser.add_argument(
        '--kan-backend',
        type=str,
        default=PHASE3_MANUAL_CONFIG['kan_backend'],
        choices=list(ALLOWED_BACKENDS),
        help='Backend для train-запуска PhysicsKANConditional',
    )
    parser.add_argument(
        '--train-backends',
        type=str,
        default=PHASE3_MANUAL_CONFIG['train_backends'],
        help='Список backend-ов для последовательного train-запуска через запятую (переопределяет --kan-backend)',
    )
    parser.add_argument(
        '--benchmark-backends',
        type=str,
        default=PHASE3_MANUAL_CONFIG['benchmark_backends'],
        help='Список backend для benchmark через запятую',
    )
    parser.add_argument('--benchmark-train-steps', type=int, default=PHASE3_MANUAL_CONFIG['benchmark_train_steps'], help='Число train шагов на backend в benchmark')
    parser.add_argument('--benchmark-val-steps', type=int, default=PHASE3_MANUAL_CONFIG['benchmark_val_steps'], help='Число val шагов на backend в benchmark')
    parser.add_argument('--benchmark-train-batch-size', type=int, default=PHASE3_MANUAL_CONFIG['benchmark_train_batch_size'], help='Train batch size в benchmark')
    parser.add_argument('--benchmark-val-batch-size', type=int, default=PHASE3_MANUAL_CONFIG['benchmark_val_batch_size'], help='Val batch size в benchmark')
    parser.add_argument('--benchmark-device', type=str, default=PHASE3_MANUAL_CONFIG['benchmark_device'], choices=['auto', 'cpu', 'cuda'], help='Устройство для benchmark')
    parser.add_argument('--checkpoint-frequency', type=int, default=PHASE3_MANUAL_CONFIG['checkpoint_frequency'], help='Частота сохранения чекпоинтов (<=0: epochs+1)')
    parser.add_argument('--no-skip-existing', action='store_true', help='Не пропускать уже обученные запуски')
    parser.add_argument('--no-summary', action='store_true', help='Не запускать авто-сводку результатов в конце')
    parser.add_argument('--no-aggregate', action='store_true', help='Не запускать расширенную агрегацию отчётов в конце')

    if len(sys.argv) == 1:
        args = _manual_namespace()
        print('[Phase3] Ручной режим: используются константы PHASE3_MANUAL_CONFIG (CLI аргументы не переданы).')
    else:
        args = parser.parse_args()

    if args.num_harmonics < 1:
        raise ValueError('num_harmonics должен быть >= 1')

    checkpoint_frequency = args.checkpoint_frequency
    if checkpoint_frequency <= 0:
        checkpoint_frequency = args.epochs + 1

    if args.mode in {'train', 'both'}:
        if args.train_backends.strip():
            train_backends = [b.strip() for b in args.train_backends.split(',') if b.strip()]
            invalid = [b for b in train_backends if b not in ALLOWED_BACKENDS]
            if invalid:
                raise ValueError(f'Неизвестные backend в --train-backends: {invalid}')
        else:
            train_backends = [args.kan_backend]

        for backend in train_backends:
            run_phase3_libraries(
                exp_name=args.exp_name,
                epochs=args.epochs,
                checkpoint_frequency=checkpoint_frequency,
                samples_per_file=args.samples_per_file,
                val_index_stride=args.val_index_stride,
                num_harmonics=args.num_harmonics,
                kan_backend=backend,
                skip_existing=not args.no_skip_existing,
            )

    if args.mode in {'benchmark', 'both'}:
        benchmark_backends = [b.strip() for b in args.benchmark_backends.split(',') if b.strip()]
        run_phase3_benchmark(
            run_name=args.exp_name,
            backends=benchmark_backends,
            num_harmonics=args.num_harmonics,
            samples_per_file=args.samples_per_file,
            val_index_stride=args.val_index_stride,
            window_size=320,
            train_batch_size=args.benchmark_train_batch_size,
            val_batch_size=args.benchmark_val_batch_size,
            train_steps=args.benchmark_train_steps,
            val_steps=args.benchmark_val_steps,
            learning_rate=0.001,
            device_name=args.benchmark_device,
        )

    if not args.no_summary:
        run_phase3_summary(run_name_filter=args.exp_name)

    if not args.no_aggregate:
        run_phase3_aggregate(run_name_filter=args.exp_name)


if __name__ == '__main__':
    main()
