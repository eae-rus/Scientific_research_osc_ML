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

from osc_tools.data_management import DatasetManager
from osc_tools.ml.config import DataConfig, TrainingConfig
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment
from osc_tools.ml.precomputed_dataset import PrecomputedDataset


PHASE3_PRECOMPUTED_CSV = 'test_precomputed_phase3.csv'
PHASE3_FEATURE_MODE = 'phase_polar_h1_angle'


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
        stride=4,
    )

    target_cols = get_target_columns('base_sequential', df)
    print(f"  [Целевые колонки (base_sequential): {len(target_cols)} классов]")
    print(f"  [KAN backend: {kan_backend}]")

    data_config = DataConfig(
        path=str(metadata_file),
        window_size=window_size,
        batch_size=64,
        mode='multilabel',
        norm_coef_path=str(norm_coef_path),
    )
    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=0.001,
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
    parser = argparse.ArgumentParser(description='Фаза 3: минимальный запуск сравнения KAN библиотек')
    parser.add_argument('--exp-name', type=str, default='libraries_baseline', help='Короткий идентификатор запуска')
    parser.add_argument('--epochs', type=int, default=30, help='Количество эпох')
    parser.add_argument('--samples-per-file', type=int, default=12, help='Окон на файл для train')
    parser.add_argument(
        '--num-harmonics',
        type=int,
        default=9,
        help='Количество гармоник для Фазы 3 (по умолчанию 9; угол сохраняется только для 1-й гармоники)',
    )
    parser.add_argument(
        '--kan-backend',
        type=str,
        default='baseline',
        choices=['baseline', 'efficient'],
        help='Backend для KAN-голов PhysicsKANConditional',
    )
    parser.add_argument('--checkpoint-frequency', type=int, default=31, help='Частота сохранения чекпоинтов')
    parser.add_argument('--no-skip-existing', action='store_true', help='Не пропускать уже обученные запуски')
    args = parser.parse_args()

    if args.num_harmonics < 1:
        raise ValueError('num_harmonics должен быть >= 1')

    checkpoint_frequency = args.checkpoint_frequency
    if checkpoint_frequency <= 0:
        checkpoint_frequency = args.epochs + 1

    run_phase3_libraries(
        exp_name=args.exp_name,
        epochs=args.epochs,
        checkpoint_frequency=checkpoint_frequency,
        samples_per_file=args.samples_per_file,
        num_harmonics=args.num_harmonics,
        kan_backend=args.kan_backend,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == '__main__':
    main()
