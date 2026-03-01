"""Тесты для PrecomputedDataset."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.precomputed_dataset import create_precomputed_dataset


@pytest.fixture
def data_dir() -> Path:
    data_path = PROJECT_ROOT / 'data' / 'ml_datasets'
    main_csv = data_path / 'labeled_2025_12_03.csv'
    if not main_csv.exists():
        pytest.skip(f"Нет основного датасета: {main_csv}")
    return data_path


def test_precomputed_dataset_basic(data_dir: Path):
    ds = create_precomputed_dataset(
        str(data_dir),
        window_size=320,
        feature_mode='phase_polar',
        sampling_strategy='snapshot'
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x is not None and y is not None


def test_precomputed_dataset_modes(data_dir: Path):
    ds_sym = create_precomputed_dataset(
        str(data_dir),
        window_size=320,
        feature_mode='symmetric',
        sampling_strategy='snapshot'
    )
    x_sym, y_sym = ds_sym[0]
    assert x_sym is not None and y_sym is not None

    ds_raw = create_precomputed_dataset(
        str(data_dir),
        window_size=320,
        feature_mode='raw',
        sampling_strategy='snapshot'
    )
    x_raw, y_raw = ds_raw[0]
    assert x_raw is not None and y_raw is not None
