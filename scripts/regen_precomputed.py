"""Регенерация test_precomputed.csv."""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from osc_tools.data_management import DatasetManager

dm = DatasetManager('data/ml_datasets')
dm.create_precomputed_test_csv(force=True)
print("Done!")
