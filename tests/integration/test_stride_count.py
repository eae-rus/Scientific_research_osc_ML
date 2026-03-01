"""Тест полной оценки со stride=1."""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from scripts.evaluation.aggregate_reports import evaluate_full_test_dataset, load_config, _test_data_cache
import time

# Очищаем кэш
_test_data_cache.clear()

data_dir = str(ROOT / 'data' / 'ml_datasets')
exp_dir = ROOT / 'experiments' / 'phase2_6' / 'Exp2.6.1_stride_SimpleMLP_medium_stride'
config = load_config(exp_dir / 'config.json')

print(f"Модель: {config.get('model', {}).get('name')}")
print(f"Input size: {config.get('model', {}).get('params', {}).get('input_size')}")
print()

print("Запуск полной оценки (stride=1, batch_size=2048)...")
start = time.time()
results = evaluate_full_test_dataset(exp_dir, config, data_dir, batch_size=2048)
elapsed = time.time() - start

print(f"\n=== Результаты ===")
print(f"Время: {elapsed:.2f}s")
print(f"Точек: {results['full_eval_samples']:,}")
print(f"Скорость: {results['full_eval_samples']/elapsed:,.0f} точек/сек")
print(f"Full Best Acc: {results['full_best_acc']:.4f}")
print(f"Full Best F1: {results['full_best_f1']:.4f}")
