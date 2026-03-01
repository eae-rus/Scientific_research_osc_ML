"""
Интеграционный тест для проверки функции полной оценки на тестовом датасете.

Запуск: python -m pytest tests/integration/test_full_eval.py -v
Или напрямую: python tests/integration/test_full_eval.py
"""
import sys
from pathlib import Path

# Добавляем корень проекта
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Сбрасываем кэш при каждом запуске для тестирования
import scripts.evaluation.aggregate_reports as ar
ar._test_data_cache.clear()

from scripts.evaluation.aggregate_reports import (
    _get_test_data_cached, 
    _create_model_from_config,
    evaluate_full_test_dataset,
    load_config
)

import torch
import numpy as np
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.data_management.dataset_manager import DatasetManager

def main():
    print("=" * 60)
    print(">>> Тест ПОЛНОЙ оценки с OscillogramDataset <<<")
    print("=" * 60)
    
    data_dir = str(ROOT / 'data' / 'ml_datasets')
    
    print("\n>>> Тест создания модели <<<")
    exp_dir = ROOT / 'experiments' / 'phase2_6' / 'Exp2.6.1_stride_SimpleMLP_medium_stride'
    config = load_config(exp_dir / 'config.json')
    model_params = config.get("model", {}).get("params", {})
    print(f'Config model: {config.get("model", {}).get("name")}')
    print(f'Expected input_size: {model_params.get("input_size")}')
    
    model = _create_model_from_config(config)
    print(f'Model created: {model is not None}')
    
    # Загружаем веса
    print("\n>>> Загрузка весов модели <<<")
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Веса загружены")
    
    # Загрузка данных через OscillogramDataset
    print("\n>>> Загрузка тестовых данных через OscillogramDataset <<<")
    dm = DatasetManager(data_dir)
    test_df = dm.load_test_df(precomputed=False)
    test_df = test_df.with_row_index("row_nr")
    print(f"Test DataFrame: {len(test_df)} строк")
    
    # Создаём индексы (как при обучении - samples_per_file)
    indices = OscillogramDataset.create_indices(
        test_df,
        window_size=320,
        mode='val',
        samples_per_file=10  # Небольшое количество для быстрого теста
    )
    print(f"Созданные индексы: {len(indices)}")
    
    # Создаём датасет
    norm_coef_path = Path(data_dir) / 'norm_coef_all_v1.4.csv'
    test_ds = OscillogramDataset(
        dataframe=test_df,
        indices=indices[:100],  # Только 100 для теста
        window_size=320,
        mode='classification',
        feature_mode='phase_polar',
        sampling_strategy='stride',
        downsampling_stride=16,
        target_columns=['Target_Normal', 'Target_ML_1', 'Target_ML_2', 'Target_ML_3'],
        target_level='base_labels',
        physical_normalization=True,
        norm_coef_path=str(norm_coef_path),
        augment=False
    )
    print(f"Dataset создан: {len(test_ds)} сэмплов")
    
    # Проверяем формат данных
    x, y = test_ds[0]
    print(f"\n>>> Формат данных <<<")
    print(f"x.shape: {x.shape}")
    print(f"x.min: {x.min():.4f}, x.max: {x.max():.4f}, x.mean: {x.mean():.4f}")
    print(f"y: {y.numpy()}")
    
    # Подсчёт точности на 100 примерах
    print("\n>>> Оценка на 100 сэмплах <<<")
    correct = 0
    total = len(test_ds)
    pred_counts = {}
    gt_counts = {}
    
    for i in range(total):
        x, y = test_ds[i]
        x_flat = x.flatten().unsqueeze(0)
        with torch.no_grad():
            output = model(x_flat)
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float()
        
        if (pred[0] == y).all().item():
            correct += 1
        
        pred_key = tuple(pred[0].int().tolist())
        gt_key = tuple(y.int().tolist())
        pred_counts[pred_key] = pred_counts.get(pred_key, 0) + 1
        gt_counts[gt_key] = gt_counts.get(gt_key, 0) + 1
    
    print(f'Accuracy: {correct/total:.2%}')
    print("\nРаспределение предсказаний:")
    for k, v in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\nGround Truth:")
    for k, v in sorted(gt_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    
    # Тест полной оценки
    print("\n>>> Тест evaluate_full_test_dataset <<<")
    results = evaluate_full_test_dataset(exp_dir, config, data_dir, batch_size=128, use_gpu=True)
    print('\nResults:')
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0:
            print(f'  {k}: {[round(x, 4) for x in v]}')
        elif isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    
    print("\n>>> Тест завершён <<<")

if __name__ == '__main__':
    main()
