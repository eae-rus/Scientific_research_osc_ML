import polars as pl
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
import time
import pandas as pd

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.models import (
    SimpleMLP, SimpleCNN, ResNet1D, 
    SimpleKAN, ConvKAN
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_tensor, device='cpu', num_repeats=100):
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Прогрев (несколько вызовов перед измерением)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_repeats):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_repeats
    return avg_time * 1000  # миллисекунды

def evaluate_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def main():
    # 1. Настройка путей
    METADATA_FILE = ROOT_DIR / 'ML_model' / 'MLOps dataset' / 'labeled_2025_12_03.csv'
    BASELINE_DIR = ROOT_DIR / 'experiments' / 'baseline'
    KAN_DIR = ROOT_DIR / 'experiments' / 'kan'
    
    # 2. Загрузка данных и воспроизведение сплита
    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    ml_cols = [c for c in df.columns if c.startswith('ML_')]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8).alias(c) 
        for c in ml_cols
    ])
    
    ml_data = df.select(ml_cols).to_numpy()
    combos = []
    for row in ml_data:
        active = [ml_cols[i] for i, val in enumerate(row) if val == 1]
        if not active:
            combos.append("Normal")
        else:
            combos.append(",".join(sorted(active)))
            
    df = df.with_columns(pl.Series("target_class", combos))
    
    le = LabelEncoder()
    y_enc = le.fit_transform(combos)
    df = df.with_columns(pl.Series("target_enc", y_enc))
    num_classes = len(le.classes_)
    
    feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    
    # Индексы (выбор стартовых позиций для окон)
    df = df.with_row_count("row_nr")
    file_stats = df.group_by("file_name").agg([
        pl.col("row_nr").min().alias("start_idx"),
        pl.len().alias("length")
    ])
    
    WINDOW_SIZE = 640
    valid_files = file_stats.filter(pl.col("length") >= WINDOW_SIZE)
    indices = valid_files["start_idx"].to_list()
    
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * 0.8)
    val_indices = indices[split_idx:]
    
    print(f"Val samples: {len(val_indices)}")
    
    # 3. Подготовка DataLoader (валидационный датасет)
    val_ds = OscillogramDataset(
        dataframe=df,
        indices=val_indices,
        window_size=WINDOW_SIZE,
        mode='classification',
        feature_mode='raw',
        feature_columns=feature_cols,
        target_columns="target_enc"
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 4. Определение моделей для оценки
    # Примечание: параметры должны совпадать с параметрами, использованными при обучении
    models_config = [
        {
            "name": "SimpleMLP",
            "path": BASELINE_DIR / "SimpleMLP.pt",
            "class": SimpleMLP,
            "params": {"input_size": len(feature_cols) * WINDOW_SIZE, "output_size": num_classes}
        },
        {
            "name": "SimpleCNN",
            "path": BASELINE_DIR / "SimpleCNN.pt",
            "class": SimpleCNN,
            "params": {"in_channels": len(feature_cols), "num_classes": num_classes}
        },
        {
            "name": "ResNet1D",
            "path": BASELINE_DIR / "ResNet1D.pt",
            "class": ResNet1D,
            "params": {"in_channels": len(feature_cols), "num_classes": num_classes}
        },
        {
            "name": "SimpleKAN",
            "path": KAN_DIR / "SimpleKAN.pt",
            "class": SimpleKAN,
            "params": {
                "input_size": len(feature_cols) * WINDOW_SIZE, 
                "output_size": num_classes,
                "grid_size": 5,
                "hidden_sizes": [64, 32]
            }
        },
        {
            "name": "ConvKAN",
            "path": KAN_DIR / "ConvKAN.pt",
            "class": ConvKAN,
            "params": {
                "in_channels": len(feature_cols), 
                "num_classes": num_classes,
                "grid_size": 5,
                "base_filters": 8
            }
        }
    ]
    
    results = []
    
    device = 'cpu' # Оценка на CPU для честного сравнения времени инференса
    print(f"Evaluating on {device}...")
    
    # Заглушка для измерения времени инференса
    dummy_input = torch.randn(1, len(feature_cols), WINDOW_SIZE).to(device)
    # Для MLP/SimpleKAN требуется сплющенный вход
    dummy_input_flat = dummy_input.view(1, -1)
    
    for cfg in models_config:
        print(f"Evaluating {cfg['name']}...")
        if not cfg['path'].exists():
            print(f"  Model file not found: {cfg['path']}")
            continue
            
        try:
            model = cfg['class'](**cfg['params'])
            model.load_state_dict(torch.load(cfg['path'], map_location=device))
            
            # Количество параметров
            n_params = count_parameters(model)

            # Время инференса
            if cfg['name'] in ['SimpleMLP', 'SimpleKAN']:
                inf_time = measure_inference_time(model, dummy_input_flat, device)
            else:
                inf_time = measure_inference_time(model, dummy_input, device)
                
            # Точность / F1
            # Примечание: MLP/KAN ожидают сплющенный вход, но `Dataset` возвращает (B, C, T).
            # Вариант решения — сплющивать тензор перед передачей в модель в цикле оценки.
            
            preds, labels = evaluate_model_wrapper(model, val_loader, device, cfg['name'])
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            
            results.append({
                "Model": cfg['name'],
                "Params": n_params,
                "Accuracy": acc,
                "F1-Macro": f1,
                "Inference Time (ms)": inf_time
            })
            
        except Exception as e:
            print(f"  Error evaluating {cfg['name']}: {e}")

    # Print Results
    results_df = pd.DataFrame(results)
    print("\n=== Evaluation Results ===")
    print(results_df.to_markdown(index=False))
    
    # Save to CSV
    results_df.to_csv(ROOT_DIR / 'reports' / 'phase2_baseline_metrics.csv', index=False)

def evaluate_model_wrapper(model, dataloader, device, model_name):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    # Флаг: требуется ли сплющивание входа для модели
    is_flat_model = model_name in ['SimpleMLP', 'SimpleKAN']
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            if is_flat_model:
                x = x.view(x.size(0), -1)
                
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    main()
