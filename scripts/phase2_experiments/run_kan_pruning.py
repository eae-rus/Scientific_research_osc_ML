import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import polars as pl
from sklearn.preprocessing import LabelEncoder

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from osc_tools.ml.models import ConvKAN
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.kan_conv.KANLinear import KANLinear
from osc_tools.ml.config import ModelConfig

def load_data(batch_size=64):
    # Загрузка тестового датасета
    data_path = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    
    print(f"Загрузка данных из {data_path}")
    df = pl.read_csv(data_path, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # Обработка целевых меток
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
    
    # Создание индексов
    df = df.with_row_count("row_nr")
    file_stats = df.group_by("file_name").agg([
        pl.col("row_nr").min().alias("start_idx"),
        pl.len().alias("length")
    ])
    
    WINDOW_SIZE = 640
    valid_files = file_stats.filter(pl.col("length") >= WINDOW_SIZE)
    indices = valid_files["start_idx"].to_list()
    
    # Путь к файлу коэффициентов нормализации по сенсорам
    norm_coef_path = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    
    # Разбиение выборки
    generator = torch.Generator().manual_seed(42)
    # Здесь мы разбиваем датасет случайным образом (фиксированный seed для воспроизводимости).
    
    dataset = OscillogramDataset(
        dataframe=df,
        indices=indices,
        window_size=WINDOW_SIZE,
        feature_mode='symmetric',
        target_columns="target_enc",
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, len(le.classes_)

def load_model(model_path, device, num_classes):
    # Создаём модель ConvKAN с параметрами, согласованными с предыдущими экспериментами.
    # Замечание: ожидается, что для режима 'symmetric' число входных каналов ~= 12.
    in_channels = 12 

    model = ConvKAN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=8,
        grid_size=5,
        base_activation=torch.nn.SiLU
    )

    # Загружаем состояния модели (без строгой проверки совпадений ключей, т.к. может быть добавлен mask)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def calculate_importance(model, loader, device):
    """
    Рассчитывает метрики важности для всех слоёв `KANLinear`.
    Важность определяется как среднее значение |phi(x)| по собранным примерам.
    """
    importances = {}
    
    # Hook для сбора входов в слои KANLinear
    inputs = {}
    def get_input_hook(name):
        def hook(model, input, output):
            # input — это кортеж (x,)
            # Нам нужен только набор примеров входов для оценки важности
            if name not in inputs:
                inputs[name] = []
            if len(inputs[name]) < 1000: # Ограничение числа сохранённых образцов
                inputs[name].append(input[0].detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            hooks.append(module.register_forward_hook(get_input_hook(name)))
            
    # Прогоним небольшое число батчей для сбора входов
    print("Сбор активаций...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            model(x)
            if i >= 5: # 5 батчей достаточно для грубой оценки
                break
                
    # Удаляем hooks
    for h in hooks:
        h.remove()
        
    # Вычисляем метрики важности
    print("Вычисление метрик важности...")
    for name, module in model.named_modules():
        if isinstance(module, KANLinear) and name in inputs:
            # Concatenate collected inputs
            x_sample = torch.cat(inputs[name], dim=0).to(device) # (N, in_features)
            
            # Рассчитываем L1-норму функции активации для каждой связи
            # Нужно повторить логику прямого прохода, но посчитать по входным признакам отдельно
            # KANLinear: output = base + spline
            # score[out_i, in_j] = mean(|phi_{i,j}(x)|)
            
            batch_size = x_sample.size(0)
            scores = torch.zeros(module.out_features, module.in_features, device=device)
            
            # При необходимости можно обрабатывать кусками для экономии памяти; здесь N небольшой (~300)
            
            # Precompute splines: (N, in, grid+order)
            bases = module.b_splines(x_sample)
            
            for out_i in range(module.out_features):
                for in_j in range(module.in_features):
                    # Базовая часть
                    y_base = module.base_weight[out_i, in_j] * module.base_activation(x_sample[:, in_j])
                    
                    # Сплайн-часть
                    # bases[:, in_j, :] : (N, coeff)
                    # spline_weight[out_i, in_j, :] : (coeff)
                    y_spline = torch.matmul(bases[:, in_j, :], module.spline_weight[out_i, in_j, :])
                    
                    if module.enable_standalone_scale_spline:
                        y_spline = y_spline * module.spline_scaler[out_i, in_j]
                        
                    y_total = y_base + y_spline
                    scores[out_i, in_j] = torch.mean(torch.abs(y_total))
            
            importances[name] = scores.cpu()
            
    return importances

def prune_model(model, importances, threshold):
    """
    Применяет маску прореживания на основе оценок важности.
    """
    total_params = 0
    active_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, KANLinear) and name in importances:
            scores = importances[name]
            # Создаём маску: 1 если score >= threshold, иначе 0
            mask = (scores >= threshold).float().to(module.mask.device)
            
            # Обновляем буфер `mask` в слое
            module.mask.copy_(mask)
            
            # Подсчёт связей
            # Для простоты считаем просто количество связей (edges)
            n_links = module.in_features * module.out_features
            n_active = mask.sum().item()
            
            total_params += n_links
            active_params += n_active
            
    return active_params, total_params

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = y.long()
            if y.dim() > 1: y = y.squeeze()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return total_loss / total, correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    
    model_path = ROOT_DIR / 'experiments' / 'kan_activations' / 'ConvKAN_SiLU.pt'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print("Загрузка данных...")
    test_loader, num_classes = load_data()
    
    print(f"Загрузка модели (num_classes={num_classes})...")
    model = load_model(model_path, device, num_classes)
    
    # Базовая оценка
    print("Оценка базовой модели...")
    base_loss, base_acc = evaluate(model, test_loader, device)
    print(f"Базовоe: Loss: {base_loss:.4f}, Acc: {base_acc:.4f}")
    
    # Сбор метрик важности
    importances = calculate_importance(model, test_loader, device)
    
    # Pruning thresholds
    thresholds = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2]
    results = []
    
    print("\nЗапуск экспериментов по прореживанию...")
    print(f"{'Порог':<10} | {'Активные связи':<12} | {'Разреженность (%)':<16} | {'Acc':<8} | {'Loss':<8}")
    print("-" * 80)
    
    for thresh in thresholds:
        active, total = prune_model(model, importances, thresh)
        sparsity = 100 * (1 - active / total)
        
        loss, acc = evaluate(model, test_loader, device)
        
        print(f"{thresh:<10.4f} | {int(active):<12} | {sparsity:<16.2f} | {acc:<8.4f} | {loss:<8.4f}")
        
        results.append({
            'threshold': thresh,
            'active_edges': int(active),
            'total_edges': total,
            'sparsity': sparsity,
            'accuracy': acc,
            'loss': loss
        })
        
    # Save results
    save_path = ROOT_DIR / 'reports' / 'pruning_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nРезультаты сохранены в {save_path}")
    
    # Plot
    thresholds_val = [r['threshold'] for r in results]
    accs = [r['accuracy'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Порог (лог шкала)')
    ax1.set_ylabel('Точность (Accuracy)', color=color)
    ax1.plot(thresholds_val, accs, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.axhline(y=base_acc, color='gray', linestyle='--', label='Базовая точность')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Разреженность (%)', color=color)
    ax2.plot(thresholds_val, sparsities, marker='s', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Прореживание KAN: Точность vs Порог')
    fig.tight_layout()
    plt.savefig(ROOT_DIR / 'reports' / 'figures' / 'kan_pruning.png')
    print("График сохранён в reports/figures/kan_pruning.png")

if __name__ == '__main__':
    main()
