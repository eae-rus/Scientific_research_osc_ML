import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Добавляем корень проекта в путь импорта, чтобы можно было запускать как скрипт
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from osc_tools.ml.kan_conv.KANLinear import KANLinear

def plot_kan_activation(layer: KANLinear, in_idx: int, out_idx: int, x_range=(-1, 1), num_points=100, ax=None):
    """
    Строит график выученной функции активации для конкретной связи в слое KANLinear.

    Аргументы:
        layer: экземпляр `KANLinear`.
        in_idx: индекс входного канала.
        out_idx: индекс выходного нейрона.
        x_range: диапазон значений x для построения графика.
        num_points: число точек на сетке для построения.
        ax: объект matplotlib.axes (если None — создаётся новый).
    """
    device = layer.base_weight.device
    x = torch.linspace(x_range[0], x_range[1], num_points).to(device)
    
    # 1. Base Part: w_b * SiLU(x)
    w_b = layer.base_weight[out_idx, in_idx]
    y_base = w_b * torch.nn.SiLU()(x)
    
    # 2. Spline Part
    # Нам нужно вычислить B-сплайны только для одного входного измерения.
    # layer.b_splines ожидает (Batch, InFeatures).
    # Создадим фиктивный вход, где нужный канал меняется, а остальные 0.
    dummy_input = torch.zeros(num_points, layer.in_features).to(device)
    dummy_input[:, in_idx] = x
    
    # Вычисляем базисы
    # bases shape: (Batch, InFeatures, GridSize + Order)
    bases = layer.b_splines(dummy_input)
    bases_i = bases[:, in_idx, :] # (Batch, GridSize + Order)
    
    w_s = layer.spline_weight[out_idx, in_idx, :] # (GridSize + Order)
    
    if layer.enable_standalone_scale_spline:
        scaler = layer.spline_scaler[out_idx, in_idx]
    else:
        scaler = 1.0
        
    y_spline = torch.matmul(bases_i, w_s) * scaler
    
    y_total = y_base + y_spline
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        
    x_np = x.detach().cpu().numpy()
    y_total_np = y_total.detach().cpu().numpy()
    y_base_np = y_base.detach().cpu().numpy()
    y_spline_np = y_spline.detach().cpu().numpy()
    
    ax.plot(x_np, y_total_np, label='Сумма', linewidth=2)
    ax.plot(x_np, y_base_np, '--', label='Базовая (SiLU)', alpha=0.5)
    ax.plot(x_np, y_spline_np, ':', label='Сплайн', alpha=0.5)
    
    ax.set_title(f"Вход {in_idx} → Выход {out_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_kan_layer_grid(layer: KANLinear, max_inputs=5, max_outputs=5):
    """
    Строит сетку графиков активаций для слоя (первые N входов/выходов).

    Подписи осей и заголовки также на русском языке.
    """
    n_in = min(layer.in_features, max_inputs)
    n_out = min(layer.out_features, max_outputs)
    
    fig, axes = plt.subplots(n_out, n_in, figsize=(3*n_in, 2.5*n_out), sharex=True, sharey=True)
    
    if n_out == 1 and n_in == 1:
        axes = np.array([[axes]])
    elif n_out == 1:
        axes = axes[None, :]
    elif n_in == 1:
        axes = axes[:, None]
        
    for i in range(n_out):
        for j in range(n_in):
            plot_kan_activation(layer, j, i, ax=axes[i, j])
            if i == n_out - 1:
                axes[i, j].set_xlabel(f"Вход {j}")
            if j == 0:
                axes[i, j].set_ylabel(f"Выход {i}")
                
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Пример использования для проверки
    print("Запуск примера визуализации KAN слоя...")
    layer = KANLinear(in_features=3, out_features=2, grid_size=5)
    plot_kan_layer_grid(layer)
