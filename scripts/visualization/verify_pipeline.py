import sys
import os
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import torch

# Добавляем корень проекта в путь импорта
sys.path.append(str(Path(__file__).parent.parent.parent))

from osc_tools.ml.dataset import OscillogramDataset

def plot_verification(dataset, idx=None, output_path=None, search_line_only=False):
    """
    Визуализация шагов пайплайна: исходные данные -> умный селектор.
    Разделяет токи и напряжения на разные графики для наглядности.
    """
    if output_path is None:
        output_path = str(Path(__file__).parent.parent.parent / 'reports' / 'pipeline_verification.png')
    
    # Создаем папку если её нет
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if idx is None and search_line_only:
        idx = find_line_voltage_sample(dataset, max_search=500)
    
    if idx is None:
        # Случайный индекс
        rand_idx = np.random.randint(0, len(dataset))
    else:
        rand_idx = idx

    print(f"Анализ индекса: {rand_idx}")
    
    # 1. Получаем индекс и вырезаем исходный DataFrame
    index_item = dataset.indices[rand_idx]
    if isinstance(index_item, (tuple, list)):
        file_start, file_len = index_item
        if file_len <= dataset.window_size:
            start_idx = file_start
        else:
            max_offset = file_len - dataset.window_size
            offset = np.random.randint(0, max_offset + 1)
            start_idx = file_start + offset
    else:
        start_idx = index_item

    sample_df = dataset.data.slice(start_idx, dataset.window_size)
    
    file_name = "Unknown"
    if 'file_name' in sample_df.columns:
        file_name = sample_df['file_name'][0]
    
    print(f"Файл: {file_name}")

    # --- Подготовка групп сигналов ---
    all_cols = sample_df.columns
    currents = [c for c in all_cols if c in ['IA', 'IB', 'IC', 'IN']]
    voltages_bb = [c for c in all_cols if 'BB' in c]
    voltages_cl = [c for c in all_cols if 'CL' in c]
    voltages_other = [c for c in all_cols if c in ['UA', 'UB', 'UC', 'UN', 'UAB', 'UBC', 'UCA']]

    # Цветовая схема
    def get_color(name):
        # Убираем 'BB' и 'CL' для правильного определения фазы
        clean_name = name.replace(' BB', '').replace(' CL', '')
        # Возвращает (color, linestyle)
        if 'AB' in clean_name:
            return 'yellow', '--'
        if 'BC' in clean_name:
            return 'green', '--'
        if 'CA' in clean_name:
            return 'red', '--'
        if 'A' in clean_name and 'B' not in clean_name and 'C' not in clean_name:
            return 'yellow', '-'
        if 'B' in clean_name and 'A' not in clean_name and 'C' not in clean_name:
            return 'green', '-'
        if 'C' in clean_name and 'A' not in clean_name and 'B' not in clean_name:
            return 'red', '-'
        if 'N' in clean_name or '0' in clean_name:
            return 'blue', '-'
        return 'gray', '-'

    fig, axes = plt.subplots(5, 1, figsize=(15, 25), sharex=True)
    
    def plot_on_ax(ax, cols, title, ylabel):
        ax.set_title(title, fontsize=12, fontweight='bold')
        for c in cols:
            try:
                data = sample_df[c].cast(pl.Float32).to_numpy()
                color, linestyle = get_color(c)
                ax.plot(data, label=c, color=color, linestyle=linestyle, alpha=0.8)
            except:
                pass
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)

    # 1. Исходные Токи
    plot_on_ax(axes[0], currents, f"1. Исходные токи - {file_name}", "Ток (A)")

    # 2. Исходные Напряжения СШ (BB)
    plot_on_ax(axes[1], voltages_bb, "2. Исходные напряжения СШ (BB)", "Напряжение (V)")

    # 3. Исходные Напряжения КЛ и прочие
    plot_on_ax(axes[2], voltages_cl + voltages_other, "3. Исходные напряжения КЛ и прочие", "Напряжение (V)")

    # --- Обработка Smart Selector ---
    if dataset.physical_normalization and 'file_name' in sample_df.columns:
        try:
             sample_df_norm = dataset._apply_physical_normalization(sample_df, file_name)
        except:
            sample_df_norm = sample_df
    else:
        sample_df_norm = sample_df
        
    smart_data = dataset._get_standardized_raw_data(sample_df_norm)
    # Порядок в smart_data: IA, IB, IC, In, UA, UB, UC, Un
    
    labels_i = ['IA', 'IB', 'IC', 'IN']
    labels_u = ['UA', 'UB', 'UC', 'UN']

    # 4. Smart Selector - Токи
    ax = axes[3]
    ax.set_title("4. Умный селектор: Токи (Стандартизировано)", fontsize=12, fontweight='bold')
    for i in range(4):
        color, linestyle = get_color(labels_i[i])
        ax.plot(smart_data[:, i], label=labels_i[i], color=color, linestyle=linestyle)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Ток (норм)")

    # 5. Smart Selector - Напряжения
    ax = axes[4]
    ax.set_title("5. Умный селектор: Напряжения (Стандартизировано)", fontsize=12, fontweight='bold')
    for i in range(4, 8):
        idx_u = i - 4
        color, linestyle = get_color(labels_u[idx_u])
        ax.plot(smart_data[:, i], label=labels_u[idx_u], color=color, linestyle=linestyle)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Напр. (норм)")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Визуализация сохранена в {output_path}")

def find_line_voltage_sample(dataset, max_search=200):
    """
    Ищет в датасете первый пример, где отсутствуют фазные напряжения 
    и используются линейные для восстановления.
    """
    print(f"Поиск осциллограммы только с линейными напряжениями (макс. поиск: {max_search})...")
    
    for i in range(min(len(dataset), max_search)):
        index_item = dataset.indices[i]
        if isinstance(index_item, (tuple, list)):
            start_idx = index_item[0]
        else:
            start_idx = index_item
            
        sample_df = dataset.data.slice(start_idx, dataset.window_size)
        _, u_type = dataset._get_best_voltage_channels(sample_df)
        
        if u_type == 'line':
            print(f"Найдена подходящая осциллограмма на индексе {i}!")
            return i
            
    print("Осциллограмма только с линейными напряжениями не найдена в пределах поиска.")
    return None

def main():
    # Путь к датасету
    DATA_PATH = Path("data/ml_datasets/labeled_2025_12_03.csv")
    
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        return

    print("Загрузка датасета...")
    df = pl.read_csv(DATA_PATH)
    
    window_size = 320
    indices = OscillogramDataset.create_indices(df, window_size, mode='train')
    
    dataset = OscillogramDataset(
        dataframe=df,
        indices=indices,
        window_size=window_size,
        feature_mode='raw',
        physical_normalization=True, # Включаем для проверки расчётов
        norm_coef_path="data/ml_datasets/norm_coef_all_v1.4.csv"
    )
    
    # Параметр: искать ли осциллограмму только с линейными напряжениями
    search_line_only = False  # По умолчанию False (случайный выбор)
    
    plot_verification(dataset, search_line_only=search_line_only)


if __name__ == "__main__":
    print("Старт скрипта верификации...")
    main()
