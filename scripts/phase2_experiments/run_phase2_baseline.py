import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset

def main():
    # 1. Setup Paths
    DATA_DIR = ROOT_DIR / 'raw_data' / 'Output'
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    norm_coef_path = METADATA_FILE.parent / 'norm_coef_all_v1.4.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'baseline'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка метаданных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # 2. Загрузка данных
    print(f"Загрузка данных из {METADATA_FILE}")
    # Используем read_csv, так как файл ~50MB и помещается в память.
    # При росте размера лучше переключиться на scan_csv и ленивую обработку.
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # 3. Обработка целевых меток
    print("Формирование целевых меток...")
    ml_cols = [c for c in df.columns if c.startswith('ML_')]
    
    # Заполняем пропуски нулями и приводим к целочисленному типу
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8).alias(c) 
        for c in ml_cols
    ])
    
    # Формируем строковую комбинацию активных ML_ колонок (или "Normal" если нет)
    ml_data = df.select(ml_cols).to_numpy()
    combos = []
    for row in ml_data:
        active = [ml_cols[i] for i, val in enumerate(row) if val == 1]
        if not active:
            combos.append("Normal")
        else:
            combos.append(",".join(sorted(active)))
            
    df = df.with_columns(pl.Series("target_class", combos))
    
    # Кодируем метки
    le = LabelEncoder()
    y_enc = le.fit_transform(combos)
    df = df.with_columns(pl.Series("target_enc", y_enc))
    
    num_classes = len(le.classes_)
    print(f"Число классов: {num_classes}")
    print(f"Классы: {le.classes_}")
    
    # 4. Выбор признаков
    feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    print(f"Найдено {len(feature_cols)} признаковых колонок")
    
    # 5. Создание индексов (по одному индексу на файл с достаточной длиной)
    print("Генерация индексов...")
    # Нужно убедиться, что окна не выходят за границы отдельных файлов.
    df = df.with_row_count("row_nr")
    
    # Группируем по file_name, получаем начальный индекс и длину
    file_stats = df.group_by("file_name").agg([
        pl.col("row_nr").min().alias("start_idx"),
        pl.len().alias("length")
    ])
    
    WINDOW_SIZE = 640
    
    valid_files = file_stats.filter(pl.col("length") >= WINDOW_SIZE)
    print(f"Валидных файлов (>= {WINDOW_SIZE}): {len(valid_files)} из {len(file_stats)}")
    
    # Для простоты используем стартовый индекс каждого валидного файла
    indices = valid_files["start_idx"].to_list()
    
    # 6. Разделение Train/Val
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # 7. Запуск экспериментов
    
    # Конфигурация данных
    data_config = DataConfig(
        path=str(METADATA_FILE),
        window_size=WINDOW_SIZE,
        batch_size=32,
        features=feature_cols,
        target="target_enc",
        mode="classification"
    )
    
    # Конфигурация обучения
    train_config = TrainingConfig(
        epochs=30,
        learning_rate=0.001,
        device='cuda',
        save_dir=str(EXPERIMENTS_DIR),
        experiment_name="baseline"
    )
    
    models_to_run = [
        ("SimpleMLP", {"input_size": len(feature_cols) * WINDOW_SIZE, "output_size": num_classes}),
        ("SimpleCNN", {"in_channels": len(feature_cols), "num_classes": num_classes}),
        ("ResNet1D", {"in_channels": len(feature_cols), "num_classes": num_classes})
    ]
    
    for model_name, model_params in models_to_run:
        print(f"\nЗапуск эксперимента: {model_name}")
        
        config = ExperimentConfig(
            model=ModelConfig(name=model_name, params=model_params),
            data=data_config,
            training=train_config
        )
        
        # Инициализация Runner
        runner = ExperimentRunner(config)
        
        # Создаём DataLoader'ы вручную, Runner не знает про наш in-memory DataFrame и индексы
        train_ds = OscillogramDataset(
            dataframe=df,
            indices=train_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='raw',
            feature_columns=feature_cols,
            target_columns="target_enc"
        , physical_normalization=True, norm_coef_path=str(norm_coef_path))
        
        val_ds = OscillogramDataset(
            dataframe=df,
            indices=val_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='raw',
            feature_columns=feature_cols,
            target_columns="target_enc"
        , physical_normalization=True, norm_coef_path=str(norm_coef_path))
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False)
        
        runner.train(train_loader, val_loader)
        
        # Сохраняем модель
        torch.save(runner.model.state_dict(), EXPERIMENTS_DIR / f"{model_name}.pt")
        print(f"Сохранено: {model_name}")

if __name__ == "__main__":
    main()
