import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Загрузка метрик из файла jsonl."""
    metrics = []
    try:
        with open(metrics_path, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
    except Exception as e:
        print(f"Ошибка чтения {metrics_path}: {e}", file=sys.stderr)
    return metrics

def load_config(config_path: Path) -> Dict[str, Any]:
    """Загрузка конфигурации из файла json."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {config_path}: {e}", file=sys.stderr)
        return {}

def plot_learning_curves(metrics: List[Dict[str, Any]], save_path: Path):
    """Генерация графиков обучения."""
    df = pd.DataFrame(metrics)
    if df.empty:
        return
        
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Learning Curves (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy/F1
    plt.subplot(1, 2, 2)
    if 'val_acc' in df.columns:
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    if 'val_f1' in df.columns:
        plt.plot(df['epoch'], df['val_f1'], label='Val F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def aggregate_reports(root_dir: str, output_file: str = None, plot: bool = False):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_file: Опциональный путь для сохранения агрегированного отчета (CSV).
        plot: Генерировать ли графики обучения для каждого эксперимента.
    """
    root_path = Path(root_dir)
    experiments = []

    print(f"Сканирование {root_path} на наличие экспериментов...")

    # Найти все файлы metrics.jsonl
    for metrics_file in root_path.rglob("metrics.jsonl"):
        exp_dir = metrics_file.parent
        config_file = exp_dir / "config.json"
        
        # Загрузка данных
        metrics = load_metrics(metrics_file)
        config = load_config(config_file) if config_file.exists() else {}
        
        if not metrics:
            continue

        # Генерация графиков
        if plot:
            plot_path = exp_dir / "learning_curves.png"
            plot_learning_curves(metrics, plot_path)

        # Найти лучшую эпоху (на основе min val_loss)
        # Если val_loss недоступна (например, только train), взять последнюю эпоху
        if any('val_loss' in m for m in metrics):
            best_epoch = min(metrics, key=lambda x: x.get('val_loss', float('inf')))
        else:
            best_epoch = metrics[-1]

        # Извлечение информации
        exp_data = {
            "Experiment": exp_dir.name,
            "Model": config.get('model', {}).get('name', 'Unknown'),
            "Params": str(config.get('model', {}).get('params', {})),
            "Epochs": len(metrics),
            "Best Epoch": best_epoch.get('epoch'),
            "Num Params": best_epoch.get('num_params', 0),
            "Train Loss": best_epoch.get('train_loss'),
            "Val Loss": best_epoch.get('val_loss'),
            "Val Acc": best_epoch.get('val_acc'),
            "Val F1": best_epoch.get('val_f1'),
            "Inf Time (ms)": best_epoch.get('inf_time_ms', 0.0),
            "Val B.Acc": best_epoch.get('val_balanced_acc', 0.0),
            "Time (s)": best_epoch.get('time'),
            "Path": str(exp_dir)
        }
        
        # Добавить некоторые детали конфигурации, если доступны
        if 'data' in config:
            exp_data['Window'] = config['data'].get('window_size')
            exp_data['Batch'] = config['data'].get('batch_size')
        
        experiments.append(exp_data)

    if not experiments:
        print("Эксперименты не найдены.")
        return

    # Создать DataFrame
    df = pd.DataFrame(experiments)
    
    # Сортировать по Val Loss
    if 'Val Loss' in df.columns:
        df = df.sort_values('Val Loss')

    # Отобразить
    print("\nАгрегированный отчет:")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # Сохранить
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nОтчет сохранен в {output_file}")

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ ===
    # Отредактируйте параметры ниже для быстрого запуска без аргументов командной строки.
    MANUAL_RUN = True
    
    # ROOT_DIR: Папка, где лежат результаты ваших экспериментов (metrics.jsonl, config.json).
    ROOT_DIR = "experiments/phase2_5"
    
    # OUTPUT_CSV: Имя файла для сохранения таблицы с результатами.
    OUTPUT_CSV = "reports/phase2_5_summary.csv"
    
    # GENERATE_PLOTS: Если True, для каждого эксперимента будут построены графики обучения.
    GENERATE_PLOTS = True

    if MANUAL_RUN:
        # Создаем папку для отчетов, если её нет
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        aggregate_reports(ROOT_DIR, OUTPUT_CSV, GENERATE_PLOTS)
    else:
        # === ВЕРСИЯ 2: ЗАПУСК ЧЕРЕЗ CLI (Командную строку) ===
        parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
        parser.add_argument("root_dir", type=str, help="Корневая директория, содержащая эксперименты (например, experiments/)")
        parser.add_argument("--output", type=str, default=None, help="Путь к выходному файлу CSV")
        parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
        
        args = parser.parse_args()
        aggregate_reports(args.root_dir, args.output, args.plot)
