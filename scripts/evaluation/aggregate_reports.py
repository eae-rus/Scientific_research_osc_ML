import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any
import sys

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

def aggregate_reports(root_dir: str, output_file: str = None):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_file: Опциональный путь для сохранения агрегированного отчета (CSV).
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
            "Train Loss": best_epoch.get('train_loss'),
            "Val Loss": best_epoch.get('val_loss'),
            "Val Acc": best_epoch.get('val_acc'),
            "Val F1": best_epoch.get('val_f1'),
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
    parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
    parser.add_argument("root_dir", type=str, help="Корневая директория, содержащая эксперименты (например, experiments/)")
    parser.add_argument("--output", type=str, default=None, help="Путь к выходному файлу CSV")
    
    args = parser.parse_args()
    
    aggregate_reports(args.root_dir, args.output)
