import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

# Добавляем корень проекта в путь импорта
ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR_PROJECT))

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

def plot_comparison(all_histories: Dict[str, List[Dict[str, Any]]], save_dir: Path):
    """
    Генерирует сравнительные графики для всех экспериментов.
    """
    if not all_histories:
        return
        
    metrics_to_plot = ['train_loss', 'val_loss', 'val_acc', 'val_f1']
    titles = ['Comparison: Train Loss', 'Comparison: Validation Loss', 'Comparison: Validation Accuracy', 'Comparison: Validation F1 (Macro)']
    
    plt.figure(figsize=(24, 5))
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        plt.subplot(1, 4, i + 1)
        
        for exp_name, history in all_histories.items():
            df = pd.DataFrame(history)
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=exp_name)
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric.split('_')[-1].capitalize())
        if len(all_histories) < 10: # Чтобы легенда не перекрывала всё, если много моделей
            plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_metrics.png")
    plt.close()

def benchmark_model_cpu(exp_dir: Path, config: Dict[str, Any], iterations: int = 1000) -> float:
    """Замер скорости инференса на CPU для конкретной модели на пустых (dummy) данных."""
    try:
        model_name = config.get('model', {}).get('name')
        params = config.get('model', {}).get('params', {})
        
        # Динамический импорт моделей
        from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN, ResNet1D
        from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN
        
        models_map = {
            'SimpleMLP': SimpleMLP, 'SimpleCNN': SimpleCNN, 'ResNet1D': ResNet1D,
            'SimpleKAN': SimpleKAN, 'ConvKAN': ConvKAN, 'PhysicsKAN': PhysicsKAN
        }
        
        if model_name not in models_map:
            return 0.0
            
        model = models_map[model_name](**params).cpu()
        model.eval()
        
        # Загрузка весов (если есть)
        model_path = exp_dir / "best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])

        # Подготовка данных (dummy)
        if model_name in ['SimpleMLP', 'SimpleKAN']:
            input_size = params.get('input_size', 2560)
            dummy_input = torch.randn(1, input_size)
        else:
            in_channels = params.get('in_channels', 8)
            # Пытаемся определить размер временно окна (320, 20 или 2)
            pts_options = [320, 20, 10, 2]
            dummy_input = None
            for pts in pts_options:
                try:
                    test_input = torch.randn(1, in_channels, pts)
                    with torch.no_grad():
                        model(test_input)
                    dummy_input = test_input
                    break
                except:
                    continue
            
            if dummy_input is None:
                return 0.0

        # Warm-up
        with torch.no_grad():
            for _ in range(20):
                model(dummy_input)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        
        avg_time_ms = ((time.time() - start) * 1000) / iterations
        return avg_time_ms
        
    except Exception as e:
        # print(f"Benchmark error for {exp_dir.name}: {e}")
        return 0.0

def aggregate_reports(root_dir: str, output_file: str = None, plot: bool = False, benchmark: bool = False):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_file: Опциональный путь для сохранения агрегированного отчета (CSV).
        plot: Генерировать ли графики обучения для каждого эксперимента.
        benchmark: Выполнять ли глубокий бенчмарк на CPU.
    """
    root_path = Path(root_dir)
    experiments = []
    all_histories = {}

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
            
        all_histories[exp_dir.name] = metrics

        # Генерация графиков
        if plot:
            plot_path = exp_dir / "learning_curves.png"
            plot_learning_curves(metrics, plot_path)

        # Найти лучшую эпоху (на основе min val_loss)
        if any('val_loss' in m for m in metrics):
            best_epoch = min(metrics, key=lambda x: x.get('val_loss', float('inf')))
        else:
            best_epoch = metrics[-1]

        # Извлечение информации
        # Сначала пробуем взять из нового поля model_info в конфиге
        num_params = config.get('model_info', {}).get('num_params', best_epoch.get('num_params', 0))
        
        cpu_inf = best_epoch.get('cpu_inf_time_ms', 0.0)
        if benchmark:
            print(f"Бенчмарк CPU для {exp_dir.name}...")
            cpu_inf = benchmark_model_cpu(exp_dir, config, iterations=1000)

        exp_data = {
            "Experiment": exp_dir.name,
            "Model": config.get('model', {}).get('name', 'Unknown'),
            "Params": str(config.get('model', {}).get('params', {})),
            "Epochs": len(metrics),
            "Best Epoch": best_epoch.get('epoch'),
            "Num Params": num_params,
            "Train Loss": best_epoch.get('train_loss'),
            "Val Loss": best_epoch.get('val_loss'),
            "Val Acc": best_epoch.get('val_acc'),
            "Val F1": best_epoch.get('val_f1'),
            "Inf Time (ms)": best_epoch.get('inf_time_ms', 0.0),
            "CPU Inf (ms)": cpu_inf,
            "Val B.Acc": best_epoch.get('val_balanced_acc', 0.0),
            "Time (s)": best_epoch.get('epoch_time', best_epoch.get('time', 0.0)),
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
        
    # Сравнительные графики
    if plot:
        # Сохраняем рядом с CSV или в папку root_path
        save_dir = Path(output_file).parent if output_file else root_path
        plot_comparison(all_histories, save_dir)
        print(f"Сравнительные графики сохранены в {save_dir / 'comparison_metrics.png'}")

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ ===
    # Отредактируйте параметры ниже для быстрого запуска без аргументов командной строки.
    MANUAL_RUN = True
    
    # ROOT_DIR: Папка, где лежат результаты ваших экспериментов (metrics.jsonl, config.json).
    ROOT_DIR = "experiments/phase2_5"
    
    # OUTPUT_CSV: Имя файла для сохранения таблицы с результатами.
    OUTPUT_CSV = "reports/phase2_5_all_summary.csv"
    
    # GENERATE_PLOTS: Если True, для каждого эксперимента будут построены графики обучения.
    GENERATE_PLOTS = True
    
    # RUN_BENCHMARK: Если True, будет выполнен глубокий замер скорости инференса на CPU (1000 итераций).
    RUN_BENCHMARK = False # Можно включить при финальной агрегации

    if MANUAL_RUN:
        # Создаем папку для отчетов, если её нет
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        aggregate_reports(ROOT_DIR, OUTPUT_CSV, GENERATE_PLOTS, RUN_BENCHMARK)
    else:
        # === ВЕРСИЯ 2: ЗАПУСК ПО УМОЛЧАНИЮ (CLI / Командная строка) ===
        parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
        parser.add_argument("root_dir", type=str, help="Корневая директория, содержащая эксперименты (например, experiments/)")
        parser.add_argument("--output", type=str, default=None, help="Путь к выходному файлу CSV")
        parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
        parser.add_argument("--benchmark", action="store_true", help="Выполнить глубокий бенчмарк на CPU")
        
        args = parser.parse_args()
        aggregate_reports(args.root_dir, args.output, args.plot, args.benchmark)
