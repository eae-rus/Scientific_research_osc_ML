"""
Визуализация отчётов: ReportVisualizer, кривые обучения, сравнительные графики.

Включает plot_learning_curves, plot_comparison, combine_training_histories.
"""
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.evaluation._core.config_resolvers import parse_experiment_info, extract_base_exp_id


def plot_learning_curves(metrics: List[Dict[str, Any]], save_path: Path):
    """Генерация графиков обучения."""
    df = pd.DataFrame(metrics)
    if df.empty:
        return
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Learning Curves (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
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
    """Генерирует сравнительные графики для всех экспериментов."""
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
        if len(all_histories) < 10:
            plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_metrics.png")
    plt.close()


def combine_training_histories(experiments_dir: Path, output_path: Path):
    """
    Объединяет файлы metrics.jsonl всех экспериментов в один текстовый файл.
    """
    print(f"Combining training histories into {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        metrics_files = sorted(list(experiments_dir.rglob("metrics.jsonl")))
        
        for metrics_path in metrics_files:
            exp_dir = metrics_path.parent
            outfile.write(f"Model: {exp_dir.name}\n")
            
            try:
                with open(metrics_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                outfile.write("\n")
            except Exception as e:
                print(f"Error reading {metrics_path}: {e}")


class ReportVisualizer:
    """Визуализатор отчётов: групповые кривые, Парето."""
    
    TEXTS = {
        'ru': {
            'loss_title': "Опыт {}: Лосс валидации (Обрезано по 2.0)",
            'f1_title': "Опыт {}: F1-Macro (Валидация)",
            'epoch': "Эпоха",
            'loss': "Лосс",
            'f1': "F1 Score",
            'pareto_title': "Кривая Парето: Точность vs Скорость инференса (CPU)",
            'latency': "Задержка (мс) - Лог. шкала",
            'val_f1_label': "F1-Macro (Валидация)",
            'model': "Модель",
            'complexity': "Сложность",
            'groups_desc': "Генерация групповых графиков"
        },
        'en': {
            'loss_title': "Exp {}: Validation Loss (Clipped at 2.0)",
            'f1_title': "Exp {}: F1-Macro Score",
            'epoch': "Epoch",
            'loss': "Loss",
            'f1': "F1 Score",
            'pareto_title': "Pareto Frontier: Accuracy vs Inference Speed (CPU)",
            'latency': "Latency (ms) - Log Scale",
            'val_f1_label': "Validation F1-Macro",
            'model': "Model",
            'complexity': "Complexity",
            'groups_desc': "Generating group plots"
        }
    }

    def __init__(self, output_root: Path, lang: str = 'ru'):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lang = lang if lang in self.TEXTS else 'ru'
        self.t = self.TEXTS[self.lang]
        
        self.color_map = {
            'MLP': '#95a5a6',
            'CNN': '#3498db',
            'ResNet': '#2ecc71',
            'SimpleKAN': '#e67e22',
            'ConvKAN': '#e74c3c',
            'PhysicsKAN': '#9b59b6',
            'cPhysicsKAN': '#f1c40f',
            'Unknown': '#34495e'
        }

    def plot_group_curves(self, group_label: str, group_df: pd.DataFrame, histories: Dict[str, List[Dict]], group_file_id: str):
        """Рисует сравнение всех моделей внутри одной группы."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for name, history in histories.items():
            if not history: continue
            df_h = pd.DataFrame(history)
            info = parse_experiment_info(name)
            
            arch_suffix = "H" if info['arch_type'] == "Hierarchical" else "B"
            model_label = f"{info['model_family']} ({arch_suffix}-{info['complexity']})"
            
            color = self.color_map.get(info['model_family'], self.color_map['Unknown'])
            ls = '--' if info['arch_type'] == "Hierarchical" else '-'
            
            val_loss = df_h['val_loss'].clip(upper=2.0)
            ax1.plot(df_h['epoch'], val_loss, label=model_label, color=color, linestyle=ls, alpha=0.8)
            
            if 'val_f1' in df_h.columns:
                ax2.plot(df_h['epoch'], df_h['val_f1'], label=model_label, color=color, linestyle=ls, alpha=0.8)

        ax1.set_title(self.t['loss_title'].format(group_label))
        ax1.set_xlabel(self.t['epoch'])
        ax1.set_ylabel(self.t['loss'])
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(self.t['f1_title'].format(group_label))
        ax2.set_xlabel(self.t['epoch'])
        ax2.set_ylabel(self.t['f1'])
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        save_path = self.output_root / "groups" / f"{group_file_id}_comparison_{self.lang}.png"
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_pareto(self, summary_df: pd.DataFrame):
        """Рисует график 'Точность vs Скорость'."""
        plt.figure(figsize=(10, 7))
        if 'CPU Inf (ms)' not in summary_df.columns:
            return
            
        df_plot = summary_df[summary_df['CPU Inf (ms)'] > 0].copy()
        if df_plot.empty: return

        df_plot['DisplayModel'] = df_plot.apply(
            lambda x: f"{x['Model']} ({x['arch_type'][0]})" if 'arch_type' in x and x['arch_type'] != 'Base' else x['Model'], axis=1
        )

        sns.scatterplot(
            data=df_plot, 
            x='CPU Inf (ms)', 
            y='Val F1', 
            hue='Model', 
            palette=self.color_map,
            style='Complexity',
            s=120, 
            alpha=0.8
        )
        
        plt.xscale('log')
        plt.title(self.t['pareto_title'])
        plt.xlabel(self.t['latency'])
        plt.ylabel(self.t['val_f1_label'])
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_root / f"summary_pareto_{self.lang}.png", dpi=200)
        plt.close()
