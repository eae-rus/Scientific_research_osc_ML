"""
Расширенный модуль визуализации для сравнения моделей.

Содержит:
1. Графики Парето (8 вариантов по разным метрикам и осям)
2. Графики обучения по сложности (объединённые для моделей одной сложности)
3. Парето по типам данных (для каждой модели отдельно)
4. Дополнительные визуализации для анализа

Структура выходных папок:
    output_dir/
    ├── pareto_cpu_time/         # Парето по CPU time
    │   ├── pareto_val_f1_cpu.png
    │   ├── pareto_val_acc_cpu.png
    │   ├── pareto_full_f1_cpu.png
    │   └── pareto_full_acc_cpu.png
    ├── pareto_params/           # Парето по количеству параметров
    │   ├── pareto_val_f1_params.png
    │   ├── pareto_val_acc_params.png
    │   ├── pareto_full_f1_params.png
    │   └── pareto_full_acc_params.png
    ├── learning_curves/         # Кривые обучения по сложности
    │   ├── light/
    │   ├── medium/
    │   └── heavy/
    ├── data_type_analysis/      # Анализ по типам данных
    │   ├── MLP/
    │   ├── CNN/
    │   └── ...
    └── additional/              # Дополнительные графики
        ├── heatmaps/
        ├── boxplots/
        └── radar/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import re

# Настройки matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class AdvancedVisualizer:
    """Расширенный визуализатор для отчётов по экспериментам."""
    
    # Цветовая палитра для моделей
    MODEL_COLORS = {
        'MLP': '#95a5a6',
        'CNN': '#3498db',
        'ResNet': '#2ecc71',
        'SimpleKAN': '#e67e22',
        'ConvKAN': '#e74c3c',
        'PhysicsKAN': '#9b59b6',
        'Unknown': '#34495e'
    }
    
    # Цветовая палитра для типов данных (Features)
    FEATURE_COLORS = {
        'Raw': '#2ecc71',
        'PhasePolar': '#3498db',
        'PhaseRect': '#1abc9c',
        'Symmetric': '#9b59b6',
        'Power': '#e74c3c',
        'AB': '#f39c12',
        'Unknown': '#95a5a6'
    }
    
    # Маркеры для sampling стратегий
    SAMPLING_MARKERS = {
        'Stride': 'o',      # Круг
        'Snapshot': 's',    # Квадрат
        'NoneSampling': '^',# Треугольник
        'Unknown': 'D'      # Ромб
    }
    
    # Маркеры для сложности
    COMPLEXITY_MARKERS = {
        'Light': 'o',
        'Medium': 's',
        'Heavy': '^',
        'Unknown': 'D'
    }
    
    # Тексты на русском и английском
    TEXTS = {
        'ru': {
            'pareto_val_f1_cpu': 'Парето: Val F1 vs Время инференса (CPU)',
            'pareto_val_acc_cpu': 'Парето: Val Accuracy vs Время инференса (CPU)',
            'pareto_full_f1_cpu': 'Парето: Full Test F1 (лучшая) vs Время инференса (CPU)',
            'pareto_full_acc_cpu': 'Парето: Full Test Accuracy (лучшая) vs Время инференса (CPU)',
            'pareto_val_f1_params': 'Парето: Val F1 vs Количество параметров',
            'pareto_val_acc_params': 'Парето: Val Accuracy vs Количество параметров',
            'pareto_full_f1_params': 'Парето: Full Test F1 (лучшая) vs Количество параметров',
            'pareto_full_acc_params': 'Парето: Full Test Accuracy (лучшая) vs Количество параметров',
            'cpu_time_ms': 'Время инференса CPU (мс) - лог. шкала',
            'params': 'Количество параметров - лог. шкала',
            'val_f1': 'Val F1-Macro',
            'val_acc': 'Val Accuracy',
            'full_f1': 'Full Test F1-Macro (лучшая)',
            'full_acc': 'Full Test Accuracy (лучшая)',
            'model': 'Модель',
            'complexity': 'Сложность',
            'features': 'Тип данных',
            'sampling': 'Sampling',
            'learning_title': 'Кривые обучения - {} (сложность: {})',
            'epoch': 'Эпоха',
            'loss': 'Loss',
            'f1': 'F1 Score',
            'acc': 'Accuracy',
            'data_type_title': 'Парето по типам данных: {}',
        },
        'en': {
            'pareto_val_f1_cpu': 'Pareto: Val F1 vs CPU Inference Time',
            'pareto_val_acc_cpu': 'Pareto: Val Accuracy vs CPU Inference Time',
            'pareto_full_f1_cpu': 'Pareto: Full Test F1 (best) vs CPU Inference Time',
            'pareto_full_acc_cpu': 'Pareto: Full Test Accuracy (best) vs CPU Inference Time',
            'pareto_val_f1_params': 'Pareto: Val F1 vs Number of Parameters',
            'pareto_val_acc_params': 'Pareto: Val Accuracy vs Number of Parameters',
            'pareto_full_f1_params': 'Pareto: Full Test F1 (best) vs Number of Parameters',
            'pareto_full_acc_params': 'Pareto: Full Test Accuracy (best) vs Number of Parameters',
            'cpu_time_ms': 'CPU Inference Time (ms) - log scale',
            'params': 'Number of Parameters - log scale',
            'val_f1': 'Val F1-Macro',
            'val_acc': 'Val Accuracy',
            'full_f1': 'Full Test F1-Macro (best)',
            'full_acc': 'Full Test Accuracy (best)',
            'model': 'Model',
            'complexity': 'Complexity',
            'features': 'Feature Type',
            'sampling': 'Sampling',
            'learning_title': 'Learning Curves - {} (complexity: {})',
            'epoch': 'Epoch',
            'loss': 'Loss',
            'f1': 'F1 Score',
            'acc': 'Accuracy',
            'data_type_title': 'Pareto by Data Types: {}',
        }
    }
    
    def __init__(self, output_root: Path, lang: str = 'ru'):
        """
        Инициализация визуализатора.
        
        Args:
            output_root: Корневая директория для сохранения графиков
            lang: Язык графиков ('ru' или 'en')
        """
        self.output_root = Path(output_root)
        self.lang = lang if lang in self.TEXTS else 'ru'
        self.t = self.TEXTS[self.lang]
        
        # Создаём структуру папок
        self._create_directories()
    
    def _create_directories(self):
        """Создаёт структуру папок для выходных файлов."""
        dirs = [
            'pareto_cpu_time',
            'pareto_params',
            'learning_curves/light',
            'learning_curves/medium', 
            'learning_curves/heavy',
            'data_type_analysis',
            'additional/heatmaps',
            'additional/boxplots',
            'additional/radar',
            'additional/rankings'
        ]
        for d in dirs:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)
    
    def _get_best_full_metric(self, row: pd.Series, metric: str) -> float:
        """
        Возвращает лучшее значение между Full Best и Full Final.
        
        Args:
            row: Строка DataFrame
            metric: 'f1' или 'acc'
        
        Returns:
            Лучшее значение метрики
        """
        if metric == 'f1':
            best = row.get('Full Best F1', 0.0)
            final = row.get('Full Final F1', 0.0)
        else:  # acc
            best = row.get('Full Best Acc', 0.0)
            final = row.get('Full Final Acc', 0.0)
        
        # Обработка NaN
        best = best if pd.notna(best) else 0.0
        final = final if pd.notna(final) else 0.0
        
        return max(best, final)
    
    # =========================================================================
    # ЧАСТЬ 1: ГРАФИКИ ПАРЕТО (8 вариантов)
    # =========================================================================
    
    def plot_all_pareto(self, df: pd.DataFrame):
        """
        Строит все 8 графиков Парето.
        
        1-4: По CPU time (Val F1, Val Acc, Full F1, Full Acc)
        5-8: По Params (Val F1, Val Acc, Full F1, Full Acc)
        """
        # Добавляем колонку с лучшими Full метриками
        df = df.copy()
        df['Best Full F1'] = df.apply(lambda r: self._get_best_full_metric(r, 'f1'), axis=1)
        df['Best Full Acc'] = df.apply(lambda r: self._get_best_full_metric(r, 'acc'), axis=1)
        
        # По CPU time
        self._plot_pareto(
            df, x_col='CPU Inf (ms)', y_col='Val F1',
            x_label=self.t['cpu_time_ms'], y_label=self.t['val_f1'],
            title=self.t['pareto_val_f1_cpu'],
            save_path=self.output_root / 'pareto_cpu_time' / 'pareto_val_f1_cpu.png'
        )
        
        self._plot_pareto(
            df, x_col='CPU Inf (ms)', y_col='Val Acc',
            x_label=self.t['cpu_time_ms'], y_label=self.t['val_acc'],
            title=self.t['pareto_val_acc_cpu'],
            save_path=self.output_root / 'pareto_cpu_time' / 'pareto_val_acc_cpu.png'
        )
        
        self._plot_pareto(
            df, x_col='CPU Inf (ms)', y_col='Best Full F1',
            x_label=self.t['cpu_time_ms'], y_label=self.t['full_f1'],
            title=self.t['pareto_full_f1_cpu'],
            save_path=self.output_root / 'pareto_cpu_time' / 'pareto_full_f1_cpu.png'
        )
        
        self._plot_pareto(
            df, x_col='CPU Inf (ms)', y_col='Best Full Acc',
            x_label=self.t['cpu_time_ms'], y_label=self.t['full_acc'],
            title=self.t['pareto_full_acc_cpu'],
            save_path=self.output_root / 'pareto_cpu_time' / 'pareto_full_acc_cpu.png'
        )
        
        # По Params
        self._plot_pareto(
            df, x_col='Params', y_col='Val F1',
            x_label=self.t['params'], y_label=self.t['val_f1'],
            title=self.t['pareto_val_f1_params'],
            save_path=self.output_root / 'pareto_params' / 'pareto_val_f1_params.png'
        )
        
        self._plot_pareto(
            df, x_col='Params', y_col='Val Acc',
            x_label=self.t['params'], y_label=self.t['val_acc'],
            title=self.t['pareto_val_acc_params'],
            save_path=self.output_root / 'pareto_params' / 'pareto_val_acc_params.png'
        )
        
        self._plot_pareto(
            df, x_col='Params', y_col='Best Full F1',
            x_label=self.t['params'], y_label=self.t['full_f1'],
            title=self.t['pareto_full_f1_params'],
            save_path=self.output_root / 'pareto_params' / 'pareto_full_f1_params.png'
        )
        
        self._plot_pareto(
            df, x_col='Params', y_col='Best Full Acc',
            x_label=self.t['params'], y_label=self.t['full_acc'],
            title=self.t['pareto_full_acc_params'],
            save_path=self.output_root / 'pareto_params' / 'pareto_full_acc_params.png'
        )
        
        print(f"[Pareto] Сохранено 8 графиков Парето в {self.output_root}")
    
    def _plot_pareto(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        save_path: Path
    ):
        """Строит один график Парето."""
        # Фильтруем данные
        df_plot = df[(df[x_col] > 0) & (df[y_col] > 0)].copy()
        if df_plot.empty:
            warnings.warn(f"Нет данных для графика: {title}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Фиксированный порядок моделей и сложностей для согласованности между графиками
        model_order = ['MLP', 'CNN', 'ResNet', 'SimpleKAN', 'ConvKAN', 'PhysicsKAN', 'Unknown']
        complexity_order = ['Light', 'Medium', 'Heavy', 'Unknown']
        
        # Scatter plot с разными маркерами для сложности
        for complexity in complexity_order:
            if complexity not in df_plot['Complexity'].unique():
                continue
            df_c = df_plot[df_plot['Complexity'] == complexity]
            marker = self.COMPLEXITY_MARKERS.get(complexity, 'o')
            
            for model in model_order:
                if model not in df_c['Model'].unique():
                    continue
                df_m = df_c[df_c['Model'] == model]
                color = self.MODEL_COLORS.get(model, self.MODEL_COLORS['Unknown'])
                
                # Унифицированный формат метки: Model (L/M/H)
                complexity_short = complexity[0] if complexity and complexity != 'Unknown' else '?'
                label = f"{model} ({complexity_short})"
                
                plt.scatter(
                    df_m[x_col], df_m[y_col],
                    c=color, marker=marker,
                    s=100, alpha=0.7,
                    label=label
                )
        
        # Выделяем фронт Парето (опционально)
        pareto_front = self._compute_pareto_front(df_plot, x_col, y_col)
        if len(pareto_front) > 1:
            pareto_df = df_plot.iloc[pareto_front].sort_values(x_col)
            plt.plot(
                pareto_df[x_col], pareto_df[y_col],
                'k--', alpha=0.5, linewidth=1.5,
                label='Фронт Парето' if self.lang == 'ru' else 'Pareto Front'
            )
        
        plt.xscale('log')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, which='both', alpha=0.3)
        
        # Легенда - сортируем для согласованности между графиками
        handles, labels = plt.gca().get_legend_handles_labels()
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        
        plt.legend(
            unique_handles, unique_labels,
            bbox_to_anchor=(1.05, 1), loc='upper left',
            fontsize=9
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _compute_pareto_front(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        minimize_x: bool = True,
        maximize_y: bool = True
    ) -> List[int]:
        """
        Вычисляет индексы точек на фронте Парето.
        
        По умолчанию: минимизируем X (время/параметры), максимизируем Y (точность).
        """
        points = df[[x_col, y_col]].values
        n = len(points)
        
        if minimize_x:
            points[:, 0] = -points[:, 0]  # Инвертируем для минимизации
        if not maximize_y:
            points[:, 1] = -points[:, 1]
        
        # Находим недоминируемые точки
        pareto_idx = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # j доминирует i, если j >= i по всем и j > i хотя бы по одному
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)
        
        return pareto_idx
    
    # =========================================================================
    # ЧАСТЬ 2: ГРАФИКИ ОБУЧЕНИЯ ПО СЛОЖНОСТИ
    # =========================================================================
    
    def plot_learning_curves_by_experiment(
        self,
        df: pd.DataFrame,
        histories: Dict[str, List[Dict[str, Any]]]
    ):
        """
        Строит графики обучения, группируя ВСЕ модели одного эксперимента вместе.
        
        Принцип группировки: "все параметры одинаковые, отличается только тип модели".
        То есть на одном графике - все модели с одинаковым ExpID.
        
        Args:
            df: DataFrame с информацией об экспериментах
            histories: Словарь {exp_name: [метрики по эпохам]}
        """
        unique_exp_ids = df['ExpID'].unique()
        
        for exp_id in unique_exp_ids:
            if exp_id == 'Unknown':
                continue
            
            exp_df = df[df['ExpID'] == exp_id]
            
            # Собираем истории для всех моделей этого эксперимента
            exp_histories = {}
            for _, row in exp_df.iterrows():
                exp_name = row['Path']
                if exp_name in histories:
                    exp_histories[exp_name] = histories[exp_name]
            
            if not exp_histories:
                continue
            
            self._plot_experiment_learning_curves(exp_id, exp_histories, exp_df)
        
        print(f"[Learning Curves] Сохранены графики обучения в {self.output_root / 'learning_curves'}")
    
    def _plot_experiment_learning_curves(
        self,
        exp_id: str,
        histories: Dict[str, List[Dict]],
        df_info: pd.DataFrame
    ):
        """Строит график обучения для одного эксперимента (все модели вместе)."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        metrics_config = [
            ('val_loss', self.t['loss'], 'Val Loss', True),   # clip_upper
            ('val_f1', self.t['f1'], 'Val F1', False),
            ('val_acc', self.t['acc'], 'Val Accuracy', False)
        ]
        
        legend_handles = []
        legend_labels = []
        
        for exp_name, history in histories.items():
            if not history:
                continue
            
            df_h = pd.DataFrame(history)
            
            # Определяем модель из DataFrame
            model_row = df_info[df_info['Path'] == exp_name]
            if not model_row.empty:
                model_name = model_row.iloc[0]['Model']
                complexity = model_row.iloc[0].get('Complexity', 'Unknown')
                arch_type = model_row.iloc[0].get('arch_type', 'Base')
            else:
                model_name = 'Unknown'
                complexity = 'Unknown'
                arch_type = 'Base'
            
            color = self.MODEL_COLORS.get(model_name, self.MODEL_COLORS['Unknown'])
            # Разные стили линий для сложности
            ls_map = {'Light': '-', 'Medium': '--', 'Heavy': ':'}
            ls = ls_map.get(complexity, '-')
            
            # Пунктирнее для Hierarchical
            if arch_type == 'Hierarchical':
                ls = (0, (3, 1, 1, 1))  # dashdotdot
            
            # Метка: Model (Complexity)
            label = f"{model_name} ({complexity[0] if complexity else '?'})"
            
            for ax, (metric, ylabel, metric_name, clip_upper) in zip(axes, metrics_config):
                if metric in df_h.columns:
                    values = df_h[metric]
                    if clip_upper:
                        values = values.clip(upper=2.0)
                    line, = ax.plot(
                        df_h['epoch'], values,
                        color=color, linestyle=ls, alpha=0.8, linewidth=1.5
                    )
            
            # Сохраняем для легенды (только один раз на модель)
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=1.5))
            legend_labels.append(label)
        
        # Настройка осей
        for ax, (metric, ylabel, metric_name, _) in zip(axes, metrics_config):
            ax.set_xlabel(self.t['epoch'])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric_name}")
            ax.grid(True, alpha=0.3)
            if metric in ['val_f1', 'val_acc']:
                ax.set_ylim([0, 1.0])
        
        # Легенда справа от последнего графика
        # Убираем дубликаты
        unique_labels = {}
        for h, l in zip(legend_handles, legend_labels):
            if l not in unique_labels:
                unique_labels[l] = h
        
        axes[-1].legend(
            unique_labels.values(), unique_labels.keys(),
            bbox_to_anchor=(1.05, 1), loc='upper left',
            fontsize=9
        )
        
        # Определяем параметры эксперимента для заголовка
        if not df_info.empty:
            first_row = df_info.iloc[0]
            features = first_row.get('Features', '?')
            sampling = first_row.get('Sampling', '?')
            title_info = f"Features: {features}, Sampling: {sampling}"
        else:
            title_info = ""
        
        fig.suptitle(
            f"Exp {exp_id}: Кривые обучения ({title_info})" if self.lang == 'ru' 
            else f"Exp {exp_id}: Learning Curves ({title_info})",
            fontsize=14
        )
        
        plt.tight_layout()
        
        # Создаём папку и сохраняем
        save_dir = self.output_root / 'learning_curves'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'exp_{exp_id}_all_models.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Старый метод оставляем для совместимости, но теперь вызываем новый
    def plot_learning_curves_by_complexity(
        self,
        df: pd.DataFrame,
        histories: Dict[str, List[Dict[str, Any]]]
    ):
        """
        Строит графики обучения, группируя модели по эксперименту.
        
        ИЗМЕНЕНО: Теперь группирует по ExpID, а не по сложности.
        Все модели одного эксперимента на одном графике.
        
        Args:
            df: DataFrame с информацией об экспериментах
            histories: Словарь {exp_name: [метрики по эпохам]}
        """
        self.plot_learning_curves_by_experiment(df, histories)
    
    def _plot_complexity_learning_curves(
        self,
        exp_id: str,
        complexity: str,
        histories: Dict[str, List[Dict]],
        df_info: pd.DataFrame
    ):
        """Строит график обучения для конкретной сложности."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_config = [
            ('val_loss', self.t['loss'], 'Loss', True),   # clip_upper
            ('val_f1', self.t['f1'], 'F1', False),
            ('val_acc', self.t['acc'], 'Accuracy', False)
        ]
        
        for exp_name, history in histories.items():
            if not history:
                continue
            
            df_h = pd.DataFrame(history)
            
            # Определяем модель из DataFrame
            model_row = df_info[df_info['Path'] == exp_name]
            if not model_row.empty:
                model_name = model_row.iloc[0]['Model']
                arch_type = model_row.iloc[0].get('arch_type', 'Base')
            else:
                model_name = 'Unknown'
                arch_type = 'Base'
            
            color = self.MODEL_COLORS.get(model_name, self.MODEL_COLORS['Unknown'])
            ls = '--' if arch_type == 'Hierarchical' else '-'
            arch_suffix = 'H' if arch_type == 'Hierarchical' else 'B'
            label = f"{model_name} ({arch_suffix})"
            
            for ax, (metric, ylabel, metric_name, clip_upper) in zip(axes, metrics_config):
                if metric in df_h.columns:
                    values = df_h[metric]
                    if clip_upper:
                        values = values.clip(upper=2.0)
                    ax.plot(
                        df_h['epoch'], values,
                        label=label, color=color,
                        linestyle=ls, alpha=0.8
                    )
        
        # Настройка осей
        for ax, (metric, ylabel, metric_name, _) in zip(axes, metrics_config):
            ax.set_xlabel(self.t['epoch'])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric_name}")
            ax.grid(True, alpha=0.3)
            if metric in ['val_f1', 'val_acc']:
                ax.set_ylim([0, 1.0])
        
        # Легенда на последнем графике
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle(
            self.t['learning_title'].format(exp_id, complexity),
            fontsize=14
        )
        
        plt.tight_layout()
        save_path = self.output_root / 'learning_curves' / complexity.lower() / f'exp_{exp_id}_{complexity.lower()}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # ЧАСТЬ 3: ПАРЕТО ПО ТИПАМ ДАННЫХ ДЛЯ КАЖДОЙ МОДЕЛИ
    # =========================================================================
    
    def plot_pareto_by_data_type(self, df: pd.DataFrame):
        """
        Строит графики Парето для каждой модели отдельно.
        Цвет = тип данных (Features), маркер = sampling стратегия.
        
        Ось X: CPU время инференса
        Ось Y: лучшая из Full Best F1 и Full Final F1
        """
        df = df.copy()
        df['Best Full F1'] = df.apply(lambda r: self._get_best_full_metric(r, 'f1'), axis=1)
        
        # Фильтруем модели с данными
        df_valid = df[(df['CPU Inf (ms)'] > 0) & (df['Best Full F1'] > 0)]
        
        unique_models = df_valid['Model'].unique()
        
        for model in unique_models:
            df_model = df_valid[df_valid['Model'] == model]
            if df_model.empty:
                continue
            
            self._plot_model_data_type_pareto(model, df_model)
        
        print(f"[Data Type Analysis] Сохранены графики в {self.output_root / 'data_type_analysis'}")
    
    def _plot_model_data_type_pareto(self, model_name: str, df: pd.DataFrame):
        """Строит Парето-график для одной модели по типам данных."""
        plt.figure(figsize=(12, 8))
        
        # Создаём папку для модели
        model_dir = self.output_root / 'data_type_analysis' / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Строим точки с разными цветами для Features и маркерами для Sampling
        for feature_type in df['Features'].unique():
            df_f = df[df['Features'] == feature_type]
            color = self.FEATURE_COLORS.get(feature_type, self.FEATURE_COLORS['Unknown'])
            
            for sampling in df_f['Sampling'].unique():
                df_s = df_f[df_f['Sampling'] == sampling]
                marker = self.SAMPLING_MARKERS.get(sampling, self.SAMPLING_MARKERS['Unknown'])
                
                plt.scatter(
                    df_s['CPU Inf (ms)'],
                    df_s['Best Full F1'],
                    c=color, marker=marker,
                    s=120, alpha=0.7,
                    label=f"{feature_type} / {sampling}"
                )
                
                # Добавляем подписи сложности
                for _, row in df_s.iterrows():
                    complexity_label = row['Complexity'][0] if row['Complexity'] else '?'
                    plt.annotate(
                        complexity_label,
                        (row['CPU Inf (ms)'], row['Best Full F1']),
                        textcoords='offset points',
                        xytext=(5, 5),
                        fontsize=8,
                        alpha=0.7
                    )
        
        plt.xscale('log')
        plt.xlabel(self.t['cpu_time_ms'], fontsize=12)
        plt.ylabel(self.t['full_f1'], fontsize=12)
        plt.title(self.t['data_type_title'].format(model_name), fontsize=14)
        plt.grid(True, which='both', alpha=0.3)
        
        # Легенда
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(), by_label.keys(),
            bbox_to_anchor=(1.05, 1), loc='upper left',
            fontsize=9, title=f"{self.t['features']} / {self.t['sampling']}"
        )
        
        plt.tight_layout()
        plt.savefig(model_dir / f'pareto_data_types_{model_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # ЧАСТЬ 4: ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ
    # =========================================================================
    
    # Названия классов для radar charts
    CLASS_NAMES = {
        'ru': ['Норма', 'Коммутации', 'Аномалии', 'Аварии'],
        'en': ['Normal', 'Switching', 'Abnormal', 'Fault']
    }
    
    def plot_additional_visualizations(self, df: pd.DataFrame):
        """
        Строит дополнительные полезные визуализации.
        
        Включает:
        1. Heatmap: Модель x Тип данных (средняя F1)
        2. Heatmap: Модель x Тип данных (лучшая F1)
        3. Boxplot: Распределение F1 по моделям
        4. Radar chart: Сравнение топ-моделей по классам (если есть per_class данные)
        5. Ranking: Таблица-рейтинг моделей
        """
        df = df.copy()
        df['Best Full F1'] = df.apply(lambda r: self._get_best_full_metric(r, 'f1'), axis=1)
        
        # Средние heatmaps
        self._plot_heatmap_model_features(df, aggfunc='mean', suffix='mean')
        self._plot_heatmap_model_sampling(df, aggfunc='mean', suffix='mean')
        
        # Лучшие heatmaps
        self._plot_heatmap_model_features(df, aggfunc='max', suffix='best')
        self._plot_heatmap_model_sampling(df, aggfunc='max', suffix='best')
        
        # Boxplots
        self._plot_boxplot_f1_by_model(df)
        self._plot_boxplot_f1_by_features(df)
        self._plot_complexity_comparison(df)
        
        # Radar charts (если есть per-class F1)
        self._plot_radar_charts(df)
        
        # Rankings
        self._save_ranking_table(df)
        
        print(f"[Additional] Сохранены дополнительные графики в {self.output_root / 'additional'}")
    
    def _plot_heatmap_model_features(self, df: pd.DataFrame, aggfunc: str = 'mean', suffix: str = 'mean'):
        """Heatmap: Модель vs Тип данных."""
        pivot = df.pivot_table(
            values='Best Full F1',
            index='Model',
            columns='Features',
            aggfunc=aggfunc
        )
        
        if pivot.empty:
            return
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot, annot=True, fmt='.3f',
            cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.5
        )
        
        title_map = {
            'mean': ('Средняя F1 по типам данных', 'Mean F1 by Feature Type'),
            'best': ('Лучшая F1 по типам данных', 'Best F1 by Feature Type'),
            'max': ('Лучшая F1 по типам данных', 'Best F1 by Feature Type')
        }
        title = title_map.get(suffix, title_map['mean'])
        plt.title(title[0] if self.lang == 'ru' else title[1], fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/heatmaps' / f'heatmap_model_features_{suffix}.png', dpi=150)
        plt.close()
    
    def _plot_heatmap_model_sampling(self, df: pd.DataFrame, aggfunc: str = 'mean', suffix: str = 'mean'):
        """Heatmap: Модель vs Sampling."""
        pivot = df.pivot_table(
            values='Best Full F1',
            index='Model',
            columns='Sampling',
            aggfunc=aggfunc
        )
        
        if pivot.empty:
            return
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot, annot=True, fmt='.3f',
            cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.5
        )
        
        title_map = {
            'mean': ('Средняя F1 по стратегии Sampling', 'Mean F1 by Sampling Strategy'),
            'best': ('Лучшая F1 по стратегии Sampling', 'Best F1 by Sampling Strategy'),
            'max': ('Лучшая F1 по стратегии Sampling', 'Best F1 by Sampling Strategy')
        }
        title = title_map.get(suffix, title_map['mean'])
        plt.title(title[0] if self.lang == 'ru' else title[1], fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/heatmaps' / f'heatmap_model_sampling_{suffix}.png', dpi=150)
        plt.close()
    
    def _plot_radar_charts(self, df: pd.DataFrame):
        """
        Строит radar charts для сравнения моделей по классам.
        
        Требует наличия колонок: Class_0_F1, Class_1_F1, Class_2_F1, Class_3_F1
        """
        # Проверяем наличие per-class метрик
        per_class_cols = ['Class_0_F1', 'Class_1_F1', 'Class_2_F1', 'Class_3_F1']
        if not all(col in df.columns for col in per_class_cols):
            print("  [Radar] Пропуск: нет per-class метрик в данных")
            return
        
        # Фильтруем модели с валидными данными
        df_valid = df[df[per_class_cols].notna().all(axis=1)]
        if df_valid.empty:
            print("  [Radar] Пропуск: нет моделей с валидными per-class метриками")
            return
        
        # Выбираем топ-5 моделей по общей F1
        top_models = df_valid.nlargest(5, 'Best Full F1')
        
        # Radar chart
        self._plot_radar_top_models(top_models, per_class_cols)

        # Radar charts: топ-3 внутри каждой модели
        self._plot_radar_top3_per_model(df_valid, per_class_cols)
        
        # Сравнение по типам моделей (усреднённое)
        self._plot_radar_by_model_type(df_valid, per_class_cols)
        
        print(f"  [Radar] Сохранены radar charts в {self.output_root / 'additional/radar'}")
    
    def _plot_radar_top_models(self, df: pd.DataFrame, class_cols: List[str]):
        """Radar chart для топ-5 моделей."""
        class_names = self.CLASS_NAMES[self.lang]
        
        # Количество осей
        num_vars = len(class_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for idx, (_, row) in enumerate(df.iterrows()):
            values = [row[col] for col in class_cols]
            values += values[:1]  # Замыкаем
            
            model_name = row['Model']
            complexity = row.get('Complexity', '?')
            color = self.MODEL_COLORS.get(model_name, self.MODEL_COLORS['Unknown'])
            
            label = f"{model_name} ({complexity[0] if complexity else '?'})"
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(
            'Топ-5 моделей: F1 по классам' if self.lang == 'ru' else 'Top-5 Models: F1 by Class',
            fontsize=14, y=1.08
        )
        
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/radar' / 'radar_top5_models.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_radar_top3_per_model(self, df: pd.DataFrame, class_cols: List[str]):
        """Radar chart: топ-3 конфигурации внутри каждой модели."""
        class_names = self.CLASS_NAMES[self.lang]

        # Группируем по типу модели
        grouped = df.groupby('Model')
        if grouped.ngroups == 0:
            return

        num_vars = len(class_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        for model_name, group_df in grouped:
            top3 = group_df.nlargest(3, 'Best Full F1')
            if len(top3) < 2:
                # Для одиночной модели радар не информативен
                continue

            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
            palette = sns.color_palette('tab10', n_colors=len(top3))

            for idx, (_, row) in enumerate(top3.iterrows()):
                values = [row[col] for col in class_cols]
                values += values[:1]

                complexity = row.get('Complexity', '?')
                features = row.get('Features', '?')
                sampling = row.get('Sampling', '?')
                label = f"{complexity} | {features} | {sampling}"

                color = palette[idx]
                ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(class_names, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax.grid(True)

            plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
            plt.title(
                f"Топ-3 конфигурации: {model_name}" if self.lang == 'ru' else f"Top-3 Configs: {model_name}",
                fontsize=14, y=1.08
            )

            safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
            out_path = self.output_root / 'additional/radar' / f"radar_top3_{safe_name}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_radar_by_model_type(self, df: pd.DataFrame, class_cols: List[str]):
        """Radar chart: усреднённые значения по типам моделей."""
        class_names = self.CLASS_NAMES[self.lang]
        
        # Группируем по типу модели
        grouped = df.groupby('Model')[class_cols].mean()
        
        if grouped.empty:
            return
        
        num_vars = len(class_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for model_name, row in grouped.iterrows():
            values = row.tolist()
            values += values[:1]
            
            color = self.MODEL_COLORS.get(model_name, self.MODEL_COLORS['Unknown'])
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(
            'Средняя F1 по классам (по типам моделей)' if self.lang == 'ru' 
            else 'Mean F1 by Class (by Model Type)',
            fontsize=14, y=1.08
        )
        
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/radar' / 'radar_by_model_type.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplot_f1_by_model(self, df: pd.DataFrame):
        """Boxplot: распределение F1 по моделям."""
        plt.figure(figsize=(12, 6))
        
        order = df.groupby('Model')['Best Full F1'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df, x='Model', y='Best Full F1',
            order=order, palette=self.MODEL_COLORS
        )
        sns.stripplot(
            data=df, x='Model', y='Best Full F1',
            order=order, color='black', alpha=0.5, size=4
        )
        
        plt.title(
            'Распределение F1 по моделям' if self.lang == 'ru' else 'F1 Distribution by Model',
            fontsize=14
        )
        plt.ylabel(self.t['full_f1'])
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/boxplots' / 'boxplot_f1_by_model.png', dpi=150)
        plt.close()
    
    def _plot_boxplot_f1_by_features(self, df: pd.DataFrame):
        """Boxplot: распределение F1 по типам данных."""
        plt.figure(figsize=(12, 6))
        
        order = df.groupby('Features')['Best Full F1'].median().sort_values(ascending=False).index
        
        # Создаём palette для features
        palette = {k: self.FEATURE_COLORS.get(k, '#95a5a6') for k in order}
        
        sns.boxplot(
            data=df, x='Features', y='Best Full F1',
            order=order, palette=palette
        )
        sns.stripplot(
            data=df, x='Features', y='Best Full F1',
            order=order, color='black', alpha=0.5, size=4
        )
        
        plt.title(
            'Распределение F1 по типам данных' if self.lang == 'ru' else 'F1 Distribution by Feature Type',
            fontsize=14
        )
        plt.ylabel(self.t['full_f1'])
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/boxplots' / 'boxplot_f1_by_features.png', dpi=150)
        plt.close()
    
    def _plot_complexity_comparison(self, df: pd.DataFrame):
        """Сравнение моделей по сложности (групповой bar chart)."""
        # Группируем по Model и Complexity
        grouped = df.groupby(['Model', 'Complexity'])['Best Full F1'].mean().unstack()
        
        if grouped.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Порядок сложности
        complexity_order = ['Light', 'Medium', 'Heavy']
        cols = [c for c in complexity_order if c in grouped.columns]
        
        if not cols:
            return
        
        grouped[cols].plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_title(
            'Средняя F1 по моделям и сложности' if self.lang == 'ru' else 'Mean F1 by Model and Complexity',
            fontsize=14
        )
        ax.set_ylabel(self.t['full_f1'])
        ax.set_xlabel(self.t['model'])
        ax.set_ylim([0, 1])
        ax.legend(title=self.t['complexity'])
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_root / 'additional/boxplots' / 'complexity_comparison.png', dpi=150)
        plt.close()
    
    def _save_ranking_table(self, df: pd.DataFrame):
        """Сохраняет таблицу рейтинга моделей."""
        # Топ по Best Full F1
        ranking = df.nlargest(20, 'Best Full F1')[
            ['ExpID', 'Model', 'Complexity', 'Features', 'Sampling', 
             'Best Full F1', 'Val F1', 'CPU Inf (ms)', 'Params']
        ].copy()
        
        ranking = ranking.round({
            'Best Full F1': 4,
            'Val F1': 4,
            'CPU Inf (ms)': 2
        })
        
        # Сохраняем в CSV
        ranking.to_csv(
            self.output_root / 'additional/rankings' / 'top20_models.csv',
            index=False
        )
        
        # Сохраняем в markdown
        with open(self.output_root / 'additional/rankings' / 'top20_models.md', 'w', encoding='utf-8') as f:
            f.write("# Топ-20 моделей по Full Test F1\n\n")
            f.write(ranking.to_markdown(index=False))
    
    # =========================================================================
    # ОСНОВНОЙ МЕТОД: ГЕНЕРАЦИЯ ВСЕХ ГРАФИКОВ
    # =========================================================================
    
    def generate_all_plots(
        self,
        df: pd.DataFrame,
        histories: Optional[Dict[str, List[Dict]]] = None
    ):
        """
        Генерирует все графики.
        
        Args:
            df: DataFrame с результатами экспериментов
            histories: Словарь историй обучения (опционально)
        """
        print(f"\n=== Генерация расширенных графиков (язык: {self.lang}) ===")
        
        # 1. Графики Парето
        print("  [1/4] Графики Парето...")
        self.plot_all_pareto(df)
        
        # 2. Кривые обучения по сложности
        if histories:
            print("  [2/4] Кривые обучения по сложности...")
            self.plot_learning_curves_by_complexity(df, histories)
        else:
            print("  [2/4] Пропуск кривых обучения (нет данных истории)")
        
        # 3. Парето по типам данных для каждой модели
        print("  [3/4] Парето по типам данных...")
        self.plot_pareto_by_data_type(df)
        
        # 4. Дополнительные визуализации
        print("  [4/4] Дополнительные визуализации...")
        self.plot_additional_visualizations(df)
        
        print(f"\n=== Готово! Все графики сохранены в: {self.output_root} ===")


# =============================================================================
# ДЕМОНСТРАЦИЯ / ТЕСТИРОВАНИЕ
# =============================================================================

if __name__ == "__main__":
    # Пример использования
    import sys
    from pathlib import Path
    
    # Добавляем корень проекта
    ROOT_DIR = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    
    # Путь к summary_report.csv
    report_path = ROOT_DIR / "reports" / "Exp_2_5_and_start_Exp_2_6" / "summary_report.csv"
    
    if report_path.exists():
        print(f"Загрузка данных из {report_path}")
        df = pd.read_csv(report_path)
        
        # Создаём визуализатор
        output_dir = ROOT_DIR / "reports" / "Exp_2_5_and_start_Exp_2_6" / "figures_advanced"
        viz = AdvancedVisualizer(output_dir, lang='ru')
        
        # Генерируем графики (без histories, т.к. они не загружены)
        viz.generate_all_plots(df, histories=None)
    else:
        print(f"Файл отчёта не найден: {report_path}")
        print("Запустите сначала aggregate_reports.py для генерации summary_report.csv")
