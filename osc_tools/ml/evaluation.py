"""
Модуль для многоуровневой оценки качества моделей (Multi-Level Evaluation).

Поддерживает оценку точности на разных уровнях гранулярности меток:
- Base (4 класса): Normal, ML_1 (Коммутации), ML_2 (Аномалии), ML_3 (Аварии)
- Level 1: ML_1, ML_2, ML_3 (без Normal)
- Level 2: ML_X_Y (подкатегории)
- Full: Все ML_* метки

Автор: Scientific Research OSC ML Team
Дата: 2026-01-22
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, balanced_accuracy_score,
    classification_report, multilabel_confusion_matrix
)
import polars as pl

from osc_tools.ml.labels import (
    get_ml_columns, 
    get_target_columns, 
    get_label_hierarchy,
    add_base_labels
)


class MultiLevelEvaluator:
    """
    Класс для многоуровневой оценки качества моделей.
    
    Позволяет:
    1. Оценивать модель на разных уровнях гранулярности
    2. Вычислять Hierarchical Accuracy (обучение на Full, проверка на Base)
    3. Сравнивать производительность на разных уровнях иерархии
    """
    
    def __init__(
        self,
        target_level: str = 'full',
        base_target_columns: List[str] = None,
        full_target_columns: List[str] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            target_level: Уровень, на котором обучалась модель ('base', 'full', 'full_by_levels')
            base_target_columns: Колонки базовых меток (4 класса)
            full_target_columns: Колонки полных меток (все ML_*)
            threshold: Порог для бинаризации предсказаний (для multilabel)
        """
        self.target_level = target_level
        self.base_columns = base_target_columns or ["Target_Normal", "Target_ML_1", "Target_ML_2", "Target_ML_3"]
        self.full_columns = full_target_columns or []
        self.threshold = threshold
        
        # Строим маппинг Full -> Base для Hierarchical Accuracy
        self._build_hierarchy_mapping()
        
    def _build_hierarchy_mapping(self):
        """
        Строит маппинг от полных меток к базовым.
        
        ML_1_*, ML_1 -> Target_ML_1
        ML_2_*, ML_2 -> Target_ML_2
        ML_3_*, ML_3 -> Target_ML_3
        """
        self.full_to_base_idx = {}
        
        for i, col in enumerate(self.full_columns):
            if col.startswith('ML_1'):
                self.full_to_base_idx[i] = 1  # Target_ML_1
            elif col.startswith('ML_2'):
                self.full_to_base_idx[i] = 2  # Target_ML_2
            elif col.startswith('ML_3'):
                self.full_to_base_idx[i] = 3  # Target_ML_3
                
    def _aggregate_to_base(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Агрегирует полные предсказания/метки в базовые (4 класса).
        
        Логика: если хотя бы одна подкатегория ML_X_* предсказана/активна,
        то базовая категория Target_ML_X также активна.
        
        Args:
            predictions: Предсказания модели (N, num_full_classes)
            targets: Истинные метки (N, num_full_classes)
            
        Returns:
            Tuple[base_predictions, base_targets] с формой (N, 4)
        """
        n_samples = predictions.shape[0]
        
        # Базовые метки: [Normal, ML_1, ML_2, ML_3]
        base_preds = np.zeros((n_samples, 4), dtype=np.float32)
        base_targets = np.zeros((n_samples, 4), dtype=np.float32)
        
        # Агрегируем по категориям
        for full_idx, base_idx in self.full_to_base_idx.items():
            if full_idx < predictions.shape[1]:
                base_preds[:, base_idx] = np.maximum(base_preds[:, base_idx], predictions[:, full_idx])
            if full_idx < targets.shape[1]:
                base_targets[:, base_idx] = np.maximum(base_targets[:, base_idx], targets[:, full_idx])
        
        # Normal = 1, если все остальные = 0
        base_preds[:, 0] = (base_preds[:, 1:].max(axis=1) == 0).astype(np.float32)
        base_targets[:, 0] = (base_targets[:, 1:].max(axis=1) == 0).astype(np.float32)
        
        return base_preds, base_targets
    
    def _aggregate_to_level(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        level: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Агрегирует предсказания до указанного уровня иерархии.
        
        Args:
            predictions: Полные предсказания
            targets: Полные метки
            level: Уровень агрегации (1 = ML_X, 2 = ML_X_Y)
            
        Returns:
            Tuple[level_predictions, level_targets, column_names]
        """
        # Определяем колонки нужного уровня
        level_cols = [c for c in self.full_columns if c.count('_') == level]
        level_col_indices = [self.full_columns.index(c) for c in level_cols if c in self.full_columns]
        
        if not level_col_indices:
            # Нет колонок этого уровня — возвращаем пустые
            return np.zeros((predictions.shape[0], 0)), np.zeros((targets.shape[0], 0)), []
        
        # Агрегируем: для каждой колонки уровня берем max по её поддереву
        n_samples = predictions.shape[0]
        level_preds = np.zeros((n_samples, len(level_cols)), dtype=np.float32)
        level_targets = np.zeros((n_samples, len(level_cols)), dtype=np.float32)
        
        for i, col in enumerate(level_cols):
            # Находим все дочерние колонки (начинающиеся с col_)
            child_indices = [
                self.full_columns.index(c) 
                for c in self.full_columns 
                if c.startswith(col + '_') or c == col
            ]
            
            # Агрегируем
            for idx in child_indices:
                if idx < predictions.shape[1]:
                    level_preds[:, i] = np.maximum(level_preds[:, i], predictions[:, idx])
                if idx < targets.shape[1]:
                    level_targets[:, i] = np.maximum(level_targets[:, i], targets[:, idx])
        
        return level_preds, level_targets, level_cols
    
    def evaluate_multilabel(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Вычисляет метрики для multilabel классификации.
        
        Args:
            predictions: Предсказания (N, num_classes) — вероятности или бинарные
            targets: Истинные метки (N, num_classes) — бинарные
            average: Тип усреднения для F1/Precision/Recall
            
        Returns:
            Словарь с метриками
        """
        # Бинаризация предсказаний
        preds_binary = (predictions >= self.threshold).astype(np.int32)
        targets_binary = targets.astype(np.int32)
        
        # Защита от пустых классов
        if targets_binary.sum() == 0:
            return {
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'exact_match': 0.0
            }
        
        metrics = {
            'f1': f1_score(targets_binary, preds_binary, average=average, zero_division=0),
            'precision': precision_score(targets_binary, preds_binary, average=average, zero_division=0),
            'recall': recall_score(targets_binary, preds_binary, average=average, zero_division=0),
            'accuracy': accuracy_score(targets_binary.flatten(), preds_binary.flatten()),
            'exact_match': (preds_binary == targets_binary).all(axis=1).mean()
        }
        
        return metrics
    
    def evaluate_per_class(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Вычисляет метрики для каждого класса отдельно.
        
        Returns:
            Словарь {class_name: {f1, precision, recall}}
        """
        preds_binary = (predictions >= self.threshold).astype(np.int32)
        targets_binary = targets.astype(np.int32)
        
        n_classes = predictions.shape[1]
        class_names = class_names or [f"Class_{i}" for i in range(n_classes)]
        
        per_class_metrics = {}
        
        for i, name in enumerate(class_names):
            if i >= n_classes:
                break
                
            per_class_metrics[name] = {
                'f1': f1_score(targets_binary[:, i], preds_binary[:, i], zero_division=0),
                'precision': precision_score(targets_binary[:, i], preds_binary[:, i], zero_division=0),
                'recall': recall_score(targets_binary[:, i], preds_binary[:, i], zero_division=0),
                'support': int(targets_binary[:, i].sum())
            }
        
        return per_class_metrics
    
    def compute_hierarchical_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """
        Вычисляет Hierarchical Accuracy — точность на базовом уровне
        при обучении на полном наборе меток.
        
        Это ключевая метрика для эксперимента 2.6.4.
        
        Returns:
            Словарь с:
            - base_metrics: метрики на уровне 4 классов
            - full_metrics: метрики на полном наборе
            - level1_metrics: метрики на уровне ML_X
            - level2_metrics: метрики на уровне ML_X_Y
            - per_class_base: метрики по каждому базовому классу
        """
        results = {}
        
        # 1. Метрики на полном уровне
        results['full_metrics'] = self.evaluate_multilabel(predictions, targets)
        results['full_per_class'] = self.evaluate_per_class(
            predictions, targets, self.full_columns
        )
        
        # 2. Агрегация до базовых 4 классов
        base_preds, base_targets = self._aggregate_to_base(predictions, targets)
        results['base_metrics'] = self.evaluate_multilabel(base_preds, base_targets)
        results['base_per_class'] = self.evaluate_per_class(
            base_preds, base_targets, self.base_columns
        )
        
        # 3. Метрики на Level 1 (ML_1, ML_2, ML_3)
        level1_preds, level1_targets, level1_cols = self._aggregate_to_level(predictions, targets, level=1)
        if level1_cols:
            results['level1_metrics'] = self.evaluate_multilabel(level1_preds, level1_targets)
            results['level1_per_class'] = self.evaluate_per_class(
                level1_preds, level1_targets, level1_cols
            )
        
        # 4. Метрики на Level 2 (ML_X_Y)
        level2_preds, level2_targets, level2_cols = self._aggregate_to_level(predictions, targets, level=2)
        if level2_cols:
            results['level2_metrics'] = self.evaluate_multilabel(level2_preds, level2_targets)
            results['level2_per_class'] = self.evaluate_per_class(
                level2_preds, level2_targets, level2_cols
            )
        
        return results
    
    def format_report(self, results: Dict[str, Any]) -> str:
        """
        Форматирует результаты многоуровневой оценки в читаемый отчёт.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("МНОГОУРОВНЕВАЯ ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
        lines.append("=" * 60)
        
        # Base Level
        if 'base_metrics' in results:
            lines.append("\n### Базовый уровень (4 класса) ###")
            m = results['base_metrics']
            lines.append(f"  F1 (macro): {m['f1']:.4f}")
            lines.append(f"  Precision:  {m['precision']:.4f}")
            lines.append(f"  Recall:     {m['recall']:.4f}")
            lines.append(f"  Exact Match:{m['exact_match']:.4f}")
            
            if 'base_per_class' in results:
                lines.append("\n  По классам:")
                for cls, metrics in results['base_per_class'].items():
                    lines.append(f"    {cls}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, N={metrics['support']}")
        
        # Full Level
        if 'full_metrics' in results:
            lines.append("\n### Полный уровень (все ML_*) ###")
            m = results['full_metrics']
            lines.append(f"  F1 (macro): {m['f1']:.4f}")
            lines.append(f"  Precision:  {m['precision']:.4f}")
            lines.append(f"  Recall:     {m['recall']:.4f}")
        
        # Level 1
        if 'level1_metrics' in results:
            lines.append("\n### Уровень 1 (ML_1, ML_2, ML_3) ###")
            m = results['level1_metrics']
            lines.append(f"  F1 (macro): {m['f1']:.4f}")
        
        # Level 2
        if 'level2_metrics' in results:
            lines.append("\n### Уровень 2 (ML_X_Y) ###")
            m = results['level2_metrics']
            lines.append(f"  F1 (macro): {m['f1']:.4f}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


def evaluate_model_multilevel(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    target_columns: List[str],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Удобная функция для многоуровневой оценки модели.
    
    Args:
        model: Обученная модель
        dataloader: DataLoader с тестовыми данными
        device: Устройство (CPU/CUDA)
        target_columns: Список целевых колонок (полный набор ML_*)
        threshold: Порог бинаризации
        
    Returns:
        Результаты многоуровневой оценки
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            
            # Применяем sigmoid для multilabel
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(probs)
            all_targets.append(y.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Создаем evaluator
    evaluator = MultiLevelEvaluator(
        target_level='full',
        full_target_columns=target_columns,
        threshold=threshold
    )
    
    # Вычисляем Hierarchical Accuracy
    results = evaluator.compute_hierarchical_accuracy(predictions, targets)
    
    return results


def compute_confusion_at_levels(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_columns: List[str]
) -> Dict[str, np.ndarray]:
    """
    Вычисляет confusion matrices на разных уровнях иерархии.
    
    Returns:
        Словарь {level_name: confusion_matrix}
    """
    evaluator = MultiLevelEvaluator(
        target_level='full',
        full_target_columns=target_columns
    )
    
    # Бинаризация
    preds_binary = (predictions >= 0.5).astype(np.int32)
    targets_binary = targets.astype(np.int32)
    
    results = {}
    
    # Full level confusion
    results['full'] = multilabel_confusion_matrix(targets_binary, preds_binary)
    
    # Base level confusion
    base_preds, base_targets = evaluator._aggregate_to_base(predictions, targets)
    base_preds_bin = (base_preds >= 0.5).astype(np.int32)
    base_targets_bin = base_targets.astype(np.int32)
    results['base'] = multilabel_confusion_matrix(base_targets_bin, base_preds_bin)
    
    return results
