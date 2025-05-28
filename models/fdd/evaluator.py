from abc import ABC
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import json
import os

from common.utils import get_available_device
from datasets.base_dataset import SlidingWindowDataset


class FDDEvaluator(ABC):
    """Enhanced evaluator for FDD dataset with comprehensive metrics."""

    def __init__(self, config, model, dataset, experiment_dir=None):
        """
        Initialize the evaluator.

        Args:
            config: Configuration dictionary
            model: PyTorch model
            dataset: FDD dataset
            experiment_dir: Directory to save experiment results
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = get_available_device()
        self.model = model
        self.dataset = dataset
        self.test_loader = None
        self.results_path = None

    def setup(self):
        """Set up the model, dataset and data loaders."""

        # Load parameters
        window_size = self.config.get('data', {}).get('window_size', None)
        batch_size = self.config.get('training', {}).get('batch_size', 128)
        stride = self.config.get('data', {}).get('stride', 10)

        # Move model to device
        self.model = self.model.to(self.device)

        # Create dataloader
        test_dataset = SlidingWindowDataset(
            df=self.dataset.df[self.dataset.test_mask],
            target=self.dataset.target[self.dataset.test_mask].astype(int),
            window_size=window_size,
            stride=stride
        )
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f" Тестовый набор: {len(test_dataset)} окон")

    def evaluate(self):
        """Evaluate the model on the test dataset with comprehensive metrics."""
        print(" Начало расширенной оценки модели...")

        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            # Progress bar
            pbar = tqdm(self.test_loader, desc="Evaluating on test set")

            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device).long()

                # Forward pass
                output = self.model(data)

                # Get probabilities
                probabilities = torch.softmax(output, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())

                # Get targets
                all_targets.extend(target.cpu().numpy())

                # Get class predictions
                _, preds = torch.max(output, 1)
                all_predictions.extend(preds.detach().cpu().numpy())

        # Convert to numpy arrays for metric calculation
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate comprehensive metrics
        metrics = self._calculate_all_metrics(all_targets, all_predictions, all_probabilities)

        # Create detailed results
        results = {
            'metrics': metrics,
            'data_info': {
                'total_samples': len(all_targets),
                'class_distribution': {
                    'class_0': int(np.sum(all_targets == 0)),
                    'class_1': int(np.sum(all_targets == 1)),
                    'class_2': int(np.sum(all_targets == 2))
                }
            },
            'predictions_summary': {
                'predicted_class_0': int(np.sum(all_predictions == 0)),
                'predicted_class_1': int(np.sum(all_predictions == 1)),
                'predicted_class_2': int(np.sum(all_predictions == 2))
            }
        }

        # Print detailed results
        self._print_detailed_results(metrics)

        # Save detailed results
        if self.experiment_dir:
            self._save_detailed_results(results)

        return results

    def _calculate_all_metrics(self, y_true, y_pred, y_prob):
        """Calculate all metrics including F1, accuracy, precision, recall, ROC-AUC"""

        metrics = {}

        # === BASIC METRICS ===

        # Accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # F1-scores (per class and averages)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        metrics['f1_scores'] = f1_per_class.tolist()
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))

        # Precision (per class and averages)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['precision_scores'] = precision_per_class.tolist()
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))

        # Recall (per class and averages)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_scores'] = recall_per_class.tolist()
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))

        # === ROC-AUC METRICS ===

        try:
            # Multi-class ROC-AUC (One-vs-Rest)
            metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo'))

            # Per-class ROC-AUC (One-vs-Rest)
            roc_auc_per_class = []
            for class_idx in range(3):
                # Binarize the problem for each class
                y_true_binary = (y_true == class_idx).astype(int)
                y_prob_binary = y_prob[:, class_idx]

                if len(np.unique(y_true_binary)) > 1:  # Check if both classes are present
                    auc = roc_auc_score(y_true_binary, y_prob_binary)
                    roc_auc_per_class.append(float(auc))
                else:
                    roc_auc_per_class.append(0.0)  # or np.nan

            metrics['roc_auc_per_class'] = roc_auc_per_class

        except Exception as e:
            print(f"⚠  Warning: Could not calculate ROC-AUC: {e}")
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
            metrics['roc_auc_per_class'] = [0.0, 0.0, 0.0]

        # === CONFUSION MATRIX ===
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()

        # === DETAILED CLASSIFICATION REPORT ===
        class_names = ['Normal', 'Anomaly', 'Emergency']
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['classification_report'] = class_report

        return metrics

    def _print_detailed_results(self, metrics):
        """Print comprehensive results in a readable format"""

        print("\n" + "=" * 80)
        print(" ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("=" * 80)

        # Overall Accuracy
        print(f" Общая точность (Accuracy): {metrics['accuracy']:.4f}")

        print("\n МЕТРИКИ ПО КЛАССАМ:")
        print("-" * 80)
        print(f"{'Класс':<12} {'F1-Score':<10} {'Precision':<12} {'Recall':<10} {'ROC-AUC':<10}")
        print("-" * 80)

        class_names = ['Норма (0)', 'Аномалии (1)', 'Аварии (2)']
        for i, class_name in enumerate(class_names):
            f1 = metrics['f1_scores'][i]
            precision = metrics['precision_scores'][i]
            recall = metrics['recall_scores'][i]
            roc_auc = metrics['roc_auc_per_class'][i]

            print(f"{class_name:<12} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f} {roc_auc:<10.4f}")

        print("\n УСРЕДНЕННЫЕ МЕТРИКИ:")
        print("-" * 50)
        print(f"F1 Macro:        {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted:     {metrics['f1_weighted']:.4f}")
        print(f"Precision Macro: {metrics['precision_macro']:.4f}")
        print(f"Recall Macro:    {metrics['recall_macro']:.4f}")
        print(f"ROC-AUC OvR:     {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC-AUC OvO:     {metrics['roc_auc_ovo']:.4f}")

        # Highlight critical class (Emergency)
        print(f"\n КРИТИЧЕСКИЙ КЛАСС (АВАРИИ):")
        print("-" * 40)
        print(f"F1-Score:  {metrics['f1_scores'][2]:.4f}")
        print(f"Precision: {metrics['precision_scores'][2]:.4f}")
        print(f"Recall:    {metrics['recall_scores'][2]:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc_per_class'][2]:.4f}")

        # Confusion Matrix
        print(f"\n МАТРИЦА ОШИБОК:")
        print("-" * 30)
        conf_matrix = np.array(metrics['confusion_matrix'])

        print("        Predicted")
        print("      0     1     2")
        for i, row in enumerate(conf_matrix):
            print(f"  {i} {row[0]:5d} {row[1]:5d} {row[2]:5d}")

        print("=" * 80)

    def _save_detailed_results(self, results):
        """Save detailed results to JSON files"""

        # Create metrics directory
        metrics_dir = os.path.join(self.experiment_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Save main results
        results_file = os.path.join(metrics_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Save summary for easy comparison
        summary = {
            'accuracy': results['metrics']['accuracy'],
            'f1_scores': results['metrics']['f1_scores'],
            'precision_scores': results['metrics']['precision_scores'],
            'recall_scores': results['metrics']['recall_scores'],
            'roc_auc_per_class': results['metrics']['roc_auc_per_class'],
            'f1_macro': results['metrics']['f1_macro'],
            'roc_auc_ovr': results['metrics']['roc_auc_ovr'],
            'emergency_class_metrics': {
                'f1': results['metrics']['f1_scores'][2],
                'precision': results['metrics']['precision_scores'][2],
                'recall': results['metrics']['recall_scores'][2],
                'roc_auc': results['metrics']['roc_auc_per_class'][2]
            }
        }

        summary_file = os.path.join(metrics_dir, "metrics_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f" Результаты сохранены:")
        print(f"   - Полные: {results_file}")
        print(f"   - Сводка: {summary_file}")

    def get_metrics_for_comparison(self):
        """Return key metrics for model comparison"""
        results = self.evaluate()
        metrics = results['metrics']

        return {
            'model_name': self.config.get('model', {}).get('name', 'Unknown'),
            'accuracy': metrics['accuracy'],
            'f1_emergency': metrics['f1_scores'][2],
            'precision_emergency': metrics['precision_scores'][2],
            'recall_emergency': metrics['recall_scores'][2],
            'roc_auc_emergency': metrics['roc_auc_per_class'][2],
            'f1_macro': metrics['f1_macro'],
            'roc_auc_ovr': metrics['roc_auc_ovr']
        }

    def analyze_classification_thresholds(self, y_true, y_prob):
        """Анализ оптимальных порогов для классификации аварий"""

        print("\n АНАЛИЗ ПОРОГОВ КЛАССИФИКАЦИИ ДЛЯ АВАРИЙ")
        print("-" * 60)

        # Анализ для класса аварий (класс 2)
        emergency_class = 2

        # Бинарная задача: аварии vs не-аварии
        y_true_binary = (y_true == emergency_class).astype(int)
        y_prob_emergency = y_prob[:, emergency_class]

        # Precision-Recall кривая
        precision, recall, thresholds_pr = precision_recall_curve(y_true_binary, y_prob_emergency)

        # Найти оптимальный порог (максимальный F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else 0.5
        optimal_f1 = f1_scores[optimal_idx]

        print(f" Стандартный порог (0.5):")
        y_pred_standard = (y_prob_emergency > 0.5).astype(int)
        f1_standard = f1_score(y_true_binary, y_pred_standard)
        precision_standard = precision_score(y_true_binary, y_pred_standard, zero_division=0)
        recall_standard = recall_score(y_true_binary, y_pred_standard, zero_division=0)

        print(f"   F1: {f1_standard:.4f}, Precision: {precision_standard:.4f}, Recall: {recall_standard:.4f}")

        print(f"\n Оптимальный порог ({optimal_threshold:.3f}):")
        y_pred_optimal = (y_prob_emergency > optimal_threshold).astype(int)
        f1_optimal = f1_score(y_true_binary, y_pred_optimal)
        precision_optimal = precision_score(y_true_binary, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_true_binary, y_pred_optimal, zero_division=0)

        print(f"   F1: {f1_optimal:.4f}, Precision: {precision_optimal:.4f}, Recall: {recall_optimal:.4f}")
        print(f"   Улучшение F1: {f1_optimal - f1_standard:+.4f}")

        # Анализ разных стратегий
        strategies = {
            'Conservative (high precision)': 0.8,  # Меньше ложных тревог
            'Balanced': optimal_threshold,  # Оптимальный F1
            'Aggressive (high recall)': 0.3  # Поймать больше аварий
        }

        print(f"\n СРАВНЕНИЕ СТРАТЕГИЙ:")
        print("-" * 60)
        print(f"{'Стратегия':<25} {'Порог':<8} {'F1':<8} {'Precision':<12} {'Recall':<8}")
        print("-" * 60)

        for strategy_name, threshold in strategies.items():
            y_pred_strategy = (y_prob_emergency > threshold).astype(int)
            f1_strategy = f1_score(y_true_binary, y_pred_strategy, zero_division=0)
            precision_strategy = precision_score(y_true_binary, y_pred_strategy, zero_division=0)
            recall_strategy = recall_score(y_true_binary, y_pred_strategy, zero_division=0)

            print(
                f"{strategy_name:<25} {threshold:<8.3f} {f1_strategy:<8.3f} {precision_strategy:<12.3f} {recall_strategy:<8.3f}")

        return {
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            'standard_f1': f1_standard,
            'improvement': f1_optimal - f1_standard
        }

    def create_performance_report(self, metrics, threshold_analysis=None):
        """Создает детальный отчет производительности"""

        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'name': self.config.get('model', {}).get('name', 'Unknown'),
                'window_size': self.config.get('data', {}).get('window_size', 'Unknown'),
                'loss_function': self.config.get('training', {}).get('loss_function', 'Unknown')
            },
            'key_metrics': {
                'accuracy': metrics['accuracy'],
                'f1_emergency': metrics['f1_scores'][2],
                'precision_emergency': metrics['precision_scores'][2],
                'recall_emergency': metrics['recall_scores'][2],
                'roc_auc_emergency': metrics['roc_auc_per_class'][2]
            },
            'all_classes_performance': {
                'normal': {
                    'f1': metrics['f1_scores'][0],
                    'precision': metrics['precision_scores'][0],
                    'recall': metrics['recall_scores'][0],
                    'roc_auc': metrics['roc_auc_per_class'][0]
                },
                'anomaly': {
                    'f1': metrics['f1_scores'][1],
                    'precision': metrics['precision_scores'][1],
                    'recall': metrics['recall_scores'][1],
                    'roc_auc': metrics['roc_auc_per_class'][1]
                },
                'emergency': {
                    'f1': metrics['f1_scores'][2],
                    'precision': metrics['precision_scores'][2],
                    'recall': metrics['recall_scores'][2],
                    'roc_auc': metrics['roc_auc_per_class'][2]
                }
            },
            'business_impact': {
                'false_emergency_rate': self._calculate_false_emergency_rate(metrics),
                'missed_emergency_rate': self._calculate_missed_emergency_rate(metrics),
                'system_reliability_score': self._calculate_reliability_score(metrics)
            }
        }

        if threshold_analysis:
            report['threshold_optimization'] = threshold_analysis

        return report

    def _calculate_false_emergency_rate(self, metrics):
        """Рассчитывает долю ложных тревог от всех предсказанных аварий"""
        # Это обратная величина precision для аварий
        precision_emergency = metrics['precision_scores'][2]
        return 1.0 - precision_emergency if precision_emergency > 0 else 1.0

    def _calculate_missed_emergency_rate(self, metrics):
        """Рассчитывает долю пропущенных аварий от всех реальных аварий"""
        # Это обратная величина recall для аварий
        recall_emergency = metrics['recall_scores'][2]
        return 1.0 - recall_emergency if recall_emergency > 0 else 1.0

    def _calculate_reliability_score(self, metrics):
        """Рассчитывает общий показатель надежности системы"""
        # Взвешенная комбинация метрик с акцентом на аварии
        weights = {'accuracy': 0.2, 'f1_emergency': 0.4, 'roc_auc_emergency': 0.4}

        score = (weights['accuracy'] * metrics['accuracy'] +
                 weights['f1_emergency'] * metrics['f1_scores'][2] +
                 weights['roc_auc_emergency'] * metrics['roc_auc_per_class'][2])

        return score