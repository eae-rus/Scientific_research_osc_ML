"""
Оценка физической baseline-модели (детерминированный алгоритм ОЗЗ)
на тестовых данных с сохранением результатов в формате, совместимом
с нейросетевыми экспериментами.

Запуск:
    python scripts/evaluation/evaluate_physics_baseline.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.analysis.ozz_physics import predict_ozz_physics
from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.data_management.ozz_split import add_ozz_target_columns, OZZ_TARGET_COLS
from osc_tools.ml.labels import clean_labels


def _get_signal_cols(df: pl.DataFrame) -> List[str]:
    """Определяет 8 аналоговых колонок в стандартном порядке."""
    available = set(df.columns)
    i_cols = ['IA', 'IB', 'IC', 'IN']
    for candidates in [
        ['UA BB', 'UB BB', 'UC BB', 'UN BB'],
        ['UA CL', 'UB CL', 'UC CL', 'UN CL'],
        ['UA', 'UB', 'UC', 'UN'],
    ]:
        if all(c in available for c in candidates):
            return i_cols + candidates
    raise ValueError("Не найдены колонки напряжений")


def evaluate_physics_on_files(
    df: pl.DataFrame,
    window_size: int = 320,
    fs: int = 1600,
    target_window_mode: str = 'any_in_window',
    **physics_kwargs
) -> Dict[str, Any]:
    """
    Оценивает физическую модель пофайлово.

    Args:
        df: DataFrame с данными и Target_OZZ_* колонками.
        window_size: Размер окна.
        fs: Частота дискретизации.
        target_window_mode: 'point' или 'any_in_window'.

    Returns:
        Словарь с метриками и массивами предсказаний.
    """
    signal_cols = _get_signal_cols(df)
    target_cols = OZZ_TARGET_COLS

    # Проверяем наличие целевых колонок
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Колонка {col} отсутствует. Вызовите add_ozz_target_columns(df) сначала.")

    file_names = df['file_name'].unique().to_list()
    all_preds = []
    all_targets = []
    all_probs = []  # Для совместимости с NN — бинарные вероятности

    start_time = time.time()

    for fname in file_names:
        file_df = df.filter(pl.col('file_name') == fname)
        n = len(file_df)
        if n < window_size:
            continue

        signals = file_df.select(signal_cols).to_numpy().astype(np.float64)
        target_data = {col: file_df[col].to_numpy().astype(np.int8) for col in target_cols}

        # Предрассчитаем метки для any_in_window
        if target_window_mode == 'any_in_window':
            kernel = np.ones(window_size, dtype=np.int32)
            for col in target_cols:
                raw = target_data[col].astype(np.int32)
                counts = np.convolve(raw, kernel, mode='full')[:n]
                win_any = (counts > 0).astype(np.int8)
                if n >= window_size:
                    win_any[:window_size - 1] = 0
                target_data[col] = win_any

        for start_idx in range(0, n - window_size + 1):
            window = signals[start_idx:start_idx + window_size, :]  # (T, 8)
            pred_class = predict_ozz_physics(window, fs=fs, **physics_kwargs)

            # Формируем вектор предсказания (3 класса: OZZ, decay, dpozz)
            # Классы НЕ взаимоисключающие: ДПОЗЗ и затухающее → тоже ОЗЗ
            pred_vec = np.zeros(len(target_cols), dtype=np.int8)
            prob_vec = np.zeros(len(target_cols), dtype=np.float32)
            if pred_class is not None:
                # Любой ненулевой класс → ОЗЗ (Target_OZZ = 1)
                pred_vec[0] = 1     # Target_OZZ
                prob_vec[0] = 1.0
                if pred_class == 1:     # Затухающее
                    pred_vec[1] = 1
                    prob_vec[1] = 1.0
                elif pred_class == 2:   # ДПОЗЗ
                    pred_vec[2] = 1
                    prob_vec[2] = 1.0

            # Целевая метка: point (конец окна) или any_in_window (уже вычислено)
            target_idx = start_idx + window_size - 1
            target_vec = np.array(
                [target_data[col][target_idx] for col in target_cols],
                dtype=np.int8
            )

            all_preds.append(pred_vec)
            all_targets.append(target_vec)
            all_probs.append(prob_vec)

    elapsed = time.time() - start_time

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    probs_arr = np.array(all_probs)

    # Расчёт метрик
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

    f1_per_class = f1_score(targets_arr, preds_arr, average=None, zero_division=0)
    f1_macro = f1_score(targets_arr, preds_arr, average='macro', zero_division=0)
    precision_macro = precision_score(targets_arr, preds_arr, average='macro', zero_division=0)
    recall_macro = recall_score(targets_arr, preds_arr, average='macro', zero_division=0)

    report = classification_report(
        targets_arr, preds_arr,
        target_names=target_cols,
        zero_division=0
    )

    return {
        'predictions': preds_arr,
        'targets': targets_arr,
        'probabilities': probs_arr,
        'target_cols': target_cols,
        'f1_per_class': dict(zip(target_cols, f1_per_class.tolist())),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'classification_report': report,
        'elapsed_seconds': elapsed,
        'n_samples': len(preds_arr),
        'n_files': len(file_names),
    }


def save_as_experiment(
    results: Dict[str, Any],
    experiment_name: str = "PhysicsBaseline_OZZ",
    save_dir: Path = None,
) -> Path:
    """
    Сохраняет результаты физической модели в формате эксперимента,
    совместимом с aggregate_reports.

    Returns:
        Путь к директории эксперимента.
    """
    if save_dir is None:
        save_dir = ROOT_DIR / "experiments" / "phase2_6"

    exp_dir = save_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем config.json (для совместимости с aggregate_reports)
    config = {
        "model": {
            "name": "PhysicsBaseline",
            "params": {
                "algorithm": "deterministic_ozz",
                "u0_threshold": 0.03333333333333333,
                "deriv_peak_sigma": 5.0,
                "min_dpozz_peaks": 3,
                "decay_ratio": 0.30,
                "num_classes": 3,
                "num_params": 0,
                "note": "Порог соответствует 10В вторичных при U_norm = U/(3*Ub_base), Ub_base=100В",
            }
        },
        "data": {
            "window_size": 320,
            "feature_mode": "raw",
            "sampling_strategy": "none",
            "target_level": "ozz",
            "target_window_mode": "any_in_window",
        },
        "training": {
            "epochs": 0,
            "learning_rate": 0,
            "experiment_name": experiment_name,
        }
    }

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Сохраняем метрики как history.json (1 «эпоха»)
    history = {
        "train_loss": [0.0],
        "val_loss": [0.0],
        "val_f1_macro": [results['f1_macro']],
        "val_precision_macro": [results['precision_macro']],
        "val_recall_macro": [results['recall_macro']],
        "val_f1_per_class": {k: [v] for k, v in results['f1_per_class'].items()},
        "elapsed_seconds": results['elapsed_seconds'],
        "algorithm": "PhysicsBaseline (deterministic)",
    }

    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    # Сохраняем текстовый отчёт
    with open(exp_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Physics Baseline — ОЗЗ/ДПОЗЗ Classification Report\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(results['classification_report'])
        f.write(f"\nF1 Macro: {results['f1_macro']:.4f}\n")
        f.write(f"Precision Macro: {results['precision_macro']:.4f}\n")
        f.write(f"Recall Macro: {results['recall_macro']:.4f}\n")
        f.write(f"\nВремя оценки: {results['elapsed_seconds']:.1f} s\n")
        f.write(f"Всего окон: {results['n_samples']}\n")
        f.write(f"Файлов: {results['n_files']}\n")

    # Сохраняем предсказания в multilabel формате (6 столбцов: 3 GT + 3 pred).
    # Это позволяет оценивать каждый таргет независимо как бинарный классификатор.
    target_cols = results.get('target_cols', OZZ_TARGET_COLS)
    y_true_arr = np.asarray(results.get('targets', []))
    y_pred_arr = np.asarray(results.get('predictions', []))

    if y_true_arr.ndim == 2 and y_true_arr.shape[1] >= 3:
        pred_df = pd.DataFrame({
            'y_true_ozz': y_true_arr[:, 0].astype(int),
            'y_true_decay': y_true_arr[:, 1].astype(int),
            'y_true_dpozz': y_true_arr[:, 2].astype(int),
            'y_pred_ozz': y_pred_arr[:, 0].astype(int),
            'y_pred_decay': y_pred_arr[:, 1].astype(int),
            'y_pred_dpozz': y_pred_arr[:, 2].astype(int),
        })
        pred_df.to_csv(exp_dir / 'test_predictions_best.csv', index=False)
        pred_df.to_csv(exp_dir / 'test_predictions_final.csv', index=False)

    print(f"[PhysicsBaseline] Результаты сохранены в {exp_dir}")
    return exp_dir


def main():
    """Запуск оценки физической baseline модели на тестовых данных."""
    DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    print("=" * 60)
    print("Оценка физической baseline-модели (ОЗЗ/ДПОЗЗ)")
    print("=" * 60)

    dm = DatasetManager(str(DATA_DIR), norm_coef_path=str(NORM_COEF_PATH))
    dm.ensure_train_test_split()

    print("Загрузка тестовых данных...")
    test_df = dm.load_test_df(precomputed=False)
    test_df = clean_labels(test_df)
    test_df = add_ozz_target_columns(test_df)

    print(f"Тестовая выборка: {len(test_df):,} точек")

    # Запуск оценки
    results = evaluate_physics_on_files(
        test_df,
        window_size=320,
        fs=1600,
        target_window_mode='any_in_window',
    )

    print(f"\n{results['classification_report']}")
    print(f"F1 Macro: {results['f1_macro']:.4f}")
    print(f"Время: {results['elapsed_seconds']:.1f} с")

    # Сохраняем как эксперимент
    save_as_experiment(results)

    # Также на train для полноты
    print("\nОценка на обучающей выборке...")
    train_df = dm.load_train_df()
    train_df = clean_labels(train_df)
    train_df = add_ozz_target_columns(train_df)

    results_train = evaluate_physics_on_files(
        train_df,
        window_size=320,
        fs=1600,
        target_window_mode='any_in_window'
    )

    print(f"\nTrain F1 Macro: {results_train['f1_macro']:.4f}")
    save_as_experiment(results_train, experiment_name="PhysicsBaseline_OZZ_train")


if __name__ == "__main__":
    main()
