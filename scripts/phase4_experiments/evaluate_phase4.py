"""
Скрипт оценки моделей Фазы 4: Physical KAN-Transformer.

Загружает fine-tuned чекпоинт, вычисляет метрики на валидационной выборке.
Поддерживает сравнение нескольких экспериментов (SSL vs random init, разные сложности).

Метрики:
  - Macro-F1 (порог 0.7)
  - Per-class F1, Precision, Recall
  - ROC-AUC (macro и per-class)
  - Confusion matrix (per class)
  - Inference time

Примеры:
  python scripts/phase4_experiments/evaluate_phase4.py --checkpoint experiments/phase4/finetune_.../best_model.pt
  python scripts/phase4_experiments/evaluate_phase4.py --compare-dir experiments/phase4/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.augmented_dataset import AugmentedSpectralDataset, compute_num_channels, compute_spectral_from_raw
from osc_tools.ml.labels import get_target_columns


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

PREDICTION_THRESHOLD = 0.7  # Порог для бинарных предсказаний (по запросу)


def compute_full_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """Вычисляет полный набор метрик для всех классов.

    Args:
        preds: (N, C) вероятности после sigmoid
        targets: (N, C) бинарные метки
        target_columns: названия классов
        threshold: порог бинаризации

    Returns:
        dict с метриками
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )

    preds_bin = (preds >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)
    num_classes = targets.shape[1]

    metrics: dict = {'threshold': threshold}

    # --- Macro метрики ---
    metrics['macro_f1'] = float(f1_score(targets_int, preds_bin, average='macro', zero_division=0))
    metrics['macro_precision'] = float(precision_score(targets_int, preds_bin, average='macro', zero_division=0))
    metrics['macro_recall'] = float(recall_score(targets_int, preds_bin, average='macro', zero_division=0))

    # ROC-AUC
    try:
        valid_cols = [c for c in range(num_classes) if len(np.unique(targets_int[:, c])) > 1]
        if valid_cols:
            metrics['macro_roc_auc'] = float(roc_auc_score(
                targets_int[:, valid_cols], preds[:, valid_cols], average='macro',
            ))
        else:
            metrics['macro_roc_auc'] = 0.0
    except ValueError:
        metrics['macro_roc_auc'] = 0.0

    # --- Per-class метрики ---
    per_class_f1 = f1_score(targets_int, preds_bin, average=None, zero_division=0)
    per_class_prec = precision_score(targets_int, preds_bin, average=None, zero_division=0)
    per_class_rec = recall_score(targets_int, preds_bin, average=None, zero_division=0)

    metrics['per_class'] = {}
    for i, col in enumerate(target_columns):
        cls_metrics = {
            'f1': float(per_class_f1[i]),
            'precision': float(per_class_prec[i]),
            'recall': float(per_class_rec[i]),
            'support_pos': int(targets_int[:, i].sum()),
            'support_neg': int((1 - targets_int[:, i]).sum()),
        }

        # Per-class ROC-AUC
        try:
            if len(np.unique(targets_int[:, i])) > 1:
                cls_metrics['roc_auc'] = float(roc_auc_score(targets_int[:, i], preds[:, i]))
            else:
                cls_metrics['roc_auc'] = 0.0
        except ValueError:
            cls_metrics['roc_auc'] = 0.0

        # Confusion matrix (per class: binary)
        cm = confusion_matrix(targets_int[:, i], preds_bin[:, i], labels=[0, 1])
        cls_metrics['tn'] = int(cm[0, 0])
        cls_metrics['fp'] = int(cm[0, 1])
        cls_metrics['fn'] = int(cm[1, 0])
        cls_metrics['tp'] = int(cm[1, 1])

        metrics['per_class'][col] = cls_metrics

    # Exact match accuracy
    metrics['exact_match'] = float((preds_bin == targets_int).all(axis=1).mean())

    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Выполняет inference на всём DataLoader.

    Returns:
        (all_preds, all_targets, time_sec) — вероятности и метки
    """
    model.eval()
    all_preds, all_targets = [], []
    t0 = time.perf_counter()

    for x, y in loader:
        x = x.to(device)
        out = model(x, mode='classify')
        logits = out['classify'][:, -1, :]  # Последняя зона
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(y.numpy())

    elapsed = time.perf_counter() - t0
    return np.concatenate(all_preds), np.concatenate(all_targets), elapsed


def measure_inference_latency(
    model: nn.Module,
    num_channels: int,
    num_steps: int,
    device: torch.device,
    num_repeats: int = 100,
) -> float:
    """Измеряет среднюю latency на одном sample (мс)."""
    model.eval()
    x = torch.randn(1, num_channels, num_steps, device=device)

    # Прогрев
    for _ in range(10):
        model(x, mode='classify')

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_repeats):
        model(x, mode='classify')

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / num_repeats
    return elapsed * 1000  # мс


# ---------------------------------------------------------------------------
# Разметка осциллограммы скользящим окном с усреднением
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_oscillogram(
    model: nn.Module,
    raw_data: np.ndarray,
    config: dict,
    device: torch.device,
    step: int = 32,
) -> np.ndarray:
    """Разметка осциллограммы скользящим окном с усреднением перекрывающихся предсказаний.

    Алгоритм:
    1. Скользящее окно шагом step=32 (1 период) по raw_data
    2. Для каждого окна: спектральная обработка → inference → вероятности
    3. Пообразцовое усреднение: для каждой точки осциллограммы берём
       среднее от всех окон, покрывающих эту точку.

    Args:
        model: fine-tuned модель в eval mode
        raw_data: (N, 8) сырая осциллограмма (IA, IB, IC, IN, UA, UB, UC, UN)
        config: конфиг модели (window_size, num_harmonics, sub_periods, ...)
        device: torch device
        step: шаг скользящего окна в отсчётах (32 = 1 период)

    Returns:
        (N, num_classes) усреднённые вероятности для каждого отсчёта
    """
    model.eval()
    N = raw_data.shape[0]
    window_size = config.get('window_size', 320)
    num_harmonics = config.get('num_harmonics', 9)
    sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if config.get('use_low_harmonics', True) else []
    include_symmetric = config.get('include_symmetric', True)
    stride = config.get('downsampling_stride', 16)
    num_classes = config.get('num_classes', 4)

    # Контекст для низших гармоник (backward-looking)
    max_lh_window = max(sub_periods) * 32 if sub_periods else 0
    context_before = max(0, max_lh_window - 1)

    # Аккумуляторы: сумма вероятностей и число покрытий
    prob_sum = np.zeros((N, num_classes), dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int32)

    fft_window = 32
    warmup = fft_window

    # Скользящее окно
    starts = list(range(0, max(1, N - window_size + 1), step))
    print(f"  Разметка: {len(starts)} окон, шаг={step}, window={window_size}")

    for win_start in starts:
        win_end = min(win_start + window_size, N)

        # Расширяем контекстом для низших гармоник
        ctx_start = max(0, win_start - context_before)
        raw_ext = raw_data[ctx_start:win_end].copy()
        ctx_offset = win_start - ctx_start

        # Padding при нехватке контекста
        if ctx_start == 0 and win_start < context_before:
            pad_len = context_before - win_start
            pad = np.repeat(raw_data[0:1], pad_len, axis=0)
            raw_ext = np.concatenate([pad, raw_ext], axis=0)
            ctx_offset = context_before

        # Спектральная обработка
        spectral = compute_spectral_from_raw(
            raw_ext, num_harmonics, sub_periods if sub_periods else None, include_symmetric,
        )

        # Обрезаем до окна
        spectral = spectral[ctx_offset: ctx_offset + window_size]

        # Warmup + stride
        if spectral.shape[0] > warmup:
            spectral = spectral[warmup:]
        spectral = spectral[::stride]

        # Inference
        X = torch.from_numpy(spectral.T.copy()).unsqueeze(0).to(device)  # (1, C, T)
        out = model(X, mode='classify')
        logits = out['classify']  # (1, num_zones, num_classes)
        probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()  # (num_zones, num_classes)

        # Маппинг зон обратно на raw отсчёты
        # Зоны начинаются с warmup, шаг = stride
        n_zones = probs.shape[0]
        for z in range(n_zones):
            # Абсолютная позиция центра зоны
            raw_pos = win_start + warmup + z * stride
            # Зона покрывает [raw_pos, raw_pos + stride)
            z_start = raw_pos
            z_end = min(raw_pos + stride, N)
            if z_start >= N:
                break
            prob_sum[z_start:z_end] += probs[z]
            coverage[z_start:z_end] += 1

    # Усреднение
    mask_covered = coverage > 0
    result = np.zeros((N, num_classes), dtype=np.float32)
    result[mask_covered] = (prob_sum[mask_covered] / coverage[mask_covered, np.newaxis]).astype(np.float32)

    # Непокрытые точки (начало осциллограммы) — берём ближайшее предсказание
    if not mask_covered.all():
        first_covered = np.argmax(mask_covered)
        if first_covered > 0:
            result[:first_covered] = result[first_covered]

    return result


# ---------------------------------------------------------------------------
# Загрузка и создание модели
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Загружает модель из fine-tuning чекпоинта.

    Returns:
        (model, config)
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']

    model_type = config.get('model_type', 'PhysicalKANTransformer')
    num_classes = config.get('num_classes', 4)
    zone_size = config.get('zone_size', 1)

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            kan_grid_size=config.get('kan_grid_size', 5),
            dropout=0.0,  # Dropout off при inference
            max_seq_len=64,
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            dropout=0.0,
            max_seq_len=64,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # Загружаем веса
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def prepare_val_dataloader(
    config: dict,
) -> tuple[DataLoader, list[str]]:
    """Создаёт валидационный DataLoader (без аугментации)."""
    data_path = Path(config['data_dir']) / config['precomputed_file']
    df = pl.read_csv(str(data_path))

    target_columns = get_target_columns(config.get('target_level', 'base'), df)

    # Разделение файлов (тот же seed что при обучении)
    files = sorted(df['file_name'].unique().to_list())
    rng = np.random.RandomState(config.get('seed', 42))
    rng.shuffle(files)
    n_val = max(1, int(len(files) * config.get('val_split', 0.2)))
    val_files = files[:n_val]
    val_df = df.filter(pl.col('file_name').is_in(val_files))

    window_size = config['window_size']
    stride = config.get('downsampling_stride', 16)

    use_low_harmonics = config.get('use_low_harmonics', False)
    include_symmetric = config.get('include_symmetric', True)
    sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if use_low_harmonics else []
    future_periods = config.get('future_periods', 0)

    # Правильное окно для индексации (с учётом будущих периодов)
    full_window = window_size + future_periods * 32

    if use_low_harmonics:
        val_boundaries = AugmentedSpectralDataset.compute_file_boundaries(val_df)
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=full_window, mode='val', stride=stride,
        )
        val_ds = AugmentedSpectralDataset(
            dataframe=val_df,
            file_boundaries=val_boundaries,
            indices=val_indices,
            window_size=window_size,
            num_harmonics=config.get('num_harmonics', 9),
            sub_periods=sub_periods if sub_periods else None,
            include_symmetric=include_symmetric,
            downsampling_stride=stride,
            future_periods=future_periods,
            mask_ratio=0.0,
            augmenter=None,
            target_columns=target_columns,
            mode='classify',
            target_window_mode='any_in_window',
        )
    else:
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=window_size, mode='val', stride=stride,
        )
        val_ds = PrecomputedDataset(
            dataframe=val_df,
            indices=val_indices,
            window_size=window_size,
            feature_mode=config.get('feature_mode', 'phase_polar'),
            target_columns=target_columns,
            sampling_strategy='stride',
            downsampling_stride=stride,
            target_position=window_size - 1,
            target_window_mode='any_in_window',
            num_harmonics=config.get('num_harmonics', 9),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('val_batch_size', 64),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return val_loader, target_columns


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def print_report(metrics: dict, target_columns: list[str], model_info: dict) -> None:
    """Печатает форматированный отчёт."""
    print("\n" + "=" * 70)
    print("ОТЧЁТ ОЦЕНКИ МОДЕЛИ ФАЗЫ 4")
    print("=" * 70)

    print(f"\nМодель:     {model_info.get('model_type', 'N/A')}")
    print(f"Чекпоинт:   {model_info.get('checkpoint', 'N/A')}")
    print(f"Параметры:  {model_info.get('num_params', 'N/A'):,}")
    print(f"d_model:    {model_info.get('d_model', 'N/A')}")
    print(f"Слои:       {model_info.get('num_layers', 'N/A')}")
    print(f"Каналов:    {model_info.get('num_input_channels', 'N/A')}")
    if 'latency_ms' in model_info:
        print(f"Latency:    {model_info['latency_ms']:.2f} мс/sample")
    if 'inference_time' in model_info:
        print(f"Inference:  {model_info['inference_time']:.2f} с (полная val)")

    print(f"\n--- Общие метрики (порог = {metrics['threshold']}) ---")
    print(f"  Macro-F1:        {metrics['macro_f1']:.4f}")
    print(f"  Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro-ROC-AUC:   {metrics['macro_roc_auc']:.4f}")
    print(f"  Exact Match:     {metrics['exact_match']:.4f}")

    print(f"\n--- Per-class метрики ---")
    header = f"  {'Класс':<25s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'AUC':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for col in target_columns:
        c = metrics['per_class'].get(col, {})
        print(
            f"  {col:<25s} "
            f"{c.get('f1', 0):.4f} "
            f"{c.get('precision', 0):.4f} "
            f"{c.get('recall', 0):.4f} "
            f"{c.get('roc_auc', 0):.4f} "
            f"{c.get('tp', 0):5d} "
            f"{c.get('fp', 0):5d} "
            f"{c.get('fn', 0):5d} "
            f"{c.get('tn', 0):5d}"
        )

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    save_report: bool = True,
) -> dict:
    """Оценивает один fine-tuning чекпоинт."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    # Загрузка модели
    model, config = load_model_from_checkpoint(ckpt_path, device)
    num_params = sum(p.numel() for p in model.parameters())

    # Подготовка данных
    val_loader, target_columns = prepare_val_dataloader(config)

    # Inference
    preds, targets, inference_time = run_inference(model, val_loader, device)
    print(f"Inference: {preds.shape[0]} samples за {inference_time:.2f} с")

    # Метрики
    metrics = compute_full_metrics(preds, targets, target_columns)

    # Latency
    num_channels = config['num_input_channels']
    # Примерное число шагов после stride
    num_steps = max(1, (config['window_size'] - 32) // config.get('downsampling_stride', 16))
    latency_ms = measure_inference_latency(model, num_channels, num_steps, device)

    model_info = {
        'checkpoint': str(ckpt_path),
        'model_type': config.get('model_type', 'N/A'),
        'num_params': num_params,
        'd_model': config.get('d_model'),
        'num_layers': config.get('num_layers'),
        'num_input_channels': config.get('num_input_channels'),
        'latency_ms': latency_ms,
        'inference_time': inference_time,
        'num_samples': preds.shape[0],
    }

    # Печать
    print_report(metrics, target_columns, model_info)

    # Сохранение
    if save_report:
        report_dir = ckpt_path.parent
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'metrics': metrics,
            'config': config,
        }
        report_path = report_dir / 'evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nОтчёт сохранён: {report_path}")

    return metrics


def compare_experiments(experiment_dir: str) -> None:
    """Сравнивает все эксперименты fine-tuning в директории."""
    exp_dir = Path(experiment_dir)
    results = []

    # Ищем все finetune директории с best_model.pt
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith('finetune_'):
            continue
        best_model = subdir / 'best_model.pt'
        if not best_model.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Эксперимент: {subdir.name}")
        print(f"{'='*60}")

        try:
            metrics = evaluate_checkpoint(str(best_model), save_report=True)
            results.append({
                'experiment': subdir.name,
                'macro_f1': metrics['macro_f1'],
                'macro_roc_auc': metrics['macro_roc_auc'],
                'exact_match': metrics['exact_match'],
            })
        except Exception as e:
            print(f"  Ошибка: {e}")

    if results:
        print(f"\n\n{'='*60}")
        print("СВОДНАЯ ТАБЛИЦА")
        print(f"{'='*60}")
        print(f"{'Эксперимент':<50s} {'F1':>6s} {'AUC':>6s} {'EM':>6s}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x['macro_f1'], reverse=True):
            print(
                f"{r['experiment']:<50s} "
                f"{r['macro_f1']:.4f} "
                f"{r['macro_roc_auc']:.4f} "
                f"{r['exact_match']:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к fine-tuning чекпоинту')
    parser.add_argument('--compare-dir', type=str, default=None,
                        help='Директория для сравнения всех экспериментов')
    parser.add_argument('--mark', type=str, default=None,
                        help='Путь к CSV-файлу осциллограммы для разметки (sliding window)')
    parser.add_argument('--mark-step', type=int, default=32,
                        help='Шаг скользящего окна при разметке (отсчёты, default=32=1 период)')
    parser.add_argument('--no-save', action='store_true',
                        help='Не сохранять отчёт')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mark and args.checkpoint:
        # Режим разметки осциллограммы
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, config = load_model_from_checkpoint(Path(args.checkpoint), device)

        mark_path = Path(args.mark)
        print(f"Разметка осциллограммы: {mark_path}")
        raw_df = pl.read_csv(str(mark_path))

        raw_channels = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
        raw_data = raw_df.select(raw_channels).to_numpy().astype(np.float32)

        probs = mark_oscillogram(model, raw_data, config, device, step=args.mark_step)

        # Сохраняем результат
        target_cols = get_target_columns(config.get('target_level', 'base'))
        out_df = pl.DataFrame({col: probs[:, i] for i, col in enumerate(target_cols)})
        out_path = mark_path.with_suffix('.marking.csv')
        out_df.write_csv(str(out_path))
        print(f"Разметка сохранена: {out_path} ({probs.shape[0]} отсчётов, {probs.shape[1]} классов)")

    elif args.compare_dir:
        compare_experiments(args.compare_dir)
    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint, save_report=not args.no_save)
    else:
        # По умолчанию — сравнить все эксперименты
        default_dir = PROJECT_ROOT / 'experiments' / 'phase4'
        if default_dir.exists():
            compare_experiments(str(default_dir))
        else:
            print(f"Директория экспериментов не найдена: {default_dir}")
            print("Используйте --checkpoint или --compare-dir")


if __name__ == '__main__':
    main()
