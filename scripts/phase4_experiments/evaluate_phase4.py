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

from osc_tools.ml.models.transformer import (
    PhysicalKANTransformer, PhysicalMLPTransformer, BaselineTransformer,
)
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.augmented_dataset import AugmentedSpectralDataset, compute_num_channels, compute_spectral_from_raw
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

PREDICTION_THRESHOLD = 0.5  # Порог для бинарных предсказаний (стандартный для sigmoid)


def find_optimal_thresholds(
    preds: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    grid_steps: int = 101,
) -> dict[str, dict]:
    """Поиск оптимального порога per-class через grid search по F1.

    Для каждого класса перебираем пороги от 0.01 до 0.99 и выбираем
    тот, при котором F1 максимален. Также считаем macro-F1 для
    оптимальных порогов.

    Args:
        preds: (N, C) вероятности после sigmoid
        targets: (N, C) бинарные метки
        target_columns: названия классов
        grid_steps: число точек сетки порогов

    Returns:
        dict с ключами 'per_class' (пороги и F1 для каждого класса),
        'thresholds' (dict class→float), 'macro_f1_optimal'
    """
    from sklearn.metrics import f1_score

    thresholds_grid = np.linspace(0.01, 0.99, grid_steps)
    targets_int = targets.astype(np.int32)
    num_classes = targets.shape[1]

    result: dict = {'per_class': {}, 'thresholds': {}}
    best_preds_bin = np.zeros_like(targets_int)

    for i, col in enumerate(target_columns):
        best_f1 = -1.0
        best_thr = 0.5

        # Если класс не представлен — оставляем порог 0.5
        if targets_int[:, i].sum() == 0 or (1 - targets_int[:, i]).sum() == 0:
            result['per_class'][col] = {'threshold': 0.5, 'f1': 0.0}
            result['thresholds'][col] = 0.5
            best_preds_bin[:, i] = (preds[:, i] >= 0.5).astype(np.int32)
            continue

        for thr in thresholds_grid:
            pred_bin = (preds[:, i] >= thr).astype(np.int32)
            f1_val = f1_score(targets_int[:, i], pred_bin, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thr = float(thr)

        result['per_class'][col] = {'threshold': best_thr, 'f1': float(best_f1)}
        result['thresholds'][col] = best_thr
        best_preds_bin[:, i] = (preds[:, i] >= best_thr).astype(np.int32)

    # Macro-F1 с оптимальными порогами
    result['macro_f1_optimal'] = float(
        f1_score(targets_int, best_preds_bin, average='macro', zero_division=0)
    )
    return result


# ---------------------------------------------------------------------------
# Метрики границ событий (Smearing / Delay)
# ---------------------------------------------------------------------------

def compute_boundary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    target_columns: list[str],
    threshold: float = PREDICTION_THRESHOLD,
    stride_samples: int = 16,
) -> dict:
    """Вычисляет метрики точности границ для каждого класса.

    Для каждого файла (осциллограммы) и каждого класса:
    - **delay** — задержка обнаружения: расстояние от начала реального события
      до первого предсказанного положительного (в зонах). Положительное = опоздание.
    - **smearing_onset** — ширина «нарастания» на начале события: число зон
      между первым и последним переключением 0→1 в окрестности true onset.
    - **smearing_offset** — ширина «спада» в конце события: число зон между
      первым и последним переключением 1→0 в окрестности true offset.
    - **false_alarm_zones** — число зон с pred=1 за пределами любого реального события.

    Args:
        probs: (N_zones, C) вероятности (позонные, НЕ window-level)
        targets: (N_zones, C) бинарные метки (позонные)
        target_columns: названия классов
        threshold: порог бинаризации
        stride_samples: число отсчётов в одной зоне (для перевода в мс)

    Returns:
        dict с метриками per-class и средними
    """
    preds_bin = (probs >= threshold).astype(np.int32)
    targets_bin = targets.astype(np.int32)
    num_classes = targets.shape[1]
    result: dict = {'per_class': {}, 'stride_samples': stride_samples}

    for ci, col in enumerate(target_columns):
        t = targets_bin[:, ci]
        p = preds_bin[:, ci]

        # Найти все непрерывные сегменты событий в targets
        events = _find_segments(t)
        # Найти все предсказанные сегменты
        pred_events = _find_segments(p)

        delays = []
        smearing_onsets = []
        smearing_offsets = []
        false_zones = 0

        for ev_start, ev_end in events:
            # delay: расстояние от ev_start до первого p[i]=1 внутри [ev_start, ev_end)
            first_pred = None
            for i in range(ev_start, ev_end):
                if p[i] == 1:
                    first_pred = i
                    break
            if first_pred is not None:
                delays.append(first_pred - ev_start)
            else:
                # Полный промах — событие не обнаружено
                delays.append(ev_end - ev_start)  # максимальная задержка = длина события

            # smearing_onset: в окрестности ev_start (±5 зон)
            # подсчёт зон, где предсказание «колеблется»
            margin = 5
            onset_region = p[max(0, ev_start - margin):min(len(p), ev_start + margin)]
            if len(onset_region) > 0:
                transitions = int(np.sum(np.abs(np.diff(onset_region))))
                smearing_onsets.append(transitions)
            else:
                smearing_onsets.append(0)

            # smearing_offset: в окрестности ev_end
            offset_region = p[max(0, ev_end - margin):min(len(p), ev_end + margin)]
            if len(offset_region) > 0:
                transitions = int(np.sum(np.abs(np.diff(offset_region))))
                smearing_offsets.append(transitions)
            else:
                smearing_offsets.append(0)

        # false_alarm_zones: pred=1 вне любого реального события
        event_mask = np.zeros(len(t), dtype=bool)
        for ev_start, ev_end in events:
            event_mask[ev_start:ev_end] = True
        false_zones = int(np.sum(p[~event_mask]))

        cls_result = {
            'num_events': len(events),
            'num_pred_events': len(pred_events),
            'delays_zones': delays,
            'mean_delay_zones': float(np.mean(delays)) if delays else 0.0,
            'mean_delay_samples': float(np.mean(delays) * stride_samples) if delays else 0.0,
            'smearing_onset': smearing_onsets,
            'mean_smearing_onset': float(np.mean(smearing_onsets)) if smearing_onsets else 0.0,
            'smearing_offset': smearing_offsets,
            'mean_smearing_offset': float(np.mean(smearing_offsets)) if smearing_offsets else 0.0,
            'false_alarm_zones': false_zones,
        }
        result['per_class'][col] = cls_result

    # Средние по всем классам
    all_delays = []
    all_smear_on = []
    all_smear_off = []
    for col in target_columns:
        cr = result['per_class'][col]
        all_delays.extend(cr['delays_zones'])
        all_smear_on.extend(cr['smearing_onset'])
        all_smear_off.extend(cr['smearing_offset'])

    result['mean_delay_zones'] = float(np.mean(all_delays)) if all_delays else 0.0
    result['mean_delay_samples'] = result['mean_delay_zones'] * stride_samples
    result['mean_smearing_onset'] = float(np.mean(all_smear_on)) if all_smear_on else 0.0
    result['mean_smearing_offset'] = float(np.mean(all_smear_off)) if all_smear_off else 0.0

    return result


def _find_segments(binary: np.ndarray) -> list[tuple[int, int]]:
    """Находит непрерывные сегменты единиц в бинарном массиве.
    
    Returns:
        Список (start, end) — полуоткрытые интервалы [start, end)
    """
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(binary):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary)))
    return segments


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

    Метки теперь позонные (B, T_zones, C). Для window-level метрик
    усредняем вероятности по зонам, метки — max по зонам.

    Returns:
        (all_preds, all_targets, time_sec) — window-level вероятности и метки
    """
    model.eval()
    all_preds, all_targets = [], []
    t0 = time.perf_counter()

    for x, y in loader:
        x = x.to(device)
        out = model(x, mode='classify')
        logits = out['classify']  # (B, num_zones, num_classes)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        # Window-level: среднее по зонам / max по зонам
        probs_mean = probs.mean(axis=1)  # (B, C)
        all_preds.append(probs_mean)

        y_np = y.numpy()
        if y_np.ndim == 3:
            y_np = y_np.max(axis=1)  # (B, C)
        all_targets.append(y_np)

    elapsed = time.perf_counter() - t0
    return np.concatenate(all_preds), np.concatenate(all_targets), elapsed


@torch.no_grad()
def run_inference_zone_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Inference с zone-level выходом (без усреднения по зонам).

    Конкатенирует все зоны всех батчей в один длинный массив.
    Используется для boundary-метрик (smearing, delay).

    Returns:
        (zone_preds, zone_targets) — позонные данные, shape (N_total_zones, C)
    """
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        out = model(x, mode='classify')
        logits = out['classify']  # (B, T_zones, C)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        # Flatten batch: (B, T, C) → (B*T, C)
        B, T, C = probs.shape
        all_preds.append(probs.reshape(-1, C))

        y_np = y.numpy()
        if y_np.ndim == 3:
            all_targets.append(y_np.reshape(-1, y_np.shape[-1]))
        else:
            # Если метки window-level — растянуть на T зон
            all_targets.append(np.repeat(y_np, T, axis=0))

    return np.concatenate(all_preds), np.concatenate(all_targets)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Разметка осциллограммы скользящим окном с усреднением перекрывающихся предсказаний.

    Алгоритм:
    1. Скользящее окно шагом step=32 (1 период) по raw_data
    2. Для каждого окна: FFT → stride фазоров → polar/symmetric → inference
    3. Пообразцовое усреднение: для каждой точки — среднее от всех покрывающих окон.

    Edge handling (Q10):
    - Первые ~(fft_window) отсчётов НЕ покрыты — возвращаются как NaN
    - coverage[i] показывает число окон, покрывающих i-й отсчёт
    - Надёжность предсказания растёт с coverage (1 окно < 5 окон)

    Args:
        model: fine-tuned модель в eval mode
        raw_data: (N, 8) сырая осциллограмма (IA, IB, IC, IN, UA, UB, UC, UN)
        config: конфиг модели (window_size, num_harmonics, sub_periods, ...)
        device: torch device
        step: шаг скользящего окна в отсчётах (32 = 1 период)

    Returns:
        (probs, coverage):
        - probs: (N, num_classes) усреднённые вероятности (NaN для непокрытых точек)
        - coverage: (N,) число окон, покрывающих каждый отсчёт (0 = нет данных)
    """
    model.eval()
    N = raw_data.shape[0]
    window_size = config.get('window_size', 320)
    num_harmonics = config.get('num_harmonics', 9)
    sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if config.get('use_low_harmonics', True) else []
    include_symmetric = config.get('include_symmetric', True)
    stride = config.get('downsampling_stride', 16)
    num_classes = config.get('num_classes', 4)

    fft_window = 32

    # Аккумуляторы: сумма вероятностей и число покрытий
    prob_sum = np.zeros((N, num_classes), dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int32)

    # Скользящее окно
    starts = list(range(0, max(1, N - window_size + 1), step))
    print(f"  Разметка: {len(starts)} окон, шаг={step}, window={window_size}")

    for win_start in starts:
        win_end = min(win_start + window_size, N)
        raw_win = raw_data[win_start:win_end].copy()

        # Дополняем нулями если короче
        if len(raw_win) < window_size:
            raw_win = np.concatenate([
                raw_win,
                np.zeros((window_size - len(raw_win), 8), dtype=np.float32),
            ], axis=0)

        # FFT → stride комплексных фазоров → polar/symmetric (уже прорежено)
        spectral = compute_spectral_from_raw(
            raw_win, num_harmonics, sub_periods if sub_periods else None,
            include_symmetric, stride=stride, warmup=fft_window,
        )  # (T_strided, C)

        # Inference
        X = torch.from_numpy(spectral.T.copy()).unsqueeze(0).to(device)
        out = model(X, mode='classify')
        logits = out['classify']  # (1, num_zones, num_classes)
        probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

        # Маппинг зон обратно на raw отсчёты
        n_zones = probs.shape[0]
        for z in range(n_zones):
            raw_pos = win_start + fft_window + z * stride
            z_end = min(raw_pos + stride, N)
            if raw_pos >= N:
                break
            prob_sum[raw_pos:z_end] += probs[z]
            coverage[raw_pos:z_end] += 1

    # Усреднение
    mask_covered = coverage > 0
    result = np.full((N, num_classes), np.nan, dtype=np.float32)
    result[mask_covered] = (prob_sum[mask_covered] / coverage[mask_covered, np.newaxis]).astype(np.float32)

    # Непокрытые точки остаются NaN — это честный edge handling (Q10).
    # Вызывающий код должен учитывать coverage при визуализации.
    n_uncovered = int((~mask_covered).sum())
    if n_uncovered > 0:
        print(f"  Edge handling: {n_uncovered} отсчётов без покрытия (первые ~{np.argmax(mask_covered)} точек)")

    return result, coverage


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
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            cls_head_type=config.get('cls_head_type', 'kan'),
            dropout=0.0,  # Dropout off при inference
            max_seq_len=config.get('max_seq_len', 128),
        )
    elif model_type == 'PhysicalMLPTransformer':
        model = PhysicalMLPTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            kan_grid_size=config.get('kan_grid_size', 5),
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            cls_head_type=config.get('cls_head_type', 'mlp'),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', 128),
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            zone_size=zone_size,
            cls_head_type=config.get('cls_head_type', 'linear'),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', 128),
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

    # Подготовка меток (нужна для ozz, base_sequential и др.)
    target_level = config.get('target_level', 'base')
    if target_level in ('base_sequential', 'ozz', 'full', 'full_by_levels'):
        df = prepare_labels_for_experiment(df, target_level)

    target_columns = get_target_columns(target_level, df)

    # Разделение файлов (тот же алгоритм что при обучении)
    if target_level == 'ozz':
        from osc_tools.data_management.ozz_split import (
            stratified_ozz_split, classify_file_ozz,
        )
        train_files, val_files, _ = stratified_ozz_split(
            df,
            test_size=config.get('val_split', 0.2),
            random_state=config.get('seed', 42),
            min_test_per_class=1,
        )
        # Перебалансировка: гарантируем представительство в обоих сплитах
        file_classes = {}
        for fname in train_files + val_files:
            fdf = df.filter(pl.col('file_name') == fname)
            file_classes[fname] = classify_file_ozz(fdf)
        for cls in ('dpozz', 'decay', 'stable'):
            train_cls = [f for f in train_files if file_classes[f] == cls]
            val_cls = [f for f in val_files if file_classes[f] == cls]
            if len(train_cls) == 0 and len(val_cls) >= 2:
                move_file = val_cls[0]
                val_files.remove(move_file)
                train_files.append(move_file)
    else:
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
    future_zones = config.get('future_zones', 0)
    # Обратная совместимость со старыми конфигами
    if future_zones == 0 and config.get('future_periods', 0) > 0:
        future_zones = config['future_periods'] * 32 // stride
    zone_target_aggregation = config.get('zone_target_aggregation', 'max')

    # Правильное окно для индексации (с учётом будущих зон)
    full_window = window_size + future_zones * stride

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
            future_zones=future_zones,
            mask_ratio=0.0,
            augmenter=None,
            target_columns=target_columns,
            mode='classify',
            target_window_mode='any_in_window',
            zone_target_aggregation=zone_target_aggregation,
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

    # Метрики с порогом по умолчанию
    metrics = compute_full_metrics(preds, targets, target_columns)

    # Оптимальные пороги per-class
    opt_thresholds = find_optimal_thresholds(preds, targets, target_columns)
    metrics['optimal_thresholds'] = opt_thresholds
    # Метрики с оптимальными порогами
    metrics_opt = compute_full_metrics(
        preds, targets, target_columns,
        threshold=None,  # Не используется — мы подменяем бинаризацию ниже
    )
    # Пересчёт с per-class порогами
    preds_bin_opt = np.zeros_like(preds, dtype=np.int32)
    for i, col in enumerate(target_columns):
        thr = opt_thresholds['thresholds'].get(col, 0.5)
        preds_bin_opt[:, i] = (preds[:, i] >= thr).astype(np.int32)
    from sklearn.metrics import f1_score as _f1
    metrics['macro_f1_optimal'] = float(
        _f1(targets.astype(np.int32), preds_bin_opt, average='macro', zero_division=0)
    )

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

    # Таблица оптимальных порогов
    print("\n--- Оптимальные пороги per-class ---")
    print(f"  {'Класс':<25s} {'Порог':>8s} {'F1_opt':>8s} {'F1_0.5':>8s}")
    print("  " + "-" * 55)
    for col in target_columns:
        opt = opt_thresholds['per_class'].get(col, {})
        std = metrics['per_class'].get(col, {})
        print(
            f"  {col:<25s} "
            f"{opt.get('threshold', 0.5):8.3f} "
            f"{opt.get('f1', 0):8.4f} "
            f"{std.get('f1', 0):8.4f}"
        )
    print(f"\n  Macro-F1 (порог=0.5):      {metrics['macro_f1']:.4f}")
    print(f"  Macro-F1 (оптимал.):       {metrics['macro_f1_optimal']:.4f}")
    delta = metrics['macro_f1_optimal'] - metrics['macro_f1']
    print(f"  Прирост:                   {delta:+.4f}")

    # --- Boundary-метрики (smearing / delay) ---
    print("\n--- Метрики границ (zone-level) ---")
    try:
        zone_preds, zone_targets = run_inference_zone_level(model, val_loader, device)
        stride_samples = config.get('downsampling_stride', 16)
        boundary = compute_boundary_metrics(
            zone_preds, zone_targets, target_columns,
            threshold=PREDICTION_THRESHOLD, stride_samples=stride_samples,
        )
        metrics['boundary'] = boundary

        print(f"  {'Класс':<25s} {'Событий':>8s} {'Delay(зон)':>11s} {'Delay(отсч)':>12s} {'Smear_on':>9s} {'Smear_off':>10s} {'FP_зон':>7s}")
        print("  " + "-" * 85)
        for col in target_columns:
            bc = boundary['per_class'].get(col, {})
            print(
                f"  {col:<25s} "
                f"{bc.get('num_events', 0):8d} "
                f"{bc.get('mean_delay_zones', 0):11.2f} "
                f"{bc.get('mean_delay_samples', 0):12.1f} "
                f"{bc.get('mean_smearing_onset', 0):9.2f} "
                f"{bc.get('mean_smearing_offset', 0):10.2f} "
                f"{bc.get('false_alarm_zones', 0):7d}"
            )
        print(f"\n  Среднее: delay={boundary['mean_delay_zones']:.2f} зон "
              f"({boundary['mean_delay_samples']:.0f} отсч.), "
              f"smear_on={boundary['mean_smearing_onset']:.2f}, "
              f"smear_off={boundary['mean_smearing_offset']:.2f}")
    except Exception as e:
        print(f"  Не удалось вычислить boundary-метрики: {e}")

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

        # Отдельно сохраняем оптимальные пороги
        thr_path = report_dir / 'optimal_thresholds.json'
        with open(thr_path, 'w', encoding='utf-8') as f:
            json.dump(opt_thresholds, f, indent=2, ensure_ascii=False)
        print(f"Пороги сохранены: {thr_path}")

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

        probs, coverage = mark_oscillogram(model, raw_data, config, device, step=args.mark_step)

        # Сохраняем результат (NaN для непокрытых точек остаются)
        target_cols = get_target_columns(config.get('target_level', 'base'))
        data_dict = {col: probs[:, i] for i, col in enumerate(target_cols)}
        data_dict['coverage'] = coverage.astype(np.float32)
        out_df = pl.DataFrame(data_dict)
        out_path = mark_path.with_suffix('.marking.csv')
        out_df.write_csv(str(out_path))
        print(f"Разметка сохранена: {out_path} ({probs.shape[0]} отсчётов, {probs.shape[1]} классов)")

    elif args.compare_dir:
        compare_experiments(args.compare_dir)
    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint, save_report=not args.no_save)
    else:
        print("Используйте --checkpoint, --compare-dir или --mark")
        print("Или запустите файл напрямую (ручной режим в __main__).")


if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Для запуска: просто выполните файл (F5 в VS Code или python evaluate_phase4.py)
    # Для CLI: python evaluate_phase4.py --checkpoint path/to/best_model.pt
    # =================================================================

    MANUAL_RUN = True

    if MANUAL_RUN:
        # --- Путь к чекпоинту для оценки ---
        CHECKPOINT = 'experiments/phase4/finetune_PhysicalKANTransformer_20260319_201650/best_model.pt'

        # --- Или сравнить все эксперименты в директории ---
        COMPARE_DIR = None  # 'experiments/phase4'

        # --- Режим разметки одного файла ---
        MARK_FILE = None   # 'data/ml_datasets/some_oscillogram.csv'
        MARK_STEP = 32     # Шаг скользящего окна (32 = 1 период)

        # --- Сохранять отчёт ---
        SAVE_REPORT = True

        # =================================================================

        if MARK_FILE and CHECKPOINT:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, config = load_model_from_checkpoint(
                Path(PROJECT_ROOT / CHECKPOINT), device,
            )
            mark_path = Path(PROJECT_ROOT / MARK_FILE)
            raw_df = pl.read_csv(str(mark_path))
            raw_channels = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
            raw_data = raw_df.select(raw_channels).to_numpy().astype(np.float32)
            probs, coverage = mark_oscillogram(model, raw_data, config, device, step=MARK_STEP)
            target_cols = get_target_columns(config.get('target_level', 'base'))
            data_dict = {col: probs[:, i] for i, col in enumerate(target_cols)}
            data_dict['coverage'] = coverage.astype(np.float32)
            out_df = pl.DataFrame(data_dict)
            out_path = mark_path.with_suffix('.marking.csv')
            out_df.write_csv(str(out_path))
            print(f"Разметка сохранена: {out_path}")
        elif COMPARE_DIR:
            compare_experiments(str(PROJECT_ROOT / COMPARE_DIR))
        elif CHECKPOINT:
            evaluate_checkpoint(
                str(PROJECT_ROOT / CHECKPOINT),
                save_report=SAVE_REPORT,
            )
        else:
            print("Укажите CHECKPOINT или COMPARE_DIR в ручном режиме.")
    else:
        main()
