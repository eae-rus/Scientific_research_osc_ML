"""
Fine-tuning скрипт для Фазы 4: Physical KAN-Transformer.

Инициализирует модель из SSL-чекпоинта и дообучает на задачу классификации
аварийных режимов (4 базовых класса: Normal, ML_1, ML_2, ML_3).

Зональная классификация: zone_size=1, каждый временной шаг (при stride=16
= полпериода) получает свою предсказанную метку.

Примеры:
  python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_.../best_model.pt
  python scripts/phase4_experiments/run_phase4_finetune.py --checkpoint experiments/phase4/pretrain_.../best_model.pt --model BaselineTransformer
  python scripts/phase4_experiments/run_phase4_finetune.py --smoke  # быстрый smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

# Добавляем корень проекта в PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.augmented_dataset import (
    AugmentedSpectralDataset, compute_num_channels, compute_stride, SAMPLES_PER_PERIOD,
    standardize_voltage_columns,
)
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.labels import (
    get_target_columns, prepare_labels_for_experiment,
    clean_labels, add_base_labels,
)


# ---------------------------------------------------------------------------
# Уровни сложности модели
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS = {
    'light':  {'d_model': 48,  'num_layers': 6,  'num_heads': 4},
    'medium': {'d_model': 64,  'num_layers': 8,  'num_heads': 8},
    'heavy':  {'d_model': 64,  'num_layers': 16, 'num_heads': 8},
}


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

NUM_BASE_CLASSES = 3  # Target_ML_1, Target_ML_2, Target_ML_3 (без Normal)
NUM_OZZ_CLASSES = 3   # Target_OZZ, Target_OZZ_decay, Target_OZZ_dpozz

def get_finetune_config() -> dict:
    """Конфигурация fine-tuning."""
    return {
        # Данные
        'window_size': 320,
        'downsampling_stride': 16,        # Вычисляется автоматически из stride_fraction
        'stride_fraction': 2,             # Доля периода: 2 = полпериода (16), 4 = четверть (8)
        'feature_mode': 'phase_polar',
        'num_harmonics': 9,
        'sub_periods': [2, 4, 6, 10],
        'use_augmentation': True,
        'use_low_harmonics': True,
        'include_symmetric': True,        # Симметричные составляющие (I1,I2,I0,U1,U2,U0)
        'future_zones': 4,                # Зон вперёд (4 зоны модели, не зависит от частоты дискретизации)
        'zone_target_aggregation': 'max', # Агрегация меток внутри зоны: 'max' | 'mean'
        'train_batches_per_epoch': 64,    # Сколько batch-ов случайно брать за эпоху
        'val_stride_multiplier': 4,       # Во сколько раз реже брать окна на валидации
        'batch_size': 32,
        'val_batch_size': 64,
        'num_workers': 8,                  # параллельная предподготовка (FFT) в фоне
        'val_split': 0.2,                 # Доля файлов для валидации (0.0–1.0), определяет размер валидационного набора
        'target_level': 'base3',           # 3 класса (Normal = все ниже порога)

        # Модель (должно совпадать с pretrain; по умолчанию = light)
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 220,  # 8 × (9+4) × 2 = 208 + 6 × 2 = 12 = 220
        'd_model': 48,
        'num_heads': 4,
        'num_layers': 6,
        'kan_grid_size': 5,
        'dropout': 0.1,
        'cls_head_type': 'kan',
        'use_angle_gate': True,           # DirectionalRelayGate (направленный орган)
        'use_mixed_layer_norm': False,    # False = AmpOnlyLayerNorm (углы не норм.)

        # Fine-tuning специфика
        'num_classes': NUM_BASE_CLASSES, # Число выходных классов для головы классификатора (автоматически корректируется по target_columns)
        'zone_size': 1,                  # Каждый временной шаг = отдельная зона
        'supervision_mode': 'zone',      # 'zone' | 'window' | 'last_zone'

        # Обучение
        'epochs': 50,
        'lr_backbone': 0.5e-3,     # Низкий LR для backbone (уже обучен SSL)
        'lr_head':       1e-3,      # Высокий LR для новой головы
        'weight_decay': 1e-5,    # L2-регуляризация (weight decay) для AdamW, помогает бороться с переобучением
        'warmup_epochs': 2,      # Число эпох линейного warmup для LR (0 = без разогрева)
        'scheduler': 'cosine_warm_restarts',  # 'cosine' | 'cosine_warm_restarts'
        'scheduler_T0': 50,      # Длина первого цикла для WarmRestarts (в эпохах)
        'scheduler_T_mult': 2,   # Множитель длины следующего цикла (50→100→200...)
        'scheduler_eta_min': 1e-7,  # Минимальный LR
        'use_amp': True,         # Включить mixed precision (AMP) на CUDA для ускорения и экономии памяти
        'grad_clip': 1.0,        # Максимальная норма градиентов для clip_grad_norm_ (предотвращает взрыв градиентов)
        'accumulation_steps': 8, # effective batch = 32 × 8 = 256

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,
        'seed': 42,

        # Данные (путь)
        'data_dir': str(PROJECT_ROOT / 'data' / 'ml_datasets'),
        'precomputed_file': 'labeled_2025_12_03.csv',
    }


# ---------------------------------------------------------------------------
# Создание модели с классификационной головой
# ---------------------------------------------------------------------------

def create_model_for_finetune(
    config: dict,
    ssl_checkpoint_path: str | None = None,
) -> tuple[nn.Module, dict]:
    """Создаёт модель для fine-tuning, опционально загружая SSL-чекпоинт.

    Стратегия загрузки:
    1. Создаём модель с num_classes (имеет и ssl_head, и cls_head)
    2. Если есть SSL-чекпоинт — загружаем веса backbone (strict=False,
       чтобы пропустить ssl_head если размеры не совпадают)
    3. cls_head инициализируется случайно (новая задача)

    Returns:
        (model, ssl_info) — модель и информация от SSL-чекпоинта
    """
    model_type = config['model_type']

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            zone_size=config['zone_size'],
            kan_grid_size=config['kan_grid_size'],
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            cls_head_type=config.get('cls_head_type', 'kan'),
            num_future_zones=config.get('num_future_zones', 0),
            dropout=config['dropout'],
            max_seq_len=config.get('max_seq_len', 128),
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(
            num_input_channels=config['num_input_channels'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            zone_size=config['zone_size'],
            cls_head_type=config.get('cls_head_type', 'linear'),
            num_future_zones=config.get('num_future_zones', 0),
            dropout=config['dropout'],
            max_seq_len=config.get('max_seq_len', 128),
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    ssl_info = {}

    if ssl_checkpoint_path is not None:
        ckpt_path = Path(ssl_checkpoint_path)
        if ckpt_path.exists():
            print(f"Загрузка SSL-чекпоинта: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            ssl_state = ckpt['model_state_dict']

            # Фильтруем cls_head (его нет в SSL-чекпоинте) — strict=False
            missing, unexpected = model.load_state_dict(ssl_state, strict=False)
            print(f"  Пропущенные ключи (ожидаемо): {missing}")
            if unexpected:
                print(f"  Неожиданные ключи: {unexpected}")

            ssl_info = {
                'ssl_epoch': ckpt.get('epoch', -1),
                'ssl_val_loss': ckpt.get('best_val_loss', float('inf')),
                'ssl_config': ckpt.get('config', {}),
            }
            print(f"  SSL эпоха: {ssl_info['ssl_epoch']}, val_loss: {ssl_info['ssl_val_loss']:.4f}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: SSL-чекпоинт не найден: {ckpt_path}")

    return model, ssl_info


# ---------------------------------------------------------------------------
# Подготовка данных (с метками)
# ---------------------------------------------------------------------------

def _split_files_train_val(
    df: pl.DataFrame, val_split: float, seed: int,
    target_level: str = 'base',
    split_from: str | None = None,
) -> tuple[list[str], list[str]]:
    """Разделяет файлы на train/val (детерминированно).

    Для target_level='ozz' использует стратифицированное разделение,
    гарантирующее представительство каждого подкласса ОЗЗ в обоих сплитах.

    Если split_from указан — загружает split из split.json предыдущего эксперимента.
    """
    # Загрузка фиксированного split из предыдущего эксперимента
    if split_from:
        split_path = Path(split_from) / 'split.json'
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            train_files = split_data['train_files']
            val_files = split_data['val_files']
            # Проверяем, что файлы из split существуют в текущем DataFrame
            available = set(df['file_name'].unique().to_list())
            missing = set(train_files + val_files) - available
            if missing:
                print(f"  [!] {len(missing)} файлов из split не найдены в данных, пропускаю их")
                train_files = [f for f in train_files if f in available]
                val_files = [f for f in val_files if f in available]
            print(f"  [Q6: Split загружен из {split_path}]")
            print(f"    train: {len(train_files)} файлов, val: {len(val_files)} файлов")
            return train_files, val_files
        else:
            print(f"  [!] split.json не найден в {split_from}, генерирую новый split")
    if target_level == 'ozz':
        from osc_tools.data_management.ozz_split import (
            stratified_ozz_split, classify_file_ozz,
        )
        train_files, val_files, stats = stratified_ozz_split(
            df, test_size=val_split, random_state=seed, min_test_per_class=1,
        )

        # Гарантируем представительство в ОБОИХ сплитах:
        # если train пуст по какому-то классу, а val имеет >= 2 файла → перенести 1 из val в train
        train_set = set(train_files)
        val_set = set(val_files)
        file_classes = {}
        for fname in train_files + val_files:
            fdf = df.filter(pl.col('file_name') == fname)
            file_classes[fname] = classify_file_ozz(fdf)

        for cls in ('dpozz', 'decay', 'stable'):
            train_cls = [f for f in train_files if file_classes[f] == cls]
            val_cls = [f for f in val_files if file_classes[f] == cls]
            if len(train_cls) == 0 and len(val_cls) >= 2:
                # Переносим первый файл из val в train
                move_file = val_cls[0]
                val_files.remove(move_file)
                train_files.append(move_file)
                print(f"  [!] Перенос {move_file[:30]}... ({cls}) из val->train для баланса")

        print(f"  [Стратифицированный OZZ split]")
        print(f"    train: {len(train_files)} файлов, val: {len(val_files)} файлов")
        return train_files, val_files

    files = sorted(df['file_name'].unique().to_list())
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]
    return train_files, list(val_files)


def prepare_finetune_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Создаёт train/val DataLoader-ы для fine-tuning (с метками).

    Если use_augmentation или use_low_harmonics — AugmentedSpectralDataset (on-the-fly FFT).
    Иначе — PrecomputedDataset (legacy).

    Returns:
        (train_loader, val_loader, target_columns)
    """
    data_path = Path(config['data_dir']) / config['precomputed_file']
    print(f"Загрузка данных: {data_path}")
    df = pl.read_csv(str(data_path), infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
    print(f"  Строк: {df.height:,}, Колонок: {df.width}")

    # Стандартизация колонок напряжений (если raw CSV с 'UA BB' вместо 'UA')
    df = standardize_voltage_columns(df)

    # Подготовка меток: если raw CSV без Target_* — создаём их
    if 'Target_Normal' not in df.columns:
        df = clean_labels(df)
        df = add_base_labels(df)
        print("  Метки подготовлены из ML_* колонок")

    # Подготовка меток (нужна для base_sequential, ozz и др.)
    target_level = config['target_level']
    if target_level in ('base_sequential', 'ozz', 'full', 'full_by_levels'):
        print(f"  [Подготовка меток для уровня: {target_level}]")
        df = prepare_labels_for_experiment(df, target_level)

    # Целевые колонки
    target_columns = get_target_columns(target_level, df)
    print(f"  Целевые колонки ({target_level}): {target_columns}")

    # Автоматическая коррекция num_classes в config
    if config.get('num_classes') != len(target_columns):
        print(f"  [!] Корректировка num_classes: {config.get('num_classes')} -> {len(target_columns)}")
        config['num_classes'] = len(target_columns)

    # Разделяем файлы
    train_files, val_files = _split_files_train_val(
        df, config['val_split'], config['seed'],
        target_level=target_level,
        split_from=config.get('split_from'),
    )
    # Сохраняем split в config для последующей записи в файл
    config['_split_data'] = {'train_files': train_files, 'val_files': val_files}
    train_df = df.filter(pl.col('file_name').is_in(train_files))
    val_df = df.filter(pl.col('file_name').is_in(val_files))
    print(f"  Файлов: train={len(train_files)}, val={len(val_files)}")
    print(f"  Строк: train={train_df.height:,}, val={val_df.height:,}")

    # Stride из доли периода
    stride_frac = config.get('stride_fraction', 2)
    stride = compute_stride(SAMPLES_PER_PERIOD, stride_frac)
    config['downsampling_stride'] = stride
    val_stride = max(stride, stride * config.get('val_stride_multiplier', 4))
    window_size = config['window_size']

    use_augmented = config.get('use_augmentation', False) or config.get('use_low_harmonics', False)

    if use_augmented:
        sub_periods = config.get('sub_periods', [2, 4, 6, 10]) if config.get('use_low_harmonics') else []
        include_symmetric = config.get('include_symmetric', True)
        augmenter = TimeSeriesAugmenter() if config.get('use_augmentation') else None
        future_zones = config.get('future_zones', 0)
        # Обратная совместимость: если задан future_periods, конвертируем
        if future_zones == 0 and config.get('future_periods', 0) > 0:
            future_zones = config['future_periods'] * 32 // stride
            config['future_zones'] = future_zones
        zone_target_aggregation = config.get('zone_target_aggregation', 'max')

        # Пересчёт num_input_channels
        num_lh = len(sub_periods)
        actual_channels = compute_num_channels(config.get('num_harmonics', 9), num_lh, include_symmetric)
        if config.get('num_input_channels') != actual_channels:
            print(f"  [!] Корректировка num_input_channels: {config['num_input_channels']} -> {actual_channels}")
            config['num_input_channels'] = actual_channels

        # Индексы: учитываем будущие зоны (future_raw = future_zones * stride)
        full_window = window_size + future_zones * stride
        train_indices = PrecomputedDataset.create_indices(
            train_df, window_size=full_window, mode='val', stride=stride,
        )
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=full_window, mode='val', stride=val_stride,
        )
        print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,} (val_stride={val_stride})")

        train_boundaries = AugmentedSpectralDataset.compute_file_boundaries(train_df)
        val_boundaries = AugmentedSpectralDataset.compute_file_boundaries(val_df)

        train_ds = AugmentedSpectralDataset(
            dataframe=train_df,
            file_boundaries=train_boundaries,
            indices=train_indices,
            window_size=window_size,
            num_harmonics=config['num_harmonics'],
            sub_periods=sub_periods if sub_periods else None,
            include_symmetric=include_symmetric,
            downsampling_stride=stride,
            future_zones=future_zones,
            mask_ratio=0.0,         # Нет маскирования при finetune
            augmenter=augmenter,
            target_columns=target_columns,
            mode='classify',
            target_window_mode='any_in_window',
            zone_target_aggregation=zone_target_aggregation,
        )
        val_ds = AugmentedSpectralDataset(
            dataframe=val_df,
            file_boundaries=val_boundaries,
            indices=val_indices,
            window_size=window_size,
            num_harmonics=config['num_harmonics'],
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
        # Legacy: PrecomputedDataset
        train_indices = PrecomputedDataset.create_indices(
            train_df, window_size=window_size, mode='val', stride=stride,
        )
        val_indices = PrecomputedDataset.create_indices(
            val_df, window_size=window_size, mode='val', stride=val_stride,
        )
        print(f"  Окон: train={len(train_indices):,}, val={len(val_indices):,} (val_stride={val_stride})")

        train_ds = PrecomputedDataset(
            dataframe=train_df,
            indices=train_indices,
            window_size=window_size,
            feature_mode=config['feature_mode'],
            target_columns=target_columns,
            sampling_strategy='stride',
            downsampling_stride=stride,
            target_position=window_size - 1,
            target_window_mode='any_in_window',
            num_harmonics=config['num_harmonics'],
        )
        val_ds = PrecomputedDataset(
            dataframe=val_df,
            indices=val_indices,
            window_size=window_size,
            feature_mode=config['feature_mode'],
            target_columns=target_columns,
            sampling_strategy='stride',
            downsampling_stride=stride,
            target_position=window_size - 1,
            target_window_mode='any_in_window',
            num_harmonics=config['num_harmonics'],
        )

    # num_future_zones для FuturePredictionHead — напрямую из future_zones
    if use_augmented and future_zones > 0:
        zone_size = config.get('zone_size', 1)
        num_future_zones = future_zones // zone_size
        config['num_future_zones'] = num_future_zones
        print(f"  Future zones: {num_future_zones} (future_zones={future_zones}, "
              f"stride={stride}, zone_size={zone_size})")
    else:
        config['num_future_zones'] = 0

    # Балансировка выборки
    sample_weights = _compute_sample_weights(train_ds, config)
    # Сохраним для повторного использования при каждой эпохе
    config['_sample_weights'] = sample_weights

    train_loader = _build_train_epoch_loader(train_ds, config, epoch=0, sample_weights=sample_weights)
    val_loader = DataLoader(
        val_ds, batch_size=config['val_batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True,
        **_extra_loader_kwargs(config),
    )

    return train_loader, val_loader, target_columns


# ---------------------------------------------------------------------------
# Балансировка выборки
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    dataset: torch.utils.data.Dataset,
    config: dict,
) -> Optional[np.ndarray]:
    """Вычисляет веса для WeightedRandomSampler.

    Стратегия combine: для каждого окна определяем «класс» по доминирующей
    метке, и даём обратный вес пропорционально частоте этого класса.
    Также лимитируем число окон от одного файла (per-file cap).

    Args:
        dataset: train dataset с ._targets_np и .indices
        config: конфиг с 'balanced_sampling' и 'max_windows_per_file'

    Returns:
        (N,) float weights или None, если балансировка отключена
    """
    if not config.get('balanced_sampling', False):
        return None

    # Нужны: метки для каждого окна и файловая принадлежность
    has_targets = hasattr(dataset, '_targets_np') and hasattr(dataset, 'indices')
    if not has_targets:
        print("  [WARNING] balanced_sampling: dataset не поддерживает — пропускаем")
        return None

    indices = dataset.indices
    targets_np = dataset._targets_np
    window_size = getattr(dataset, 'window_size', config.get('window_size', 320))
    n_samples = len(indices)

    # 1. Класс каждого окна (по доминирующей метке в окне)
    labels = np.zeros(n_samples, dtype=np.int32)
    for i, start_idx in enumerate(indices):
        end_idx = min(start_idx + window_size, len(targets_np))
        window_targets = targets_np[start_idx:end_idx]  # (W, C)
        # Любое событие в окне → кодируем как бинарный multi-label
        any_event = window_targets.max(axis=0)  # (C,)
        if any_event.sum() == 0:
            labels[i] = 0  # Normal
        else:
            # Кодируем как номер доминирующего класса (1-indexed)
            labels[i] = int(any_event.argmax()) + 1

    # 2. Обратные частоты → веса
    unique_labels, counts = np.unique(labels, return_counts=True)
    freq = {lbl: cnt for lbl, cnt in zip(unique_labels, counts)}
    weights = np.array([1.0 / freq[lbl] for lbl in labels], dtype=np.float64)

    # 3. Per-file cap: ограничить вклад одного файла
    max_per_file = config.get('max_windows_per_file', 0)
    if max_per_file > 0 and hasattr(dataset, '_file_bounds'):
        file_counts = {}
        for i, start_idx in enumerate(indices):
            for fi, (fstart, fend) in enumerate(dataset._file_bounds):
                if fstart <= start_idx < fend:
                    file_counts.setdefault(fi, []).append(i)
                    break
        for fi, sample_indices in file_counts.items():
            if len(sample_indices) > max_per_file:
                # Уменьшаем вес лишних сэмплов пропорционально
                scale = max_per_file / len(sample_indices)
                for si in sample_indices:
                    weights[si] *= scale

    # Нормализация (сумма = N)
    weights = weights / weights.sum() * n_samples

    print(f"  [Balanced sampling] classes: {dict(zip(unique_labels.tolist(), counts.tolist()))}, "
          f"max_per_file={max_per_file}")
    return weights


# ---------------------------------------------------------------------------
# Обучение и валидация
# ---------------------------------------------------------------------------

def _extra_loader_kwargs(config: dict) -> dict:
    """Доп. kwargs для DataLoader: persistent_workers + prefetch_factor при num_workers > 0."""
    nw = config.get('num_workers', 0)
    if nw > 0:
        return {'persistent_workers': True, 'prefetch_factor': 2}
    return {}


def _build_train_epoch_loader(
    train_ds: torch.utils.data.Dataset,
    config: dict,
    epoch: int,
    sample_weights: Optional[np.ndarray] = None,
) -> DataLoader:
    """Строит train DataLoader с фиксированной случайной подвыборкой на эпоху.

    Если sample_weights задан — использует WeightedRandomSampler.
    """
    from torch.utils.data import WeightedRandomSampler

    batches_per_epoch = config.get('train_batches_per_epoch')
    total_samples = len(train_ds)

    # Число элементов в эпохе
    if batches_per_epoch is not None:
        target_samples = int(batches_per_epoch) * config['batch_size']
        if target_samples <= 0 or target_samples >= total_samples:
            target_samples = total_samples
    else:
        target_samples = total_samples

    # Weighted sampler или обычный shuffle/subset
    if sample_weights is not None:
        weights_tensor = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=target_samples,
            replacement=True,  # Для балансировки нужен replacement
        )
        return DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            sampler=sampler,
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            **_extra_loader_kwargs(config),
        )

    # Без балансировки — стандартная логика
    if batches_per_epoch is None:
        return DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            **_extra_loader_kwargs(config),
        )

    target_samples = int(batches_per_epoch) * config['batch_size']
    total_samples = len(train_ds)
    if target_samples <= 0 or target_samples >= total_samples:
        return DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
            **_extra_loader_kwargs(config),
        )

    rng = np.random.default_rng(config.get('seed', 42) + epoch)
    sampled_indices = rng.choice(total_samples, size=target_samples, replace=False).tolist()
    sampler = SubsetRandomSampler(sampled_indices)
    return DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        sampler=sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        **_extra_loader_kwargs(config),
    )
def _reduce_logits_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervision_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Приводит logits/targets к форме для loss и метрик.

    `zone`: обучаем настоящее зонирование, как token classification / 1D-segmentation.
    `window`: одно решение на всё окно.
    `last_zone`: решение только по последней зоне окна.
    """
    if supervision_mode == 'zone':
        B, T_z, C_cls = logits.shape
        T_y = targets.shape[1]
        # Выравнивание при рассогласовании модели и меток (edge case)
        if T_z != T_y:
            T_min = min(T_z, T_y)
            logits = logits[:, :T_min, :]
            targets = targets[:, :T_min, :]
            T_z = T_min
        return logits.reshape(B * T_z, C_cls).float(), targets.reshape(B * T_z, C_cls).float()
    if supervision_mode == 'window':
        return logits.mean(dim=1).float(), targets.max(dim=1).values.float()
    if supervision_mode == 'last_zone':
        return logits[:, -1, :].float(), targets[:, -1, :].float()
    raise ValueError(f"Неизвестный supervision_mode: {supervision_mode}")


def _compute_supervision_pos_weight(
    loader: DataLoader,
    supervision_mode: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """pos_weight в той же редукции, в которой считается loss."""
    total = 0
    pos_counts = None
    for _, y in loader:
        y = y.float()
        if y.dim() == 2:
            y_used = y
        elif supervision_mode == 'zone':
            y_used = y.reshape(-1, y.shape[-1])
        elif supervision_mode == 'window':
            y_used = y.max(dim=1).values
        elif supervision_mode == 'last_zone':
            y_used = y[:, -1, :]
        else:
            raise ValueError(f"Неизвестный supervision_mode: {supervision_mode}")

        if pos_counts is None:
            pos_counts = torch.zeros(y_used.shape[-1], dtype=torch.float64)
        pos_counts += y_used.sum(dim=0).double()
        total += y_used.shape[0]

    if pos_counts is None:
        raise ValueError("Нет данных для вычисления pos_weight")

    neg_counts = total - pos_counts
    pos_safe = pos_counts.clone()
    pos_safe[pos_safe == 0] = 1.0
    weights = neg_counts / pos_safe
    return weights.float().to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    accumulation_steps: int = 1,
    epoch_idx: int = 0,
    total_epochs: int = 1,
) -> dict[str, float]:
    """Одна эпоха fine-tuning с gradient accumulation.

    Модель выдаёт (B, num_zones, num_classes). Loss считается
    по всем зонам: каждая зона (полпериода) имеет свою метку.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    t0 = time.perf_counter()
    supervision_mode = getattr(loader.dataset, 'supervision_mode', 'zone')

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch_idx + 1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, (x, y) in enumerate(progress):
        x = x.to(device)       # (B, C, T)
        y = y.to(device)       # (B, T_zones, num_classes) — позонные метки

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x, mode='classify')
            logits = out['classify']  # (B, num_zones, num_classes)

        logits_used, targets_used = _reduce_logits_targets(logits, y, supervision_mode)
        loss = loss_fn(logits_used, targets_used)

        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * (accumulation_steps if accumulation_steps > 1 else 1)
        if not np.isnan(loss_val):
            total_loss += loss_val
            num_batches += 1

        # Метрики: в той же постановке, что и loss
        with torch.no_grad():
            probs_used = torch.sigmoid(logits_used).cpu().numpy()
            targets_np = targets_used.cpu().numpy()
            all_preds.append(probs_used)
            all_targets.append(targets_np)
        progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

    progress.close()

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    metrics['time_sec'] = elapsed

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int = 0,
    total_epochs: int = 1,
) -> dict[str, float]:
    """Валидация по всем зонам."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    t0 = time.perf_counter()
    supervision_mode = getattr(loader.dataset, 'supervision_mode', 'zone')

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Val   {epoch_idx + 1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for x, y in progress:
        x = x.to(device)       # (B, C, T)
        y = y.to(device)       # (B, T_zones, num_classes)

        out = model(x, mode='classify')
        logits = out['classify']  # (B, num_zones, num_classes)

        logits_used, targets_used = _reduce_logits_targets(logits, y, supervision_mode)
        loss = loss_fn(logits_used, targets_used)

        total_loss += loss.item()
        num_batches += 1

        probs_used = torch.sigmoid(logits_used).cpu().numpy()
        targets_np = targets_used.cpu().numpy()
        all_preds.append(probs_used)
        all_targets.append(targets_np)
        progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

    progress.close()

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(num_batches, 1)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss
    metrics['time_sec'] = elapsed

    return metrics


def _compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Вычисляет Macro-F1, ROC-AUC, Accuracy.

    Args:
        preds: (N, C) вероятности после sigmoid
        targets: (N, C) бинарные метки
        threshold: порог бинаризации (0.5 — стандартный для sigmoid)
    """
    from sklearn.metrics import f1_score, roc_auc_score

    preds_bin = (preds >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    metrics = {}

    # Средние вероятности по классам (для диагностики)
    metrics['mean_prob'] = float(preds.mean())
    for c in range(preds.shape[1]):
        metrics[f'mean_prob_cls_{c}'] = float(preds[:, c].mean())

    # Macro-F1
    metrics['macro_f1'] = float(f1_score(
        targets_int, preds_bin, average='macro', zero_division=0,
    ))

    # Per-class F1
    per_cls_f1 = f1_score(targets_int, preds_bin, average=None, zero_division=0)
    for i, f in enumerate(per_cls_f1):
        metrics[f'f1_class_{i}'] = float(f)

    # ROC-AUC (macro, только для классов с обоими значениями 0 и 1)
    try:
        valid_cols = []
        for c in range(targets.shape[1]):
            if len(np.unique(targets_int[:, c])) > 1:
                valid_cols.append(c)
        if valid_cols:
            metrics['roc_auc'] = float(roc_auc_score(
                targets_int[:, valid_cols], preds[:, valid_cols],
                average='macro',
            ))
        else:
            metrics['roc_auc'] = 0.0
    except ValueError:
        metrics['roc_auc'] = 0.0

    # Exact match
    metrics['exact_match'] = float((preds_bin == targets_int).all(axis=1).mean())

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_columns: list[str],
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    """Вычисляет и сохраняет confusion matrix для каждого класса в текстовый файл."""
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds = []
    all_targets = []
    supervision_mode = getattr(loader.dataset, 'supervision_mode', 'zone')

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x, mode='classify')
            logits = out['classify']
            logits_used, targets_used = _reduce_logits_targets(logits, y, supervision_mode)
            probs = torch.sigmoid(logits_used).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets_used.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    preds_bin = (all_preds >= threshold).astype(np.int32)
    targets_int = all_targets.astype(np.int32)

    lines = [f"Confusion Matrix (threshold={threshold})", "=" * 50, ""]
    for i, col in enumerate(target_columns):
        cm = confusion_matrix(targets_int[:, i], preds_bin[:, i], labels=[0, 1])
        lines.append(f"--- {col} ---")
        lines.append(f"  TN={cm[0, 0]:6d}  FP={cm[0, 1]:6d}")
        lines.append(f"  FN={cm[1, 0]:6d}  TP={cm[1, 1]:6d}")
        lines.append("")

    text = "\n".join(lines)
    save_path.write_text(text, encoding='utf-8')
    print(f"  Confusion matrix сохранена: {save_path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_f1: float,
    config: dict,
    best_val_loss: float = float('inf'),
) -> None:
    """Сохраняет fine-tuning checkpoint."""
    state = {
        'epoch': epoch,
        'best_val_f1': best_val_f1,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: torch.amp.GradScaler | None = None,
    reset_optimizer: bool = False,
) -> dict:
    """Загружает checkpoint. Возвращает мета-информацию (epoch, best_val_f1)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if reset_optimizer:
        print("  [!] Optimizer, scheduler and epoch state will be reset (starting from 0).")
        return {
            'epoch': -1,
            'best_val_f1': 0.0,
            'best_val_loss': float('inf'),
        }

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return {
        'epoch': ckpt.get('epoch', 0),
        'best_val_f1': ckpt.get('best_val_f1', 0.0),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
    }


# ---------------------------------------------------------------------------
# Основной цикл fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    config: dict,
    ssl_checkpoint: str | None = None,
    resume_path: str | None = None,
    reset_optimizer: bool = False,
) -> Path:
    """Supervised fine-tuning на задачу классификации.

    1. Загружает данные с метками
    2. Создаёт модель, инициализирует из SSL-чекпоинта
    3. Обучает с раздельным LR (backbone vs head)
    4. Сохраняет лучшую модель по Macro-F1

    Returns:
        Путь к директории с результатами
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Информативное именование — включаем target_level в имя директории
    run_name = config.get('run_name')
    if run_name:
        dir_name = run_name
    else:
        target_level = config.get('target_level', 'base')
        dir_name = f"finetune_{config['model_type']}_{target_level}_{timestamp}"
    save_dir = Path(config['save_dir']) / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FINE-TUNING — {config['model_type']}")
    if resume_path:
        print(f"ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ: {resume_path}")
    else:
        print(f"SSL чекпоинт: {ssl_checkpoint or 'НЕТ (random init)'}")
    print(f"Сохранение: {save_dir}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Данные (ВНИМАНИЕ: может корректировать num_classes в config!) ---
    train_loader, val_loader, target_columns = prepare_finetune_dataloaders(config)

    # Сохраняем конфиг ПОСЛЕ подготовки данных (num_classes уже скорректирован)
    # Убираем внутренние данные split из сохраняемого конфига
    split_data = config.pop('_split_data', None)
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Сохраняем split в отдельный файл для воспроизводимости
    if split_data:
        with open(save_dir / 'split.json', 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Split сохранён: {save_dir / 'split.json'}")

    # Статистика классов
    _print_class_distribution(train_loader, target_columns)

    # --- Модель ---
    model, ssl_info = create_model_for_finetune(config, ssl_checkpoint)
    model = model.to(device)
    n_params = model.num_parameters()
    print(f"Параметры модели: {n_params:,}")

    supervision_mode = config.get('supervision_mode', 'zone')
    train_loader.dataset.supervision_mode = supervision_mode
    val_loader.dataset.supervision_mode = supervision_mode

    # --- Loss: BCEWithLogitsLoss с pos_weight в той же постановке, что и supervision ---
    pos_weight = _compute_supervision_pos_weight(
        train_loader,
        supervision_mode=supervision_mode,
        device=device,
    )
    print(f"supervision_mode: {supervision_mode}")
    print(f"cls_head_type: {config.get('cls_head_type', 'kan')}")
    print(f"use_angle_gate: {config.get('use_angle_gate', True)}")
    print(f"use_mixed_layer_norm: {config.get('use_mixed_layer_norm', False)}")
    print(f"pos_weight ({supervision_mode}): {pos_weight.cpu().numpy()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Оптимизатор: раздельный LR ---
    # Backbone (всё кроме cls_head) — низкий LR
    # cls_head — высокий LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'cls_head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr_backbone']},
        {'params': head_params, 'lr': config['lr_head']},
    ], weight_decay=config['weight_decay'])

    print(f"Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={config['lr_backbone']})")
    print(f"Head params: {sum(p.numel() for p in head_params):,} (lr={config['lr_head']})")

    # --- LR Scheduler ---
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config['epochs']
    scheduler_type = config.get('scheduler', 'cosine_warm_restarts')
    eta_min = config.get('scheduler_eta_min', 1e-7)

    if scheduler_type == 'cosine_warm_restarts':
        T_0 = config.get('scheduler_T0', 50)
        T_mult = config.get('scheduler_T_mult', 2)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, T_0), T_mult=T_mult, eta_min=eta_min,
        )
        print(f"LR Scheduler: CosineAnnealingWarmRestarts "
              f"(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    else:
        # Классический CosineAnnealing (monotonic decay)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min,
        )
        print(f"LR Scheduler: CosineAnnealingLR "
              f"(T_max={total_epochs - warmup_epochs}, eta_min={eta_min})")

    # --- AMP ---
    scaler = None
    if config['use_amp'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision: включён (AMP)")

    # --- Resume/SSL Start ---
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float('inf')  # отслеживаем лучший loss отдельно

    if resume_path is not None:
        resume_p = Path(resume_path)
        if resume_p.exists():
            print(f"Восстановление из чекпоинта: {resume_p}")
            meta = load_checkpoint(
                resume_p, model, optimizer, cosine_scheduler, scaler,
                reset_optimizer=reset_optimizer,
            )
            start_epoch = meta['epoch'] + 1
            best_val_f1 = meta['best_val_f1']
            best_val_loss = meta.get('best_val_loss', float('inf'))
            if not reset_optimizer:
                print(f"  Продолжаем с эпохи {start_epoch}, best_val_f1={best_val_f1:.4f}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: чекпоинт не найден: {resume_p}")
    # --- Лог ---
    history = []
    log_path = save_dir / 'training_log.jsonl'

    print(f"\nНачало fine-tuning: {total_epochs} эпох, batch_size={config['batch_size']}")
    print(f"Целевые классы: {target_columns}")
    accum = config.get('accumulation_steps', 1)
    print(f"Gradient accumulation: {accum} шагов -> effective batch = {config['batch_size'] * accum}")
    if config.get('train_batches_per_epoch') is not None:
        print(f"Train batches per epoch: {config['train_batches_per_epoch']} (случайная подвыборка)")
    print("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        train_loader = _build_train_epoch_loader(
            train_loader.dataset, config, epoch,
            sample_weights=config.get('_sample_weights'),
        )

        # Warmup LR (линейный от 0 до target)
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]['lr'] = config['lr_backbone'] * frac
            optimizer.param_groups[1]['lr'] = config['lr_head'] * frac

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, config.get('grad_clip', 1.0),
            accumulation_steps=accum,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        # Val
        val_metrics = validate(
            model, val_loader, loss_fn, device,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        # LR step
        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        # Логирование
        is_best = val_metrics['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['macro_f1']

        # отслеживаем лучший loss отдельно
        is_best_loss = val_metrics['loss'] < best_val_loss
        if is_best_loss:
            best_val_loss = val_metrics['loss']

        vram_mb = 0.0
        if device.type == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['macro_f1'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['macro_f1'],
            'val_roc_auc': val_metrics.get('roc_auc', 0.0),
            'val_exact_match': val_metrics.get('exact_match', 0.0),
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'train_time': train_metrics['time_sec'],
            'val_time': val_metrics['time_sec'],
            'vram_mb': vram_mb,
            'is_best': is_best,
            'is_best_loss': is_best_loss,
        }
        # Per-class F1 на валидации
        for i, col in enumerate(target_columns):
            record[f'val_f1_{col}'] = val_metrics.get(f'f1_class_{i}', 0.0)

        history.append(record)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        marker = ' ★' if is_best else (' ★L' if is_best_loss else '')
        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        print(
            f"Epoch {epoch + 1:3d}/{total_epochs} | "
            f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"F1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}{marker} | "
            f"AUC={val_metrics.get('roc_auc', 0):.4f} | "
            f"lr={lr_bb:.1e}/{lr_hd:.1e} | "
            f"p̄={val_metrics.get('mean_prob', 0):.3f} | "
            f"time={train_metrics['time_sec'] + val_metrics['time_sec']:.1f}s"
        )
        # Per-class F1 (каждую эпоху)
        cls_f1s = [f"{val_metrics.get(f'f1_class_{i}', 0):.3f}" for i in range(len(target_columns))]
        cls_probs = [f"{val_metrics.get(f'mean_prob_cls_{i}', 0):.3f}" for i in range(len(target_columns))]
        print(f"         Per-class F1: {cls_f1s}  |  mean_prob: {cls_probs}")

        # Checkpointing
        # TODO: рассмотреть взвешенный чекпоинт (weighted: α*F1 + β*(1-loss)),
        #       чтобы выбирать модель, которая хороша и по F1, и по loss одновременно.
        save_checkpoint(
            save_dir / 'latest_checkpoint.pt',
            model, optimizer, cosine_scheduler, scaler,
            epoch, best_val_f1, config, best_val_loss,
        )
        if is_best:
            save_checkpoint(
                save_dir / 'best_model.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config, best_val_loss,
            )
        # отдельный чекпоинт по лучшему loss
        if is_best_loss:
            save_checkpoint(
                save_dir / 'best_model_loss.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config, best_val_loss,
            )
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint(
                save_dir / f'checkpoint_epoch_{epoch:03d}.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config, best_val_loss,
            )

        # Детекция переобучения: N эпох подряд val_loss не улучшается
        overfit_patience = config.get('overfit_patience', 20)
        if len(history) >= overfit_patience:
            recent_losses = [r['val_loss'] for r in history[-overfit_patience:]]
            if all(recent_losses[i] >= recent_losses[0] for i in range(1, len(recent_losses))):
                print(f"  ⚠ ВНИМАНИЕ: val_loss не улучшается {overfit_patience} эпох подряд "
                      f"(возможно переобучение)")

    # Итоги
    print("\n" + "=" * 60)
    print("FINE-TUNING ЗАВЕРШЁН")
    print(f"  Лучший val Macro-F1: {best_val_f1:.4f}")
    if history:
        best_epoch = max(history, key=lambda r: r['val_f1'])['epoch']
        print(f"  Лучшая эпоха: {best_epoch}")
        best_rec = history[best_epoch]
        print(f"  ROC-AUC: {best_rec.get('val_roc_auc', 0):.4f}")
        print(f"  Per-class F1:")
        for col in target_columns:
            f1_val = best_rec.get(f'val_f1_{col}', 0)
            print(f"    {col}: {f1_val:.4f}")
    print(f"  Результаты: {save_dir}")
    print("=" * 60)

    # Confusion matrix на лучшей модели
    _save_confusion_matrix(
        model, val_loader, loss_fn, device, target_columns,
        save_dir / 'confusion_matrix.txt',
    )

    # Сохраняем финальную сводку
    summary = {
        'model_type': config['model_type'],
        'ssl_checkpoint': ssl_checkpoint,
        'best_epoch': best_epoch if history else -1,
        'best_val_f1': best_val_f1,
        'best_val_roc_auc': history[best_epoch].get('val_roc_auc', 0) if history else 0,
        'total_epochs': total_epochs,
        'num_params': n_params,
        'target_columns': target_columns,
        'ssl_info': ssl_info,
    }
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV-экспорт кривых обучения — для matplotlib
    if history:
        import csv
        csv_path = save_dir / 'training_curves.csv'
        fieldnames = list(history[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"  Кривые обучения: {csv_path}")

    return save_dir


def _print_class_distribution(loader: DataLoader, target_columns: list[str]) -> None:
    """Печатает распределение классов на уровне зон и на уровне окон."""
    total_zones = 0
    total_windows = 0
    zone_pos = None
    win_pos = None
    for _, y in loader:
        y_f = y.float()
        # Window-level: any_in_window
        if y_f.dim() == 3:
            y_win = y_f.max(dim=1).values  # (B, C)
            y_zone = y_f.reshape(-1, y_f.shape[-1])  # (B*T, C)
        else:
            y_win = y_f
            y_zone = y_f

        if zone_pos is None:
            zone_pos = torch.zeros(y_zone.shape[-1], dtype=torch.float64)
            win_pos = torch.zeros(y_win.shape[-1], dtype=torch.float64)
        zone_pos += y_zone.sum(dim=0).double()
        win_pos += y_win.sum(dim=0).double()
        total_zones += y_zone.shape[0]
        total_windows += y_win.shape[0]

    if zone_pos is None:
        return

    print(f"\n  Распределение классов (зон={total_zones}, окон={total_windows}):")
    for i, col in enumerate(target_columns):
        z_n = int(zone_pos[i].item())
        z_pct = 100.0 * z_n / total_zones if total_zones > 0 else 0
        w_n = int(win_pos[i].item())
        w_pct = 100.0 * w_n / total_windows if total_windows > 0 else 0
        print(f"    {col}: зон {z_n} ({z_pct:.1f}%), окон {w_n} ({w_pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к SSL-чекпоинту для инициализации')
    parser.add_argument('--model', choices=['PhysicalKANTransformer', 'BaselineTransformer'],
                        default='PhysicalKANTransformer', help='Тип модели')
    parser.add_argument('--complexity', choices=['light', 'medium', 'heavy'],
                        default=None, help='Уровень сложности модели')
    parser.add_argument('--epochs', type=int, default=None, help='Число эпох')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--target-level', choices=['base', 'base3', 'ozz', 'base_sequential'],
                        default=None, help='Задача классификации: base (4 класса), base3 (3 класса, без Normal), ozz (3 класса ОЗЗ)')
    parser.add_argument('--cls-head-type',
                        choices=['kan', 'mlp', 'linear'],
                        default=None,
                        help='Тип классификационной головы')
    parser.add_argument('--supervision-mode',
                        choices=['zone', 'window', 'last_zone'],
                        default=None,
                        help='Как считать loss: по всем зонам, по окну или только по последней зоне')
    parser.add_argument('--no-angle-gate', action='store_true',
                        help='Отключить DirectionalRelayGate (направленный орган)')
    parser.add_argument('--mixed-layer-norm', action='store_true',
                        help='Использовать стандартный LayerNorm на всём векторе (amp+angle)')
    parser.add_argument('--smoke', action='store_true', help='Smoke-test (2 эпохи)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Отключить аугментацию на сырых данных')
    parser.add_argument('--no-low-harmonics', action='store_true',
                        help='Отключить низшие гармоники')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help='Шаги gradient accumulation (override)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к finetune-чекпоинту для продолжения')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Сбросить оптимизатор и начать с 0 эпохи (использовать веса из resume)')
    parser.add_argument('--split-from', type=str, default=None,
                        help='Путь к директории эксперимента, из которой взять split.json')
    parser.add_argument('--balanced', action='store_true',
                        help='Включить балансировку выборки по классам')
    parser.add_argument('--max-windows-per-file', type=int, default=0,
                        help='Максимум окон от одного файла (0 = без лимита')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_finetune_config()

    config['model_type'] = args.model
    if args.target_level is not None:
        config['target_level'] = args.target_level
    if args.cls_head_type is not None:
        config['cls_head_type'] = args.cls_head_type
    if args.supervision_mode is not None:
        config['supervision_mode'] = args.supervision_mode
    if hasattr(args, 'no_angle_gate') and args.no_angle_gate:
        config['use_angle_gate'] = False
    if hasattr(args, 'mixed_layer_norm') and args.mixed_layer_norm:
        config['use_mixed_layer_norm'] = True
    if args.complexity:
        level = COMPLEXITY_LEVELS[args.complexity]
        config.update(level)
        print(f"Сложность: {args.complexity} -> {level}")
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.accumulation_steps is not None:
        config['accumulation_steps'] = args.accumulation_steps
    if args.no_augmentation:
        config['use_augmentation'] = False
    if args.no_low_harmonics:
        config['use_low_harmonics'] = False
        config['include_symmetric'] = False
        config['num_input_channels'] = 144
    if args.split_from:
        config['split_from'] = args.split_from
    if args.balanced:
        config['balanced_sampling'] = True
    if args.max_windows_per_file > 0:
        config['max_windows_per_file'] = args.max_windows_per_file
    if args.smoke:
        config['epochs'] = 2
        config['batch_size'] = 4
        config['val_batch_size'] = 4
        config['use_augmentation'] = False
        config['use_low_harmonics'] = False
        config['include_symmetric'] = False
        config['num_input_channels'] = 144
        config['accumulation_steps'] = 1
        config['future_zones'] = 0

    finetune(
        config,
        ssl_checkpoint=args.checkpoint,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
    )


if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Раскомментируйте нужный блок и запустите файл напрямую.
    # Для CLI: python run_phase4_finetune.py --checkpoint path/to/best_model.pt
    # =================================================================

    # --- Тип модели (должен совпадать с pretrain) ---
    MODEL_TYPE = 'PhysicalKANTransformer'   # 'PhysicalKANTransformer' | 'BaselineTransformer'

    # --- Сложность (должна совпадать с pretrain) ---
    SELECTED_COMPLEXITY = 'light'           # 'light' | 'medium' | 'heavy'

    # --- Путь к SSL-чекпоинту (None = random init) ---
    # SSL_CHECKPOINT = None
    SSL_CHECKPOINT = 'experiments/phase4/pretrain_PhysicalKANTransformer_20260327_155618/best_model.pt'

    # --- Продолжение fine-tuning ---
    # RESUME_PATH = None      # Путь к чекпоинту finetune (latest_checkpoint.pt)
    RESUME_PATH = 'experiments/phase4/finetune_PhysicalKANTransformer_20260328_230623/best_model.pt'
    RESET_OPTIMIZER = True # True, если нужно сбросить оптимизатор и начать с 0 эпохи

    # --- Эпохи ---
    EPOCHS = 100

    # --- Задача классификации ---
    # 'base' — 4 класса (Normal, ML_1, ML_2, ML_3)
    # 'ozz'  — 3 класса ОЗЗ (Target_OZZ, Target_OZZ_decay, Target_OZZ_dpozz)
    TARGET_LEVEL = 'ozz'
    CLS_HEAD_TYPE = 'kan'   # 'kan' | 'mlp' | 'linear'
    SUPERVISION_MODE = 'zone'             # 'zone' | 'window' | 'last_zone'
    USE_ANGLE_GATE = True                       # DirectionalRelayGate (направленный орган)
    USE_MIXED_LAYER_NORM = False                # False = AmpOnlyLayerNorm

    # --- Аугментация и признаки ---
    USE_AUGMENTATION = True
    USE_LOW_HARMONICS = True
    INCLUDE_SYMMETRIC = True                # Симметричные составляющие
    FUTURE_ZONES = 4                        # Зон вперёд (шаги модели, не зависит от частоты дискретизации)
    ZONE_TARGET_AGGREGATION = 'max'         # 'max' | 'mean'

    # --- Stride (доля периода: 2 = полпериода=16, 4 = четверть=8) ---
    STRIDE_FRACTION = 2
    VAL_STRIDE_MULTIPLIER = 5 # 4               # Валидация реже, чем обучение
    TRAIN_BATCHES_PER_EPOCH = 128 # 64           # Случайных batch-ов за эпоху

    # --- Gradient accumulation ---
    ACCUMULATION_STEPS = 8

    # --- Частота чекпоинтов ---
    CHECKPOINT_FREQUENCY = 5

    # =================================================================

    config = get_finetune_config()
    config['model_type'] = MODEL_TYPE
    config['target_level'] = TARGET_LEVEL
    config['cls_head_type'] = CLS_HEAD_TYPE
    config['supervision_mode'] = SUPERVISION_MODE
    config['use_angle_gate'] = USE_ANGLE_GATE
    config['use_mixed_layer_norm'] = USE_MIXED_LAYER_NORM
    config['epochs'] = EPOCHS
    config['use_augmentation'] = USE_AUGMENTATION
    config['use_low_harmonics'] = USE_LOW_HARMONICS
    config['include_symmetric'] = INCLUDE_SYMMETRIC
    config['future_zones'] = FUTURE_ZONES
    config['zone_target_aggregation'] = ZONE_TARGET_AGGREGATION
    config['accumulation_steps'] = ACCUMULATION_STEPS
    config['checkpoint_frequency'] = CHECKPOINT_FREQUENCY
    config['stride_fraction'] = STRIDE_FRACTION
    config['val_stride_multiplier'] = VAL_STRIDE_MULTIPLIER
    config['train_batches_per_epoch'] = TRAIN_BATCHES_PER_EPOCH

    level = COMPLEXITY_LEVELS[SELECTED_COMPLEXITY]
    config.update(level)

    # Пересчёт каналов
    num_lh = len(config['sub_periods']) if USE_LOW_HARMONICS else 0
    config['num_input_channels'] = compute_num_channels(
        config['num_harmonics'], num_lh, INCLUDE_SYMMETRIC,
    )
    if not USE_LOW_HARMONICS and not INCLUDE_SYMMETRIC:
        config['num_input_channels'] = 144

    finetune(
        config,
        ssl_checkpoint=SSL_CHECKPOINT,
        resume_path=RESUME_PATH,
        reset_optimizer=RESET_OPTIMIZER,
    )
