"""
Fine-tuning скрипт для Этапа 4.5: Simulated_OZZ_v1.

Обучает Physical KAN-Transformer / BaselineTransformer на задачу
классификации типов ОЗЗ/ДПОЗЗ (4 класса + implicit Normal) на
симулированных осциллограммах RTDS.

Ключевые отличия от ``run_phase4_finetune.py``:
- **Lazy dataset:** файлы грузятся по требованию (62 ГБ ≫ RAM).
- **Без ресэмплирования:** FFT напрямую по сырым данным (Fs ~ 38.4..40 кГц, per-file).
- **3I0 → IN, 3U0 → UN:** каналы нулевой последовательности.
- **Stride = 1/8 периода** (96..100 отсчётов в зависимости от Fs).
- **Split 80/20:** стратификация по типу дуги X (1..4).

Примеры::

    python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py --smoke
    python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py --max-files 100
    python scripts/phase4_experiments/sim_ozz/run_phase4_finetune_sim_ozz.py --resume experiments/phase4/sim_ozz_.../latest_checkpoint.pt
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
from tqdm.auto import tqdm

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем общие функции из оригинального finetune
from scripts.phase4_experiments.run_phase4_finetune import (
    COMPLEXITY_LEVELS,
    create_model_for_finetune,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    _reduce_logits_targets,
    _compute_supervision_pos_weight,
    _save_confusion_matrix,
    _print_class_distribution,
)

from osc_tools.ml.augmented_dataset import (
    compute_num_channels,
    AugmentedSpectralDataset,
    SAMPLES_PER_PERIOD,
    FFT_WINDOW,
)
from osc_tools.ml.augmentation import (
    TimeSeriesAugmenter,
    NeutralChannelDropout,
    PhaseCurrentDropout,
    CompositeAugmenter,
)
from osc_tools.ml.simulated_ozz_dataset import (
    SimOZZFileIndex,
    SimOZZLazyDataset,
    ARC_TYPES,
    TARGET_COLUMNS,
)
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from osc_tools.data_management.no_ozz_filter import get_no_ozz_files, filter_no_ozz_dataframe
from osc_tools.ml.labels import get_target_columns, clean_labels
from osc_tools.ml.balanced_dataset import BalancedConcatDataset, BalancedEpochSampler
from osc_tools.ml.precomputed_dataset import PrecomputedDataset


# ---------------------------------------------------------------------------
# Конфигурация для Simulated_OZZ_v1
# ---------------------------------------------------------------------------

NUM_SIM_OZZ_CLASSES = 4   # Stable, Petersen, PetersSlepian, Beliakov


def _sim_ozz_worker_init(worker_id: int) -> None:
    """Инициализация воркера DataLoader.

    Уменьшает LRU-кэш SimOZZLazyDataset в каждом воркере,
    чтобы суммарное потребление RAM не взрывалось
    (N воркеров × cache_size → N × cache_size/N = исходный cache_size).
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    n_workers = info.num_workers
    ds = info.dataset
    # Обходим вложенные обёртки (BalancedConcatDataset, ConcatDataset)
    datasets = ds.datasets if hasattr(ds, 'datasets') else [ds]
    for sub_ds in datasets:
        cache = getattr(sub_ds, '_cache', None)
        if cache is not None and hasattr(cache, '_maxsize'):
            cache._maxsize = max(10, cache._maxsize // n_workers)


# ---------------------------------------------------------------------------
# Быстрая оценка статистики (без прохода по DataLoader)
# ---------------------------------------------------------------------------

def _print_sim_ozz_class_stats(config: dict, target_columns: list[str]) -> None:
    """Печатает статистику классов из метаданных (без итерации по DataLoader).

    Для lazy SimOZZ dataset полный проход по DataLoader занял бы часы,
    поэтому считаем из split / class_ids.
    """
    split_data = config.get('_split_data')
    train_files = split_data['train_files'] if split_data else []
    n_train_sim = len(train_files)

    # Считаем файлы по типу дуги из имён файлов
    from collections import Counter
    arc_counts = Counter()
    import re
    for fname in train_files:
        m = re.match(r'^OZZ_(\d+)_', fname, re.IGNORECASE)
        if m:
            arc_counts[int(m.group(1))] += 1

    print(f"\n  Статистика классов (из метаданных, быстрая оценка):")
    print(f"    SimOZZ train файлов: {n_train_sim:,}")
    for x_type in sorted(arc_counts.keys()):
        name = ARC_TYPES[x_type] if x_type <= len(ARC_TYPES) else f"X={x_type}"
        print(f"      X={x_type} ({name}): {arc_counts[x_type]:,} файлов")

    if config.get('use_real_no_ozz'):
        print(f"    + реальные «не ОЗЗ» файлов (Balanced sampler уравняет)")
    print()


def _estimate_pos_weight_from_metadata(
    config: dict,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Оценивает pos_weight без итерации по DataLoader.

    В SimOZZ каждый файл содержит нормальное начало + ОЗЗ участок.
    Из разметки файлов ~40-60% зон содержат ОЗЗ (зависит от момента
    замыкания T). Для balanced dataset с 5 классами (4 ОЗЗ + 1 Normal)
    каждый класс видит одинаковое число элементов, поэтому pos_weight
    должен отражать баланс внутри зон.

    Эмпирическая оценка (из ~19k файлов):
    - Каждый файл ~72 зоны, из них ~30-40 зон с ОЗЗ (в зависимости от T)
    - С balanced sampler: каждый класс = 1/5 батча
    - Внутри SimOZZ элемента: ~50% зон с ОЗЗ
    - Для Normal (реальные): 100% зон с target [0,0,0,0]

    pos_weight = neg_count / pos_count ≈ (0.8*72 + 0.2*72) / (0.8*0.5*72/4)
    Упрощённо: ~3-5 для каждого класса.
    Стартуем с pos_weight=4.0 (будет пересчитан после 1-й эпохи если нужно).
    """
    n_classes = config.get('num_classes', NUM_SIM_OZZ_CLASSES)
    # Консервативная оценка: ~25% зон в среднем относятся к данному классу
    # (из выборки sampler: 4/5 — SimOZZ по ~50% ОЗЗ зон / 4 класса + 1/5 Normal)
    # pos_rate ≈ (4/5 * 0.5 / 4) = 0.1 → pos_weight ≈ 9
    # Но с учётом нормализации и balanced: ~4.0 хорошая стартовая оценка
    pw = torch.full((n_classes,), 4.0, dtype=torch.float32)
    if device is not None:
        pw = pw.to(device)
    return pw


def get_sim_ozz_config() -> dict:
    """Конфигурация fine-tuning для Simulated_OZZ_v1."""
    return {
        # Данные
        'data_dir': str(PROJECT_ROOT / 'data' / 'Simulated_OZZ_v1'),
        'target_level': 'sim_ozz',
        'num_harmonics': 9,
        'sub_periods': [2, 4, 6, 10],
        'include_symmetric': True,
        'stride_fraction': 8,               # 1/8 периода (96..100 отсчётов, per-file)
        'num_periods_window': 10,           # 10 периодов ≈ 7690 отсчётов
        'use_augmentation': True,           # Аугментация с dropout IN/UN
        'always_drop_neutral': False,       # True = ВСЕГДА обнулять IN+UN (для доучивания без 0-ых)
        'zone_target_aggregation': 'max',
        'val_split': 0.2,                   # 80/20
        'train_batches_per_epoch': 64,
        'batch_size': 32,
        'val_batch_size': 64,
        'num_workers': 8,                   # параллельная загрузка + FFT в фоне
        'prefetch_factor': 2,               # кол-во батчей на воркер в очереди
        'cache_size': 500,                  # LRU-кэш файлов (делится между воркерами)

        # Модель
        'model_type': 'PhysicalKANTransformer',
        'num_input_channels': 220,          # 8 × (9+4) × 2 + 6 × 2
        'd_model': 48,
        'num_heads': 4,
        'num_layers': 6,
        'kan_grid_size': 5,
        'dropout': 0.1,
        'cls_head_type': 'kan',
        'use_angle_gate': True,
        'use_mixed_layer_norm': False,
        'max_seq_len': 128,                 # >= 72 зон = (num_periods_window-1)*stride_fraction

        # Fine-tuning
        'num_classes': NUM_SIM_OZZ_CLASSES,
        'zone_size': 1,
        'supervision_mode': 'zone',
        'num_future_zones': 0,

        # Обучение
        'epochs': 50,
        'lr_backbone': 1e-3,               # Нет SSL → единый LR
        'lr_head': 1e-3,
        'weight_decay': 1e-5,
        'warmup_epochs': 2,
        'scheduler': 'cosine_warm_restarts',
        'scheduler_T0': 50,
        'scheduler_T_mult': 2,
        'scheduler_eta_min': 1e-7,
        'use_amp': True,
        'grad_clip': 1.0,
        'accumulation_steps': 8,

        # Сохранение
        'save_dir': str(PROJECT_ROOT / 'experiments' / 'phase4'),
        'checkpoint_frequency': 5,
        'seed': 42,

        # --- Реальные осциллограммы (класс «не ОЗЗ») ---
        'real_csv_path': str(PROJECT_ROOT / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'),
        'norm_coef_path': str(PROJECT_ROOT / 'data' / 'ml_datasets' / 'norm_coef_all_v1.4.csv'),
        'use_real_no_ozz': True,             # Добавить 5-й класс из реальных данных
        'real_window_size': 320,             # 10 периодов × 32 spp @ 1600 Гц
    }


# ---------------------------------------------------------------------------
# Подготовка реальных «не ОЗЗ» осциллограмм
# ---------------------------------------------------------------------------

def _standardize_voltage_columns(df: pl.DataFrame) -> pl.DataFrame:
    """UA BB -> UA, UB BB -> UB и т.д. Приоритет: BB > CL."""
    rename_map = {}
    for target in ['UA', 'UB', 'UC', 'UN']:
        if target not in df.columns:
            for src in [f'{target} BB', f'{target} CL']:
                if src in df.columns:
                    rename_map[src] = target
                    break
    if rename_map:
        df = df.rename(rename_map)
    return df


def prepare_real_no_ozz_dataset(
    config: dict,
    split: str = 'train',
    val_split: float = 0.2,
    seed: int = 42,
) -> Optional[AugmentedSpectralDataset]:
    """Подготавливает Dataset реальных осциллограмм без ОЗЗ.

    Формат выхода: ``(C, T_zones) = (220, 72)`` — совместимо с SimOZZ,
    если stride_fraction=8 (stride=4 при spp=32 → (320-32)/4 = 72 зоны).

    Returns:
        AugmentedSpectralDataset или None (если данных нет / отключено).
    """
    if not config.get('use_real_no_ozz', False):
        return None

    csv_path = Path(config['real_csv_path'])
    if not csv_path.exists():
        print(f"  [!] Реальный CSV не найден: {csv_path}")
        return None

    print(f"\n--- Подготовка реальных «не ОЗЗ» данных ---")
    print(f"  CSV: {csv_path}")

    df = pl.read_csv(str(csv_path), infer_schema_length=5000)
    df = _standardize_voltage_columns(df)
    df = clean_labels(df)

    # Фильтруем файлы без ОЗЗ
    no_ozz_files, stats = get_no_ozz_files(df, verbose=True)
    if not no_ozz_files:
        print("  [!] Нет файлов без ОЗЗ!")
        return None

    # Split на train/val по файлам
    rng = np.random.default_rng(seed)
    rng.shuffle(no_ozz_files)
    n_val = max(1, int(len(no_ozz_files) * val_split))
    if split == 'train':
        selected_files = no_ozz_files[n_val:]
    else:
        selected_files = no_ozz_files[:n_val]

    print(f"  {split}: {len(selected_files)} файлов из {len(no_ozz_files)} no-OZZ")

    df_split = df.filter(pl.col('file_name').is_in(selected_files))

    # Убеждаемся, что каналы IN/UN числовые (могут быть String)
    for ch in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
        if ch in df_split.columns:
            df_split = df_split.with_columns(
                pl.col(ch).cast(pl.Float32, strict=False).fill_null(0.0).alias(ch)
            )

    # Физическая нормализация (TODO: подключить после определения номиналов)
    # Пока без нормализации — нормализацию добавим после получения
    # результатов scan_sim_ozz_statistics.py

    # Индексы и boundaries
    file_boundaries = AugmentedSpectralDataset.compute_file_boundaries(df_split)
    window_size = config['real_window_size']  # 320
    stride = SAMPLES_PER_PERIOD // config['stride_fraction']  # 32 // 8 = 4

    indices = PrecomputedDataset.create_indices(
        df_split, window_size=window_size, mode='val', stride=stride,
    )
    print(f"  Строк: {df_split.height:,}, окон: {len(indices):,}")

    # Создаём Dataset с пустыми target_columns → все метки = 0 (класс «не ОЗЗ»)
    # Используем те же target_columns что и SimOZZ (4 столбца),
    # но все они = 0 для этих осциллограмм.
    target_columns = get_target_columns(config['target_level'])

    # Добавляем нулевые target-колонки (их нет в реальном CSV)
    for tc in target_columns:
        if tc not in df_split.columns:
            df_split = df_split.with_columns(
                pl.lit(0, dtype=pl.Int8).alias(tc)
            )

    ds = AugmentedSpectralDataset(
        dataframe=df_split,
        file_boundaries=file_boundaries,
        indices=indices,
        window_size=window_size,
        num_harmonics=config['num_harmonics'],
        sub_periods=config.get('sub_periods'),
        include_symmetric=config.get('include_symmetric', True),
        downsampling_stride=stride,
        future_zones=0,
        mask_ratio=0.0,
        augmenter=None,
        target_columns=target_columns,
        mode='classify',
        target_window_mode='any_in_window',
        zone_target_aggregation=config['zone_target_aggregation'],
    )

    print(f"  AugmentedSpectralDataset: {len(ds):,} элементов")
    # Проверка формы
    if len(ds) > 0:
        X, Y = ds[0]
        print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    return ds


# ---------------------------------------------------------------------------
# Общие kwargs для DataLoader (multi-worker + prefetch)
# ---------------------------------------------------------------------------

def _loader_kwargs(config: dict, is_train: bool = True) -> dict:
    """Возвращает общие kwargs для DataLoader с учётом num_workers.

    Если num_workers > 0: persistent_workers=True (кэш живёт между эпохами),
    worker_init_fn уменьшает LRU-кэш, prefetch_factor задаёт глубину очереди.
    """
    nw = config.get('num_workers', 0)
    kwargs: dict = {
        'num_workers': nw,
        'pin_memory': True,
        'drop_last': is_train,
    }
    if nw > 0:
        kwargs['persistent_workers'] = True
        kwargs['worker_init_fn'] = _sim_ozz_worker_init
        pf = config.get('prefetch_factor', 2)
        if pf is not None:
            kwargs['prefetch_factor'] = pf
    return kwargs


# ---------------------------------------------------------------------------
# Подготовка данных (lazy)
# ---------------------------------------------------------------------------

def prepare_sim_ozz_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Создаёт train/val DataLoader-ы для Simulated_OZZ_v1 (lazy loading).

    Returns:
        (train_loader, val_loader, target_columns)
    """
    data_dir = Path(config['data_dir'])
    max_files = config.get('max_files')

    nw = config.get('num_workers', 0)
    if nw > 0:
        pf = config.get('prefetch_factor', 2)
        print(f"DataLoader: {nw} воркеров, prefetch_factor={pf}, "
              f"persistent_workers=True")

    print(f"Сканирование файлов: {data_dir}")
    file_index = SimOZZFileIndex.from_directory(
        data_dir, max_files=max_files, verbose=True,
    )

    # Стратифицированный split
    all_names = [fi.path.name for fi in file_index.files]
    train_names, val_names, _, stats = stratified_sim_ozz_split(
        all_names,
        val_split=config['val_split'],
        seed=config['seed'],
    )
    print(f"  Split: train={len(train_names)}, val={len(val_names)}")
    for split_name, split_stats in stats.items():
        if any(v > 0 for v in split_stats.values()):
            print(f"    {split_name}: {split_stats}")

    # Сохраняем split
    config['_split_data'] = {
        'train_files': train_names,
        'val_files': val_names,
    }

    train_paths = [data_dir / n for n in train_names]
    val_paths = [data_dir / n for n in val_names]

    target_columns = get_target_columns(config['target_level'])
    config['num_classes'] = len(target_columns)

    # Пересчёт num_input_channels
    num_lh = len(config.get('sub_periods', [2, 4, 6, 10]))
    include_sym = config.get('include_symmetric', True)
    actual_ch = compute_num_channels(config['num_harmonics'], num_lh, include_sym)
    if config.get('num_input_channels') != actual_ch:
        print(f"  [!] num_input_channels: {config['num_input_channels']} -> {actual_ch}")
        config['num_input_channels'] = actual_ch

    # Общие параметры dataset
    ds_kwargs = dict(
        file_index=file_index,
        num_periods_window=config['num_periods_window'],
        stride_fraction=config['stride_fraction'],
        num_harmonics=config['num_harmonics'],
        sub_periods=config.get('sub_periods'),
        include_symmetric=include_sym,
        target_level=config['target_level'],
        mode='classify',
        zone_target_aggregation=config['zone_target_aggregation'],
        cache_size=config.get('cache_size', 500),
    )

    # Аугментация: inversion + phase shuffle + дропаут IN/UN
    train_augmenter = None
    val_augmenter = None
    always_drop = config.get('always_drop_neutral', False)

    if config.get('use_augmentation', True):
        base_aug = TimeSeriesAugmenter({
            'p_inversion': 0.5,
            'p_scaling': 0.2,
            'p_noise': 0.0,
            'p_phase_shuffling': 0.33,
            'p_drop_channel': 0.0,   # дропаут фазных отключен (есть NeutralChannelDropout)
        })
        # Дропаут одной фазы тока (имитация отсутствующего ТТ)
        phase_dropout = PhaseCurrentDropout(
            fill_value=0.0,
            p_all=0.45,
            p_drop_b=0.45,
            p_drop_a=0.05,
            p_drop_c=0.05,
        )
        if always_drop:
            # Режим «без нулевых»: IN+UN обнуляются ВСЕГДА (и train, и val)
            neutral_dropout = NeutralChannelDropout(
                fill_value=0.0,
                p_all=0.0,
                p_drop_in=0.0,
                p_drop_in_un=1.0,
            )
            train_augmenter = CompositeAugmenter(base_aug, phase_dropout, neutral_dropout)
            # Для val: только маскирование IN/UN, без прочих аугментаций
            val_augmenter = NeutralChannelDropout(
                fill_value=0.0,
                p_all=0.0,
                p_drop_in=0.0,
                p_drop_in_un=1.0,
            )
            print(f"  Аугментация: inversion + scaling + phase_shuffle + "
                  f"PhaseCurrentDropout(45%B/5%A/5%C) + "
                  f"NeutralDropout(ALWAYS drop IN+UN)")
        else:
            neutral_dropout = NeutralChannelDropout(
                fill_value=0.0,
                p_all=1.0 / 3,
                p_drop_in=1.0 / 3,
                p_drop_in_un=1.0 / 3,
            )
            train_augmenter = CompositeAugmenter(base_aug, phase_dropout, neutral_dropout)
            print(f"  Аугментация: inversion + scaling + phase_shuffle + "
                  f"PhaseCurrentDropout(45%B/5%A/5%C) + "
                  f"NeutralDropout(1/3 all, 1/3 -IN, 1/3 -IN-UN)")
    elif always_drop:
        # Без аугментации, но с маскированием IN/UN
        train_augmenter = NeutralChannelDropout(
            fill_value=0.0, p_all=0.0, p_drop_in=0.0, p_drop_in_un=1.0,
        )
        val_augmenter = train_augmenter
        print(f"  Аугментация: ТОЛЬКО NeutralDropout(ALWAYS drop IN+UN)")

    train_ds = SimOZZLazyDataset(
        file_paths=train_paths,
        augmenter=train_augmenter,
        verbose=True,
        **ds_kwargs,
    )

    val_ds = SimOZZLazyDataset(
        file_paths=val_paths,
        augmenter=val_augmenter,
        verbose=True,
        **ds_kwargs,
    )

    # supervision_mode — устанавливаем на dataset
    for ds in (train_ds, val_ds):
        ds.supervision_mode = config.get('supervision_mode', 'zone')

    # ---------------------------------------------------------------
    # 5-й класс: реальные осциллограммы без ОЗЗ
    # ---------------------------------------------------------------
    real_train_ds = prepare_real_no_ozz_dataset(
        config, split='train', val_split=config['val_split'], seed=config['seed'],
    )
    real_val_ds = prepare_real_no_ozz_dataset(
        config, split='val', val_split=config['val_split'], seed=config['seed'],
    )

    if real_train_ds is not None and len(real_train_ds) > 0:
        # --- Комбинированный train dataset ---
        # Распределяем class_ids: для SimOZZ каждое окно получает class_id
        # по типу дуги X файла (0=Stable, 1Petersen=, 2=PetersSlepian, 3=Beliakov).
        # Для real → class_id = 4 (Normal / не ОЗЗ).
        sim_class_ids = []
        for path, _ in train_ds._indices:
            fi = train_ds._file_infos[path.name]
            sim_class_ids.append(fi.meta['x'] - 1)  # X=1..4 -> 0..3

        real_class_ids = [4] * len(real_train_ds)

        combined_class_ids = sim_class_ids + real_class_ids
        combined_train = BalancedConcatDataset(
            datasets=[train_ds, real_train_ds],
            class_ids=combined_class_ids,
            class_names=list(ARC_TYPES) + ['Normal_real'],
        )
        combined_train.supervision_mode = config.get('supervision_mode', 'zone')
        combined_train.print_stats()

        # Balanced sampler
        # Ограничиваем размер «эпохи» через train_batches_per_epoch,
        # иначе полный проход по ~580k lazy-элементам займёт часы.
        batches_per_epoch = config.get('train_batches_per_epoch', 64)
        batch_size = config['batch_size']
        n_classes = len(combined_train.indices_by_class)
        # samples_per_class: из заданного лимита батчей
        max_samples_per_class = max(1, (batches_per_epoch * batch_size) // n_classes)
        # Явно заданный samples_per_class имеет приоритет
        spc = config.get('samples_per_class') or max_samples_per_class
        spc = min(spc, max_samples_per_class)

        train_sampler = BalancedEpochSampler(
            combined_train,
            samples_per_class=spc,
            seed=config['seed'],
        )
        print(f"  BalancedEpochSampler: {len(train_sampler):,} элементов/эпоха "
              f"({spc:,} на класс, {n_classes} классов, "
              f"~{len(train_sampler) // batch_size} батчей)")

        train_loader = DataLoader(
            combined_train,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            **_loader_kwargs(config, is_train=True),
        )
        config['_train_sampler'] = train_sampler  # для set_epoch()
    else:
        print("\n  [!] Реальные данные не загружены — обучение только на SimOZZ")
        # Ограничиваем число элементов в эпохе (lazy dataset медленный)
        from torch.utils.data import SubsetRandomSampler as _SRS
        batches_per_epoch = config.get('train_batches_per_epoch', 64)
        train_n = min(len(train_ds), batches_per_epoch * config['batch_size'])
        train_indices = np.random.default_rng(config['seed']).choice(
            len(train_ds), size=train_n, replace=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            sampler=_SRS(train_indices.tolist()),
            **_loader_kwargs(config, is_train=True),
        )

    # Val — ограниченная подвыборка (lazy-dataset тоже медленный для val)
    val_batches = config.get('val_batches_per_epoch', 32)
    if real_val_ds is not None and len(real_val_ds) > 0:
        from torch.utils.data import ConcatDataset as TorchConcatDataset, SubsetRandomSampler
        combined_val = TorchConcatDataset([val_ds, real_val_ds])
        combined_val.supervision_mode = config.get('supervision_mode', 'zone')
        val_n = min(len(combined_val), val_batches * config['val_batch_size'])
        val_indices = np.random.default_rng(config['seed']).choice(
            len(combined_val), size=val_n, replace=False,
        )
        val_loader = DataLoader(
            combined_val,
            batch_size=config['val_batch_size'],
            sampler=SubsetRandomSampler(val_indices.tolist()),
            **_loader_kwargs(config, is_train=False),
        )
        print(f"  Val: {val_n:,} элементов (~{val_batches} батчей) "
              f"из {len(combined_val):,} (SimOZZ + real)")
    else:
        from torch.utils.data import SubsetRandomSampler
        val_n = min(len(val_ds), val_batches * config['val_batch_size'])
        val_indices = np.random.default_rng(config['seed']).choice(
            len(val_ds), size=val_n, replace=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config['val_batch_size'],
            sampler=SubsetRandomSampler(val_indices.tolist()),
            **_loader_kwargs(config, is_train=False),
        )
        print(f"  Val: {val_n:,} элементов (~{val_batches} батчей) "
              f"из {len(val_ds):,}")

    return train_loader, val_loader, target_columns


# ---------------------------------------------------------------------------
# Основной цикл fine-tuning (адаптирован из run_phase4_finetune)
# ---------------------------------------------------------------------------

def finetune_sim_ozz(
    config: dict,
    resume_path: str | None = None,
    reset_optimizer: bool = False,
) -> Path:
    """Fine-tuning на Simulated_OZZ_v1."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = config.get('run_name')
    if run_name:
        dir_name = run_name
    else:
        dir_name = f"sim_ozz_finetune_{config['model_type']}_{timestamp}"
    save_dir = Path(config['save_dir']) / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SIM_OZZ FINE-TUNING — {config['model_type']}")
    if resume_path:
        print(f"ПРОДОЛЖЕНИЕ: {resume_path}")
    else:
        print("Random init (без SSL)")
    print(f"Сохранение: {save_dir}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Данные ---
    train_loader, val_loader, target_columns = prepare_sim_ozz_dataloaders(config)

    # Сохраняем конфиг и split
    split_data = config.pop('_split_data', None)
    # Убираем non-serializable перед сохранением
    config_save = {k: v for k, v in config.items() if not k.startswith('_')}
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    if split_data:
        with open(save_dir / 'split.json', 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Split: {save_dir / 'split.json'}")

    # Статистика классов (из метаданных, без прохода по DataLoader)
    _print_sim_ozz_class_stats(config, target_columns)

    # --- Модель (random init) ---
    model, _ = create_model_for_finetune(config, ssl_checkpoint_path=None)
    model = model.to(device)
    n_params = model.num_parameters()
    print(f"Параметры: {n_params:,}")

    supervision_mode = config.get('supervision_mode', 'zone')

    # --- Loss ---
    # Для lazy dataset считаем pos_weight из метаданных (не итерируя DataLoader).
    # В SimOZZ ~половина зон каждого файла — до начала ОЗЗ (target=0),
    # остальная — одна из 4 классов. Эмпирическая оценка: ~40% зон с ОЗЗ.
    pos_weight = _estimate_pos_weight_from_metadata(config, device=device)
    print(f"pos_weight ({supervision_mode}): {pos_weight.cpu().numpy()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer (единый LR, нет SSL) ---
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

    print(f"Backbone: {sum(p.numel() for p in backbone_params):,} params "
          f"(lr={config['lr_backbone']})")
    print(f"Head: {sum(p.numel() for p in head_params):,} params "
          f"(lr={config['lr_head']})")

    # --- LR scheduler ---
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config['epochs']
    eta_min = config.get('scheduler_eta_min', 1e-7)
    T_0 = config.get('scheduler_T0', 50)
    T_mult = config.get('scheduler_T_mult', 2)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, T_0), T_mult=T_mult, eta_min=eta_min,
    )
    print(f"Scheduler: CosineWarmRestarts T_0={T_0}, T_mult={T_mult}")

    # --- AMP ---
    scaler = None
    if config['use_amp'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("AMP: ON")

    # --- Resume ---
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float('inf')

    if resume_path is not None:
        rp = Path(resume_path)
        if rp.exists():
            print(f"Восстановление: {rp}")
            meta = load_checkpoint(
                rp, model, optimizer, cosine_scheduler, scaler,
                reset_optimizer=reset_optimizer,
            )
            start_epoch = meta['epoch'] + 1
            best_val_f1 = meta['best_val_f1']
            best_val_loss = meta.get('best_val_loss', float('inf'))
            if not reset_optimizer:
                print(f"  epoch={start_epoch}, best_f1={best_val_f1:.4f}")
        else:
            print(f"  [!] Чекпоинт не найден: {rp}")

    # --- Training loop ---
    history = []
    log_path = save_dir / 'training_log.jsonl'
    accum = config.get('accumulation_steps', 1)

    print(f"\nОбучение: {total_epochs} эпох, batch={config['batch_size']}, "
          f"accum={accum} (eff={config['batch_size'] * accum})")
    print(f"Классы: {target_columns}")
    print("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        # Обновляем seed сэмплера для текущей эпохи
        sampler = config.get('_train_sampler')
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Warmup
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]['lr'] = config['lr_backbone'] * frac
            optimizer.param_groups[1]['lr'] = config['lr_head'] * frac

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, config.get('grad_clip', 1.0),
            accumulation_steps=accum,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        val_metrics = validate(
            model, val_loader, loss_fn, device,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        is_best = val_metrics['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['macro_f1']

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
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'train_time': train_metrics['time_sec'],
            'val_time': val_metrics['time_sec'],
            'vram_mb': vram_mb,
            'is_best': is_best,
            'is_best_loss': is_best_loss,
        }
        for i, col in enumerate(target_columns):
            record[f'val_f1_{col}'] = val_metrics.get(f'f1_class_{i}', 0.0)

        history.append(record)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        marker = ' ★' if is_best else (' ★L' if is_best_loss else '')
        lr_bb = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:3d}/{total_epochs} | "
            f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"F1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}{marker} | "
            f"AUC={val_metrics.get('roc_auc', 0):.4f} | "
            f"lr={lr_bb:.1e} | "
            f"time={train_metrics['time_sec'] + val_metrics['time_sec']:.1f}s"
        )
        cls_f1s = [f"{val_metrics.get(f'f1_class_{i}', 0):.3f}"
                   for i in range(len(target_columns))]
        print(f"         Per-class F1: {cls_f1s}")

        # Checkpointing
        save_checkpoint(
            save_dir / 'latest_checkpoint.pt',
            model, optimizer, cosine_scheduler, scaler,
            epoch, best_val_f1, config_save, best_val_loss,
        )
        if is_best:
            save_checkpoint(
                save_dir / 'best_model.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )
        if is_best_loss:
            save_checkpoint(
                save_dir / 'best_model_loss.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint(
                save_dir / f'checkpoint_{epoch:03d}.pt',
                model, optimizer, cosine_scheduler, scaler,
                epoch, best_val_f1, config_save, best_val_loss,
            )

    # --- Итоги ---
    print("\n" + "=" * 60)
    print("SIM_OZZ FINE-TUNING ЗАВЕРШЁН")
    print(f"  Best val F1: {best_val_f1:.4f}")
    if history:
        best_rec = max(history, key=lambda r: r['val_f1'])
        best_ep = best_rec['epoch']
        print(f"  Best epoch: {best_ep}")
        for col in target_columns:
            print(f"    {col}: {best_rec.get(f'val_f1_{col}', 0):.4f}")
    print(f"  Результаты: {save_dir}")
    print("=" * 60)

    # Confusion matrix
    _save_confusion_matrix(
        model, val_loader, loss_fn, device, target_columns,
        save_dir / 'confusion_matrix.txt',
    )

    # Summary
    summary = {
        'model_type': config['model_type'],
        'best_epoch': best_ep if history else -1,
        'best_val_f1': best_val_f1,
        'best_val_roc_auc': best_rec.get('val_roc_auc', 0) if history else 0,
        'total_epochs': total_epochs,
        'num_params': n_params,
        'target_columns': target_columns,
        'dataset': 'Simulated_OZZ_v1',
    }
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    if history:
        import csv
        csv_path = save_dir / 'training_curves.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"  Кривые: {csv_path}")

    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Phase 4.5: Simulated_OZZ Fine-tuning')
    p.add_argument('--model', choices=['PhysicalKANTransformer', 'PhysicalMLPTransformer', 'BaselineTransformer'],
                   default='PhysicalKANTransformer')
    p.add_argument('--complexity', choices=['light', 'medium', 'heavy'], default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--max-files', type=int, default=None,
                   help='Ограничить число файлов (для отладки)')
    p.add_argument('--accumulation-steps', type=int, default=None)
    p.add_argument('--smoke', action='store_true', help='Smoke-test (2 эпохи, 20 файлов)')
    p.add_argument('--resume', type=str, default=None,
                   help='Путь к чекпоинту для продолжения')
    p.add_argument('--reset-optimizer', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = get_sim_ozz_config()

    config['model_type'] = args.model
    if args.model == 'BaselineTransformer':
        config['cls_head_type'] = 'linear'
    elif args.model == 'PhysicalMLPTransformer':
        config['cls_head_type'] = 'mlp'
    if args.complexity:
        config.update(COMPLEXITY_LEVELS[args.complexity])
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.accumulation_steps is not None:
        config['accumulation_steps'] = args.accumulation_steps
    if args.max_files is not None:
        config['max_files'] = args.max_files

    if args.smoke:
        config['epochs'] = 2
        config['batch_size'] = 4
        config['val_batch_size'] = 4
        config['accumulation_steps'] = 1
        config['max_files'] = 20
        config['cache_size'] = 50
        config['include_symmetric'] = False
        num_lh = len(config['sub_periods'])
        config['num_input_channels'] = compute_num_channels(
            config['num_harmonics'], num_lh, False,
        )

    finetune_sim_ozz(
        config,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
    )


if __name__ == '__main__':
    # =================================================================
    # РУЧНОЙ ЗАПУСК ИЗ IDE (VS Code, PyCharm и т.д.)
    # =================================================================
    # Если хотите использовать аргументы командной строки, раскомментируйте:
    # main()
    # sys.exit(0)

    # 1. Выбор архитектуры
    # Доступно: 'PhysicalKANTransformer', 'PhysicalMLPTransformer', 'BaselineTransformer'
    MODEL_TYPE = 'PhysicalKANTransformer'

    # 2. Размер модели (влияет на кол-во параметров и потребление видеопамяти VRAM)
    # Доступно: 'light' (~100k), 'medium' (~500k), 'heavy' (~2M+)
    SELECTED_COMPLEXITY = 'light'

    # 3. Основные гиперпараметры обучения
    EPOCHS = 50                 # Общее количество эпох
    BATCH_SIZE = 32             # Размер батча на 1 шаг (уменьшите, если не хватает VRAM)
    ACCUMULATION_STEPS = 8      # Накопление градиентов (эфф. батч = 32 * 8 = 256)

    # 4. Ограничения данных (для тестов и отладки)
    MAX_FILES = None            # int или None. Пример: 500 (чтобы проверить на малой выборке)
    USE_REAL_NO_OZZ = True      # True = добавлять 5-й класс "Норма" из реальных осциллограмм (~959 файлов)

    # 4a. Режим «без нулевых последовательностей»
    # True = ВСЕГДА обнулять IN и UN (и при обучении, и при валидации).
    # Используйте для доучивания модели, которая будет работать только по фазным каналам.
    # На inference (inference_real_ozz.py) этот режим включён по умолчанию через mask_neutral=True.
    ALWAYS_DROP_NEUTRAL = False

    # 5. Размер «эпохи» (lazy dataset — полный проход занял бы часы!)
    # train_batches_per_epoch × batch_size элементов делится поровну на 5 классов.
    # 256 батчей × 32 = 8192 элементов, ~1638 на класс → ~13 мин/эпоха (зависит от диска)
    TRAIN_BATCHES_PER_EPOCH = 256   # Число батчей обучения за эпоху
    VAL_BATCHES_PER_EPOCH = 16      # Число батчей валидации за эпоху

    # 6. Продолжение прерванного обучения
    # Пример: RESUME_PATH = str(PROJECT_ROOT / 'experiments/phase4/.../latest_checkpoint.pt')

    RESUME_PATH = 'experiments/phase4/sim_ozz_finetune_PhysicalKANTransformer_20260427_215035/latest_checkpoint.pt'
    RESET_OPTIMIZER = True # True, если нужно сбросить оптимизатор и начать с 0 эпохи

    # =================================================================

    config = get_sim_ozz_config()
    config['model_type'] = MODEL_TYPE
    if MODEL_TYPE == 'BaselineTransformer':
        config['cls_head_type'] = 'linear'
    elif MODEL_TYPE == 'PhysicalMLPTransformer':
        config['cls_head_type'] = 'mlp'
    config['epochs'] = EPOCHS
    config['batch_size'] = BATCH_SIZE
    config['accumulation_steps'] = ACCUMULATION_STEPS
    config['use_real_no_ozz'] = USE_REAL_NO_OZZ
    config['always_drop_neutral'] = ALWAYS_DROP_NEUTRAL
    config['train_batches_per_epoch'] = TRAIN_BATCHES_PER_EPOCH
    config['val_batches_per_epoch'] = VAL_BATCHES_PER_EPOCH
    if MAX_FILES is not None:
        config['max_files'] = MAX_FILES

    level = COMPLEXITY_LEVELS[SELECTED_COMPLEXITY]
    config.update(level)

    # Каналы
    num_lh = len(config['sub_periods'])
    config['num_input_channels'] = compute_num_channels(
        config['num_harmonics'], num_lh, config['include_symmetric'],
    )

    finetune_sim_ozz(
        config,
        resume_path=RESUME_PATH,
        reset_optimizer=RESET_OPTIMIZER,
    )
