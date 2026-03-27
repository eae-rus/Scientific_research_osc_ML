"""
Визуализация разметки осциллограмм моделью Фазы 4 (Physical KAN-Transformer).

Генерирует для каждого файла из val-сплита:
- Графики токов и напряжений
- Реальные метки vs предсказания модели (дискреты)
- Кривые уверенности (confidence) по каждому классу

Поддерживает три supervision_mode для визуализации:
  - zone:     зонная разметка (каждая зона = stride отсчётов → пообразцовое усреднение)
  - window:   оконная разметка (среднее по зонам = одна вероятность на окно)
  - last_zone: последняя зона окна = предсказание (аналогично зоне, но без усреднения)

Примеры:
  python scripts/phase4_experiments/plot_phase4_marking.py  # ручной режим
  python scripts/phase4_experiments/plot_phase4_marking.py --checkpoint path/to/best_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.ml.augmented_dataset import compute_spectral_from_raw, standardize_voltage_columns
from osc_tools.ml.labels import (
    get_target_columns, prepare_labels_for_experiment,
    clean_labels, add_base_labels,
)


# ---------------------------------------------------------------------------
# Константы фазовых цветов (стандарт РЗА)
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    'IA': '#FFD700', 'IB': '#228B22', 'IC': '#FF4500', 'IN': '#1E90FF',
    'UA': '#FFD700', 'UB': '#228B22', 'UC': '#FF4500', 'UN': '#1E90FF',
}

# Цвета для классов ОЗЗ
CLASS_COLORS = {
    'Target_OZZ': '#E74C3C',
    'Target_OZZ_decay': '#F39C12',
    'Target_OZZ_dpozz': '#8E44AD',
    'Target_Normal': '#27AE60',
    'Target_ML_1': '#E74C3C',
    'Target_ML_2': '#F39C12',
    'Target_ML_3': '#8E44AD',
}


def _get_class_color(name: str, idx: int) -> str:
    """Возвращает цвет для класса по имени."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return f"C{idx % 10}"


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Загружает модель из fine-tuning чекпоинта."""
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
            dropout=0.0,
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
            cls_head_type=config.get('cls_head_type', 'linear'),
            dropout=0.0,
            max_seq_len=64,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Пообразцовая разметка скользящим окном
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_oscillogram(
    model: torch.nn.Module,
    raw_data: np.ndarray,
    config: dict,
    device: torch.device,
    step: int = 32,
    max_batch_mem_gb: float = 6.0,
) -> np.ndarray:
    """Разметка осциллограммы скользящим окном с батчированным инференсом.

    Все окна сначала предрассчитываются (FFT), затем собираются в батчи
    и подаются на GPU разом — это на порядок быстрее поштучного прогона.

    Args:
        model: модель в eval mode
        raw_data: (N, 8) сырые отсчёты (IA, IB, IC, IN, UA, UB, UC, UN)
        config: конфиг модели
        device: torch device
        step: шаг скользящего окна в отсчётах (32 = 1 период при 1600 Гц)
        max_batch_mem_gb: лимит GPU-памяти для одного батча (по умолчанию 6 ГБ)

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
    fft_window = 32

    starts = list(range(0, max(1, N - window_size + 1), step))
    n_windows = len(starts)

    # --- 1. Предрасчёт спектральных признаков для всех окон ---
    spectral_list: list[np.ndarray] = []
    for win_start in starts:
        win_end = min(win_start + window_size, N)
        raw_win = raw_data[win_start:win_end].copy()

        if len(raw_win) < window_size:
            raw_win = np.concatenate([
                raw_win,
                np.zeros((window_size - len(raw_win), 8), dtype=np.float32),
            ], axis=0)

        spectral = compute_spectral_from_raw(
            raw_win, num_harmonics, sub_periods if sub_periods else None,
            include_symmetric, stride=stride, warmup=fft_window,
        )
        # spectral: (T_out, C)  →  (C, T_out) для модели
        spectral_list.append(spectral.T.copy())

    # --- 2. Оценка размера батча по памяти ---
    sample_shape = spectral_list[0].shape  # (C, T_out)
    bytes_per_sample = sample_shape[0] * sample_shape[1] * 4  # float32
    # Учитываем ~4× overhead (веса + промежуточные активации)
    max_batch_size = max(1, int(max_batch_mem_gb * 1e9 / (bytes_per_sample * 4)))
    max_batch_size = min(max_batch_size, n_windows)

    # --- 3. Батчированный инференс ---
    all_probs: list[np.ndarray] = []
    for batch_start in range(0, n_windows, max_batch_size):
        batch_end = min(batch_start + max_batch_size, n_windows)
        batch_np = np.stack(spectral_list[batch_start:batch_end], axis=0)  # (B, C, T_out)
        X = torch.from_numpy(batch_np).to(device)
        out = model(X, mode='classify')
        logits = out['classify']  # (B, n_zones, num_classes)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)

    all_probs_np = np.concatenate(all_probs, axis=0)  # (n_windows, n_zones, num_classes)

    # --- 4. Раскладка по пообразцовой сетке ---
    prob_sum = np.zeros((N, num_classes), dtype=np.float64)
    coverage = np.zeros(N, dtype=np.int32)

    for i, win_start in enumerate(starts):
        probs = all_probs_np[i]  # (n_zones, num_classes)
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
    result = np.zeros((N, num_classes), dtype=np.float32)
    result[mask_covered] = (prob_sum[mask_covered] / coverage[mask_covered, np.newaxis]).astype(np.float32)

    # Непокрытые точки — ближайшее предсказание
    if not mask_covered.all():
        first_covered = np.argmax(mask_covered)
        if first_covered > 0:
            result[:first_covered] = result[first_covered]

    return result


# ---------------------------------------------------------------------------
# Построение графика
# ---------------------------------------------------------------------------

def plot_marking(
    out_path: Path,
    time_axis: np.ndarray,
    currents: dict[str, np.ndarray],
    voltages: dict[str, np.ndarray],
    real_labels: dict[str, np.ndarray],
    pred_probs: dict[str, np.ndarray],
    title: str,
    threshold: float = 0.5,
) -> None:
    """Строит комбинированный график: токи, напряжения, дискреты, кривые уверенности.

    Args:
        out_path: путь для сохранения PNG
        time_axis: (N,) ось времени в мс
        currents: словарь {имя: массив} для токов
        voltages: словарь {имя: массив} для напряжений
        real_labels: словарь {класс: бинарный массив} реальных меток
        pred_probs: словарь {класс: массив вероятностей} предсказаний модели
        title: заголовок графика
        threshold: порог бинаризации
    """
    labels = list(real_labels.keys())
    n_classes = len(labels)
    amplitudes = np.arange(1, n_classes + 1)

    # Высоты подграфиков: токи, напряжения, дискреты, N × уверенность
    height_ratios = [1.1, 1.1, 0.8] + [0.6] * n_classes
    fig = plt.figure(figsize=(16, 10 + 1.2 * n_classes))
    gs = fig.add_gridspec(nrows=3 + n_classes, ncols=1, height_ratios=height_ratios)

    def get_phase_color(name: str, idx: int) -> str:
        name_uc = name.upper()
        for key, color in PHASE_COLORS.items():
            if key in name_uc:
                return color
        return f"C{idx % 10}"

    # --- Токи ---
    ax_curr = fig.add_subplot(gs[0, 0])
    for i, (name, data) in enumerate(currents.items()):
        ax_curr.plot(time_axis, data, label=name, color=get_phase_color(name, i), linewidth=1.2)
    ax_curr.set_ylabel("Токи")
    ax_curr.legend(loc='upper right', fontsize=8)
    ax_curr.grid(True, alpha=0.3, linestyle=':')

    # --- Напряжения ---
    ax_volt = fig.add_subplot(gs[1, 0], sharex=ax_curr)
    for i, (name, data) in enumerate(voltages.items()):
        ax_volt.plot(time_axis, data, label=name, color=get_phase_color(name, i), linewidth=1.2)
    ax_volt.set_ylabel("Напряжения")
    ax_volt.legend(loc='upper right', fontsize=8)
    ax_volt.grid(True, alpha=0.3, linestyle=':')

    # --- Дискреты (реальные сверху, предсказания снизу) ---
    ax_disc = fig.add_subplot(gs[2, 0], sharex=ax_curr)
    for i, label_name in enumerate(labels):
        color = _get_class_color(label_name, i)
        pred_bin = (pred_probs[label_name] >= threshold).astype(np.int8)

        # Реальные — наверх
        real_pos = np.where(real_labels[label_name] > 0, amplitudes[i], np.nan)
        ax_disc.scatter(time_axis, real_pos, marker='o', s=16, alpha=0.8,
                        color=color, label=f"GT: {label_name}")

        # Предсказания — вниз
        pred_pos = np.where(pred_bin > 0, -amplitudes[i], np.nan)
        ax_disc.scatter(time_axis, pred_pos, marker='s', s=14, alpha=0.6,
                        color=color, label=f"Pred: {label_name}")

    ax_disc.axhline(0, color='black', linewidth=1)
    ax_disc.set_ylim(-n_classes - 0.5, n_classes + 0.5)
    y_ticks = np.concatenate([-amplitudes[::-1], amplitudes])
    y_tick_labels = [f"P:{l}" for l in labels[::-1]] + [f"G:{l}" for l in labels]
    ax_disc.set_yticks(y_ticks)
    ax_disc.set_yticklabels(y_tick_labels, fontsize=7)
    ax_disc.set_ylabel("Дискреты (GT:+, Pred:-)")
    ax_disc.grid(True, alpha=0.3, linestyle=':')
    ax_disc.legend(loc='upper right', ncols=2, fontsize=7)

    # --- Кривые уверенности по каждому классу ---
    for i, label_name in enumerate(labels):
        ax_conf = fig.add_subplot(gs[3 + i, 0], sharex=ax_curr)
        color = _get_class_color(label_name, i)
        probs = pred_probs.get(label_name, np.zeros_like(time_axis))

        ax_conf.plot(time_axis, probs, color=color, linewidth=1.2, alpha=0.6)
        mask_above = probs >= threshold
        if np.any(mask_above):
            ax_conf.scatter(time_axis[mask_above], probs[mask_above],
                            color=color, s=8, alpha=0.9)

        # Полоса реальной метки (фон)
        real = real_labels[label_name]
        if np.any(real > 0):
            ax_conf.fill_between(time_axis, 0, 1,
                                 where=real > 0, alpha=0.1, color=color)

        ax_conf.axhline(threshold, color='red', linewidth=0.9,
                        linestyle='--', alpha=0.7)
        ax_conf.set_ylim(-0.02, 1.02)
        ax_conf.set_yticks([0.0, threshold, 1.0])
        ax_conf.set_yticklabels(["0", f"{threshold:.2f}", "1"], fontsize=7)
        ax_conf.set_ylabel(label_name, fontsize=8)
        ax_conf.grid(True, alpha=0.3, linestyle=':')

    # Нижняя ось
    ax_conf.set_xlabel("Время, мс")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Основная функция генерации графиков
# ---------------------------------------------------------------------------

def generate_marking_plots(
    checkpoint_path: str,
    output_dir: str,
    data_dir: str = 'data/ml_datasets',
    precomputed_file: str = 'test_precomputed.csv',
    split: str = 'val',
    step: int = 32,
    threshold: float = 0.5,
    include_zero_current: bool = True,
    include_zero_voltage: bool = True,
    max_files: Optional[int] = None,
    selected_files: Optional[list[str]] = None,
) -> None:
    """Генерирует графики разметки для всех файлов из указанного сплита.

    Args:
        checkpoint_path: путь к fine-tuning чекпоинту (best_model.pt)
        output_dir: папка для сохранения графиков
        data_dir: папка с датасетами
        precomputed_file: имя CSV-файла данных
        split: 'val' | 'train' | 'all'
        step: шаг скользящего окна (32 = 1 период)
        threshold: порог бинаризации
        include_zero_current: отображать ток нуля (IN)
        include_zero_voltage: отображать напряжение нуля (UN)
        max_files: ограничение на число файлов (None = все)
        selected_files: конкретные файлы для построения
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Загрузка модели
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    model, config = load_model(ckpt_path, device)
    model_type = config.get('model_type', 'N/A')
    target_level = config.get('target_level', 'base')
    supervision_mode = config.get('supervision_mode', 'zone')
    print(f"Модель: {model_type}, target_level={target_level}, supervision={supervision_mode}")

    # Данные
    data_path = Path(data_dir) / precomputed_file
    print(f"Загрузка данных: {data_path.name}")
    df = pl.read_csv(str(data_path), infer_schema_length=50000, null_values=["NA", "nan", "null", ""])

    # Стандартизация колонок напряжений (UA BB → UA и т.д.)
    df = standardize_voltage_columns(df)

    # Подготовка меток: если это raw CSV без Target_* — создаём их
    if 'Target_Normal' not in df.columns:
        df = clean_labels(df)
        df = add_base_labels(df)
        print("  Метки подготовлены из ML_* колонок")

    # Подготовка дополнительных меток
    if target_level in ('base_sequential', 'ozz', 'full', 'full_by_levels'):
        df = prepare_labels_for_experiment(df, target_level)

    target_columns = get_target_columns(target_level, df)
    print(f"Целевые классы: {target_columns}")

    # Split (тот же алгоритм, что при обучении)
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
        # Перебалансировка: если train пуст по классу, а val >= 2 — переносим
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
        train_files = [f for f in files if f not in set(val_files)]

    print(f"Всего уникальных файлов в датасете: {len(train_files) + len(val_files)}")
    print(f"  train: {len(train_files)}, val: {len(val_files)}")

    if split == 'val':
        plot_files = val_files
    elif split == 'train':
        plot_files = train_files
    else:
        plot_files = train_files + val_files

    if selected_files:
        selected_set = set(selected_files)
        plot_files = [f for f in plot_files if f in selected_set]

    if max_files is not None:
        plot_files = plot_files[:max_files]

    print(f"Файлов для визуализации ({split}): {len(plot_files)}")

    # Сигнальные колонки
    current_cols = ['IA', 'IB', 'IC']
    if include_zero_current:
        current_cols.append('IN')
    voltage_cols = ['UA', 'UB', 'UC']
    if include_zero_voltage:
        voltage_cols.append('UN')
    raw_channels = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']

    # Папка вывода
    out_base = Path(output_dir)
    exp_name = ckpt_path.parent.name
    out_dir = out_base / f"marking_{exp_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Выходная папка: {out_dir}")

    sampling_rate = 1600  # Частота дискретизации

    for file_idx, file_name in enumerate(plot_files):
        file_df = df.filter(pl.col('file_name') == file_name)
        N = file_df.height
        if N < config.get('window_size', 320) + 1:
            print(f"  Пропуск {file_name}: слишком мало отсчётов ({N})")
            continue

        print(f"  [{file_idx + 1}/{len(plot_files)}] {file_name} ({N} отсчётов)")

        # Сигналы
        available_cols = set(file_df.columns)
        currents = {}
        for c in current_cols:
            if c in available_cols:
                currents[c] = file_df[c].to_numpy().astype(np.float32)
        voltages = {}
        for c in voltage_cols:
            if c in available_cols:
                voltages[c] = file_df[c].to_numpy().astype(np.float32)

        # Реальные метки
        real_labels = {}
        for col in target_columns:
            if col in available_cols:
                real_labels[col] = file_df[col].to_numpy().astype(np.int8)
            else:
                real_labels[col] = np.zeros(N, dtype=np.int8)

        # Raw данные для FFT
        raw_cols_available = [c for c in raw_channels if c in available_cols]
        if len(raw_cols_available) < 8:
            print(f"    Пропуск: не все 8 каналов ({raw_cols_available})")
            continue
        raw_data = file_df.select(raw_channels).to_numpy().astype(np.float32)

        # Пообразцовая разметка скользящим окном
        probs = mark_oscillogram(model, raw_data, config, device, step=step)

        # Формируем словарь вероятностей
        pred_probs = {}
        for i, col in enumerate(target_columns):
            pred_probs[col] = probs[:, i] if i < probs.shape[1] else np.zeros(N, dtype=np.float32)

        # Ось времени
        time_axis = np.arange(N) * 1000.0 / sampling_rate

        # Имя файла для графика
        name_hash = hashlib.md5(f"{file_name}|{exp_name}".encode('utf-8')).hexdigest()[:12]
        out_name = f"mark_{name_hash}.png"
        out_path = out_dir / out_name

        title = f"{model_type} | {supervision_mode} | {file_name}"
        plot_marking(
            out_path=out_path,
            time_axis=time_axis,
            currents=currents,
            voltages=voltages,
            real_labels=real_labels,
            pred_probs=pred_probs,
            title=title,
            threshold=threshold,
        )

    print(f"\nГотово! Графики сохранены в: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 4: Визуализация разметки')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к fine-tuning чекпоинту (best_model.pt)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Папка для сохранения графиков')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'train', 'all'],
                        help='Какие файлы визуализировать')
    parser.add_argument('--step', type=int, default=32,
                        help='Шаг скользящего окна (32 = 1 период)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Порог бинаризации')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Максимум файлов для визуализации')
    parser.add_argument('--no-zero-current', action='store_true',
                        help='Не отображать ток нуля (IN)')
    parser.add_argument('--no-zero-voltage', action='store_true',
                        help='Не отображать напряжение нуля (UN)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        print("Используйте --checkpoint или ручной режим в __main__")
        return

    generate_marking_plots(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir or str(PROJECT_ROOT / 'reports' / 'phase4'),
        split=args.split,
        step=args.step,
        threshold=args.threshold,
        include_zero_current=not args.no_zero_current,
        include_zero_voltage=not args.no_zero_voltage,
        max_files=args.max_files,
    )


if __name__ == '__main__':
    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА ЧЕРЕЗ КОНСТАНТЫ
    # Для запуска: просто выполните файл (F5 или python plot_phase4_marking.py)
    # Для CLI: python plot_phase4_marking.py --checkpoint path/to/best_model.pt
    # =================================================================

    MANUAL_RUN = True

    if MANUAL_RUN:
        # --- Путь к чекпоинту finetune ---
        CHECKPOINT = 'experiments/phase4/base_finetune_PhysicalKANTransformer_20260321_154837/latest_checkpoint.pt'
        # CHECKPOINT = 'experiments/phase4/base_finetune_PhysicalKANTransformer_20260321_154837/best_model.pt'

        
        # --- Папка вывода ---
        OUTPUT_DIR = 'reports/phase4'

        # --- Данные ---
        # Можно указать любой CSV: labeled_2025_12_03.csv (все 990 файлов),
        # train.csv (792 файла), test_precomputed.csv (198 файлов)
        DATA_DIR = 'data/ml_datasets'
        PRECOMPUTED_FILE = 'labeled_2025_12_03.csv'

        # --- Split: 'val' | 'train' | 'all' ---
        SPLIT = 'train'

        # --- Параметры визуализации ---
        STEP = 1           # Шаг скользящего окна (32 = 1 период)
        THRESHOLD = 0.5     # Порог бинаризации
        INCLUDE_ZERO_CURRENT = True
        INCLUDE_ZERO_VOLTAGE = True

        # --- Ограничения ---
        MAX_FILES = None    # None = все файлы из сплита
        SELECTED_FILES = None  # ['some_file.cfg', ...] или None

        # =================================================================

        generate_marking_plots(
            checkpoint_path=str(PROJECT_ROOT / CHECKPOINT),
            output_dir=str(PROJECT_ROOT / OUTPUT_DIR),
            data_dir=str(PROJECT_ROOT / DATA_DIR),
            precomputed_file=PRECOMPUTED_FILE,
            split=SPLIT,
            step=STEP,
            threshold=THRESHOLD,
            include_zero_current=INCLUDE_ZERO_CURRENT,
            include_zero_voltage=INCLUDE_ZERO_VOLTAGE,
            max_files=MAX_FILES,
            selected_files=SELECTED_FILES,
        )
    else:
        main()
