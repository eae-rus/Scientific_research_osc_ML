import argparse
import re
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment
from osc_tools.analysis.ozz_physics import predict_ozz_physics, u0_threshold_raw_to_normalized

# Переиспользуем проверенные функции создания модели и загрузки весов
from scripts.evaluation._core.model_utils import _create_model_from_config, _load_state_dict_safe
from scripts.evaluation._core.config_resolvers import parse_experiment_info


def _find_experiment_dir(exp_name: str) -> Path:
    """Ищет директорию эксперимента по имени в папке experiments."""
    exp_root = ROOT_DIR / "experiments"
    if not exp_root.exists():
        raise FileNotFoundError(f"Не найдена папка experiments: {exp_root}")

    # Прямое совпадение
    direct = exp_root / exp_name
    if direct.exists() and direct.is_dir():
        return direct

    # Рекурсивный поиск
    candidates = [p for p in exp_root.rglob(exp_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Эксперимент не найден: {exp_name}")

    # Берём первый найденный
    return candidates[0]


def _resolve_target_level(exp_name: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Определяем target_level по config (приоритетно) или имени эксперимента."""
    cfg_level = None
    if config is not None:
        cfg_level = str(config.get('data', {}).get('target_level', '')).strip().lower()
    if cfg_level:
        return cfg_level

    name = exp_name.lower()
    if 'base_sequential' in name:
        return 'base_sequential'
    if 'ozz' in name or '2.6.11' in name:
        return 'ozz'
    if 'full_by_levels' in name or ('hier_' in name and '2.6.4' in name):
        return 'full_by_levels'
    if '2.6.4' in name and 'full' in name:
        return 'full'
    return 'base'


def _resolve_target_window_mode(config: Dict[str, Any], exp_name: str) -> str:
    """Определяем режим формирования метки по окну."""
    data_cfg = config.get('data', {})
    mode = data_cfg.get('target_window_mode')
    if isinstance(mode, str) and mode:
        return mode
    name = exp_name.lower()
    if 'win_any' in name or 'window_any' in name:
        return 'any_in_window'
    return 'point'


def _resolve_feature_mode(config: Dict[str, Any], exp_name: str) -> str:
    """Определяем feature_mode по конфигу или имени эксперимента."""
    data_features = config.get('data', {}).get('features')
    if isinstance(data_features, list) and data_features:
        return data_features[0]
    if isinstance(data_features, str) and data_features:
        return data_features

    name = exp_name.lower()
    if 'raw' in name:
        return 'raw'
    if 'symmetric_polar' in name:
        return 'symmetric_polar'
    if 'symmetric' in name:
        return 'symmetric'
    if 'phase_complex' in name:
        return 'phase_complex'
    if 'phase_polar' in name or 'polar' in name:
        return 'phase_polar'
    if 'alpha_beta' in name:
        return 'alpha_beta'
    if 'power' in name:
        return 'power'
    return 'phase_polar'


def _resolve_sampling_strategy(exp_name: str, input_size: int, in_channels: Optional[int], window_size: int) -> Tuple[str, int]:
    """Определяем sampling_strategy и stride по имени/конфигу."""
    name = exp_name.lower()
    if 'stride' in name:
        return 'stride', 16
    if 'snapshot' in name:
        return 'snapshot', 32
    if 'none_sampl' in name:
        return 'none', 1

    # Фолбэк по input_size
    if in_channels:
        seq_len = input_size // in_channels if in_channels else 0
    elif input_size > 0:
        if input_size <= 32:
            seq_len = 2
        else:
            seq_len = input_size // 16
    else:
        seq_len = 18

    if seq_len >= window_size:
        return 'none', 1
    if seq_len <= 4:
        return 'snapshot', 32
    return 'stride', 16


def _resolve_num_harmonics(
    feature_mode: str,
    sampling_strategy: str,
    downsampling_stride: int,
    window_size: int,
    in_channels: Optional[int],
    input_size: int
) -> Tuple[int, Optional[int]]:
    """Определяем число гармоник и корректный in_channels."""
    base_ch = 0
    if feature_mode in ['symmetric', 'symmetric_polar']:
        base_ch = 12
    elif feature_mode in ['phase_polar', 'phase_complex']:
        base_ch = 16
    elif feature_mode in ['raw', 'power']:
        base_ch = 8
    elif feature_mode == 'alpha_beta':
        base_ch = 6

    num_harmonics = 1
    derived_in_channels = None

    if input_size > 0 and base_ch > 0:
        if sampling_strategy == 'snapshot':
            pts = 64 if feature_mode == 'raw' else 2
            derived_in_channels = input_size // pts
        elif sampling_strategy == 'none':
            derived_in_channels = input_size // window_size
        else:
            if feature_mode == 'raw':
                pts = window_size // downsampling_stride
            else:
                pts = (window_size - 32) // downsampling_stride
            pts = max(1, pts)
            derived_in_channels = input_size // pts

    if derived_in_channels and base_ch > 0 and derived_in_channels % base_ch == 0:
        derived_harmonics = max(1, derived_in_channels // base_ch)
        if in_channels is None or (derived_in_channels != in_channels and derived_harmonics > 1):
            in_channels = derived_in_channels
        num_harmonics = max(num_harmonics, derived_harmonics)
    elif in_channels and base_ch > 0 and in_channels % base_ch == 0:
        num_harmonics = max(1, in_channels // base_ch)

    return num_harmonics, in_channels


def _safe_name(name: str, max_len: int = 80) -> str:
    """Безопасное имя для файлов (с ограничением длины)."""
    safe = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(name))
    if len(safe) <= max_len:
        return safe
    # Укорачиваем и добавляем хэш для уникальности
    digest = hashlib.md5(safe.encode('utf-8')).hexdigest()[:8]
    return f"{safe[:max_len]}_{digest}"


def _select_signal_columns(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    """Определяет доступные имена токов и напряжений в тестовом DataFrame."""
    cols = set(df.columns)

    # Токи
    current_candidates = ['IA', 'IB', 'IC', 'IN']
    if all(c in cols for c in current_candidates):
        current_cols = current_candidates
    else:
        # Фолбэк на первые 3-4 совпадения по шаблону
        current_cols = [c for c in df.columns if c.upper().startswith('I')][:4]

    # Напряжения
    if all(c in cols for c in ['UA', 'UB', 'UC', 'UN']):
        voltage_cols = ['UA', 'UB', 'UC', 'UN']
    elif all(c in cols for c in ['UA BB', 'UB BB', 'UC BB', 'UN BB']):
        voltage_cols = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
    elif all(c in cols for c in ['UA CL', 'UB CL', 'UC CL', 'UN CL']):
        voltage_cols = ['UA CL', 'UB CL', 'UC CL', 'UN CL']
    else:
        voltage_cols = [c for c in df.columns if c.upper().startswith('U')][:4]

    return current_cols, voltage_cols


def _build_window_any_labels(
    df: pl.DataFrame,
    target_cols: List[str],
    window_size: int
) -> Dict[str, np.ndarray]:
    """Строит метки по правилу "событие было в окне" (сдвиг вправо)."""
    labels = {}
    kernel = np.ones(window_size, dtype=np.int32)

    for col in target_cols:
        raw = df[col].to_numpy().astype(np.int32)
        # trailing window sum для каждого времени
        counts = np.convolve(raw, kernel, mode='full')
        window_any = (counts[:len(raw)] > 0).astype(np.int8)
        if len(raw) >= window_size:
            window_any[:window_size - 1] = 0
        labels[col] = window_any

    return labels


def _apply_time_range(
    time_axis: np.ndarray,
    series: Dict[str, np.ndarray],
    start_ms: float,
    end_ms: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Обрезает временной диапазон для набора сигналов."""
    mask = (time_axis >= start_ms) & (time_axis <= end_ms)
    return time_axis[mask], {k: v[mask] for k, v in series.items()}


def _plot_marking(
    out_path: Path,
    time_axis: np.ndarray,
    currents: Dict[str, np.ndarray],
    voltages: Dict[str, np.ndarray],
    real_labels: Dict[str, np.ndarray],
    pred_labels: Dict[str, np.ndarray],
    title: str,
    pred_probs: Optional[Dict[str, np.ndarray]] = None,
    plot_mode: str = 'discrete',
    threshold: float = 0.5
) -> None:
    """Строит график токов/напряжений и дискретов (реальные сверху, предикт снизу)."""
    labels = list(real_labels.keys())
    amplitudes = np.arange(1, len(labels) + 1)

    # Стандартная раскраска: Желтый (A), Зеленый (B), Красный (C), Синий (N)
    PHASE_COLORS = {
        'A': '#FFD700', 'B': '#228B22', 'C': '#FF4500', 'N': '#1E90FF',
        'IA': '#FFD700', 'IB': '#228B22', 'IC': '#FF4500', 'IN': '#1E90FF',
        'UA': '#FFD700', 'UB': '#228B22', 'UC': '#FF4500', 'UN': '#1E90FF'
    }

    def get_color(name: str, idx: int) -> str:
        name_uc = str(name).upper()
        # Поиск по ключам IA/UA и т.д.
        for key in ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']:
            if key in name_uc: return PHASE_COLORS[key]
        # Поиск по суффиксам/вхождениям фаз
        if ' A' in name_uc or name_uc.endswith(' A') or name_uc.endswith('(A)'): return PHASE_COLORS['A']
        if ' B' in name_uc or name_uc.endswith(' B') or name_uc.endswith('(B)'): return PHASE_COLORS['B']
        if ' C' in name_uc or name_uc.endswith(' C') or name_uc.endswith('(C)'): return PHASE_COLORS['C']
        if ' N' in name_uc or name_uc.endswith(' N') or name_uc.endswith('(N)'): return PHASE_COLORS['N']
        return f"C{idx % 10}"

    if plot_mode == 'confidence':
        height_ratios = [1.1, 1.1, 0.8] + [0.6] * len(labels)
        fig = plt.figure(figsize=(16, 10 + 1.2 * len(labels)))
        gs = fig.add_gridspec(nrows=3 + len(labels), ncols=1, height_ratios=height_ratios)

        ax_curr = fig.add_subplot(gs[0, 0])
        for i, (name, data) in enumerate(currents.items()):
            ax_curr.plot(time_axis, data, label=name, color=get_color(name, i), linewidth=1.2)
        ax_curr.set_ylabel("Токи")
        ax_curr.legend(loc='upper right')
        ax_curr.grid(True, alpha=0.3, linestyle=':')

        ax_volt = fig.add_subplot(gs[1, 0], sharex=ax_curr)
        for i, (name, data) in enumerate(voltages.items()):
            ax_volt.plot(time_axis, data, label=name, color=get_color(name, i), linewidth=1.2)
        ax_volt.set_ylabel("Напряжения")
        ax_volt.legend(loc='upper right')
        ax_volt.grid(True, alpha=0.3, linestyle=':')

        ax_disc = fig.add_subplot(gs[2, 0], sharex=ax_curr)
        # Реальные метки — в плюс
        for i, label_name in enumerate(labels):
            color = get_color(label_name, i)
            real_positions = [amplitudes[i] if v else np.nan for v in real_labels[label_name]]
            ax_disc.scatter(time_axis, real_positions, marker='o', s=16, alpha=0.8, color=color, label=f"GT: {label_name}")

        # Предсказания — в минус
        for i, label_name in enumerate(labels):
            color = get_color(label_name, i)
            pred_positions = [-amplitudes[i] if v else np.nan for v in pred_labels[label_name]]
            ax_disc.scatter(time_axis, pred_positions, marker='s', s=14, alpha=0.6, color=color, label=f"Pred: {label_name}")

        ax_disc.axhline(0, color='black', linewidth=1)
        ax_disc.set_ylim(-len(labels) - 0.5, len(labels) + 0.5)

        # Настраиваем y-ticks для дискретов
        y_ticks = np.concatenate([-amplitudes[::-1], amplitudes])
        y_tick_labels = [f"P:{l}" for l in labels[::-1]] + [f"G:{l}" for l in labels]
        ax_disc.set_yticks(y_ticks)
        ax_disc.set_yticklabels(y_tick_labels, fontsize=7)

        ax_disc.set_ylabel("Дискреты (GT:+, Pred:-)")
        ax_disc.grid(True, alpha=0.3, linestyle=':')
        ax_disc.legend(loc='upper right', ncols=2, fontsize=8)

        for i, label_name in enumerate(labels):
            ax_conf = fig.add_subplot(gs[3 + i, 0], sharex=ax_curr)
            color = get_color(label_name, i)
            probs = pred_probs.get(label_name, np.zeros_like(time_axis)) if pred_probs else np.zeros_like(time_axis)

            ax_conf.plot(time_axis, probs, color=color, linewidth=1.2, alpha=0.6)
            mask = probs >= threshold
            if np.any(mask):
                ax_conf.scatter(time_axis[mask], probs[mask], color=color, s=8, alpha=0.9)

            ax_conf.axhline(threshold, color='red', linewidth=0.9, linestyle='--', alpha=0.7)
            ax_conf.set_ylim(-0.02, 1.02)
            ax_conf.set_yticks([0.0, threshold, 1.0])
            ax_conf.set_yticklabels(["0", f"{threshold:.2f}", "1"], fontsize=7)
            ax_conf.set_ylabel(label_name, fontsize=8)
            ax_conf.grid(True, alpha=0.3, linestyle=':')

        ax_conf.set_xlabel("Время, мс")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    plt.figure(figsize=(16, 10))

    ax_curr = plt.subplot(3, 1, 1)
    for i, (name, data) in enumerate(currents.items()):
        ax_curr.plot(time_axis, data, label=name, color=get_color(name, i), linewidth=1.2)
    ax_curr.set_ylabel("Токи")
    ax_curr.legend(loc='upper right')
    ax_curr.grid(True, alpha=0.3, linestyle=':')

    ax_volt = plt.subplot(3, 1, 2)
    for i, (name, data) in enumerate(voltages.items()):
        ax_volt.plot(time_axis, data, label=name, color=get_color(name, i), linewidth=1.2)
    ax_volt.set_ylabel("Напряжения")
    ax_volt.legend(loc='upper right')
    ax_volt.grid(True, alpha=0.3, linestyle=':')

    ax_disc = plt.subplot(3, 1, 3)

    # Реальные метки — в плюс
    for i, label_name in enumerate(labels):
        color = get_color(label_name, i)
        real_positions = [amplitudes[i] if v else np.nan for v in real_labels[label_name]]
        ax_disc.scatter(time_axis, real_positions, marker='o', s=16, alpha=0.8, color=color, label=f"GT: {label_name}")

    # Предсказания — в минус
    for i, label_name in enumerate(labels):
        color = get_color(label_name, i)
        pred_positions = [-amplitudes[i] if v else np.nan for v in pred_labels[label_name]]
        ax_disc.scatter(time_axis, pred_positions, marker='s', s=14, alpha=0.6, color=color, label=f"Pred: {label_name}")

    ax_disc.axhline(0, color='black', linewidth=1)
    ax_disc.set_ylim(-len(labels) - 0.5, len(labels) + 0.5)

    # Настраиваем y-ticks для дискретов
    y_ticks = np.concatenate([-amplitudes[::-1], amplitudes])
    y_tick_labels = [f"P:{l}" for l in labels[::-1]] + [f"G:{l}" for l in labels]
    ax_disc.set_yticks(y_ticks)
    ax_disc.set_yticklabels(y_tick_labels, fontsize=7)

    ax_disc.set_ylabel("Дискреты (GT:+, Pred:-)")
    ax_disc.set_xlabel("Время, мс")
    ax_disc.grid(True, alpha=0.3, linestyle=':')
    ax_disc.legend(loc='upper right', ncols=2, fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def generate_marking_plots_for_model(
    exp_name: str,
    output_dir: Path,
    data_dir: Path,
    include_zero_current: bool = True,
    include_zero_voltage: bool = True,
    split: str = 'test',
    plot_mode: str = 'discrete',
    threshold: float = 0.5,
    inference_backend: str = 'auto',
    selected_files: Optional[List[str]] = None,
    file_time_ranges_ms: Optional[Dict[str, Tuple[float, float]]] = None
) -> None:
    """
    Генерация графиков разметки для осциллограмм по выбранной модели.
    
    Args:
        exp_name: Имя эксперимента.
        output_dir: Папка для сохранения отчётов.
        data_dir: Папка с датасетами.
        include_zero_current: Отображать ток нуля (IN).
        include_zero_voltage: Отображать напряжение нуля (UN).
        split: Какие данные использовать — 'test' или 'train'.
        plot_mode: 'discrete' (как раньше) или 'confidence' (уверенность по классам).
        threshold: Порог для подсветки уверенности.
        inference_backend: 'auto' | 'nn' | 'physics'.
            'physics' — использовать формульный алгоритм predict_ozz_physics.
        selected_files: Ограничить список файлов конкретным набором.
        file_time_ranges_ms: Диапазоны времени по файлам (мс) для обрезки графиков.
    """
    exp_dir = _find_experiment_dir(exp_name)
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config.json: {config_path}")

    config = json_load(config_path)

    # Параметры модели/данных
    window_size = config.get('data', {}).get('window_size', 320)
    sampling_rate = config.get('data', {}).get('sampling_rate', 1600)
    input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
    in_channels = config.get('model', {}).get('params', {}).get('in_channels')

    feature_mode = _resolve_feature_mode(config, exp_name)
    sampling_strategy, downsampling_stride = _resolve_sampling_strategy(
        exp_name, input_size, in_channels, window_size
    )

    num_harmonics, in_channels = _resolve_num_harmonics(
        feature_mode, sampling_strategy, downsampling_stride, window_size, in_channels, input_size
    )

    target_level = _resolve_target_level(exp_name, config)
    target_window_mode = _resolve_target_window_mode(config, exp_name)

    # Подготовка данных
    dm = DatasetManager(str(data_dir))
    if split == 'train':
        data_df = dm.load_train_df() # precomputed=False
        split_label = 'train'
    else:
        data_df = dm.load_test_df(precomputed=True)
        split_label = 'test'
    
    data_df = data_df.with_row_index("row_nr")
    data_df = prepare_labels_for_experiment(data_df, target_level)

    target_cols = get_target_columns(target_level, data_df)

    model_name_cfg = str(config.get('model', {}).get('name', '')).strip()
    backend = inference_backend.strip().lower()
    if backend not in ('auto', 'nn', 'physics'):
        raise ValueError("inference_backend должен быть 'auto', 'nn' или 'physics'")

    use_physics = (backend == 'physics') or (
        backend == 'auto' and model_name_cfg.lower() == 'physicsbaseline'
    )

    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not use_physics:
        model = _create_model_from_config(config)
        if model is None:
            raise ValueError("Не удалось создать модель из config.json")

        ckpt_path = exp_dir / "final_model.pt"
        if not ckpt_path.exists():
            ckpt_path = exp_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Не найден чекпоинт модели (best/final) в {exp_dir}")

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        _load_state_dict_safe(model, checkpoint, exp_dir.name, 'marking')
        model = model.to(device)
        model.eval()

    # Добавляем суффикс split к папке вывода для разделения train/test
    out_dir = output_dir / "marking_plots" / f"{exp_name}_{split_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Генерация графиков разметки для split='{split_label}'")
    print(f"Выходная папка: {out_dir}")

    # Работа по файлам
    file_names = data_df['file_name'].unique().to_list()
    if selected_files:
        selected_set = set(selected_files)
        file_names = [name for name in file_names if name in selected_set]
    for file_name in tqdm(file_names, desc=f"Разметка {split_label} осциллограмм"):
        file_df = data_df.filter(pl.col('file_name') == file_name)
        if len(file_df) < window_size + 1:
            continue

        # Маппинг сигналов
        current_cols, voltage_cols = _select_signal_columns(file_df)

        currents = {c: file_df[c].to_numpy() for c in current_cols[:3]}
        if include_zero_current and len(current_cols) > 3:
            currents[current_cols[3]] = file_df[current_cols[3]].to_numpy()

        voltages = {c: file_df[c].to_numpy() for c in voltage_cols[:3]}
        if include_zero_voltage and len(voltage_cols) > 3:
            voltages[voltage_cols[3]] = file_df[voltage_cols[3]].to_numpy()

        # Индексы скользящих окон
        indices = list(range(0, len(file_df) - window_size + 1))

        # Предсказания по всем окнам — со сглаживанием через аккумуляцию
        # Каждое окно покрывает точки [start_idx ... start_idx + window_size - 1].
        # Для каждой точки накапливаем сумму вероятностей и число покрытий.
        n_points = len(file_df)
        pred_prob_accum = {col: np.zeros(n_points, dtype=np.float64) for col in target_cols}
        pred_count = np.zeros(n_points, dtype=np.int32)
        pred_labels = {col: np.zeros(n_points, dtype=np.int8) for col in target_cols}
        pred_probs = {col: np.zeros(n_points, dtype=np.float32) for col in target_cols}

        if target_window_mode == 'any_in_window':
            real_labels = _build_window_any_labels(file_df, target_cols, window_size)
        else:
            real_labels = {col: file_df[col].to_numpy().astype(np.int8) for col in target_cols}

        if use_physics:
            if target_level != 'ozz':
                raise ValueError("Формульный backend поддерживается только для target_level='ozz'")

            signal_cols = current_cols + voltage_cols
            signal_arr = file_df.select(signal_cols).to_numpy().astype(np.float64)

            model_params = config.get('model', {}).get('params', {})
            u0_thr_norm = float(model_params.get('u0_threshold', u0_threshold_raw_to_normalized(10.0)))
            delay_samples = int(model_params.get('operate_delay_samples', 32))
            delay_periods = float(model_params.get('operate_delay_periods', 0.0))

            for start_idx in indices:
                end_idx = start_idx + window_size
                window = signal_arr[start_idx:end_idx, :]
                pred_class = predict_ozz_physics(
                    window,
                    fs=sampling_rate,
                    u0_threshold=u0_thr_norm,
                    operate_delay_samples=delay_samples,
                    operate_delay_periods=delay_periods,
                )

                prob_vec = np.zeros(len(target_cols), dtype=np.float32)
                if pred_class is not None:
                    prob_vec[0] = 1.0
                    if pred_class == 1 and len(target_cols) > 1:
                        prob_vec[1] = 1.0
                    elif pred_class == 2 and len(target_cols) > 2:
                        prob_vec[2] = 1.0

                end_cover = min(start_idx + window_size, n_points)
                pred_count[start_idx:end_cover] += 1
                for k, col in enumerate(target_cols):
                    pred_prob_accum[col][start_idx:end_cover] += float(prob_vec[k])
        else:
            ds = OscillogramDataset(
                dataframe=file_df,
                indices=indices,
                window_size=window_size,
                mode='classification',
                feature_mode=feature_mode,
                sampling_strategy=sampling_strategy,
                downsampling_stride=downsampling_stride,
                target_columns=target_cols,
                target_level=target_level if target_level != 'base' else 'base_labels',
                target_window_mode=target_window_mode,
                physical_normalization=True,
                norm_coef_path=config.get('data', {}).get('norm_coef_path'),
                num_harmonics=num_harmonics,
                augment=False
            )

            data_mode = config.get('data', {}).get('mode', 'multilabel')
            model_name = model.__class__.__name__
            is_conditional_model = 'conditional' in model_name.lower() or 'conditional' in exp_name.lower()
            if target_level == 'base_sequential' and data_mode == 'multilabel':
                data_mode = 'multitask_conditional'

            batch_size = 64
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                x_batch = []
                for start_idx in batch_indices:
                    x, _ = ds[start_idx]
                    x_batch.append(x)

                x_tensor = torch.stack(x_batch).to(device)
                with torch.no_grad():
                    outputs = model(x_tensor)

                if data_mode == 'classification':
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    pred_matrix = np.zeros((len(preds), len(target_cols)), dtype=np.int8)
                    pred_matrix[np.arange(len(preds)), preds] = 1
                    prob_matrix = probs
                elif data_mode == 'multitask_conditional' or is_conditional_model or target_level == 'base_sequential':
                    probs = torch.sigmoid(outputs[:, :4]).cpu().numpy()
                    pred_matrix = (probs > threshold).astype(np.int8)
                    prob_matrix = probs.copy()

                    normal_mask = pred_matrix[:, 0:1] == 1
                    pred_matrix[:, 1:] = np.where(normal_mask, 0, pred_matrix[:, 1:])
                    prob_matrix[:, 1:] = np.where(normal_mask, 0.0, prob_matrix[:, 1:])

                    ml3_mask = pred_matrix[:, 3] == 1
                    pred_matrix[:, 2] = np.where(ml3_mask, 0, pred_matrix[:, 2])
                    prob_matrix[:, 2] = np.where(ml3_mask, 0.0, prob_matrix[:, 2])
                else:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    pred_matrix = (probs > threshold).astype(np.int8)
                    prob_matrix = probs

                for j, start_idx in enumerate(batch_indices):
                    end_cover = min(start_idx + window_size, n_points)
                    pred_count[start_idx:end_cover] += 1
                    for k, col in enumerate(target_cols):
                        pred_prob_accum[col][start_idx:end_cover] += float(prob_matrix[j, k])

        # Сглаживание: средняя вероятность = сумма / количество покрытий
        for col in target_cols:
            mask = pred_count > 0
            pred_probs[col][mask] = (pred_prob_accum[col][mask] / pred_count[mask]).astype(np.float32)
            pred_labels[col] = (pred_probs[col] >= threshold).astype(np.int8)

        # Ось времени
        time_axis = np.arange(len(file_df)) * 1000 / sampling_rate

        if file_time_ranges_ms and file_name in file_time_ranges_ms:
            start_ms, end_ms = file_time_ranges_ms[file_name]
            mask = (time_axis >= start_ms) & (time_axis <= end_ms)
            time_axis = time_axis[mask]
            currents = {k: v[mask] for k, v in currents.items()}
            voltages = {k: v[mask] for k, v in voltages.items()}
            real_labels = {k: v[mask] for k, v in real_labels.items()}
            pred_labels = {k: v[mask] for k, v in pred_labels.items()}
            pred_probs = {k: v[mask] for k, v in pred_probs.items()}

        info = parse_experiment_info(exp_name)
        model_family = info.get("model_family", "Model")
        title = f"{model_family} | {exp_name} | {file_name}"
        name_hash = hashlib.md5(f"{file_name}|{exp_name}".encode('utf-8')).hexdigest()[:12]
        out_name = f"mark_{name_hash}.png"
        out_path = out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        _plot_marking(
            out_path=out_path,
            time_axis=time_axis,
            currents=currents,
            voltages=voltages,
            real_labels=real_labels,
            pred_labels=pred_labels,
            title=title,
            pred_probs=pred_probs,
            plot_mode=plot_mode,
            threshold=threshold
        )


def json_load(path: Path) -> Dict[str, Any]:
    """Безопасная загрузка JSON."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графиков разметки данных моделью")
    parser.add_argument("--exp", type=str, default=None,
                        help="Имя эксперимента (папка в experiments)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Папка отчёта (туда сохраняются графики)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Папка датасетов")
    parser.add_argument("--no-zero-current", action="store_true",
                        help="Отключить отображение I0 (IN)")
    parser.add_argument("--no-zero-voltage", action="store_true",
                        help="Отключить отображение U0 (UN)")
    parser.add_argument("--split", type=str, default='test', choices=['train', 'test'],
                        help="Какие данные использовать: 'train' или 'test' (по умолчанию: test)")
    parser.add_argument("--plot-mode", type=str, default='discrete', choices=['discrete', 'confidence'],
                        help="Режим графика: дискреты или уверенность")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Порог для уверенности модели")
    parser.add_argument("--backend", type=str, default='auto', choices=['auto', 'nn', 'physics'],
                        help="Бэкенд инференса: нейро-сеть или формульный алгоритм")
    parser.add_argument("--files", type=str, default=None,
                        help="Список файлов через запятую для построения")
    parser.add_argument("--time-ranges-json", type=str, default=None,
                        help="JSON с диапазонами времени (мс) по файлам")

    args, _ = parser.parse_known_args()

    # === РУЧНОЙ РЕЖИМ (как в других скриптах) ===
    MANUAL_RUN = False

    if MANUAL_RUN or args.exp is None:
        # EXP_NAME = "Exp_2.6.1_PhysicsKAN_medium_phase_polar_stride_base_weights_aug"
        EXP_NAME = "Exp_2.6.9_cPhysicsKAN_heavy_phase_polar_stride_base_weights_aug"
        OUTPUT_DIR = "reports/Exp_2_5_and_start_Exp_2_6"
        DATA_DIR = "data/ml_datasets"
        INCLUDE_ZERO_CURRENT = True
        INCLUDE_ZERO_VOLTAGE = True
        SPLIT = 'train'  # 'train' или 'test'
        PLOT_MODE = 'confidence'  # 'discrete' или 'confidence'
        THRESHOLD = 0.5
        INFERENCE_BACKEND = 'auto'  # 'auto' | 'nn' | 'physics'

        # Пример выбора файлов и диапазонов времени (мс)
        # SELECTED_FILES = ["file_001.cfg", "file_002.cfg"]
        # FILE_TIME_RANGES_MS = {
        #     "file_001.cfg": (120.0, 420.0),
        #     "file_002.cfg": (800.0, 1400.0)
        # }
        SELECTED_FILES = None
        FILE_TIME_RANGES_MS = None
    else:
        EXP_NAME = args.exp
        OUTPUT_DIR = args.output_dir or "reports/Exp_2_5_and_start_Exp_2_6"
        DATA_DIR = args.data_dir or "data/ml_datasets"
        INCLUDE_ZERO_CURRENT = not args.no_zero_current
        INCLUDE_ZERO_VOLTAGE = not args.no_zero_voltage
        SPLIT = args.split
        PLOT_MODE = args.plot_mode
        THRESHOLD = args.threshold
        INFERENCE_BACKEND = args.backend
        SELECTED_FILES = args.files.split(',') if args.files else None
        FILE_TIME_RANGES_MS = None
        if args.time_ranges_json:
            with open(args.time_ranges_json, 'r', encoding='utf-8') as f:
                FILE_TIME_RANGES_MS = json.load(f)

    generate_marking_plots_for_model(
        exp_name=EXP_NAME,
        output_dir=ROOT_DIR / OUTPUT_DIR,
        data_dir=ROOT_DIR / DATA_DIR,
        include_zero_current=INCLUDE_ZERO_CURRENT,
        include_zero_voltage=INCLUDE_ZERO_VOLTAGE,
        split=SPLIT,
        plot_mode=PLOT_MODE,
        threshold=THRESHOLD,
        inference_backend=INFERENCE_BACKEND,
        selected_files=SELECTED_FILES,
        file_time_ranges_ms=FILE_TIME_RANGES_MS
    )
