import argparse
import re
import sys
import hashlib
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

# Переиспользуем проверенные функции создания модели и загрузки весов
from scripts.evaluation.aggregate_reports import _create_model_from_config, _load_state_dict_safe


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


def _resolve_target_level(exp_name: str) -> str:
    """Определяем target_level по имени эксперимента."""
    name = exp_name.lower()
    if 'full_by_levels' in name or ('hier_' in name and '2.6.4' in name):
        return 'full_by_levels'
    if '2.6.4' in name and 'full' in name:
        return 'full'
    return 'base'


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


def _plot_marking(
    out_path: Path,
    time_axis: np.ndarray,
    currents: Dict[str, np.ndarray],
    voltages: Dict[str, np.ndarray],
    real_labels: Dict[str, np.ndarray],
    pred_labels: Dict[str, np.ndarray],
    title: str
) -> None:
    """Строит график токов/напряжений и дискретов (реальные сверху, предикт снизу)."""
    plt.figure(figsize=(16, 10))

    ax_curr = plt.subplot(3, 1, 1)
    for name, data in currents.items():
        ax_curr.plot(time_axis, data, label=name, linewidth=1.2)
    ax_curr.set_ylabel("Токи")
    ax_curr.legend(loc='upper right')
    ax_curr.grid(True, alpha=0.3, linestyle=':')

    ax_volt = plt.subplot(3, 1, 2)
    for name, data in voltages.items():
        ax_volt.plot(time_axis, data, label=name, linewidth=1.2)
    ax_volt.set_ylabel("Напряжения")
    ax_volt.legend(loc='upper right')
    ax_volt.grid(True, alpha=0.3, linestyle=':')

    ax_disc = plt.subplot(3, 1, 3)
    labels = list(real_labels.keys())
    amplitudes = np.arange(1, len(labels) + 1)

    # Реальные метки — в плюс
    for i, label_name in enumerate(labels):
        color = f"C{i % 10}"
        real_positions = [amplitudes[i] if v else np.nan for v in real_labels[label_name]]
        ax_disc.scatter(time_axis, real_positions, marker='o', s=16, alpha=0.8, color=color, label=f"GT: {label_name}")

    # Предсказания — в минус
    for i, label_name in enumerate(labels):
        color = f"C{i % 10}"
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
    include_zero_voltage: bool = True
) -> None:
    """Генерация графиков разметки для всех тестовых осциллограмм по выбранной модели."""
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

    target_level = _resolve_target_level(exp_name)

    # Подготовка данных
    dm = DatasetManager(str(data_dir))
    test_df = dm.load_test_df(precomputed=False)
    test_df = test_df.with_row_index("row_nr")
    test_df = prepare_labels_for_experiment(test_df, target_level)

    target_cols = get_target_columns(target_level, test_df)

    # Модель
    model = _create_model_from_config(config)
    if model is None:
        raise ValueError("Не удалось создать модель из config.json")

    ckpt_path = exp_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = exp_dir / "final_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт модели (best/final) в {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    _load_state_dict_safe(model, checkpoint, exp_dir.name, 'marking')
    model = model.to(device)
    model.eval()

    out_dir = output_dir / "marking_plots" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Работа по файлам
    file_names = test_df['file_name'].unique().to_list()
    for file_name in tqdm(file_names, desc="Разметка тестовых осциллограмм"):
        file_df = test_df.filter(pl.col('file_name') == file_name)
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

        # Датасет для предсказаний
        indices = list(range(0, len(file_df) - window_size + 1))
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
            physical_normalization=True,
            norm_coef_path=config.get('data', {}).get('norm_coef_path'),
            num_harmonics=num_harmonics,
            augment=False
        )

        # Предсказания по всем окнам
        pred_labels = {col: np.zeros(len(file_df), dtype=np.int8) for col in target_cols}
        real_labels = {col: file_df[col].to_numpy().astype(np.int8) for col in target_cols}

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

            if config.get('data', {}).get('mode', 'multilabel') == 'classification':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                pred_matrix = np.zeros((len(preds), len(target_cols)), dtype=np.int8)
                pred_matrix[np.arange(len(preds)), preds] = 1
            else:
                probs = torch.sigmoid(outputs)
                pred_matrix = (probs > 0.5).int().cpu().numpy()

            for j, start_idx in enumerate(batch_indices):
                target_idx = start_idx + (window_size - 1)
                if target_idx >= len(file_df):
                    continue
                for k, col in enumerate(target_cols):
                    pred_labels[col][target_idx] = pred_matrix[j, k]

        # Ось времени
        time_axis = np.arange(len(file_df)) * 1000 / sampling_rate

        title = f"{exp_name} | {file_name}"
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
            title=title
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

    args, _ = parser.parse_known_args()

    # === РУЧНОЙ РЕЖИМ (как в других скриптах) ===
    MANUAL_RUN = True

    if MANUAL_RUN or args.exp is None:
        EXP_NAME = "Exp_2.6.1_PhysicsKAN_medium_phase_polar_stride_base_weights_aug"
        OUTPUT_DIR = "reports/Exp_2_5_and_start_Exp_2_6"
        DATA_DIR = "data/ml_datasets"
        INCLUDE_ZERO_CURRENT = True
        INCLUDE_ZERO_VOLTAGE = True
    else:
        EXP_NAME = args.exp
        OUTPUT_DIR = args.output_dir or "reports/Exp_2_5_and_start_Exp_2_6"
        DATA_DIR = args.data_dir or "data/ml_datasets"
        INCLUDE_ZERO_CURRENT = not args.no_zero_current
        INCLUDE_ZERO_VOLTAGE = not args.no_zero_voltage

    generate_marking_plots_for_model(
        exp_name=EXP_NAME,
        output_dir=ROOT_DIR / OUTPUT_DIR,
        data_dir=ROOT_DIR / DATA_DIR,
        include_zero_current=INCLUDE_ZERO_CURRENT,
        include_zero_voltage=INCLUDE_ZERO_VOLTAGE
    )
