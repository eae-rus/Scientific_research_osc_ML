"""
Визуализация реконструкции сигнала из pretrain-модели (SSL).

Загружает pretrain-чекпоинт, прогоняет осциллограмму через модель в режиме SSL,
восстанавливает сигнал из спектральных признаков (вход и выход) и строит
сравнительные графики.

Три линии на каждом канале:
  1. Оригинальный сигнал (сырые данные из CSV)
  2. Идеальная реконструкция (FFT h1-9 от оригинала → IFFT)
  3. Реконструкция из SSL-модели (output polar → IFFT)

Использование:
  # CLI-режим
  python scripts/phase4_experiments/plot_pretrain_reconstruction.py \\
      --checkpoint experiments/phase4/pretrain_PhysicalKANTransformer_*/best_model.pt \\
      --csv data/ml_datasets/labeled.csv \\
      --file-name "some_osc.cfg"

  # Ручной запуск (MANUAL_RUN = True) — настройки ниже в блоке __main__
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

# --- Пути проекта ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.augmented_dataset import (
    RAW_CHANNELS,
    FFT_WINDOW,
    SAMPLES_PER_PERIOD,
    DEFAULT_SUB_PERIODS,
    compute_spectral_from_raw,
    compute_stride,
    standardize_voltage_columns,
)
from osc_tools.ml.models.transformer import PhysicalKANTransformer, BaselineTransformer
from osc_tools.visualization.spectral_reconstruction import (
    plot_reconstruction,
    plot_harmonic_comparison,
    reconstruct_channel_from_polar,
)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Загрузка pretrain-модели
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

def load_pretrain_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Загружает pretrain-модель из чекпоинта.

    Returns:
        (model в режиме SSL, config)
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})

    model_type = config.get('model_type', 'PhysicalKANTransformer')

    common_kwargs = dict(
        num_input_channels=config.get('num_input_channels', 220),
        d_model=config.get('d_model', 48),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 6),
        dropout=0.0,
        max_seq_len=64,
    )

    if model_type == 'PhysicalKANTransformer':
        model = PhysicalKANTransformer(
            kan_grid_size=config.get('kan_grid_size', 5),
            use_angle_gate=config.get('use_angle_gate', True),
            use_mixed_layer_norm=config.get('use_mixed_layer_norm', False),
            **common_kwargs,
        )
    elif model_type == 'BaselineTransformer':
        model = BaselineTransformer(**common_kwargs)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Извлечение данных
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

def extract_oscillogram_data(
    csv_path: str | Path,
    file_name: str,
    window_start: int = 0,
    window_size: int = 320,
    stride: int | None = None,
    warmup: int = FFT_WINDOW,
    num_harmonics: int = 9,
    sub_periods: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает raw-данные и polar-признаки одной осциллограммы.

    Args:
        csv_path: путь к CSV с данными
        file_name: имя файла осциллограммы
        window_start: начальный отсчёт окна
        window_size: размер окна (в raw-отсчётах)
        stride: шаг FFT (None = auto)
        warmup: warmup для FFT
        num_harmonics: число стандартных гармоник
        sub_periods: низшие гармоники

    Returns:
        (raw, polar) — raw shape (T, 8), polar shape (T_steps, C_polar)
    """
    if sub_periods is None:
        sub_periods = DEFAULT_SUB_PERIODS
    if stride is None:
        stride = compute_stride()

    df = pl.read_csv(csv_path)
    df = standardize_voltage_columns(df)

    # Фильтруем нужную осциллограмму
    osc = df.filter(pl.col('file_name') == file_name)
    if len(osc) == 0:
        available = df['file_name'].unique().to_list()[:10]
        raise ValueError(
            f"Осциллограмма '{file_name}' не найдена. "
            f"Доступные (первые 10): {available}"
        )

    # Извлекаем raw-каналы
    raw_cols = [c for c in RAW_CHANNELS if c in osc.columns]
    if len(raw_cols) < 8:
        print(f"  WARN: только {len(raw_cols)} raw-каналов из 8: {raw_cols}")

    raw_full = osc.select(raw_cols).to_numpy().astype(np.float64)

    # Берём окно
    end = min(window_start + window_size, len(raw_full))
    raw = raw_full[window_start:end]

    # Если не хватает каналов до 8, заполняем нулями
    if raw.shape[1] < 8:
        padded = np.zeros((raw.shape[0], 8), dtype=np.float64)
        padded[:, :raw.shape[1]] = raw
        raw = padded

    # Вычисляем polar-признаки
    polar = compute_spectral_from_raw(
        raw.astype(np.float32),
        num_harmonics=num_harmonics,
        sub_periods=sub_periods,
        include_symmetric=True,
        stride=stride,
        warmup=warmup,
    )

    return raw, polar


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Инференс SSL
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

@torch.no_grad()
def run_ssl_inference(
    model: torch.nn.Module,
    polar: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Прогоняет polar-признаки через модель в SSL-режиме.

    Args:
        model: pretrain-модель
        polar: (T_steps, C_polar) входные признаки

    Returns:
        (T_steps, C_polar) реконструированные polar-признаки
    """
    # (1, C, T) — формат модели
    x = torch.from_numpy(polar.T[np.newaxis]).float().to(device)

    output = model(x, mode='ssl')
    ssl_out = output['ssl']  # (1, C, T)

    # Обратно в (T_steps, C_polar)
    recon = ssl_out[0].cpu().numpy().T  # (T, C)
    return recon


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Главная функция визуализации
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

def visualize_pretrain_reconstruction(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    file_name: str,
    window_start: int = 0,
    window_size: int = 320,
    stride: int | None = None,
    num_harmonics: int = 9,
    save_dir: str | Path | None = None,
    show: bool = True,
):
    """Полный пайплайн визуализации pretrain-реконструкции.

    Args:
        checkpoint_path: путь к pretrain-чекпоинту
        csv_path: путь к CSV с данными
        file_name: имя осциллограммы
        window_start: начальный отсчёт
        window_size: размер окна (в raw-отсчётах)
        stride: шаг FFT (None = auto)
        num_harmonics: число гармоник
        save_dir: папка для сохранения (None → только показ)
        show: показывать ли графики (plt.show)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warmup = FFT_WINDOW

    if stride is None:
        stride = compute_stride()

    print(f"=== Визуализация pretrain-реконструкции ===")
    print(f"  Чекпоинт: {checkpoint_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Осциллограмма: {file_name}")
    print(f"  Окно: [{window_start}, {window_start + window_size}), stride={stride}")
    print(f"  Device: {device}")

    # 1. Загрузка модели
    model, config = load_pretrain_model(Path(checkpoint_path), device)
    print(f"  Модель: {config.get('model_type', '?')}, "
          f"d_model={config.get('d_model', '?')}")

    # 2. Извлечение данных
    raw, input_polar = extract_oscillogram_data(
        csv_path, file_name,
        window_start=window_start,
        window_size=window_size,
        stride=stride,
        warmup=warmup,
        num_harmonics=num_harmonics,
    )
    print(f"  Raw shape: {raw.shape}, Polar shape: {input_polar.shape}")

    # 3. SSL инференс
    model_polar = run_ssl_inference(model, input_polar, device)
    print(f"  SSL output shape: {model_polar.shape}")

    # 4. Подготовка выходной папки
    save_path_waveform = None
    save_path_harmonics = None
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe_name = file_name.replace('.', '_').replace('/', '_')
        save_path_waveform = str(out / f'recon_waveform_{safe_name}.png')
        save_path_harmonics = str(out / f'recon_harmonics_{safe_name}.png')

    # 5. Графики реконструкции сигнала
    plot_reconstruction(
        raw=raw,
        input_polar=input_polar,
        model_polar=model_polar,
        stride=stride,
        warmup=warmup,
        num_harmonics=num_harmonics,
        save_path=save_path_waveform if save_dir else None,
        title_prefix=f'{file_name}: ',
    )

    # 6. Графики сравнения амплитуд гармоник (для тока фазы A)
    plot_harmonic_comparison(
        input_polar=input_polar,
        model_polar=model_polar,
        ch=0,  # IA
        num_harmonics=num_harmonics,
        save_path=save_path_harmonics if save_dir else None,
        title_prefix=f'{file_name}: ',
    )

    print("  Готово!")
    return raw, input_polar, model_polar


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# CLI / Ручной запуск
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

# ═══════════════ РУЧНОЙ РЕЖИМ ═══════════════
MANUAL_RUN = False  # True → игнорирует CLI, использует параметры ниже

# Настройки ручного режима:
MANUAL_CONFIG = {
    'checkpoint': str(PROJECT_ROOT / 'experiments' / 'phase4'
                      / 'ЗДЕСЬ_ПАПКА_ПРЕТРЕЙНА' / 'best_model.pt'),
    'csv': str(PROJECT_ROOT / 'data' / 'ml_datasets' / 'labeled.csv'),
    'file_name': 'ЗДЕСЬ_ИМЯ_ОСЦ.cfg',
    'window_start': 0,
    'window_size': 480,  # 15 периодов (10 + 5 для наглядности)
    'stride': None,      # None = auto (16 = полпериода)
    'num_harmonics': 9,
    'save_dir': str(PROJECT_ROOT / 'reports' / 'pretrain_reconstruction'),
}
# ════════════════════════════════════════════


def main():
    if MANUAL_RUN:
        cfg = MANUAL_CONFIG
        visualize_pretrain_reconstruction(
            checkpoint_path=cfg['checkpoint'],
            csv_path=cfg['csv'],
            file_name=cfg['file_name'],
            window_start=cfg['window_start'],
            window_size=cfg['window_size'],
            stride=cfg['stride'],
            num_harmonics=cfg['num_harmonics'],
            save_dir=cfg['save_dir'],
        )
        return

    parser = argparse.ArgumentParser(
        description='Визуализация pretrain SSL-реконструкции',
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Путь к pretrain-чекпоинту (.pt)')
    parser.add_argument('--csv', required=True,
                        help='Путь к CSV с данными')
    parser.add_argument('--file-name', required=True,
                        help='Имя осциллограммы (file_name в CSV)')
    parser.add_argument('--window-start', type=int, default=0,
                        help='Начальный отсчёт окна')
    parser.add_argument('--window-size', type=int, default=480,
                        help='Размер окна в отсчётах (default: 480 = 15 периодов)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Шаг FFT (default: auto = 16)')
    parser.add_argument('--num-harmonics', type=int, default=9,
                        help='Число гармоник (default: 9)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Папка для сохранения PNG')
    parser.add_argument('--no-show', action='store_true',
                        help='Не показывать графики (только сохранить)')
    args = parser.parse_args()

    visualize_pretrain_reconstruction(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        file_name=args.file_name,
        window_start=args.window_start,
        window_size=args.window_size,
        stride=args.stride,
        num_harmonics=args.num_harmonics,
        save_dir=args.save_dir,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
