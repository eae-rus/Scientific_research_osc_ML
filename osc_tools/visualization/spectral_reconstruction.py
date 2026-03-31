"""Восстановление временного сигнала из спектральных признаков.

Обратное преобразование: polar features (amp, angle) → complex phasors → IFFT → signal.
Используется для визуальной проверки качества SSL pretrain и корректности
спектрального представления.

Формат спектральных признаков (polar):
    [mag_ch0_h1, ang_ch0_h1, mag_ch0_h2, ..., mag_ch0_lh1, ang_ch0_lh1, ...]
    Для каждого из 8 каналов: (num_harmonics + num_sub_periods) × 2 значений.
    Затем опционально 6 × 2 значения для симметричных составляющих.

Соответствие FFT ↔ polar:
    phasor_k = 2 * FFT[k] / N   →   amp = |phasor_k|, angle = arg(phasor_k) - ref_angle
    Восстановление: FFT[k] = phasor_k / 2, FFT[N-k] = conj(FFT[k]), signal = IFFT(FFT) * N
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from osc_tools.ml.augmented_dataset import (
    RAW_CHANNELS, NUM_RAW, FFT_WINDOW, SAMPLES_PER_PERIOD,
    DEFAULT_SUB_PERIODS, NUM_SYMMETRIC,
)


# ──────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────

def _channel_index_range(
    ch: int,
    num_harmonics: int = 9,
    num_sub_periods: int = 4,
) -> tuple[int, int]:
    """Возвращает (start, end) индексов в polar-векторе для канала ch.

    Каждый канал занимает (num_harmonics + num_sub_periods) * 2 значений.
    """
    h_total = num_harmonics + num_sub_periods
    start = ch * h_total * 2
    end = start + h_total * 2
    return start, end


def polar_to_complex(
    polar: np.ndarray,
    ch: int,
    num_harmonics: int = 9,
    num_sub_periods: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает комплексные фазоры из polar-вектора для одного канала.

    Args:
        polar: (..., C_polar) массив polar-признаков
        ch: индекс канала (0-7)
        num_harmonics: кол-во стандартных гармоник
        num_sub_periods: кол-во низших гармоник

    Returns:
        (harmonics, low_harmonics) — комплексные фазоры,
        shapes (..., num_harmonics) и (..., num_sub_periods).
    """
    start, _ = _channel_index_range(ch, num_harmonics, num_sub_periods)
    h_total = num_harmonics + num_sub_periods

    amps = polar[..., start:start + h_total * 2:2]
    angles = polar[..., start + 1:start + h_total * 2:2]
    phasors = amps * np.exp(1j * angles)

    harmonics = phasors[..., :num_harmonics]
    low = phasors[..., num_harmonics:]
    return harmonics, low


def reconstruct_one_period(
    phasors: np.ndarray,
    samples_per_period: int = SAMPLES_PER_PERIOD,
) -> np.ndarray:
    """Восстанавливает один период сигнала из комплексных фазоров гармоник 1..H.

    Args:
        phasors: (num_harmonics,) комплексные фазоры
        samples_per_period: число отсчётов в периоде (= размер FFT-окна)

    Returns:
        (samples_per_period,) восстановленный сигнал
    """
    N = samples_per_period
    fft_array = np.zeros(N, dtype=np.complex128)
    H = len(phasors)

    for k in range(H):
        bin_idx = k + 1  # гармоника 1 → bin 1
        if bin_idx >= N:
            break
        half = phasors[k] / 2.0  # phasor = 2 * FFT[k]/N
        fft_array[bin_idx] = half
        conj_idx = N - bin_idx
        if conj_idx != bin_idx:
            fft_array[conj_idx] = np.conj(half)

    signal = np.real(np.fft.ifft(fft_array)) * N
    return signal.astype(np.float64)


def reconstruct_one_period_vectorized(
    phasors: np.ndarray,
    samples_per_period: int = SAMPLES_PER_PERIOD,
) -> np.ndarray:
    """Батчевое восстановление периодов.

    Args:
        phasors: (n_steps, num_harmonics) комплексные фазоры

    Returns:
        (n_steps, samples_per_period)
    """
    n_steps, H = phasors.shape
    N = samples_per_period
    fft_array = np.zeros((n_steps, N), dtype=np.complex128)

    max_h = min(H, N - 1)
    for k in range(max_h):
        bin_idx = k + 1
        half = phasors[:, k] / 2.0
        fft_array[:, bin_idx] = half
        conj_idx = N - bin_idx
        if conj_idx != bin_idx:
            fft_array[:, conj_idx] = np.conj(half)

    signals = np.real(np.fft.ifft(fft_array, axis=1)) * N
    return signals


# ──────────────────────────────────────────────────────────────────────────
# Полная реконструкция сигнала (overlap-add)
# ──────────────────────────────────────────────────────────────────────────

def reconstruct_channel_from_polar(
    polar: np.ndarray,
    ch: int,
    stride: int,
    warmup: int = 0,
    num_harmonics: int = 9,
    num_sub_periods: int = 4,
    total_length: Optional[int] = None,
) -> np.ndarray:
    """Восстанавливает временной сигнал одного канала из polar-признаков.

    Реконструкция через overlap-add: на каждом шаге синтезируется один
    период из гармоник 1..num_harmonics, периоды накладываются и усредняются.

    Args:
        polar: (T_steps, C_polar) спектральные polar-признаки
        ch: индекс канала (0=IA, 1=IB, ..., 4=UA, 5=UB, 6=UC, 7=UN)
        stride: шаг между позициями FFT-окон (в отсчётах)
        warmup: сдвиг начала (warmup период для FFT)
        num_harmonics: только стандартные гармоники используются для реконструкции
        num_sub_periods: кол-во низших гармоник (для правильного парсинга)
        total_length: желаемая длина output-сигнала (обрезка/дополнение)

    Returns:
        (total_length,) восстановленный сигнал
    """
    harmonics, _ = polar_to_complex(polar, ch, num_harmonics, num_sub_periods)
    # harmonics shape: (T_steps, num_harmonics)

    N = SAMPLES_PER_PERIOD
    n_steps = harmonics.shape[0]
    out_len = warmup + (n_steps - 1) * stride + N
    if total_length is not None:
        out_len = max(out_len, total_length)

    signal = np.zeros(out_len, dtype=np.float64)
    count = np.zeros(out_len, dtype=np.float64)

    # Батчевая реконструкция всех периодов
    periods = reconstruct_one_period_vectorized(harmonics, N)

    for step in range(n_steps):
        start = warmup + step * stride
        end = start + N
        if end > out_len:
            break
        signal[start:end] += periods[step]
        count[start:end] += 1.0

    # Усреднение перекрывающихся участков
    mask = count > 0
    signal[mask] /= count[mask]

    if total_length is not None:
        signal = signal[:total_length]

    return signal


def reconstruct_ideal_from_raw(
    raw_1ch: np.ndarray,
    stride: int,
    warmup: int = 0,
    num_harmonics: int = 9,
) -> np.ndarray:
    """Идеальная реконструкция из сырого сигнала: FFT → h1..H → IFFT.

    Показывает, сколько информации теряется при ограничении до H гармоник.
    Результат в абсолютной фазе (без ссылочного сдвига).

    Args:
        raw_1ch: (T,) одноканальный сырой сигнал
        stride: шаг FFT (в отсчётах)
        warmup: сдвиг начала
        num_harmonics: число гармоник для реконструкции

    Returns:
        (T,) восстановленный сигнал (с overlap-add)
    """
    N = FFT_WINDOW
    T = len(raw_1ch)

    positions = np.arange(warmup, T - N + 1, stride)
    if len(positions) == 0:
        return np.zeros_like(raw_1ch)

    # Батчевое FFT
    windows = np.lib.stride_tricks.sliding_window_view(raw_1ch, N)
    batch_windows = windows[positions]
    fft_all = np.fft.fft(batch_windows, axis=1)

    # Обнуляем всё, кроме гармоник 1..num_harmonics
    fft_filtered = np.zeros_like(fft_all)
    max_h = min(num_harmonics, N // 2)
    for k in range(1, max_h + 1):
        fft_filtered[:, k] = fft_all[:, k]
        fft_filtered[:, N - k] = fft_all[:, N - k]

    # IFFT
    periods = np.real(np.fft.ifft(fft_filtered, axis=1))

    # Overlap-add
    signal = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, pos in enumerate(positions):
        signal[pos:pos + N] += periods[i]
        count[pos:pos + N] += 1.0

    mask = count > 0
    signal[mask] /= count[mask]
    return signal


# ──────────────────────────────────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────────────────────────────────

def plot_reconstruction(
    raw: np.ndarray,
    input_polar: np.ndarray,
    model_polar: Optional[np.ndarray],
    stride: int,
    warmup: int = 0,
    num_harmonics: int = 9,
    num_sub_periods: int = 4,
    channels: Optional[list[int]] = None,
    save_path: Optional[str] = None,
    title_prefix: str = '',
    figsize: tuple[float, float] = (18, 20),
):
    """Строит сравнительные графики реконструкции для всех каналов.

    3 линии на каждом подграфике:
    1. Оригинальный сигнал (сырые данные)
    2. Идеальная реконструкция (FFT h1-H из raw → IFFT)
    3. Реконструкция из модели (SSL output → polar → IFFT)

    Args:
        raw: (T, 8) сырые данные
        input_polar: (T_steps, C_polar) входные polar-признаки
        model_polar: (T_steps, C_polar) выход SSL модели (None → без линии модели)
        stride: шаг FFT
        warmup: начальный сдвиг
        num_harmonics: число гармоник
        num_sub_periods: число низших суб-периодов
        channels: список индексов каналов (None → все 8)
        save_path: путь для сохранения PNG (None → plt.show())
        title_prefix: префикс заголовка
        figsize: размер фигуры
    """
    import matplotlib.pyplot as plt

    if channels is None:
        channels = list(range(NUM_RAW))
    n_ch = len(channels)

    fig, axes = plt.subplots(n_ch, 1, figsize=figsize, sharex=True)
    if n_ch == 1:
        axes = [axes]

    T = raw.shape[0]
    t_axis = np.arange(T)

    for i, ch in enumerate(channels):
        ax = axes[i]
        ch_name = RAW_CHANNELS[ch] if ch < len(RAW_CHANNELS) else f'Ch{ch}'

        # 1. Оригинальный сигнал
        ax.plot(t_axis, raw[:, ch], color='black', alpha=0.6,
                linewidth=0.8, label='Оригинал')

        # 2. Идеальная реконструкция (FFT из raw → h1-H → IFFT)
        ideal = reconstruct_ideal_from_raw(
            raw[:, ch], stride, warmup, num_harmonics,
        )
        ax.plot(t_axis, ideal, color='blue', alpha=0.7,
                linewidth=0.8, linestyle='--', label=f'Идеал (h1-{num_harmonics})')

        # 3. Реконструкция из polar-признаков модели
        if model_polar is not None:
            model_recon = reconstruct_channel_from_polar(
                model_polar, ch, stride, warmup,
                num_harmonics, num_sub_periods, T,
            )
            ax.plot(t_axis[:len(model_recon)], model_recon[:T],
                    color='red', alpha=0.7, linewidth=0.8,
                    label='SSL модель')

        ax.set_ylabel(ch_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Отсчёты', fontsize=10)
    fig.suptitle(f'{title_prefix}Реконструкция сигнала из спектра', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Сохранено: {save_path}")
    else:
        plt.show()


def plot_harmonic_comparison(
    input_polar: np.ndarray,
    model_polar: np.ndarray,
    ch: int,
    num_harmonics: int = 9,
    num_sub_periods: int = 4,
    save_path: Optional[str] = None,
    title_prefix: str = '',
):
    """Сравнение амплитуд гармоник: вход vs выход SSL модели.

    Средние амплитуды по всем временным шагам для каждой гармоники.
    """
    import matplotlib.pyplot as plt

    h_in, _ = polar_to_complex(input_polar, ch, num_harmonics, num_sub_periods)
    h_out, _ = polar_to_complex(model_polar, ch, num_harmonics, num_sub_periods)

    amp_in = np.mean(np.abs(h_in), axis=0)
    amp_out = np.mean(np.abs(h_out), axis=0)

    ch_name = RAW_CHANNELS[ch] if ch < len(RAW_CHANNELS) else f'Ch{ch}'
    x = np.arange(1, num_harmonics + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.35
    ax.bar(x - w / 2, amp_in, w, label='Вход (ground truth)', color='steelblue')
    ax.bar(x + w / 2, amp_out, w, label='SSL модель', color='salmon')
    ax.set_xlabel('Гармоника')
    ax.set_ylabel('Средняя амплитуда')
    ax.set_title(f'{title_prefix}{ch_name}: Амплитуды гармоник')
    ax.legend()
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Сохранено: {save_path}")
    else:
        plt.show()
