from __future__ import annotations

import torch


def fft_calc(input_tensor: torch.Tensor, count_harmonic: int = 1):
    """Возвращает спектры предаварийного и аварийного окон."""
    prev_signals = torch.fft.rfft(input_tensor[:, :, :32])
    current_signals = torch.fft.rfft(input_tensor[:, :, 32:])
    return prev_signals[:, :, : count_harmonic + 1], current_signals[:, :, : count_harmonic + 1]


def fft_calc_abs_angle(input_tensor: torch.Tensor, count_harmonic: int = 1):
    """Возвращает амплитуды и углы для предаварийного/аварийного окон."""
    prev_signals = torch.fft.rfft(input_tensor[:, :, :32])[:, :, : count_harmonic + 1]
    current_signals = torch.fft.rfft(input_tensor[:, :, 32:])[:, :, : count_harmonic + 1]
    return (
        torch.abs(prev_signals),
        torch.abs(current_signals),
        torch.angle(prev_signals),
        torch.angle(current_signals),
    )


def create_signal_group(
    x: torch.Tensor,
    currents: list[int],
    voltages: list[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Формирует группу комплексных сигналов ток+напряжение."""
    if device is None:
        device = x.device
    group = torch.zeros(x.size(0), len(currents), x.size(2), device=device, dtype=torch.cfloat)
    for i, (ic, iv) in enumerate(zip(currents, voltages)):
        if ic != -1 and iv != -1:
            group[:, i, :] = x[:, ic, :] + 1j * x[:, iv, :]
        elif ic == -1:
            group[:, i, :] = 1j * x[:, iv, :]
        else:
            group[:, i, :] = x[:, ic, :]
    return group


def create_line_group(
    x: torch.Tensor,
    ic_l: list[tuple[int, int]],
    voltages: list[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Формирует линейные комплексные сигналы по разностям токов и напряжениям."""
    if device is None:
        device = x.device
    group = torch.zeros(x.size(0), len(ic_l), x.size(2), device=device, dtype=torch.cfloat)
    for i, (ic_pair, iv) in enumerate(zip(ic_l, voltages)):
        group[:, i, :] = x[:, ic_pair[0], :] - x[:, ic_pair[1], :] + 1j * x[:, iv, :]
    return group
