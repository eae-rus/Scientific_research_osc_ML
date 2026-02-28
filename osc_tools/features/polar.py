import numpy as np
from typing import Tuple, Optional


def calculate_polar_features_multiharmonic(
    complex_features: np.ndarray,
    ref_signal: Optional[np.ndarray] = None,
    ref_mag_threshold: float = 1e-4
) -> np.ndarray:
    """
    Преобразует комплексные фазоры в полярные координаты с ПОГАРМОНИЧНОЙ коррекцией угла.
    
    DFT-фазор k-й гармоники вращается со скоростью k*ω₀. Для корректного вычисления
    относительного угла необходимо вычитать угол СООТВЕТСТВУЮЩЕЙ k-й гармоники
    опорного сигнала, а не только 1-й.
    
    Алгоритм для k-й гармоники (k=1,2,...):
    1. Если амплитуда k-й гармоники опорного сигнала > порога:
       relative_angle = angle(X_k) - angle(ref_k)  →  СТАБИЛЬНО
    2. Иначе (опорный фазор слишком мал для данной гармоники):
       relative_angle = angle(X_k) - k * angle(ref_h1)  →  СТАБИЛЬНО
       (Синтетический опорный угол: k-кратный угол 1-й гармоники компенсирует вращение)
    
    Args:
        complex_features: (Time, n_channels, n_harmonics) комплексные фазоры.
        ref_signal: (Time, n_harmonics) опорный сигнал (обычно UA или IA).
                    Если None, используются абсолютные углы (без коррекции).
        ref_mag_threshold: Порог амплитуды k-й гармоники опорного сигнала для
                          переключения на синтетический опорный угол (k * angle_h1).
    
    Returns:
        (Time, n_channels * n_harmonics * 2) — чередование [Mag, Angle] 
        в порядке: [ch0_h1_mag, ch0_h1_angle, ch0_h2_mag, ch0_h2_angle, ..., chN_hH_angle].
    """
    time_steps, n_channels, n_harmonics = complex_features.shape
    
    magnitudes = np.abs(complex_features)   # (T, C, H)
    angles = np.angle(complex_features)     # (T, C, H)
    
    if ref_signal is not None:
        ref_angles_all = np.angle(ref_signal)       # (T, H)
        ref_h1_angle = ref_angles_all[:, 0]         # (T,) — угол 1-й гармоники
        
        for h in range(n_harmonics):
            k = h + 1  # Номер гармоники (1-based)
            h_ref_mag = np.nanmean(np.abs(ref_signal[:, h]))
            
            if h_ref_mag > ref_mag_threshold:
                # Вычитаем угол k-й гармоники опорного сигнала
                angles[:, :, h] -= ref_angles_all[:, h][:, np.newaxis]
            else:
                # Опорный фазор для k-й гармоники слишком мал —
                # используем синтетический: k * angle(ref_h1)
                # Это корректно компенсирует вращение DFT-фазора,
                # т.к. wrap(α - k*wrap(θ)) = wrap(α - k*θ)
                angles[:, :, h] -= (k * ref_h1_angle)[:, np.newaxis]
        
        # Нормализация углов в [-π, π]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
    
    # Сборка результата в порядке: [ch0_h0_mag, ch0_h0_angle, ch0_h1_mag, ...]
    result = np.empty((time_steps, n_channels * n_harmonics * 2), dtype=np.float32)
    for ch in range(n_channels):
        for h in range(n_harmonics):
            idx = (ch * n_harmonics + h) * 2
            result[:, idx] = magnitudes[:, ch, h]
            result[:, idx + 1] = angles[:, ch, h]
    
    return result
