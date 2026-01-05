import numpy as np
from typing import Tuple, Optional

def calculate_polar_features(
    phasors: np.ndarray, 
    reference_phasor: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Преобразует комплексные фазоры в (Модуль, Относительный Угол).
    
    Args:
        phasors: Массив комплексных чисел (Time, Channels) или (Channels,).
        reference_phasor: Опорный фазор для расчета относительного угла.
                          Если None, используется угол самого фазора (абсолютный угол).
                          Обычно это напряжение фазы А.
                          
    Returns:
        features: Массив (Time, Channels * 2) где чередуются [Mag, Angle, Mag, Angle...].
    """
    magnitude = np.abs(phasors)
    angle = np.angle(phasors)
    
    if reference_phasor is not None:
        ref_angle = np.angle(reference_phasor)
        # Если reference_phasor - массив той же длины, что и phasors (по времени)
        if ref_angle.shape == angle.shape:
             angle = angle - ref_angle
        # Если reference_phasor - одномерный (для каждого момента времени один угол), а phasors - (Time, Channels)
        elif ref_angle.ndim == 1 and angle.ndim == 2:
             angle = angle - ref_angle[:, None]
        else:
             # Fallback broadcasting
             angle = angle - ref_angle
             
    # Нормализация угла в диапазон [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    
    # Объединение Magnitude и Angle
    # (Time, Channels) -> (Time, Channels, 2) -> (Time, Channels * 2)
    if phasors.ndim == 2:
        time_steps, channels = phasors.shape
        features = np.empty((time_steps, channels * 2), dtype=np.float32)
        features[:, 0::2] = magnitude
        features[:, 1::2] = angle
    else:
        # 1D case
        channels = len(phasors)
        features = np.empty(channels * 2, dtype=np.float32)
        features[0::2] = magnitude
        features[1::2] = angle
        
    return features
