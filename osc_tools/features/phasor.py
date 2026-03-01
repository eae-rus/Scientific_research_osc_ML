import numpy as np
from typing import Tuple, Optional

def calculate_symmetrical_components(phasor_a: np.ndarray, phasor_b: np.ndarray, phasor_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчет симметричных составляющих (прямая, обратная, нулевая).
    
    Args:
        phasor_a, phasor_b, phasor_c: Комплексные фазоры фазных величин.
        
    Returns:
        phasor_1, phasor_2, phasor_0: Прямая, обратная и нулевая последовательности.
    """
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a * a
    
    phasor_0 = (phasor_a + phasor_b + phasor_c) / 3.0
    phasor_1 = (phasor_a + a * phasor_b + a2 * phasor_c) / 3.0
    phasor_2 = (phasor_a + a2 * phasor_b + a * phasor_c) / 3.0
    
    return phasor_1, phasor_2, phasor_0

def calculate_symmetrical_components_from_line(u_ab: np.ndarray, u_bc: np.ndarray, u_ca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Расчет симметричных составляющих (прямая, обратная) из линейных напряжений.
    Нулевая составляющая (U0) не может быть вычислена из линейных напряжений.
    
    Args:
        u_ab, u_bc, u_ca: Комплексные фазоры линейных напряжений.
        
    Returns:
        u_1, u_2: Прямая и обратная последовательности фазных напряжений.
    """
    # Сначала считаем симметричные составляющие системы линейных векторов
    ul_1, ul_2, _ = calculate_symmetrical_components(u_ab, u_bc, u_ca)
    
    # Пересчет в фазные
    # U_line_1 = U_phase_1 * sqrt(3) * exp(j30)
    # U_phase_1 = U_line_1 / (sqrt(3) * exp(j30))
    
    # U_line_2 = U_phase_2 * sqrt(3) * exp(-j30)
    # U_phase_2 = U_line_2 / (sqrt(3) * exp(-j30))
    
    sqrt3 = np.sqrt(3)
    exp_j30 = np.exp(1j * np.pi / 6)
    exp_minus_j30 = np.exp(-1j * np.pi / 6)
    
    u_1 = ul_1 / (sqrt3 * exp_j30)
    u_2 = ul_2 / (sqrt3 * exp_minus_j30)
    
    return u_1, u_2

def calculate_impedance(voltage: np.ndarray, current: np.ndarray, min_current_threshold: float = 1e-6) -> np.ndarray:
    """
    Расчет комплексного сопротивления Z = V / I.
    
    Args:
        voltage: Комплексное напряжение.
        current: Комплексный ток.
        min_current_threshold: Порог тока для предотвращения деления на ноль.
        
    Returns:
        impedance: Комплексное сопротивление.
    """
    current_safe = current.copy()
    zero_current_mask = np.abs(current_safe) < min_current_threshold
    current_safe[zero_current_mask] = np.nan # Замена на NaN для избежания деления на 0
    impedance = (voltage / current_safe).astype(complex)
    # Защита от слишком малых токов
    impedance[zero_current_mask] = 1/min_current_threshold + (1/min_current_threshold)*1j # Задаём максимальный порог, чтобы исключить nan
    return impedance

def calculate_power(voltage: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчет комплексной (S), активной (P) и реактивной (Q) мощности.
    
    Args:
        voltage: Комплексное напряжение.
        current: Комплексный ток.
        
    Returns:
        s_complex, p_active, q_reactive
    """
    s_complex = voltage * np.conj(current)
    p_active = s_complex.real
    q_reactive = s_complex.imag
    # s_apparent = np.abs(s_complex) # Модуль можно получить позже из s_complex
    return s_complex, p_active, q_reactive

def calculate_linear_voltages(ua: np.ndarray, ub: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчет линейных напряжений из фазных.
    
    Args:
        ua, ub, uc: Фазные напряжения.
        
    Returns:
        uab, ubc, uca: Линейные напряжения.
    """
    uab = ua - ub
    ubc = ub - uc
    uca = uc - ua
    return uab, ubc, uca
