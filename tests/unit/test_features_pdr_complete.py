"""
Comprehensive Layer 1 tests for osc_tools.features.pdr_calculator module.

Tests PDR (расстояние до места повреждения) calculations using real functions
that actually exist in the module (sliding_window_fft, calculate_symmetrical_components, etc).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPDRCalculatorImport:
    """Test that PDR calculator module imports correctly."""
    
    def test_pdr_calculator_module_imports(self):
        """Test pdr_calculator module can be imported."""
        try:
            from osc_tools.features import pdr_calculator
            assert pdr_calculator is not None
        except ImportError as e:
            pytest.fail(f"Cannot import pdr_calculator: {e}")
    
    def test_key_pdr_functions_exist(self):
        """Test that key PDR calculation functions exist."""
        from osc_tools.features.pdr_calculator import (
            sliding_window_fft,
            calculate_symmetrical_components,
            calculate_impedance,
            calculate_power,
        )
        
        assert callable(sliding_window_fft)
        assert callable(calculate_symmetrical_components)
        assert callable(calculate_impedance)
        assert callable(calculate_power)


class TestSlidingWindowFFT:
    """Test sliding_window_fft function."""
    
    def test_sliding_window_fft_callable(self):
        """Test function is callable."""
        from osc_tools.features.pdr_calculator import sliding_window_fft
        assert callable(sliding_window_fft)
    
    def test_sliding_window_fft_basic(self):
        """Test basic FFT windowing on sine wave."""
        from osc_tools.features.pdr_calculator import sliding_window_fft
        
        fs = 1600
        f = 50
        t = np.arange(0, 1, 1/fs)
        signal = np.sin(2*np.pi*f*t)
        
        result = sliding_window_fft(signal, window_size=160, num_harmonics=5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestCalculateImpedance:
    """Test impedance calculation functions."""
    
    def test_calculate_impedance_callable(self):
        """Test function is callable."""
        from osc_tools.features.pdr_calculator import calculate_impedance
        assert callable(calculate_impedance)
    
    def test_calculate_impedance_with_arrays(self):
        """Test impedance calculation with proper arrays."""
        from osc_tools.features.pdr_calculator import calculate_impedance
        
        # Create arrays (not scalars)
        voltage = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        
        result = calculate_impedance(voltage, current)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(voltage)
        assert np.all(np.isfinite(result))


class TestCalculatePower:
    """Test calculate_power function."""
    
    def test_calculate_power_callable(self):
        """Test function is callable."""
        from osc_tools.features.pdr_calculator import calculate_power
        assert callable(calculate_power)
    
    def test_calculate_power_returns_tuple(self):
        """Test power calculation returns three components."""
        from osc_tools.features.pdr_calculator import calculate_power
        
        voltage = np.array([1.0, 2.0, 3.0])
        current = np.array([0.5, 1.0, 1.5])
        
        p, q, s = calculate_power(voltage, current)
        
        assert isinstance(p, np.ndarray)
        assert isinstance(q, np.ndarray)
        assert isinstance(s, np.ndarray)


class TestSymmetricalComponents:
    """Test calculate_symmetrical_components function."""
    
    def test_symmetrical_components_callable(self):
        """Test function is callable."""
        from osc_tools.features.pdr_calculator import calculate_symmetrical_components
        assert callable(calculate_symmetrical_components)
    
    def test_symmetrical_components_balanced(self):
        """Test with balanced three-phase system."""
        from osc_tools.features.pdr_calculator import calculate_symmetrical_components
        
        # Balanced system
        pa = np.array([1.0, 0.866j])
        pb = np.array([1.0 * np.exp(-2j*np.pi/3), 0.866j * np.exp(-2j*np.pi/3)])
        pc = np.array([1.0 * np.exp(2j*np.pi/3), 0.866j * np.exp(2j*np.pi/3)])
        
        result = calculate_symmetrical_components(pa, pb, pc)
        
        assert isinstance(result, tuple)
        assert len(result) == 3

