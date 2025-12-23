"""
Tests for osc_tools.features.normalization module.

Layer 2: Adaptive Layer (API Contracts)
Focuses on smoke tests only - verify module loads and basic structures work.
Do NOT test full functionality due to complex initialization requirements.
"""

import pytest
from pathlib import Path
import sys

# Setup sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNormalizationModuleImport:
    """Test that the module can be imported."""
    
    def test_normalize_module_can_import(self):
        """Test features.normalization module can be imported."""
        try:
            from osc_tools.features.normalization import CreateNormOsc
            assert CreateNormOsc is not None
        except ImportError:
            pytest.fail("Cannot import CreateNormOsc from osc_tools.features.normalization")
    
    def test_normalize_constants_defined(self):
        """Test that normalization constants are defined."""
        from osc_tools.features import normalization
        assert hasattr(normalization, 'VOLTAGE_T1_S')
        assert hasattr(normalization, 'CURRENT_T1_S')
        assert hasattr(normalization, 'NOISE_FACTOR')


class TestNormalizationStructure:
    """Test basic structure of normalization module."""
    
    def test_create_norm_osc_is_class(self):
        """Test CreateNormOsc is a proper class."""
        from osc_tools.features.normalization import CreateNormOsc
        assert isinstance(CreateNormOsc, type)
    
    def test_create_norm_osc_has_key_methods(self):
        """Test CreateNormOsc has expected methods."""
        from osc_tools.features.normalization import CreateNormOsc
        
        expected_methods = [
            'generate_VT_cols',
            'generate_CT_cols',
            '_determine_voltage_status',
            '_determine_current_status',
        ]
        
        for method_name in expected_methods:
            assert hasattr(CreateNormOsc, method_name), \
                f"CreateNormOsc missing method: {method_name}"
            # Should handle empty file list
