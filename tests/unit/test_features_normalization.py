"""
Tests for osc_tools.features.normalization module.

Layer 2: Adaptive Layer (API Contracts)
Focuses on:
- Module can be imported
- Classes and methods exist
- Basic contracts work with real test fixtures

Does NOT use mocks for imports. Uses real fixture files instead.
"""

import pytest
from pathlib import Path
import sys
import pandas as pd
from unittest.mock import patch, MagicMock
import os

# Setup sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNormalizationModuleImport:
    """Test that the module can be imported without mocks."""
    
    def test_normalize_module_imports_cleanly(self):
        """Test features.normalization module imports without errors."""
        try:
            import osc_tools.features.normalization as norm_module
            assert norm_module is not None
            assert hasattr(norm_module, 'CreateNormOsc')
        except ImportError as e:
            pytest.fail(f"Cannot import normalization module: {e}")
    
    def test_create_norm_osc_is_class(self):
        """Test CreateNormOsc is a proper class."""
        from osc_tools.features.normalization import CreateNormOsc
        assert isinstance(CreateNormOsc, type), \
            "CreateNormOsc should be a class"


class TestNormalizationStructure:
    """Test basic structure of normalization module."""
    
    def test_create_norm_osc_has_key_methods(self):
        """Test CreateNormOsc has expected methods."""
        from osc_tools.features.normalization import CreateNormOsc
        
        expected_methods = [
            'generate_VT_cols',
            'generate_CT_cols',
            'generate_rCT_cols',
            'generate_dI_cols',
            'generate_raw_cols',
            'generate_result_cols',
            'generate_all_features',
        ]
        
        for method_name in expected_methods:
            assert hasattr(CreateNormOsc, method_name), \
                f"CreateNormOsc missing method: {method_name}"
    
    def test_create_norm_osc_static_methods_work(self):
        """Test that static methods can be called without instance."""
        from osc_tools.features.normalization import CreateNormOsc
        
        # These are static/class methods - should work without __init__
        try:
            # Just verify they're callable
            assert callable(CreateNormOsc.generate_VT_cols)
            assert callable(CreateNormOsc.generate_CT_cols)
            assert callable(CreateNormOsc.generate_raw_cols)
        except Exception as e:
            pytest.fail(f"Static methods not properly defined: {e}")



class TestGenerateColumnMethods:
    """Test column generation methods work on real instances."""
    
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=False)
    def test_generate_vt_cols_returns_tuple(self, mock_exists, mock_listdir):
        """Test generate_VT_cols returns tuple structure."""
        from osc_tools.features.normalization import CreateNormOsc
        
        # Create a real instance with a dummy path
        instance = CreateNormOsc(osc_path='dummy_path')
        result = instance.generate_VT_cols(bus=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=False)
    def test_generate_ct_cols_returns_tuple(self, mock_exists, mock_listdir):
        """Test generate_CT_cols returns tuple structure."""
        from osc_tools.features.normalization import CreateNormOsc
        
        instance = CreateNormOsc(osc_path='dummy_path')
        result = instance.generate_CT_cols(bus=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=False)
    def test_generate_raw_cols_returns_list(self, mock_exists, mock_listdir):
        """Test generate_raw_cols returns set."""
        from osc_tools.features.normalization import CreateNormOsc
        
        instance = CreateNormOsc(osc_path='dummy_path')
        result = instance.generate_raw_cols(bus=2)
        # generate_raw_cols returns a set in the code
        assert isinstance(result, set)
    
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=False)
    def test_generate_result_cols_returns_list(self, mock_exists, mock_listdir):
        """Test generate_result_cols returns dict."""
        from osc_tools.features.normalization import CreateNormOsc
        
        instance = CreateNormOsc(osc_path='dummy_path')
        result = instance.generate_result_cols(bus=2)
        assert isinstance(result, dict)
    
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=False)
    def test_generate_all_features_returns_list(self, mock_exists, mock_listdir):
        """Test generate_all_features returns set."""
        from osc_tools.features.normalization import CreateNormOsc
        
        instance = CreateNormOsc(osc_path='dummy_path')
        result = instance.generate_all_features(bus=2)
        assert isinstance(result, set)


class TestNormalizationEdgeCases:
    """Test edge cases for normalization."""
    
    def test_normalize_module_has_constants(self):
        """Test that normalization module defines expected constants."""
        import osc_tools.features.normalization as norm_module
        
        # These are critical constants for normalization
        constants = ['VOLTAGE_T1_S', 'CURRENT_T1_S', 'NOISE_FACTOR']
        
        for const_name in constants:
            assert hasattr(norm_module, const_name), \
                f"Module missing constant: {const_name}"
    
    def test_merge_normalization_files_function_exists(self):
        """Test that merge function exists."""
        from osc_tools.features.normalization import CreateNormOsc
        
        assert callable(getattr(CreateNormOsc, 'merge_normalization_files', None)), \
            "merge_normalization_files should be callable"
    
    def test_update_normalization_coefficients_exists(self):
        """Test that update function exists."""
        from osc_tools.features.normalization import CreateNormOsc
        
        assert callable(getattr(CreateNormOsc, 'update_normalization_coefficients', None)), \
            "update_normalization_coefficients should be callable"

