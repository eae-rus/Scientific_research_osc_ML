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


class TestNormalizationLogic:
    """Detailed tests for normalization logic in CreateNormOsc."""

    @pytest.fixture
    def norm_instance(self):
        with patch('os.listdir', return_value=[]), \
             patch('os.path.exists', return_value=False):
            from osc_tools.features.normalization import CreateNormOsc
            return CreateNormOsc(osc_path='dummy_path')

    def test_determine_voltage_status_noise(self, norm_instance):
        """Test voltage status for noise level."""
        # m1 <= VOLTAGE_T1_S (20*sqrt(2) approx 28.28)
        # m1 <= NOISE_FACTOR * mx (1.5 * mx)
        m1 = 10.0
        mx = 8.0
        h1_df = pd.DataFrame({'U1': [10.0]})
        coef_p_s = {'U1': 1.0}
        
        ps, base = norm_instance._determine_voltage_status(m1, mx, h1_df, ['U1'], coef_p_s)
        assert ps == 's'
        assert base == 'Noise'

    def test_determine_voltage_status_primary(self, norm_instance):
        """Test voltage status for primary values (?1)."""
        # m1 <= VOLTAGE_T1_S
        # m1 > NOISE_FACTOR * mx
        # max_primary > VOLTAGE_T1_P (500*sqrt(2) approx 707)
        m1 = 10.0
        mx = 2.0
        h1_df = pd.DataFrame({'U1': [10.0]})
        coef_p_s = {'U1': 100.0} # primary = 1000 > 707
        
        ps, base = norm_instance._determine_voltage_status(m1, mx, h1_df, ['U1'], coef_p_s)
        assert ps == 'p'
        assert base == '?1'

    def test_determine_current_status_nominal_5(self, norm_instance):
        """Test current status for 5A nominal."""
        # CURRENT_T1_S < m1 <= CURRENT_T2_S (0.03*sqrt(2) < m1 <= 30*sqrt(2))
        # m1 > NOISE_FACTOR * mx
        m1 = 5.0
        mx = 1.0
        h1_df = pd.DataFrame({'I1': [5.0]})
        coef_p_s = {'I1': 1.0}
        
        ps, base = norm_instance._determine_current_status(m1, mx, h1_df, ['I1'], coef_p_s)
        assert ps == 's'
        assert base == 5

    def test_determine_residual_current_noise(self, norm_instance):
        """Test residual current status for noise."""
        # m1 <= RESIDUAL_CURRENT_T1_S and mx <= RESIDUAL_CURRENT_T1_S
        m1 = 0.01
        mx = 0.01
        ps, base = norm_instance._determine_residual_current_status(m1, mx)
        assert ps == 's'
        assert base == 'Noise'

    def test_analyze_basic(self, norm_instance):
        """Test analyze method with synthetic data."""
        h1_df = pd.DataFrame({
            'U | BusBar-1 | phase: A': [100.0],
            'I | Bus-1 | phase: A': [5.0],
            'I | Bus-1 | phase: N': [0.1]
        })
        hx_df = pd.DataFrame({
            'U | BusBar-1 | phase: A': [1.0],
            'I | Bus-1 | phase: A': [0.1],
            'I | Bus-1 | phase: N': [0.01]
        })
        coef_p_s = {
            'U | BusBar-1 | phase: A': 1.0,
            'I | Bus-1 | phase: A': 1.0,
            'I | Bus-1 | phase: N': 1.0
        }
        
        result = norm_instance.analyze('test_file.cfg', h1_df, hx_df, coef_p_s)
        
        assert isinstance(result, pd.DataFrame)
        assert result.at[0, 'name'] == 'test_file'
        assert '1Ub_PS' in result.columns
        assert '1Ip_PS' in result.columns
        assert '1Iz_PS' in result.columns

    def test_merge_normalization_files_basic(self):
        """Test merge_normalization_files utility."""
        from osc_tools.features.normalization import CreateNormOsc
        import shutil
        
        # Use a local temp dir to avoid PermissionError on Windows
        local_tmp = PROJECT_ROOT / "tests" / "unit" / "tmp_merge_test"
        if local_tmp.exists():
            shutil.rmtree(local_tmp)
        local_tmp.mkdir(parents=True)
        
        try:
            # Create dummy CSV files
            df1 = pd.DataFrame({'name': ['file1'], 'norm': ['YES'], '1Ub_PS': ['s']})
            df2 = pd.DataFrame({'name': ['file2'], 'norm': ['NO'], '1Ub_PS': ['p']})
            
            csv1 = local_tmp / "norm_1.csv"
            csv2 = local_tmp / "norm_2.csv"
            df1.to_csv(csv1, index=False)
            df2.to_csv(csv2, index=False)
            
            output_csv = local_tmp / "merged.csv"
            
            CreateNormOsc.merge_normalization_files(
                input_paths_or_folder=str(local_tmp),
                output_csv_path=str(output_csv),
                file_pattern=r"norm_.*\.csv"
            )
            
            assert os.path.exists(output_csv)
            merged_df = pd.read_csv(output_csv)
            assert len(merged_df) == 2
            assert 'file1' in merged_df['name'].values
            assert 'file2' in merged_df['name'].values
        finally:
            if local_tmp.exists():
                shutil.rmtree(local_tmp)

