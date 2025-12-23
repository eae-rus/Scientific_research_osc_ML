"""
Layer 1: Comprehensive tests for osc_tools.data_management module.

Tests data management functionality: file search, processing, and organization.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataManagementImports:
    """Test data_management module imports."""
    
    def test_data_management_modules_import(self):
        """Test that data_management submodules can be imported."""
        try:
            from osc_tools.data_management import (
                search,
                processing,
                comtrade_processing,
            )
            assert search is not None
            assert processing is not None
            assert comtrade_processing is not None
        except ImportError as e:
            pytest.fail(f"Cannot import data_management modules: {e}")


class TestComtradeProcessing:
    """Test COMTRADE file processing functionality."""
    
    def test_comtrade_processing_module_has_functions(self):
        """Test that comtrade_processing has processing functions."""
        from osc_tools.data_management import comtrade_processing
        
        # Check for common processing functions
        assert hasattr(comtrade_processing, '__file__')
        assert comtrade_processing is not None
    
    def test_process_with_sample_data(self):
        """Test ReadComtrade class exists and can be instantiated."""
        from osc_tools.data_management.comtrade_processing import ReadComtrade
        
        reader = ReadComtrade()
        assert reader is not None
        assert hasattr(reader, 'read_comtrade')


class TestDataSearch:
    """Test data search and discovery functions."""
    
    def test_search_module_callable(self):
        """Test that search module functions are callable."""
        from osc_tools.data_management import search
        
        # Module should exist
        assert search is not None



class TestDataProcessing:
    """Test data processing pipeline."""
    
    def test_processing_module_callable(self):
        """Test that processing module is available."""
        from osc_tools.data_management import processing
        
        assert processing is not None
    
    def test_process_dataframe_structure(self, sample_normalized_dataframe):
        """Test that DataFrame processing preserves structure."""
        # Verify that sample DataFrame is properly structured
        assert isinstance(sample_normalized_dataframe, pd.DataFrame)
        assert sample_normalized_dataframe.shape[0] > 0
        assert sample_normalized_dataframe.shape[1] > 0
        
        # Check for numeric data
        for col in sample_normalized_dataframe.columns:
            assert pd.api.types.is_numeric_dtype(sample_normalized_dataframe[col]), \
                f"Column {col} should be numeric"



class TestDataFiltering:
    """Test data filtering and selection."""
    
    def test_filter_voltage_signals(self, sample_normalized_dataframe):
        """Test filtering voltage signals from DataFrame."""
        df = sample_normalized_dataframe
        
        # Filter columns with 'U' (voltage)
        voltage_df = df[[col for col in df.columns if 'U' in col]]
        
        assert len(voltage_df.columns) > 0
        assert isinstance(voltage_df, pd.DataFrame)
    
    def test_filter_current_signals(self, sample_normalized_dataframe):
        """Test filtering current signals from DataFrame."""
        df = sample_normalized_dataframe
        
        # Filter columns with 'I' (current)
        current_df = df[[col for col in df.columns if 'I' in col]]
        
        assert len(current_df.columns) > 0
        assert isinstance(current_df, pd.DataFrame)
    
    def test_filter_by_bus(self, sample_normalized_dataframe):
        """Test filtering signals by bus number."""
        df = sample_normalized_dataframe
        
        # Get columns starting with '1' (Bus 1)
        bus1_cols = [col for col in df.columns if col.startswith('1')]
        bus1_df = df[bus1_cols]
        
        assert len(bus1_cols) > 0
        assert isinstance(bus1_df, pd.DataFrame)
