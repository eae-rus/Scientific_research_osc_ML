import unittest
import os
import sys
import shutil
import json
import pandas as pd
import numpy as np # For test_check_activity_norm_disabled_in_config if creating data on the fly

# Add project root to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer
from analysis.activity_filter import OscillogramActivityFilter, ChannelType

BASE_TEST_ROOT_DIR = os.path.dirname(__file__)
SAMPLE_DATA_ROOT = os.path.join(BASE_TEST_ROOT_DIR, "sample_data")
COMTRADE_SAMPLES_PATH = os.path.join(SAMPLE_DATA_ROOT, "comtrade_files", "for_activity_filter")
CONFIG_SAMPLES_PATH = os.path.join(SAMPLE_DATA_ROOT, "config_files")

class TestOscillogramActivityFilter(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.join(BASE_TEST_ROOT_DIR, "temp_activity_filter_data")
        self.source_dir = os.path.join(self.base_temp_dir, "source_af")

        if not os.path.exists(SAMPLE_DATA_ROOT):
            self.fail(f"Sample data root directory not found at {SAMPLE_DATA_ROOT}. Run test_data_setup.py.")
        if not os.path.exists(COMTRADE_SAMPLES_PATH):
            self.fail(f"Sample COMTRADE directory for activity filter tests not found: {COMTRADE_SAMPLES_PATH}")
        if not os.path.exists(CONFIG_SAMPLES_PATH):
            self.fail(f"Sample config files directory not found: {CONFIG_SAMPLES_PATH}")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.source_dir, exist_ok=True)

        # Load main config for the filter
        activity_config_path = os.path.join(CONFIG_SAMPLES_PATH, "activity_filter_config.json")
        self.assertTrue(os.path.exists(activity_config_path), f"Activity filter config not found: {activity_config_path}")
        with open(activity_config_path, 'r') as f:
            self.config_dict = json.load(f)

        # Setup normalizer
        norm_coeffs_path = os.path.join(CONFIG_SAMPLES_PATH, "norm_coeffs_for_filter_test.csv")
        self.assertTrue(os.path.exists(norm_coeffs_path), f"Norm coeffs for filter test not found: {norm_coeffs_path}")
        self.normalizer = OscillogramNormalizer(norm_coef_file_path=norm_coeffs_path, is_print_message=False)

        # Filters for testing
        self.filter_with_norm = OscillogramActivityFilter(config=self.config_dict, normalizer=self.normalizer, show_progress_bars=False)

        config_raw = self.config_dict.copy()
        config_raw['use_norm_osc'] = False # Force raw path for this filter instance
        self.filter_raw_only = OscillogramActivityFilter(config=config_raw, normalizer=None, show_progress_bars=False)

        self.is_print_message_tests = False # For direct calls if needed, though filter verbose is from its config

        # Copy sample files to temp source directory
        sample_files_to_copy = [
            "osc_af_active.cfg", "osc_af_active.dat",
            "osc_af_empty_noise.cfg", "osc_af_empty_noise.dat",
            "osc_af_empty_stable.cfg", "osc_af_empty_stable.dat",
            "osc_af_active_fornorm.cfg", "osc_af_active_fornorm.dat"
        ]
        for fname in sample_files_to_copy:
            src = os.path.join(COMTRADE_SAMPLES_PATH, fname)
            dest = os.path.join(self.source_dir, fname)
            self.assertTrue(os.path.exists(src), f"Required sample file not found: {src}")
            shutil.copy2(src, dest)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_check_activity_raw_active(self):
        cfg_path = os.path.join(self.source_dir, "osc_af_active.cfg")
        osc_obj = Oscillogram(cfg_path)
        # This filter instance is configured with use_norm_osc=False
        is_active = self.filter_raw_only.check_activity(osc_obj)
        self.assertTrue(is_active, "osc_af_active should be detected as active by raw filter.")

    def test_check_activity_raw_empty_noise(self):
        cfg_path = os.path.join(self.source_dir, "osc_af_empty_noise.cfg")
        osc_obj = Oscillogram(cfg_path)
        is_active = self.filter_raw_only.check_activity(osc_obj)
        self.assertFalse(is_active, "osc_af_empty_noise should be detected as inactive by raw filter.")

    def test_check_activity_raw_empty_stable(self):
        cfg_path = os.path.join(self.source_dir, "osc_af_empty_stable.cfg")
        osc_obj = Oscillogram(cfg_path)
        is_active = self.filter_raw_only.check_activity(osc_obj)
        self.assertFalse(is_active, "osc_af_empty_stable should be detected as inactive by raw filter.")

    def test_check_activity_normalized_active(self):
        cfg_path = os.path.join(self.source_dir, "osc_af_active_fornorm.cfg")
        osc_obj = Oscillogram(cfg_path)
        # This filter instance is configured with use_norm_osc=True and has a normalizer
        is_active = self.filter_with_norm.check_activity(osc_obj)
        self.assertTrue(is_active, "osc_af_active_fornorm should be detected as active by filter with normalization.")

    def test_check_activity_norm_disabled_in_config_explicitly(self):
        config_no_norm = self.config_dict.copy()
        config_no_norm['use_norm_osc'] = False # Explicitly disable normalization in config

        filter_temp = OscillogramActivityFilter(config=config_no_norm, normalizer=self.normalizer, show_progress_bars=False)

        cfg_path = os.path.join(self.source_dir, "osc_af_active_fornorm.cfg")
        osc_obj = Oscillogram(cfg_path)

        # Since use_norm_osc is False, it should take the raw path.
        # osc_af_active_fornorm has high amplitude raw signals.
        # Whether it's active depends on raw_signal_analysis thresholds vs its H1 characteristics.
        # Assuming its raw H1 characteristics (after relativization if clean) meet raw thresholds.
        # The sample data for osc_af_active_fornorm has large values (2000-10000).
        # Initial H1 for this will be large. If it passes clean check, h1_for_relative_norm_peak will be large.
        # Then h1_to_analyze will be h1_series_rms / (h1_for_relative_norm_peak / sqrt(2)).
        # This makes h1_to_analyze values small (around 1.0 if signal is stable H1).
        # The thresholds_raw_voltage_relative are delta:0.1, std_dev:0.05.
        # The active_for_norm_data has a step change, so delta should be significant.
        is_active = filter_temp.check_activity(osc_obj)
        self.assertTrue(is_active, "osc_af_active_fornorm should be active via raw path if norm is disabled in config.")


    def test_filter_directory(self):
        output_csv = os.path.join(self.base_temp_dir, "active_report.csv")
        # filter_with_norm is configured with use_norm_osc = True
        self.filter_with_norm.filter_directory(self.source_dir, output_csv)

        self.assertTrue(os.path.exists(output_csv))
        df = pd.read_csv(output_csv)
        self.assertListEqual(list(df.columns), ['active_files'])

        active_files_found = set(df['active_files'].tolist())

        # Expected active files:
        # - osc_af_active.cfg (should pass raw analysis path as normalizer might not have its entry or norm=NO)
        #   Actually, if normalizer is provided, it will attempt normalization.
        #   If "osc_af_active" is in norm_coeffs_for_filter_test.csv and norm="YES", it uses normalized path.
        #   If not, it uses raw path. Let's assume it's in norm_coeffs_for_filter_test.csv.
        # - osc_af_active_fornorm.cfg (should pass normalized analysis path)
        expected_active = {"osc_af_active.cfg", "osc_af_active_fornorm.cfg"}

        self.assertSetEqual(active_files_found, expected_active)
        self.assertNotIn("osc_af_empty_noise.cfg", active_files_found)
        self.assertNotIn("osc_af_empty_stable.cfg", active_files_found)

if __name__ == '__main__':
    if not os.path.exists(SAMPLE_DATA_ROOT):
        print(f"Warning: Test sample data directory not found at {SAMPLE_DATA_ROOT}")
        print(f"Please run 'python tests/test_data_setup.py' from the project root to generate test data first.")
    unittest.main()
