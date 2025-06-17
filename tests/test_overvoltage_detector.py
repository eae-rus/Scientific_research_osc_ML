import unittest
import os
import sys
import shutil
import json
import csv
import pandas as pd
import numpy as np

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer
from raw_to_csv.raw_to_csv import OscillogramToCsvConverter
from analysis.overvoltage_detector import OvervoltageDetector


class TestOvervoltageDetector(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.abspath("tests/temp_overvoltage_data")
        self.sample_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))
        self.sample_data_comtrade_path = os.path.join(self.sample_data_root, "comtrade_files", "for_spef_overvoltage_1")
        self.sample_data_config_path = os.path.join(self.sample_data_root, "config_files")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.source_dir = os.path.join(self.base_temp_dir, "source_ov")
        os.makedirs(self.source_dir, exist_ok=True)

        # Load config
        config_file_path = os.path.join(self.sample_data_config_path, "overvoltage_detector_config.json")
        with open(config_file_path, 'r') as f:
            self.config_dict = json.load(f)

        # Load norm coefficients
        self.norm_coeffs_path = os.path.join(self.sample_data_config_path, "norm_coeffs_for_ov_test.csv")
        self.assertTrue(os.path.exists(self.norm_coeffs_path), f"Norm coeffs file not found: {self.norm_coeffs_path}")
        self.norm_coef_df = pd.read_csv(self.norm_coeffs_path)

        self.normalizer = OscillogramNormalizer(norm_coef_file_path=self.norm_coeffs_path, is_print_message=False)

        # Paths to dict_analog_names and dict_discrete_names
        analog_names_path = os.path.join(self.sample_data_config_path, "dict_analog_names_ov_test.json")
        discrete_names_path = os.path.join(self.sample_data_config_path, "dict_discrete_names_ov_test.json")
        self.assertTrue(os.path.exists(analog_names_path), f"Analog names JSON not found: {analog_names_path}")
        self.assertTrue(os.path.exists(discrete_names_path), f"Discrete names JSON not found: {discrete_names_path}")

        self.bus_splitter = OscillogramToCsvConverter(
            normalizer=self.normalizer,
            dict_analog_names_path=analog_names_path,
            dict_discrete_names_path=discrete_names_path,
            is_print_message=False
        )

        self.detector = OvervoltageDetector(
            config=self.config_dict,
            normalizer=self.normalizer,
            bus_splitter=self.bus_splitter,
            norm_coef_df=self.norm_coef_df,
            show_progress_bars=False
        )

        # Copy sample CFG/DAT files
        sample_files_to_copy = ["spef_present.cfg", "spef_present.dat",
                                "no_spef.cfg", "no_spef.dat",
                                "spef_low_ov.cfg", "spef_low_ov.dat",
                                "spef_symmetrical_phases.cfg", "spef_symmetrical_phases.dat"]
        for file_name in sample_files_to_copy:
            src_file = os.path.join(self.sample_data_comtrade_path, file_name)
            dst_file = os.path.join(self.source_dir, file_name)
            self.assertTrue(os.path.exists(src_file), f"Sample file not found: {src_file}")
            shutil.copy(src_file, dst_file)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def _create_sample_bus_df(self, u0_un_values, ua_vals, ub_vals, uc_vals, prefix="BB"):
        data = {
            f'U0/Un {prefix}': u0_un_values, # This column name is derived from bus_splitter logic
            f'UA {prefix}': ua_vals,
            f'UB {prefix}': ub_vals,
            f'UC {prefix}': uc_vals,
            f'UN {prefix}': np.zeros(len(u0_un_values)) # UN is not directly used in _find_spef_zones or _calc_max_ov
        }
        # Ensure all arrays have the same length
        min_len = min(len(v) for v in data.values())
        for k in data:
            data[k] = data[k][:min_len]

        df = pd.DataFrame(data)
        # Add a time index for realism if needed, though not strictly used by current methods
        df.index = pd.to_timedelta(np.arange(len(df)) * 0.0001, unit='s') # Example time index
        return df

    def test_find_spef_zones_present(self):
        # U0/Un exceeds threshold (0.1) for more than duration_ms (e.g., 1ms from config -> 5 samples if 5kHz)
        # Assuming samples_per_period = 20 (for 50Hz, implies 1kHz sampling rate)
        # Config: "SPEF_THRESHOLD_U0_UN": 0.1, "SPEF_MIN_DURATION_MS": 1.0
        # If sampling rate is 1kHz (1 sample per ms), 1ms = 1 sample.
        # Let's use 20 samples/period for 50Hz, so Fs = 1000 Hz. 1ms = 1 sample.
        # The config actually implies SPEF_MIN_DURATION_SAMPLES = Fs_Hz * SPEF_MIN_DURATION_MS / 1000
        # If Fs_Hz from oscillogram is e.g. 1000Hz, then SPEF_MIN_DURATION_SAMPLES = 1000 * 1.0 / 1000 = 1 sample.
        # Let's make it clearer: 5 samples of SPEF.
        u0_un = [0.05, 0.05, 0.15, 0.20, 0.25, 0.18, 0.12, 0.05, 0.05] # 5 samples > 0.1
        ua = [1.0]*len(u0_un) # Amplitudes don't matter for zone finding unless filter is on
        ub = [1.0]*len(u0_un)
        uc = [1.0]*len(u0_un)
        bus_group_df = self._create_sample_bus_df(u0_un, ua, ub, uc)

        # Detector's config has SPEF_MIN_DURATION_MS: 1.0.
        # Assume oscillogram sample rate is 1000 Hz for this test (1 sample/ms)
        # This means min_duration_samples will be 1.
        # The method _find_spef_zones needs samples_per_period for phase similarity, not directly for duration.
        # The actual duration check uses Fs_Hz from self.detector._current_fs_hz, set during analyze_oscillogram
        # For direct testing _find_spef_zones, we need to mock this or use a typical value.
        # Let's assume Fs_Hz = 1000 for this test.
        self.detector._current_fs_hz = 1000.0 # Mocking this for the test

        zones = self.detector._find_spef_zones(bus_group_df, group_prefix="BB", samples_per_period=20)
        self.assertTrue(len(zones) > 0, "SPEF zones should be detected")
        # Expecting zone from index 2 to 6 (inclusive start, exclusive end for typical slicing,
        # but _find_spef_zones returns (start_index, end_index_inclusive) )
        # The logic is: `if end_index - start_index + 1 >= min_duration_samples:`
        # Here, start_index = 2, end_index = 6. Duration = 6-2+1 = 5 samples.
        # min_duration_samples = 1000 Hz * 1.0 ms / 1000 = 1 sample. 5 >= 1.
        self.assertEqual(zones[0], (2, 6))


    def test_find_spef_zones_similar_amplitude_filter_on(self):
        u0_un = [0.05, 0.15, 0.20, 0.25, 0.18, 0.05] # SPEF condition met
        ua =    [1.0,  1.0,  1.01, 1.0,  0.99, 1.0] # Very similar amplitudes
        ub =    [1.0,  1.0,  1.0,  1.01, 1.0,  1.0]
        uc =    [1.0,  0.99, 1.0,  1.0,  1.01, 1.0]
        bus_group_df = self._create_sample_bus_df(u0_un, ua, ub, uc)

        config_filter_on = self.config_dict.copy()
        config_filter_on['SIMILAR_AMPLITUDES_FILTER_ENABLED'] = True
        # Threshold for similarity: "SIMILAR_AMPLITUDES_THRESHOLD_PU": 0.05
        # Max diff in any zone for UA, UB, UC should be < 0.05 PU
        # In the zone [1,4]: ua_max_diff = 0.01, ub_max_diff = 0.01, uc_max_diff = 0.01. All < 0.05. So it *is* similar.

        detector_temp = OvervoltageDetector(
            config=config_filter_on,
            normalizer=self.normalizer,
            bus_splitter=self.bus_splitter,
            norm_coef_df=self.norm_coef_df
        )
        detector_temp._current_fs_hz = 1000.0 # Mock Fs_Hz

        zones = detector_temp._find_spef_zones(bus_group_df, group_prefix="BB", samples_per_period=20)
        self.assertEqual(len(zones), 0, "SPEF zones should be filtered out due to similar amplitudes")

    def test_find_spef_zones_similar_amplitude_filter_off(self):
        u0_un = [0.05, 0.15, 0.20, 0.25, 0.18, 0.05] # SPEF condition met
        ua =    [1.0,  1.0,  1.01, 1.0,  0.99, 1.0] # Very similar amplitudes
        ub =    [1.0,  1.0,  1.0,  1.01, 1.0,  1.0]
        uc =    [1.0,  0.99, 1.0,  1.0,  1.01, 1.0]
        bus_group_df = self._create_sample_bus_df(u0_un, ua, ub, uc)

        config_filter_off = self.config_dict.copy()
        config_filter_off['SIMILAR_AMPLITUDES_FILTER_ENABLED'] = False

        detector_temp = OvervoltageDetector(
            config=config_filter_off,
            normalizer=self.normalizer,
            bus_splitter=self.bus_splitter,
            norm_coef_df=self.norm_coef_df
        )
        detector_temp._current_fs_hz = 1000.0 # Mock Fs_Hz

        zones = detector_temp._find_spef_zones(bus_group_df, group_prefix="BB", samples_per_period=20)
        self.assertTrue(len(zones) > 0, "SPEF zones should be detected when filter is off")
        self.assertEqual(zones[0], (1, 4)) # Zone from index 1 to 4

    def test_calculate_max_overvoltage_in_zones_valid(self):
        # Peak PU value of 2.5. RMS = 2.5 / sqrt(2)
        ua_peak = 2.5
        ua_vals = [1.0, 1.0, ua_peak, 1.0, 1.0, 1.0] # Max at index 2
        ub_vals = [1.0] * len(ua_vals)
        uc_vals = [1.0] * len(ua_vals)
        u0_un_vals = [0.0] * len(ua_vals) # Not used by this method
        bus_group_df = self._create_sample_bus_df(u0_un_vals, ua_vals, ub_vals, uc_vals)

        zones = [(1, 3)] # Zone includes the peak at index 2. (start_idx, end_idx_inclusive)

        # nominal_voltage_ph_rms is used to calculate physical voltage if data is not PU.
        # Here, data is assumed to be PU, so nominal_voltage_ph_rms = 1.0 for PU output.
        # The method calculates max_pu_excursion = max_val_in_zone / (nominal_voltage_ph_rms * sqrt(2))
        # If nominal_voltage_ph_rms is 1.0 (meaning data is already in PU), then
        # max_pu_excursion = max_peak_in_zone / sqrt(2) which is max RMS in zone.
        ov_factor = self.detector._calculate_max_overvoltage_in_zones(bus_group_df, "BB", zones, 1.0)

        # The method returns max_rms_in_zone / nominal_voltage_ph_rms.
        # If nominal_voltage_ph_rms = 1.0, it returns max_rms_in_zone.
        # Max peak is 2.5. Max RMS is 2.5 / sqrt(2).
        expected_ov_factor = ua_peak / np.sqrt(2)
        self.assertAlmostEqual(ov_factor, expected_ov_factor, places=3)

    def test_analyze_oscillogram_overvoltage_detected(self):
        cfg_path = os.path.join(self.source_dir, "spef_present.cfg")
        dat_path = os.path.join(self.source_dir, "spef_present.dat")
        osc_obj = Oscillogram(cfg_path, dat_path)
        osc_obj.load()

        result = self.detector.analyze_oscillogram(osc_obj)
        self.assertIsNotNone(result, "Overvoltage should be detected in spef_present.cfg")
        if result: # Check to satisfy linter if result can be None
            self.assertGreater(result['overvoltage_factor'], 1.7, "Overvoltage factor should be > 1.7")

    def test_analyze_oscillogram_no_overvoltage(self):
        cfg_path = os.path.join(self.source_dir, "no_spef.cfg")
        dat_path = os.path.join(self.source_dir, "no_spef.dat")
        osc_obj = Oscillogram(cfg_path, dat_path)
        osc_obj.load()

        result = self.detector.analyze_oscillogram(osc_obj)
        self.assertIsNone(result, "No overvoltage should be detected in no_spef.cfg")

    def test_analyze_oscillogram_spef_low_ov(self):
        cfg_path = os.path.join(self.source_dir, "spef_low_ov.cfg")
        dat_path = os.path.join(self.source_dir, "spef_low_ov.dat")
        osc_obj = Oscillogram(cfg_path, dat_path)
        osc_obj.load()

        # Expect SPEF, but overvoltage below threshold from config: "OVERVOLTAGE_THRESHOLD_PU_RMS": 1.7
        result = self.detector.analyze_oscillogram(osc_obj)
        self.assertIsNone(result, "Overvoltage should be None if below threshold in spef_low_ov.cfg")

    def test_analyze_oscillogram_symmetrical_phases_filter_on(self):
        cfg_path = os.path.join(self.source_dir, "spef_symmetrical_phases.cfg")
        dat_path = os.path.join(self.source_dir, "spef_symmetrical_phases.dat")
        osc_obj = Oscillogram(cfg_path, dat_path)
        osc_obj.load()

        # Assuming SIMILAR_AMPLITUDES_FILTER_ENABLED = True in the default config
        self.assertTrue(self.detector.config.get("SIMILAR_AMPLITUDES_FILTER_ENABLED", False))
        result = self.detector.analyze_oscillogram(osc_obj)
        self.assertIsNone(result, "No overvoltage due to symmetrical phase filter in spef_symmetrical_phases.cfg")

    def test_analyze_directory(self):
        output_csv_path = os.path.join(self.base_temp_dir, "ov_report.csv")
        log_path = os.path.join(self.base_temp_dir, "ov_log.txt")

        self.detector.analyze_directory(self.source_dir, output_csv_path, log_path)

        self.assertTrue(os.path.exists(output_csv_path))
        report_df = pd.read_csv(output_csv_path)

        self.assertFalse(report_df.empty, "Report CSV should not be empty")

        # Check for spef_present.cfg results
        spef_present_hash = Oscillogram(os.path.join(self.source_dir, "spef_present.cfg")).file_hash
        spef_present_row = report_df[report_df['file_hash'] == spef_present_hash]
        self.assertFalse(spef_present_row.empty, "spef_present.cfg should be in the report")
        self.assertGreater(spef_present_row.iloc[0]['overvoltage_factor'], 1.7)

        # Check that no_spef.cfg is not listed as having overvoltage (it might be in log, but not CSV)
        no_spef_hash = Oscillogram(os.path.join(self.source_dir, "no_spef.cfg")).file_hash
        self.assertFalse(no_spef_hash in report_df['file_hash'].values,
                         "no_spef.cfg should not be in the overvoltage report CSV")

        # Check that spef_low_ov.cfg is not listed
        spef_low_ov_hash = Oscillogram(os.path.join(self.source_dir, "spef_low_ov.cfg")).file_hash
        self.assertFalse(spef_low_ov_hash in report_df['file_hash'].values,
                         "spef_low_ov.cfg should not be in the overvoltage report CSV")

        # Check that spef_symmetrical_phases.cfg is not listed (if filter is on by default)
        if self.config_dict.get("SIMILAR_AMPLITUDES_FILTER_ENABLED", False):
            symmetrical_hash = Oscillogram(os.path.join(self.source_dir, "spef_symmetrical_phases.cfg")).file_hash
            self.assertFalse(symmetrical_hash in report_df['file_hash'].values,
                             "spef_symmetrical_phases.cfg should not be in the overvoltage report CSV when filter is on")


    def test_copy_spef_oscillograms(self):
        report_csv_path = os.path.join(self.base_temp_dir, "ov_report_for_copy.csv")
        log_path = os.path.join(self.base_temp_dir, "ov_log_for_copy.txt")

        # Generate a report first
        self.detector.analyze_directory(self.source_dir, report_csv_path, log_path)
        self.assertTrue(os.path.exists(report_csv_path), "Report CSV should be generated before copy test")

        copied_dir = os.path.join(self.base_temp_dir, "copied_ov_files")
        # Ensure copied_dir does not exist or is empty
        if os.path.exists(copied_dir):
            shutil.rmtree(copied_dir)
        os.makedirs(copied_dir)

        OvervoltageDetector.copy_spef_oscillograms(report_csv_path, self.source_dir, copied_dir)

        # Check if spef_present.cfg and .dat are copied
        self.assertTrue(os.path.exists(os.path.join(copied_dir, "spef_present.cfg")))
        self.assertTrue(os.path.exists(os.path.join(copied_dir, "spef_present.dat")))

        # Check that no_spef.cfg and .dat are NOT copied
        self.assertFalse(os.path.exists(os.path.join(copied_dir, "no_spef.cfg")))
        self.assertFalse(os.path.exists(os.path.join(copied_dir, "no_spef.dat")))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
