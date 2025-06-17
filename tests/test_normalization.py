import unittest
import os
import sys
import shutil
import pandas as pd
import numpy as np
import csv

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.oscillogram import Oscillogram
from normalization.normalization import NormalizationCoefficientGenerator, OscillogramNormalizer

class TestNormalizationCoefficientGenerator(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.abspath("tests/temp_norm_coeff_gen_data")
        self.sample_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))
        self.comtrade_files_path = os.path.join(self.sample_data_root, "comtrade_files", "for_normalization_1")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.source_dir = os.path.join(self.base_temp_dir, "source_oscillograms")
        os.makedirs(self.source_dir, exist_ok=True)

        # The class expects osc_path to be where the norm.csv will be written.
        # It internally uses self.osc_path + "norm.csv"
        self.coeff_generator = NormalizationCoefficientGenerator(
            osc_path=self.base_temp_dir, # Output directory for norm.csv
            prev_norm_csv_path="",
            bus=1, # Default bus, can be overridden in methods if needed
            is_print_message=False
        )
        self.coeff_generator.df_norm = pd.DataFrame() # Initialize an empty df_norm

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def _run_analyze_test(self, h1_data, hx_data, coef_p_s_data, current_or_voltage, ch_name, file_id, expected_ps, expected_base):
        h1_df = pd.DataFrame(h1_data)
        hx_df = pd.DataFrame(hx_data)
        # coef_p_s needs to be a Series with a specific structure if used, or mock its effect
        # For simplicity, we'll often set values that bypass complex coef_p_s lookups or assume default behavior.
        # The 'analyze' method primarily uses coef_p_s for 'kV_rating_primary' and 'kV_rating_secondary'.
        # Let's create a minimal coef_p_s DataFrame structure if needed by a specific scenario.
        # For now, most scenarios depend on direct values or thresholds not complex lookups in coef_p_s.

        # Mocking necessary attributes that would be set during full `normalization` run
        self.coeff_generator.current_or_voltage = current_or_voltage
        self.coeff_generator.ch_name = ch_name
        self.coeff_generator.file = file_id # 'file' attribute is used as file_id in analyze

        # Call analyze - it appends to self.coeff_generator.df_norm
        self.coeff_generator.analyze(h1_df, hx_df, pd.DataFrame()) # Empty coef_p_s for now

        # Retrieve the last added row
        result_row = self.coeff_generator.df_norm.iloc[-1]

        self.assertEqual(result_row['_PS'], expected_ps)
        if isinstance(expected_base, str):
            self.assertEqual(result_row['_base'], expected_base)
        else: # Assuming float/int for numerical bases
            self.assertAlmostEqual(result_row['_base'], expected_base, places=1)


    def test_analyze_method_noise_scenario(self):
        # Scenario: H1 <= 1.5 * Hx (noise condition)
        # H1[0] is secondary, H1[1] is primary
        # Hx[0] is secondary, Hx[1] is primary
        h1_data = {'channel_0': [1.0, 10.0]} # Secondary H1 = 1.0
        hx_data = {'channel_0': [0.8, 5.0]}  # Secondary Hx = 0.8.  1.0 <= 1.5 * 0.8 (1.0 <= 1.2) is FALSE.
                                             # Let's make Hx larger: Hx_sec = 0.7, 1.0 <= 1.5 * 0.7 => 1.0 <= 1.05 (TRUE)
        h1_data_noise = {'ch': [1.0, 10.0]}
        hx_data_noise = {'ch': [0.7, 5.0]} # 1.0 / 0.7 = 1.42... < 1.5, so it's noise.
                                           # Correction: the condition is H1_s <= 1.5 * Hx_s
                                           # 1.0 <= 1.5 * 0.7  => 1.0 <= 1.05. This is noise.
        self._run_analyze_test(h1_data_noise, hx_data_noise, {}, 'I', 'Ia', 'file1', 's', 'Noise')


    def test_analyze_method_high_primary_scenario(self):
        # Scenario: H1_secondary is low (e.g. < 0.001), but H1_primary is high (e.g. > 0.1)
        # This implies measurement from primary side (or issue)
        h1_data = {'ch': [0.0005, 0.5]} # H1_s low, H1_p high
        hx_data = {'ch': [0.0001, 0.01]} # Hx low, no noise
        self._run_analyze_test(h1_data, hx_data, {}, 'U', 'Ua', 'file2', 'p', '?1')


    def test_analyze_method_distorted_scenario(self):
        # Scenario: H1 > 1.5 * Hx (not noise), but H1 is low (e.g. < 0.01 for 'U', < 0.001 for 'I')
        # and not meeting other specific conditions. This often falls into '?2'
        h1_data = {'ch': [0.005, 0.1]} # H1_s = 0.005 (low for 'U' type)
        hx_data = {'ch': [0.001, 0.01]} # H1_s / Hx_s = 5 > 1.5 (not noise)
        self._run_analyze_test(h1_data, hx_data, {}, 'U', 'Ub', 'file3', 's', '?2')

        h1_data_I = {'ch': [0.0005, 0.1]} # H1_s = 0.0005 (low for 'I' type)
        hx_data_I = {'ch': [0.0001, 0.01]} # H1_s / Hx_s = 5 > 1.5 (not noise)
        self._run_analyze_test(h1_data_I, hx_data_I, {}, 'I', 'Ib', 'file3_I', 's', '?2')


    def test_analyze_method_voltage_tier1_scenario(self):
        # Scenario: Voltage, secondary side, H1_s clearly indicates a voltage tier.
        # Example: H1_s is around 100 / sqrt(3) = 57.7. Let's use 58.0
        # Tiers: [6, 10, 15, 20, 27, 35, 100, 110, 150, 220, 330, 400, 500, 750]
        # Closest to 58.0 is 100.0 (as 100/sqrt(3)). Or, if it's phase-to-phase, it's 100.
        # The logic compares H1_s to `v / np.sqrt(3)` and `v`.
        # If H1_s = 58, then 58 * sqrt(3) = 100.4. Closest tier is 100.
        h1_data = {'ch': [58.0, 0.1]} # H1_s = 58.0
        hx_data = {'ch': [0.1, 0.01]}  # Not noise: 58.0 / 0.1 = 580 > 1.5
        self._run_analyze_test(h1_data, hx_data, {}, 'U', 'Uc', 'file4', 's', 100.0)


    def test_normalization_method_integration(self):
        # Copy sample COMTRADE files to the source_dir for this test generator instance
        cfg_file_name = "for_norm_coeffs.cfg"
        dat_file_name = "for_norm_coeffs.dat"
        shutil.copy(os.path.join(self.comtrade_files_path, cfg_file_name), self.source_dir)
        shutil.copy(os.path.join(self.comtrade_files_path, dat_file_name), self.source_dir)

        # The NormalizationCoefficientGenerator's osc_path is base_temp_dir.
        # It will scan source_dir if its own osc_path is a directory and empty of CFGs.
        # Or, more directly, we can point its 'path_to_files_for_normalization'
        self.coeff_generator.path_to_files_for_normalization = self.source_dir

        self.coeff_generator.normalization()

        output_csv_path = os.path.join(self.base_temp_dir, "norm.csv")
        self.assertTrue(os.path.exists(output_csv_path))

        df_output = pd.read_csv(output_csv_path)
        self.assertFalse(df_output.empty)

        # Expected channels from a hypothetical 'for_norm_coeffs.cfg'
        # Example: file has Ua, Ub, Uc, Ia, Ib, Ic.
        # This depends heavily on the content of the sample file.
        # For now, just check if some rows are generated for the file.
        # The file_hash would be generated by Oscillogram(cfg_path).file_hash
        osc_obj = Oscillogram(os.path.join(self.source_dir, cfg_file_name))
        expected_file_hash = osc_obj.file_hash

        self.assertTrue(expected_file_hash in df_output['file'].values)
        # Example: Check if a Ua channel was processed
        self.assertTrue('Ua' in df_output[df_output['file'] == expected_file_hash]['name'].values)


    def test_static_update_dataframe(self):
        input_csv_path = os.path.join(self.base_temp_dir, "input_update.csv")
        output_csv_path = os.path.join(self.base_temp_dir, "output_update.csv")
        file_list_path = os.path.join(self.base_temp_dir, "file_list.txt")

        # Create dummy input.csv
        pd.DataFrame({
            'file': ['hash1', 'hash2', 'hash3'],
            'name': ['Ua', 'Ub', 'Uc'],
            'other_col': [1,2,3]
        }).to_csv(input_csv_path, index=False)
        # Create dummy file_list.txt (files to keep)
        with open(file_list_path, 'w') as f:
            f.write("hash1\n")
            f.write("hash3\n")

        NormalizationCoefficientGenerator.update_dataframe_static(
            input_csv_path, output_csv_path, file_list_path, self.base_temp_dir # temp_folder_for_intermediate_storage
        )
        self.assertTrue(os.path.exists(output_csv_path))
        df_out = pd.read_csv(output_csv_path)
        self.assertEqual(len(df_out), 2)
        self.assertTrue('hash1' in df_out['file'].values)
        self.assertTrue('hash3' in df_out['file'].values)
        self.assertFalse('hash2' in df_out['file'].values)


    def test_static_correct_df(self):
        input_errors_path = os.path.join(self.base_temp_dir, "input_errors.csv")
        output_corrected_path = os.path.join(self.base_temp_dir, "output_corrected.csv")
        # Create dummy input_errors.csv
        pd.DataFrame({
            'file': ['h1', 'h2', 'h3', 'h4'],
            'name': ['Ua', 'Ub', 'Uc', 'Ud'],
            '_base': [100.0, 'Noise', '?1', 50.0],
            'norm': ['YES', 'NO', 'hz', 'YES']
        }).to_csv(input_errors_path, index=False)

        NormalizationCoefficientGenerator.correct_df_static(input_errors_path, output_corrected_path)
        self.assertTrue(os.path.exists(output_corrected_path))
        df_out = pd.read_csv(output_corrected_path)

        self.assertEqual(df_out[df_out['file'] == 'h1']['norm'].iloc[0], 'YES') # Stays YES
        self.assertEqual(df_out[df_out['file'] == 'h2']['norm'].iloc[0], 'NO')  # Stays NO
        self.assertEqual(df_out[df_out['file'] == 'h3']['norm'].iloc[0], 'NO')  # hz -> NO
        self.assertEqual(df_out[df_out['file'] == 'h3']['_base'].iloc[0], 0)    # Base becomes 0 for hz


    def test_static_merge_normalization_files(self):
        norm1_path = os.path.join(self.base_temp_dir, "norm1.csv")
        norm2_path = os.path.join(self.base_temp_dir, "norm2.csv")
        merged_path = os.path.join(self.base_temp_dir, "merged_norm.csv")

        pd.DataFrame({
            'file': ['h1', 'h2'], 'name': ['Ua', 'Ub'], '_base': [100, 200], 'extra_col1': ['a','b']
        }).to_csv(norm1_path, index=False)
        pd.DataFrame({
            'file': ['h2', 'h3'], 'name': ['Ub', 'Uc'], '_base': [250, 300], 'extra_col2': ['x','y']
        }).to_csv(norm2_path, index=False)

        # Files to merge
        files_to_merge_list = [norm1_path, norm2_path]

        NormalizationCoefficientGenerator.merge_normalization_files_static(
            files_to_merge_list, merged_path, merge_on_file_and_name=True
        )
        self.assertTrue(os.path.exists(merged_path))
        df_merged = pd.read_csv(merged_path)

        self.assertEqual(len(df_merged), 3) # h1(Ua), h2(Ub), h3(Uc)
        # h2/Ub from norm1 should be taken (first encountered)
        self.assertEqual(df_merged[df_merged['file'] == 'h2']['_base'].iloc[0], 200)
        self.assertTrue('extra_col1' in df_merged.columns)
        self.assertTrue('extra_col2' in df_merged.columns)


class TestOscillogramNormalizer(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.abspath("tests/temp_normalizer_data")
        self.sample_data_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data", "config_files"))

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.sample_norm_coeffs_src_path = os.path.join(self.sample_data_config_path, "sample_norm_coeffs.csv")
        self.sample_norm_coeffs_path = os.path.join(self.base_temp_dir, "sample_norm_coeffs.csv")
        shutil.copy(self.sample_norm_coeffs_src_path, self.sample_norm_coeffs_path)

        self.normalizer = OscillogramNormalizer(self.sample_norm_coeffs_path, is_print_message=False)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_normalize_bus_signals_valid(self):
        # From sample_norm_coeffs.csv, for file_id='hash123abc', name='Ua':
        # _base = 57.73, _PS = 's', Factor = 1.0, Ur_nom_kV = 0.1, Ir_nom_A = 5
        # Normalization for voltage: val / (Factor * _base_from_csv * (1 if _PS == 's' else Ur_nom_kV))
        # If _PS='s', Factor=1, _base=57.73: val / 57.73
        # To get 1.0 PU, input val should be 57.73

        # The column names in bus_df are like "U | BusName | phase: A"
        # The 'name' in norm_coeffs.csv is "Ua", "Ia" etc.
        # OscillogramNormalizer maps "U | BusName | phase: A" to "Ua" internally.

        bus_df_data = {"U | BusBar-1 | phase: A": [57.73], "I | Feeder-X | phase: A": [10.0]}
        bus_df = pd.DataFrame(bus_df_data)
        file_id = "hash123abc" # Exists in sample_norm_coeffs.csv

        normalized_df = self.normalizer.normalize_bus_signals(bus_df.copy(), file_id) # Pass copy
        self.assertIsNotNone(normalized_df)
        self.assertAlmostEqual(normalized_df["U | BusBar-1 | phase: A"].iloc[0], 1.0, places=1)

        # For current: name='Ia', _base=10, _PS='s', Factor=1, Ir_nom_A=5
        # Norm: val / (Factor * _base_from_csv * (1 if _PS == 's' else Ir_nom_A))
        # If _PS='s', Factor=1, _base=10: val / 10. Input 10.0 -> 1.0 PU
        self.assertAlmostEqual(normalized_df["I | Feeder-X | phase: A"].iloc[0], 1.0, places=1)


    def test_normalize_bus_signals_no_coeffs_for_file(self):
        bus_df = pd.DataFrame({"U | BusBar-1 | phase: A": [100.0]})
        normalized_df = self.normalizer.normalize_bus_signals(bus_df.copy(), "unknown_hash")
        self.assertIsNone(normalized_df, "Should return None if file_id not in coeffs")

    def test_normalize_bus_signals_norm_disallowed(self):
        # file_id='hash456def' has norm='NO' in sample_norm_coeffs.csv
        bus_df = pd.DataFrame({"U | BusBar-1 | phase: A": [100.0]})
        file_id = "hash456def"
        normalized_df = self.normalizer.normalize_bus_signals(bus_df.copy(), file_id)
        self.assertIsNone(normalized_df, "Should return None if norm is 'NO'")

        # file_id='hash789xyz' has norm='hz'
        file_id_hz = "hash789xyz"
        normalized_df_hz = self.normalizer.normalize_bus_signals(bus_df.copy(), file_id_hz)
        self.assertIsNone(normalized_df_hz, "Should return None if norm is 'hz'")


    def test_normalizer_init_no_coef_file(self):
        bad_path = os.path.join(self.base_temp_dir, "non_existent_coeffs.csv")
        norm_temp = OscillogramNormalizer(bad_path, is_print_message=False)
        self.assertTrue(norm_temp.norm_coef.empty, "norm_coef should be empty if file not found")

        bus_df = pd.DataFrame({'U | BusBar-1 | phase: A': [1.0]})
        res = norm_temp.normalize_bus_signals(bus_df, "any_file_id")
        self.assertIsNone(res, "normalize_bus_signals should return None if norm_coef is empty")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
