import unittest
import os
import sys
import shutil
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.oscillogram import Oscillogram
from normalization.normalization import OscillogramNormalizer
from raw_to_csv.raw_to_csv import OscillogramToCsvConverter

class TestOscillogramToCsvConverter(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.abspath("tests/temp_converter_data")
        self.sample_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.source_dir = os.path.join(self.base_temp_dir, "source_oscillograms")
        os.makedirs(self.source_dir, exist_ok=True)
        self.csv_output_dir = os.path.join(self.base_temp_dir, "csv_output")
        os.makedirs(self.csv_output_dir, exist_ok=True)
        self.config_dir = os.path.join(self.base_temp_dir, "config_files")
        os.makedirs(self.config_dir, exist_ok=True)

        # Copy sample COMTRADE files
        comtrade_src_valid1 = os.path.join(self.sample_data_root, "comtrade_files", "valid_cfg_dat_1")
        shutil.copy(os.path.join(comtrade_src_valid1, "sample_A.cfg"), self.source_dir)
        shutil.copy(os.path.join(comtrade_src_valid1, "sample_A.dat"), self.source_dir)

        comtrade_src_converter = os.path.join(self.sample_data_root, "comtrade_files", "for_converter")
        # sample_B.cfg should have a digital signal for event testing
        shutil.copy(os.path.join(comtrade_src_converter, "sample_B.cfg"), self.source_dir)
        shutil.copy(os.path.join(comtrade_src_converter, "sample_B.dat"), self.source_dir)

        # Copy sample config files
        config_src = os.path.join(self.sample_data_root, "config_files")
        self.norm_coeffs_path = os.path.join(self.config_dir, "test_norm.csv")
        shutil.copy(os.path.join(config_src, "sample_norm_coeffs.csv"), self.norm_coeffs_path)

        self.analog_names_path = os.path.join(self.config_dir, "test_analog_names.json")
        shutil.copy(os.path.join(config_src, "dict_analog_names_sample.json"), self.analog_names_path)

        self.discrete_names_path = os.path.join(self.config_dir, "test_discrete_names.json")
        shutil.copy(os.path.join(config_src, "dict_discrete_names_sample.json"), self.discrete_names_path)

        self.normalizer = OscillogramNormalizer(norm_coef_file_path=self.norm_coeffs_path, is_print_message=False)

        self.converter = OscillogramToCsvConverter(
            normalizer=self.normalizer,
            raw_path=self.source_dir, # Path to oscillogram files
            csv_path=self.csv_output_dir, # Path where output CSVs will be saved
            dict_analog_names_path=self.analog_names_path,
            dict_discrete_names_path=self.discrete_names_path,
            is_print_message=False,
            show_progress_bars=False
        )

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_create_csv_basic_conversion(self):
        output_csv_name = "basic_output.csv"
        self.converter.create_csv(output_csv_name)

        output_file_path = os.path.join(self.csv_output_dir, output_csv_name)
        self.assertTrue(os.path.exists(output_file_path))

        df = pd.read_csv(output_file_path)
        self.assertFalse(df.empty)

        # Check for renamed columns (example based on typical dict_analog_names_sample.json)
        # e.g., "U | LS ROSSIJA | phase: A" might become "U_A LS ROSSIJA" or "U_A LR"
        # This depends on the content of test_analog_names.json and rename_bus_columns
        # A generic check:
        self.assertTrue(any(col.startswith("U_A") or col.startswith("IA_") for col in df.columns))

        # Verify data normalization (qualitative check: values should be small, typically around -2 to 2 PU)
        # Example: check a voltage column if 'U_A LR' exists after renaming
        # This requires knowing a specific column that gets normalized from sample_A.cfg
        # and its corresponding entry in sample_norm_coeffs.csv
        # For 'hash123abc' (sample_A.cfg) and 'Ua' (mapped from "U | LS ROSSIJA | phase: A"),
        # _base is 57.73. If original value was ~5773, normalized should be ~1.0
        # Let's assume such a column exists after renaming and processing
        # This part is highly dependent on sample data alignment.
        # A simple check: are most values in a known analog column small?
        # Find a column that looks like a normalized analog signal
        analog_signal_col = None
        for col in df.columns:
            if (col.startswith("U_") or col.startswith("I_")) and \
               not col.endswith("_event") and \
               df[col].dtype in [np.float64, np.int64]: # ensure it's numeric
                # Further check: is it a common phase?
                if "_A " in col or "_B " in col or "_C " in col or "_N " in col: # Space after phase
                    analog_signal_col = col
                    break

        if analog_signal_col:
            # Check if values are roughly in PU range (e.g. mean close to 0, std dev around 0.5-1.5)
            # This is a loose check, actual values depend on the specific signal
            # A more robust check would be to compare with expected PU values based on norm_coeffs
            self.assertTrue(df[analog_signal_col].mean() < 10 and df[analog_signal_col].mean() > -10) # Rough check
            self.assertTrue(df[analog_signal_col].std() < 10) # Rough check for PU scale
        else:
            print(f"Warning: Could not find a clearly normalized analog column for basic value check in {df.columns}")


    def test_create_csv_with_cut_out_area(self):
        self.converter.number_periods = 2 # Number of periods before and after event
        output_csv_name = "cut_output.csv"
        self.converter.create_csv(output_csv_name, is_cut_out_area=True)

        output_file_path = os.path.join(self.csv_output_dir, output_csv_name)
        self.assertTrue(os.path.exists(output_file_path))

        df = pd.read_csv(output_file_path)
        self.assertFalse(df.empty)

        # sample_B.cfg is designed to have an event.
        # Its original Oscillogram object:
        osc_B = Oscillogram(os.path.join(self.source_dir, "sample_B.cfg"),
                            os.path.join(self.source_dir, "sample_B.dat"))
        osc_B.load()
        total_samples_original_B = len(osc_B.time)
        samples_per_period_B = int(osc_B.sample_rates[0][1] / osc_B.cfg.frequency)


        # Filter for rows related to sample_B events.
        # The file_name column in the CSV should contain the original filename part.
        # And an event suffix if cut_out_area was applied.
        df_sample_B_events = df[df['file_name'].str.contains("sample_B") & df['file_name'].str.contains("_event")]
        self.assertFalse(df_sample_B_events.empty, "No event data found for sample_B in cut output.")

        # Expected duration of cut area: (2 * num_periods * samples_per_period) + event_duration_samples
        # Event duration is usually 1 sample for discrete signals.
        # Total length around 2 * 2 * samples_per_period_B + (a few samples for event itself, let's say 1 to 10)
        expected_min_len = 2 * self.converter.number_periods * samples_per_period_B
        # Max len could be slightly more due to how event window is centered or extends
        expected_max_len = (2 * self.converter.number_periods + 2) * samples_per_period_B # Generous upper bound for a short event

        # Since one event can be split into multiple rows if data is long, sum lengths
        total_event_rows_for_sample_B = len(df_sample_B_events)

        self.assertLess(total_event_rows_for_sample_B, total_samples_original_B)
        self.assertGreaterEqual(total_event_rows_for_sample_B, expected_min_len,
                                f"Cut event data too short. Got {total_event_rows_for_sample_B}, expected min {expected_min_len}")
        # self.assertLessEqual(total_event_rows_for_sample_B, expected_max_len,
        #                         f"Cut event data too long. Got {total_event_rows_for_sample_B}, expected max {expected_max_len}")


    def test_create_csv_simple_output_generation(self):
        main_csv_name = "main_for_simple.csv"
        self.converter.create_csv(main_csv_name, is_simple_csv=True)

        simple_csv_path = os.path.join(self.csv_output_dir, f"{os.path.splitext(main_csv_name)[0]}_simple.csv")
        self.assertTrue(os.path.exists(simple_csv_path))

        df_simple = pd.read_csv(simple_csv_path)
        self.assertFalse(df_simple.empty)

        expected_simple_cols = ['opr_swch', 'abnorm_evnt', 'emerg_evnt', 'normal']
        for col in expected_simple_cols:
            self.assertIn(col, df_simple.columns)

        # Assert that original detailed ML columns (from self.converter.ml_all) are NOT present
        # self.converter.ml_all contains raw ML signal names like "MLsignal_1_1_1_1"
        # After renaming by rename_bus_columns they might be "ML_1_1_1"
        # The simple CSV should not have these.
        for col in df_simple.columns:
            self.assertFalse(col.startswith("ML_"), "Detailed ML columns should not be in simple CSV")
            self.assertFalse(col.startswith("MLsignal_"), "Detailed ML columns should not be in simple CSV")


    def test_split_buses_logic(self):
        osc_A_path = os.path.join(self.source_dir, "sample_A.cfg")
        osc_A = Oscillogram(cfg_file_path=osc_A_path)
        osc_A.load()

        # split_buses expects a DataFrame from Oscillogram and applies normalization if normalizer is set
        # The converter's normalizer is already set up.
        split_df_list = self.converter.split_buses(osc_A.data_frame.copy(), "sample_A.cfg", osc_A.file_hash)

        self.assertTrue(isinstance(split_df_list, list))
        self.assertTrue(len(split_df_list) > 0, "split_buses should return at least one DataFrame")

        split_df_example = split_df_list[0] # Assuming at least one bus configuration
        self.assertFalse(split_df_example.empty)

        # Check file_name column for bus suffix (e.g., "sample_A_bus1" or specific name from dict)
        # The actual bus name depends on dict_analog_names_sample.json
        # Example: if "U | LS ROSSIJA | phase: A" maps to bus "LR", expect "sample_A_LR"
        # This requires knowing the content of dict_analog_names_sample.json
        # For now, check if 'file_name' exists and has 'sample_A' prefix.
        self.assertIn('file_name', split_df_example.columns)
        self.assertTrue(split_df_example['file_name'].iloc[0].startswith("sample_A"))

        # Assert expected columns for "bus1" (from dict_analog_names_sample.json) are present
        # e.g. "U | LS ROSSIJA | phase: A" after mapping.
        # This also depends heavily on the content of dict_analog_names_sample.json
        # A generic check:
        self.assertTrue(any(col.startswith("U |") or col.startswith("I |") for col in split_df_example.columns))


    def test_rename_bus_columns_logic(self):
        # Create a sample DataFrame with columns like "U | BusBar-1 | phase: A", "MLsignal_1_1_1_1"
        # And a discrete signal that is NOT an ML signal, e.g. "Discrete_Raw_Name"
        # which maps via dict_discrete_names to "Discrete_Mapped_Name"

        # From dict_discrete_names_sample.json: "EVENT_TRIGGER_1" -> "MLsignal_1_1_1_1"
        # Let's add another: "OTHER_DISCRETE" -> "ControlSignalX"
        sample_data = {
            "U | BusBar-1 | phase: A": [1.0, 2.0],
            "I | Feeder-X | phase: C": [3.0, 4.0],
            "EVENT_TRIGGER_1": [0, 1], # This is raw name, before discrete dict mapping
            "OTHER_DISCRETE": [1,0]    # This is raw name
        }
        df = pd.DataFrame(sample_data)

        # The rename_bus_columns method first applies discrete name mapping
        # then renames based on analog dict and ML signal parsing.

        # Simulate discrete mapping that would happen inside create_one_df before rename_bus_columns
        # discrete_name_mapping = {"EVENT_TRIGGER_1": "MLsignal_1_1_1_1", "OTHER_DISCRETE": "ControlSignalX"}
        # df_mapped_discrete = df.rename(columns=discrete_name_mapping)
        # In the actual code, this mapping is more complex using self.converter.digital_chan_name_protocol
        # For this unit test, we assume digital_chan_name_protocol is applied correctly before rename_bus_columns.
        # The input to rename_bus_columns will have digital channels already mapped if they were in dict_discrete_names.
        # So, we should use the *mapped* names if testing rename_bus_columns in isolation.
        # However, rename_bus_columns itself *also* consults self.digital_chan_name_protocol for unmapped discrete.

        # Let's assume digital_chan_name_protocol is correctly populated.
        # self.converter.digital_chan_name_protocol = {"EVENT_TRIGGER_1": "MLsignal_1_1_1_1", "OTHER_DISCRETE": "ControlSignalX"}

        renamed_df = self.converter.rename_bus_columns(df.copy(), is_use_ML=True, is_use_discrete=True)

        # Expected renaming from dict_analog_names_sample.json:
        # "U | BusBar-1 | phase: A" -> "U_A BB1" (if "BusBar-1" maps to "BB1")
        # "I | Feeder-X | phase: C" -> "I_C FX" (if "Feeder-X" maps to "FX")
        # This requires specific knowledge of dict_analog_names_sample.json
        # For a generic test:
        self.assertTrue(any(col.startswith("U_A") for col in renamed_df.columns))
        self.assertTrue(any(col.startswith("I_C") for col in renamed_df.columns))

        # MLsignal_1_1_1_1 (from EVENT_TRIGGER_1 after discrete map) should become ML_1_1_1
        self.assertIn("ML_1_1_1", renamed_df.columns)

        # OTHER_DISCRETE mapped to ControlSignalX (if not an ML signal) should be kept as is or with a prefix
        # The current rename_bus_columns logic might drop non-ML discrete if is_use_discrete=False for ML part
        # If is_use_discrete=True, it should keep them, possibly with prefix.
        # Based on code: if it's in digital_chan_name_protocol and not ML, it's kept.
        # If it's NOT in digital_chan_name_protocol, it's kept.
        # The renaming for discrete is basically just ensuring they are strings.
        self.assertIn("ControlSignalX", renamed_df.columns) # Assuming "OTHER_DISCRETE" mapped to this via protocol

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
