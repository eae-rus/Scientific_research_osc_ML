import unittest
import os
import sys
import shutil
import csv
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Optional # For type hints

# Add project root to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from preprocessing.signal_names import SignalNameManager

BASE_TEST_ROOT_DIR = os.path.dirname(__file__)
SAMPLE_DATA_ROOT = os.path.join(BASE_TEST_ROOT_DIR, "sample_data")
COMTRADE_SAMPLES_PATH = os.path.join(SAMPLE_DATA_ROOT, "comtrade_files")

class TestSignalNameManagerFindAndMerge(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.join(BASE_TEST_ROOT_DIR, "temp_signal_names_data")
        self.source_dir = os.path.join(self.base_temp_dir, "source_cfgs")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.source_dir, exist_ok=True)

        # Copy necessary sample files for testing find_signal_names
        # Ensure these files are created by test_data_setup.py
        self.names_A_cfg_orig = os.path.join(COMTRADE_SAMPLES_PATH, "for_signal_names_1", "names_A.cfg")
        self.names_A_dat_orig = os.path.join(COMTRADE_SAMPLES_PATH, "for_signal_names_1", "names_A.dat")
        self.cp1251_cfg_orig = os.path.join(COMTRADE_SAMPLES_PATH, "encoding_tests", "cp1251_encoded.cfg")
        self.cp1251_dat_orig = os.path.join(COMTRADE_SAMPLES_PATH, "encoding_tests", "cp1251_encoded.dat")

        if not (os.path.exists(self.names_A_cfg_orig) and \
                os.path.exists(self.names_A_dat_orig) and \
                os.path.exists(self.cp1251_cfg_orig) and \
                os.path.exists(self.cp1251_dat_orig)):
            self.fail(f"One or more sample files not found. Ensure test_data_setup.py has run. Searched in {COMTRADE_SAMPLES_PATH}")

        shutil.copy2(self.names_A_cfg_orig, self.source_dir)
        shutil.copy2(self.names_A_dat_orig, self.source_dir)
        shutil.copy2(self.cp1251_cfg_orig, self.source_dir)
        shutil.copy2(self.cp1251_dat_orig, self.source_dir)

        self.manager = SignalNameManager(verbose_logging=False, show_progress_bars=False)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_find_signal_names_all(self):
        output_csv = os.path.join(self.base_temp_dir, "catalog_all.csv")
        self.manager.find_signal_names(self.source_dir, signal_type_to_find='all', output_csv_path=output_csv)

        self.assertTrue(os.path.exists(output_csv))
        df = pd.read_csv(output_csv)
        self.assertListEqual(list(df.columns), ['Signal Name', 'Universal Code', 'Count'])

        # Expected signals from names_A.cfg (content from prompt):
        # 1,SigAnalog1 | phase: X,X,,V...  -> "SigAnalog1|phase:X"
        # 2,SigAnalogShared | phase: Y,Y,,V... -> "SigAnalogShared|phase:Y"
        # 3,DigitalRawName1,,,1 -> "DigitalRawName1"
        # Expected from cp1251_encoded.cfg:
        # 1,АналогСИмя | phase: Ф,Ф,,V... -> "АналогСИмя|phase:Ф"
        # 2,ДискретИмя,,,1 -> "ДискретИмя"
        expected_signals_counts = {
            "SigAnalog1|phase:X": 1,
            "SigAnalogShared|phase:Y": 1,
            "DigitalRawName1": 1,
            "АналогСИмя|phase:Ф": 1,
            "ДискретИмя": 1
        }
        self.assertEqual(len(df), len(expected_signals_counts))
        for _, row in df.iterrows():
            self.assertIn(row['Signal Name'], expected_signals_counts)
            self.assertEqual(row['Count'], expected_signals_counts[row['Signal Name']])
            self.assertEqual(row['Universal Code'], '-')

    def test_find_signal_names_analog_only(self):
        output_csv = os.path.join(self.base_temp_dir, "catalog_analog.csv")
        self.manager.find_signal_names(self.source_dir, signal_type_to_find='analog', output_csv_path=output_csv)

        self.assertTrue(os.path.exists(output_csv))
        df = pd.read_csv(output_csv)
        self.assertListEqual(list(df.columns), ['Signal Name', 'Universal Code', 'Count'])

        expected_analog_signals = {
            "SigAnalog1|phase:X": 1,
            "SigAnalogShared|phase:Y": 1,
            "АналогСИмя|phase:Ф": 1
        }
        self.assertEqual(len(df), len(expected_analog_signals))
        for _, row in df.iterrows():
            self.assertIn(row['Signal Name'], expected_analog_signals)
            self.assertTrue("Digital" not in row['Signal Name'] and "Дискрет" not in row['Signal Name'])

    def test_find_signal_names_digital_only(self):
        output_csv = os.path.join(self.base_temp_dir, "catalog_digital.csv")
        self.manager.find_signal_names(self.source_dir, signal_type_to_find='digital', output_csv_path=output_csv)

        self.assertTrue(os.path.exists(output_csv))
        df = pd.read_csv(output_csv)
        self.assertListEqual(list(df.columns), ['Signal Name', 'Universal Code', 'Count'])

        expected_digital_signals = {
            "DigitalRawName1": 1,
            "ДискретИмя": 1
        }
        self.assertEqual(len(df), len(expected_digital_signals))
        for _, row in df.iterrows():
            self.assertIn(row['Signal Name'], expected_digital_signals)
            self.assertTrue("| phase:" not in row['Signal Name'])


    def test_merge_signal_code_csvs_merge_values(self):
        csv1_path = os.path.join(self.base_temp_dir, "catalog1.csv")
        csv1_data = [
            ['Signal Name', 'Universal Code', 'Count'],
            ['Ua | phase:A', 'U_A', '10'],
            ['Ia | phase:B', '-', '5'],
            ['Xyz', 'XYZ_OLD', '3']
        ]
        with open(csv1_path, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(csv1_data)

        csv2_path = os.path.join(self.base_temp_dir, "catalog2.csv")
        csv2_data = [
            ['Signal Name', 'Universal Code', 'Count'],
            ['Ua | phase:A', 'UA_revised', '7'], # Common key
            ['Ic | phase:C', 'I_C', '3'],      # New key
            ['Xyz', '?', '2']                 # Common key, old universal code was better
        ]
        with open(csv2_path, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(csv2_data)

        merged_path = os.path.join(self.base_temp_dir, "merged1.csv")
        self.manager.merge_signal_code_csvs(csv1_path, csv2_path, merged_path, is_merge_values=True, verbose_logging_method=self.is_print_message_tests)

        self.assertTrue(os.path.exists(merged_path))
        df = pd.read_csv(merged_path)

        # Expected: Ua count = 17, code U_A. Xyz count = 5, code XYZ_OLD. Sorted by count.
        # Order: Ua | phase:A, Ia | phase:B, Xyz, Ic | phase:C (if sorted by count then original order for ties)
        # The sort is by value, then by key (implicitly if values are same)
        # Expected order by count desc: Ua (17), Ia (5), Xyz (5), Ic (3)

        self.assertEqual(df[df['Signal Name'] == 'Ua | phase:A']['Count'].iloc[0], 17)
        self.assertEqual(df[df['Signal Name'] == 'Ua | phase:A']['Universal Code'].iloc[0], 'U_A')
        self.assertEqual(df[df['Signal Name'] == 'Xyz']['Count'].iloc[0], 5)
        self.assertEqual(df[df['Signal Name'] == 'Xyz']['Universal Code'].iloc[0], 'XYZ_OLD')
        self.assertTrue('Ia | phase:B' in df['Signal Name'].values)
        self.assertTrue('Ic | phase:C' in df['Signal Name'].values)
        self.assertEqual(len(df), 4)


    def test_merge_signal_code_csvs_overwrite_values(self):
        csv1_path = os.path.join(self.base_temp_dir, "catalog3.csv")
        csv1_data = [
            ['Signal Name', 'Universal Code', 'Count'],
            ['Ua | phase:A', 'U_A', '10'],
            ['Ia | phase:B', '-', '5']
        ]
        with open(csv1_path, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(csv1_data)

        csv2_path = os.path.join(self.base_temp_dir, "catalog4.csv")
        csv2_data = [
            ['Signal Name', 'Universal Code', 'Count'],
            ['Ua | phase:A', 'UA_revised', '7'], # Common key
            ['Ic | phase:C', 'I_C', '3']       # New key
        ]
        with open(csv2_path, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(csv2_data)

        merged_path = os.path.join(self.base_temp_dir, "merged2.csv")
        self.manager.merge_signal_code_csvs(csv1_path, csv2_path, merged_path, is_merge_values=False, verbose_logging_method=self.is_print_message_tests)

        self.assertTrue(os.path.exists(merged_path))
        df = pd.read_csv(merged_path)

        # Expected: Ua count = 7 (from new), code U_A. Sorted by count.
        # Order: Ua | phase:A (7), Ia | phase:B (5), Ic | phase:C (3)
        self.assertEqual(df[df['Signal Name'] == 'Ua | phase:A']['Count'].iloc[0], 7)
        self.assertEqual(df[df['Signal Name'] == 'Ua | phase:A']['Universal Code'].iloc[0], 'U_A') # Old code takes precedence
        self.assertTrue('Ia | phase:B' in df['Signal Name'].values)
        self.assertTrue('Ic | phase:C' in df['Signal Name'].values)
        self.assertEqual(len(df), 3)

class TestSignalNameManagerRenameAndDuplicates(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.join(BASE_TEST_ROOT_DIR, "temp_signal_names_rename_data")
        self.sample_data_comtrade_path = os.path.join(SAMPLE_DATA_ROOT, "comtrade_files")
        self.sample_data_config_path = os.path.join(SAMPLE_DATA_ROOT, "config_files")

        self.source_dir = os.path.join(self.base_temp_dir, "source_cfgs_rename")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.source_dir, exist_ok=True)

        # Copy names_A.cfg and .dat for testing (ensure it's the version with all needed signals)
        # This version of names_A.cfg should be created by test_data_setup.py as per prompt.
        # For this test, we'll assume it has the structure defined in the prompt for this subtask.
        # If not, these tests might fail or need adjustment based on actual names_A.cfg content.
        names_A_cfg_orig_path = os.path.join(self.sample_data_comtrade_path, "for_signal_names_1", "names_A.cfg")
        names_A_dat_orig_path = os.path.join(self.sample_data_comtrade_path, "for_signal_names_1", "names_A.dat")

        self.assertTrue(os.path.exists(names_A_cfg_orig_path), f"names_A.cfg not found at {names_A_cfg_orig_path}")
        self.assertTrue(os.path.exists(names_A_dat_orig_path), f"names_A.dat not found at {names_A_dat_orig_path}")

        self.test_cfg_path = os.path.join(self.source_dir, "names_A.cfg")
        shutil.copy2(names_A_cfg_orig_path, self.test_cfg_path)
        shutil.copy2(names_A_dat_orig_path, os.path.join(self.source_dir, "names_A.dat"))

        # Copy signal_rename_map.csv
        rename_map_orig_path = os.path.join(self.sample_data_config_path, "signal_rename_map.csv")
        self.assertTrue(os.path.exists(rename_map_orig_path), f"signal_rename_map.csv not found at {rename_map_orig_path}")
        self.current_rename_map_csv_path = os.path.join(self.base_temp_dir, "current_rename_map.csv")
        shutil.copy2(rename_map_orig_path, self.current_rename_map_csv_path)

        self.manager = SignalNameManager(verbose_logging=False, show_progress_bars=False)
        self.is_print_message_tests = False # For controlling prints from manager methods

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def _read_cfg_lines(self, cfg_path: str) -> List[str]:
        with open(cfg_path, 'r', encoding='utf-8') as f: # Assuming utf-8 for test files created by setup
            return f.readlines()

    def _get_signal_field(self, line: str, field_index: int) -> str:
        parts = line.split(',')
        if len(parts) > field_index:
            return parts[field_index].strip()
        return ""

    def test_rename_signals_from_csv(self):
        # Ensure names_A.cfg has "SigToRenameCSV | phase: R" and "DigitalToRenameCSV"
        # And signal_rename_map.csv maps them.
        # Original names_A.cfg (from test_data_setup.py for this subtask):
        # 1,SigToRenameCSV | phase: R,R,,V...
        # 5,DigitalToRenameCSV,,,1

        self.manager.rename_signals_from_csv(
            self.source_dir,
            self.current_rename_map_csv_path,
            signal_type_to_rename='all',
            verbose_logging_method=self.is_print_message_tests
        )

        lines = self._read_cfg_lines(self.test_cfg_path)

        # Check analog rename (line index 2 in CFG, which is lines[2])
        # Field 1 is raw_name_field
        self.assertEqual(self._get_signal_field(lines[2], 1), "AnalogNewNameFromCSV")
        self.assertEqual(self._get_signal_field(lines[2], 2), "R", "Phase field for analog should be unchanged")

        # Check digital rename (line index 6 in CFG, which is lines[6])
        self.assertEqual(self._get_signal_field(lines[6], 1), "DigitalNewNameFromCSV")


    def test_rename_single_signal(self):
        # names_A.cfg has "SingleRenameTarget | phase: S" as raw_name_field on line 5 (index 4)
        target_line_index = 5 # CFG line: 4,SingleRenameTarget | phase: S,S...
        old_raw_name = "SingleRenameTarget | phase: S"
        new_raw_name = "SignalWasRenamed"

        self.manager.rename_single_signal(
            self.source_dir,
            old_name_pattern=old_raw_name,
            new_name_field_value=new_raw_name,
            verbose_logging_method=self.is_print_message_tests
        )
        lines = self._read_cfg_lines(self.test_cfg_path)
        self.assertEqual(self._get_signal_field(lines[target_line_index], 1), new_raw_name)
        # Ensure phase field is still there for analog if it was part of original line structure
        self.assertEqual(self._get_signal_field(lines[target_line_index], 2), "S")


    def test_manage_duplicate_signals_report_only(self):
        # names_A.cfg has two "AnalogDupe | phase: D"
        # Combined name for these is "AnalogDupe|phase:D"
        report_csv = os.path.join(self.base_temp_dir, "dupes.csv")
        self.manager.manage_duplicate_signals(
            self.source_dir,
            output_csv_duplicates_path=report_csv,
            auto_rename_analog=False,
            verbose_logging_method=self.is_print_message_tests
        )

        self.assertTrue(os.path.exists(report_csv))
        df = pd.read_csv(report_csv)

        # Check if the duplicate is reported
        duplicate_entry = df[df['signal_name'] == 'AnalogDupe|phase:D'] # Combined name format
        self.assertEqual(len(duplicate_entry), 1)
        self.assertEqual(duplicate_entry['count'].iloc[0], 2)
        self.assertEqual(duplicate_entry['file_name'].iloc[0], 'names_A.cfg')

        # Verify original file is unchanged
        lines_after = self._read_cfg_lines(self.test_cfg_path)
        # Line 3 (index 2) and 4 (index 3) in CFG are the duplicates
        self.assertEqual(self._get_signal_field(lines_after[3], 1), "AnalogDupe | phase: D")
        self.assertEqual(self._get_signal_field(lines_after[4], 1), "AnalogDupe | phase: D")

    def test_manage_duplicate_signals_auto_rename_analog(self):
        report_csv = os.path.join(self.base_temp_dir, "dupes_after_rename.csv")
        rename_log_csv = os.path.join(self.base_temp_dir, "rename_log.csv")

        self.manager.manage_duplicate_signals(
            self.source_dir,
            output_csv_duplicates_path=report_csv,
            auto_rename_analog=True,
            output_csv_rename_log_path=rename_log_csv,
            verbose_logging_method=self.is_print_message_tests
        )

        self.assertTrue(os.path.exists(rename_log_csv))
        log_df = pd.read_csv(rename_log_csv)

        # Expect one rename action for the second "AnalogDupe | phase: D"
        self.assertEqual(len(log_df), 1)
        action = log_df.iloc[0]
        self.assertEqual(action['old_combined_name'], 'AnalogDupe|phase:D')
        self.assertEqual(action['old_raw_name_field'], 'AnalogDupe | phase: D')
        # Default renaming strategy appends _1, _2 etc.
        # The first occurrence is kept, second is renamed.
        self.assertEqual(action['new_raw_name_field'], 'AnalogDupe | phase: D_1')
        self.assertTrue(action['new_combined_name'].startswith('AnalogDupe|phase:D_1')) # Phase part should be there

        lines_after = self._read_cfg_lines(self.test_cfg_path)
        # Original at CFG line 3 (index 2 in lines list)
        self.assertEqual(self._get_signal_field(lines_after[3], 1), "AnalogDupe | phase: D")
        # Renamed at CFG line 4 (index 3 in lines list)
        self.assertEqual(self._get_signal_field(lines_after[4], 1), "AnalogDupe | phase: D_1")
        self.assertEqual(self._get_signal_field(lines_after[4], 2), "D") # Phase preserved

if __name__ == '__main__':
    if not os.path.exists(SAMPLE_DATA_ROOT):
        print(f"Warning: Test sample data directory not found at {SAMPLE_DATA_ROOT}")
        print(f"Please run 'python tests/test_data_setup.py' from the project root to generate test data first.")
    unittest.main()
