import unittest
import os
import sys
import shutil
import datetime
import csv
import hashlib
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union # For type hints if needed by moved methods

# Add project root to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from filesystem.organizer import FileOrganizer

# Define the base path for sample data relative to this test file's location
BASE_TEST_ROOT_DIR = os.path.dirname(__file__) # Should be /app/tests
SAMPLE_DATA_ROOT = os.path.join(BASE_TEST_ROOT_DIR, "sample_data")
COMTRADE_SAMPLES_PATH = os.path.join(SAMPLE_DATA_ROOT, "comtrade_files")

class TestFileOrganizer(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.join(BASE_TEST_ROOT_DIR, "temp_organizer_data")

        # Ensure sample data is available
        if not os.path.exists(SAMPLE_DATA_ROOT):
            self.fail(f"Sample data directory not found at {SAMPLE_DATA_ROOT}. Run test_data_setup.py.")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir)

        self.organizer = FileOrganizer()
        self.is_print_message_tests = False # Control verbosity of tested methods

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_update_modification_times(self):
        test_subdir = os.path.join(self.base_temp_dir, "mod_time_test")
        os.makedirs(test_subdir)

        file_to_test = os.path.join(test_subdir, "test_file.txt")
        with open(file_to_test, "w") as f:
            f.write("some content")

        initial_stat = os.stat(file_to_test)
        # Ensure we sleep a bit if system clock resolution is low, to see a change
        # However, setting to a specific old date should always show a change.

        custom_dt = datetime.datetime(2020, 1, 1, 10, 30, 0)
        self.organizer.update_modification_times(test_subdir, new_mod_time=custom_dt)

        updated_stat = os.stat(file_to_test)
        self.assertAlmostEqual(updated_stat.st_mtime, custom_dt.timestamp())

    def test_extract_frequencies_from_cfg_valid_utf8(self):
        cfg_path = os.path.join(COMTRADE_SAMPLES_PATH, "valid_cfg_dat_1_subdir", "sample_A.cfg")
        self.assertTrue(os.path.exists(cfg_path), f"Test file not found: {cfg_path}")

        f_net, f_rate = self.organizer.extract_frequencies_from_cfg(cfg_path, encoding='utf-8')
        self.assertEqual(f_net, 50)
        self.assertEqual(f_rate, 1000)

    def test_extract_frequencies_from_cfg_encoding_cp1251(self):
        cfg_path = os.path.join(COMTRADE_SAMPLES_PATH, "encoding_tests", "cp1251_encoded.cfg")
        self.assertTrue(os.path.exists(cfg_path), f"Test file not found: {cfg_path}")
        # From test_data_setup.py: "50\n1\n1000,1" -> 50Hz, 1000sps
        f_net, f_rate = self.organizer.extract_frequencies_from_cfg(cfg_path, encoding='cp1251')
        self.assertEqual(f_net, 50)
        self.assertEqual(f_rate, 1000)

    def test_extract_frequencies_from_cfg_malformed(self):
        cfg_path = os.path.join(COMTRADE_SAMPLES_PATH, "malformed_cfg_files", "malformed_cfg_1.cfg")
        self.assertTrue(os.path.exists(cfg_path), f"Test file not found: {cfg_path}")

        f_net, f_rate = self.organizer.extract_frequencies_from_cfg(cfg_path, encoding='utf-8')
        self.assertIsNone(f_net, "Network frequency should be None for malformed CFG")
        self.assertIsNone(f_rate, "Sample rate should be None for malformed CFG")

    def test_group_files_by_frequency_and_rate(self):
        group_source_dir = os.path.join(self.base_temp_dir, "group_source")
        os.makedirs(group_source_dir)

        # Pair 1: sample_A (50Hz, 1000sps)
        shutil.copy2(os.path.join(COMTRADE_SAMPLES_PATH, "valid_cfg_dat_1_subdir", "sample_A.cfg"), group_source_dir)
        shutil.copy2(os.path.join(COMTRADE_SAMPLES_PATH, "valid_cfg_dat_1_subdir", "sample_A.dat"), group_source_dir)

        # Pair 2: valid_cff_1.cff (60Hz, 2000sps) - needs DAT for grouping by FileOrganizer
        # FileOrganizer.group_files_by_frequency_and_rate currently expects .cfg/.dat pairs.
        # To test grouping with a different frequency, let's use another .cfg/.dat pair.
        # We'll use cp866_encoded as sample_B, assuming its frequencies are different.
        # From test_data_setup.py, cp866_encoded.cfg is also 50Hz, 1000sps.
        # We need a file with different frequencies for a meaningful grouping test.
        # Let's assume test_data_setup.py creates "sample_B.cfg/dat" with 60Hz, 2000sps.
        # For now, we'll simulate this by copying and renaming one of the existing files
        # and then creating a new CFG content for it.

        sample_B_path = os.path.join(group_source_dir, "sample_B.cfg")
        sample_B_dat_path = os.path.join(group_source_dir, "sample_B.dat")
        cfg_content_B = """SampleStationB,DeviceB,1999
1,1A,0D
1,U_PhaseB,B,,kV,1.0,0,0,-32767,32767,10,0.1,P
60.0
1
2000,3
01/01/2024,11:00:00.000000
01/01/2024,11:00:00.001000
ASCII
1.0
"""
        dat_content_B = "1,0,5.67\n2,500,6.78\n3,1000,7.89\n"
        with open(sample_B_path, "w", encoding="utf-8") as f: f.write(cfg_content_B)
        with open(sample_B_dat_path, "w", encoding="utf-8") as f: f.write(dat_content_B)

        # Malformed CFG (should remain)
        shutil.copy2(os.path.join(COMTRADE_SAMPLES_PATH, "malformed_cfg_files", "malformed_cfg_1.cfg"), group_source_dir)
        # CFG with no DAT (should remain)
        with open(os.path.join(group_source_dir, "cfg_no_dat.cfg"), "w") as f: f.write("dummy content")

        self.organizer.group_files_by_frequency_and_rate(group_source_dir, is_print_message=self.is_print_message_tests)

        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "f_network = 50 and f_rate = 1000", "sample_A.cfg")))
        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "f_network = 50 and f_rate = 1000", "sample_A.dat")))
        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "f_network = 60 and f_rate = 2000", "sample_B.cfg")))
        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "f_network = 60 and f_rate = 2000", "sample_B.dat")))

        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "malformed_cfg_1.cfg")), "Malformed CFG should remain in source if not processed.")
        self.assertTrue(os.path.exists(os.path.join(group_source_dir, "cfg_no_dat.cfg")), "CFG without DAT should remain in source.")

    def test_generate_dat_hash_inventory(self):
        inventory_source_dir = os.path.join(self.base_temp_dir, "inventory_source")

        # DAT file 1 for 50Hz, 1000sps group
        dir1 = os.path.join(inventory_source_dir, "f_network = 50 and f_rate = 1000")
        os.makedirs(dir1)
        dat1_path = os.path.join(dir1, "dat1_sampleA.dat")
        shutil.copy2(os.path.join(COMTRADE_SAMPLES_PATH, "valid_cfg_dat_1_subdir", "sample_A.dat"), dat1_path)
        h1 = ""
        with open(dat1_path, 'rb') as f: h1 = hashlib.md5(f.read()).hexdigest()

        # DAT file 2 for 60Hz, 2000sps group
        dir2 = os.path.join(inventory_source_dir, "f_network = 60 and f_rate = 2000")
        os.makedirs(dir2)
        dat2_content_B = "1,0,5.67\n2,500,6.78\n3,1000,7.89\n" # Content for sample_B.dat used above
        dat2_path = os.path.join(dir2, "dat2_sampleB.dat")
        with open(dat2_path, "w", encoding="utf-8") as f: f.write(dat2_content_B)
        h2 = ""
        with open(dat2_path, 'rb') as f: h2 = hashlib.md5(f.read()).hexdigest()

        output_csv = os.path.join(self.base_temp_dir, "inventory.csv")
        self.organizer.generate_dat_hash_inventory(inventory_source_dir, output_csv)

        self.assertTrue(os.path.exists(output_csv))

        df = pd.read_csv(output_csv)
        self.assertListEqual(list(df.columns), ['Номер', 'Частота сети', 'Частота дискретизации', 'Имя файла', 'Хеш'])
        self.assertEqual(len(df), 2)

        # Check content (order might vary, so check for presence of rows)
        row1_expected = [1, "50", "1000", "dat1_sampleA.dat", h1]
        row2_expected = [2, "60", "2000", "dat2_sampleB.dat", h2]

        # Convert df rows to lists of strings for easier comparison
        df_values_as_str_lists = []
        for _, row in df.iterrows():
            df_values_as_str_lists.append([str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]), str(row.iloc[3]), str(row.iloc[4])])

        # Convert expected rows to lists of strings
        row1_expected_str = [str(x) for x in row1_expected]
        row2_expected_str = [str(x) for x in row2_expected]

        self.assertIn(row1_expected_str, df_values_as_str_lists)
        self.assertIn(row2_expected_str, df_values_as_str_lists)

if __name__ == '__main__':
    if not os.path.exists(SAMPLE_DATA_ROOT):
        print(f"Warning: Test sample data directory not found at {SAMPLE_DATA_ROOT}")
        print(f"Please run 'python tests/test_data_setup.py' from the project root to generate test data first.")
    unittest.main()
