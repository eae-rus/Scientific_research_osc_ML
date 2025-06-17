import unittest
import os
import sys
import shutil
import hashlib
import pandas as pd # Not strictly needed for these tests, but often useful

# Add project root to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from preprocessing.anonymizer import DataAnonymizer

# Define the base path for sample data relative to this test file's location
# Assumes this test file is in tests/ and sample_data is also in tests/
BASE_TEST_ROOT_DIR = os.path.dirname(__file__) # Should be /app/tests
SAMPLE_DATA_BASE_PATH = os.path.join(BASE_TEST_ROOT_DIR, "sample_data", "comtrade_files")

class TestDataAnonymizer(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.join(BASE_TEST_ROOT_DIR, "temp_anonymizer_data")

        # Paths to original sample files (these should exist from test_data_setup.py)
        self.path_to_anonymize_cfg_orig = os.path.join(SAMPLE_DATA_BASE_PATH, "for_anonymizer_1", "to_anonymize.cfg")
        self.path_to_anonymize_dat_orig = os.path.join(SAMPLE_DATA_BASE_PATH, "for_anonymizer_1", "to_anonymize.dat")
        self.path_to_cp1251_cfg_orig = os.path.join(SAMPLE_DATA_BASE_PATH, "encoding_tests", "cp1251_encoded.cfg")
        self.path_to_cp1251_dat_orig = os.path.join(SAMPLE_DATA_BASE_PATH, "encoding_tests", "cp1251_encoded.dat")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.anonymizer = DataAnonymizer(verbose_logging=False, show_progress_bars=False)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def test_anonymize_file_success_utf8(self):
        # Create a specific subdirectory for this test to avoid name clashes
        test_subdir = os.path.join(self.base_temp_dir, "test_success_utf8")
        os.makedirs(test_subdir, exist_ok=True)

        current_cfg_path = os.path.join(test_subdir, "to_anonymize.cfg")
        current_dat_path = os.path.join(test_subdir, "to_anonymize.dat")

        shutil.copy2(self.path_to_anonymize_cfg_orig, current_cfg_path)
        shutil.copy2(self.path_to_anonymize_dat_orig, current_dat_path)

        expected_hash = ""
        with open(current_dat_path, 'rb') as f_dat:
            expected_hash = hashlib.md5(f_dat.read()).hexdigest()

        result = self.anonymizer.anonymize_file(current_cfg_path, encoding='utf-8')
        self.assertTrue(result, "Anonymization should succeed for utf-8 file.")

        self.assertFalse(os.path.exists(current_cfg_path), "Original CFG file should be renamed.")
        self.assertFalse(os.path.exists(current_dat_path), "Original DAT file should be renamed.")

        new_cfg_path = os.path.join(test_subdir, expected_hash + ".cfg")
        new_dat_path = os.path.join(test_subdir, expected_hash + ".dat")
        self.assertTrue(os.path.exists(new_cfg_path), "New CFG file with hash name should exist.")
        self.assertTrue(os.path.exists(new_dat_path), "New DAT file with hash name should exist.")

        with open(new_cfg_path, 'r', encoding='utf-8') as f_cfg:
            lines = f_cfg.readlines()

        self.assertEqual(lines[0].strip(), ",,1991", "Station and device ID should be anonymized, rev_year preserved.")

        # For "to_anonymize.cfg": 1,1A,0D -> 1 channel total.
        # Line indices: 0=header, 1=counts, 2=analog_ch1
        # 3=freq, 4=nrates, 5=samp_rate/endsamp, 6=start_ts, 7=trigger_ts
        # So, start_ts is lines[6], trigger_ts is lines[7]
        # However, the anonymizer logic uses total_channels + 5 and total_channels + 6
        # total_channels = 1. start_date_line_index = 1 + 1+1+1+1+1 = 6. trigger_date_line_index = 7.
        self.assertEqual(lines[6].strip(), "01/01/0001,00:00:00.000000", "Start timestamp should be anonymized.")
        self.assertEqual(lines[7].strip(), "01/01/0001,00:00:00.000000", "Trigger timestamp should be anonymized.")


    def test_anonymize_file_dat_missing(self):
        test_subdir = os.path.join(self.base_temp_dir, "test_dat_missing")
        os.makedirs(test_subdir, exist_ok=True)

        temp_cfg_path = os.path.join(test_subdir, "only_cfg.cfg")
        shutil.copy2(self.path_to_anonymize_cfg_orig, temp_cfg_path) # Copy only CFG

        result = self.anonymizer.anonymize_file(temp_cfg_path, encoding='utf-8')
        self.assertFalse(result, "Anonymization should fail if DAT file is missing.")
        self.assertTrue(os.path.exists(temp_cfg_path), "Original CFG file should still exist if DAT was missing.")

    def test_anonymize_directory_mixed_content(self):
        source_for_dir_test = os.path.join(self.base_temp_dir, "source_for_dir_test")
        os.makedirs(source_for_dir_test, exist_ok=True)

        # Copy set 1 (to_anonymize)
        shutil.copy2(self.path_to_anonymize_cfg_orig, os.path.join(source_for_dir_test, "to_anonymize.cfg"))
        shutil.copy2(self.path_to_anonymize_dat_orig, os.path.join(source_for_dir_test, "to_anonymize.dat"))
        hash_anonymize = ""
        with open(os.path.join(source_for_dir_test, "to_anonymize.dat"), 'rb') as f: hash_anonymize = hashlib.md5(f.read()).hexdigest()

        # Copy set 2 (cp1251)
        shutil.copy2(self.path_to_cp1251_cfg_orig, os.path.join(source_for_dir_test, "cp1251_encoded.cfg"))
        shutil.copy2(self.path_to_cp1251_dat_orig, os.path.join(source_for_dir_test, "cp1251_encoded.dat"))
        hash_cp1251 = ""
        with open(os.path.join(source_for_dir_test, "cp1251_encoded.dat"), 'rb') as f: hash_cp1251 = hashlib.md5(f.read()).hexdigest()

        # Create a CFG with no DAT
        with open(os.path.join(source_for_dir_test, "only_cfg.cfg"), "w") as f: f.write("dummy cfg content")
        # Create a non-COMTRADE file
        with open(os.path.join(source_for_dir_test, "other.txt"), "w") as f: f.write("text file")

        self.anonymizer.anonymize_directory(source_for_dir_test)

        # Check anonymized files
        self.assertTrue(os.path.exists(os.path.join(source_for_dir_test, f"{hash_anonymize}.cfg")))
        self.assertTrue(os.path.exists(os.path.join(source_for_dir_test, f"{hash_anonymize}.dat")))
        self.assertTrue(os.path.exists(os.path.join(source_for_dir_test, f"{hash_cp1251}.cfg")))
        self.assertTrue(os.path.exists(os.path.join(source_for_dir_test, f"{hash_cp1251}.dat")))

        # Check originals are gone
        self.assertFalse(os.path.exists(os.path.join(source_for_dir_test, "to_anonymize.cfg")))
        self.assertFalse(os.path.exists(os.path.join(source_dir_test, "cp1251_encoded.cfg")))

        # Check other files
        self.assertTrue(os.path.exists(os.path.join(source_for_dir_test, "other.txt")), "Non-COMTRADE file should remain.")

        protected_log_path = os.path.join(source_for_dir_test, "protected_files.txt")
        self.assertTrue(os.path.exists(protected_log_path), "Protected files log should be created.")

        with open(protected_log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        self.assertIn("only_cfg.cfg", log_content, "Log should contain entry for CFG without DAT.")

if __name__ == '__main__':
    # This allows running the tests directly from this file
    # Ensure test_data_setup.py has been run once to create sample_data
    if not os.path.exists(SAMPLE_DATA_BASE_PATH):
        print(f"Warning: Test sample data directory not found at {SAMPLE_DATA_BASE_PATH}")
        print(f"Please run 'python tests/test_data_setup.py' from the project root to generate test data first.")
    unittest.main()
