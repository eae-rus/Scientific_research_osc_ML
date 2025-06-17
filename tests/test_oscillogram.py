import unittest
import os
import sys
import datetime
import pandas as pd
import hashlib

# Add project root to sys.path to allow importing core.oscillogram
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.oscillogram import Oscillogram

# Define the base path for sample data relative to this test file's location
# Assuming this test file is in tests/ and sample_data is also in tests/
BASE_SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

class TestOscillogramLoading(unittest.TestCase):

    def test_load_valid_cfg_dat(self):
        cfg_path = os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "valid_cfg_dat_1_subdir", "sample_A.cfg")
        dat_path = os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "valid_cfg_dat_1_subdir", "sample_A.dat")

        self.assertTrue(os.path.exists(cfg_path), f"CFG file not found for test: {cfg_path}")
        self.assertTrue(os.path.exists(dat_path), f"DAT file not found for test: {dat_path}")

        osc = Oscillogram(cfg_path)

        self.assertEqual(osc.station_name, "SampleStationA")
        self.assertEqual(osc.rec_dev_id, "DeviceA")
        self.assertEqual(osc.rev_year, "1999") # comtrade_APS stores rev_year as str
        self.assertEqual(osc.frequency, 50.0)

        # Specifics from sample_A.cfg created by test_data_setup.py
        # total analog + digital channels
        self.assertEqual(osc.channels_count, 2)
        self.assertEqual(osc.analog_count, 1)
        self.assertEqual(osc.status_count, 1)
        self.assertEqual(osc.total_samples, 3) # From samp_count in CFG

        self.assertIsInstance(osc.start_timestamp, datetime.datetime)
        # Date from sample_A.cfg: 01/01/2024,10:00:00.000000
        self.assertEqual(osc.start_timestamp, datetime.datetime(2024, 1, 1, 10, 0, 0, 0))

        self.assertListEqual(osc.analog_channel_ids, ["U_PhaseA"])
        self.assertListEqual(osc.status_channel_ids, ["Status_1"])

        self.assertIsInstance(osc.data_frame, pd.DataFrame)
        self.assertEqual(len(osc.data_frame), 3) # Number of samples

        expected_columns = {'time', 'U_PhaseA', 'Status_1'}
        self.assertTrue(expected_columns.issubset(set(osc.data_frame.columns)))

        # Calculate hash of the DAT file
        dat_hash = None
        with open(dat_path, 'rb') as f_dat:
            dat_hash = hashlib.md5(f_dat.read()).hexdigest()
        self.assertEqual(osc.file_hash, dat_hash)

    def test_load_valid_cff(self):
        cff_path = os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "valid_cff_1.cff")
        self.assertTrue(os.path.exists(cff_path), f"CFF file not found for test: {cff_path}")

        osc = Oscillogram(cff_path)

        self.assertEqual(osc.station_name, "SampleStationCFF")
        self.assertEqual(osc.rec_dev_id, "DeviceCFF")
        self.assertEqual(osc.rev_year, "2013")
        self.assertEqual(osc.frequency, 60.0)
        self.assertEqual(osc.channels_count, 1)
        self.assertEqual(osc.analog_count, 1)
        self.assertEqual(osc.status_count, 0)
        self.assertEqual(osc.total_samples, 2)

        self.assertIsInstance(osc.start_timestamp, datetime.datetime)
        # Date from valid_cff_1.cff: 15/07/2024,12:30:00.123456
        self.assertEqual(osc.start_timestamp, datetime.datetime(2024, 7, 15, 12, 30, 0, 123456))

        self.assertListEqual(osc.analog_channel_ids, ["I_PhaseC"])
        self.assertListEqual(osc.status_channel_ids, [])

        self.assertIsInstance(osc.data_frame, pd.DataFrame)
        self.assertEqual(len(osc.data_frame), 2)

        expected_cff_columns = {'time', 'I_PhaseC'}
        self.assertTrue(expected_cff_columns.issubset(set(osc.data_frame.columns)))

        # For CFF with embedded ASCII data, Oscillogram._dat_file_path might be None or the CFF path itself.
        # If _dat_file_path is None, hash might be of filename. If it's CFF path, hash is of CFF.
        # The current Oscillogram._calculate_file_hash logic:
        # Tries self._dat_file_path first. If CFF, self._dat_file_path is None initially.
        # Then it might try self.filepath if _dat_file_path isn't valid.
        # So, for CFF, it should hash the CFF file itself if no external DAT is specified/found.
        cff_hash = None
        with open(cff_path, 'rb') as f_cff:
            cff_hash = hashlib.md5(f_cff.read()).hexdigest()
        self.assertEqual(osc.file_hash, cff_hash)


    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            Oscillogram("nonexistent_dummy_file.cfg")

    def test_load_malformed_cfg(self):
        malformed_cfg_path = os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "malformed_cfg_files", "malformed_cfg_1.cfg")
        self.assertTrue(os.path.exists(malformed_cfg_path), f"Malformed CFG file not found for test: {malformed_cfg_path}")

        # Oscillogram wraps comtrade_APS parsing errors into RuntimeError
        with self.assertRaises(RuntimeError):
            Oscillogram(malformed_cfg_path)

class TestOscillogramDataAccess(unittest.TestCase):
    def setUp(self):
        """Set up for data access tests using a valid oscillogram."""
        self.cfg_path = os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "valid_cfg_dat_1_subdir", "sample_A.cfg")
        self.assertTrue(os.path.exists(self.cfg_path), f"CFG file not found for setUp: {self.cfg_path}")
        try:
            self.osc = Oscillogram(self.cfg_path)
        except Exception as e:
            self.fail(f"Oscillogram instantiation failed in setUp: {e}")

        # Expected data from sample_A.dat (1,0,1.23,0; 2,1000,2.34,1; 3,2000,3.45,0)
        self.expected_U_PhaseA_data = [1.23, 2.34, 3.45]
        self.expected_Status_1_data = [0, 1, 0]
        # Expected time values in seconds (sample_A.cfg: 1000Hz sample rate)
        # DAT timestamps: 0, 1000us, 2000us
        self.expected_time_values = [0.0, 0.001, 0.002]


    def test_cfg_property(self):
        self.assertIsNotNone(self.osc.cfg)
        self.assertEqual(self.osc.cfg.station_name, "SampleStationA")
        self.assertEqual(self.osc.cfg.rec_dev_id, "DeviceA")
        self.assertEqual(self.osc.cfg.rev_year, "1999")

    def test_time_values_property(self):
        # comtrade_APS returns time as array.array, which behaves like a list
        self.assertTrue(isinstance(self.osc.time_values, list) or isinstance(self.osc.time_values, type(array.array('f'))))
        self.assertEqual(len(self.osc.time_values), 3)
        for i, expected_time in enumerate(self.expected_time_values):
            self.assertAlmostEqual(self.osc.time_values[i], expected_time, places=6)

    def test_get_analog_series_valid(self):
        series = self.osc.get_analog_series("U_PhaseA")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(series.name, "U_PhaseA")
        # Compare values, ignoring dtype for flexibility (e.g. float64 vs float32)
        pd.testing.assert_series_equal(
            series,
            pd.Series(self.expected_U_PhaseA_data, name="U_PhaseA", index=self.expected_time_values), # Add index for proper comparison
            check_dtype=False,
            check_index_type=False # Index might be float64 from Series, vs array.array's time
        )

    def test_get_analog_series_invalid(self):
        with self.assertRaises(KeyError):
            self.osc.get_analog_series("NonExistentAnalogChannel")

    def test_get_status_series_valid(self):
        series = self.osc.get_status_series("Status_1")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(series.name, "Status_1")
        pd.testing.assert_series_equal(
            series,
            pd.Series(self.expected_Status_1_data, name="Status_1", index=self.expected_time_values),
            check_dtype=False,
            check_index_type=False
        )

    def test_get_status_series_invalid(self):
        with self.assertRaises(KeyError):
            self.osc.get_status_series("NonExistentStatusChannel")

if __name__ == '__main__':
    # This allows running the tests directly from this file
    # Ensure test_data_setup.py has been run once to create sample_data

    # Need to import array for type check in test_time_values_property if using it
    import array

    if not os.path.exists(BASE_SAMPLE_DATA_DIR):
        print("Warning: Test sample data directory not found.")
        print(f"Please run 'python tests/test_data_setup.py' from the project root to generate test data first.")

    # Check if test_data_setup.py needs to be run
    # This is a simple check, more robust would be to verify specific files needed by tests
    if not os.path.exists(os.path.join(BASE_SAMPLE_DATA_DIR, "comtrade_files", "valid_cfg_dat_1_subdir", "sample_A.cfg")):
        print("Sample data for TestOscillogramDataAccess not found. Attempting to run test_data_setup.py...")
        try:
            # Assuming test_data_setup.py is in the same directory as this test file
            # Or adjust path if it's elsewhere e.g. project root/tests/
            setup_script_path = os.path.join(os.path.dirname(__file__), "test_data_setup.py")
            if os.path.exists(setup_script_path):
                # This execution might be problematic depending on environment and cwd for test_data_setup.py
                # It's generally better to run setup script separately.
                # For now, just print a stronger message.
                print(f"Please ensure '{setup_script_path}' has been run successfully.")
            else:
                print(f"Setup script {setup_script_path} not found.")
        except Exception as e:
            print(f"Error trying to run test_data_setup.py: {e}")

    unittest.main()
