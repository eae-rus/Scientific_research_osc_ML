import unittest
import os
import sys
import shutil
import json
import hashlib
import zipfile
import py7zr

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from filesystem.finder import OscillogramFinder, TYPE_OSC
from core.oscillogram import Oscillogram

# Attempt to import aspose.zip, set a flag
HAS_ASPOSE_ZIP = False
try:
    import aspose.zip as az
    HAS_ASPOSE_ZIP = True
except ImportError:
    # print("Aspose.ZIP not available, RAR tests might be skipped or fail if RARs are present.")
    pass


class TestOscillogramFinder(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = os.path.abspath("tests/temp_finder_data")
        self.sample_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))
        self.sample_data_comtrade_path = os.path.join(self.sample_data_root, "comtrade_files")
        self.sample_data_archive_path = os.path.join(self.sample_data_root, "archives")
        self.sample_data_config_path = os.path.join(self.sample_data_root, "config_files")

        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)
        os.makedirs(self.base_temp_dir, exist_ok=True)

        self.source_dir = os.path.join(self.base_temp_dir, "source_files")
        os.makedirs(self.source_dir, exist_ok=True)
        self.dest_dir = os.path.join(self.base_temp_dir, "dest_files")
        os.makedirs(self.dest_dir, exist_ok=True)

        # Populate source_dir
        # 1. CFG/DAT
        cfg_dat_src_dir = os.path.join(self.sample_data_comtrade_path, "valid_cfg_dat_1")
        shutil.copy(os.path.join(cfg_dat_src_dir, "sample_A.cfg"), self.source_dir)
        shutil.copy(os.path.join(cfg_dat_src_dir, "sample_A.dat"), self.source_dir)

        # 2. CFF
        cff_src_file = os.path.join(self.sample_data_comtrade_path, "valid_cff_1", "valid_cff_1.cff")
        shutil.copy(cff_src_file, self.source_dir)

        # 3. Dummy BRS (ensure this type is handled if TYPE_OSC.BRESLER exists)
        if hasattr(TYPE_OSC, 'BRESLER'): # Check if BRESLER type is defined
             with open(os.path.join(self.source_dir, "sample_X.brs"), "w") as f:
                f.write("dummy bresler content")

        # 4. Archives
        shutil.copy(os.path.join(self.sample_data_archive_path, "archive_A.zip"), self.source_dir)
        shutil.copy(os.path.join(self.sample_data_archive_path, "archive_B.7z"), self.source_dir)
        # Optional: Add a .rar if HAS_ASPOSE_ZIP and sample data exists
        if HAS_ASPOSE_ZIP and os.path.exists(os.path.join(self.sample_data_archive_path, "archive_C.rar")):
            shutil.copy(os.path.join(self.sample_data_archive_path, "archive_C.rar"), self.source_dir)


        # 5. Non-oscillogram file
        with open(os.path.join(self.source_dir, "non_osc.txt"), "w") as f:
            f.write("some text")

        # Create a subdirectory in source to test preserve_dir_structure
        self.source_subdir = os.path.join(self.source_dir, "subdir_src")
        os.makedirs(self.source_subdir, exist_ok=True)
        shutil.copy(os.path.join(cfg_dat_src_dir, "sample_B.cfg"), self.source_subdir) # Assuming sample_B exists
        shutil.copy(os.path.join(cfg_dat_src_dir, "sample_B.dat"), self.source_subdir)


        self.finder = OscillogramFinder(is_print_message=False, show_progress_bars=False)

    def tearDown(self):
        if os.path.exists(self.base_temp_dir):
            shutil.rmtree(self.base_temp_dir)

    def _get_file_hash(self, file_path):
        h = hashlib.md5()
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in 64k chunks
                if not data:
                    break
                h.update(data)
        return h.hexdigest()

    def test_copy_all_types_no_hashes_no_structure(self):
        # file_type_flags=None means all known types are processed
        copied_files_info = self.finder.copy_new_oscillograms(
            self.source_dir, self.dest_dir,
            use_hashes=False,
            preserve_dir_structure=False,
            file_type_flags=None
        )

        # Assertions for files directly in source_dir
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_A.cfg")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_A.dat")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFF", "valid_cff_1.cff")))
        if hasattr(TYPE_OSC, 'BRESLER'):
            self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "BRESLER", "sample_X.brs")))

        # Assertions for files from archives (archive_A.zip contains sample_arc_A.cfg/dat)
        # These should also go into COMTRADE_CFG_DAT when not preserving structure
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_arc_A.cfg")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_arc_A.dat")))

        # archive_B.7z contains sample_arc_A.cfg_in7z (renamed sample_arc_A.cfg)
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_arc_A.cfg_in7z")))
        # Assuming .dat also exists for it
        # self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_arc_A.dat_in7z")))

        # Assertions for files in source_subdir (sample_B.cfg/dat)
        # When not preserving structure, these should also be in the flat COMTRADE_CFG_DAT
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_B.cfg")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_B.dat")))

        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "non_osc.txt"))) # Should not be copied
        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "_hash_table.json"))) # use_hashes=False

    def test_copy_selective_types_with_hashes_and_structure(self):
        # Hash of sample_A.dat (which should exist from setup)
        original_sample_A_dat_path = os.path.join(self.sample_data_comtrade_path, "valid_cfg_dat_1", "sample_A.dat")
        dat_hash_A = self._get_file_hash(original_sample_A_dat_path)
        self.assertIsNotNone(dat_hash_A, "Failed to hash sample_A.dat for test setup")

        initial_hashes = {dat_hash_A: ["source_files/sample_A.cfg", "source_files/sample_A.dat"]}

        # Write initial_hashes to a temporary hash table file for the finder to load
        initial_hash_table_path = os.path.join(self.dest_dir, "_hash_table.json")
        with open(initial_hash_table_path, 'w') as f:
            json.dump(initial_hashes, f)

        file_flags = {type_enum: False for type_enum in TYPE_OSC}
        file_flags[TYPE_OSC.COMTRADE_CFF] = True
        file_flags[TYPE_OSC.ARCH_ZIP] = True
        file_flags[TYPE_OSC.COMTRADE_CFG_DAT] = True # To extract CFG/DAT from zip

        copied_files_info = self.finder.copy_new_oscillograms(
            self.source_dir, self.dest_dir,
            copied_hashes_input=None, # Finder will load from _hash_table.json in dest_dir
            use_hashes=True,
            preserve_dir_structure=True,
            file_type_flags=file_flags
        )

        # 1. sample_A.cfg/dat should NOT be copied because its hash is in initial_hashes
        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "sample_A.cfg")))
        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "sample_A.dat")))
        # Also check subdir just in case
        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "COMTRADE_CFG_DAT", "sample_A.cfg")))


        # 2. valid_cff_1.cff IS copied (COMTRADE_CFF type is enabled) and structure preserved
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "valid_cff_1.cff")))

        # 3. sample_arc_A.cfg/dat (from archive_A.zip) IS copied
        # With preserve_dir_structure=True, files from archive root are extracted to dest_dir root.
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "sample_arc_A.cfg")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "sample_arc_A.dat")))

        # 4. sample_X.brs should NOT be copied (BRESLER type is False in file_flags)
        if hasattr(TYPE_OSC, 'BRESLER'):
            self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "sample_X.brs")))
            self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "BRESLER", "sample_X.brs")))


        # 5. sample_B.cfg/dat from source_dir/subdir_src
        # COMTRADE_CFG_DAT is True, preserve_dir_structure is True. Hash not in initial_hashes.
        # So it should be copied to dest_dir/subdir_src/sample_B.cfg
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "subdir_src", "sample_B.cfg")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "subdir_src", "sample_B.dat")))


        # 6. Verify hash tables
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "_hash_table.json")))
        newly_copied_json_path = None
        for fname in os.listdir(self.dest_dir):
            if fname.startswith("newly_copied_hashes_") and fname.endswith(".json"):
                newly_copied_json_path = os.path.join(self.dest_dir, fname)
                break
        self.assertIsNotNone(newly_copied_json_path, "Newly copied hashes JSON not found")

        with open(os.path.join(self.dest_dir, "_hash_table.json"), 'r') as f:
            main_hashes = json.load(f)
        with open(newly_copied_json_path, 'r') as f:
            newly_copied_hashes = json.load(f)

        self.assertIn(dat_hash_A, main_hashes) # Original hash should still be there

        # Hash of sample_arc_A.dat (from zip)
        # Need to create a dummy .dat for sample_arc_A.cfg to hash it for the test data setup
        # For this test, let's assume test_data_setup.py creates sample_data/comtrade_files/for_archive_1/sample_arc_A.dat
        original_arc_A_dat_path = os.path.join(self.sample_data_comtrade_path, "for_archive_1", "sample_arc_A.dat")
        if os.path.exists(original_arc_A_dat_path): # Make this check conditional on test data setup
            dat_hash_arc_A = self._get_file_hash(original_arc_A_dat_path)
            self.assertIsNotNone(dat_hash_arc_A)
            self.assertIn(dat_hash_arc_A, main_hashes)
            self.assertIn(dat_hash_arc_A, newly_copied_hashes)
            # Check path in main_hashes for sample_arc_A.dat (preserve_dir_structure=True)
            self.assertEqual(main_hashes[dat_hash_arc_A][1], "sample_arc_A.dat")


        # Hash for sample_B.dat
        original_sample_B_dat_path = os.path.join(self.sample_data_comtrade_path, "valid_cfg_dat_1", "sample_B.dat")
        dat_hash_B = self._get_file_hash(original_sample_B_dat_path)
        self.assertIsNotNone(dat_hash_B)
        self.assertIn(dat_hash_B, main_hashes)
        self.assertIn(dat_hash_B, newly_copied_hashes)
        # Check path for sample_B.dat (preserve_dir_structure=True)
        self.assertEqual(main_hashes[dat_hash_B][1], os.path.join("subdir_src", "sample_B.dat"))


    def test_find_terminal_hashes_from_json(self):
        input_json_path = os.path.join(self.sample_data_config_path, "terminal_hashes_input.json")
        self.assertTrue(os.path.exists(input_json_path), f"Test input file not found: {input_json_path}")

        output_json_path = os.path.join(self.base_temp_dir, "term_out.json")

        # Assuming terminal_hashes_input.json has:
        # { "terminal_data": { "123": ["hash1", "hash2"], "456": ["hash3"] } }
        OscillogramFinder.find_terminal_hashes_from_json(input_json_path, [123, 789], output_json_path)

        self.assertTrue(os.path.exists(output_json_path))
        with open(output_json_path, 'r') as f:
            result = json.load(f)

        self.assertIn("123", result)
        self.assertEqual(result["123"], ["hash1", "hash2"])
        self.assertNotIn("456", result) # Not requested
        self.assertNotIn("789", result) # Requested but not in input file

    def test_find_oscillograms_with_neutral_current(self):
        neutral_dir = os.path.join(self.base_temp_dir, "neutral_test_src")
        os.makedirs(neutral_dir, exist_ok=True)

        # Files for this test are expected in sample_data/comtrade_files/specific_signals_1/
        neutral_src_dir = os.path.join(self.sample_data_comtrade_path, "specific_signals_1")

        # File with neutral current
        shutil.copy(os.path.join(neutral_src_dir, "neutral_current.cfg"), neutral_dir)
        shutil.copy(os.path.join(neutral_src_dir, "neutral_current.dat"), neutral_dir)

        # File without neutral current (using signals_for_checker as example)
        shutil.copy(os.path.join(neutral_src_dir, "signals_for_checker.cfg"), neutral_dir)
        shutil.copy(os.path.join(neutral_src_dir, "signals_for_checker.dat"), neutral_dir)

        output_txt_path = os.path.join(self.base_temp_dir, "neutral_report.txt")

        self.finder.find_oscillograms_with_neutral_current(neutral_dir, output_txt_path)

        self.assertTrue(os.path.exists(output_txt_path))
        with open(output_txt_path, 'r') as f:
            content = f.read()

        self.assertIn("neutral_current.cfg", content)
        self.assertNotIn("signals_for_checker.cfg", content)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
