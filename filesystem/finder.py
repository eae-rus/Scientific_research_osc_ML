import os
import shutil
import hashlib
import json
import datetime
import py7zr
import zipfile
from enum import Enum
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Set # Added Set for type hints
from core.oscillogram import Oscillogram # Added import

# Attempt to import aspose.zip, but allow failure if not installed
try:
    import aspose.zip as az
except ImportError:
    az = None # Will check for None before using

# Define TYPE_OSC Enum as it exists in search_oscillograms.py
class TYPE_OSC(Enum):
    #  ?? WTF ??
    COMTRADE_CFG_DAT = "COMTRADE files (*.cfg/*.dat)"
    COMTRADE_CFF = "COMTRADE files (*.cff)"
    ARCH_7Z = "Archive files (*.7z)"
    ARCH_ZIP = "Archive files (*.zip)"
    ARCH_RAR = "Archive files (*.rar)"
    EVA = "EVA files (*.eva)"
    BRESLER = "Bresler files (*.Испытание)" # Using a more generic name for the extension type
    NIIM = "NIIM files (*.Авария)" # Using a more generic name
    KRUG = "KRUG files (*. Αναγν)" # Using a more generic name
    EKRA = "EKRA files (*.ekg)"
    # Add other types as needed

class OscillogramFinder:
    def __init__(self, is_print_message: bool = False):
        self.is_print_message = is_print_message
        self.CFG_EXTENSION = '.cfg'
        self.DAT_EXTENSION = '.dat'
        self.CFF_EXTENSION = '.cff'
        self.ARCH_7Z_EXTENSION = '.7z'
        self.ARCH_ZIP_EXTENSION = '.zip'
        self.ARCH_RAR_EXTENSION = '.rar'
        self.EVA_EXTENSION = '.eva'
        self.BRESLER_EXTENSION = '.испытание' # Example, adjust to actual extension if different
        self.NIIM_EXTENSION = '.авария'       # Example
        self.KRUG_EXTENSION = '.αναγν'       # Example
        self.EKRA_EXTENSION = '.ekg'

        # For tracking hashes during a single top-level run of copy_new_oscillograms
        self._current_run_newly_copied_hashes: Dict[str, str] = {}


    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError:
            if self.is_print_message:
                print(f"Error reading file for hashing: {file_path}")
            return None
        except Exception as e:
            if self.is_print_message:
                print(f"Unexpected error hashing file {file_path}: {e}")
            return None

    def _process_file(self, file_path: str, root_path: str, source_dir_base: str, dest_dir_base: str,
                      copied_hashes: Dict[str, str], new_copied_hashes_this_run: Dict[str, str],
                      preserve_dir_structure: bool, use_hashes: bool,
                      type_osc_enum_member: TYPE_OSC, specific_folder_name: str) -> int:
        file_name = os.path.basename(file_path)

        file_to_hash = file_path # For most single files, hash the file itself

        # For COMTRADE CFG/DAT, this method is called for the CFG, but DAT is primary for hash
        # This specific case is handled by _process_comtrade_file calling this for CFG after DAT.
        # So, if we are here for a CFG via _process_comtrade_file, the DAT hash is already managed.

        file_hash = self._calculate_file_hash(file_to_hash)
        if use_hashes and file_hash is None:
            if self.is_print_message: print(f"  Skipping {file_name} (hashing failed).")
            return 0

        if use_hashes and file_hash in copied_hashes:
            if self.is_print_message and self.is_print_message == "full_log": # very verbose
                 print(f"  Skipping {file_name} (hash {file_hash} already exists in main table).")
            return 0

        if use_hashes and file_hash in new_copied_hashes_this_run:
            if self.is_print_message and self.is_print_message == "full_log":
                 print(f"  Skipping {file_name} (hash {file_hash} already copied in this run).")
            return 0

        relative_dir = ""
        if preserve_dir_structure:
            relative_dir = os.path.relpath(root_path, source_dir_base)
            if relative_dir == ".": relative_dir = ""

        dest_subdir = os.path.join(dest_dir_base, specific_folder_name, relative_dir)
        os.makedirs(dest_subdir, exist_ok=True)

        dest_file_path = os.path.join(dest_subdir, file_name)

        try:
            shutil.copy2(file_path, dest_file_path)
            if self.is_print_message:
                print(f"  Copied: {file_name} to {dest_subdir}")
            if use_hashes and file_hash: # file_hash could be None if hashing failed but use_hashes was false initially
                new_copied_hashes_this_run[file_hash] = os.path.join(relative_dir, file_name) # Store relative path
            return 1
        except Exception as e:
            if self.is_print_message:
                print(f"  Error copying {file_name} to {dest_subdir}: {e}")
            return 0

    def _process_comtrade_file(self, cfg_file_path: str, root_path: str, source_dir_base: str, dest_dir_base: str,
                               copied_hashes: Dict[str, str], new_copied_hashes_this_run: Dict[str, str],
                               preserve_dir_structure: bool, use_hashes: bool) -> int:
        cfg_file_name = os.path.basename(cfg_file_path)
        dat_file_name = cfg_file_name[:-len(self.CFG_EXTENSION)] + self.DAT_EXTENSION
        dat_file_path = os.path.join(root_path, dat_file_name)

        if not os.path.exists(dat_file_path):
            if self.is_print_message:
                print(f"  Skipping CFG {cfg_file_name} (DAT file {dat_file_name} not found).")
            return 0

        dat_hash = self._calculate_file_hash(dat_file_path)
        if use_hashes:
            if dat_hash is None:
                if self.is_print_message: print(f"  Skipping {cfg_file_name} (DAT hashing failed).")
                return 0
            if dat_hash in copied_hashes:
                if self.is_print_message and self.is_print_message == "full_log":
                    print(f"  Skipping {cfg_file_name} (DAT hash {dat_hash} already exists in main table).")
                return 0
            if dat_hash in new_copied_hashes_this_run: # Check if DAT already copied in this run
                if self.is_print_message and self.is_print_message == "full_log":
                    print(f"  Skipping {cfg_file_name} (DAT hash {dat_hash} already copied in this run).")
                return 0

        # Determine destination paths
        specific_folder_name = "COMTRADE_CFG_DAT"
        relative_dir = ""
        if preserve_dir_structure:
            relative_dir = os.path.relpath(root_path, source_dir_base)
            if relative_dir == ".": relative_dir = ""

        dest_subdir = os.path.join(dest_dir_base, specific_folder_name, relative_dir)
        os.makedirs(dest_subdir, exist_ok=True)

        dest_cfg_path = os.path.join(dest_subdir, cfg_file_name)
        dest_dat_path = os.path.join(dest_subdir, dat_file_name)

        copied_cfg = False
        copied_dat = False
        try:
            shutil.copy2(cfg_file_path, dest_cfg_path)
            copied_cfg = True
            shutil.copy2(dat_file_path, dest_dat_path)
            copied_dat = True

            if self.is_print_message:
                print(f"  Copied: {cfg_file_name} and {dat_file_name} to {dest_subdir}")
            if use_hashes and dat_hash: # dat_hash must exist if we are here
                 new_copied_hashes_this_run[dat_hash] = os.path.join(relative_dir, dat_file_name) # Store relative path of DAT
            return 1 # Count as one pair
        except Exception as e:
            if self.is_print_message:
                print(f"  Error copying pair {cfg_file_name}/{dat_file_name} to {dest_subdir}: {e}")
            # Rollback if one part failed
            if copied_cfg and not copied_dat and os.path.exists(dest_cfg_path): os.remove(dest_cfg_path)
            if copied_dat and not copied_cfg and os.path.exists(dest_dat_path): os.remove(dest_dat_path) # Should not happen
            return 0


    def _process_archive_file(self, archive_file_path: str, root_path: str, source_dir_base: str, dest_dir_base: str,
                              copied_hashes: Dict[str, str], new_copied_hashes_this_run: Dict[str, str],
                              preserve_dir_structure: bool, use_hashes: bool,
                              file_type_flags: Dict[TYPE_OSC, bool], # Pass flags for recursive call
                              current_archive_depth: int = 0, max_archive_depth: int = 2) -> int:

        if current_archive_depth >= max_archive_depth:
            if self.is_print_message:
                print(f"  Skipping archive {archive_file_path} (max recursion depth {max_archive_depth} reached).")
            return 0

        file_name = os.path.basename(archive_file_path)
        temp_extract_path = os.path.join(dest_dir_base, "_temp_extract", file_name + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))

        copied_count_from_archive = 0
        try:
            os.makedirs(temp_extract_path, exist_ok=True)
            if self.is_print_message: print(f"  Extracting archive: {file_name} to {temp_extract_path}")

            if archive_file_path.lower().endswith(self.ARCH_7Z_EXTENSION):
                with py7zr.SevenZipFile(archive_file_path, mode='r') as z:
                    z.extractall(path=temp_extract_path)
            elif archive_file_path.lower().endswith(self.ARCH_ZIP_EXTENSION):
                with zipfile.ZipFile(archive_file_path, 'r') as z:
                    z.extractall(path=temp_extract_path)
            elif az and archive_file_path.lower().endswith(self.ARCH_RAR_EXTENSION):
                with az.Archive(archive_file_path) as z:
                    z.extract_to_directory(temp_extract_path)
            else:
                if self.is_print_message: print(f"  Unsupported archive type: {file_name}")
                return 0

            if self.is_print_message: print(f"  Successfully extracted {file_name}.")

            # Recursively call copy_new_oscillograms on the extracted content
            # Note: The dest_dir for this recursive call should still be based on dest_dir_base,
            # and preserve_dir_structure should adjust paths relative to temp_extract_path.
            # The specific_folder_name will be determined by the file types found within the archive.

            # Create a temporary sub-instance or call a modified internal walker
            # For simplicity, let's call a generalized walker that can be part of copy_new_oscillograms
            # or a new helper. The original passed all flags.

            # The key is how the relative path for structure preservation is handled.
            # New source_dir_base for recursive call is temp_extract_path.
            # Files copied from archive will go into dest_dir_base / specific_type_folder / relative_path_inside_archive.

            # This internal call needs to use the *same* new_copied_hashes_this_run dictionary
            # to ensure hashes are unique across the entire top-level operation.
            copied_count_from_archive = self._walk_and_process(
                source_dir=temp_extract_path,
                dest_dir_base=dest_dir_base, # Keep original top-level destination
                source_dir_base_for_relative_paths=temp_extract_path, # Paths inside archive are relative to here
                copied_hashes=copied_hashes,
                new_copied_hashes_this_run=new_copied_hashes_this_run, # Pass down the same dict
                preserve_dir_structure=preserve_dir_structure,
                use_hashes=use_hashes,
                file_type_flags=file_type_flags,
                current_archive_depth=current_archive_depth + 1,
                max_archive_depth=max_archive_depth
                # progress_callback, stop_processing_fn, is_write_names_fn are not used by _walk_and_process directly
            )

        except Exception as e:
            if self.is_print_message:
                print(f"  Error processing archive {file_name}: {e}")
        finally:
            if os.path.exists(temp_extract_path):
                try:
                    shutil.rmtree(temp_extract_path)
                except Exception as e:
                    if self.is_print_message: print(f"  Error cleaning up temp directory {temp_extract_path}: {e}")

        return copied_count_from_archive

    def _walk_and_process(self, source_dir: str, dest_dir_base: str, source_dir_base_for_relative_paths: str,
                          copied_hashes: Dict[str, str], new_copied_hashes_this_run: Dict[str, str],
                          preserve_dir_structure: bool, use_hashes: bool,
                          file_type_flags: Dict[TYPE_OSC, bool],
                          current_archive_depth: int, max_archive_depth: int,
                          pbar: Optional[tqdm] = None) -> int:
        """Internal walker to process files and sub-archives."""
        newly_copied_count = 0
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_ext_lower = os.path.splitext(file_name)[1].lower()

                # COMTRADE CFG/DAT
                if file_ext_lower == self.CFG_EXTENSION and file_type_flags.get(TYPE_OSC.COMTRADE_CFG_DAT, False):
                    newly_copied_count += self._process_comtrade_file(
                        file_path, root, source_dir_base_for_relative_paths, dest_dir_base,
                        copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes
                    )
                # COMTRADE CFF
                elif file_ext_lower == self.CFF_EXTENSION and file_type_flags.get(TYPE_OSC.COMTRADE_CFF, False):
                    newly_copied_count += self._process_file(
                        file_path, root, source_dir_base_for_relative_paths, dest_dir_base,
                        copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes,
                        TYPE_OSC.COMTRADE_CFF, "COMTRADE_CFF"
                    )
                # Archives
                elif file_ext_lower == self.ARCH_7Z_EXTENSION and file_type_flags.get(TYPE_OSC.ARCH_7Z, False):
                    newly_copied_count += self._process_archive_file(
                        file_path, root, source_dir_base_for_relative_paths, dest_dir_base,
                        copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes,
                        file_type_flags, current_archive_depth, max_archive_depth
                    )
                elif file_ext_lower == self.ARCH_ZIP_EXTENSION and file_type_flags.get(TYPE_OSC.ARCH_ZIP, False):
                    newly_copied_count += self._process_archive_file(
                        file_path, root, source_dir_base_for_relative_paths, dest_dir_base,
                        copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes,
                        file_type_flags, current_archive_depth, max_archive_depth
                    )
                elif file_ext_lower == self.ARCH_RAR_EXTENSION and file_type_flags.get(TYPE_OSC.ARCH_RAR, False):
                    if az is None and self.is_print_message: print("Aspose.Zip for RAR not available. Skipping .rar files.")
                    elif az:
                        newly_copied_count += self._process_archive_file(
                            file_path, root, source_dir_base_for_relative_paths, dest_dir_base,
                            copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes,
                            file_type_flags, current_archive_depth, max_archive_depth
                        )
                # Other specific types
                elif file_ext_lower == self.EVA_EXTENSION and file_type_flags.get(TYPE_OSC.EVA, False):
                    newly_copied_count += self._process_file(file_path, root, source_dir_base_for_relative_paths, dest_dir_base, copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes, TYPE_OSC.EVA, "EVA")
                elif file_name.lower().endswith(self.BRESLER_EXTENSION) and file_type_flags.get(TYPE_OSC.BRESLER, False): # Note: use endswith for multi-dot extensions
                    newly_copied_count += self._process_file(file_path, root, source_dir_base_for_relative_paths, dest_dir_base, copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes, TYPE_OSC.BRESLER, "BRESLER")
                elif file_name.lower().endswith(self.NIIM_EXTENSION) and file_type_flags.get(TYPE_OSC.NIIM, False):
                    newly_copied_count += self._process_file(file_path, root, source_dir_base_for_relative_paths, dest_dir_base, copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes, TYPE_OSC.NIIM, "NIIM")
                elif file_name.lower().endswith(self.KRUG_EXTENSION) and file_type_flags.get(TYPE_OSC.KRUG, False):
                    newly_copied_count += self._process_file(file_path, root, source_dir_base_for_relative_paths, dest_dir_base, copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes, TYPE_OSC.KRUG, "KRUG")
                elif file_ext_lower == self.EKRA_EXTENSION and file_type_flags.get(TYPE_OSC.EKRA, False):
                    newly_copied_count += self._process_file(file_path, root, source_dir_base_for_relative_paths, dest_dir_base, copied_hashes, new_copied_hashes_this_run, preserve_dir_structure, use_hashes, TYPE_OSC.EKRA, "EKRA")

                if pbar: pbar.update(1) # Update progress for each file considered
        return newly_copied_count

    def copy_new_oscillograms(self, source_dir: str, dest_dir: str,
                              copied_hashes_input: Optional[Dict[str, str]] = None,
                              preserve_dir_structure: bool = True, use_hashes: bool = True,
                              file_type_flags: Optional[Dict[TYPE_OSC, bool]] = None,
                              max_archive_depth: int = 2,
                              progress_callback=None, stop_processing_fn=None,
                              is_write_names_fn=None) -> int: # Added is_write_names_fn for compatibility

        if not os.path.isdir(source_dir):
            if self.is_print_message: print(f"Error: Source directory '{source_dir}' does not exist.")
            return 0
        os.makedirs(dest_dir, exist_ok=True)

        copied_hashes: Dict[str, str] = {}
        if use_hashes:
            if copied_hashes_input is not None:
                copied_hashes = dict(copied_hashes_input) # Use provided, make a copy
                if self.is_print_message: print(f"Using provided hash table with {len(copied_hashes)} entries.")
            else:
                hash_table_path = os.path.join(dest_dir, "_hash_table.json")
                if os.path.exists(hash_table_path):
                    try:
                        with open(hash_table_path, 'r', encoding='utf-8') as f:
                            copied_hashes = json.load(f)
                        if self.is_print_message: print(f"Loaded hash table from {hash_table_path} with {len(copied_hashes)} entries.")
                    except Exception as e:
                        if self.is_print_message: print(f"Error loading hash table {hash_table_path}: {e}. Starting with empty table.")
                        copied_hashes = {}
                else:
                    if self.is_print_message: print("No existing hash table found. Starting fresh.")

        # Initialize new_copied_hashes for this specific run at the top level
        self._current_run_newly_copied_hashes.clear()

        if file_type_flags is None: # Default: process all known types
            file_type_flags = {osc_type: True for osc_type in TYPE_OSC}
            if self.is_print_message: print("No file type flags specified, processing all known types.")

        # Count total files for tqdm progress bar if not using callback
        total_files = 0
        if progress_callback is None:
            for _, _, files_in_walk in os.walk(source_dir):
                total_files += len(files_in_walk)

        pbar = None
        if progress_callback is None and self.is_print_message : # Only use tqdm if no callback and verbose
            pbar = tqdm(total=total_files, desc="Copying oscillograms")

        newly_copied_total_count = self._walk_and_process(
            source_dir=source_dir,
            dest_dir_base=dest_dir,
            source_dir_base_for_relative_paths=source_dir, # For top-level call, base is source_dir
            copied_hashes=copied_hashes,
            new_copied_hashes_this_run=self._current_run_newly_copied_hashes,
            preserve_dir_structure=preserve_dir_structure,
            use_hashes=use_hashes,
            file_type_flags=file_type_flags,
            current_archive_depth=0,
            max_archive_depth=max_archive_depth,
            pbar=pbar
        )

        if pbar: pbar.close()

        if use_hashes:
            # Save the main hash table (updated with any new hashes from this run)
            # The original logic was to add new_copied_hashes to copied_hashes at the end
            for h, path_rel in self._current_run_newly_copied_hashes.items():
                 if h not in copied_hashes: # Should always be true due to checks in _process*
                      copied_hashes[h] = path_rel

            hash_table_path = os.path.join(dest_dir, "_hash_table.json")
            try:
                with open(hash_table_path, 'w', encoding='utf-8') as f:
                    json.dump(copied_hashes, f, indent=4, ensure_ascii=False)
                if self.is_print_message: print(f"Main hash table saved to {hash_table_path} with {len(copied_hashes)} total entries.")
            except Exception as e:
                if self.is_print_message: print(f"Error saving main hash table: {e}")

            # Save the new_copied_hashes for this run specifically
            if self._current_run_newly_copied_hashes:
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                new_hashes_filename = f"newly_copied_hashes_{ts}.json"
                new_hashes_path = os.path.join(dest_dir, new_hashes_filename)
                try:
                    with open(new_hashes_path, 'w', encoding='utf-8') as f:
                        json.dump(self._current_run_newly_copied_hashes, f, indent=4, ensure_ascii=False)
                    if self.is_print_message: print(f"Newly copied hashes for this run saved to {new_hashes_path} ({len(self._current_run_newly_copied_hashes)} entries).")
                except Exception as e:
                    if self.is_print_message: print(f"Error saving new hashes table: {e}")
            elif self.is_print_message:
                 print("No new files were copied in this run.")

        # Compatibility for is_write_names_fn placeholder
        if is_write_names_fn is not None and callable(is_write_names_fn):
            try:
                is_write_names_fn() # Call it if provided
            except Exception as e:
                if self.is_print_message: print(f"Error executing is_write_names_fn callback: {e}")


        if self.is_print_message:
            print(f"Oscillogram copy process complete. Total newly copied files/pairs: {newly_copied_total_count}")
        return newly_copied_total_count

    @staticmethod
    def find_terminal_hashes_from_json(input_json_path: str, terminal_numbers_to_find: List[int],
                                       output_json_path: str, is_print_message: bool = False) -> None:
        if not os.path.exists(input_json_path):
            if is_print_message: print(f"Error: Input JSON file not found: {input_json_path}")
            return

        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            if is_print_message: print(f"Error reading or parsing input JSON {input_json_path}: {e}")
            return

        if not isinstance(data, dict):
            if is_print_message: print("Error: Input JSON content is not a dictionary (terminal to hashes map).")
            return

        found_hashes_map = {}
        terminal_numbers_to_find_str = {str(t) for t in terminal_numbers_to_find} # Convert target terminals to string for matching

        if is_print_message: print(f"Searching for terminals: {terminal_numbers_to_find_str}")

        for terminal_key, hash_list in tqdm(data.items(), desc="Processing terminals", disable=not is_print_message):
            if str(terminal_key) in terminal_numbers_to_find_str:
                if isinstance(hash_list, list):
                    found_hashes_map[str(terminal_key)] = hash_list
                    if is_print_message: print(f"  Found terminal {terminal_key} with {len(hash_list)} hashes.")
                elif is_print_message:
                    print(f"  Warning: Hashes for terminal {terminal_key} are not in a list format. Skipping.")

        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f_out:
                json.dump(found_hashes_map, f_out, indent=4, ensure_ascii=False)
            if is_print_message:
                print(f"Found hashes for specified terminals saved to: {output_json_path}")
                print(f"Total terminals found: {len(found_hashes_map)}")
        except Exception as e:
            if is_print_message: print(f"Error writing output JSON to {output_json_path}: {e}")


    def find_oscillograms_with_neutral_current(self, cfg_directory: str, output_txt_path: str,
                                               is_print_message: bool = False) -> None:
        if not os.path.isdir(cfg_directory):
            if is_print_message: print(f"Error: Source directory '{cfg_directory}' not found.")
            return

        found_files = []
        cfg_files_to_scan = [os.path.join(r, f) for r, _, fs in os.walk(cfg_directory) for f in fs if f.lower().endswith(self.CFG_EXTENSION)]

        if is_print_message: print(f"Found {len(cfg_files_to_scan)} CFG files to scan for neutral current.")

        for cfg_file_path in tqdm(cfg_files_to_scan, desc="Scanning for neutral current", disable=not is_print_message):
            try:
                osc = Oscillogram(cfg_file_path) # Using Oscillogram to load

                # analog_channel_ids from Oscillogram are already full names
                for channel_name in osc.analog_channel_ids:
                    # Example channel name: "I | Bus-1 | phase: N"
                    # Split by " | " and check parts. Be robust to spacing.
                    parts = [p.strip().upper() for p in channel_name.split('|')] # Uppercase for case-insensitive compare

                    is_current = False
                    is_neutral = False

                    for part in parts:
                        if part == 'I': # Check for current type identifier
                            is_current = True
                        if "PHASE: N" in part or part == 'N': # Check for neutral phase identifier
                            is_neutral = True

                    if is_current and is_neutral:
                        found_files.append(os.path.basename(cfg_file_path)[:-len(self.CFG_EXTENSION)]) # Add hash/basename
                        if is_print_message and len(found_files) <= 10 : # Print first few findings
                             print(f"  Found neutral current in: {os.path.basename(cfg_file_path)} (Channel: '{channel_name}')")
                        break # Found neutral current in this file, no need to check other channels

            except RuntimeError as e: # Catch errors from Oscillogram loading (e.g. ComtradeReadError)
                 if is_print_message: print(f"  Error processing {os.path.basename(cfg_file_path)}: {e}")
            except Exception as e:
                if is_print_message: print(f"  Unexpected error with {os.path.basename(cfg_file_path)}: {e}")

        output_dir = os.path.dirname(output_txt_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f_out:
                for file_name_hash in sorted(list(set(found_files))): # Write unique names, sorted
                    f_out.write(f"{file_name_hash}\n")
            if is_print_message:
                print(f"List of oscillograms with neutral current saved to: {output_txt_path}")
                print(f"Total files found with neutral current: {len(found_files)}")
        except Exception as e:
            if is_print_message: print(f"Error writing output TXT to {output_txt_path}: {e}")
