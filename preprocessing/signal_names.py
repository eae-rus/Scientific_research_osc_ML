import os
import csv
import re
from collections import Counter

class SignalNameManager:
    def __init__(self):
        pass

    def _extract_signal_names_from_cfg(self, cfg_file_path: str, encoding: str,
                                       include_analog: bool, include_digital: bool) -> list[str]:
        signal_names = []
        try:
            with open(cfg_file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()

            if len(lines) < 2: # Need at least the header and the counts line
                return []

            # Parse second line for signal counts (e.g., "12,6A,6D")
            parts = lines[1].strip().split(',')
            if len(parts) < 3:
                return [] # Line format is incorrect

            try:
                # Extract numbers, removing non-numeric characters like 'A' or 'D'
                # Example: "6A" -> 6
                count_analog_signals = int(re.sub(r'\D', '', parts[1])) if parts[1] else 0
                count_digital_signals = int(re.sub(r'\D', '', parts[2])) if parts[2] else 0
            except ValueError:
                return [] # Could not parse signal counts

            current_line_idx = 2 # Channel definitions start from the third line (index 2)

            # Extract Analog Signals
            if include_analog:
                for _ in range(count_analog_signals):
                    if current_line_idx >= len(lines): break
                    signal_line_parts = lines[current_line_idx].strip().split(',')
                    if len(signal_line_parts) > 2: # Need at least index, name, phase
                        name = signal_line_parts[1].strip()
                        phase = signal_line_parts[2].strip()
                        # Replicate specific combined name format from original ProcessingOscillograms
                        if name and phase: # Ensure both are non-empty
                            combined_name = name.replace(' ', '') + ' | phase:' + phase.replace(' ', '')
                            signal_names.append(combined_name)
                        elif name: # If phase is missing, just use name
                            signal_names.append(name.replace(' ', ''))
                    elif len(signal_line_parts) > 1: # Only name available
                         name = signal_line_parts[1].strip()
                         if name:
                            signal_names.append(name.replace(' ', ''))
                    current_line_idx += 1
            else:
                # If not including analog, still need to advance past their lines
                current_line_idx += count_analog_signals

            # Extract Digital Signals
            if include_digital:
                for _ in range(count_digital_signals):
                    if current_line_idx >= len(lines): break
                    signal_line_parts = lines[current_line_idx].strip().split(',')
                    if len(signal_line_parts) > 1: # Name is the second field
                        name = signal_line_parts[1].strip()
                        if name: # Ensure name is not empty
                            signal_names.append(name) # No phase for digital in this context
                    current_line_idx += 1

        except UnicodeDecodeError:
            return [] # Signal encoding error to caller
        except IOError:
            # print(f"Warning (_extract_signal_names_from_cfg): IOError for {cfg_file_path} with {encoding}")
            return [] # Signal file read error to caller
        except Exception:
            # print(f"Warning (_extract_signal_names_from_cfg): Unexpected error for {cfg_file_path} with {encoding}: {e}")
            return [] # Signal other error to caller
        return signal_names

    def find_signal_names(self, source_dir: str, signal_type_to_find: str = 'all',
                          output_csv_path: str = 'signal_catalog.csv',
                          is_print_message: bool = False) -> None:

        if not os.path.isdir(source_dir):
            if is_print_message: print(f"Error: Source directory '{source_dir}' not found.")
            return

        include_analog = signal_type_to_find.lower() in ['analog', 'all']
        include_digital = signal_type_to_find.lower() in ['digital', 'all']

        if not include_analog and not include_digital:
            if is_print_message: print("Neither analog nor digital signals selected for finding. Exiting.")
            return

        all_signal_names_counter = Counter()

        cfg_files_to_process = []
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                if file_name.lower().endswith(".cfg"):
                    cfg_files_to_process.append(os.path.join(root, file_name))

        if is_print_message:
            total_files = len(cfg_files_to_process)
            print(f"Found {total_files} .cfg files to scan for signal names in {source_dir}.")

        processed_count = 0
        for cfg_file_path in cfg_files_to_process:
            processed_count += 1
            if is_print_message and processed_count % 100 == 0:
                print(f"Scanning file {processed_count}/{total_files}: {os.path.basename(cfg_file_path)}")

            encodings_to_try = ['utf-8', 'windows-1251', 'cp866']
            file_signal_names = []
            extracted_successfully_for_file = False

            for encoding in encodings_to_try:
                # _extract_signal_names_from_cfg returns [] on encoding error or if no signals of the type are found.
                # We need a way to distinguish "bad encoding" from "file has no signals".
                # For now, if it returns anything (even an empty list not due to encoding error), we assume encoding was okay.
                # A more robust way would be for _extract to raise specific encoding errors.
                # Given current _extract, if it returns [], we can't be sure if it's encoding or no signals.
                # However, the prompt implies _extract returns [] on encoding error, so this logic is okay.

                # Let's test readability with the encoding first.
                try:
                    with open(cfg_file_path, 'r', encoding=encoding) as f_test:
                        f_test.readline() # Try to read one line to check encoding
                except UnicodeDecodeError:
                    if is_print_message and processed_count <= 20 : # Print for first few files to show encoding attempts
                         print(f"  Skipping encoding {encoding} for {os.path.basename(cfg_file_path)} due to decode error.")
                    continue # Try next encoding
                except Exception: # Other read errors
                    continue


                current_names = self._extract_signal_names_from_cfg(cfg_file_path, encoding,
                                                                  include_analog, include_digital)

                # If current_names is not empty, it means we found signals with this encoding.
                # If current_names IS empty, it could be because:
                # 1. Encoding was fine, but no relevant signals in file.
                # 2. _extract_signal_names_from_cfg had an internal issue (other than UnicodeDecode which it returns [] for).
                # We assume if we reach here, the encoding was "probably" okay for reading structure, even if no names found.
                file_signal_names.extend(current_names)
                extracted_successfully_for_file = True # At least one encoding was processed without decode error
                if is_print_message and len(current_names) > 0:
                     print(f"  Extracted {len(current_names)} signal names from {os.path.basename(cfg_file_path)} using {encoding}.")
                break # Successfully processed with this encoding (or determined no signals with it)

            if extracted_successfully_for_file: # If any encoding allowed processing
                all_signal_names_counter.update(file_signal_names) # Update with names found (could be empty)
            elif is_print_message: # If all encodings failed at the read test stage
                print(f"  Warning: Could not read or extract signal names from {os.path.basename(cfg_file_path)} with any tried encoding.")

        sorted_signal_names = all_signal_names_counter.most_common()

        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                if is_print_message: print(f"Error creating output directory {output_dir}: {e}")
                return

        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Signal Name', 'Universal Code', 'Count'])
                for name, count in sorted_signal_names:
                    writer.writerow([name, '-', count])
            if is_print_message:
                print(f"Signal name catalog saved to: {output_csv_path} with {len(sorted_signal_names)} unique names.")
        except IOError as e:
            if is_print_message: print(f"Error writing signal catalog CSV to {output_csv_path}: {e}")
        except Exception as e:
            if is_print_message: print(f"An unexpected error occurred while writing CSV {output_csv_path}: {e}")

    def _parse_analog_signal_name_for_section(self, signal_name_field: str) -> dict | None:
        """
        Parses an analog signal name field (e.g., "U | BusBar-1" or "U | BusBar")
        to find a numeric section for potential incrementing.
        The goal is to separate a prefix, an existing section number, and a suffix.
        Example target: "Name-1" -> prefix="Name-", section_num="1", suffix=""
                       "Name-1-Details" -> prefix="Name-", section_num="1", suffix="-Details" (less ideal, prefer suffix to be part of prefix if not clearly separated)
                       "Name1" -> prefix="Name", section_num="1", suffix=""
        This specific regex targets names ending in "-<number>" or "<number>" (no hyphen).
        """
        # Pattern 1: "Anything-Number" (e.g., "BusBar-1", "Signal-ABC-12")
        match = re.fullmatch(r'(.*[^\d])-(\d+)', signal_name_field)
        if match:
            prefix, section_num = match.groups()
            return {'prefix': prefix + "-", 'section_num': section_num, 'suffix': "", 'original_name_field': signal_name_field}

        # Pattern 2: "AnythingNumber" (e.g., "BusBar1", "SignalABC12") - no hyphen, number at the end
        match = re.fullmatch(r'(.*[^\d])(\d+)', signal_name_field)
        if match:
            prefix, section_num = match.groups()
            # Check if what's before the number is non-empty, otherwise it's just a number.
            if prefix: # Ensure there's a non-numeric part before the number
                 return {'prefix': prefix, 'section_num': section_num, 'suffix': "", 'original_name_field': signal_name_field}

        # If no pattern matched for modification, return None or a structure indicating it's not auto-modifiable this way.
        return {'prefix': signal_name_field, 'section_num': None, 'suffix': "", 'original_name_field': signal_name_field}


    def _get_raw_signal_info_from_lines(self, lines: list[str], file_path_for_log: str = "") -> list[dict]:
        """
        Extracts raw signal information (name as it appears in field[1], line index, type) from CFG lines.
        Returns a list of dicts: [{'line_index': int, 'raw_name': str, 'is_analog': bool, 'original_line': str}]
        """
        signal_info_list = []
        if len(lines) < 2: return signal_info_list

        parts = lines[1].strip().split(',')
        if len(parts) < 3: return signal_info_list

        try:
            count_analog = int(re.sub(r'\D', '', parts[1])) if parts[1] else 0
            count_digital = int(re.sub(r'\D', '', parts[2])) if parts[2] else 0
        except ValueError:
            # print(f"Warning (_get_raw_signal_info_from_lines): Could not parse signal counts in {file_path_for_log}")
            return signal_info_list

        current_line_idx = 2 # Channel definitions start at index 2

        for i in range(count_analog):
            if current_line_idx >= len(lines): break
            line_content = lines[current_line_idx].strip()
            signal_line_parts = line_content.split(',')
            if len(signal_line_parts) > 1:
                raw_name = signal_line_parts[1].strip()
                phase = signal_line_parts[2].strip() if len(signal_line_parts) > 2 else ""
                # For analog, the "effective" name for duplication might be name + phase
                # but for renaming, we modify raw_name (field[1])
                # The combined name is used for _finding_ duplicates to rename.
                combined_name_for_dupe_check = raw_name.replace(' ', '') + ' | phase:' + phase.replace(' ', '') if raw_name and phase else raw_name.replace(' ', '')

                signal_info_list.append({
                    'line_index': current_line_idx,
                    'raw_name_field': raw_name, # This is what we modify
                    'combined_name_for_dupe_check': combined_name_for_dupe_check, # This is what we check for duplication
                    'is_analog': True,
                    'original_line': lines[current_line_idx],
                    'phase_field': phase
                })
            current_line_idx += 1

        for i in range(count_digital):
            if current_line_idx >= len(lines): break
            line_content = lines[current_line_idx].strip()
            signal_line_parts = line_content.split(',')
            if len(signal_line_parts) > 1:
                raw_name = signal_line_parts[1].strip()
                signal_info_list.append({
                    'line_index': current_line_idx,
                    'raw_name_field': raw_name,
                    'combined_name_for_dupe_check': raw_name, # For digital, raw name is used for dupe check
                    'is_analog': False,
                    'original_line': lines[current_line_idx]
                })
            current_line_idx += 1
        return signal_info_list

    def _rename_duplicate_analog_signals_in_lines(self, cfg_lines: list[str], file_path_for_log: str, is_print_message: bool = False) -> tuple[list[str], bool, list[dict]]:
        modified_lines = list(cfg_lines) # Work on a copy
        changes_made = False
        rename_actions = []

        signal_infos = self._get_raw_signal_info_from_lines(modified_lines, file_path_for_log)

        # We need all current "combined_name_for_dupe_check" to ensure new names are unique across the file
        all_current_combined_names = {info['combined_name_for_dupe_check'] for info in signal_infos}

        # Count occurrences of combined analog names
        analog_combined_names_to_check = [info['combined_name_for_dupe_check'] for info in signal_infos if info['is_analog']]
        name_counts = Counter(analog_combined_names_to_check)

        # Iterate through original signal infos to decide which ones to rename
        # We need to iterate multiple times if a rename causes a new conflict (unlikely with _N suffix, but good to be aware)
        # For simplicity, this pass renames. A more robust system might need iterative renaming.

        processed_original_indices = set() # To ensure we only rename duplicates after the first occurrence

        for original_idx, signal_info in enumerate(signal_infos):
            if not signal_info['is_analog']:
                continue

            current_combined_name = signal_info['combined_name_for_dupe_check']

            # If this name is a duplicate and we haven't processed its first occurrence yet
            if name_counts[current_combined_name] > 1:
                # Find all occurrences of this combined name
                indices_of_this_name = [
                    i for i, s_info in enumerate(signal_infos)
                    if s_info['is_analog'] and s_info['combined_name_for_dupe_check'] == current_combined_name
                ]

                first_occurrence_processed = False
                for master_list_idx in indices_of_this_name:
                    if master_list_idx in processed_original_indices: # If this specific instance was already handled (e.g. it was the first one)
                        first_occurrence_processed = True # Mark that the first one was conceptually "kept"
                        break

                if not first_occurrence_processed: # Keep the first one, mark it as processed
                    processed_original_indices.add(indices_of_this_name[0])
                    # Now, iterate all OTHER instances of this duplicate name to rename them
                    for i in range(len(indices_of_this_name)):
                        if i == 0: continue # Skip the first one, which we are keeping as is

                        signal_to_rename_info = signal_infos[indices_of_this_name[i]]
                        line_idx_to_change = signal_to_rename_info['line_index']
                        old_raw_name_field = signal_to_rename_info['raw_name_field']
                        old_phase_field = signal_to_rename_info['phase_field']

                        parsed_name = self._parse_analog_signal_name_for_section(old_raw_name_field)

                        new_raw_name_field = None
                        next_num = 1

                        # Try appending _section_num format
                        while True:
                            temp_raw_name = f"{parsed_name['prefix']}{parsed_name['section_num']}_{next_num}{parsed_name['suffix']}" if parsed_name['section_num'] else f"{parsed_name['prefix']}_{next_num}{parsed_name['suffix']}"

                            # Construct the new combined name for uniqueness check
                            temp_combined_name = temp_raw_name.replace(' ', '') + ' | phase:' + old_phase_field.replace(' ', '') if old_phase_field else temp_raw_name.replace(' ', '')

                            if temp_combined_name not in all_current_combined_names:
                                new_raw_name_field = temp_raw_name
                                break
                            next_num += 1
                            if next_num > 1000: # Safety break
                                if is_print_message: print(f"  Error: Could not find unique name for {old_raw_name_field} in {file_path_for_log} after 1000 tries.")
                                new_raw_name_field = None # Mark as failed to find unique name
                                break

                        if new_raw_name_field:
                            # Modify the line
                            line_to_modify = modified_lines[line_idx_to_change]
                            parts = line_to_modify.split(',')
                            parts[1] = new_raw_name_field # Update the name field
                            modified_lines[line_idx_to_change] = ','.join(parts) # Ensure newline is preserved if original had it
                            if not modified_lines[line_idx_to_change].endswith('\n') and cfg_lines[line_idx_to_change].endswith('\n'):
                                modified_lines[line_idx_to_change] += '\n'

                            changes_made = True
                            new_combined_name = new_raw_name_field.replace(' ', '') + ' | phase:' + old_phase_field.replace(' ', '') if old_phase_field else new_raw_name_field.replace(' ', '')
                            action = {
                                'file_path': os.path.basename(file_path_for_log),
                                'line_index': line_idx_to_change + 1, # 1-based for log
                                'old_combined_name': current_combined_name,
                                'new_combined_name': new_combined_name,
                                'old_raw_name_field': old_raw_name_field,
                                'new_raw_name_field': new_raw_name_field
                            }
                            rename_actions.append(action)
                            if is_print_message:
                                print(f"  LogRename: File: {action['file_path']}, Line: {action['line_index']}, OldCombined: '{action['old_combined_name']}', NewCombined: '{action['new_combined_name']}' (Raw: '{old_raw_name_field}' -> '{new_raw_name_field}')")

                            # Update our reference set for future uniqueness checks in this file
                            all_current_combined_names.remove(current_combined_name) # Remove one instance of the old name count
                            all_current_combined_names.add(new_combined_name)
                            # Update signal_infos as well if we were to iterate again (not done in this simplified version)
                        else:
                            if is_print_message: print(f"  Failed to generate unique name for duplicate: {current_combined_name} at line {line_idx_to_change+1}")

                # Mark all instances of this duplicated name as processed ( conceptually, first one kept, others renamed)
                for master_list_idx in indices_of_this_name:
                    processed_original_indices.add(master_list_idx)

        return modified_lines, changes_made, rename_actions


    def manage_duplicate_signals(self, source_dir: str,
                                 output_csv_duplicates_path: str,
                                 auto_rename_analog: bool = False,
                                 output_csv_rename_log_path: str = "rename_log.csv",
                                 cfg_encodings_to_try: list = None,
                                 is_print_message: bool = False) -> None:

        if cfg_encodings_to_try is None:
            cfg_encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

        files_with_duplicates_log = [] # List of tuples: (file_path, duplicate_info_str)
        all_renaming_actions_log = []  # List of dicts from _rename_duplicate_analog_signals_in_lines

        cfg_files_to_scan = []
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                if file_name.lower().endswith(".cfg"):
                    cfg_files_to_scan.append(os.path.join(root, file_name))

        if is_print_message:
            print(f"Found {len(cfg_files_to_scan)} .cfg files to scan for duplicate signal names.")

        file_scan_count = 0
        for cfg_file_path in cfg_files_to_scan:
            file_scan_count +=1
            if is_print_message and file_scan_count % 50 == 0:
                print(f"Scanning file for duplicates {file_scan_count}/{len(cfg_files_to_scan)}: {os.path.basename(cfg_file_path)}")

            determined_encoding = None
            cfg_lines = None
            for encoding in cfg_encodings_to_try:
                try:
                    with open(cfg_file_path, 'r', encoding=encoding) as f:
                        cfg_lines = f.readlines()
                    determined_encoding = encoding
                    if is_print_message and file_scan_count <=10 : print(f"  Successfully read {os.path.basename(cfg_file_path)} with encoding {encoding}")
                    break
                except UnicodeDecodeError:
                    if is_print_message and file_scan_count <=10 : print(f"  Encoding {encoding} failed for {os.path.basename(cfg_file_path)}")
                    continue
                except Exception as e:
                    if is_print_message: print(f"  Error reading {os.path.basename(cfg_file_path)} with {encoding}: {e}")
                    continue # Skip this file if fundamental read error other than encoding

            if not cfg_lines or not determined_encoding:
                if is_print_message: print(f"  Could not read or determine encoding for {os.path.basename(cfg_file_path)}. Skipping duplicate check.")
                files_with_duplicates_log.append({'file_path': cfg_file_path, 'file_name': os.path.basename(cfg_file_path), 'error': 'Could not read file with any encoding'})
                continue

            # Extract "raw" signal names (field[1]) and their line numbers for duplicate checking
            # Using combined name for duplicate check as per original logic for analog signals
            current_file_signal_infos = self._get_raw_signal_info_from_lines(cfg_lines, cfg_file_path)
            # We are interested in duplicates of 'combined_name_for_dupe_check'
            names_for_counting = [info['combined_name_for_dupe_check'] for info in current_file_signal_infos]
            name_counts = Counter(names_for_counting)

            duplicates_found_in_file = False
            for name, count in name_counts.items():
                if count > 1:
                    duplicates_found_in_file = True
                    # Log detailed info about which names are duplicated
                    line_indices = [info['line_index'] + 1 for info in current_file_signal_infos if info['combined_name_for_dupe_check'] == name]
                    duplicate_entry = {
                        'file_path': cfg_file_path,
                        'file_name': os.path.basename(cfg_file_path),
                        'signal_name': name,
                        'count': count,
                        'lines': str(line_indices)
                    }
                    files_with_duplicates_log.append(duplicate_entry)
                    if is_print_message:
                         print(f"  Duplicate found in {os.path.basename(cfg_file_path)}: '{name}' (Count: {count}, Lines: {line_indices})")

            if duplicates_found_in_file and auto_rename_analog:
                if is_print_message: print(f"  Attempting to auto-rename analog duplicates in {os.path.basename(cfg_file_path)}...")
                modified_lines, changes_made, rename_actions = self._rename_duplicate_analog_signals_in_lines(cfg_lines, cfg_file_path, is_print_message)

                if changes_made:
                    try:
                        with open(cfg_file_path, 'w', encoding=determined_encoding) as f_write:
                            f_write.writelines(modified_lines)
                        if is_print_message: print(f"  Successfully wrote changes to {os.path.basename(cfg_file_path)}")
                        all_renaming_actions_log.extend(rename_actions)
                    except Exception as e:
                        if is_print_message: print(f"  Error writing modified file {os.path.basename(cfg_file_path)}: {e}")
                elif is_print_message:
                     print(f"  No analog signal renames were performed for {os.path.basename(cfg_file_path)} (possibly no renamable analog duplicates or no unique names found).")

        # Write duplicate log
        if files_with_duplicates_log:
            output_dup_dir = os.path.dirname(output_csv_duplicates_path)
            if output_dup_dir and not os.path.exists(output_dup_dir): os.makedirs(output_dup_dir, exist_ok=True)
            try:
                with open(output_csv_duplicates_path, 'w', newline='', encoding='utf-8') as csvfile:
                    # Adjust header if 'error' key might not exist for all entries
                    fieldnames = ['file_path', 'file_name', 'signal_name', 'count', 'lines', 'error']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for entry in files_with_duplicates_log:
                         writer.writerow(entry) # entry is already a dict
                if is_print_message: print(f"Duplicate signals log saved to: {output_csv_duplicates_path}")
            except IOError as e:
                if is_print_message: print(f"Error writing duplicates log CSV: {e}")

        # Write rename log
        if all_renaming_actions_log:
            output_rename_dir = os.path.dirname(output_csv_rename_log_path)
            if output_rename_dir and not os.path.exists(output_rename_dir): os.makedirs(output_rename_dir, exist_ok=True)
            try:
                with open(output_csv_rename_log_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['file_path', 'line_index', 'old_combined_name', 'new_combined_name', 'old_raw_name_field', 'new_raw_name_field']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_renaming_actions_log)
                if is_print_message: print(f"Renaming actions log saved to: {output_csv_rename_log_path}")
            except IOError as e:
                if is_print_message: print(f"Error writing rename log CSV: {e}")
        elif auto_rename_analog and is_print_message:
            print("No renaming actions were logged.")

        if is_print_message: print("Finished managing duplicate signals.")

    def rename_signals_from_csv(self, source_dir: str, csv_map_path: str,
                                signal_type_to_rename: str = 'all',
                                cfg_encodings_to_try: list = None,
                                csv_encoding: str = 'utf-8',
                                csv_delimiter: str = ',',
                                is_print_message: bool = False) -> None:
        if cfg_encodings_to_try is None:
            cfg_encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

        if not os.path.isfile(csv_map_path):
            if is_print_message: print(f"Error: CSV mapping file not found: {csv_map_path}")
            return

        name_map = {}
        try:
            with open(csv_map_path, mode='r', encoding=csv_encoding) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=csv_delimiter)
                if 'Key' not in reader.fieldnames or 'universal_code' not in reader.fieldnames:
                    if is_print_message: print(f"Error: CSV map must contain 'Key' and 'universal_code' columns. Found: {reader.fieldnames}")
                    return
                for row in reader:
                    key = row['Key']
                    universal_code = row['universal_code']
                    if universal_code and universal_code not in ['-', '?','']: # Skip if no valid new name
                        name_map[key] = universal_code
        except Exception as e:
            if is_print_message: print(f"Error reading or parsing CSV map {csv_map_path}: {e}")
            return

        if not name_map:
            if is_print_message: print("No valid signal name mappings loaded from CSV. Nothing to rename.")
            return

        if is_print_message: print(f"Loaded {len(name_map)} signal renaming rules from {csv_map_path}.")

        cfg_files_to_process = [os.path.join(r, f) for r, _, fs in os.walk(source_dir) for f in fs if f.lower().endswith(".cfg")]
        if is_print_message: print(f"Found {len(cfg_files_to_process)} .cfg files to process for renaming.")

        processed_file_count = 0
        for cfg_file_path in cfg_files_to_process:
            processed_file_count += 1
            if is_print_message and processed_file_count % 50 == 0:
                print(f"Processing file for CSV rename {processed_file_count}/{len(cfg_files_to_process)}: {os.path.basename(cfg_file_path)}")

            determined_encoding = None
            cfg_lines = None
            for encoding in cfg_encodings_to_try:
                try:
                    with open(cfg_file_path, 'r', encoding=encoding) as f:
                        cfg_lines = f.readlines()
                    determined_encoding = encoding
                    break
                except UnicodeDecodeError: continue
                except Exception: continue

            if not cfg_lines or not determined_encoding:
                if is_print_message: print(f"  Could not read {os.path.basename(cfg_file_path)} with any encoding. Skipping.")
                continue

            signal_infos = self._get_raw_signal_info_from_lines(cfg_lines, cfg_file_path)
            modified_this_file = False

            temp_cfg_lines = list(cfg_lines) # Work on a copy

            for info in signal_infos:
                # Determine which signal type to process
                if signal_type_to_rename == 'analog' and not info['is_analog']: continue
                if signal_type_to_rename == 'digital' and info['is_analog']: continue

                current_combined_name = info['combined_name_for_dupe_check']
                if current_combined_name in name_map:
                    new_raw_name_field = name_map[current_combined_name]
                    old_raw_name_field = info['raw_name_field']

                    if old_raw_name_field != new_raw_name_field:
                        line_idx = info['line_index']
                        original_line_content = temp_cfg_lines[line_idx]
                        parts = original_line_content.split(',')

                        # Ensure we're only changing the name part (parts[1])
                        if len(parts) > 1:
                            parts[1] = new_raw_name_field
                            # Reconstruct line, preserving original number of fields and newline
                            reconstructed_line = ','.join(parts)
                            if not reconstructed_line.endswith('\n') and original_line_content.endswith('\n'):
                                reconstructed_line += '\n'

                            temp_cfg_lines[line_idx] = reconstructed_line
                            modified_this_file = True
                            if is_print_message:
                                print(f"  In {os.path.basename(cfg_file_path)} (line {line_idx+1}): Renamed '{current_combined_name}' (raw: '{old_raw_name_field}') to raw '{new_raw_name_field}'")

            if modified_this_file:
                try:
                    with open(cfg_file_path, 'w', encoding=determined_encoding) as f_write:
                        f_write.writelines(temp_cfg_lines)
                    if is_print_message: print(f"  Successfully updated {os.path.basename(cfg_file_path)} with renames from CSV.")
                except Exception as e:
                    if is_print_message: print(f"  Error writing changes to {os.path.basename(cfg_file_path)}: {e}")

        if is_print_message: print("Finished renaming signals from CSV.")


    def rename_single_signal(self, source_dir: str,
                             old_name_pattern: str, new_name_field_value: str,
                             cfg_encodings_to_try: list = None,
                             is_print_message: bool = False) -> None:
        if cfg_encodings_to_try is None:
            cfg_encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

        if not old_name_pattern: # Cannot replace an empty pattern
            if is_print_message: print("Error: 'old_name_pattern' cannot be empty.")
            return

        cfg_files_to_process = [os.path.join(r, f) for r, _, fs in os.walk(source_dir) for f in fs if f.lower().endswith(".cfg")]
        if is_print_message: print(f"Found {len(cfg_files_to_process)} .cfg files to process for single signal rename.")

        processed_file_count = 0
        renamed_in_any_file_count = 0

        for cfg_file_path in cfg_files_to_process:
            processed_file_count += 1
            if is_print_message and processed_file_count % 50 == 0:
                 print(f"Processing file for single rename {processed_file_count}/{len(cfg_files_to_process)}: {os.path.basename(cfg_file_path)}")

            determined_encoding = None
            cfg_lines = None
            for encoding in cfg_encodings_to_try:
                try:
                    with open(cfg_file_path, 'r', encoding=encoding) as f:
                        cfg_lines = f.readlines()
                    determined_encoding = encoding
                    break
                except UnicodeDecodeError: continue
                except Exception: continue

            if not cfg_lines or not determined_encoding:
                if is_print_message: print(f"  Could not read {os.path.basename(cfg_file_path)} with any encoding. Skipping.")
                continue

            signal_infos = self._get_raw_signal_info_from_lines(cfg_lines, cfg_file_path)
            modified_this_file = False
            temp_cfg_lines = list(cfg_lines)

            for info in signal_infos:
                # old_name_pattern is a simple string to be found exactly in raw_name_field
                if info['raw_name_field'] == old_name_pattern:
                    if info['raw_name_field'] != new_name_field_value:
                        line_idx = info['line_index']
                        original_line_content = temp_cfg_lines[line_idx]
                        parts = original_line_content.split(',')
                        if len(parts) > 1:
                            parts[1] = new_name_field_value
                            reconstructed_line = ','.join(parts)
                            if not reconstructed_line.endswith('\n') and original_line_content.endswith('\n'):
                                reconstructed_line += '\n'

                            temp_cfg_lines[line_idx] = reconstructed_line
                            modified_this_file = True
                            if is_print_message:
                                print(f"  In {os.path.basename(cfg_file_path)} (line {line_idx+1}): Replaced raw name '{old_name_pattern}' with '{new_name_field_value}'")

            if modified_this_file:
                try:
                    with open(cfg_file_path, 'w', encoding=determined_encoding) as f_write:
                        f_write.writelines(temp_cfg_lines)
                    renamed_in_any_file_count+=1
                    if is_print_message: print(f"  Successfully updated {os.path.basename(cfg_file_path)} with single signal rename.")
                except Exception as e:
                    if is_print_message: print(f"  Error writing changes to {os.path.basename(cfg_file_path)}: {e}")

        if is_print_message: print(f"Finished single signal rename. Changes made in {renamed_in_any_file_count} file(s).")


    def merge_signal_code_csvs(self, old_csv_path: str, new_csv_path: str, merged_csv_path: str,
                               old_csv_encoding: str = 'utf-8', new_csv_encoding: str = 'utf-8',
                               old_csv_delimiter: str = ',', new_csv_delimiter: str = ',',
                               is_merge_values: bool = True, is_print_message: bool = False) -> None:

        merged_data = {} # Key: {'universal_code': str, 'Value': int}

        # Read OLD CSV
        try:
            with open(old_csv_path, mode='r', encoding=old_csv_encoding) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=old_csv_delimiter)
                if not all(col in reader.fieldnames for col in ['Key', 'universal_code', 'Value']):
                     if is_print_message: print(f"Error: Old CSV {old_csv_path} must have 'Key', 'universal_code', 'Value' columns. Found: {reader.fieldnames}")
                     return
                for row in reader:
                    try:
                        merged_data[row['Key']] = {
                            'universal_code': row['universal_code'] if row['universal_code'] not in ['-', ''] else '?', # Default to ? if undefined
                            'Value': int(row['Value'])
                        }
                    except ValueError:
                        if is_print_message: print(f"Warning: Skipping row in {old_csv_path} due to invalid 'Value': {row}")
        except FileNotFoundError:
            if is_print_message: print(f"Info: Old CSV {old_csv_path} not found. Starting fresh merge.")
        except Exception as e:
            if is_print_message: print(f"Error reading old CSV {old_csv_path}: {e}")
            return

        # Read NEW CSV and merge
        try:
            with open(new_csv_path, mode='r', encoding=new_csv_encoding) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=new_csv_delimiter)
                if not all(col in reader.fieldnames for col in ['Key', 'universal_code', 'Value']): # Assuming new also has Key, universal_code, Value
                     if is_print_message: print(f"Error: New CSV {new_csv_path} must have 'Key', 'universal_code', 'Value' columns. Found: {reader.fieldnames}")
                     return
                for row in reader:
                    key = row['Key']
                    new_universal_code = row['universal_code'] if row['universal_code'] not in ['-', ''] else '?'
                    try:
                        new_value = int(row['Value'])
                    except ValueError:
                        if is_print_message: print(f"Warning: Skipping row in {new_csv_path} for key '{key}' due to invalid 'Value': {row}")
                        continue

                    if key in merged_data:
                        # Key exists, decide how to merge
                        if is_merge_values:
                            merged_data[key]['Value'] += new_value
                        else: # Overwrite value
                            merged_data[key]['Value'] = new_value

                        # Universal code: old takes precedence if it's not default '?'
                        if merged_data[key]['universal_code'] == '?':
                            merged_data[key]['universal_code'] = new_universal_code
                        # else, keep the old one
                    else:
                        # New key
                        merged_data[key] = {
                            'universal_code': new_universal_code,
                            'Value': new_value
                        }
        except FileNotFoundError:
            if is_print_message: print(f"Error: New CSV {new_csv_path} not found. Cannot merge.")
            return # Cannot proceed without new CSV
        except Exception as e:
            if is_print_message: print(f"Error reading new CSV {new_csv_path}: {e}")
            return

        # Sort by 'Value' (count) descending
        sorted_merged_list = sorted(merged_data.items(), key=lambda item: item[1]['Value'], reverse=True)

        # Write merged data
        output_dir = os.path.dirname(merged_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(merged_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Key', 'universal_code', 'Value']) # Header
                for key, data in sorted_merged_list:
                    writer.writerow([key, data['universal_code'], data['Value']])
            if is_print_message:
                print(f"Merged signal code CSV saved to: {merged_csv_path} with {len(sorted_merged_list)} entries.")
        except IOError as e:
            if is_print_message: print(f"Error writing merged CSV to {merged_csv_path}: {e}")
        except Exception as e:
            if is_print_message: print(f"An unexpected error occurred while writing merged CSV {merged_csv_path}: {e}")
