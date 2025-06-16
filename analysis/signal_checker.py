import os
import csv
import re

class SignalChecker:
    def __init__(self, custom_signal_check_logic: callable = None):
        self._signal_check_logic = custom_signal_check_logic if custom_signal_check_logic else self._default_signal_check_logic

    def _parse_cfg_signals(self, cfg_file_path: str, encoding: str, is_print_message: bool = False) -> list[dict] | None:
        signal_details_list = []
        try:
            with open(cfg_file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()

            if len(lines) < 2:
                # Suppressing print here as check_file_signals will report overall failure if no encoding works
                # if is_print_message: print(f"Warning (_parse_cfg_signals): File {cfg_file_path} too short.")
                return None

            parts = lines[1].strip().split(',')
            if len(parts) < 3:
                # if is_print_message: print(f"Warning (_parse_cfg_signals): Second line in {cfg_file_path} has incorrect format.")
                return None

            try:
                count_analog = int(re.sub(r'\D', '', parts[1].strip())) if parts[1].strip() else 0
                count_digital = int(re.sub(r'\D', '', parts[2].strip())) if parts[2].strip() else 0
            except ValueError:
                # if is_print_message: print(f"Warning (_parse_cfg_signals): Could not parse signal counts from {cfg_file_path}.")
                return None

            current_line_idx = 2 # Start of analog channel definitions

            # Analog signals
            for _ in range(count_analog):
                if current_line_idx >= len(lines): break
                line_content = lines[current_line_idx].strip()
                current_line_idx += 1
                fields = line_content.split(',')
                if len(fields) <= 1: continue

                raw_name_field = fields[1].strip()
                name_parts = [p.strip() for p in raw_name_field.split('|')]

                details = {'raw_name': raw_name_field, 'type': 'analog', 'signal_type': None, 'location_type': None, 'section_number': None, 'phase': None}
                if not name_parts: continue

                details['signal_type'] = name_parts[0]
                if len(name_parts) > 1:
                    location_section_part = name_parts[1]
                    match = re.fullmatch(r"^(.*?)-?(\d+)$", location_section_part) # Optional hyphen before number
                    if match:
                        details['location_type'] = match.group(1).strip()
                        details['section_number'] = match.group(2).strip()
                    else:
                        details['location_type'] = location_section_part.strip()

                # Phase for analog signals is typically the 3rd field in the CFG line (fields[2])
                # but also check if embedded in name_parts like "phase: A"
                if len(fields) > 2 and fields[2].strip():
                     details['phase'] = fields[2].strip()
                elif len(name_parts) > 2 and "phase:" in name_parts[2].lower(): # Check last part of split name
                    details['phase'] = name_parts[2].split(':')[-1].strip()

                signal_details_list.append(details)

            # Digital signals
            for _ in range(count_digital):
                if current_line_idx >= len(lines): break
                line_content = lines[current_line_idx].strip()
                current_line_idx += 1
                fields = line_content.split(',')
                if len(fields) <= 1: continue

                raw_name_field = fields[1].strip()
                name_parts = [p.strip() for p in raw_name_field.split('|')]
                details = {'raw_name': raw_name_field, 'type': 'digital', 'signal_type': None, 'location_type': None, 'section_number': None, 'phase': None}
                if not name_parts: continue

                details['signal_type'] = name_parts[0]
                if len(name_parts) > 1:
                    location_section_part = name_parts[1]
                    match = re.fullmatch(r"^(.*?)-?(\d+)$", location_section_part) # Optional hyphen
                    if match:
                        details['location_type'] = match.group(1).strip()
                        details['section_number'] = match.group(2).strip()
                    else:
                        details['location_type'] = location_section_part.strip()

                if len(name_parts) > 2 and "phase:" in name_parts[2].lower(): # Check last part of split name for phase
                     details['phase'] = name_parts[2].split(':')[-1].strip()
                # Digital phase might also be in field[2] of CFG like analog, if structure is consistent
                elif len(fields) > 2 and fields[2].strip(): # Digital "phase" or state info
                     details['phase'] = fields[2].strip()


                signal_details_list.append(details)

            return signal_details_list

        except UnicodeDecodeError:
            # No print here, handled by caller if all encodings fail
            return None
        except Exception as e:
            if is_print_message: print(f"Warning (_parse_cfg_signals): Error parsing {cfg_file_path} with {encoding}: {e}")
            return None


    def _default_signal_check_logic(self, file_signals: list[dict]) -> bool:
        has_voltage_busbar_1 = False
        has_voltage_cableline_1 = False
        has_voltage_busbar_2 = False
        has_voltage_cableline_2 = False
        has_current_bus_1_ac = False
        has_current_bus_2_ac = False
        r_has_pdr_bus_1 = False
        i_has_pdr_bus_1 = False

        voltage_busbar_1_phases = set()
        voltage_cableline_1_phases = set()
        voltage_busbar_2_phases = set()
        voltage_cableline_2_phases = set()
        current_bus_1_phases = set()
        current_bus_2_phases = set()
        r_pdr_bus_1_phases = set()
        i_pdr_bus_1_phases = set()

        for signal in file_signals:
            s_type = signal.get('signal_type')
            l_type = signal.get('location_type')
            s_num = signal.get('section_number')
            phase = signal.get('phase')

            if s_type == 'U':
                if l_type == 'BusBar' and s_num == '1': voltage_busbar_1_phases.add(phase)
                elif l_type == 'CableLine' and s_num == '1': voltage_cableline_1_phases.add(phase)
                elif l_type == 'BusBar' and s_num == '2': voltage_busbar_2_phases.add(phase)
                elif l_type == 'CableLine' and s_num == '2': voltage_cableline_2_phases.add(phase)
            elif s_type == 'I':
                if l_type == 'Bus' and s_num == '1': current_bus_1_phases.add(phase)
                elif l_type == 'Bus' and s_num == '2': current_bus_2_phases.add(phase)
            elif s_type == 'PDR':
                if l_type == 'Bus' and s_num == '1': r_pdr_bus_1_phases.add(phase)
            elif s_type == 'PDR_ideal':
                if l_type == 'Bus' and s_num == '1': i_pdr_bus_1_phases.add(phase)

        required_abc_phases = {'A', 'B', 'C'}
        has_voltage_busbar_1 = required_abc_phases.issubset(voltage_busbar_1_phases)
        has_voltage_cableline_1 = required_abc_phases.issubset(voltage_cableline_1_phases)
        has_voltage_busbar_2 = required_abc_phases.issubset(voltage_busbar_2_phases)
        has_voltage_cableline_2 = required_abc_phases.issubset(voltage_cableline_2_phases)

        required_ac_phases_for_current = {'A', 'C'}
        has_current_bus_1_ac = required_ac_phases_for_current.issubset(current_bus_1_phases) and len(current_bus_1_phases) >=2
        has_current_bus_2_ac = required_ac_phases_for_current.issubset(current_bus_2_phases) and len(current_bus_2_phases) >=2

        r_has_pdr_bus_1 = {'PS'}.issubset(r_pdr_bus_1_phases) or required_abc_phases.issubset(r_pdr_bus_1_phases)
        i_has_pdr_bus_1 = {'PS'}.issubset(i_pdr_bus_1_phases) or required_abc_phases.issubset(i_pdr_bus_1_phases)

        voltage_condition = (has_voltage_busbar_1 or has_voltage_cableline_1) and \
                            (has_voltage_busbar_2 or has_voltage_cableline_2)
        current_condition = has_current_bus_1_ac and has_current_bus_2_ac
        pdr_condition = r_has_pdr_bus_1 or i_has_pdr_bus_1

        return voltage_condition and current_condition and pdr_condition


    def check_file_signals(self, cfg_file_path: str, encodings_to_try: list = None, is_print_message: bool = False) -> str:
        if encodings_to_try is None:
            encodings_to_try = ['utf-8', 'windows-1251', 'cp866']

        parsed_signals = None
        used_encoding = None
        for encoding in encodings_to_try:
            # if is_print_message: print(f"  Attempting to parse {os.path.basename(cfg_file_path)} with {encoding}...")
            parsed_signals = self._parse_cfg_signals(cfg_file_path, encoding, is_print_message=False) # Reduce noise from _parse
            if parsed_signals is not None:
                used_encoding = encoding
                if is_print_message: print(f"  Successfully parsed {os.path.basename(cfg_file_path)} with {used_encoding}.")
                break

        if parsed_signals is None:
            if is_print_message: print(f"  Failed to parse {os.path.basename(cfg_file_path)} with any encoding.")
            return "Error"

        if not parsed_signals:
            if is_print_message: print(f"  No signals parsed from {os.path.basename(cfg_file_path)} (encoding: {used_encoding}). Will be marked 'No'.")
            # Logic treats empty signal list as "No" unless check_logic handles it differently
            pass

        check_passed = self._signal_check_logic(parsed_signals)

        result = "Yes" if check_passed else "No"
        if is_print_message: print(f"  Check result for {os.path.basename(cfg_file_path)}: {result}")
        return result

    def check_signals_in_directory(self, source_dir: str, output_csv_path: str,
                                   custom_signal_check_logic: callable = None,
                                   cfg_encodings_to_try: list = None,
                                   is_print_message: bool = False) -> None:
        if not os.path.isdir(source_dir):
            if is_print_message: print(f"Error: Source directory '{source_dir}' not found.")
            return

        original_check_logic = self._signal_check_logic # Backup
        if custom_signal_check_logic:
            self._signal_check_logic = custom_signal_check_logic

        results = []
        cfg_files_to_process = []
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                if file_name.lower().endswith(".cfg"):
                    cfg_files_to_process.append(os.path.join(root, file_name))

        if is_print_message:
            total_to_check = len(cfg_files_to_process)
            print(f"Found {total_to_check} .cfg files to check signals in {source_dir}.")

        processed_count = 0
        for cfg_file_path in cfg_files_to_process:
            processed_count +=1
            if is_print_message and processed_count % 50 == 0 :
                print(f"Checking file {processed_count}/{total_to_check}: {os.path.basename(cfg_file_path)}")

            file_hash_name = os.path.basename(cfg_file_path)[:-4]
            status = self.check_file_signals(cfg_file_path, cfg_encodings_to_try, is_print_message=is_print_message) # Pass flag
            results.append({'filename': file_hash_name, 'contains_required_signals': status})

        if custom_signal_check_logic: # Restore
            self._signal_check_logic = original_check_logic

        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                 if is_print_message: print(f"Error creating output directory {output_dir}: {e}")
                 return

        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'contains_required_signals']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            if is_print_message: print(f"Signal check results saved to: {output_csv_path}")
        except IOError as e:
            if is_print_message: print(f"Error writing results CSV to {output_csv_path}: {e}")
        except Exception as e:
            if is_print_message: print(f"An unexpected error occurred while writing results CSV: {e}")
