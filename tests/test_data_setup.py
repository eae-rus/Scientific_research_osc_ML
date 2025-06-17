import os
import shutil
import json
import csv
import numpy as np # For sample data generation
import zipfile
import py7zr
from typing import List, Dict, Any # For type hints

# Attempt to import aspose.zip for RAR creation, not strictly necessary for setup if RARs are pre-made or skipped
try:
    import aspose.zip as az
    HAS_ASPOSE_ZIP = True
except ImportError:
    HAS_ASPOSE_ZIP = False
    # print("Warning: Aspose.Zip for .NET not found. RAR creation/full testing might be limited.")

# --- Base Test Data Directory ---
# Assume this script is in the 'tests' directory or project root for path to work.
# If script is in tests/, then tests/sample_data is correct.
# If script is in root/, then tests/sample_data is also correct.
BASE_TEST_DIR = "tests"
SAMPLE_DATA_DIR = os.path.join(BASE_TEST_DIR, "sample_data")

# Subdirectories for organization
COMTRADE_FILES_DIR = os.path.join(SAMPLE_DATA_DIR, "comtrade_files")
ARCHIVES_DIR = os.path.join(SAMPLE_DATA_DIR, "archives")
CONFIG_FILES_DIR = os.path.join(SAMPLE_DATA_DIR, "config_files") # For CSVs, JSONs

# --- Helper to create COMTRADE files ---
def create_comtrade_sample(base_path: str, name: str, cfg_content: str, dat_content: str, cfg_encoding='utf-8', dat_encoding='utf-8'):
    os.makedirs(base_path, exist_ok=True)
    cfg_file_path = os.path.join(base_path, f"{name}.cfg")
    dat_file_path = os.path.join(base_path, f"{name}.dat")
    with open(cfg_file_path, "w", encoding=cfg_encoding) as f:
        f.write(cfg_content)
    with open(dat_file_path, "w", encoding=dat_encoding) as f:
        f.write(dat_content)
    # print(f"Created sample: {cfg_file_path} & {dat_file_path}")
    return cfg_file_path, dat_file_path

def setup_sample_test_data():
    """
    Sets up a directory structure with various sample files for testing.
    """
    if os.path.exists(SAMPLE_DATA_DIR):
        print(f"Cleaning up existing sample data directory: {SAMPLE_DATA_DIR}")
        shutil.rmtree(SAMPLE_DATA_DIR)

    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    os.makedirs(COMTRADE_FILES_DIR, exist_ok=True)
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    os.makedirs(CONFIG_FILES_DIR, exist_ok=True)
    print(f"Created base directories in: {SAMPLE_DATA_DIR}")

    # --- 1. COMTRADE Files ---
    valid_cfg_dat_1_path = os.path.join(COMTRADE_FILES_DIR, "valid_cfg_dat_1_subdir") # Changed to subdir
    cfg_content_A = """SampleStationA,DeviceA,1999
2,1A,1D
1,U_PhaseA,A,,kV,1.0,0.0,0,-32767,32767,10,0.1,P
2,Status_1,,,1
50.0
1
1000,3
01/01/2024,10:00:00.000000
01/01/2024,10:00:00.001000
ASCII
1.0
"""
    dat_content_A = """1,0,1.23,0
2,1000,2.34,1
3,2000,3.45,0
"""
    create_comtrade_sample(valid_cfg_dat_1_path, "sample_A", cfg_content_A, dat_content_A)

    cff_content_1 = """--- file type: CFF Comtrade version: 1.00 type: ascii ---
--- file type: CFG ---
SampleStationCFF,DeviceCFF,2013
1,1A,0D
1,I_PhaseC,C,,kA,1.0,0,0,-1000,1000,5,1,S
60.0
1
2000,2
15/07/2024,12:30:00.123456
15/07/2024,12:30:00.123456
ASCII
1.0
--- file type: DAT ascii: 2 ---
1,0,0.5
2,500,0.75
--- file type: END ---
"""
    with open(os.path.join(COMTRADE_FILES_DIR, "valid_cff_1.cff"), "w", encoding="utf-8") as f:
        f.write(cff_content_1)
    # print(f"Created sample: {os.path.join(COMTRADE_FILES_DIR, 'valid_cff_1.cff')}")

    anonymizer_test_path = os.path.join(COMTRADE_FILES_DIR, "for_anonymizer_1")
    cfg_to_anonymize = """MyHomeStation,MyDeviceID,1991
1,1A,0D
1,Voltage,A,,V,1,0,0,-100,100,100,1,P
50
1
100,10
25/12/2023,11:22:33.444555
25/12/2023,11:22:33.544555
ASCII
1
"""
    dat_to_anonymize = "1,0,10\n2,10000,20\n"
    create_comtrade_sample(anonymizer_test_path, "to_anonymize", cfg_to_anonymize, dat_to_anonymize)

    # --- 2. Config/CSV/JSON Files ---
    norm_coeffs_path = os.path.join(CONFIG_FILES_DIR, "sample_norm_coeffs.csv")
    norm_coeffs_data = [
        ["name", "norm", "1Ub_PS", "1Ub_base", "1Ip_PS", "1Ip_base", "1Uc_base"], # Added 1Uc_base for Overvoltage
        ["hash123abc", "YES", "s", (10000.0/np.sqrt(3)), "p", 5.0, (10000.0/np.sqrt(3))],
        ["hash456def", "NO", "s", (6000.0/np.sqrt(3)), "s", "Noise", (6000.0/np.sqrt(3))]
    ]
    with open(norm_coeffs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(norm_coeffs_data)
    # print(f"Created sample: {norm_coeffs_path}")

    analog_names_path = os.path.join(CONFIG_FILES_DIR, "dict_analog_names_sample.json")
    analog_names_content = {
        "bus1": { "UA": ["U_PhaseA", "U | BusBar-1 | phase: A"], "IA": ["I_PhaseA_alt"] },
        "bus2": { "UC": ["U_PhaseC_Bus2"] } }
    with open(analog_names_path, "w", encoding="utf-8") as f: json.dump(analog_names_content, f, indent=4)
    # print(f"Created sample: {analog_names_path}")

    discrete_names_path = os.path.join(CONFIG_FILES_DIR, "dict_discrete_names_sample.json")
    with open(discrete_names_path, "w", encoding="utf-8") as f: json.dump({},f) # Empty for now

    rename_map_path = os.path.join(CONFIG_FILES_DIR, "signal_rename_map.csv")
    rename_map_data = [
        ["Key", "universal_code"],
        ["U_PhaseA", "Ua_New_From_RawFieldName"],
        ["U_PhaseA | phase:A", "Ua_New_From_CombinedName"] # Example for combined name key
    ]
    with open(rename_map_path, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(rename_map_data)
    # print(f"Created sample: {rename_map_path}")

    # --- 3. Archive Files ---
    archive_content_path = os.path.join(COMTRADE_FILES_DIR, "for_archive_1") # Content will be created here
    cfg_arc_A_content = """ArcStationA,ArcDeviceA,1999
1,1A,0D
1,U_Arc,A,,kV,1.0,0,0,-10,10,1,1,P
50.0
1
100,1
01/01/2024,01:00:00.000000
01/01/2024,01:00:00.000000
ASCII
1.0
"""
    dat_arc_A_content = "1,0,1.0\n"
    create_comtrade_sample(archive_content_path, "sample_arc_A", cfg_arc_A_content, dat_arc_A_content)

    zip_file_path = os.path.join(ARCHIVES_DIR, "archive_A.zip")
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.write(os.path.join(archive_content_path, "sample_arc_A.cfg"), arcname="sample_arc_A.cfg")
        zf.write(os.path.join(archive_content_path, "sample_arc_A.dat"), arcname="sample_arc_A.dat")
    # print(f"Created sample archive: {zip_file_path}")

    sz_file_path = os.path.join(ARCHIVES_DIR, "archive_B.7z")
    with py7zr.SevenZipFile(sz_file_path, 'w') as szf:
        szf.write(os.path.join(archive_content_path, "sample_arc_A.cfg"), arcname="sample_arc_A_in7z.cfg")
        szf.write(os.path.join(archive_content_path, "sample_arc_A.dat"), arcname="sample_arc_A_in7z.dat")
    # print(f"Created sample archive: {sz_file_path}")

    # Dummy RAR if aspose.zip is not available or fails
    rar_file_path = os.path.join(ARCHIVES_DIR, "archive_C.rar")
    if HAS_ASPOSE_ZIP:
        try:
            # Aspose.zip might require specific paths or file objects.
            # This creates an empty RAR file. Adding files can be complex.
            with az.Archive() as archive:
                archive.save(rar_file_path)
            # print(f"Created (empty) RAR: {rar_file_path}")
        except Exception as e:
            # print(f"Could not create RAR with Aspose.Zip: {e}. Creating dummy RAR.")
            with open(rar_file_path, "w") as f: f.write("dummy_rar_content_cannot_create")
    else:
        with open(rar_file_path, "w") as f: f.write("dummy_rar_content_aspose_missing")

    # --- 4. Malformed CFG ---
    print("\n--- Creating Malformed CFG files ---")
    malformed_cfg_path = os.path.join(COMTRADE_FILES_DIR, "malformed_cfg_files")
    os.makedirs(malformed_cfg_path, exist_ok=True)
    malformed_cfg_content_1 = """SampleStation,Device,1999
# Missing total channel line
1,U_PhaseA,A,,kV,1.0,0.0,0,-32767,32767,10,0.1,P
50.0
1
1000,3
01/01/2024,10:00:00.000000
01/01/2024,10:00:00.001000
ASCII
1.0
"""
    with open(os.path.join(malformed_cfg_path, "malformed_cfg_1.cfg"), "w", encoding="utf-8") as f:
        f.write(malformed_cfg_content_1)
    # print(f"Created: {os.path.join(malformed_cfg_path, 'malformed_cfg_1.cfg')}")

    # --- 5. Empty DAT ---
    print("\n--- Creating Empty DAT files ---")
    empty_dat_path = os.path.join(COMTRADE_FILES_DIR, "empty_dat_1")
    os.makedirs(empty_dat_path, exist_ok=True)
    empty_dat_cfg_content = """SampleStation,Device,1999
1,1A,0D
1,U_PhaseA,A,,kV,1.0,0.0,0,-32767,32767,10,0.1,P
50.0
1
1000,0 # 0 samples
01/01/2024,10:00:00.000000
01/01/2024,10:00:00.000000
ASCII
1.0
"""
    empty_dat_content = "" # Empty DAT file
    create_comtrade_sample(empty_dat_path, "empty_A", empty_dat_cfg_content, empty_dat_content)

    # --- 6. Specific Signals for SignalChecker ---
    print("\n--- Creating Specific Signals for SignalChecker ---")
    specific_signals_path = os.path.join(COMTRADE_FILES_DIR, "specific_signals_1")
    os.makedirs(specific_signals_path, exist_ok=True)
    # Content for signals_for_checker.cfg (to pass _default_signal_check_logic)
    checker_cfg_content = """Station,Dev,2013
11,10A,1D
1,U | BusBar-1 | phase: A,A,,V,1,0,0,0,0,1,1,P
2,U | BusBar-1 | phase: B,B,,V,1,0,0,0,0,1,1,P
3,U | BusBar-1 | phase: C,C,,V,1,0,0,0,0,1,1,P
4,U | CableLine-2 | phase: A,A,,V,1,0,0,0,0,1,1,P
5,U | CableLine-2 | phase: B,B,,V,1,0,0,0,0,1,1,P
6,U | CableLine-2 | phase: C,C,,V,1,0,0,0,0,1,1,P
7,I | Bus-1 | phase: A,A,,A,1,0,0,0,0,1,1,P
8,I | Bus-1 | phase: C,C,,A,1,0,0,0,0,1,1,P
9,I | Bus-2 | phase: A,A,,A,1,0,0,0,0,1,1,P
10,I | Bus-2 | phase: C,C,,A,1,0,0,0,0,1,1,P
11,PDR | Bus-1 | phase: PS,,,1
50.0
1
1000,3
01/01/2024,00:00:00.000
01/01/2024,00:00:01.000
ASCII
1.0
"""
    checker_dat_content = "1,0,0,0,0,0,0,0,0,0,0,0,1\n2,1000,0,0,0,0,0,0,0,0,0,0,1\n3,2000,0,0,0,0,0,0,0,0,0,0,1\n"
    create_comtrade_sample(specific_signals_path, "signals_for_checker", checker_cfg_content, checker_dat_content)

    # Content for neutral_current.cfg
    neutral_cfg_content = """Station,Dev,1999
2,2A,0D
1,Ua,A,,V,1,0,0,0,100,1,1,P
2,I | Bus 1 | phase: N,N,,A,1,0,0,0,100,1,1,P
50.0
1
1000,2
01/01/2024,00:00:00.000
01/01/2024,00:00:00.001
ASCII
1.0
"""
    neutral_dat_content = "1,0,10,5\n2,1000,10,5\n"
    create_comtrade_sample(specific_signals_path, "neutral_current", neutral_cfg_content, neutral_dat_content)

    # --- 7. For Normalization Coefficient Generation ---
    print("\n--- Creating files for Normalization Coefficient Generation ---")
    norm_gen_path = os.path.join(COMTRADE_FILES_DIR, "for_normalization_1")
    os.makedirs(norm_gen_path, exist_ok=True)

    time_norm = np.linspace(0, 2*np.pi*5, 100) # 5 periods, 100 samples
    dat_rows_norm = 100

    cfg_norm_base = """StationN,DeviceN,1999
{num_analog}A,{num_analog}A,0D
{analog_lines}50.0
1
1000,{dat_rows}
01/01/2023,00:00:00.000
01/01/2023,00:00:00.000
ASCII
1.0
"""
    # Low amplitude signal (Noise)
    analog_low = [("I_low_noise", 1.0, 1.0, np.sin(time_norm) * 0.01)] # H1 ~ 0.01, CURRENT_T1_S ~ 0.042
    # High primary signal
    analog_high_P = [("U_high_primary", 10000.0, 10.0, np.sin(time_norm) * 20)] # H1 втор = 20, H1 перв = 20000
    # High harmonics
    signal_distorted_norm = np.sin(time_norm) * 50 + np.sin(time_norm * 3) * 40 # H1=50, H3=40
    analog_distorted = [("U_distorted_harmonics", 10000.0, 100.0, signal_distorted_norm)]

    all_norm_analog_data = analog_low + analog_high_P + analog_distorted

    analog_lines_content = ""
    for idx, (ch_name, prim, sec, _) in enumerate(all_norm_analog_data):
        analog_lines_content += f"{idx+1},{ch_name},,,V,1,0,0,-32767,32767,{prim},{sec},P\n"

    final_cfg_norm_content = cfg_norm_base.format(
        num_analog=len(all_norm_analog_data),
        analog_lines=analog_lines_content,
        dat_rows=dat_rows_norm
    )

    dat_norm_content = ""
    for i in range(dat_rows_norm):
        dat_norm_content += f"{i+1},{i*1000}" # n, timestamp_us
        for _, _, _, values in all_norm_analog_data:
            dat_norm_content += f",{values[i % len(values)]:.6f}"
        dat_norm_content += "\n"
    create_comtrade_sample(norm_gen_path, "for_norm_coeffs", final_cfg_norm_content, dat_norm_content)

    # --- 8. For SPEF/Overvoltage ---
    print("\n--- Creating files for SPEF/Overvoltage Detection ---")
    spef_ov_path = os.path.join(COMTRADE_FILES_DIR, "for_spef_overvoltage_1")
    os.makedirs(spef_ov_path, exist_ok=True)

    time_spef = np.linspace(0, 2*np.pi*10, 200) # 200 samples
    nominal_peak = 10000.0 * np.sqrt(2) / np.sqrt(3) # For 10kV system, phase peak

    ua_overvoltage_data = np.concatenate([np.sin(time_spef[:80])*nominal_peak,
                                          np.sin(time_spef[80:120])*nominal_peak*1.8, # Overvoltage 1.8 PU
                                          np.sin(time_spef[120:])*nominal_peak])
    un_spef_data = np.concatenate([np.zeros(80), np.sin(time_spef[80:120])*0.15*nominal_peak, np.zeros(80)]) # U0 = 0.15 PU

    spef_analog_data = [
        ("U | BusBar-1 | phase: A", ua_overvoltage_data),
        ("U | BusBar-1 | phase: B", np.sin(time_spef - 2*np.pi/3)*nominal_peak*0.6),
        ("U | BusBar-1 | phase: C", np.sin(time_spef + 2*np.pi/3)*nominal_peak*0.6),
        ("U | BusBar-1 | phase: N", un_spef_data)
    ]
    create_comtrade_sample(spef_ov_path, "spef_present", cfg_content_A, dat_content_A) # Reusing generic CFG for structure
    # Need to use create_workflow_cfg_dat or similar for easier DAT content generation matching analog_ch_details
    # For now, reusing create_comtrade_sample and manually crafting DAT for spef_present
    dat_spef_present_content = ""
    for i in range(200):
        dat_spef_present_content += f"{i+1},{i*1000}"
        for _, s_data in spef_analog_data: dat_spef_present_content += f",{s_data[i]:.6f}"
        dat_spef_present_content += "\n"
    with open(os.path.join(spef_ov_path, "spef_present.dat"), "w", encoding="utf-8") as f: f.write(dat_spef_present_content)
    # Create a CFG that matches the channel names in spef_analog_data
    cfg_spef_present_content = f"SPEFStation,SPEFDevice,1999\n{len(spef_analog_data)},{len(spef_analog_data)}A,0D\n"
    for idx, (ch_name, _) in enumerate(spef_analog_data):
         cfg_spef_present_content += f"{idx+1},{ch_name},,,V,1,0,0,-32767,32767,1,1,P\n"
    cfg_spef_present_content += "50.0\n1\n1000,200\n01/01/2024,00:00:00.000000\n01/01/2024,00:00:00.001000\nASCII\n1.0\n"
    with open(os.path.join(spef_ov_path, "spef_present.cfg"), "w", encoding="utf-8") as f: f.write(cfg_spef_present_content)


    normal_analog_data = [
        ("U | BusBar-1 | phase: A", np.sin(time_spef)*nominal_peak),
        ("U | BusBar-1 | phase: B", np.sin(time_spef - 2*np.pi/3)*nominal_peak),
        ("U | BusBar-1 | phase: C", np.sin(time_spef + 2*np.pi/3)*nominal_peak)
    ]
    # create_comtrade_sample(spef_ov_path, "no_spef", normal_analog_data, "") # Needs DAT
    dat_no_spef_content = ""
    for i in range(200):
        dat_no_spef_content += f"{i+1},{i*1000}"
        for _, s_data in normal_analog_data: dat_no_spef_content += f",{s_data[i]:.6f}"
        dat_no_spef_content += "\n"
    cfg_no_spef_content = f"NormalStation,NormalDevice,1999\n{len(normal_analog_data)},{len(normal_analog_data)}A,0D\n"
    for idx, (ch_name, _) in enumerate(normal_analog_data):
         cfg_no_spef_content += f"{idx+1},{ch_name},,,V,1,0,0,-32767,32767,1,1,P\n"
    cfg_no_spef_content += "50.0\n1\n1000,200\n01/01/2024,00:00:00.000000\n01/01/2024,00:00:00.001000\nASCII\n1.0\n"
    create_comtrade_sample(spef_ov_path, "no_spef", cfg_no_spef_content, dat_no_spef_content)


    # --- 9. For Signal Names ---
    print("\n--- Creating files for Signal Names testing ---")
    signal_names_path = os.path.join(COMTRADE_FILES_DIR, "for_signal_names_1")
    os.makedirs(signal_names_path, exist_ok=True)
    cfg_names_A_content = """StationSN,DeviceSN1,1999
3,2A,1D
1,Ua | phase:A,A,,V,1,0,0,0,100,1,1,P
2,Ua | phase:A,A,,V,1,0,0,0,100,1,1,P
3,DigitalX,,,1
50.0
1
1000,3
01/01/2024,10:00:00.000
01/01/2024,10:00:00.001
ASCII
1.0
"""
    dat_names_A_content = "1,0,10,10,1\n2,1000,12,12,0\n3,2000,14,14,1\n"
    create_comtrade_sample(signal_names_path, "names_A", cfg_names_A_content, dat_names_A_content)

    cfg_names_B_content = """StationSN,DeviceSN2,1999
2,1A,1D
1,Ub | phase:B,B,,V,1,0,0,0,100,1,1,P
2,DigitalY,,,0
50.0
1
1000,2
01/01/2024,11:00:00.000
01/01/2024,11:00:00.001
ASCII
1.0
"""
    dat_names_B_content = "1,0,20,0\n2,1000,22,1\n"
    create_comtrade_sample(signal_names_path, "names_B", cfg_names_B_content, dat_names_B_content)

    # --- 10. Encoding Test Files ---
    print("\n--- Creating Encoding Test files ---")
    encoding_test_path = os.path.join(COMTRADE_FILES_DIR, "encoding_tests")
    os.makedirs(encoding_test_path, exist_ok=True)

    cfg_cp1251_content = "СтанцияCP1251,УстройствоCP1251,1999\n1,1A,0D\n1,НапряжениеАБС,АБС,,В,1,0,0,0,1,1,1,P\n50\n1\n1000,1\n01/01/2000,00:00:00\n01/01/2000,00:00:00\nASCII\n1"
    dat_cp1251_content = "1,0,123\n"
    create_comtrade_sample(encoding_test_path, "cp1251_encoded", cfg_cp1251_content, dat_cp1251_content, cfg_encoding='cp1251', dat_encoding='cp1251')

    cfg_cp866_content = "СтанцияCP866,ДевайсCP866,1999\n1,1A,0D\n1,ТокXYZ,XYZ,,А,1,0,0,0,1,1,1,P\n50\n1\n1000,1\n01/01/2000,00:00:00\n01/01/2000,00:00:00\nASCII\n1"
    dat_cp866_content = "1,0,456\n"
    create_comtrade_sample(encoding_test_path, "cp866_encoded", cfg_cp866_content, dat_cp866_content, cfg_encoding='cp866', dat_encoding='cp866')

    # --- 11. Additional Config Files ---
    print("\n--- Creating Additional Config files ---")
    with open(os.path.join(CONFIG_FILES_DIR, "empty_norm_coeffs.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "norm", "1Ub_PS", "1Ub_base", "1Ip_PS", "1Ip_base"]) # Only headers
    # print(f"Created: {os.path.join(CONFIG_FILES_DIR, 'empty_norm_coeffs.csv')}")

    terminal_hashes_content = {
        "hash_term1_file1": ["file1.cfg", "path/to/T00123/file1.cfg"],
        "hash_term1_file2": ["file2.dat", "path/somewhere/T00123/file2.dat"],
        "hash_term2_file3": ["file3.cff", "path/other/T00456/file3.cff"]
    }
    with open(os.path.join(CONFIG_FILES_DIR, "terminal_hashes_input.json"), "w", encoding="utf-8") as f:
        json.dump(terminal_hashes_content, f, indent=4)
    # print(f"Created: {os.path.join(CONFIG_FILES_DIR, 'terminal_hashes_input.json')}")

    activity_filter_sample_config = {
        'channels_to_analyze_patterns': ['i ', 'u '],
        'current_channel_id_patterns': ['i '],
        'voltage_channel_id_patterns': ['u '],
        'use_norm_osc': True,
        'norm_yes_phrase': "YES",
        'thresholds_current_normalized': {'delta': 0.1, 'std_dev': 0.05, 'max_abs': 0.1},
        'thresholds_voltage_normalized': {'delta': 0.05, 'std_dev': 0.02, 'max_abs': 0.05},
        'raw_signal_analysis': {
            'initial_window_check_periods': 2,
            'h1_vs_hx_ratio_threshold_U': 10,
            'h1_vs_hx_ratio_threshold_I': 5,
            'min_initial_h1_amplitude_for_rel_norm': 0.01,
            'thresholds_raw_current_relative': {'delta': 0.5, 'std_dev': 0.3, 'max_abs': 0.5},
            'thresholds_raw_voltage_relative': {'delta': 0.3, 'std_dev': 0.2, 'max_abs': 0.3}
        },
        'verbose': False
    }
    with open(os.path.join(CONFIG_FILES_DIR, "activity_filter_config.json"), "w", encoding="utf-8") as f:
        json.dump(activity_filter_sample_config, f, indent=4)
    # print(f"Created: {os.path.join(CONFIG_FILES_DIR, 'activity_filter_config.json')}")

    overvoltage_detector_sample_config = {
        'VALID_NOMINAL_VOLTAGES': [6000.0, 10000.0, 35000.0, 110000.0, 220000.0, 6000.0/np.sqrt(3), 10000.0/np.sqrt(3)],
        'SPEF_THRESHOLD_U0': 0.05,
        'SPEF_THRESHOLD_Un': 0.03,
        'SPEF_MIN_DURATION_PERIODS': 2,
        'SIMILAR_AMPLITUDES_FILTER_ENABLED': True,
        'SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE': 0.10,
        'overvoltage_report_bins': [0, 1.1, 1.3, 1.5, 2.0, 10.0],
        'overvoltage_report_labels': ["<1.1", "1.1-1.3", "1.3-1.5", "1.5-2.0", ">2.0"],
        'norm_yes_phrase': "YES",
        'verbose': False
    }
    with open(os.path.join(CONFIG_FILES_DIR, "overvoltage_detector_config.json"), "w", encoding="utf-8") as f:
        json.dump(overvoltage_detector_sample_config, f, indent=4)
    # print(f"Created: {os.path.join(CONFIG_FILES_DIR, 'overvoltage_detector_config.json')}")

    print(f"\nSample data setup complete in {SAMPLE_DATA_DIR}")
    print("Directory structure:")
    for root, dirs, files in os.walk(SAMPLE_DATA_DIR):
        level = root.replace(SAMPLE_DATA_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')

if __name__ == '__main__':
    setup_sample_test_data()
