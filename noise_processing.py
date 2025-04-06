import csv
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

def generate_analog_signals_name(is_print_to_console: bool = False, is_print_to_json: bool = False, path_to_json_file: str = "") -> dict:
    """
    The function generates groups with all the names of the signals
      
    Args:
        is_print_to_console (bool): printing to the console
        is_print_to_json (bool): printing to a json file
        path_to_json_file (str): path to the json file

    Returns:
        dict - dictionary with groups and names of signals
    """
    dict_names = {}
    for i in range(1, 9):
        if is_print_to_console:
            print(f"------------ Bus {i} ------------")
        dict_names_new = {}
        dict_names_new[f"U BusBar {i}"] = []
        dict_names_new[f"U CableLine {i}"] = []
        dict_names_new[f"I phase {i}"] = []
        dict_names_new[f"I zero {i}"] = []

        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: A")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: B")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: C")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: N")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: AB")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: BC")
        dict_names_new [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: CA")
        
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: A")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: B")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: C")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: N")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: AB")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: BC")
        dict_names_new[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: CA")
        
        dict_names_new[f"I phase {i}"].append(f"I | Bus-{i} | phase: A")
        dict_names_new[f"I phase {i}"].append(f"I | Bus-{i} | phase: B")
        dict_names_new[f"I phase {i}"].append(f"I | Bus-{i} | phase: C")
        
        dict_names_new[f"I zero {i}"].append(f"I | Bus-{i} | phase: N")
        
        # with other types of sensors, as long as these are Rogovsky belts and capacitive dividers
        dict_names_new[f"U_raw BusBar {i}"] = []
        dict_names_new[f"U_raw CableLine {i}"] = []
        dict_names_new[f"I_raw phase {i}"] = []
        dict_names_new[f"I_raw zero {i}"] = []
        
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: A")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: B")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: C")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: N")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: AB")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: BC")
        dict_names_new [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: CA")
        
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: A")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: B")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: C")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: N")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: AB")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: BC")
        dict_names_new[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: CA")
        
        dict_names_new[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: A")
        dict_names_new[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: B")
        dict_names_new[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: C")
        
        dict_names_new[f"I_raw zero {i}"].append(f"I_raw | Bus-{i} | phase: N")
        
        
        dict_names[f"Bus {i}"] = dict_names_new
        if is_print_to_console:
            print(dict_names_new)
    
    dict_names_new = {}
    dict_names_new["Diff 1"] = []
    dict_names_new["Diff 1"].append("I | dif-1 | phase: A")
    dict_names_new["Diff 1"].append("I | dif-1 | phase: B")
    dict_names_new["Diff 1"].append("I | dif-1 | phase: C")
    dict_names_new["Diff 1"].append("I | braking-1 | phase: A")
    dict_names_new["Diff 1"].append("I | braking-1 | phase: B")
    dict_names_new["Diff 1"].append("I | braking-1 | phase: C")
    if is_print_to_console:
        print("------------ Diff 1 ------------")
        print(dict_names_new)
    
    dict_names["Diff current"] = dict_names_new
    
    if is_print_to_json:
        path = path_to_json_file + '/dict_analog_names.json'
        with open(path, 'w') as fp:
            json.dump(dict_names, fp)
    
    return dict_names

def generate_discrete_signals_name(is_print_to_console: bool = False, is_print_to_json: bool = False, path_to_json_file: str = "") -> dict:
    """
    The function generates groups with all the names of the signals
      
    Args:
        is_print_to_console (bool): printing to the console
        is_print_to_json (bool): printing to a json file
        path_to_json_file (str): path to the json file

    Returns:
        dict - dictionary with groups and names of signals
    """
    dict_names = {}
    
    dict_names_new = {}
    dict_names_new["FABT"], dict_names_new["RNM"] = [], []
    dict_names_new["FABT"].append("FABT | Bus-0 | operation")
    dict_names_new["RNM"].append("RNM | Bus-0 | operation")
    dict_names[f"Bus 0 (common signals)"] = dict_names_new
    if is_print_to_console:
        print(f"------------ bus 0 (general signals) ------------")
        print(dict_names_new)
    
    dict_names["Bus 0 (common signals)"] = dict_names_new
    
    for i in range(1, 9):
        if is_print_to_console:
            print(f"------------ bus {i} ------------")
        dict_names_new = {}
        dict_names_new[f"PDR B{i}"] = []
        dict_names_new[f"IFB B{i}"] = []
        dict_names_new[f"BC B{i}"] = []
        dict_names_new[f"BC Bypass B{i}"] = []
        dict_names_new[f"FABT B{i}"] = []
        dict_names_new[f"RNM B{i}"] = []

        dict_names_new[f"PDR B{i}"].append(f"PDR | Bus-{i} | phase: A")
        dict_names_new[f"PDR B{i}"].append(f"PDR | Bus-{i} | phase: B")
        dict_names_new[f"PDR B{i}"].append(f"PDR | Bus-{i} | phase: C")
        dict_names_new[f"PDR B{i}"].append(f"PDR | Bus-{i} | phase: PS") # Direct sequence or common signal
        # "PDR_ideal" - for scientific research
        dict_names_new[f"PDR B{i}"].append(f"PDR_ideal | Bus-{i} | phase: A")
        dict_names_new[f"PDR B{i}"].append(f"PDR_ideal | Bus-{i} | phase: B")
        dict_names_new[f"PDR B{i}"].append(f"PDR_ideal | Bus-{i} | phase: C")
        dict_names_new[f"PDR B{i}"].append(f"PDR_ideal | Bus-{i} | phase: PS") # Direct sequence or common signal
        
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | open")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | close")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | open | phase: A")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | open | phase: B")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | open | phase: C")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | close | phase: A")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | close | phase: B")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | close | phase: C")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command open")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command close")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command open | phase: A")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command open | phase: B")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command open | phase: C")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command close | phase: A")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command close | phase: B")
        dict_names_new[f"IFB B{i}"].append(f"IFB | Bus-{i} | command close | phase: C")
        
        dict_names_new[f"BC B{i}"].append(f"BC | Bus-{i} | open")
        dict_names_new[f"BC B{i}"].append(f"BC | Bus-{i} | close")
        dict_names_new[f"BC B{i}"].append(f"BC | Bus-{i} | command open")
        dict_names_new[f"BC B{i}"].append(f"BC | Bus-{i} | command close")
        
        dict_names_new[f"BC Bypass B{i}"].append(f"BC Bypass | Bus-{i} | open")
        dict_names_new[f"BC Bypass B{i}"].append(f"BC Bypass | Bus-{i} | close")
        dict_names_new[f"BC Bypass B{i}"].append(f"BC Bypass | Bus-{i} | command open")
        dict_names_new[f"BC Bypass B{i}"].append(f"BC Bypass | Bus-{i} | command close")
        
        dict_names_new[f"FABT B{i}"].append(f"FABT | Bus-{i} | operation")
        dict_names_new[f"RNM B{i}"].append(f"RNM | Bus-{i} | operation")
    
        dict_names[f"Bus {i}"] = dict_names_new
        if is_print_to_console:
            print(dict_names_new)
    
    if is_print_to_json:
        path = path_to_json_file + '/dict_discrete_names.json'
        with open(path, "w") as file:
            json.dump(dict_names, file)
    return dict_names

def generate_group_signals_from_csv(is_print_to_console: bool = False) -> dict:
    """
    The function generates groups with all the names of the signals

    Args:

    Returns:
        dict - dictionary with groups and names of signals
    """
    dict_group_names = {}
    dict_group_names["U BusBar"] = []
    dict_group_names["U CableLine"] = []
    dict_group_names["I phase"] = []
    dict_group_names["I zero"] = []
    dict_group_names["U_raw BusBar"] = []
    dict_group_names["U_raw CableLine"] = []
    dict_group_names["I_raw phase"] = []
    dict_group_names["I_raw zero"] = []
    
    # TODO: you can think of another type of data to speed up
    for i in range(1, 8):
        dict_group_names["U BusBar"].append(f"{i}Ub_base")
        dict_group_names["U CableLine"].append(f"{i}Uc_base")
        dict_group_names["I phase"].append(f"{i}Ip_base")
        dict_group_names["I zero"].append(f"{i}Iz_base")
        
        # with other types of sensors, while these are Rogovsky belts and capacitive dividers
        # TODO: there are no such names yet and a simple mark is being made in the waveform
        dict_group_names["U_raw BusBar"].append(f"{i}Ub_PS")
        dict_group_names["U_raw CableLine"].append(f"{i}Uc_base")
        dict_group_names["I_raw phase"].append(f"{i}Ip_base")
        dict_group_names["I_raw zero"].append(f"{i}Iz_base")

    dict_group_names["Diff"] = []
    dict_group_names["Diff"].append("dId_base") # FIXME: Why is that? Can a Dif?

    if is_print_to_console:
        print(dict_group_names)
   
    return dict_group_names

def noise_processing(source_dir: str, path_to_csv_file: str, 
                     is_use_qestion_1: bool = False, is_use_qestion_2: bool = False, is_use_qestion_3: bool = False) -> None:
    """
    The function processes the csv file and marks up the data with noise or causing questions

    Args:
        source_dir (str): the directory with the file.
        path_to_csv_file: (str): The address of the csv file.

    Returns:
        None
    """
    csv_group_names = generate_group_signals_from_csv()
    csv_group_base = {}
    
    new_csv_file = []
    
    with open(path_to_csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            if not (row["norm"] == "хз" or row["norm"] == "НЕТ"): # raw - then we mark it up separately
                new_csv_file.append(row)
                continue # That is, only noisy (хз) and with questions (Нет) are processed
            
            csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = -1, -1, -1, -1
            
            is_correct_nominal = True
            # search for denominations
            for key, value in row.items():
                if not value: # missing value
                    continue
                
                if not (value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()) ):
                    continue # it's not a number
                int_value = round(float(value)) # rounding is enough, because sometimes 100.0 is written simply
                
                if key in csv_group_names["U BusBar"]:
                    if csv_group_base["U BusBar"] == -1 or csv_group_base["U BusBar"] == int_value:
                        csv_group_base["U BusBar"] = int_value
                    else:
                        is_correct_nominal = False
                        
                elif key in csv_group_names["U CableLine"]:
                    if csv_group_base["U CableLine"] == -1 or csv_group_base["U CableLine"] == int_value:
                        csv_group_base["U CableLine"] = int_value
                    else:
                        is_correct_nominal = False
                
                elif key in csv_group_names["I phase"]:
                    if csv_group_base["I phase"] == -1 or csv_group_base["I phase"] == int_value:
                        csv_group_base["I phase"] = int_value
                    else:
                        is_correct_nominal = False
                        
                elif key in csv_group_names["I zero"]:
                    if csv_group_base["I zero"] == -1 or csv_group_base["U zero"] == int_value:
                        csv_group_base["I zero"] = int_value
                    else:
                        is_correct_nominal = False
            
            if not is_correct_nominal:
                print(f"Different values were detected, the {row['name']} waveform is not being processed")
                new_csv_file.append(row)
                continue
            
            # WORKING WITH NOISE
            if row["norm"] == "хз":
                # setting the denominations
                # TODO: rewrite into one function
                if csv_group_base["U BusBar"] == -1 and csv_group_base["U CableLine"] == -1 and csv_group_base["I phase"] == -1 and csv_group_base["I zero"] == -1:
                    # there are no denominations, we assume that all denominations are basic
                    csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = 100, 100, 5, 1
                else:
                    if csv_group_base["U BusBar"] == -1:
                        if csv_group_base["U CableLine"] != -1:
                            csv_group_base["U BusBar"] = csv_group_base["U CableLine"]
                        else:
                            csv_group_base["U BusBar"] = 100
                    if csv_group_base["U CableLine"] == -1:
                        csv_group_base["U CableLine"] = csv_group_base["U BusBar"] # she can't be "-1" anymore
                    if csv_group_base["I phase"] == -1:
                        csv_group_base["I phase"] = 5
                    if csv_group_base["I zero"] == -1:
                        csv_group_base["I zero"] = 1

                for key, value in row.items():
                    if not value: # missing value
                        continue
                    
                    if value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()):
                        continue # this number does not need to be changed
                    
                    if not (key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                            key in csv_group_names["I phase"] or key in csv_group_names["I zero"]):
                        continue # skipping the non-interesting
                    
                    if ((key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                         key in csv_group_names["I phase"] or key in csv_group_names["I zero"]) 
                        and row[key] != "шум"):
                        print("Why? Inconsistencies in the noise have been detected")
                    
                    if key in csv_group_names["U BusBar"]:
                        row[key] = str(csv_group_base["U BusBar"])
                    elif key in csv_group_names["U CableLine"]:
                        row[key] = str(csv_group_base["U CableLine"])
                    elif key in csv_group_names["I phase"]:
                        row[key] = str(csv_group_base["I phase"])
                    elif key in csv_group_names["I zero"]:
                        row[key] = str(csv_group_base["I zero"])
                
                row["norm"] = "ДА"
                new_csv_file.append(row)
            
            # WORKING WITH FIRST-ORDER QUESTIONS (?1)
            if row["norm"] == "НЕТ":
                # setting the denominations
                # BUT! it should be borne in mind that this can not be done for all questions.
                # TODO: rewrite into one function
                if csv_group_base["U BusBar"] == -1 and csv_group_base["U CableLine"] == -1 and csv_group_base["I phase"] == -1 and csv_group_base["I zero"] == -1:
                    # we cannot say that this is not a problem in the ranges, so we move on to the next iteration
                    csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = 100, 100, 5, 1
                else:
                    if csv_group_base["U BusBar"] == -1:
                        if csv_group_base["U CableLine"] != -1:
                            csv_group_base["U BusBar"] = csv_group_base["U CableLine"]
                        else:
                            csv_group_base["U BusBar"] = 100
                    if csv_group_base["U CableLine"] == -1:
                        csv_group_base["U CableLine"] = csv_group_base["U BusBar"] # she can't be "-1" anymore
                    if csv_group_base["I phase"] == -1:
                        csv_group_base["I phase"] = 5
                    if csv_group_base["I zero"] == -1:
                        csv_group_base["I zero"] = 1

                coun_undefined = 0 # number of undefined questions
                for key, value in row.items():
                    if not value: # missing value
                        continue
                    
                    if value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()):
                        continue # this number does not need to be changed
                    
                    if not (key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                            key in csv_group_names["I phase"] or key in csv_group_names["I zero"]):
                        continue # skipping the non-interesting
                    
                    # we consider processing appropriate only for these aspects
                    if not (row[key] == "шум" or row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3): 
                        coun_undefined += 1
                        continue
                    
                    if (key in csv_group_names["U BusBar"] and 
                        (row[key] == "шум" or row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3)):
                        row[key] = csv_group_base["U BusBar"]
                    if (key in csv_group_names["U CableLine"] and 
                        (row[key] == "шум" or row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3)):
                        row[key] = csv_group_base["U CableLine"]
                    if (key in csv_group_names["I phase"] and 
                        (row[key] == "шум" or row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3)):
                        row[key] = csv_group_base["I phase"]
                    if (row[key] == "шум" or key in csv_group_names["I zero"] and 
                        (row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3)):
                        row[key] = csv_group_base["I zero"]

                if coun_undefined == 0:
                    row['norm'] = 'ДА'
                new_csv_file.append(row)
    
    new_csv_file_path = ''.join((source_directory, "/new_norm_file.csv"))
    csv_columns = []
    with open(path_to_csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_columns = reader.fieldnames
        
    with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in new_csv_file:
            writer.writerow(data)

def set_base_values(source_directory: str, path_to_csv_file: str, base_values: dict, osc_list: list) -> None:
    """
    The function processes the csv file and sets the nominal value of the selected waveforms
      
    Args:
        source_directory (str): The directory containing the files to update.
        path_to_csv_file: (str): The address of the csv file.
        base_values (dict): dictionary with denominations
        osc_list (list): list of waveform names

    Returns:
        None
    """
    with open(path_to_csv_file, mode='r', encoding='utf-8') as file:
        new_csv_file = []
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            if row['name'] in osc_list:
                for key, value in base_values.items():
                    if key in base_values:
                        row[key] = base_values[key]
            new_csv_file.append(row)

        new_csv_file_path = ''.join((source_directory, "/set_base_file.csv"))
        csv_columns = reader.fieldnames
        
        with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in new_csv_file:
                writer.writerow(data)
            
    
# Example of using the function
# Path to the source directory
source_directory = '/For normalization'
path_to_csv_file = '/For normalization/new_norm_file.csv'

# 108
base_values = {"1Ub_PS":"s", "1Ub_base": "100", "1Uc_PS":"s", "1Uc_base": "100", "1Ip_PS":"s", "1Ip_base": "0.1",
               "2Ub_PS":"s", "2Ub_base": "100", "2Uc_PS":"s", "2Uc_base": "100", "2Ip_PS":"s", "2Ip_base": "0.1"}
# 209
#base_values = {"1Ub_PS":"s", "1Ub_base": "400", "1Uc_PS":"s", "1Uc_base": "400", "1Ip_PS":"s", "1Ip_base": "0.1",
#               "2Ub_PS":"s", "2Ub_base": "400", "2Uc_PS":"s", "2Uc_base": "400", "2Ip_PS":"s", "2Ip_base": "0.1"}
# 331
#base_values = {"1Uc_PS":"s", "1Uc_base": "0.31542", "1Ip_PS":"s", "1Ip_base": "0.45",
#               "2Uc_PS":"s", "2Uc_base": "0.31542", "2Ip_PS":"s", "2Ip_base": "0.45"}
# 363
#base_values = {"1Uc_PS":"s", "1Uc_base": "0.1105", "1Ip_PS":"s", "1Ip_base": "1.41",
#               "2Uc_PS":"s", "2Uc_base": "0.1105", "2Ip_PS":"s", "2Ip_base": "1.41"}
osc_list = []
try:
    path = '/For normalization/_osc_name_dict_v1.json'
    with open(path, 'r', encoding='utf-8') as file:
        hash_table = json.load(file)
        osc_list = hash_table["t00108"] # t00108, t00209, t00331, t00363
except:
    print("Failed to read hash_table from JSON file")

test = generate_analog_signals_name(is_print_to_json=True, path_to_json_file='/For normalization')
test = generate_discrete_signals_name(is_print_to_json=True, path_to_json_file='/For normalization')
# test = generate_group_signals_from_csv(is_print_to_console = True)
# noise_processing(source_directory, path_to_csv_file, is_use_qestion_1 = True, is_use_qestion_2 = True, is_use_qestion_3 = True)
# set_base_values(source_directory, path_to_csv_file, base_values, osc_list)
    