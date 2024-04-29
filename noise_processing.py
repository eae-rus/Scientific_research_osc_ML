import os
import csv

def generate_group_signals(is_print_to_console: bool = False) -> dict:
    """
    Функция генерирует группы со всеми именами сигналов
    Вероятно, она пока не нужна.
      
    Args:

    Returns:
        dict - словарь с группами и именами сигналов
    """
    dict_group_names = {}
    # TODO: можно подумать о другом типе данных для ускорения
    for i in range(1, 8):
        if is_print_to_console:
            print(f"------------ секция {i} ------------")
        # истинные данные
        dict_new_group_names = {}
        dict_new_group_names[f"U BusBar {i}"] = []
        dict_new_group_names[f"U CableLine {i}"] = []
        dict_new_group_names[f"I phase {i}"] = []
        dict_new_group_names[f"I zero {i}"] = []

        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: A")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: B")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: C")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: N")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: AB")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: BC")
        dict_new_group_names [f"U BusBar {i}"].append(f"U | BusBar-{i} | phase: CA")
        
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: A")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: B")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: C")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: N")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: AB")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: BC")
        dict_new_group_names[f"U CableLine {i}"].append(f"U | CableLine-{i} | phase: CA")
        
        dict_new_group_names[f"I phase {i}"].append(f"I | Bus-{i} | phase: A")
        dict_new_group_names[f"I phase {i}"].append(f"I | Bus-{i} | phase: B")
        dict_new_group_names[f"I phase {i}"].append(f"I | Bus-{i} | phase: C")
        
        dict_new_group_names[f"I zero {i}"].append(f"I | Bus-{i} | zero")
        
        # с других типов датчиков, пока это чэто пояса Роговского и ёмкостные делители
        dict_new_group_names[f"U_raw BusBar {i}"] = []
        dict_new_group_names[f"U_raw CableLine {i}"] = []
        dict_new_group_names[f"I_raw phase {i}"] = []
        dict_new_group_names[f"I_raw zero {i}"] = []
        
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: A")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: B")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: C")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: N")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: AB")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: BC")
        dict_new_group_names [f"U_raw BusBar {i}"].append(f"U_raw | BusBar-{i} | phase: CA")
        
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: A")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: B")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: C")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: N")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: AB")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: BC")
        dict_new_group_names[f"U_raw CableLine {i}"].append(f"U_raw | CableLine-{i} | phase: CA")
        
        dict_new_group_names[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: A")
        dict_new_group_names[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: B")
        dict_new_group_names[f"I_raw phase {i}"].append(f"I_raw | Bus-{i} | phase: C")
        
        dict_new_group_names[f"I_raw zero {i}"].append(f"I_raw | Bus-{i} | zero")
        
        
        dict_group_names[f"Bus {i}"] = dict_new_group_names
        if is_print_to_console:
            print(dict_new_group_names)
    
    dict_new_group_names = {}
    dict_new_group_names["Diff 1"] = []
    dict_new_group_names["Diff 1"].append("I | dif-1 | phase: A")
    dict_new_group_names["Diff 1"].append("I | dif-1 | phase: B")
    dict_new_group_names["Diff 1"].append("I | dif-1 | phase: C")
    dict_new_group_names["Diff 1"].append("I | braking-1 | phase: A")
    dict_new_group_names["Diff 1"].append("I | braking-1 | phase: B")
    dict_new_group_names["Diff 1"].append("I | braking-1 | phase: C")
    if is_print_to_console:
        print("------------ Diff 1 ------------")
        print(dict_new_group_names)
    
    dict_group_names["Diff current"] = dict_new_group_names
    
    return dict_group_names

def generate_group_signals_from_csv(is_print_to_console: bool = False) -> dict:
    """
    Функция генерирует группы со всеми именами сигналов
    Вероятно, она пока не нужна.
      
    Args:

    Returns:
        dict - словарь с группами и именами сигнало
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
    
    # TODO: можно подумать о другом типе данных для ускорения
    for i in range(1, 8):
        # истинные данные
        dict_group_names["U BusBar"].append(f"{i}Ub_base")
        dict_group_names["U CableLine"].append(f"{i}Uc_base")
        dict_group_names["I phase"].append(f"{i}Ip_base")
        dict_group_names["I zero"].append(f"{i}Iz_base")
        
        # с других типов датчиков, пока это чэто пояса Роговского и ёмкостные делители
        # TODO: пока отсутствуют такие имена и делается прост пометка в осциллограмме
        dict_group_names["U_raw BusBar"].append(f"{i}Ub_PS")
        dict_group_names["U_raw CableLine"].append(f"{i}Uc_base")
        dict_group_names["I_raw phase"].append(f"{i}Ip_base")
        dict_group_names["I_raw zero"].append(f"{i}Iz_base")

    dict_group_names["Diff"] = []
    dict_group_names["Diff"].append("dId_base") # FIXME: почему так? Dif может?

    if is_print_to_console:
        print(dict_group_names)
   
    return dict_group_names

def noise_processing(source_dir: str, path_to_csv_file: str, 
                     is_use_qestion_1: bool = False, is_use_qestion_2: bool = False, is_use_qestion_3: bool = False) -> None:
    """
    Функция обрабатывает csv файл и размечается данные с шумом или вызывающие вопросы
      
    Args:
        source_dir (str): каталог с файлом.
        path_to_csv_file: (str): адрес csv файла.

    Returns:
        None
    """
    csv_group_names = generate_group_signals_from_csv()
    csv_group_base = {}
    
    new_csv_file = []
    
    with open(path_to_csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        #iterration = 0 # FIXME: для первых тестов
        for row in reader:
            #if iterration >= 10:
            #    break
            #iterration += 1
            if not (row["norm"] == "хз" or row["norm"] == "НЕТ"): # raw - потом разметим отдельно
                new_csv_file.append(row)
                continue # То есть обрабатываются только шумные (хз) и с вопросами (НЕТ)
            
            csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = -1, -1, -1, -1
            
            is_correct_nominal = True
            # поиск номиналов
            for key, value in row.items():
                if not value: # отсутствует значение
                    continue
                
                if not (value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()) ):
                    continue # это не число
                int_value = round(float(value)) # достаточно округление, ибо иногда прост пишется 100.0
                
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
                print(f"Обнаружены разные номиналы, осцилограмма {row['name']} не обрабатывается")
                new_csv_file.append(row)
                continue
            
            # РАБОТА С ШУМОМ
            if row["norm"] == "хз":
                # задание номиналов
                # TODO: переписать в одну функцию
                if csv_group_base["U BusBar"] == -1 and csv_group_base["U CableLine"] == -1 and csv_group_base["I phase"] == -1 and csv_group_base["I zero"] == -1:
                    # нет номиналов, принимаем, что все номиналы базовые
                    csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = 100, 100, 5, 1
                else:
                    if csv_group_base["U BusBar"] == -1:
                        if csv_group_base["U CableLine"] != -1:
                            csv_group_base["U BusBar"] = csv_group_base["U CableLine"]
                        else:
                            csv_group_base["U BusBar"] = 100
                    if csv_group_base["U CableLine"] == -1:
                        csv_group_base["U CableLine"] = csv_group_base["U BusBar"] # она уже не может быть "-1"
                    if csv_group_base["I phase"] == -1:
                        csv_group_base["I phase"] = 5
                    if csv_group_base["I zero"] == -1:
                        csv_group_base["I zero"] = 1

                for key, value in row.items():
                    if not value: # отсутствует значение
                        continue
                    
                    if value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()):
                        continue # это число и менять не надо
                    
                    if not (key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                            key in csv_group_names["I phase"] or key in csv_group_names["I zero"]):
                        continue # пропуск не интересующего
                    
                    if ((key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                         key in csv_group_names["I phase"] or key in csv_group_names["I zero"]) 
                        and row[key] != "шум"):
                        print("Why? Обнаружены несотыковки в шуме")
                    
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
            
            # РАБОТА С ВОПРОСАМИ ПЕРОВОГО ПОРЯДКА (?1)
            if row["norm"] == "НЕТ":
                # задание номиналов
                # НО! стоит иметь ввиду, что это можно сделать не под все вопросы.
                # TODO: переписать в одну функцию
                if csv_group_base["U BusBar"] == -1 and csv_group_base["U CableLine"] == -1 and csv_group_base["I phase"] == -1 and csv_group_base["I zero"] == -1:
                    # не можем утверждать, чо это не в диапазонах проблема, поэтому переходим на следующую итерацию
                    csv_group_base["U BusBar"], csv_group_base["U CableLine"], csv_group_base["I phase"], csv_group_base["I zero"] = 100, 100, 5, 1
                else:
                    if csv_group_base["U BusBar"] == -1:
                        if csv_group_base["U CableLine"] != -1:
                            csv_group_base["U BusBar"] = csv_group_base["U CableLine"]
                        else:
                            csv_group_base["U BusBar"] = 100
                    if csv_group_base["U CableLine"] == -1:
                        csv_group_base["U CableLine"] = csv_group_base["U BusBar"] # она уже не может быть "-1"
                    if csv_group_base["I phase"] == -1:
                        csv_group_base["I phase"] = 5
                    if csv_group_base["I zero"] == -1:
                        csv_group_base["I zero"] = 1

                coun_undefined = 0 # количество неопределенных вопросов
                for key, value in row.items():
                    if not value: # отсутствует значение
                        continue
                    
                    if value.isdigit() or (value.count('.') == 1 and value.replace('.', '', 1).isdigit()):
                        continue # это число и менять не надо
                    
                    if not (key in csv_group_names["U BusBar"] or key in csv_group_names["U CableLine"] or
                            key in csv_group_names["I phase"] or key in csv_group_names["I zero"]):
                        continue # пропуск не интересующего
                    
                    if not (row[key] == "шум" or row[key] == "?1" and is_use_qestion_1 or row[key] == "?2" and is_use_qestion_2 or row[key] == "?3" and is_use_qestion_3): # обработку считаем уместной только для этих аспектов
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
    
    
# Пример использования функции
# Путь к исходной директории
source_directory = 'D:/DataSet/Для нормировки'

path_to_csv_file = 'D:/DataSet/Для нормировки/norm_1600_v1.csv'

# test = generate_group_signals(is_print_to_console = True)
# test = generate_group_signals_from_csv(is_print_to_console = True)
noise_processing(source_directory, path_to_csv_file, is_use_qestion_1 = True, is_use_qestion_2 = True, is_use_qestion_3 = True)
    