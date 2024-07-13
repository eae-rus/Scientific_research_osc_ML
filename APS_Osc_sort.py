import os
import shutil
import hashlib
import json
import datetime

# создаю exe файл через 
# pip install auto-py-to-exe
# и затем опять же в командной строке
# python -m auto_py_to_exe

# Функция для обхода файловой системы
def Search_and_copy_new_oscillograms(source_dir: str, dest_dir: str, hash_table: dict = {}, 
                                     is_copy_saving_the_folder_structure: bool = True, is_use_hash: bool = True,
                                     is_use_comtrade: bool = True, is_use_brs: bool = True) -> None:
    """
    Копирует файлы .cfg и соответствующие файлы .dat из исходного каталога в целевой каталог, отслеживая скопированные файлы.

    Функция проходит через исходный каталог, идентифицирует файлы. cfg, вычисляет их хэш-значения, 
    копирует их в целевой каталог вместе с файлами .dat, обновляет хэш-таблицу скопированными файлами и сохраняет хэш-таблицы в JSON-файлах.

    Args:
        source_dir (str): путь к исходному каталогу.
        dest_dir (str): путь к целевому каталогу.
        hash_table (dict): хэш-таблица для отслеживания скопированных файлов (по умолчанию-пустой словарь).
        is_copy_saving_the_folder_structure (bool): сохранять ли структуру директорий в целевом каталоге?

    Returns:
        None.
        При этом, создаются новые файлы в целевом каталоге:
        - обновляется файл "_hash_table"
        - создаётся новый файл "_new_hash_table"
    """
    # hash_table - хэш-таблица для отслеживания скопированных файлов
    new_hash_table = {}
    count_new_files = 0
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if is_use_comtrade and file.lower().endswith(".cfg"):  # Если файл имеет расширение .cfg
                file = file[:-4] + ".cfg" # изменяем шрифт типа файла на строчный.
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
                is_exist = os.path.exists(dat_file_path)              
                if is_exist:
                    with open(dat_file_path, 'rb') as f:  # Открываем dat файл для чтения в бинарном режиме
                        file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму dat файла
                        if file_hash not in hash_table or not is_use_hash:
                            dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
                            
                            if is_copy_saving_the_folder_structure:
                                dest_path = os.path.join(dest_dir, dest_subdir, file)  # Формируем путь для копирования cfg файла
                                dat_dest_path = os.path.join(dest_dir, dest_subdir, dat_file)  # Формируем путь для копирования dat файла
                            else:
                                dest_path = os.path.join(dest_dir, file)
                                dat_dest_path = os.path.join(dest_dir, dat_file)
                            
                            if not os.path.exists(dest_path):
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                                shutil.copy2(file_path, dest_path)  # Копируем cfg файл в целевую директорию
                            
                            if not os.path.exists(dat_dest_path):
                                os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                                shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию
                                
                            hash_table[file_hash] = (file, file_path)  # Добавляем хэш-сумму файла в хэш-таблицу
                            new_hash_table[file_hash] = (file, file_path)
                            count_new_files += 1
            
            if is_use_brs and file.lower().endswith(".brs"):  # Если файл имеет расширение .brs (характерно для Бреслера)
                file = file[:-4] + ".brs" # изменяем шрифт типа файла на строчный.
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                with open(file_path, 'rb') as f: # Открываем brs файл для чтения в бинарном режиме
                    file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму dat файла
                    if file_hash not in hash_table or not is_use_hash:
                        dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
                        if is_copy_saving_the_folder_structure:
                            dest_path_BRESELER = os.path.join(dest_dir,'BRESELER', dest_subdir, file)  # Формируем путь для копирования cfg файла
                        else:
                            dest_path_BRESELER = os.path.join(dest_dir,'BRESELER', file)  # Формируем путь для копирования cfg файла
                        if not os.path.exists(dest_path_BRESELER):
                            os.makedirs(os.path.dirname(dest_path_BRESELER), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                            shutil.copy2(file_path, dest_path_BRESELER)  # Копируем файл в целевую директорию
                        
                        if is_use_hash:  
                            hash_table[file_hash] = (file, file_path)  # Добавляем хэш-сумму файла в хэш-таблицу
                            new_hash_table[file_hash] = (file, file_path)
                        count_new_files += 1
    
    print(f"Количество новых скопированных файлов: {count_new_files}") 
    # Сохранение JSON файлов hash_table и new_hash_table                        
    hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')  # Формируем путь для сохранения hash_table
    if is_use_hash: 
        try:
            with open(hash_table_file_path, 'w') as file:
                json.dump(hash_table, file)  # Сохраняем hash_table в JSON файл
        except:
            print("Не удалось сохранить hash_table в JSON файл")
            
        data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        new_hash_table_file_path = os.path.join(dest_dir, f'new_hash_table_{data_now}.json')
        try:
            with open(new_hash_table_file_path, 'w') as file:
                json.dump(new_hash_table, file)
        except:
            print("Не удалось сохранить new_hash_table в JSON файл")


def find_all_osc_for_terminal(dest_dir: str, hash_table: dict, osc_name_dict: dict) -> None:
    """
    Ищет коды осциллограмм в хэш-таблице и добавляет их в новый хэш-список.
      
    Args:
        dest_dir (str): каталог, для сохранения словаря осциллограмм.
        hash_table (dict): хэш-таблица для отслеживания скопированных файлов.
        osc_name_dict (dict): словарь для хранения кодов осциллограмм и их хэш-сумм.
    Returns:
        None
    """
    SCM_NAME = '\\ИПМ'
    new_osc_name_dict = {}
    new_osc_name_arr = [] # для ускорения работы циклов
    for osc_name in osc_name_dict.keys():
        new_osc_name_arr.append(osc_name)
        new_osc_name_dict[osc_name] = '\\'+osc_name[1:]
        
    # Проходим по всем файлам в папке
    for key in hash_table.keys():  # имён по хэш-суммам в разы больше, чем терминалов
        for osc_name in new_osc_name_arr:
            if osc_name in hash_table[key][0]:
                osc_name_dict[osc_name].append(key)
                break # если найдено имя осциллограммы, то прерываем цикл.
            elif ('ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ' in hash_table[key][1] and new_osc_name_dict[osc_name] in hash_table[key][1] and 
                  not SCM_NAME in hash_table[key][1]):
                # сделано раздельно, чтобы отследить, что проверка добавлена не зря.
                osc_name_dict[osc_name].append(key)
                break
                
    
    osc_name_dict_file_path = os.path.join(dest_dir, '_osc_name_dict.json')  # Формируем путь для сохранения osc_name_dict
    try:
        with open(osc_name_dict_file_path, 'w') as file:
            json.dump(osc_name_dict, file)  # Сохраняем hash_table в JSON файл
    except:
        print("Не удалось сохранить osc_name_dict в JSON файл")

# Пример использования функции
# Путь к исходной директории
source_directory = 'D:/DataSet/_ALL_OSC_v2'
# Путь к целевой директории
destination_directory = 'D:/DataSet/depersonalized_ALL_OSC_v2.1'

hash_table = {}
destination_directory_hash_table = destination_directory +  '/_hash_table.json'

# source_directory = input("Введите путь в которой искать: ")
# destination_directory = input("Введите путь в которую сохранять: ")
# destination_directory_hash_table = input("Введите путь к папке с файлом '_hash_table.json': ")
try:
    with open(destination_directory_hash_table, 'r') as file:
        hash_table = json.load(file)
except:
    print("Не удалось прочитать hash_table из JSON файла")


# Search_and_copy_new_oscillograms(source_directory, destination_directory, is_copy_saving_the_folder_structure=False, is_use_brs=False)
osc_name_dict = {}
# osc_name_dict["t00108"], osc_name_dict["t00209"], osc_name_dict["t00331"], osc_name_dict["t00363"] = [], [], [], []
for i in range(1, 500): # пока лишь до 500, потом расширим
    osc_name_dict[f"t{i:05}"] = []
find_all_osc_for_terminal(destination_directory, hash_table, osc_name_dict)