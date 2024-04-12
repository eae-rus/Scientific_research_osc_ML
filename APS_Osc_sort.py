import os
import shutil
import hashlib
import json
import datetime

# Функция для обхода файловой системы
def Search_and_copy_new_oscillograms(source_dir, dest_dir, hash_table = {}):
    """
    Копирует файлы .cfg и соответствующие файлы .dat из исходного каталога в целевой каталог, отслеживая скопированные файлы.

    Функция проходит через исходный каталог, идентифицирует файлы. cfg, вычисляет их хэш-значения, 
    копирует их в целевой каталог вместе с файлами .dat, обновляет хэш-таблицу скопированными файлами и сохраняет хэш-таблицы в JSON-файлах.

    Args:
        source_dir (str): путь к исходному каталогу.
        dest_dir (str): путь к целевому каталогу.
        hash_table (dict): хэш-таблица для отслеживания скопированных файлов (по умолчанию-пустой словарь).

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
            if file.lower().endswith(".cfg"):  # Если файл имеет расширение .cfg
                file = file[:-4] + ".cfg" # изменяем шрифт типа файла на строчный.
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
                is_exist = os.path.exists(dat_file_path)              
                if is_exist:
                    with open(dat_file_path, 'rb') as f:  # Открываем dat файл для чтения в бинарном режиме
                        file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму dat файла
                        if file_hash not in hash_table:
                            dest_subdir = os.path.relpath(root, source_dir)  # Получаем относительный путь от исходной директории до текущей директории
                            dest_path = os.path.join(dest_dir, dest_subdir, file)  # Формируем путь для копирования cfg файла
                            if not os.path.exists(dest_path):
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                                shutil.copy2(file_path, dest_path)  # Копируем cfg файл в целевую директорию
                            
                            dat_dest_path = os.path.join(dest_dir, dest_subdir, dat_file)  # Формируем путь для копирования dat файла
                            if not os.path.exists(dat_dest_path):
                                os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                                shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию
                                
                            hash_table[file_hash] = (file, file_path)  # Добавляем хэш-сумму файла в хэш-таблицу
                            new_hash_table[file_hash] = (file, file_path)
                            count_new_files += 1
    
    # Сохранение JSON файлов hash_table и new_hash_table                        
    hash_table_file_path = os.path.join(dest_dir, '_hash_table.json')  # Формируем путь для сохранения hash_table
    try:
        with open(hash_table_file_path, 'w') as file:
            json.dump(hash_table, file)  # Сохраняем hash_table в JSON файл
    except:
        print("Не удалось сохранить hash_table в JSON файл")
    
    print(f"Количество новых скопированных файлов: {count_new_files}") 
    # TODO: Проверить корректность сохранения new_hash_table.
    data_now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    new_hash_table_file_path = os.path.join(dest_dir, '_new_hash_table_', data_now, '.json')
    try:
        with open(new_hash_table_file_path, 'w') as file:
            json.dump(new_hash_table, file)
    except:
        print("Не удалось сохранить new_hash_table в JSON файл")

# Пример использования функции
# Путь к исходной директории
source_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/Удалить'
source_directory = '//192.168.87.199/документы/ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ/Терминалы/БАВР/00203'
source_directory = 'C://Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/Локальное (Алексея)'
# Путь к целевой директории
destination_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/_ALL_OSC'

hash_table = {}
destination_directory_hash_table = destination_directory +  '/_hash_table.json'
try:
    with open(destination_directory_hash_table, 'r') as file:
        hash_table = json.load(file)
except:
    print("Не удалось прочитать hash_table из JSON файла")

Search_and_copy_new_oscillograms(source_directory, destination_directory, hash_table)