import os
import shutil
import hashlib
import json

# Функция для обхода файловой системы
def copy_cfg_and_dat_files(source_dir, dest_dir, hash_table = {}):
    # hash_table - хэш-таблицf для отслеживания скопированных файлов
    new_hash_table = {}
    count_new_files = 0
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = file_path[:-4] + ".dat"  # Получаем полный путь к dat файлу
                if os.path.exists(dat_file_path):
                    with open(file_path, 'rb') as f:  # Открываем cfg файл для чтения в бинарном режиме
                        file_hash = hashlib.md5(f.read()).hexdigest()  # Вычисляем хэш-сумму cfg файла
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
    new_hash_table_file_path = os.path.join(dest_dir, '_new_hash_table.json')
    try:
        with open(new_hash_table_file_path, 'w') as file:
            json.dump(new_hash_table, file)
    except:
        print("Не удалось сохранить new_hash_table в JSON файл")


def deleting_confidential_information_in_cfg_files(source_dir):
    protected_files = []  # Создаем список для хранения путей к защищенным файлам
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                try:
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        parts = lines[0].split(',')
                        if len(parts) >= 2:
                            lines[0] = ",," + parts[-1].strip() + "\n"  # Удаляем информацию о локальной информации в cfg файле
                    with open(file_path, 'w') as file:
                        file.writelines(lines)
                except:
                    protected_files.append(file_path)  # Добавляем защищенный файл в список
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
        file.write('\n'.join(protected_files))  # Сохраняем список защищенных файлов в txt файл в корне папки

# Пример использования функции
# Путь к исходной директории
source_directory = 'D:\Программирование\Fork\Scientific_research_osc_ML\Отчёты' 
# Путь к целевой директории
destination_directory = 'D:\Программирование\Fork\Scientific_research_osc_ML\Отчёты_копия'

hash_table = {}
destination_directory_hash_table = destination_directory +  '/_hash_table.json'
try:
    with open(destination_directory_hash_table, 'r') as file:
        hash_table = json.load(file)
except:
    print("Не удалось прочитать hash_table из JSON файла")

copy_cfg_and_dat_files(source_directory, destination_directory, hash_table)
# deleting_confidential_information_in_cfg_files(destination_directory)