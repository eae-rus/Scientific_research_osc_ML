import os
import shutil
import hashlib
import json

def deleting_confidential_information_in_cfg_files(source_dir):
    protected_files = []  # Создаем список для хранения путей к защищенным файлам
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        # Удаляем информацию о локальной информации в cfg файле
                        parts = lines[0].split(',')
                        if len(parts) >= 2:
                            lines[0] = ",," + parts[-1].strip() + "\n"
                        # считываем количество сигналов
                        signals, analog_signals, digital_signals = lines[1].split(',')
                        signals = int(signals)
                        # считываем параметры осциллограммы
                        frequency = int(lines[signals + 2])
                        frequency_discretization, count_point= lines[signals + 4].split(',')
                        frequency_discretization, count_point = int(frequency_discretization), int(count_point)
                        # перезапись даты
                        new_date = '01/01/0001, 01:01:01.000000\n'
                        lines[signals + 5] = new_date
                        lines[signals + 6] = new_date
                                                
                    with open(file_path, 'w') as file:
                        file.writelines(lines)
                    
                    dat_file_path = file_path[:-4] + ".dat"  # Получаем полный путь к dat файлу 
                    with open(dat_file_path, 'rb') as file: # решил делать по dat файлу, так как он точно отличается даже после всех корректировок
                        file_hash = hashlib.md5(file.read()).hexdigest()  # Вычисляем хэш-сумму cfg файла
                    os.rename(file_path, os.path.join(root, file_hash + '.cfg'))
                    os.rename(dat_file_path, os.path.join(root, file_hash + '.dat'))
                        
                except Exception as e:
                    protected_files.append(file_path)  # Добавляем защищенный файл в список
                    protected_files.append(f"Произошла ошибка при обработке cfg файла: {e}")
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
        file.write('\n'.join(protected_files))  # Сохраняем список защищенных файлов в txt файл в корне папки

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

deleting_confidential_information_in_cfg_files(destination_directory)