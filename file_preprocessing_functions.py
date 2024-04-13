import os
import shutil
import hashlib
import datetime
import csv

def copy_files_in_one_dir(source_dir, dest_dir):
    """
    Копирует файлы .cfg и соответствующие файлы .dat из дирректории "source_dir" в "dest_dir".

    Args:
        source_dir (str): исходный каталог, содержащий файлы .cfg и .dat.
        dest_dir (str): каталог назначения для копирования файлов.

    Returns:
        None
    """
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.lower().endswith(".cfg"):  # Если файл имеет расширение .cfg
                file = file[:-4] + ".cfg" # изменяем шрифт типа файла на строчный.
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
                is_exist = os.path.exists(dat_file_path) 
                if is_exist:
                    dest_path = os.path.join(dest_dir, file)  # Формируем путь для копирования cfg файла
                    if not os.path.exists(dest_path):
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого файла
                        shutil.copy2(file_path, dest_path)  # Копируем cfg файл в целевую директорию

                    dat_dest_path = os.path.join(dest_dir, dat_file)  # Формируем путь для копирования dat файла
                    if not os.path.exists(dat_dest_path):
                        os.makedirs(os.path.dirname(dat_dest_path), exist_ok=True)  # Создаем все несуществующие директории для целевого dat файла
                        shutil.copy2(dat_file_path, dat_dest_path)  # Копируем dat файл в целевую директорию

def deleting_confidential_information_in_all_files(source_dir): #FIXME: исправить наименование
    """
    Идентифицирует и обрабатывает конфиденциальную информацию в файлах. cfg в указанном исходном каталоге, а так же
    переименовывет файлы с расширением .cfg в .dat.

    Функция ищет файлы .cfg в заданном каталоге, пытается определить их кодировку (utf-8, windows-1251 или ОЕМ 866) 
    и собирает пути к файлам с потенциальной конфиденциальной информацией. 
    Он сохраняет список защищенных файлов и любые ошибки обработки в 'protected_files.txt-файл в исходном каталоге.

    Args:
        source_dir (str): каталог, содержащий файлы .cfg для проверки конфиденциальной информации.

    Returns:
        None
    """
    protected_files = []  # Создаем список для хранения путей к защищенным файлам
    for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
        for file in files:  # Имя каждого файла
            if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                # TODO: для определения кодировки и пересохранения файла в utf-8 необходимо
                # создать отдельную функцию
                try: 
                    deleting_confidential_information_in_on_file(file_path, root, 'utf-8')
                except Exception as e:
                    try:
                        deleting_confidential_information_in_on_file(file_path, root, 'windows-1251')  
                    except Exception as e:
                        try:
                            deleting_confidential_information_in_on_file(file_path, root, 'ОЕМ 866') # ОЕМ - русский язык
                        except Exception as e:
                            protected_files.append(file_path)  # Добавляем защищенный файл в список
                            protected_files.append(f"Произошла ошибка при обработке cfg файла: {e}")
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w') as file:
        file.write('\n'.join(protected_files))  # Сохраняем список защищенных файлов в txt файл в корне папки

def deleting_confidential_information_in_on_file(file_path, root, encoding_name): #FIXME: исправить наименование
    """
    Функция считывает содержимое файла. cfg, выполняет коррекцию кодировки, 
    удаляет локальную информацию, обновляет определенные строки, вычисляет хэш данных 
    и переименовывает файлы .cfg и соответствующие файлы .dat на основе вычисленного хэша.
    
    Args:
        file_path (str): путь к файлу .cfg.
        root (str): путь к корневому каталогу.
        encoding_name (str): формат кодировки, используемый для чтения файла.

    Returns:
        None
    """
    with open(file_path, 'r', encoding=encoding_name) as file:
        lines = file.readlines()
        # Удаляем информацию о локальной информации в cfg файле
        parts = lines[0].split(',')
        if len(parts) >= 2:
            lines[0] = ",," + parts[-1].strip() + "\n"
        # считываем количество сигналов
        signals, analog_signals, digital_signals = lines[1].split(',')
        signals = int(signals)
        new_date = '01/01/0001, 01:01:01.000000\n'
        lines[signals + 5] = new_date
        lines[signals + 6] = new_date

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)
    dat_file_path = file_path[:-4] + ".dat"  # Получаем полный путь к dat файлу 
    with open(dat_file_path, 'rb') as file: # решил делать по dat файлу, так как он точно отличается даже после всех корректировок
        file_hash = hashlib.md5(file.read()).hexdigest()  # Вычисляем хэш-сумму cfg файла
    os.rename(file_path, os.path.join(root, file_hash + '.cfg'))
    os.rename(dat_file_path, os.path.join(root, file_hash + '.dat'))


def date_of_change_replacement(source_dir):
    """
    Функция перебирает все файлы в данном каталоге и устанавливает время изменения каждого файла на текущую дату и время.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Получаем текущую дату и время
    current_date = datetime.datetime.now()

    # Проходим по всем файлам в папке
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        file_stat = os.stat(file_path)
        os.utime(file_path, times=(file_stat.st_atime, current_date.timestamp()))
        
        
def grouping_by_sampling_rate_and_network(source_dir):
    """
    Функция группирует файлы по частоте дискретизации и частоте сети.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                dat_file = file[:-4] + ".dat"  # Формируем имя dat файла на основе имени cfg файла
                dat_file_path = os.path.join(root, dat_file)  # Получаем полный путь к dat файлу
                is_exist = os.path.exists(dat_file_path) 
                if is_exist:
                    f_network, f_rate = extract_frequencies(file_path)

                    if f_network and f_rate:
                        dest_folder = os.path.join(source_dir, 'f_network = ' + str(f_network) + ' and f_rate = ' + str(f_rate))
                        if not os.path.exists(dest_folder):
                            os.makedirs(dest_folder)

                        shutil.move(file_path, os.path.join(dest_folder, file))
                        shutil.move(dat_file_path, os.path.join(dest_folder, dat_file))

def extract_frequencies(file_path):
    """
    Извлекает частоту сети (f_network) и частоту дискретизации (f_rate) из заданного файла ".cfg".

    Args:
        source_dir (str): путь к файлу ".cfg".

    Returns:
        tuple: кортеж, содержащий извлеченную частоту сети и частоту дискретизации.
    """
    f_network, f_rate = 0, 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # FIXME: нет защиты от защищёных и/или ошибочных файлов
            lines = file.readlines()
            if len(lines) >= 2:
                # считываем количество сигналов
                signals, analog_signals, digital_signals = lines[1].split(',')
                signals = int(signals)
                f_network = lines[signals + 2][:-1]
                f_rate, count = lines[signals + 4].split(',')
                f_network, f_rate = int(f_network), int(f_rate)
    except Exception as e:
        f_network, f_rate = 1, 1 #TODO В будущих версиях может потребовать корректировка невалидной частоты
        print(e)

    return f_network, f_rate

def find_all_name_analog_signals(source_dir):
    """
    Функция ищет все название аналоговых сигналов в comtrade файлах и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    analog_signals_name = {}
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # FIXME: нет защиты от защищёных и/или ошибочных файлов
                    lines = file.readlines()
                    if len(lines) >= 2:
                        # считываем количество сигналов
                        signals, analog_signals, digital_signals = lines[1].split(',')
                        count_analog_signals = int(analog_signals[:-1])
                        for i in range(count_analog_signals):
                            analog_signal = lines[2 + i].split(',') # получаем аналоговый сигнал
                            # TODO: добавить единую функцию формирования комбинированного названия сигнала
                            name, phase, unit = analog_signal[1], analog_signal[2], analog_signal[4] # получаем название, фазу и единицу измерения
                            name, phase, unit = name.replace(' ', ''), phase.replace(' ', ''), unit.replace(' ', '') # удаляем пробелы
                            signal_name = name + ' | phase:' + phase + ' | unit:' + unit # создаем комбинированное название сигнала
                            if signal_name not in analog_signals_name:
                                analog_signals_name[signal_name] = 1
                            else:
                                analog_signals_name[signal_name] += 1
    
    sorted_analog_signals_name = {k: v for k, v in sorted(analog_signals_name.items(), key=lambda item: item[1], reverse=True)}      
    # определям путь к csv файлу 
    csv_file = os.path.join(source_dir, 'sorted_analog_signals_name.csv')
    # записываем результаты в csv файл
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', "universal_code", 'Value'])  # Write header
        for key, value in sorted_analog_signals_name.items():
            writer.writerow([key, "-", value])

def find_all_name_digital_signals(source_dir):
    """
    Функция ищет все название дискретных сигналов в comtrade файлах и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    digital_signals_name = {}
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # FIXME: нет защиты от защищёных и/или ошибочных файлов
                    lines = file.readlines()
                    if len(lines) >= 2:
                        # считываем количество сигналов
                        signals, analog_signals, digital_signals = lines[1].split(',')
                        count_analog_signals = int(analog_signals[:-1])
                        count_digital_signals = int(digital_signals[:-2])
                        for i in range(count_digital_signals):
                            digital_signal = lines[2 + count_analog_signals + i].split(',') # получаем аналоговый сигнал
                            if len(digital_signal) == 1: # защита от некорректного количества сигналов
                                break
                            signal_name = digital_signal[1] # получаем название
                            if signal_name not in digital_signals_name:
                                digital_signals_name[signal_name] = 1
                            else:
                                digital_signals_name[signal_name] += 1
    
    sorted_digital_signals_name = {k: v for k, v in sorted(digital_signals_name.items(), key=lambda item: item[1], reverse=True)}      
    # определям путь к csv файлу 
    csv_file = os.path.join(source_dir, 'sorted_digital_signals_name.csv')
    # записываем результаты в csv файл
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', "universal_code", 'Value'])  # Write header
        for key, value in sorted_digital_signals_name.items():
            writer.writerow([key, "-", value])


def rename_analog_signals(source_dir, csv_dir):
    """
    Функция ищет все название аналоговые сигналы, которые присутствуют в базе данных, и переименовывает их к стандартным кодам.
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        csv_dir (str): адрес csv файла.

    Returns:
        None
    """
    # Загрузка CSV файла в словарь
    code_map = {}
    with open(csv_dir, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            key = row['Key']
            universal_code = row['universal_code']
            is_name_determined = universal_code != '-' and universal_code != '?'
            if is_name_determined:
                code_map[key] = universal_code
    
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # FIXME: нет защиты от защищёных и/или ошибочных файлов
                    lines = file.readlines()
                    if len(lines) >= 2:
                        # считываем количество сигналов
                        signals, analog_signals, digital_signals = lines[1].split(',')
                        count_analog_signals = int(analog_signals[:-1])
                        for i in range(count_analog_signals):
                            analog_signal = lines[2 + i].split(',') # получаем аналоговый сигнал
                            # TODO: добавить единую функцию формирования комбинированного названия сигнала
                            name, phase, unit = analog_signal[1], analog_signal[2], analog_signal[4] # получаем название, фазу и единицу измерения
                            name, phase, unit = name.replace(' ', ''), phase.replace(' ', ''), unit.replace(' ', '') # удаляем пробелы
                            signal_name = name + ' | phase:' + phase + ' | unit:' + unit # создаем комбинированное название сигнала
                            if signal_name in code_map:
                                analog_signal[1] = code_map[signal_name]
                                lines[2 + i] = ','.join(analog_signal)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)

def rename_digital_signals(source_dir, csv_dir):
    """
    Функция ищет все название аналоговые сигналы, которые присутствуют в базе данных, и переименовывает их к стандартным кодам.
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        csv_dir (str): адрес csv файла.

    Returns:
        None
    """
    # Загрузка CSV файла в словарь
    code_map = {}
    with open(csv_dir, mode='r', encoding='windows-1251') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            key = row['Key']
            universal_code = row['universal_code']
            is_name_determined = universal_code != '-' and universal_code != '?'
            if is_name_determined:
                code_map[key] = universal_code
    
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # FIXME: нет защиты от защищёных и/или ошибочных файлов
                    lines = file.readlines()
                    if len(lines) >= 2:
                        # считываем количество сигналов
                        signals, analog_signals, digital_signals = lines[1].split(',')
                        count_analog_signals = int(analog_signals[:-1])
                        count_digital_signals = int(digital_signals[:-2])
                        for i in range(count_digital_signals):
                            digital_signal = lines[2 + count_analog_signals + i].split(',') # получаем аналоговый сигнал
                            if len(digital_signal) == 1: # защита от некорректного количества сигналов
                                break
                            signal_name = digital_signal[1] # получаем название
                            if signal_name in code_map:
                                digital_signal[1] = code_map[signal_name]
                                lines[2 + count_analog_signals + i] = ','.join(digital_signal)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)

def delete_empty_line(source_dir):
    """
    удаляет из cfg файла пустые строки
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Проходим по всем файлам в папке
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".cfg"):
                file_path = os.path.join(root, file)
                new_lines = []
                with open(file_path, 'r', encoding='utf-8') as file:
                    # FIXME: нет защиты от защищёных и/или ошибочных файлов
                    lines = file.readlines()
                    for line in lines:
                        if line.strip():
                            new_lines.append(line)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(new_lines)

# Пример использования функции
# Путь к исходной директории
source_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/Удалить'
source_directory = '//192.168.87.199/документы/ОТГРУЖЕННЫЕ ТЕРМИНАЛЫ И ШКАФЫ/Терминалы/БАВР/00203'
source_directory = 'C://Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/Локальное (Алексея)'
# Путь к целевой директории
destination_directory = 'C:/Users/User/Desktop/Буфер (Алексей)/Банк осциллограмм/_до обработки/_ALL_OSC'
# Путь к csv файлу универсальных имён
csv_analog_directory = 'D:/DataSet/depersonalized_ALL_OSC_/universal_analog_signals_name v2.csv'
csv_digital_directory = 'D:/DataSet/depersonalized_ALL_OSC_/universal_digital_signals_name v2.csv'

# FIXME: Оформить описание
# 1) сперва рекомендуется переносить в одну папку
# 2) потом проходиться алгоритмом обезличивания
# 3) а затем удалять дату.

# copy_files_in_one_dir(source_directory, destination_directory)
# deleting_confidential_information_in_files(source_directory)
# date_of_change_replacement(source_directory)
# grouping_by_sampling_rate_and_network(source_directory)
# find_all_name_analog_signals(source_directory)
# find_all_name_digital_signals(source_directory)
# rename_analog_signals(source_directory, csv_analog_directory)
# rename_digital_signals(source_directory, csv_digital_directory)
delete_empty_line(source_directory)
