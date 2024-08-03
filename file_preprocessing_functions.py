import os
import shutil
import hashlib
import datetime
import csv
import json
from tqdm import tqdm

def deleting_confidential_information_in_all_files(source_dir: str) -> None:
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
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Deleting confidential information") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
            for file in files:  # Имя каждого файла
                pbar.update(1)  # Обновление прогресс-бара на один файл
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

def deleting_confidential_information_in_on_file(file_path: str, root: str, encoding_name: str) -> None:
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


def date_of_change_replacement(source_dir: str) -> None:
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
        
        
def grouping_by_sampling_rate_and_network(source_dir: str) -> None:
    """
    Функция группирует файлы по частоте дискретизации и частоте сети.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Проходим по всем файлам в папке
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Grouping by sampling rate and network") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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

def extract_frequencies(file_path: str) -> tuple:
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

def find_all_name_analog_signals(source_dir: str) -> None:
    """
    Функция ищет все название аналоговых сигналов в comtrade файлах и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    analog_signals_name = {}
    # Проходим по всем файлам в папке
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Find all name analog signals") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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
                                name, phase = analog_signal[1], analog_signal[2] # получаем название, фазу и единицу измерения
                                name, phase = name.replace(' ', ''), phase.replace(' ', '') # удаляем пробелы
                                signal_name = name + ' | phase:' + phase # создаем комбинированное название сигнала
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

def find_all_name_digital_signals(source_dir: str) -> None:
    """
    Функция ищет все название дискретных сигналов в comtrade файлах и сортирует их по частоте использования.

    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    digital_signals_name = {}
    # Проходим по всем файлам в папке
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Find all name digital signals") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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


def rename_analog_signals(source_dir: str, csv_dir: str) -> None:
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
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Rename analog signals") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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
                                name, phase = analog_signal[1], analog_signal[2] # получаем название, фазу и единицу измерения
                                name, phase = name.replace(' ', ''), phase.replace(' ', ''),  # удаляем пробелы
                                signal_name = name + ' | phase:' + phase # создаем комбинированное название сигнала
                                if signal_name in code_map:
                                    analog_signal[1] = code_map[signal_name]
                                    lines[2 + i] = ','.join(analog_signal)

                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.writelines(lines)

def rename_digital_signals(source_dir: str, csv_dir: str) -> None:
    """
    Функция ищет все название дискретные сигналы, которые присутствуют в базе данных, и переименовывает их к стандартным кодам.
      
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
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Rename digital signals") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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
                    
def rename_one_signals(source_dir: str, old_name: str, new_name: str) -> None:
    """
    Функция ищет все название сигнала с одним именем и меняет на новое.
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.
        old_name (str): старое название сигнала.
        new_name (str): новое название сигнала.

    Returns:
        None
    """
    # Проходим по всем файлам в папке
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Rename signal") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
                if file.lower().endswith(".cfg"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        # FIXME: нет защиты от защищёных и/или ошибочных файлов
                        lines = file.readlines()
                        if len(lines) >= 2:
                            # считываем количество сигналов
                            for i in range(len(lines)):
                                if old_name in lines[i]:
                                    lines[i] = lines[i].replace(old_name, new_name)

                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.writelines(lines)

def delete_empty_line(source_dir: str) -> None:
    """
    удаляет из cfg файла пустые строки
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    # Проходим по всем файлам в папке
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Delete empty line") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                pbar.update(1)  # Обновление прогресс-бара на один файл
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

def Combining_databases_of_unique_codes(old_csv_file_path: str, new_csv_file_path: str, merged_csv_file_path: str,
                                        deelimed_old_csv_file: str = ';', deelimed_new_csv_file: str = ';', is_merge_files: bool = True) -> None:
    """
    функция объединяет csv файлы с уникальными кодами сигналов
      
    Args:
        old_csv_file_path (str): адрес csv файла с уникальными кодами сигналов.
        new_csv_file_path (str): адрес csv файла с уникальными кодами сигналов.

    Returns:
        None
    """
    # Открытие старого CSV файла и чтение его в словарь
    old_code_map = {}
    with open(old_csv_file_path, mode='r', encoding='1251') as file:
        reader = csv.DictReader(file, delimiter=deelimed_old_csv_file)
        for row in reader:
            key = row['Key']
            universal_code = row['universal_code']
            value = row['Value']
            is_name_determined = universal_code != '-' and universal_code != '?'
            if is_name_determined or is_merge_files:
                old_code_map[key] = (universal_code, value)

    # Открытие новоего CSV файла и чтение его в словарь
    new_code_map = {}
    with open(new_csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=deelimed_new_csv_file)
        for row in reader:
            key = row['Key']
            universal_code = row['universal_code']
            value = row['Value']
            new_code_map[key] = (universal_code, value)
            

    # Если is_merge_files имеет значение True, объедините массив с суммированными значениями в поле "значение"
    merged_code_map = dict()
    if is_merge_files:
        merged_code_map = old_code_map.copy()
        
    for key, value in new_code_map.items():
        if is_merge_files:
            if key not in merged_code_map:
                merged_code_map[key] = value
            else:
                old_value = merged_code_map[key][1]
                new_value = value[1]
                merged_value = int(old_value) + int(new_value)
                merged_code_map[key] = (merged_code_map[key][0], str(merged_value))
        else:
            if key in old_code_map:
                merged_code_map[key] = (old_code_map[key][0] , value[1])
            else:
                merged_code_map[key] = value
    
    sorted_code_map = dict(sorted(merged_code_map.items(), key=lambda item: int(item[1][1]), reverse=True))
     # Создание нового CSV файла для хранения скопированных значений
    with open(merged_csv_file_path, mode='w', encoding='utf-8', newline='') as new_file:
        writer = csv.writer(new_file, delimiter=deelimed_new_csv_file)
        writer.writerow(['Key', 'universal_code', 'Value'])
        for key, (universal_code, value) in sorted_code_map.items():
            writer.writerow([key, universal_code, value])

def combining_json_hash_table(source_dir: str) -> None:
    """
    !!ВНИМАНИЕ!!
    Стоит использовать только для объединения файлов "hash_table"
    функция объединяет json файлы с уникальными именами осциллограмм сигналов
      
    Args:
        source_dir (str): каталог, содержащий файлы для обновления.

    Returns:
        None
    """
    combine_hash_table = {}
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Combining json hash table") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
            for file in files:  # Имя каждого файла
                pbar.update(1)  # Обновление прогресс-бара на один файл
                if file.lower().endswith(".json"):  # Если файл имеет расширение .json
                    try:
                        path = os.path.join(root, file)
                        with open(path, 'r', encoding='utf-8') as file:
                            hash_table = json.load(file)
                            for key, value in hash_table.items():
                                if key not in combine_hash_table:
                                    combine_hash_table[key] = value
                    except:
                        print("Не удалось прочитать hash_table из JSON файла")
                        
    try:
        combine_hash_table_file_path = os.path.join(source_dir, 'combine_hash_table.json')
        with open(combine_hash_table_file_path, 'w') as file:
            json.dump(combine_hash_table, file)
    except:
        print("Не удалось сохранить new_hash_table в JSON файл")


def Research_coorect_encoding_in_cfg(source_dir: str, act_function = None) -> None:
    """
    Функция ищет файлы .cfg в заданном каталоге, пытается определить их кодировку (utf-8, windows-1251 или ОЕМ 866) 
    и собирает пути к файлам. 
    Он сохраняет список защищенных файлов и любые ошибки обработки в 'protected_files.txt-файл в исходном каталоге.
    
    С требуемыми файлами применяется функция заданная в входных данныхх

    Args:
        source_dir (str): каталог, содержащий файлы .cfg для проверки конфиденциальной информации.
        act_function (function): функция, которая будет применена к файлам.

    Returns:
        None
    """
    protected_files = []  # Создаем список для хранения путей к защищенным файлам
    print("Считаем общее количество файлов в исходной директории...")
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])  # Подсчет общего количества файлов
    print(f"Общее количество файлов: {total_files}, запускаем обработку...")
    with tqdm(total=total_files, desc="Finding the correct encoding and determining the date") as pbar:  # Инициализация прогресс-бара
        for root, dirs, files in os.walk(source_dir):  # Итерируемся по всем файлам и директориям в исходной директории
            for file in files:  # Имя каждого файла
                pbar.update(1)  # Обновление прогресс-бара на один файл
                if file.endswith(".cfg"):  # Если файл имеет расширение .cfg
                    file_path = os.path.join(root, file)  # Получаем полный путь к cfg файлу
                    # TODO: для определения кодировки и пересохранения файла в utf-8 необходимо
                    # создать отдельную функцию
                    try: 
                        act_function(file_path, root, 'utf-8')
                    except Exception as e:
                        try:
                            act_function(file_path, root, 'windows-1251')  
                        except Exception as e:
                            try:
                                act_function(file_path, root, 'ОЕМ 866') # ОЕМ - русский язык
                            except Exception as e:
                                protected_files.append(file_path)  # Добавляем защищенный файл в список
                                protected_files.append(f"Произошла ошибка при обработке cfg файла: {e}")
    with open(os.path.join(source_dir, 'protected_files.txt'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(protected_files))  # Сохраняем список защищенных файлов в txt файл в корне папки

def detect_date(file_path: str, root: str, encoding_name: str, dict_all_dates = {}) -> None:
    """
    Функция определяет дату события в cfg файле.
    
    Args:
        file_path (str): путь к файлу .cfg.
        root (str): путь к корневому каталогу.
        encoding_name (str): формат кодировки, используемый для чтения файла.

    Returns:
        None
    """
    with open(file_path, 'r', encoding=encoding_name) as file:
        lines = file.readlines()
        # считываем количество сигналов
        signals, analog_signals, digital_signals = lines[1].split(',')
        signals = int(signals)
        
        # Изменяем только год в дате события
        new_date = lines[signals + 5].split(',')
        date_parts = new_date[0].split('/')
        time_parts = new_date[1].split(':')
        
        dict_date = {
            'year': date_parts[2],
            'month': date_parts[1],
            'day': date_parts[0],
            'hour': time_parts[0],
            'minute': time_parts[1],
            'second': time_parts[2]
        }
        dat_file_path = file_path[:-4] + ".dat"  # Получаем полный путь к dat файлу 
        with open(dat_file_path, 'rb') as file: # решил делать по dat файлу, так как он точно отличается даже после всех корректировок
            file_hash = hashlib.md5(file.read()).hexdigest()  # Вычисляем хэш-сумму cfg файла

        dict_all_dates[file_hash] = dict_date

# Пример использования функции
# Путь к исходной директории
source_directory = 'D:/DataSet/_ALL_OSC_v2'
source_directory = 'XXXXX'
# Путь к целевой директории
destination_directory = 'D:/DataSet/depersonalized_ALL_OSC_v2'
# Путь к csv файлу универсальных имён
csv_analog_directory = 'D:/DataSet/depersonalized_ALL_OSC_/universal_analog_signals_name v2.csv'
csv_digital_directory = 'D:/DataSet/depersonalized_ALL_OSC_/universal_digital_signals_name v2.csv'
# Пути комбинации файлов csv
old_csv_file_path = 'D:/DataSet/depersonalized_ALL_OSC_v2/universal_analog_signals_name v2.csv'
new_csv_file_path = 'D:/DataSet/depersonalized_ALL_OSC_v2/sorted_analog_signals_name.csv'
merged_csv_file_path = 'D:/DataSet/depersonalized_ALL_OSC_v2/merged.csv'

# FIXME: Оформить описание
# 1) сперва рекомендуется переносить в одну папку
# 2) потом проходиться алгоритмом обезличивания
# 3) а затем удалять дату.

# deleting_confidential_information_in_all_files(destination_directory)
# date_of_change_replacement(destination_directory)
# grouping_by_sampling_rate_and_network(destination_directory)
# find_all_name_analog_signals(destination_directory)
# find_all_name_digital_signals(destination_directory)
# rename_analog_signals(destination_directory, csv_analog_directory)
# rename_digital_signals(destination_directory, csv_digital_directory)
# rename_one_signals(destination_directory, 'I | Bus-3 | phase: N', 'U | BusBar-3 | phase: N')
# delete_empty_line(destination_directory)
Combining_databases_of_unique_codes(old_csv_file_path, new_csv_file_path, merged_csv_file_path,
                                    deelimed_old_csv_file=';',deelimed_new_csv_file=',', is_merge_files=True)
# combining_json_hash_table(destination_directory)
dict_all_dates = {}
Research_coorect_encoding_in_cfg(destination_directory, act_function=lambda file_path, root, encoding_name: detect_date(file_path, root, encoding_name, dict_all_dates))
dict_all_dates_path = destination_directory + '/dict_all_dates.json'
with open(dict_all_dates_path, 'w', encoding='utf-8') as file:
    file.write(json.dumps(dict_all_dates))