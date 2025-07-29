import os
import sys
import datetime
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from preparing_oscillograms.processing_oscillograms import ProcessingOscillograms

class StatisticalProces():
    """
    Этот класс предназначен для статистической обработки данных.
    """
    def __init__(self):
        pass # ничего не делаем
    
    def date_validation_two(self, start_date: dict, end_date: dict) -> (datetime.datetime, bool, str):
        """
        Эта функция проверяет правильность формата даты, когда у нас есть start_date и end_date.
        
        Аргументы:
            start_date (dict): Данные о событии из файла JSON.
            end_date (dict): Данные о событии из файла JSON.
        
        Возвращает:
            tuple:
             - datetime.datetime: дата и время, если они действительны, в противном случае None
             - bool: True, если действительны, False в противном случае
             - str: сообщение для пользователя
        """
        current_date = datetime.datetime.now()
        answer = {"result": True, "message": ""}
        new_date = {"year": "", "month": "", "day": "", "hour": "", "minute": "", "second": ""}
        # Проверка секунд
        if not (0 <= int(start_date['second'].split('.')[0].strip()) <= 59):
            if (0 <= int(end_date['second'].split('.')[0].strip())   <= 59):
                new_date = end_date
                answer['message'] += "Неверный формат даты: секунда должна быть от 0 до 59. Измените всю дату на end_date. "
            else:
                return (None, False, "Неверный формат даты: секунда должна быть от 0 до 59")
        else: # Все верно
            new_date['second'] = start_date['second'].strip()
        
        # Приведение к общему стандарту
        if '.' not in new_date['second']:
            new_date['second'] += '.0'
        
        # Проверка минут
        if not (0 <= int(start_date['minute']) <= 59):
            if (0 <= int(end_date['minute'])   <= 59):
                new_date = end_date
                answer['message'] += "Неверный формат даты: минута должна быть от 0 до 59. Измените всю дату на end_date. "
            else:
                return (None, False, "Неверный формат даты: минута должна быть от 0 до 59")
        else: # Все верно
            new_date['minute'] = start_date['minute']
        
        # Проверка часов
        if not (0 <= int(start_date['hour']) <= 23):
            if (0 <= int(end_date['hour']) <= 23):
                new_date = end_date
                answer['message'] += "Неверный формат даты: час должен быть от 0 до 23. Измените всю дату на end_date. "
            else:
                return (None, False, "Неверный формат даты: час должен быть от 0 до 23")
        else: # Все верно
            new_date['hour'] = start_date['hour']
        
        # Проверка дня и месяца одновременно (могут быть перепутаны местами)
        # Сначала проверяем на обнуление, было ли такое в данных
        if (int(start_date['month']) == 0 or int(start_date['day']) == 0):
            if ((1 <= int(end_date['month']) <= 12) and 1 <= int(end_date['day']) <= 31):
                new_date = end_date
                answer['message'] += "Неверный формат даты: день и месяц не должны быть нулевыми. Измените всю дату на end_date. "
            else:
                return (None, False, "Неверный формат даты: день должен быть от 1 до 31. Измените всю дату на end_date.")
        # затем проверяем, чтобы день не выходил за пределы допустимого диапазона
        # (и также проверяем месяц, так как они могут быть перепутаны местами)
        elif not ((1 <= int(start_date['day']) <= 31) and (1 <= int(start_date['month']) <= 31)):
            return (None, False, "Неверный формат даты: день должен быть от 1 до 31. Измените всю дату на end_date.")
        elif not (1 <= int(start_date['month']) <= 12):
            # перепутаны друг с другом (считаем за хороший месяц)
            if ((1 <= int(start_date['day']) <= 12) and (1 <= int(start_date['month']) <= 31)):
                new_date["month"] = start_date['day']
                new_date["day"] = start_date['month']
            # неверная дата, заменяем на end_date
            elif ((1 <= int(end_date['month']) <= 12) and 1 <= int(end_date['day']) <= 31):
                new_date["month"] = end_date['month']
                new_date["day"] = end_date['day']
                answer['message'] += "Неверный формат даты: месяц должен быть от 1 до 12. Измените всю дату на end_date. "
            elif not ((1 <= int(end_date['day']) <= 12) and 1 <= int(end_date['month']) <= 31):
                return (None, False, "Неверный формат даты: час должен быть от 0 до 23")
        else: # Все верно
            new_date["month"] = start_date['month']
            new_date["day"] = start_date['day']        
        
        # Проверяем год, который мы считаем действительным с 2001 года, другие требуют уточнения
        if not (1970 <= int(start_date['year']) <= current_date.year):
            if (1970 <= int(end_date['year']) <= current_date.year):
                # скорее всего, дата неверна, и мы ее всю меняем
                new_date = end_date
                answer['message'] += "Неверный формат даты: год должен быть между 1970 и текущим годом. Измените всю дату на end_date. "
            elif (1 <= int(start_date['year']) <= (current_date.year - 2000)):
                new_date["year"] = str(int(start_date['year']) + 2000) # считаем год правильным, просто в другом формате
            elif (70 <= int(end_date['year']) <= 99):
                new_date["year"] = str(int(end_date['year']) + 1900) # исправляем год с прошлого века
            elif (1 <= int(end_date['year']) <= (current_date.year - 2000)):
                end_date['year'] = str(int(end_date['year']) + 2000) # корректируем год
                new_date = end_date
                answer['message'] += "Неверный формат даты: год должен быть между 1970 и текущим годом. Измените всю дату на end_date. "
            elif (70 <= int(end_date['year']) <= 99):
                end_date['year'] = str(int(end_date['year']) + 1900) # корректируем год
                new_date = end_date
                answer['message'] += "Неверный формат даты: год должен быть между 1970 и текущим годом. Измените всю дату на end_date. "
            else:
                return (None, False, "Неверный формат даты: год должен быть между 1970 и текущим годом")
        else: # Все верно
            new_date["year"] = start_date['year']
            
        
        new_date_str = "{}-{}-{} {}:{}:{}".format(
            new_date['year'],
            new_date['month'],
            new_date['day'],
            new_date['hour'],
            new_date['minute'],
            new_date['second'].strip()
        )
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        answer_datetime = datetime.datetime.strptime(new_date_str, date_format)
        
        return (answer_datetime, answer['result'], answer['message'])
    
    def date_validation_one(self, end_date: dict) -> (datetime.datetime, bool, str):
        """
        Эта функция проверяет правильность формата даты, когда у нас есть одна дата (end_date).
        
        Аргументы:
            end_date (dict): Данные о событии из файла JSON.
        
        Возвращает:
            tuple:
             - datetime.datetime: дата и время, если они действительны, в противном случае None
             - bool: True, если действительны, False в противном случае
             - str: сообщение для пользователя
        """
        current_date = datetime.datetime.now()
        answer = {"result": True, "message": "Неверный формат даты"}
        new_date = {"year": "", "month": "", "day": "", "hour": "", "minute": "", "second": ""}
        # Проверка секунд
        if not (0 <= int(end_date['second'].split('.')[0].strip()) <= 59):
            return (None, False, "Неверный формат даты: секунда должна быть от 0 до 59")
        else: # Все верно
            new_date['second'] = end_date['second'].strip()
        
        # Приведение к общему стандарту
        if '.' not in new_date['second']:
            new_date['second'] += '.0'
        
        # Проверка минут
        if not (0 <= int(end_date['minute']) <= 59):
            return (None, False, "Неверный формат даты: минута должна быть от 0 до 59")
        else: # Все верно
            new_date['minute'] = end_date['minute']
        
        # Проверка часов
        if not (0 <= int(end_date['hour']) <= 23):
            return (None, False, "Неверный формат даты: час должен быть от 0 до 23")
        else: # Все верно
            new_date['hour'] = end_date['hour']
        
        # Проверка дня и месяца одновременно (могут быть перепутаны местами)
        # Сначала проверяем на обнуление, было ли такое в данных
        if (int(end_date['month']) == 0 or int(end_date['day']) == 0):
            return (None, False, "Неверный формат даты: день должен быть от 1 до 31. Измените всю дату на end_date.")
        # затем проверяем, чтобы день не выходил за пределы допустимого диапазона
        # (и также проверяем месяц, так как они могут быть перепутаны местами)
        elif not ((1 <= int(end_date['day']) <= 31) and (1 <= int(end_date['month']) <= 31)):
            return (None, False, "Неверный формат даты: день должен быть от 1 до 31. Измените всю дату на end_date.")
        elif not (1 <= int(end_date['month']) <= 12):
           # перепутаны друг с другом (считаем за хороший месяц)
            if ((1 <= int(end_date['day']) <= 12) and (1 <= int(end_date['month']) <= 31)):
                new_date["month"] = end_date['day']
                new_date["day"] = end_date['month']
            # неверная дата, заменяем на end_date
            elif not ((1 <= int(end_date['day']) <= 12) and 1 <= int(end_date['month']) <= 31):
                return (None, False, "Неверный формат даты: час должен быть от 0 до 23")
        else: # Все верно
            new_date["month"] = end_date['month']
            new_date["day"] = end_date['day']        
        
        # Проверяем год, который мы считаем действительным с 2001 года, другие требуют уточнения
        if not (1970 <= int(end_date['year']) <= current_date.year):
            if (1 <= int(end_date['year']) <= (current_date.year - 2000)):
                new_date["year"] = str(int(end_date['year']) + 2000) # считаем год правильным, просто в другом формате
            elif (70 <= int(end_date['year']) <= 99):
                new_date["year"] = str(int(end_date['year']) + 1900) # исправляем год с прошлого века
            else:
                return (None, False, "Неверный формат даты: год должен быть между 1970 и текущим годом")
        else: # Все верно
            new_date["year"] = end_date['year']
            
        
        new_date_str = "{}-{}-{} {}:{}:{}".format(
            new_date['year'],
            new_date['month'],
            new_date['day'],
            new_date['hour'],
            new_date['minute'],
            new_date['second'].strip()
        )
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        answer_datetime = datetime.datetime.strptime(new_date_str, date_format)
        
        return (answer_datetime, answer['result'], answer['message'])

    def frequency_statistics(self, source_dir: str, threshold: float = 0.1, isPrintMessege: bool = False) -> dict:
        """
        Функция группирует файлы по частоте дискретизации и частоте сети.

        Аргументы:
            source_dir (str): Каталог, содержащий файлы для обновления.
            threshold (float): Порог для рассмотрения отклонения частоты от целого числа как ошибки измерения.
            isPrintMessege (bool): Флаг, указывающий, нужно ли печатать сообщение, если частоты не найдены.

        Возвращает:
            frequency_statistics_dict dict: Словарь, содержащий частоты, сгруппированные по частоте дискретизации и сети.
        """
        # TODO: подумать об оптимизации кода, функция близка к "grouping_by_sampling_rate_and_network".
        # также функция "extract_frequencies" взята из этой библиотеки и появляется лишняя ссылка.
        process_osc = ProcessingOscillograms()
        frequency_statistics_dict = {}
        
        print("Подсчитываем общее количество файлов в исходном каталоге...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Общее количество файлов: {total_files}, начинаем обработку...")
        with tqdm(total=total_files, desc="Группировка по частоте дискретизации и сети") as pbar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    pbar.update(1)
                    if file.lower().endswith(".cfg"):
                        file_path = os.path.join(root, file)
                        dat_file = file[:-4] + ".dat"
                        dat_file_path = os.path.join(root, dat_file)
                        is_exist = os.path.exists(dat_file_path) 
                        if is_exist:
                            f_network, f_rate = process_osc.extract_frequencies(file_path=file_path, threshold=threshold, isPrintMessege=isPrintMessege)
                            if f_network and f_rate:
                                if f_network not in frequency_statistics_dict:
                                    frequency_statistics_dict[f_network] = {}
                                if f_rate not in frequency_statistics_dict[f_network]:
                                     frequency_statistics_dict[f_network][f_rate] = 0
                                frequency_statistics_dict[f_network][f_rate] += 1
                            elif f_network:
                                if isPrintMessege: print(f"В файле не найдена частота сети: {file_path}")
                            elif f_rate:
                                if isPrintMessege: print(f"В файле не найдена частота дискретизации: {file_path}")
                            else:
                                if isPrintMessege: print(f"В файле не найдены частоты: {file_path}")
        
        # сортировка по сети, а затем по количеству
        sorted_dict = sorted(frequency_statistics_dict.items(), key=lambda x: (x[0], -sum(x[1].values()), -max(x[1].values())))

        all_count = 0
        for network, rate_dict in sorted_dict:
            print(f"Сеть: {network}")
            sorted_rate_dict = sorted(rate_dict.items(), key=lambda x: x[1], reverse=True)
            for rate, count in sorted_rate_dict:
                print(f"\tЧастота: {rate}, Количество: {count}")
                all_count += count
        print(f"Общее количество: {all_count}")
        return frequency_statistics_dict
