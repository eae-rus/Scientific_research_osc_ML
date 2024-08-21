import os
import sys
import datetime
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from preparing_oscillograms.processing_oscillograms import ProcessingOscillograms

class StatisticalProces():
    """
    This class is for statistical data processing.
    """
    def __init__(self):
        pass # do nothing
    
    def date_validation_two(self, start_date: dict, end_date: dict) -> (datetime.datetime, bool, str):
        """
        This function checks the validity of the date format, when we have start_date and end_date.
        
        args:
            start_date (dict): The event data from the JSON file.
            end_date (dict): The event data from the JSON file.
        
        Returns:
            tuple:
             - datetime.datetime: date and time if valid, else None
             - bool: True if valid, False otherwise
             - str: message for the user
        """
        current_date = datetime.datetime.now()
        answer = {"result": True, "message": ""}
        new_date = {"year": "", "month": "", "day": "", "hour": "", "minute": "", "second": ""}
        # Checking the seconds
        if not (0 <= int(start_date['second'].split('.')[0].strip()) <= 59):
            if (0 <= int(end_date['second'].split('.')[0].strip())   <= 59):
                new_date = end_date
                answer['message'] += "Invalid date format: second should be between 0 and 59. Change the all date on end_date. "
            else:
                return (None, False, "Invalid date format: second should be between 0 and 59")
        else: # That's right
            new_date['second'] = start_date['second'].strip()
        
        # Bringing to a common standard
        if '.' not in new_date['second']:
            new_date['second'] += '.0'
        
        # Checking the minutes
        if not (0 <= int(start_date['minute']) <= 59):
            if (0 <= int(end_date['minute'])   <= 59):
                new_date = end_date
                answer['message'] += "Invalid date format: minute should be between 0 and 59. Change the all date on end_date. "
            else:
                return (None, False, "Invalid date format: minute should be between 0 and 59")
        else: # That's right
            new_date['minute'] = start_date['minute']
        
        # Checking the clock
        if not (0 <= int(start_date['hour']) <= 23):
            if (0 <= int(end_date['hour']) <= 23):
                new_date = end_date
                answer['message'] += "Invalid date format: hour should be between 0 and 23. Change the all date on end_date. "
            else:
                return (None, False, "Invalid date format: hour should be between 0 and 23")
        else: # That's right
            new_date['hour'] = start_date['hour']
        
        # Checking the day and month at the same time (may be mixed up in places)
        # Checking for zeroing first, has this happened in the data
        if (int(start_date['month']) == 0 or int(start_date['day']) == 0):
            if ((1 <= int(end_date['month']) <= 12) and 1 <= int(end_date['day']) <= 31):
                new_date = end_date
                answer['message'] += "Invalid date format: day and month should not be zero. Change the all date on end_date. "
            else:
                return (None, False, "Invalid date format: day should be between 1 and 31. Change the all date on end_date.")
        # then we check that the day does not exceed the limits of the acceptable range 
        # (and we also check the month, since they may be mixed up in places)
        elif not ((1 <= int(start_date['day']) <= 31) and (1 <= int(start_date['month']) <= 31)):
            return (None, False, "Invalid date format: day should be between 1 and 31. Change the all date on end_date.")        
        elif not (1 <= int(start_date['month']) <= 12):
            # confused with each other (we count for a good month)
            if ((1 <= int(start_date['day']) <= 12) and (1 <= int(start_date['month']) <= 31)):
                new_date["month"] = start_date['day']
                new_date["day"] = start_date['month']
            # incorrect date, replace with end_date
            elif ((1 <= int(end_date['month']) <= 12) and 1 <= int(end_date['day']) <= 31):
                new_date["month"] = end_date['month']
                new_date["day"] = end_date['day']
                answer['message'] += "Invalid date format: month should be between 1 and 12. Change the all date on end_date. "
            elif not ((1 <= int(end_date['day']) <= 12) and 1 <= int(end_date['month']) <= 31):
                return (None, False, "Invalid date format: hour should be between 0 and 23")
        else: # That's right
            new_date["month"] = start_date['month']
            new_date["day"] = start_date['day']        
        
        # Checking the year we consider valid since 2001, others require clarification
        if not (1970 <= int(start_date['year']) <= current_date.year):
            if (1970 <= int(end_date['year']) <= current_date.year):
                # most likely, the date is incorrect and we are changing it all
                new_date = end_date
                answer['message'] += "Invalid date format: year should be between 1970 and current year. Change the all date on end_date. "
            elif (1 <= int(start_date['year']) <= (current_date.year - 2000)):
                new_date["year"] = str(int(start_date['year']) + 2000) # we consider the year to be correct, simple in a different format
            elif (70 <= int(end_date['year']) <= 99):
                new_date["year"] = str(int(end_date['year']) + 1900) # correcting the year from the last century
            elif (1 <= int(end_date['year']) <= (current_date.year - 2000)):
                end_date['year'] = str(int(end_date['year']) + 2000) # adjusting the year
                new_date = end_date
                answer['message'] += "Invalid date format: year should be between 1970 and current year. Change the all date on end_date. "
            elif (70 <= int(end_date['year']) <= 99):
                end_date['year'] = str(int(end_date['year']) + 1900) # adjusting the year
                new_date = end_date
                answer['message'] += "Invalid date format: year should be between 1970 and current year. Change the all date on end_date. "
            else:
                return (None, False, "Invalid date format: year should be between 1970 and current year")
        else: # That's right
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
        This function checks the validity of the date format, when we have one date (end_date).
        
        args:
            end_date (dict): The event data from the JSON file.
        
        Returns:
            tuple:
             - datetime.datetime: date and time if valid, else None
             - bool: True if valid, False otherwise
             - str: message for the user
        """
        current_date = datetime.datetime.now()
        answer = {"result": True, "message": "Valid date format"}
        new_date = {"year": "", "month": "", "day": "", "hour": "", "minute": "", "second": ""}
        # Checking the seconds
        if not (0 <= int(end_date['second'].split('.')[0].strip()) <= 59):
            return (None, False, "Invalid date format: second should be between 0 and 59")
        else: # That's right
            new_date['second'] = end_date['second'].strip()
        
        # Bringing to a common standard 
        if '.' not in new_date['second']:
            new_date['second'] += '.0'
        
        # Checking the minutes
        if not (0 <= int(end_date['minute']) <= 59):
            return (None, False, "Invalid date format: minute should be between 0 and 59")
        else: # That's right
            new_date['minute'] = end_date['minute']
        
        # Checking the clock
        if not (0 <= int(end_date['hour']) <= 23):
            return (None, False, "Invalid date format: hour should be between 0 and 23")
        else: # That's right
            new_date['hour'] = end_date['hour']
        
        # Checking the day and month at the same time (may be mixed up in places)
        # Checking for zeroing first, has this happened in the data
        if (int(end_date['month']) == 0 or int(end_date['day']) == 0):
            return (None, False, "Invalid date format: day should be between 1 and 31. Change the all date on end_date.")
        # then we check that the day does not exceed the limits of the acceptable range 
        # (and we also check the month, since they may be mixed up in places)
        elif not ((1 <= int(end_date['day']) <= 31) and (1 <= int(end_date['month']) <= 31)):
            return (None, False, "Invalid date format: day should be between 1 and 31. Change the all date on end_date.")        
        elif not (1 <= int(end_date['month']) <= 12):
           # confused with each other (we count for a good month)
            if ((1 <= int(end_date['day']) <= 12) and (1 <= int(end_date['month']) <= 31)):
                new_date["month"] = end_date['day']
                new_date["day"] = end_date['month']
            # incorrect date, replace with end_date
            elif not ((1 <= int(end_date['day']) <= 12) and 1 <= int(end_date['month']) <= 31):
                return (None, False, "Invalid date format: hour should be between 0 and 23")
        else: # That's right
            new_date["month"] = end_date['month']
            new_date["day"] = end_date['day']        
        
        # Checking the year we consider valid since 2001, others require clarification
        if not (1970 <= int(end_date['year']) <= current_date.year):
            if (1 <= int(end_date['year']) <= (current_date.year - 2000)):
                new_date["year"] = str(int(end_date['year']) + 2000) # we consider the year to be correct, simple in a different format
            elif (70 <= int(end_date['year']) <= 99):
                new_date["year"] = str(int(end_date['year']) + 1900) # correcting the year from the last century
            else:
                return (None, False, "Invalid date format: year should be between 1970 and current year")
        else: # That's right
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
        The function groups files by sampling rate and network frequency.

        Args:
            source_dir (str): The directory containing the files to update.
            threshold (float): The threshold for considering frequency deviation from an integer as a measurement error.
            isPrintMessege (bool): A flag indicating whether to print a message if the frequencies are not found.

        Returns:
            frequency_statistics_dict dict: A dictionary containing the frequencies grouped by sampling rate and network.
        """
        # TODO: think about optimizing the code, the function is close to "grouping_by_sampling_rate_and_network".
        # also, the "extract_frequencies" function is taken from that library and an extra link appears.
        process_osc = ProcessingOscillograms()
        frequency_statistics_dict = {}
        
        print("We count the total number of files in the source directory...")
        total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
        print(f"Total number of files: {total_files}, starting processing...")
        with tqdm(total=total_files, desc="Grouping by sampling rate and network") as pbar:
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
                                if isPrintMessege: print(f"No network frequency found in the file: {file_path}")
                            elif f_rate:
                                if isPrintMessege: print(f"No sampling rate found in the file: {file_path}")
                            else:
                                if isPrintMessege: print(f"No frequencies found in the file: {file_path}")
        
        # sort by network and then by count
        sorted_dict = sorted(frequency_statistics_dict.items(), key=lambda x: (x[0], -sum(x[1].values()), -max(x[1].values())))

        all_count = 0
        for network, rate_dict in sorted_dict:
            print(f"Network: {network}")
            sorted_rate_dict = sorted(rate_dict.items(), key=lambda x: x[1], reverse=True)
            for rate, count in sorted_rate_dict:
                print(f"\tRate: {rate}, Count: {count}")
                all_count += count
        print(f"Total count: {all_count}")
        return frequency_statistics_dict
