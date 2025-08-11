import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft

from osc_tools.features.normalization import NormOsc
from osc_tools.io.comtrade_parser import ComtradeParser

# Этот файл будет содержать функции для специфической аналитической задачи:
# поиска осциллограмм с однофазными замыканиями на землю (ОЗЗ).

def _get_signals_from_prepared_cfg(file_path: str, encoding_name: str) -> list:
    """
    Разбирает файл .cfg, подготовленный для стандартных имен, и возвращает список полных имен сигналов (как аналоговых, так и цифровых).

    Args:
        file_path (str): путь к файлу .cfg.
        encoding_name (str): кодировка файла cfg

    Returns:
        list: список полных имен сигналов в файле cfg.
    """
    signal_names = []
    try:
        with open(file_path, 'r', encoding=encoding_name) as file:
            lines = file.readlines()
            count_all_signals_str, count_analog_signals_str, count_digital_signals_str = lines[1].split(',')
            count_analog_signals = int(count_analog_signals_str[:-1])
            count_digital_signals = int(count_digital_signals_str[:-2])

            # Обработка аналоговых сигналов
            for i in range(count_analog_signals):
                signal_name = lines[2 + i].split(',')
                name = signal_name[1].split('|')
                if len(name) < 2: # защита от некорректных строк
                    return signal_names.append({
                    'signal_type': "error_name",
                    'location_type': None,
                    'section_number': None,
                    'phase': None
                })
                signal_type = name[0].strip() # U, I
                section_info = name[1].strip().split("-")
                location_type = section_info[0] # BusBar, CableLine, Bus
                section_number = section_info[1] if len(section_info)>1 else None # 1, 2 ..., None
                phase_info = name[2].strip().split(":")
                phase = phase_info[1].strip() if len(phase_info)>1 else None # A, B, C ..., None
                signal_names.append({
                    'signal_type': signal_type,
                    'location_type': location_type,
                    'section_number': section_number,
                    'phase': phase
                })

            # Обработка цифровых сигналов
            for i in range(count_digital_signals):
                signal_name = lines[2 + count_analog_signals + i].split(',')
                if len(signal_name) < 2: # защита от некорректных строк
                    return signal_names.append({
                    'signal_type': "error_digital_signal",
                    'location_type': None,
                    'section_number': None,
                    'phase': None
                })
                name = signal_name[1].split('|')
                if len(name) < 2: # защита от некорректных строк
                    return signal_names.append({
                    'signal_type': "error_name",
                    'location_type': None,
                    'section_number': None,
                    'phase': None
                })
                signal_type = name[0].strip() # PDR...
                section_info = name[1].strip().split("-")
                location_type = section_info[0] # BusBar, CableLine, Bus
                section_number = section_info[1] if len(section_info)>1 else None # 1, 2 ..., None
                phase_info = name[-1].strip().split(":")
                phase = None # A, B, C ..., None
                if len(phase_info)>1 and (phase_info[0] == "phase"):
                    phase = phase_info[1].strip()
                signal_names.append({
                    'signal_type': signal_type,
                    'location_type': location_type,
                    'section_number': section_number,
                    'phase': phase
                })

    except Exception as e:
        print(f"Ошибка чтения файла cfg {file_path}: {e}")
    return signal_names

def _default_signal_checker(file_signals: list) -> bool:
    """
    Проверяет, содержит ли один файл comtrade необходимые сигналы (считая PDR цифровым).

    Args:
        file_signals (list): список имен сигналов.

    Returns:
        bool: True, если все необходимые сигналы присутствуют, иначе False.
    """
    has_voltage_busbar_1 = False
    has_voltage_cableline_1 = False
    has_voltage_busbar_2 = False
    has_voltage_cableline_2 = False
    has_current_bus_1_ac = False
    has_current_bus_2_ac = False
    r_has_pdr_bus_1 = False
    i_has_pdr_bus_1 = False

    voltage_busbar_1_phases = set()
    voltage_cableline_1_phases = set()
    voltage_busbar_2_phases = set()
    voltage_cableline_2_phases = set()
    current_bus_1_phases = set()
    current_bus_2_phases = set()
    r_pdr_bus_1_phases = set()
    i_pdr_bus_1_phases = set()

    for signal in file_signals:
        # проверка аналоговых сигналов
        if signal['signal_type'] == 'U':
            if signal['location_type'] == 'BusBar' and signal['section_number'] == '1':
                voltage_busbar_1_phases.add(signal['phase'])
            elif signal['location_type'] == 'CableLine' and signal['section_number'] == '1':
                voltage_cableline_1_phases.add(signal['phase'])
            elif signal['location_type'] == 'BusBar' and signal['section_number'] == '2':
                voltage_busbar_2_phases.add(signal['phase'])
            elif signal['location_type'] == 'CableLine' and signal['section_number'] == '2':
                voltage_cableline_2_phases.add(signal['phase'])
        elif signal['signal_type'] == 'I':
            if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                current_bus_1_phases.add(signal['phase'])
            elif signal['location_type'] == 'Bus' and signal['section_number'] == '2':
                current_bus_2_phases.add(signal['phase'])
        # проверка дискретных сигналов
        elif signal['signal_type'] == 'PDR': # реальный сигнал из осциллограмм
            if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                r_pdr_bus_1_phases.add(signal['phase'])
        elif signal['signal_type'] == 'PDR_ideal': # сигнал идеальной разметки
            if signal['location_type'] == 'Bus' and signal['section_number'] == '1':
                i_pdr_bus_1_phases.add(signal['phase'])

    has_voltage_busbar_1 = {'A', 'B', 'C'}.issubset(voltage_busbar_1_phases)
    has_voltage_cableline_1 = {'A', 'B', 'C'}.issubset(voltage_cableline_1_phases)
    has_voltage_busbar_2 = {'A', 'B', 'C'}.issubset(voltage_busbar_2_phases)
    has_voltage_cableline_2 = {'A', 'B', 'C'}.issubset(voltage_cableline_2_phases)

    has_current_bus_1_ac = {'A', 'C'}.issubset(current_bus_1_phases) and len(current_bus_1_phases) >= 2
    has_current_bus_2_ac = {'A', 'C'}.issubset(current_bus_2_phases) and len(current_bus_2_phases) >= 2

    r_has_pdr_bus_1 = {'PS'}.issubset(r_pdr_bus_1_phases) or {'A', 'B', 'C'}.issubset(r_pdr_bus_1_phases)
    i_has_pdr_bus_1 = {'PS'}.issubset(i_pdr_bus_1_phases) or {'A', 'B', 'C'}.issubset(i_pdr_bus_1_phases)

    voltage_condition = (has_voltage_busbar_1 or has_voltage_cableline_1) and (has_voltage_busbar_2 or has_voltage_cableline_2)
    current_condition = has_current_bus_1_ac and has_current_bus_2_ac
    pdr_condition = r_has_pdr_bus_1 or i_has_pdr_bus_1

    return voltage_condition and current_condition and pdr_condition

def find_oscillograms_with_spef(raw_path: str ='raw_data/', output_csv_path: str = "find_oscillograms_with_spef.csv",
                                norm_coef_file_path: str = 'norm_coef.csv', filter_txt_path: str = None):
    """
    Находит осциллограммы с однофазными замыканиями на землю (ОЗЗ) на основе определенных условий с использованием скользящего окна и гармонического анализа.

    Args:
        raw_path (str): путь к каталогу, содержащему файлы COMTRADE.
        output_csv_path (str): путь для сохранения CSV-файла с именами файлов ОЗЗ.
        norm_coef_file_path (str): путь к CSV-файлу с коэффициентами нормализации.
    """
    # Чтение фильтра (если указан)
    filter_set = None
    if filter_txt_path is not None:
        with open(filter_txt_path, 'r') as f:
            filter_set = set(line.strip() for line in f if line.strip())

    spef_files = []
    raw_files = sorted([file for file in os.listdir(raw_path) if 'cfg' in file])
    norm_osc = NormOsc(norm_coef_file_path=norm_coef_file_path)

    threshold = 30/400/3
    period_count = 3

    number_ocs_found = 0
    rawToCSV = ComtradeParser()

    with tqdm(total=len(raw_files), desc="Поиск ОЗЗ") as pbar:
        for file in raw_files:
            file_path = os.path.join(raw_path, file)
            filename_without_ext = file[:-4]
            if filter_set is not None and filename_without_ext not in filter_set:
                pbar.update(1)
                continue

            try:
                raw_date, raw_df = rawToCSV.readComtrade.read_comtrade(file_path)
                # TODO: Модернизировать "read_comtrade", а точнее даже функцию "comtrade.load(file_name)"
                # Так как сейчас приходится искусственно вытягивать нужные коэффициент
                samples_per_period = int(raw_date._cfg.sample_rates[0][0] / raw_date.frequency)
                samples_duration = period_count * samples_per_period
                if raw_df is None or raw_df.empty:
                    pbar.update(1)
                    continue

                buses_df = ComtradeParser.split_buses(ComtradeParser(), raw_df.reset_index(), file)
                if buses_df.empty:
                    pbar.update(1)
                    continue

                buses_df = norm_osc.normalize_bus_signals(buses_df, filename_without_ext, yes_prase="YES", is_print_error=False)
                if buses_df is None:
                    pbar.update(1)
                    continue

                for file_name, group_df in buses_df.groupby("file_name"):
                    is_spef = False
                    group_df = group_df.copy()

                    # Условие 1 и 2: 3U0 BB и 3U0 CL (объединены для эффективности)
                    u0_signals_bb_cl = {}
                    signal_names_3u0 = {"UN BB": "UN BB", "UN CL": "UN CL"} # Сопоставление имен сигналов

                    for signal_3u0_name, col_name in signal_names_3u0.items():
                        if not group_df.empty and col_name in group_df.columns:
                            u0_signal = group_df[col_name].fillna(0).values
                            u0_harmonics = np.zeros_like(u0_signal, dtype=float) # Массив для хранения значений первой гармоники

                            # Скользящее окно для расчета и проверки гармоник
                            for i in range(len(u0_signal) - samples_per_period): # Скольжение по всему сигналу
                                window = u0_signal[i:i+samples_per_period]
                                window_fft = np.abs(fft(window)) / samples_per_period
                                u0_harmonics[i] = max(window_fft[1:samples_per_period//2])

                            # Проверка скользящим окном по гармоникам
                            for i in range(len(u0_harmonics) - samples_duration):
                                window_harmonics = u0_harmonics[i:i+samples_duration]
                                if np.all(window_harmonics >= threshold): # Проверка уровня гармоник в скользящем окне
                                    is_spef = True
                                    number_ocs_found += 1
                                    spef_files.append([filename_without_ext, file_name])
                                    break # Условие выполнено, нет необходимости проверять дальнейшие условия для этого файла
                            if is_spef:
                                break # Прервать внешний цикл, если ОЗЗ найдено

                        if is_spef:
                            break # Прервать, если ОЗЗ найдено

                    if is_spef:
                        continue # Переход к следующему файлу, если ОЗЗ найдено

                    # Условие 3 и 4: Нулевая последовательность + фазные напряжения BB и CL (объединены)
                    phase_voltage_conditions = {
                        "BB": {"phases": ["UA BB", "UB BB", "UC BB"], "threshold_u0": threshold, "threshold_phase": threshold/np.sqrt(3)}, # Использование threshold_phase при необходимости для фазных напряжений
                        "CL": {"phases": ["UA CL", "UB CL", "UC CL"], "threshold_u0": threshold, "threshold_phase": threshold/np.sqrt(3)}  # Использование threshold_phase при необходимости для фазных напряжений
                    }

                    for location, condition_params in phase_voltage_conditions.items():
                        phase_names = condition_params["phases"]
                        threshold_u0 = condition_params["threshold_u0"]
                        threshold_phase = condition_params["threshold_phase"] # В настоящее время не используется, может использоваться для проверок уровня фазного напряжения

                        if not group_df.empty and all(col in group_df.columns for col in phase_names):
                            ua_signal = group_df[phase_names[0]].fillna(0).values
                            ub_signal = group_df[phase_names[1]].fillna(0).values
                            uc_signal = group_df[phase_names[2]].fillna(0).values

                            u0_3_signal = (ua_signal + ub_signal + uc_signal) / np.sqrt(3) # Расчет 3U0
                            # /np.sqrt(3) - потому что мы использовали фазный сигнал

                            u0_3_harmonics = np.zeros_like(u0_3_signal, dtype=float) # Массив для гармоник
                            for i in range(len(u0_3_signal) - samples_per_period):
                                window = u0_3_signal[i:i+samples_per_period]
                                window_fft = np.abs(fft(window)) / samples_per_period
                                u0_3_harmonics[i] = max(window_fft[1:samples_per_period//2])

                            phase_voltages = [ua_signal, ub_signal, uc_signal]
                            voltages_above_threshold = 0 # Подсчет, сколько фазных напряжений удовлетворяют условию (при необходимости)

                            # Проверка скользящим окном для гармоник 3U0
                            for i in range(len(u0_3_harmonics) - samples_duration):
                                window_harmonics_u0 = u0_3_harmonics[i:i+samples_duration]
                                if np.all(window_harmonics_u0 >= threshold_u0): # Проверка уровня гармоник 3U0 в скользящем окне
                                    voltages_above_threshold = 0 # Сброс счетчика для фазных напряжений для этого окна
                                    for v in phase_voltages:
                                        # Здесь можно добавить проверку фазного напряжения, если это необходимо, например, с использованием гармоник или значений во временной области в окне
                                        # Пока просто проверяем, существуют ли какие-либо два напряжения (согласно исходному условию)
                                        voltages_above_threshold += 1 # Увеличение, если сигнал напряжения существует (для простоты, можно улучшить)

                                    if voltages_above_threshold >= 2: # Проверка наличия как минимум двух фазных напряжений (условие из исходной задачи)
                                        is_spef = True
                                        number_ocs_found += 1
                                        spef_files.append([filename_without_ext, file_name])
                                        break # Условие выполнено, нет необходимости проверять дальнейшие условия для этого файла
                            if is_spef:
                                break # Прервать цикл по местоположению, если ОЗЗ найдено

                        if is_spef:
                            break # Прервать внешний цикл, если ОЗЗ найдено

            except Exception as e:
                print(f"Ошибка обработки файла {file}: {e}")

            pbar.update(1)

    print(f"Количество найденных образцов = {number_ocs_found}")
    df_spef = pd.DataFrame(spef_files, columns=['filename', 'file_name_bus'])
    df_spef.to_csv(output_csv_path, index=False)
    print(f"Файлы ОЗЗ сохранены в: {output_csv_path}")
