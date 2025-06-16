import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# import comtrade # Replaced by Oscillogram
from typing import Dict, Any, Tuple, List, Union, Set # Added Set
import re # Для регулярных выражений
# from typing import List, Set, Union # Duplicate typing import
import json
# from tqdm import tqdm # Duplicate tqdm import, tqdm.auto is preferred generally

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

# from dataflow.comtrade_processing import ReadComtrade # Replaced by Oscillogram
from core.oscillogram import Oscillogram

VOLTAGE_T1_S = 20 * np.sqrt(2)
VOLTAGE_T1_P = 500 * np.sqrt(2)
VOLTAGE_T2_S = 140 * np.sqrt(2)
VOLTAGE_T3_S = 560 * np.sqrt(2)

CURRENT_T1_S = 0.03 * np.sqrt(2)
CURRENT_T1_P = 20 * np.sqrt(2)
CURRENT_T2_S = 30 * np.sqrt(2)

RESIDUAL_CURRENT_T1_S = 0.02 * np.sqrt(2)
RESIDUAL_CURRENT_T2_S = 5 * np.sqrt(2)

DIFFERENTIAL_CURRENT_THRESHOLD = 30

NOISE_FACTOR = 1.5 # Коэффициент для сравнения m1 и mx (m1 <= 1.5 * mx)

# ?1 - A non-standard sensor, the secondary values are too small, but the primary values are normal.
# ?2 - A large proportion of higher harmonics, the signal is highly distorted and it is difficult to judge the nominal value.
# ?3 - Probably there is no transformation coefficient (secondary values are too large)

class NormalizationCoefficientGenerator: # Renamed from CreateNormOsc
    def __init__(self,
                 osc_path: str,
                 prev_norm_csv_path: str = "",
                 step_size: int = 1,
                 bus: int = 6
                 ):
        self.osc_path = osc_path
        # self.readComtrade = ReadComtrade() # REMOVED
        self.existing_norm_df = pd.DataFrame()
        if prev_norm_csv_path and os.path.exists(prev_norm_csv_path):
            try:
                self.existing_norm_df = pd.read_csv(prev_norm_csv_path, dtype=str)
                if hasattr(self, 'is_print_message') and self.is_print_message: # Assuming is_print_message might be added
                    print(f"Loaded existing normalization coefficients from: {prev_norm_csv_path}")
            except Exception as e:
                if hasattr(self, 'is_print_message') and self.is_print_message:
                    print(f"Error loading existing normalization coefficients from '{prev_norm_csv_path}': {e}")
        elif prev_norm_csv_path:
            if hasattr(self, 'is_print_message') and self.is_print_message:
                print(f"Warning: File of existing normalization coefficients '{prev_norm_csv_path}' not found.")

        self.window_size = 32 # Default, will be updated per file in normalization()
        self.step_size = step_size
       
        self.ru_cols, self.u_cols = self.generate_VT_cols(bus=bus)
        self.ri_cols, self.i_cols = self.generate_CT_cols(bus=bus)
        self.riz_cols, self.iz_cols = self.generate_rCT_cols(bus=bus)
        self.rid_cols, self.id_cols = self.generate_dI_cols(1)
        
        self.raw_cols = self.generate_raw_cols(bus=bus)
        
        self.result_cols = self.generate_result_cols(bus=bus)
        
        self.all_features = self.generate_all_features(bus=bus)

        self.df = pd.DataFrame(data=self.result_cols)

        self.osc_files = sorted([file for file in os.listdir(self.osc_path)
                            if "cfg" in file], reverse=False)
    
    def get_summary(self,):
        unread = []
        c = 0
        for file in tqdm(self.osc_files):
            try:
                osc_df = comtrade.load_as_dataframe(self.osc_path + file)
            except Exception as ex:
                unread.append((file, ex))
                continue
            c += 1
            if c > 30:
                break
        with open('unread_files.txt', 'w') as f:
            for line in unread:
                f.write(f"{line}\n")

    def generate_VT_cols(self, bus = 6, isRaw = False): # VT - Voltage transformer
        ru_cols = [(f"{i}Ub", f"{i}Uc") for i in range(1, bus+1)]
        u_cols = dict()
        name = "U"
        if isRaw:
            name = "U_raw"
        
        for i in range(1, bus+1):
            base = f'{name} | BusBar-' + str(i) + ' | phase: '
            u_cols[ru_cols[i-1][0]] = {base + 'A', base + 'B', base + 'C', base + 'N', 
                                     base + 'AB', base + 'BC', base + 'CA'}
            cable = f'{name} | CableLine-' + str(i) + ' | phase: '
            u_cols[ru_cols[i-1][1]] = {cable + 'A', cable + 'B', cable + 'C', cable + 'N', 
                                     cable + 'AB', cable + 'BC', cable + 'CA'}
        
        ru_cols = [item for sublist in ru_cols for item in sublist]
        return ru_cols, u_cols

    def generate_CT_cols(self, bus = 6, isRaw = False): # CT - current transformer
        ri_cols = [f"{i}Ip" for i in range(1, bus+1)]
        i_cols = dict()
        name = "I"
        if isRaw:
            name = "I_raw"
        
        for i in range(1, bus+1):
            phase = f'{name} | Bus-' + str(i) + ' | phase: '
            i_cols[ri_cols[i-1]] = {phase + 'A', phase + 'B', phase + 'C'}
            
        return ri_cols, i_cols

    def generate_rCT_cols(self, bus = 6): # Residual current transformer (rCT)
        riz_cols = [f"{i}Iz" for i in range(1, bus+1)]
        iz_cols = dict()
        
        for i in range(1, bus+1):
            iz_cols[riz_cols[i-1]] = 'I | Bus-' + str(i) + ' | phase: N'
            
        return riz_cols, iz_cols

    def generate_dI_cols(self, bus = 1): # different current (for protection - 87)
        rid_cols = [f"{i}Id" for i in range(1, bus+1)]
        id_cols = dict()
        
        for i in range(1, bus+1):
            phase_d = 'I | dif-' + str(i) + ' | phase: '
            phase_b = 'I | breaking-' + str(i) + ' | phase: '
            id_cols[rid_cols[i-1]] = {phase_d + 'A', phase_d + 'B', phase_d + 'C',
                                    phase_b + 'A', phase_b + 'B', phase_b + 'C' }
            
        return rid_cols, id_cols

    def generate_raw_cols(self, bus = 6):
        _, raw_cols_VT = self.generate_VT_cols(bus=bus, isRaw=True)
        _, raw_cols_CT = self.generate_CT_cols(bus=bus, isRaw=True)

        raw_cols = set()
        for name in raw_cols_VT.keys():
            one_set = raw_cols_VT[name]
            raw_cols.update(one_set)
        for name in raw_cols_CT.keys():
            one_set = raw_cols_CT[name]
            raw_cols.update(one_set)
        
        return set(raw_cols)

    def generate_result_cols(self, bus = 6):
        # TODO: Merge with other generation functions as they are related.
        result_cols = dict()
        result_cols['name'], result_cols['norm'] = [], []
        for i in range(1, bus+1):
            result_cols.update({
                f'{i}Ub_PS': [], f'{i}Ub_base': [], f'{i}Ub_h1': [], f'{i}Ub_hx': [],
                f'{i}Uc_PS': [], f'{i}Uc_base': [], f'{i}Uc_h1': [], f'{i}Uc_hx': [],
                f'{i}Ip_PS': [], f'{i}Ip_base': [], f'{i}Ip_h1': [], f'{i}Ip_hx': [],
                f'{i}Iz_PS': [], f'{i}Iz_base': [], f'{i}Iz_h1': [], f'{i}Iz_hx': []
            })
        result_cols.update({'dId_PS': [], 'dId_base': [], 'dId_h1': []})
        return result_cols
    
    def generate_all_features(self, bus = 6):
        # TODO: Merge with other generation functions as they are related.
        all_features = set()
        for name in self.u_cols.keys():
            one_set = self.u_cols[name]
            all_features.update(one_set)
        for name in self.i_cols:
            one_set = self.i_cols[name]
            all_features.update(one_set)
        for name in self.iz_cols.keys():
            one_set = self.iz_cols[name]
            all_features.update(one_set)
        for name in self.id_cols.keys():
            one_set = self.id_cols[name]
            all_features.update(one_set)
        all_features.update(self.raw_cols)
        
        return all_features
    
    def _get_max_primary_value(self,
                               h1_df: pd.DataFrame,
                               columns: List[str],
                               coef_p_s: Dict[str, float]) -> float:
        """Вычисляет максимальное первичное значение для заданных столбцов."""
        max_primary = 0.0
        # Используем .at[0, col] для доступа к значению в однострочном DataFrame
        for name in columns:
            if name in h1_df.columns and name in coef_p_s:
                primary_value = h1_df.at[0, name] * coef_p_s[name]
                if not np.isnan(primary_value):
                    max_primary = max(max_primary, primary_value)
        return max_primary
    
    def _determine_voltage_status(self,
                                  m1: float,
                                  mx: float,
                                  h1_df: pd.DataFrame,
                                  columns: List[str],
                                  coef_p_s: Dict[str, float]) -> Tuple[str, Any]:
        """Определяет статус (_PS, _base) для измерений напряжения."""
        if m1 <= VOLTAGE_T1_S:
            if m1 <= NOISE_FACTOR * mx:
                return 's', 'Noise'
            else:
                max_primary = self._get_max_primary_value(h1_df, columns, coef_p_s)
                if max_primary > VOLTAGE_T1_P:
                    return 'p', '?1'
                else:
                    return 's', 'Noise'
        elif VOLTAGE_T1_S < m1 <= VOLTAGE_T2_S:
            return 's', 100 if m1 > NOISE_FACTOR * mx else '?2'
        elif VOLTAGE_T2_S < m1 <= VOLTAGE_T3_S:
            return 's', 400 if m1 > NOISE_FACTOR * mx else '?2'
        else:
            # TODO: Добавить разделение определения номиналов по линейным / фазным значениям для случая, когда это превичка
            # Потому что по ним, для больниства случаев - можно определить
            return '?3', '?3' # Неопределенное состояние или выход за пределы
        
    def _determine_current_status(self,
                                m1: float,
                                mx: float,
                                h1_df: pd.DataFrame,
                                columns: List[str],
                                coef_p_s: Dict[str, float]) -> Tuple[str, Any]:
        """Определяет статус (_PS, _base) для измерений тока."""
        if m1 <= CURRENT_T1_S:
            if m1 <= NOISE_FACTOR * mx:
                return 's', 'Noise'
            else:
                max_primary = self._get_max_primary_value(h1_df, columns, coef_p_s)
                if max_primary > CURRENT_T1_P:
                    return 'p', '?1'
                else:
                    return 's', 'Noise'
        elif CURRENT_T1_S < m1 <= CURRENT_T2_S:
            return 's', 5 if m1 > NOISE_FACTOR * mx else '?2'
        else:
            return '?3', '?3'
        
    def _determine_residual_current_status(self, m1: float, mx: float) -> Tuple[str, Any]:
        """Определяет статус (_PS, _base) для измерений остаточного тока."""
        # Обратите внимание: в оригинальной логике использовалось OR для m1 и mx
        # Это может быть не совсем логично, стоит перепроверить требования
        # Здесь оставлена оригинальная логика: если хотя бы одно значение в диапазоне
        if m1 <= RESIDUAL_CURRENT_T1_S and mx <= RESIDUAL_CURRENT_T1_S:
            return 's', 'Noise'
        elif (RESIDUAL_CURRENT_T1_S < m1 <= RESIDUAL_CURRENT_T2_S and mx <= RESIDUAL_CURRENT_T2_S 
              or
              RESIDUAL_CURRENT_T1_S < mx <= RESIDUAL_CURRENT_T2_S and m1 <= RESIDUAL_CURRENT_T2_S):
            return 's', 1
        else:
            return '?3', '?3'
        
    def _determine_diff_current_status(self, m1: float) -> Tuple[str, Any]:
        """Определяет статус (_PS, _base) для измерений дифференциального тока."""
        if m1 < DIFFERENTIAL_CURRENT_THRESHOLD:
            return 's', 1
        else:
            return '?3', '?3'

    # Вспомогательная функция для определения максимального номера секции
    def _get_max_bus_number_from_columns(columns: set[str]) -> int:
        """
        Определяет максимальный номер секции из имен столбцов.
        Пример: "1Ub_PS", "12Ip_base" -> 12.
        Если номера не найдены, возвращает 1 (предполагая хотя бы одну секцию).
        """
        max_bus = 0
        # Паттерн: число в начале, за которым следует один из типов и '_'.
        # r"^(\d+)(Ub|Uc|Ip|Iz)_"
        # Более общий паттерн, если суффиксы могут отличаться или быть _PS, _base, _h1, _hx
        pattern = re.compile(r"^(\d+)(?:Ub|Uc|Ip|Iz)(?:_PS|_base|_h1|_hx)")
        found_any_number = False
        for col_name in columns:
            match = pattern.match(col_name)
            if match:
                try:
                    bus_num = int(match.group(1))
                    if bus_num > max_bus:
                        max_bus = bus_num
                    found_any_number = True
                except ValueError:
                    continue # На случай, если первая группа не число (маловероятно с таким паттерном)
        return max_bus if found_any_number else 1 # Если номеров нет, по умолчанию 1 секция

    def _should_skip_calculation(self, 
                                 results: Dict[str, Any], 
                                 group_prefix: str, 
                                 suffixes_to_check: List[str]) -> bool:
        """
        Проверяет, следует ли пропустить расчет для данной группы сигналов,
        так как все необходимые данные уже присутствуют в results.

        Args:
            results (Dict[str, Any]): Текущий словарь с результатами анализа.
            group_prefix (str): Префикс группы сигналов (например, "1Ub", "2Ip").
            suffixes_to_check (List[str]): Список суффиксов (_PS, _base),
                                           формирующих полные имена столбцов для проверки.

        Returns:
            bool: True, если все указанные поля для данной группы существуют и не пусты,
                  иначе False.
        """
        for suffix in suffixes_to_check:
            col_name = group_prefix + suffix
            value = results.get(col_name)
            # Проверка на None, пустую строку, или pandas/numpy NaN
            if value is None or str(value).strip() == '' or pd.isna(value):
                return False # Нашли отсутствующее или пустое значение, расчет нужен
        return True # Все значения на месте и не пусты, можно пропустить расчет
    
    def analyze(self,
                file_identifier: str, # Changed from file
                h1_df: pd.DataFrame,
                hx_df: pd.DataFrame,
                coef_p_s: Dict[str, float],
                existing_row: pd.DataFrame = pd.DataFrame()
                ) -> pd.DataFrame:
        """
        Анализирует данные одной осциллограммы (гармоники h1 и hx)
        и определяет базовые значения и тип подключения (первичка/вторичка).

        Args:
            file_identifier: Identifier for the oscillogram (e.g., file hash).
            h1_df: DataFrame с амплитудами первой гармоники.
            hx_df: DataFrame с максимальными амплитудами высших гармоник (>=2).
            coef_p_s: Словарь с коэффициентами трансформации (первичка/вторичка).
            existing_row: Existing normalization data for this oscillogram.

        Returns:
            Однострочный DataFrame с результатами анализа.
        """
        features = set(h1_df.columns)
        if not existing_row.empty:
            results = existing_row.iloc[0].fillna('').to_dict()
            results['name'] = file_identifier
        else:
            results: Dict[str, Any] = {'name': file_identifier, 'norm': 'YES'}

        suffixes = ['_PS', '_base']
        
        # --- Обработка напряжения ---
        for r_prefix in self.ru_cols:
            if self._should_skip_calculation(results, r_prefix, suffixes):
                # print(f"Skipping recalculation for {r_prefix} for {file[:-4]} - data exists.") # Для отладки
                continue
            
            # Находим пересечение нужных столбцов с доступными фичами
            relevant_cols = list(self.u_cols[r_prefix].intersection(features))
            if not relevant_cols: # Пропускаем, если нет данных для этой группы
                continue

            m1 = h1_df[relevant_cols].max(axis=1).iloc[0]
            mx = hx_df[relevant_cols].max(axis=1).iloc[0]

            if np.isnan(m1): # Пропускаем, если основные данные некорректны
                 continue

            results[r_prefix + '_h1'] = m1
            results[r_prefix + '_hx'] = mx

            ps_status, base_status = self._determine_voltage_status(m1, mx, h1_df, relevant_cols, coef_p_s)
            results[r_prefix + '_PS'] = ps_status
            results[r_prefix + '_base'] = base_status

        # --- Обработка тока ---
        for r_prefix in self.ri_cols:
            if self._should_skip_calculation(results, r_prefix, suffixes):
                # print(f"Skipping recalculation for {r_prefix} for {file[:-4]} - data exists.") # Для отладки
                continue
            
            relevant_cols = list(self.i_cols[r_prefix].intersection(features))
            if not relevant_cols:
                continue

            m1 = h1_df[relevant_cols].max(axis=1).iloc[0]
            mx = hx_df[relevant_cols].max(axis=1).iloc[0]

            if np.isnan(m1):
                continue

            results[r_prefix + '_h1'] = m1
            results[r_prefix + '_hx'] = mx

            ps_status, base_status = self._determine_current_status(m1, mx, h1_df, relevant_cols, coef_p_s)
            results[r_prefix + '_PS'] = ps_status
            results[r_prefix + '_base'] = base_status

        # --- Обработка тока нулевой последовательности ---
        for r_prefix in self.riz_cols:
            if self._should_skip_calculation(results, r_prefix, suffixes):
                # print(f"Skipping recalculation for {r_prefix} for {file[:-4]} - data exists.") # Для отладки
                continue
            
            col_set = set([self.iz_cols[r_prefix]])
            relevant_cols = list(col_set.intersection(features))
            if not relevant_cols:
                continue

            col_name = relevant_cols[0]
            m1 = h1_df.at[0, col_name]
            mx = hx_df.at[0, col_name]


            if np.isnan(m1):
                 continue

            results[r_prefix + '_h1'] = m1
            results[r_prefix + '_hx'] = mx

            ps_status, base_status = self._determine_residual_current_status(m1, mx)
            results[r_prefix + '_PS'] = ps_status
            results[r_prefix + '_base'] = base_status

        # --- Обработка дифференциального тока ---
        # TODO: В оригинальном коде использовался только один префикс 'dId' для всех rid_cols.
        # Уточнить, нужно ли обрабатывать каждый префикс из self.rid_cols отдельно
        # или достаточно одного общего 'dId'. Пока оставлено как в оригинале - общий.
        processed_did = False
        for r_prefix in self.rid_cols:
            if self._should_skip_calculation(results, r_prefix, suffixes):
                # print(f"Skipping recalculation for {r_prefix} for {file[:-4]} - data exists.") # Для отладки
                continue
            
            if processed_did:
                break
            relevant_cols = list(self.id_cols[r_prefix].intersection(features))
            if not relevant_cols:
                continue

            m1 = h1_df[relevant_cols].max(axis=1).iloc[0]

            if np.isnan(m1):
                continue

            results['dId_h1'] = m1

            ps_status, base_status = self._determine_diff_current_status(m1)
            results['dId_PS'] = ps_status
            results['dId_base'] = base_status
            processed_did = True # Помечаем, что обработали

        # --- Финальное определение статуса 'norm' ---
        # 1. Проверка на 'raw' (наличие "сырых" столбцов)
        if self.raw_cols.intersection(features):
             results['norm'] = 'raw'
        # 2. Проверка на неопределенность ('?')
        elif any(val in ['?1', '?2', '?3'] for val in results.values()):
             results['norm'] = 'NO' # Непонятно, требуется разобраться
        # 3. Проверка на шум ('Noise')
        elif 'Noise' in results.values():
             # TODO: "почему-то он формируется излишне - требуется перепроверить"
             results['norm'] = 'hz' # Шум

        return pd.DataFrame([results], columns=self.result_cols.keys())

    def normalization(self, bus = 6): # Parameter 'bus' seems unused here if generate_cols in __init__ uses instance's bus
        unread = []
        # Ensure result_cols is a dictionary as expected by pd.DataFrame(data=...)
        # Or, if it's for columns, pd.DataFrame(columns=list(self.result_cols.keys()))
        if not hasattr(self, 'result_cols') or not isinstance(self.result_cols, dict):
             # This should ideally be an error or handled more gracefully
             print("Error: self.result_cols is not initialized correctly for DataFrame creation.")
             return
        result_df = pd.DataFrame(columns=list(self.result_cols.keys()))

        for file_name_with_ext in tqdm(self.osc_files):
            current_file_identifier = "" # Will be set by osc_obj.file_hash
            existing_row_for_osc = pd.DataFrame()
            
            try:
                cfg_file_full_path = os.path.join(self.osc_path, file_name_with_ext)
                osc_obj = Oscillogram(cfg_file_full_path)
                current_file_identifier = osc_obj.file_hash

                if not self.existing_norm_df.empty and 'name' in self.existing_norm_df.columns:
                    match = self.existing_norm_df[self.existing_norm_df['name'].astype(str) == str(current_file_identifier)]
                    if not match.empty:
                        existing_row_for_osc = match.iloc[[0]]

                if osc_obj.data_frame is None or osc_obj.data_frame.empty:
                    if hasattr(self, 'is_print_message') and self.is_print_message:
                        print(f"Warning: No data in {file_name_with_ext}. Skipping.")
                    unread.append(file_name_with_ext)
                    continue

                osc_df = osc_obj.data_frame # This is already a DataFrame
                frequency = osc_obj.frequency

                if not osc_obj.cfg.sample_rates or len(osc_obj.cfg.sample_rates[0]) < 1 or frequency is None or frequency == 0:
                    if hasattr(self, 'is_print_message') and self.is_print_message:
                        print(f"Warning: Invalid frequency ({frequency}) or sample_rates for {file_name_with_ext}. Skipping.")
                    unread.append(file_name_with_ext)
                    continue
                samples_rate = osc_obj.cfg.sample_rates[0][0]

                if samples_rate == 0:
                    self.window_size = 32 # Default or handle error
                else:
                    self.window_size = int(samples_rate / frequency)

                if self.window_size <= 0:
                    if hasattr(self, 'is_print_message') and self.is_print_message:
                        print(f"Warning: window_size is {self.window_size} for {file_name_with_ext} (sample_rate: {samples_rate}, frequency: {frequency}). Skipping.")
                    unread.append(file_name_with_ext)
                    continue

                coef_p_s = {}
                for analog_channel in osc_obj.raw_comtrade_obj.cfg.analog_channels:
                    if analog_channel.secondary != 0:
                        coef = analog_channel.primary / analog_channel.secondary
                    else:
                        coef = float('inf')
                    coef_p_s[analog_channel.name] = coef
                
                # Filter osc_df columns to only include those in self.all_features
                # This ensures FFT is done only on relevant signals if all_features is correctly defined
                # to match what analyze expects.
                relevant_df_columns = [col for col in osc_df.columns if col in self.all_features]
                filtered_osc_df = osc_df[relevant_df_columns]

                if filtered_osc_df.empty or filtered_osc_df.shape[0] < self.window_size:
                    if hasattr(self, 'is_print_message') and self.is_print_message:
                        print(f"Warning: Not enough data points or relevant columns after filtering for {file_name_with_ext}. Skipping FFT.")
                    unread.append(file_name_with_ext)
                    continue

                # FFT calculation logic based on original code
                osc_fft = np.abs(np.fft.fft(filtered_osc_df.iloc[:self.window_size], axis=0))
                for i_fft in range(self.window_size, filtered_osc_df.shape[0], self.step_size):
                    if i_fft - self.window_size < 0 or i_fft > filtered_osc_df.shape[0]:
                        break
                    window_data = filtered_osc_df.iloc[i_fft - self.window_size: i_fft]
                    if window_data.shape[0] == self.window_size:
                        window_fft = np.abs(np.fft.fft(window_data, axis=0))
                        osc_fft = np.maximum(osc_fft, window_fft)

                h1 = np.zeros(filtered_osc_df.shape[1]) # Default to zeros
                hx = np.zeros(filtered_osc_df.shape[1]) # Default to zeros

                if self.window_size > 0 and osc_fft.shape[0] > 1: # Need at least DC and 1st harmonic
                    h1 = 2 * osc_fft[1] / self.window_size

                harmonic_count = self.window_size // 2
                if harmonic_count >= 2 and self.window_size > 0 and osc_fft.shape[0] > harmonic_count :
                    hx = 2 * np.max(osc_fft[2:harmonic_count+1], axis=0) / self.window_size

                osc_features = filtered_osc_df.columns.tolist()
                h1_df = pd.DataFrame([dict(zip(osc_features, h1))])
                hx_df = pd.DataFrame([dict(zip(osc_features, hx))])

                result = self.analyze(current_file_identifier, h1_df, hx_df, coef_p_s, existing_row=existing_row_for_osc)
                result_df = pd.concat([result_df, result], ignore_index=True)

            except FileNotFoundError: # Should be caught by Oscillogram, but good to have a catch here
                if hasattr(self, 'is_print_message') and self.is_print_message:
                     print(f"Warning: File not found for {file_name_with_ext} by NormalizationCoefficientGenerator. Skipping.")
                unread.append(file_name_with_ext)
                continue
            except RuntimeError as e:
                if hasattr(self, 'is_print_message') and self.is_print_message:
                     print(f"Warning: Runtime error loading/processing {file_name_with_ext} via Oscillogram: {e}. Skipping.")
                unread.append(file_name_with_ext)
                continue
            except Exception as e:
                if hasattr(self, 'is_print_message') and self.is_print_message: # Generic catch for safety
                    print(f"Warning: Unexpected error processing {file_name_with_ext} in NormalizationCoefficientGenerator: {e}")
                unread.append(file_name_with_ext)
                continue

        output_file_path = os.path.join(self.osc_path, 'norm.csv') # Save in the input osc_path, or make configurable
        try:
            result_df.to_csv(output_file_path, index=False)
            if hasattr(self, 'is_print_message') and self.is_print_message:
                print(f"Normalization coefficients saved to {output_file_path}")
        except Exception as e:
             if hasattr(self, 'is_print_message') and self.is_print_message:
                print(f"Error saving normalization results to {output_file_path}: {e}")

        if unread and hasattr(self, 'is_print_message') and self.is_print_message:
            print(f"Files not read or skipped during normalization: {unread}")

    @staticmethod
    def update_dataframe_static(df_path: str, processed_files_path: str, output_df_path: str, is_print_message: bool = False) -> None:
        try:
            df = pd.read_csv(df_path)
        except FileNotFoundError:
            if is_print_message: print(f"Error: DataFrame file not found at {df_path}")
            return
        except Exception as e:
            if is_print_message: print(f"Error reading DataFrame from {df_path}: {e}")
            return

        try:
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                processed_files = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            if is_print_message: print(f"Error: Processed files list not found at {processed_files_path}")
            return
        except Exception as e:
            if is_print_message: print(f"Error reading processed files list from {processed_files_path}: {e}")
            return

        if not processed_files:
            if is_print_message: print("No processed files listed. DataFrame not updated.")
            # Optionally save the original df to the output path if no changes are to be made
            # Or simply return if the convention is to only write if there are changes.
            # For consistency with original logic, let's assume it might have been intended to save.
            # However, if no processed_files, no mask is true, so df is unchanged.
            # To avoid writing an identical file, we can just return.
            return

        # Ensure 'name' column exists in DataFrame
        if 'name' not in df.columns:
            if is_print_message: print(f"Error: 'name' column not found in DataFrame from {df_path}.")
            return

        mask = df['name'].astype(str).isin(processed_files)

        # Columns to update
        columns_to_update = {
            '2Ip_PS': "s",
            '2Ip_base': "5", # Original used string "5"
            '2Ip_h1': "5",   # Original used string "5"
            '2Ip_hx': "1"    # Original used string "1"
        }

        for col, value in columns_to_update.items():
            if col in df.columns:
                df.loc[mask, col] = value
            elif is_print_message:
                print(f"Warning: Column '{col}' not found in DataFrame. Cannot update.")

        try:
            output_dir = os.path.dirname(output_df_path)
            if output_dir and not os.path.exists(output_dir): # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_df_path, index=False, encoding='utf-8')
            if is_print_message: print(f"Updated DataFrame saved to {output_df_path}")
        except Exception as e:
            if is_print_message: print(f"Error saving updated DataFrame to {output_df_path}: {e}")

    @staticmethod
    def correct_df_static(input_path: str, output_path: str, is_print_message: bool = False) -> None:
        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            if is_print_message: print(f"Error: Input CSV file not found at {input_path}")
            return
        except Exception as e:
            if is_print_message: print(f"Error reading CSV from {input_path}: {e}")
            return

        if 'norm' not in df.columns:
            if is_print_message: print(f"Error: 'norm' column not found in {input_path}. Cannot perform corrections.")
            # Optionally save the original df or return
            # df.to_csv(output_path, index=False, encoding='utf-8')
            return

        mask = df['norm'].astype(str).isin(['NO', 'hz']) # astype(str) for safety

        def correct_values_helper(row, prefix, target_values_set, default_ps, default_base):
            # target_values_set should contain strings
            candidates = [f"{i}{prefix}_base" for i in range(1, 7)] # Max 6 sections hardcoded in original
            current_row_base_values = []
            for col_candidate in candidates:
                if col_candidate in row and str(row[col_candidate]) in target_values_set:
                    current_row_base_values.append(str(row[col_candidate]))

            replacement_base = default_base # Default if no specific condition met
            if '100' in current_row_base_values: replacement_base = '100'
            elif '400' in current_row_base_values: replacement_base = '400'

            for i in range(1, 7):
                base_col = f"{i}{prefix}_base"
                ps_col = f"{i}{prefix}_PS"
                if base_col in row and str(row[base_col]) in target_values_set:
                    if base_col in df.columns: row[base_col] = replacement_base
                    if ps_col in df.columns: row[ps_col] = default_ps
            return row

        # Ensure target_values are strings for comparison
        target_vals_for_u = {'?1', '?2', 'Noise'}
        target_vals_for_i = {'?1', '?2', 'Noise'} # Assuming same for Ip based on original

        # Apply corrections row by row where mask is True
        # Create a copy of the slice to avoid SettingWithCopyWarning
        df_to_update = df[mask].copy()

        df_to_update = df_to_update.apply(lambda row: correct_values_helper(row, 'Ub', target_vals_for_u, 's', '100'), axis=1)
        df_to_update = df_to_update.apply(lambda row: correct_values_helper(row, 'Uc', target_vals_for_u, 's', '100'), axis=1)
        df_to_update = df_to_update.apply(lambda row: correct_values_helper(row, 'Ip', target_vals_for_i, 's', '5'), axis=1)

        df_to_update['norm'] = 'YES'

        # Update original DataFrame
        df.loc[mask] = df_to_update

        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            if is_print_message: print(f"Corrected DataFrame saved to {output_path}")
        except Exception as e:
            if is_print_message: print(f"Error saving corrected DataFrame to {output_path}: {e}")

    @staticmethod
    def merge_normalization_files_static(input_paths_or_folder: Union[str, List[str]],
                                         output_csv_path: str,
                                         file_pattern: str = r'^norm_.*\.csv$',
                                         combine_duplicates_by_name: bool = False,
                                         is_print_message: bool = False) -> None:
        
        input_csv_paths = []
        if isinstance(input_paths_or_folder, str) and os.path.isdir(input_paths_or_folder):
            if is_print_message: print(f"Searching for files in folder: '{input_paths_or_folder}' with pattern '{file_pattern}'")
            for root, _, files in os.walk(input_paths_or_folder):
                for filename in sorted(files): # Sort for consistent processing order
                    if re.match(file_pattern, filename, re.IGNORECASE):
                        input_csv_paths.append(os.path.join(root, filename))
            if is_print_message: print(f"Found {len(input_csv_paths)} files to merge.")
        elif isinstance(input_paths_or_folder, list):
            input_csv_paths = input_paths_or_folder
        else:
            if is_print_message: print(f"Error: 'input_paths_or_folder' must be a path to a folder or a list of file paths.")
            return

        if not input_csv_paths:
            if is_print_message: print("No files found to merge.")
            return
        
        dataframes_list = []
        all_columns_ever_seen = set()

        for file_path in input_csv_paths:
            try:
                if os.path.getsize(file_path) > 0: # Check if file is not empty
                    df = pd.read_csv(file_path, dtype=str) # Read all as string to avoid type issues
                    dataframes_list.append(df)
                    all_columns_ever_seen.update(df.columns)
                elif is_print_message:
                    print(f"  Skipping empty file: {file_path}")
            except FileNotFoundError:
                if is_print_message: print(f"  Warning: File not found {file_path}. Skipping.")
            except Exception as e:
                if is_print_message: print(f"  Error reading file '{file_path}': {e}. Skipping.")

        if not dataframes_list:
            if is_print_message: print("No DataFrames to merge (all files were empty, unreadable, or not found).")
            # Create empty CSV with max possible columns if desired, or just return
            # pd.DataFrame(columns=list(all_columns_ever_seen)).to_csv(output_csv_path, index=False, encoding='utf-8')
            return
            
        merged_df = pd.concat(dataframes_list, ignore_index=True, sort=False)
        if is_print_message:
            print(f"Initially merged {len(dataframes_list)} DataFrames into {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

        if combine_duplicates_by_name:
            if is_print_message: print("Combining duplicates by 'name' column...")
            if 'name' in merged_df.columns and not merged_df.empty:
                merged_df['name'] = merged_df['name'].astype(str)
                def first_valid_value_in_group(series):
                    for value in series:
                        if pd.notna(value) and str(value).strip() != '':
                            return value
                    return pd.NA

                # Ensure all columns that are not 'name' are passed to agg
                agg_funcs = {col: first_valid_value_in_group for col in merged_df.columns if col != 'name'}
                merged_df = merged_df.groupby('name', as_index=False).agg(agg_funcs)
                if is_print_message:
                    print(f"DataFrame after grouping by 'name': {merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")
            elif is_print_message:
                print("Warning: 'name' column not found or DataFrame empty. Cannot combine duplicates by 'name'.")
        
        # Custom column ordering
        final_ordered_columns = []
        if "name" in all_columns_ever_seen: final_ordered_columns.append("name")
        if "norm" in all_columns_ever_seen and "norm" not in final_ordered_columns: final_ordered_columns.append("norm")

        max_bus = NormalizationCoefficientGenerator._get_max_bus_number_from_columns(all_columns_ever_seen)
        if is_print_message and max_bus > 0: print(f"Max bus number determined from columns: {max_bus}")
        
        bus_types = ["Ub", "Uc", "Ip", "Iz"]
        bus_suffixes = ["_PS", "_base", "_h1", "_hx"]
        for i in range(1, max_bus + 1):
            for b_type in bus_types:
                for suffix in bus_suffixes:
                    col_name = f"{i}{b_type}{suffix}"
                    if col_name in all_columns_ever_seen and col_name not in final_ordered_columns:
                        final_ordered_columns.append(col_name)
        
        diff_current_cols = ["dId_PS", "dId_base", "dId_h1"] 
        for col_name in diff_current_cols:
            if col_name in all_columns_ever_seen and col_name not in final_ordered_columns:
                final_ordered_columns.append(col_name)
                
        processed_cols_set = set(final_ordered_columns)
        remaining_cols = sorted([col for col in all_columns_ever_seen if col not in processed_cols_set])
        final_ordered_columns.extend(remaining_cols)
        
        merged_df = merged_df.reindex(columns=final_ordered_columns)
        if is_print_message: print(f"Final column order set. Total columns: {len(final_ordered_columns)}.")

        if 'name' in merged_df.columns:
            if is_print_message: print("Sorting DataFrame by 'name' column...")
            merged_df['name'] = merged_df['name'].astype(str) # Ensure string type before sort
            merged_df.sort_values(by='name', ascending=True, inplace=True)
            
        output_dir_path = os.path.dirname(output_csv_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path, exist_ok=True)
                if is_print_message: print(f"Created output directory: {output_dir_path}")
            except OSError as e:
                if is_print_message: print(f"Error creating output directory {output_dir_path}: {e}. Cannot save.")
                return

        try:
            merged_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            if is_print_message: print(f"Merged normalization file saved to: '{output_csv_path}'")
        except Exception as e:
            if is_print_message: print(f"Error saving merged file to '{output_csv_path}': {e}")

    def update_normalization_coefficients(
        self,
        norm_coef_path: str,
        terminal_hashes_path: str,
        output_path: str,
        config: dict,
        max_bus_num_override: int = None
    ) -> None:
        """
        Корректирует файл с коэффициентами нормализации на основе хешей терминалов и конфигурации.

        Args:
            norm_coef_path (str): Путь к CSV-файлу с коэффициентами нормализации.
            terminal_hashes_path (str): Путь к JSON-файлу, где ключи - номера терминалов (строки),
                                        а значения - списки хешей осциллограмм.
                                        Пример: {"694": ["hash1", "hash2"], ...}
            output_path (str): Путь для сохранения обновленного CSV-файла.
            config (dict): Словарь с настройками для обновления.
            max_bus_num_override (int, optional): Максимальный номер секции для обработки.
                                                Если None, будет определен автоматически.
        """
        try:
            norm_df = pd.read_csv(norm_coef_path, dtype=str) # Читаем все как строки во избежание проблем с типами
            # Позже нужные значения (base_value) будут преобразованы при записи
        except FileNotFoundError:
            print(f"Ошибка: Файл коэффициентов нормализации не найден: {norm_coef_path}")
            return
        except Exception as e:
            print(f"Ошибка при чтении файла '{norm_coef_path}': {e}")
            return

        try:
            with open(terminal_hashes_path, 'r', encoding='utf-8') as f:
                terminal_hashes_data = json.load(f)
        except FileNotFoundError:
            print(f"Ошибка: JSON-файл с хешами терминалов не найден: {terminal_hashes_path}")
            return
        except json.JSONDecodeError:
            print(f"Ошибка: Не удалось декодировать JSON из файла: {terminal_hashes_path}")
            return
        except Exception as e:
            print(f"Ошибка при чтении файла '{terminal_hashes_path}': {e}")
            return

        # Создаем "обратный" словарь: {hash: terminal_number_str}
        hash_to_terminal_map = {}
        for term_num_str, hashes_list in terminal_hashes_data.items():
            for h in hashes_list:
                if h not in hash_to_terminal_map: # Первый встреченный терминал для хеша "выигрывает"
                    hash_to_terminal_map[h] = term_num_str
                # Можно добавить логику, если один хеш может быть у нескольких терминалов и это нужно учитывать
                # например, собирать список терминалов:
                else:
                    if isinstance(hash_to_terminal_map[h], list):
                        if term_num_str not in hash_to_terminal_map[h]:
                            hash_to_terminal_map[h].append(term_num_str)
                    else: # был один, станет список
                        if hash_to_terminal_map[h] != term_num_str:
                           hash_to_terminal_map[h] = [hash_to_terminal_map[h], term_num_str]

        if not hash_to_terminal_map:
            print("Предупреждение: В JSON-файле с хешами терминалов не найдено хешей для сопоставления.")
            # Решаем, что делать: выйти или сохранить копию исходного файла
            try:
                norm_df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Файл '{output_path}' сохранен без изменений, так как не было хешей для обработки.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_path}': {e}")
            return

        # Определяем максимальный номер секции
        if max_bus_num_override is not None:
            max_bus = max_bus_num_override
        else:
            max_bus = NormalizationCoefficientGenerator._get_max_bus_number_from_columns(set(norm_df.columns))
        
        print(f"Определен максимальный номер секции для обработки: {max_bus}")

        updated_rows_count = 0

        # Итерация по строкам DataFrame
        for index, row in tqdm(norm_df.iterrows(), total=norm_df.shape[0], desc="Обновление коэффициентов"):
            file_hash_name = row.get('name') # Столбец 'name' содержит хеши

            if file_hash_name and file_hash_name in hash_to_terminal_map:
                terminal_number_str = hash_to_terminal_map[file_hash_name]
                # Если хеш может принадлежать списку терминалов:
                # terminal_numbers = hash_to_terminal_map[file_hash_name]
                # terminal_number_str = terminal_numbers[0] if isinstance(terminal_numbers, list) else terminal_numbers

                updated_rows_count += 1

                # 1. Обновление графы "norm"
                norm_base_prefix = config.get("norm_base_prefix", "YES_MODIFIED") # Значение по умолчанию
                norm_df.at[index, 'norm'] = f"{norm_base_prefix}_{terminal_number_str}"

                # Обновление для каждой секции от 1 до max_bus
                for bus_idx in range(1, max_bus + 1):
                    # 2. Исправление токов (Ip)
                    current_config = config.get("currents", {})
                    if current_config.get("apply", False):
                        ps_val = current_config.get("ps_value")
                        base_val = current_config.get("base_value") # Может быть числом или строкой
                        
                        col_ps = f"{bus_idx}Ip_PS"
                        col_base = f"{bus_idx}Ip_base"
                        if col_ps in norm_df.columns:
                            norm_df.at[index, col_ps] = str(ps_val) # Приводим к строке для единообразия
                        if col_base in norm_df.columns:
                            norm_df.at[index, col_base] = str(base_val)

                    # 3. Исправление напряжений СШ (Ub)
                    vb_config = config.get("voltage_busbar", {})
                    if vb_config.get("apply", False):
                        ps_val = vb_config.get("ps_value")
                        base_val = vb_config.get("base_value")
                        
                        col_ps = f"{bus_idx}Ub_PS"
                        col_base = f"{bus_idx}Ub_base"
                        if col_ps in norm_df.columns:
                            norm_df.at[index, col_ps] = str(ps_val)
                        if col_base in norm_df.columns:
                            norm_df.at[index, col_base] = str(base_val)

                    # 4. Исправление напряжений КЛ (Uc)
                    vc_config = config.get("voltage_cableline", {})
                    if vc_config.get("apply", False):
                        ps_val = vc_config.get("ps_value")
                        base_val = vc_config.get("base_value")

                        col_ps = f"{bus_idx}Uc_PS"
                        col_base = f"{bus_idx}Uc_base"
                        if col_ps in norm_df.columns:
                            norm_df.at[index, col_ps] = str(ps_val)
                        if col_base in norm_df.columns:
                            norm_df.at[index, col_base] = str(base_val)
                    
                    # 5. Исправление токов нулевой последовательности (Iz)
                    rc_config = config.get("residual_currents", {})
                    if rc_config.get("apply", False):
                        ps_val = rc_config.get("ps_value")
                        base_val = rc_config.get("base_value")

                        col_ps = f"{bus_idx}Iz_PS"
                        col_base = f"{bus_idx}Iz_base"
                        if col_ps in norm_df.columns:
                            norm_df.at[index, col_ps] = str(ps_val)
                        if col_base in norm_df.columns:
                            norm_df.at[index, col_base] = str(base_val)
        
        if updated_rows_count == 0:
            print("Предупреждение: Ни одна строка в файле коэффициентов не соответствовала хешам из JSON. Файл не изменен.")
        else:
            print(f"Обновлено {updated_rows_count} строк.")

        # Создание директории для выходного файла, если она не существует
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): # output_dir может быть пустым, если путь - просто имя файла
            try:
                os.makedirs(output_dir)
                print(f"Создана директория: {output_dir}")
            except OSError as e:
                print(f"Ошибка при создании директории '{output_dir}': {e}. Файл может не сохраниться.")
                return

        try:
            norm_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Обновленный файл коэффициентов нормализации сохранен: {output_path}")
        except Exception as e:
            print(f"Ошибка при сохранении обновленного файла '{output_path}': {e}")

    def mark_missing_neutral_current_normalization(self,
                                                   input_csv_path: str,
                                                   neutral_current_hashes_txt_path: str,
                                                   output_csv_path: str,
                                                   max_bus_num_override: int = None
                                                   ) -> int:
        """
        Проверяет CSV-файл с коэффициентами нормализации.
        Если для осциллограммы, имеющей ток нулевой последовательности (из TXT-файла),
        все столбцы нормализации этого тока (XIz_PS, XIz_base, XIz_h1, XIz_hx) пусты,
        то в столбец '1Iz_PS' записывается "MISTAKE".

        Args:
            input_csv_path (str): Путь к исходному CSV-файлу нормализации.
            neutral_current_hashes_txt_path (str): Путь к TXT-файлу со списком хешей осциллограмм,
                                                   имеющих ток нулевой последовательности.
            output_csv_path (str): Путь для сохранения измененного CSV-файла.
            max_bus_num_override (int, optional): Принудительное задание максимального номера секции.
                                                Если None, будет определен из столбцов CSV.

        Returns:
            int: Количество найденных и помеченных ошибок (строк с "MISTAKE").
        """
        mistakes_found = 0

        try:
            df = pd.read_csv(input_csv_path, dtype=str) # Читаем все как строки
        except FileNotFoundError:
            print(f"Ошибка: CSV-файл не найден: {input_csv_path}")
            return 0
        except Exception as e:
            print(f"Ошибка при чтении CSV-файла '{input_csv_path}': {e}")
            return 0

        if 'name' not in df.columns:
            print(f"Ошибка: В CSV-файле '{input_csv_path}' отсутствует обязательный столбец 'name'.")
            return 0
        
        target_mistake_column = "1Iz_PS" # Целевой столбец для записи "MISTAKE"
        if target_mistake_column not in df.columns:
            print(f"Предупреждение: В CSV-файле '{input_csv_path}' отсутствует целевой столбец '{target_mistake_column}'. Он будет создан.")
            df[target_mistake_column] = pd.NA # или np.nan, или '' в зависимости от предпочтений для пустых ячеек

        try:
            with open(neutral_current_hashes_txt_path, 'r', encoding='utf-8') as f:
                # Убираем пустые строки и лишние пробелы, если есть
                hashes_with_neutral_current = {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Ошибка: TXT-файл с хешами не найден: {neutral_current_hashes_txt_path}")
            return 0
        
        if not hashes_with_neutral_current:
            print(f"Предупреждение: TXT-файл '{neutral_current_hashes_txt_path}' пуст. Ошибки не будут помечены.")
            # Сохраняем копию исходного файла или ничего не делаем
            try:
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"Файл '{output_csv_path}' сохранен без изменений.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_csv_path}': {e}")
            return 0

        # Определяем максимальный номер секции
        if max_bus_num_override is not None:
            max_bus = max_bus_num_override
        else:
            # Используем self._get_max_bus_number_from_columns, который уже есть в классе
            max_bus = NormalizationCoefficientGenerator._get_max_bus_number_from_columns(set(df.columns))
        
        print(f"Анализ CSV. Максимальный номер секции для Iz: {max_bus}")

        # Формируем список столбцов Iz_* для проверки
        iz_columns_to_check = []
        # Суффиксы, как в self.generate_result_cols
        iz_suffixes = ["_PS", "_base", "_h1", "_hx"] 
        for i in range(1, max_bus + 1):
            for suffix in iz_suffixes:
                col_name = f"{i}Iz{suffix}"
                if col_name in df.columns:
                    iz_columns_to_check.append(col_name)
        
        if not iz_columns_to_check:
            print(f"Предупреждение: В CSV-файле не найдено столбцов для проверки нормализации тока нулевой последовательности (вида XIz_*).")
            # Сохраняем и выходим
            try:
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"Файл '{output_csv_path}' сохранен без изменений.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_csv_path}': {e}")
            return 0
        else:
             print(f"Столбцы для проверки Iz: {iz_columns_to_check}")


        # Итерация по строкам DataFrame
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Проверка нормализации Iz"):
            file_hash_name = row['name']

            if file_hash_name in hashes_with_neutral_current:
                all_iz_fields_empty = True
                for iz_col in iz_columns_to_check:
                    value = row.get(iz_col) # Используем .get для безопасности, хотя столбцы должны быть
                    # Проверка на NaN (для численных NaN), None, и пустую строку
                    if not (pd.isna(value) or value is None or str(value).strip() == ''):
                        all_iz_fields_empty = False
                        break
                
                if all_iz_fields_empty:
                    df.loc[index, target_mistake_column] = "MISTAKE"
                    mistakes_found += 1
        
        print(f"Найдено и помечено ошибок (отсутствие данных в XIz_* при наличии сигнала N): {mistakes_found}")

        # Сохранение результата
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Обновленный CSV-файл сохранен: {output_csv_path}")
        except Exception as e:
            print(f"Ошибка при сохранении обновленного CSV-файла '{output_csv_path}': {e}")
            
        return mistakes_found
    
    
    def filter_csv_by_folder_contents(
        self,
        input_csv_path: str,
        oscillogram_folder_path: str,
        output_csv_path: str,
        name_column: str = 'name',
        file_extension: str = '.cfg'
    ) -> None:
        """
        Фильтрует строки CSV-файла, оставляя только те,
        для которых значение в столбце 'name_column' соответствует
        имени файла (без 'file_extension') в 'oscillogram_folder_path'.

        Args:
            input_csv_path (str): Путь к исходному CSV-файлу.
            oscillogram_folder_path (str): Путь к папке с файлами осциллограмм.
            output_csv_path (str): Путь для сохранения отфильтрованного CSV-файла.
            name_column (str): Имя столбца в CSV, содержащего идентификаторы (имена файлов).
                            По умолчанию 'name'.
            file_extension (str): Расширение файлов в папке для проверки (например, '.cfg').
                                По умолчанию '.cfg'. Убедитесь, что он включает точку.
        """
        print(f"Начало фильтрации CSV: '{input_csv_path}'")
        print(f"Папка с осциллограммами: '{oscillogram_folder_path}'")
        print(f"Проверяемое расширение файла: '{file_extension}'")

        # 1. Проверка существования входных путей
        if not os.path.exists(input_csv_path):
            print(f"Ошибка: Входной CSV-файл не найден: '{input_csv_path}'")
            return
        if not os.path.isdir(oscillogram_folder_path):
            print(f"Ошибка: Папка с осциллограммами не найдена: '{oscillogram_folder_path}'")
            return

        # 2. Получение имен файлов из папки осциллограмм
        try:
            existing_files_basenames: Set[str] = set()
            for filename in os.listdir(oscillogram_folder_path):
                if filename.lower().endswith(file_extension.lower()):
                    basename = os.path.splitext(filename)[0]
                    existing_files_basenames.add(basename)
            
            if not existing_files_basenames:
                print(f"Предупреждение: В папке '{oscillogram_folder_path}' не найдено файлов с расширением '{file_extension}'.")
                # Если файлов нет, то итоговый CSV будет пуст или не изменится, если и входной пуст
        except OSError as e:
            print(f"Ошибка при чтении содержимого папки '{oscillogram_folder_path}': {e}")
            return
        
        print(f"Найдено {len(existing_files_basenames)} файлов с расширением '{file_extension}' в папке.")

        # 3. Чтение и фильтрация CSV
        try:
            # Читаем столбец 'name' как строку, чтобы избежать проблем с типами, если хеши похожи на числа
            df = pd.read_csv(input_csv_path, dtype={name_column: str})
        except FileNotFoundError: # Дополнительная проверка, хотя первая уже есть
            print(f"Ошибка: Входной CSV-файл не найден (повторно): '{input_csv_path}'")
            return
        except Exception as e:
            print(f"Ошибка при чтении CSV-файла '{input_csv_path}': {e}")
            return

        if name_column not in df.columns:
            print(f"Ошибка: Столбец '{name_column}' не найден в CSV-файле '{input_csv_path}'.")
            print(f"Доступные столбцы: {df.columns.tolist()}")
            return

        original_row_count = len(df)
        print(f"Исходное количество строк в CSV: {original_row_count}")

        # Фильтрация DataFrame
        # Убедимся, что сравнение происходит со строками, на случай если dtype не сработал идеально
        mask = df[name_column].astype(str).isin(existing_files_basenames)
        filtered_df = df[mask]
        
        filtered_row_count = len(filtered_df)
        removed_row_count = original_row_count - filtered_row_count
        print(f"Количество строк после фильтрации: {filtered_row_count}")
        print(f"Количество удаленных строк: {removed_row_count}")

        # 4. Сохранение результата
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir): # output_dir может быть пустым, если путь - просто имя файла
                os.makedirs(output_dir)
                print(f"Создана директория для выходного файла: '{output_dir}'")
            
            filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Отфильтрованный CSV-файл успешно сохранен: '{output_csv_path}'")
        except Exception as e:
            print(f"Ошибка при сохранении отфильтрованного CSV-файла '{output_csv_path}': {e}")

class NormOsc:
    # TODO: Подумать о том, что получается несколько "__init__"
    def __init__(self, norm_coef_file_path: str = 'norm_coef.csv', is_print_message: bool = False): # Add type hint and is_print_message
        self.norm_coef = pd.DataFrame() # Initialize as empty DataFrame
        self.is_print_message = is_print_message # Store for use in other methods if needed
        if os.path.exists(norm_coef_file_path):
            try:
                self.norm_coef = pd.read_csv(norm_coef_file_path, encoding='utf-8')
                if self.is_print_message:
                    print(f"OscillogramNormalizer: Successfully loaded normalization coefficients from {norm_coef_file_path}")
            except Exception as e:
                if self.is_print_message:
                    print(f"OscillogramNormalizer: Error loading normalization coefficients from {norm_coef_file_path}: {e}")
                # self.norm_coef remains an empty DataFrame
        elif self.is_print_message: # File does not exist
            print(f"OscillogramNormalizer: Warning - Normalization coefficient file not found: {norm_coef_file_path}. Normalizer will not be effective.")
                
    def normalize_bus_signals(self, bus_df: pd.DataFrame, file_identifier: str,
                              yes_prase: str = "YES", is_print_error: bool = False) -> pd.DataFrame | None:
        if self.norm_coef.empty:
            if is_print_error:
                print(f"OscillogramNormalizer: Normalization coefficients are not loaded. Cannot normalize {file_identifier}.")
            return None

        # Ensure 'name' column exists and then filter
        if "name" not in self.norm_coef.columns:
            if is_print_error:
                print(f"OscillogramNormalizer: 'name' column not found in normalization coefficients. Cannot normalize {file_identifier}.")
            return None

        norm_row_matches = self.norm_coef[self.norm_coef["name"].astype(str) == str(file_identifier)]

        if norm_row_matches.empty:
            if is_print_error:
                print(f"OscillogramNormalizer: No normalization coefficients found for {file_identifier}.")
            return None

        norm_row = norm_row_matches.iloc[0]

        norm_column_value = str(norm_row.get("norm", "")) # Get 'norm' status, default to empty string
        # Original logic: if "YES" (yes_prase) is not in "YES_MODIFIED_XYZ" (norm_column_value), then skip.
        if yes_prase not in norm_column_value:
            if is_print_error:
                print(f"OscillogramNormalizer: Normalization not permitted for {file_identifier} based on 'norm' column value ('{norm_column_value}').")
            return None

        for bus in range(1, 9):
            nominal_current_series = norm_row.get(f"{bus}Ip_base")
            if nominal_current_val_str is not None and pd.notna(nominal_current_val_str):
                try:
                    nominal_current = 20.0 * float(nominal_current_val_str)
                    if nominal_current == 0:
                        if is_print_error: print(f"OscillogramNormalizer: Zero nominal current for Ip, Bus {bus_idx} for {file_identifier}. Skipping Ip norm.")
                    else:
                        for phase in ['A', 'B', 'C']:
                            col_name = f'I | Bus-{bus_idx} | phase: {phase}' # Original column name format
                            if col_name in normalized_df.columns:
                                normalized_df[col_name] = normalized_df[col_name] / nominal_current
                except (ValueError, TypeError) as e:
                    if is_print_error: print(f"OscillogramNormalizer: Invalid base current for {file_identifier}, Bus {bus_idx} (Ip): '{nominal_current_val_str}' ({e})")

            # Residual current normalization (Iz)
            nominal_current_I0_val_str = norm_row.get(f"{bus_idx}Iz_base")
            if nominal_current_I0_val_str is not None and pd.notna(nominal_current_I0_val_str):
                try:
                    nominal_current_I0 = 5.0 * float(nominal_current_I0_val_str)
                    if nominal_current_I0 == 0:
                         if is_print_error: print(f"OscillogramNormalizer: Zero nominal current for Iz, Bus {bus_idx} for {file_identifier}. Skipping Iz norm.")
                    else:
                        col_name_N = f'I | Bus-{bus_idx} | phase: N'
                        if col_name_N in normalized_df.columns:
                            normalized_df[col_name_N] = normalized_df[col_name_N] / nominal_current_I0
                except (ValueError, TypeError) as e:
                    if is_print_error: print(f"OscillogramNormalizer: Invalid base I0 current for {file_identifier}, Bus {bus_idx} (Iz): '{nominal_current_I0_val_str}' ({e})")

            # Voltage BusBar normalization (Ub)
            nominal_voltage_bb_val_str = norm_row.get(f"{bus_idx}Ub_base")
            if nominal_voltage_bb_val_str is not None and pd.notna(nominal_voltage_bb_val_str):
                try:
                    nominal_voltage_bb = 3.0 * float(nominal_voltage_bb_val_str)
                    if nominal_voltage_bb == 0:
                        if is_print_error: print(f"OscillogramNormalizer: Zero nominal voltage for Ub, Bus {bus_idx} for {file_identifier}. Skipping Ub norm.")
                    else:
                        for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']:
                            col_name = f'U | BusBar-{bus_idx} | phase: {phase}'
                            if col_name in normalized_df.columns:
                                normalized_df[col_name] = normalized_df[col_name] / nominal_voltage_bb
                except (ValueError, TypeError) as e:
                     if is_print_error: print(f"OscillogramNormalizer: Invalid base BusBar voltage for {file_identifier}, Bus {bus_idx} (Ub): '{nominal_voltage_bb_val_str}' ({e})")

            # Voltage CableLine normalization (Uc)
            nominal_voltage_cl_val_str = norm_row.get(f"{bus_idx}Uc_base")
            if nominal_voltage_cl_val_str is not None and pd.notna(nominal_voltage_cl_val_str):
                try:
                    nominal_voltage_cl = 3.0 * float(nominal_voltage_cl_val_str)
                    if nominal_voltage_cl == 0:
                        if is_print_error: print(f"OscillogramNormalizer: Zero nominal voltage for Uc, Bus {bus_idx} for {file_identifier}. Skipping Uc norm.")
                    else:
                        for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']:
                            col_name = f'U | CableLine-{bus_idx} | phase: {phase}'
                            if col_name in normalized_df.columns:
                                normalized_df[col_name] = normalized_df[col_name] / nominal_voltage_cl
                except (ValueError, TypeError) as e:
                    if is_print_error: print(f"OscillogramNormalizer: Invalid base CableLine voltage for {file_identifier}, Bus {bus_idx} (Uc): '{nominal_voltage_cl_val_str}' ({e})")
            
        return normalized_df

if __name__ == '__main__':
    # --- Вызов класса ---
    osc_path='Путь к папке для обзора'
    #osc_path='raw_data/PDR/'
    prev_norm_csv_path = "Путь к имеющемуся файла csv"
    createNormOsc = CreateNormOsc(osc_path=osc_path, prev_norm_csv_path = prev_norm_csv_path)


    
    # ---!!! Вызов новой функции !!!---
    # ---createNormOsc---
    createNormOsc.normalization()
    
    
    
    
    # ---!!! Вызов новой функции !!!---
    # ---merge_normalization_files---
    # Объединение файлов нормализации
    norm_file_path_1="Путь к первому файлу с CSV значениями"
    norm_file_path_2="Путь ко второму файлу с CSV значениями"
    input_files = [
        norm_file_path_1,
        norm_file_path_2
    ]
    output_file_path = "Путь к новому файлу с CSV значениями"
    createNormOsc.merge_normalization_files(input_files, output_file_path)
    
    
    
    # ---!!! Вызов новой функции !!!---
    # ---update_normalization_coefficients---
    # Запись коэффциентов для заранее известных терминалов (по хеш-именам)
    # Конфигурация для обновления
    # Путь для выходного файла
    mock_norm_coef_path = "Путь к имеющемуся файлу с CSV значениями"
    output_updated_norm_coef_path = "Путь к новому файлу с CSV значениями"
    mock_terminal_hashes_path = "Путь к файлу JSON содержащем информацию о группировки осциллограмм по признакам"
    
    update_config = {
        "norm_base_prefix": "YES_SCM", # Префикс для столбца 'norm'
        "currents": {
            "apply": False,        # Надо ли вносить изменения?
            "ps_value": "s",      # Новое значение для xIp_PS
            "base_value": 0.1       # Новое значение для xIp_base (str или число)
        },
        "voltage_busbar": {        # Напряжения СШ (Ub)
            "apply": False,
            "ps_value": "s",
            "base_value": 100
        },
        "voltage_cableline": {     # Напряжения КЛ (Uc)
            "apply": False,
            "ps_value": "s",
            "base_value": 100
        },
        "residual_currents": {     # Токи нулевой последовательности (Iz)
            "apply": False,
            "ps_value": "",
            "base_value": ""
        }
    }

    print(f"\n--- Запуск функции update_normalization_coefficients для тестовых данных ---")
    createNormOsc.update_normalization_coefficients(
        norm_coef_path=mock_norm_coef_path,
        terminal_hashes_path=mock_terminal_hashes_path,
        output_path=output_updated_norm_coef_path,
        config=update_config
    )
    

    
    
    # ---!!! Вызов новой функции !!!---
    # ---mark_missing_neutral_current_normalization---
    mock_norm_coef_path = "Путь к имеющемуся файлу с CSV значениями"
    neutral_current_hashes_file = "Путь к файлу с найденныеми именами осциллограмм содержащих ТТНП"
    output_updated_norm_coef_path = "Путь к новому файлу с CSV значениями"
    num_mistakes = createNormOsc.mark_missing_neutral_current_normalization(
        input_csv_path=mock_norm_coef_path,
        neutral_current_hashes_txt_path=neutral_current_hashes_file,
        output_csv_path=output_updated_norm_coef_path
    )
    print(f"Итоговое количество помеченных ошибок: {num_mistakes}")   
