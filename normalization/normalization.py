import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import comtrade
from typing import Dict, Any, Tuple, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from dataflow.comtrade_processing import ReadComtrade

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

class CreateNormOsc:
    def __init__(self,
                 osc_path,
                 prev_norm_csv_path = "",
                 step_size=1,
                 bus = 6
                 ):
        self.osc_path = osc_path
        self.readComtrade = ReadComtrade()
        self.prev_norm_csv_path = prev_norm_csv_path
        self.window_size = 32
        self.step_size = step_size
       
        self.ru_cols, self.u_cols = self.generate_VT_cols(bus=bus)
        self.ri_cols, self.i_cols = self.generate_CT_cols(bus=bus)
        self.riz_cols, self.iz_cols = self.generate_rCT_cols(bus=bus)
        self.rid_cols, self.id_cols = self.generate_dI_cols(1)
        
        self.raw_cols = self.generate_raw_cols(bus=bus)
        
        self.result_cols = self.generate_result_cols()
        
        self.all_features = self.generate_all_features()

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
        
        for c in ru_cols:
            base = f'{name} | BusBar-' + c[0][0] + ' | phase: '
            u_cols[c[0]] = {base + 'A', base + 'B', base + 'C', base + 'N', 
                            base + 'AB', base + 'BC', base + 'CA'}
            cable = f'{name} | CableLine-' + c[1][0] + ' | phase: '
            u_cols[c[1]] = {cable + 'A', cable + 'B', cable + 'C', cable + 'N', 
                        cable + 'AB', cable + 'BC', cable + 'CA'}
        
        ru_cols = [item for sublist in ru_cols for item in sublist]
        return ru_cols, u_cols

    def generate_CT_cols(self, bus = 6, isRaw = False): # CT - current transformer
        ri_cols = [f"{i}Ip" for i in range(1, bus+1)]
        i_cols = dict()
        name = "I"
        if isRaw:
            name = "I_raw"
        
        for c in ri_cols:
            phase = f'{name} | Bus-' + c[0] + ' | phase: '
            i_cols[c] = {phase + 'A', phase + 'B', phase + 'C'}
            
        return ri_cols, i_cols

    def generate_rCT_cols(self, bus = 6): # Residual current transformer (rCT)
        riz_cols = [f"{i}Iz" for i in range(1, bus+1)]
        iz_cols = dict()
        for c in riz_cols:
            iz_cols[c] = 'I | Bus-' + c[0] + ' | phase: N'
            
        return riz_cols, iz_cols

    def generate_dI_cols(self, bus = 1): # different current (for protection - 87)
        rid_cols = [f"{i}Id" for i in range(1, bus+1)]
        id_cols = dict()
        for c in rid_cols:
            phase_d = 'I | dif-' + c[0] + ' | phase: '
            phase_b = 'I | breaking-' + c[0] + ' | phase: '
            id_cols[c] = {phase_d + 'A', phase_d + 'B', phase_d + 'C',
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

    def analyze(self,
                file: str,
                h1_df: pd.DataFrame,
                hx_df: pd.DataFrame,
                coef_p_s: Dict[str, float]) -> pd.DataFrame:
        """
        Анализирует данные одной осциллограммы (гармоники h1 и hx)
        и определяет базовые значения и тип подключения (первичка/вторичка).

        Args:
            file: Имя файла осциллограммы (без расширения).
            h1_df: DataFrame с амплитудами первой гармоники.
            hx_df: DataFrame с максимальными амплитудами высших гармоник (>=2).
            coef_p_s: Словарь с коэффициентами трансформации (первичка/вторичка).

        Returns:
            Однострочный DataFrame с результатами анализа.
        """
        features = set(h1_df.columns) # Используем set для быстрого пересечения
        results: Dict[str, Any] = {'name': file[:-4], 'norm': 'YES'} # Собираем результаты в словарь

        # --- Обработка напряжения ---
        for r_prefix in self.ru_cols:
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

    def normalization(self, bus = 6, isSaveOnlyNewFilese = False):
        name_prev_norm = []
        if self.prev_norm_csv_path != "":
            prev_norm_csv = pd.read_csv(self.prev_norm_csv_path)
            name_prev_norm = prev_norm_csv["name"].values
        
        unread = []
        result_df = pd.DataFrame(data=self.result_cols)
        for file in tqdm(self.osc_files):
            name_osc = file[:-4]
            if name_osc in name_prev_norm:
                continue
            
            try:
                raw_date, osc_df = self.readComtrade.read_comtrade(self.osc_path + file)
                if raw_date == None or raw_date == None:
                    unread.append(file)
                    continue
                frequency = raw_date.cfg.frequency
                samples_rate = raw_date.cfg.sample_rates[0][0]
                self.window_size = int(samples_rate / frequency)
            except:
                unread.append(file)
                continue
            
            coef_p_s = {}
            for analog_channel in raw_date.cfg.analog_channels:
                coef = analog_channel.primary / analog_channel.secondary
                coef_p_s[analog_channel.name] = coef
            
            osc_columns = osc_df.columns
            osc_features = []
            to_drop = []
            for column in osc_columns:
                if column in self.all_features:
                    osc_features.append(column)
                else:
                    to_drop.append(column)
            osc_df.drop(columns=to_drop, inplace=True)
            osc_fft = np.abs(np.fft.fft(osc_df.iloc[:self.window_size],axis=0))
            for i in range(self.window_size, osc_df.shape[0], self.step_size):
                window_fft = np.abs(np.fft.fft(osc_df.iloc[i - self.window_size: i],axis=0))
                osc_fft = np.maximum(osc_fft, window_fft)
            h1 = 2 * osc_fft[1] / self.window_size
            harmonic_count = self.window_size // 2
            if harmonic_count >= 2:
                hx = 2 * np.max(osc_fft[2:harmonic_count+1], axis=0) / self.window_size
            else:
                hx = 0
            h1_df = pd.DataFrame([dict(zip(osc_features, h1))])
            hx_df = pd.DataFrame([dict(zip(osc_features, hx))])
            result = self.analyze(file, h1_df, hx_df, coef_p_s)
            result_df = pd.concat([result_df, result])

        result_df.to_csv('normalization/norm.csv', index=False)

class NormOsc:
    # TODO: Подумать о том, что получается несколько "__init__"
    def __init__(self, norm_coef_file_path='norm_coef.csv'):
        if os.path.exists(norm_coef_file_path):
            with open(norm_coef_file_path, "r") as file:
                self.norm_coef = pd.read_csv(file, encoding='utf-8')
                
    # TODO: подумать об унификации данной вещи, пока это локальная реализация
    # Пока прост скопировал из raw_to_csv
    def normalize_bus_signals(self, raw_df, file_name, yes_prase = "YES", is_print_error = False):
        """Нормализация аналоговых сигналов для каждой секции."""
        norm_row = self.norm_coef[self.norm_coef["name"] == file_name] # Поиск строки нормализации по имени файла
        if norm_row.empty or norm_row["norm"].values[0] != yes_prase: # Проверка наличия строки и разрешения на нормализацию
            if is_print_error:
                print(f"Предупреждение: {file_name} не найден в файле norm.csv или нормализация не разрешена.")
            return None

        for bus in range(1, 9):
            nominal_current_series = norm_row.get(f"{bus}Ip_base")
            if nominal_current_series is not None and not pd.isna(nominal_current_series.values[0]):
                nominal_current = 20 * float(nominal_current_series.values[0])
                for phase in ['A', 'B', 'C']: # Нормализация токов
                    current_col_name = f'I | Bus-{bus} | phase: {phase}'
                    if current_col_name in raw_df.columns:
                        raw_df[current_col_name] = raw_df[current_col_name] / nominal_current

            nominal_current_I0_series = norm_row.get(f"{bus}Iz_base")
            if nominal_current_I0_series is not None and not pd.isna(nominal_current_I0_series.values[0]):
                nominal_current_I0 = 5 * float(nominal_current_I0_series.values[0])
                for phase in ['N']: # Нормализация тока нулевой последовательности
                    current_I0_col_name = f'I | Bus-{bus} | phase: {phase}'
                    if current_I0_col_name in raw_df.columns:
                        raw_df[current_I0_col_name] = raw_df[current_I0_col_name] / nominal_current_I0

            nominal_voltage_bb_series = norm_row.get(f"{bus}Ub_base")
            if nominal_voltage_bb_series is not None and not pd.isna(nominal_voltage_bb_series.values[0]):
                nominal_voltage_bb = 3 * float(nominal_voltage_bb_series.values[0])
                for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # Нормализация напряжений BusBar
                    voltage_bb_col_name = f'U | BusBar-{bus} | phase: {phase}'
                    if voltage_bb_col_name in raw_df.columns:
                        raw_df[voltage_bb_col_name] = raw_df[voltage_bb_col_name] / nominal_voltage_bb

            nominal_voltage_cl_series = norm_row.get(f"{bus}Uc_base")
            if nominal_voltage_cl_series is not None and not pd.isna(nominal_voltage_cl_series.values[0]):
                nominal_voltage_cl = 3 * float(nominal_voltage_cl_series.values[0])
                for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # Нормализация напряжений CableLine
                    voltage_cl_col_name = f'U | CableLine-{bus} | phase: {phase}'
                    if voltage_cl_col_name in raw_df.columns:
                        raw_df[voltage_cl_col_name] = raw_df[voltage_cl_col_name] / nominal_voltage_cl

            # TODO: Добавить дифференциальный ток
            
        return raw_df
