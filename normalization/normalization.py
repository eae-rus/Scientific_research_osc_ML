import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import comtrade

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from dataflow.comtrade_processing import ReadComtrade

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

    def analyze(self,
                file, h1_df, hx_df):
        features = h1_df.columns
        df = pd.DataFrame({'name': [file[:-4]]})
        df['norm'] = 'YES'

        # voltage
        t1 = 20 * np.sqrt(2)
        t2 = 140 * np.sqrt(2)
        t3 = 560 * np.sqrt(2)
        for r in self.ru_cols:
            c = list(self.u_cols[r].intersection(features))
            m1 = h1_df[c].max(axis=1)[0]
            mx = hx_df[c].max(axis=1)[0]
            if np.isnan(m1):
                continue
            df[r + '_h1'] = m1
            df[r + '_hx'] = mx
            # TODO: add a check for primary values
            if m1 <= t1:
                if m1 <= 1.5 * mx:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = 'Noise'
                else: 
                    df[r + '_PS'] = 'p'
                    df[r + '_base'] = '?1'#None
            elif t1 < m1 <= t2:
                if m1 <= 1.5 * mx:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = '?2'
                else:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = 100
            elif t2 < m1 <= t3:
                if m1 <= 1.5 * mx:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = '?2'
                else:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = 400
            else:
                df[r + '_PS'] = '?3'
                df[r + '_base'] = '?3'
                
        # current
        t1 = 0.03 * np.sqrt(2)
        t2 = 30 * np.sqrt(2)
        for r in self.ri_cols:
            c = list(self.i_cols[r].intersection(features))
            m1 = h1_df[c].max(axis=1)[0]
            mx = hx_df[c].max(axis=1)[0]
            if np.isnan(m1):
                continue
            df[r + '_h1'] = m1
            df[r + '_hx'] = mx
            if m1 <= t1:
                if m1 <= 1.5 * mx:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = 'Noise'
                else:
                    df[r + '_PS'] = 'p'
                    df[r + '_base'] = '?1'
            elif t1 < m1 <= t2:
                if m1 <= 1.5 * mx:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = '?2'
                else:
                    df[r + '_PS'] = 's'
                    df[r + '_base'] = 5
            else:
                df[r + '_PS'] = '?3'
                df[r + '_base'] = '?3'
        
        # residual current    
        t1 = 0.02 * np.sqrt(2)
        t2 = 5 * np.sqrt(2)
        for r in self.riz_cols:
            c_r = set([self.iz_cols[r]])
            c = list(c_r.intersection(features))
            m1 = h1_df[c].max(axis=1)[0]
            mx = hx_df[c].max(axis=1)[0]
            if np.isnan(m1):
                continue
            
            df[r + 'Iz_h1'] = m1
            df[r + 'Iz_hx'] = mx
            if m1 <= t1 or mx <= t1:
                df[r + 'Iz_PS'] = 's'
                df[r + 'Iz_base'] = 'Noise'
            elif t1 < m1 <= t2 or t1 < mx <= t2:
                df[r + 'Iz_PS'] = 's'
                df[r + 'Iz_base'] = '1'
            else:
                df[r + 'Iz_PS'] = '?3'
                df[r + 'Iz_base'] = '?3'
        
        for r in self.rid_cols:
            c = list(self.id_cols[r].intersection(features))
            m1 = h1_df[c].max(axis=1)[0]
            if np.isnan(m1):
                continue
            
            df['dId_h1'] = m1
            if m1 < 30:
                df['dId_PS'] = 's'
                df['dId_base'] = 1
            else:
                df['dId_PS'] = '?3'
                df['dId_base'] = '?3'
                
                
        if 'Noise' in df.values:
            df['norm'] = 'hz' # Шум
        if '?1' in df.values or '?2' in df.values or '?3' in df.values:
            df['norm'] = 'NO' # Непонятно, требуется разобраться
        if list(self.raw_cols.intersection(features)):
            df['norm'] = 'raw'
        # if file[:-4] in self.raw_files:
        #     df['norm'] = 'raw'
        return df 

    def normalization(self, bus = 6, isSaveOnlyNewFilese = False):
        name_prev_norm = []
        if prev_norm_csv_path != "":
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
            except:
                unread.append(file)
                continue
            frequency = raw_date.cfg.frequency
            samples_rate = raw_date.cfg.sample_rates[0][0]
            self.window_size = int(samples_rate / frequency)
            
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
            result = self.analyze(file, h1_df, hx_df)
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