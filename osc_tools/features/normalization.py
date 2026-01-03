import os
import sys
import polars as pl
import numpy as np
from tqdm.auto import tqdm
from osc_tools.core.comtrade_custom import Comtrade
from typing import Dict, Any, Tuple, List, Optional
import re # Для регулярных выражений
from typing import List, Set, Union
import json
from tqdm import tqdm # Для визуализации прогресса

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from osc_tools.data_management.comtrade_processing import ReadComtrade

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
                 prev_norm_csv_path = "", # TODO: На будущее требуется убрать подобные переменные как обязательные, они мешают внутренним функциям
                 step_size=1,
                 bus = 6
                 ):
        self.osc_path = osc_path
        self.readComtrade = ReadComtrade()
        self.existing_norm_df = pl.DataFrame() # Инициализация пустым DataFrame
        if prev_norm_csv_path and os.path.exists(prev_norm_csv_path):
            try:
                self.existing_norm_df = pl.read_csv(prev_norm_csv_path, infer_schema_length=0) # Читаем все как строки для начала
                print(f"Загружены существующие коэффициенты из: {prev_norm_csv_path}")
            except Exception as e:
                print(f"Ошибка при загрузке существующих коэффициентов из '{prev_norm_csv_path}': {e}")
                # existing_norm_df останется пустым, обработка пойдет как без старых данных
        elif prev_norm_csv_path:
            print(f"Предупреждение: Файл существующих коэффициентов '{prev_norm_csv_path}' не найден.")
        self.window_size = 32
        self.step_size = step_size
       
        self.ru_cols, self.u_cols = self.generate_VT_cols(bus=bus)
        self.ri_cols, self.i_cols = self.generate_CT_cols(bus=bus)
        self.riz_cols, self.iz_cols = self.generate_rCT_cols(bus=bus)
        self.rid_cols, self.id_cols = self.generate_dI_cols(1)
        
        self.raw_cols = self.generate_raw_cols(bus=bus)
        
        self.result_cols = self.generate_result_cols(bus=bus)
        
        self.all_features = self.generate_all_features(bus=bus)

        self.df = pl.DataFrame(self.result_cols)

        self.osc_files = sorted([file for file in os.listdir(self.osc_path)
                            if "cfg" in file], reverse=False)
    
    def get_summary(self,):
        unread = []
        c = 0
        for file in tqdm(self.osc_files):
            try:
                rec = Comtrade()
                rec.load(self.osc_path + file)
                osc_df = rec.to_dataframe()
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
                               h1_df: pl.DataFrame,
                               columns: List[str],
                               coef_p_s: Dict[str, float]) -> float:
        """Вычисляет максимальное первичное значение для заданных столбцов."""
        max_primary = 0.0
        # Используем [0, name] для доступа к значению в однострочном DataFrame
        for name in columns:
            if name in h1_df.columns and name in coef_p_s:
                primary_value = h1_df[0, name] * coef_p_s[name]
                if not np.isnan(primary_value):
                    max_primary = max(max_primary, primary_value)
        return max_primary
    
    def _determine_voltage_status(self,
                                  m1: float,
                                  mx: float,
                                  h1_df: pl.DataFrame,
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
                                h1_df: pl.DataFrame,
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
            if value is None or str(value).strip() == '' or (isinstance(value, float) and np.isnan(value)):
                return False # Нашли отсутствующее или пустое значение, расчет нужен
        return True # Все значения на месте и не пусты, можно пропустить расчет
    
    def analyze(self,
                file: str,
                h1_df: pl.DataFrame,
                hx_df: pl.DataFrame,
                coef_p_s: Dict[str, float],
                existing_row: pl.DataFrame = pl.DataFrame()
                ) -> pl.DataFrame:
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
        if not existing_row.is_empty():
            # Преобразуем строку DataFrame в словарь, обрабатывая возможные NaN/None
            # fillna('') - чтобы не было проблем с типами при последующем сравнении или записи
            results = existing_row.fill_null('').row(0, named=True)
            results = dict(results)
            results['name'] = file[:-4] # Убедимся, что имя файла корректное
            # Статус 'norm' может быть переопределен позже, так что начальное значение из existing_row нормально
        else:
            results: Dict[str, Any] = {'name': file[:-4], 'norm': 'YES'} # Собираем результаты в словарь

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

            m1 = h1_df.select(relevant_cols).max_horizontal().item(0)
            mx = hx_df.select(relevant_cols).max_horizontal().item(0)

            if m1 is None or np.isnan(m1): # Пропускаем, если основные данные некорректны
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

            m1 = h1_df.select(relevant_cols).max_horizontal().item(0)
            mx = hx_df.select(relevant_cols).max_horizontal().item(0)

            if m1 is None or np.isnan(m1):
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
            m1 = h1_df[0, col_name]
            mx = hx_df[0, col_name]


            if m1 is None or np.isnan(m1):
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

            m1 = h1_df.select(relevant_cols).max_horizontal().item(0)

            if m1 is None or np.isnan(m1):
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

        return pl.DataFrame([results])

    def normalization(self, bus = 6):     
        unread = []
        result_df = pl.DataFrame(self.result_cols)
        for file in tqdm(self.osc_files):
            name_osc = file[:-4]
            existing_row_for_osc = pl.DataFrame() # Пустой DataFrame по умолчанию
            if not self.existing_norm_df.is_empty() and 'name' in self.existing_norm_df.columns:
                # Убедимся, что сравниваем строки, если name_osc - хеш, который может быть числом
                match = self.existing_norm_df.filter(pl.col('name').cast(pl.Utf8) == str(name_osc))
                if not match.is_empty():
                    existing_row_for_osc = match.head(1) # Берем первую строку, если вдруг дубликаты имен
            
            try:
                raw_date, osc_df = self.readComtrade.read_comtrade(self.osc_path + file)
                if raw_date is None or osc_df is None:
                    unread.append(file)
                    continue
                frequency = raw_date.cfg.frequency
                samples_rate = raw_date.cfg.sample_rates[0][0]
                self.window_size = int(samples_rate / frequency)

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
                
                osc_df = osc_df.drop(to_drop)
                osc_numpy = osc_df.to_numpy()

                osc_fft = np.abs(np.fft.fft(osc_numpy[:self.window_size],axis=0))
                for i in range(self.window_size, osc_numpy.shape[0], self.step_size):
                    window_fft = np.abs(np.fft.fft(osc_numpy[i - self.window_size: i],axis=0))
                    osc_fft = np.maximum(osc_fft, window_fft)
                h1 = 2 * osc_fft[1] / self.window_size
                harmonic_count = self.window_size // 2
                if harmonic_count >= 2:
                    hx = 2 * np.max(osc_fft[2:harmonic_count+1], axis=0) / self.window_size
                else:
                    hx = 0
                
                current_features = osc_df.columns
                h1_df = pl.DataFrame([dict(zip(current_features, h1))])
                hx_df = pl.DataFrame([dict(zip(current_features, hx))])
                
                result = self.analyze(file, h1_df, hx_df, coef_p_s, existing_row=existing_row_for_osc)
                result_df = pl.concat([result_df, result], how='vertical')
            except:
                # TODO: написать разбор более обширный, ибо ошибки могут быть разные.
                unread.append(file)
                continue


        result_df.write_csv('normalization/norm.csv')
    
    def merge_normalization_files(
        input_paths_or_folder: Union[str, List[str]],
        output_csv_path: str,
        file_pattern: str = r'^norm_.*\.csv$',
        combine_duplicates_by_name: bool = False # Новый параметр
    ) -> None:
        """
        Объединяет несколько CSV-файлов нормализации в один.
        Может принимать как список путей, так и путь к папке, где файлы ищутся по шаблону.

        Порядок столбцов:
        1. "name", "norm"
        2. Группы для каждой секции (1..N): {i}Ub_*, {i}Uc_*, {i}Ip_*, {i}Iz_* 
        (суффиксы в порядке: _PS, _base, _h1, _hx)
        3. Дифференциальные токи: dId_PS, dId_base, dId_h1
        4. Остальные столбцы в алфавитном порядке.

        Args:
            input_paths_or_folder (Union[str, List[str]]): Путь к папке или список путей к файлам.
            output_csv_path (str): Путь для сохранения объединенного CSV.
            file_pattern (str): Регулярное выражение для поиска файлов в папке.
            combine_duplicates_by_name (bool): Если True, строки с одинаковым значением
                                               в столбце 'name' будут объединены,
                                               при этом для остальных столбцов будет взято
                                               первое непустое значение из группы.
                                               По умолчанию False (простое добавление всех строк).
        """
        
        input_csv_paths = []
        if isinstance(input_paths_or_folder, str) and os.path.isdir(input_paths_or_folder):
            print(f"Поиск файлов в папке: '{input_paths_or_folder}' по шаблону '{file_pattern}'")
            for root, _, files in os.walk(input_paths_or_folder):
                for filename in sorted(files):
                    if re.match(file_pattern, filename, re.IGNORECASE):
                        input_csv_paths.append(os.path.join(root, filename))
            print(f"Найдено файлов для объединения: {len(input_csv_paths)}")
        elif isinstance(input_paths_or_folder, (list, tuple, set, np.ndarray)):
            input_csv_paths = input_paths_or_folder
        else:
            print(f"Ошибка: 'input_paths_or_folder' должен быть путём к папке или списком строк.")
            return

        if not input_csv_paths:
            print("Не найдено файлов для объединения.")
            return
        
        dataframes_list = []
        all_columns_ever_seen = set()

        for file_path in input_csv_paths:
            try:
                if os.path.getsize(file_path) > 0:
                    df = pl.read_csv(file_path, infer_schema_length=0)
                    dataframes_list.append(df)
                    all_columns_ever_seen.update(df.columns)
            except Exception as e:
                print(f"Ошибка при чтении файла '{file_path}': {e}. Файл пропускается.")

        if not dataframes_list:
            print("Не найдено DataFrame'ов для объединения (после фильтрации пустых/ошибочных файлов).")
            merged_df = pl.DataFrame()
        else:
            # Сначала объединяем все DataFrame'ы в один DataFrame.
            merged_df = pl.concat(dataframes_list, how="diagonal")
            print(f"\nПредварительно объединены {len(dataframes_list)} DataFrame'ов в "
                f"{merged_df.height} строк и {merged_df.width} столбцов.")

            if combine_duplicates_by_name:
                print("Активирован режим объединения дубликатов по столбцу 'name'.")
                if 'name' in merged_df.columns and not merged_df.is_empty():
                    # Убедимся, что столбец 'name' имеет строковый тип для надежной группировки.
                    merged_df = merged_df.with_columns(pl.col('name').cast(pl.Utf8))

                    # Заменяем пустые строки на null для корректной работы drop_nulls
                    merged_df = merged_df.with_columns([
                        pl.col(c).cast(pl.Utf8).str.strip_chars().replace("", None) 
                        for c in merged_df.columns
                    ])
                    
                    # Агрегируем: берем первое не-null значение
                    exprs = [pl.col(c).drop_nulls().first().alias(c) for c in merged_df.columns if c != 'name']
                    merged_df = merged_df.group_by('name').agg(exprs)
                    
                    print(f"DataFrame после группировки и слияния дубликатов: {merged_df.shape[0]} строк (уникальные 'name') "
                        f"и {merged_df.shape[1]} столбцов.")
                
                elif not merged_df.empty:
                    print("Предупреждение: Столбец 'name' не найден в объединенных данных или DataFrame пуст. "
                        "Объединение дубликатов по 'name' не выполнено. Данные остаются просто объединенными.")
                # Если merged_df пуст, ничего дополнительно делать не нужно.
            
            else: # combine_duplicates_by_name is False (стандартное поведение)
                print("Стандартный режим объединения DataFrame'ов (без слияния дубликатов по 'name').")
                # merged_df уже содержит результат простого объединения.

        print(f"Всего уникальных столбцов, обнаруженных во всех файлах: {len(all_columns_ever_seen)}")

        # Формирование кастомного порядка столбцов
        final_ordered_columns = []
        
        # 1. Обязательные первые столбцы
        if "name" in all_columns_ever_seen:
            final_ordered_columns.append("name")
        if "norm" in all_columns_ever_seen and "norm" not in final_ordered_columns:
            # Проверка "not in" на случай, если 'norm' как-то совпадет с 'name' (маловероятно)
            final_ordered_columns.append("norm")

        # 2. Группы по секциям
        max_bus = CreateNormOsc._get_max_bus_number_from_columns(all_columns_ever_seen)
        if max_bus > 0:
            print(f"Определен максимальный номер секции: {max_bus}")
        
        # Порядок типов и суффиксов соответствует CreateNormOsc.generate_result_cols
        bus_types = ["Ub", "Uc", "Ip", "Iz"]
        bus_suffixes = ["_PS", "_base", "_h1", "_hx"]
        
        for i in range(1, max_bus + 1): # Итерация по номерам секций
            for b_type in bus_types: # Для каждого типа измерений (U шин, U кабеля, Ток, Ток НП)
                for suffix in bus_suffixes: # Для каждого типа данных (_PS, _base, _h1, _hx)
                    col_name = f"{i}{b_type}{suffix}"
                    if col_name in all_columns_ever_seen and col_name not in final_ordered_columns:
                        final_ordered_columns.append(col_name)
        
        # 3. Дифференциальные токи
        # Порядок и набор соответствуют CreateNormOsc.generate_result_cols
        diff_current_cols = ["dId_PS", "dId_base", "dId_h1"] 
        for col_name in diff_current_cols:
            if col_name in all_columns_ever_seen and col_name not in final_ordered_columns:
                final_ordered_columns.append(col_name)
                
        # 4. Остальные столбцы (не попавшие в шаблон), отсортированные по алфавиту
        processed_cols_set = set(final_ordered_columns)
        remaining_cols = sorted([col for col in all_columns_ever_seen if col not in processed_cols_set])
        final_ordered_columns.extend(remaining_cols)
        
        # Применяем новый порядок столбцов.
        # В Polars для "reindex" мы добавляем недостающие столбцы как null и выбираем нужный порядок.
        missing_cols = [pl.lit(None).alias(c) for c in final_ordered_columns if c not in merged_df.columns]
        if missing_cols:
            merged_df = merged_df.with_columns(missing_cols)
        
        merged_df = merged_df.select(final_ordered_columns)

        print(f"Порядок столбцов определен. Итоговое количество столбцов: {len(final_ordered_columns)}.")

        # Сортировка данных по столбцу 'name'
        if 'name' in merged_df.columns:
            print("Сортировка данных по столбцу 'name'...")
            # Перед сортировкой убедимся, что 'name' имеет строковый тип для консистентности
            merged_df = merged_df.with_columns(pl.col('name').cast(pl.String))
            merged_df = merged_df.sort('name')
        else:
            # Эта ситуация маловероятна, если файлы создаются CreateNormOsc, но для полноты
            print("Предупреждение: Столбец 'name' отсутствует в объединенных данных. Сортировка данных не выполнена.")
            
        # Создание директории для выходного файла, если она не существует
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir): # output_dir может быть пустым, если путь - просто имя файла
            try:
                os.makedirs(output_dir)
                print(f"Создана директория: {output_dir}")
            except OSError as e:
                print(f"Ошибка при создании директории '{output_dir}': {e}. Файл может не сохраниться.")
                return # Выходим, если не можем создать директорию

        # Сохранение результата
        try:
            merged_df.write_csv(output_csv_path)
            print(f"Объединенный файл успешно сохранен: '{output_csv_path}'")
        except Exception as e:
            print(f"Ошибка при сохранении объединенного файла '{output_csv_path}': {e}")
    
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
            norm_df = pl.read_csv(norm_coef_path, infer_schema_length=0) # Читаем все как строки во избежание проблем с типами
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
                norm_df.write_csv(output_path)
                print(f"Файл '{output_path}' сохранен без изменений, так как не было хешей для обработки.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_path}': {e}")
            return

        # Определяем максимальный номер секции
        if max_bus_num_override is not None:
            max_bus = max_bus_num_override
        else:
            max_bus = CreateNormOsc._get_max_bus_number_from_columns(set(norm_df.columns))
        
        print(f"Определен максимальный номер секции для обработки: {max_bus}")

        updated_rows_count = 0

        # Создаем DataFrame для обновления
        updates_data = []
        for h, t in hash_to_terminal_map.items():
             t_str = str(t) # Handle list or string
             updates_data.append({'name': h, 'terminal_number': t_str})
        
        updates_df = pl.DataFrame(updates_data)
        
        # Присоединяем информацию о терминалах
        if 'name' in norm_df.columns:
            norm_df = norm_df.join(updates_df, on='name', how='left')
        else:
            print("Ошибка: столбец 'name' не найден в файле коэффициентов.")
            return

        # Маска для строк, которые нужно обновить (где есть terminal_number)
        has_terminal = pl.col('terminal_number').is_not_null()
        
        # Подсчет обновленных строк
        updated_rows_count = norm_df.filter(has_terminal).height

        # 1. Обновление графы "norm"
        norm_base_prefix = config.get("norm_base_prefix", "YES_MODIFIED")
        if 'norm' in norm_df.columns:
            norm_df = norm_df.with_columns(
                pl.when(has_terminal)
                .then(pl.format("{}_{}", pl.lit(norm_base_prefix), pl.col('terminal_number')))
                .otherwise(pl.col('norm'))
                .alias('norm')
            )

        # Обновление для каждой секции
        for bus_idx in range(1, max_bus + 1):
            # Helper function to apply updates
            def apply_update(conf_key, suffix_ps, suffix_base):
                nonlocal norm_df
                conf = config.get(conf_key, {})
                if conf.get("apply", False):
                    ps_val = str(conf.get("ps_value"))
                    base_val = str(conf.get("base_value"))
                    
                    col_ps = f"{bus_idx}{suffix_ps}"
                    col_base = f"{bus_idx}{suffix_base}"
                    
                    if col_ps in norm_df.columns:
                        norm_df = norm_df.with_columns(
                            pl.when(has_terminal).then(pl.lit(ps_val)).otherwise(pl.col(col_ps)).alias(col_ps)
                        )
                    if col_base in norm_df.columns:
                        norm_df = norm_df.with_columns(
                            pl.when(has_terminal).then(pl.lit(base_val)).otherwise(pl.col(col_base)).alias(col_base)
                        )
                return norm_df

            # 2. Исправление токов (Ip)
            norm_df = apply_update("currents", "Ip_PS", "Ip_base")
            # 3. Исправление напряжений СШ (Ub)
            norm_df = apply_update("voltage_busbar", "Ub_PS", "Ub_base")
            # 4. Исправление напряжений КЛ (Uc)
            norm_df = apply_update("voltage_cableline", "Uc_PS", "Uc_base")
            # 5. Исправление токов нулевой последовательности (Iz)
            norm_df = apply_update("residual_currents", "Iz_PS", "Iz_base")

        # Удаляем вспомогательный столбец
        norm_df = norm_df.drop('terminal_number')

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
            norm_df.write_csv(output_path)
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
            df = pl.read_csv(input_csv_path, infer_schema_length=0) # Читаем все как строки
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
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(target_mistake_column))

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
                df.write_csv(output_csv_path)
                print(f"Файл '{output_csv_path}' сохранен без изменений.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_csv_path}': {e}")
            return 0

        # Определяем максимальный номер секции
        if max_bus_num_override is not None:
            max_bus = max_bus_num_override
        else:
            # Используем self._get_max_bus_number_from_columns, который уже есть в классе
            max_bus = CreateNormOsc._get_max_bus_number_from_columns(set(df.columns))
        
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
                df.write_csv(output_csv_path)
                print(f"Файл '{output_csv_path}' сохранен без изменений.")
            except Exception as e:
                print(f"Ошибка при сохранении файла '{output_csv_path}': {e}")
            return 0
        else:
             print(f"Столбцы для проверки Iz: {iz_columns_to_check}")

        # Создаем выражение для проверки пустоты всех столбцов Iz
        # Столбец считается пустым, если он null или пустая строка (после strip)
        # В Polars read_csv(infer_schema_length=0) пустые поля могут быть null или ""
        
        # Выражение: (col is null) OR (col.str.strip_chars() == "")
        # Мы хотим проверить, что ВСЕ столбцы пустые.
        # all_empty = (col1_empty) & (col2_empty) & ...
        
        all_iz_empty_expr = pl.lit(True)
        for col_name in iz_columns_to_check:
            is_empty = pl.col(col_name).is_null() | (pl.col(col_name).str.strip_chars() == "")
            all_iz_empty_expr = all_iz_empty_expr & is_empty

        # Фильтр по хешам
        # Создаем DataFrame с хешами для join или используем is_in
        # is_in работает хорошо для списков
        is_target_hash = pl.col('name').is_in(hashes_with_neutral_current)

        # Условие ошибки: хеш в списке И все поля Iz пустые
        is_mistake = is_target_hash & all_iz_empty_expr

        # Подсчет ошибок
        mistakes_found = df.filter(is_mistake).height
        
        if mistakes_found > 0:
            df = df.with_columns(
                pl.when(is_mistake)
                .then(pl.lit("MISTAKE"))
                .otherwise(pl.col(target_mistake_column))
                .alias(target_mistake_column)
            )

        print(f"Найдено и помечено ошибок (отсутствие данных в XIz_* при наличии сигнала N): {mistakes_found}")

        # Сохранение результата
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.write_csv(output_csv_path)
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
            df = pl.read_csv(input_csv_path, schema_overrides={name_column: pl.String})
        except FileNotFoundError: # Дополнительная проверка, хотя первая уже есть
            print(f"Ошибка: Входной CSV-файл не найден (повторно): '{input_csv_path}'")
            return
        except Exception as e:
            print(f"Ошибка при чтении CSV-файла '{input_csv_path}': {e}")
            return

        if name_column not in df.columns:
            print(f"Ошибка: Столбец '{name_column}' не найден в CSV-файле '{input_csv_path}'.")
            print(f"Доступные столбцы: {df.columns}")
            return

        original_row_count = len(df)
        print(f"Исходное количество строк в CSV: {original_row_count}")

        # Фильтрация DataFrame
        # Убедимся, что сравнение происходит со строками, на случай если dtype не сработал идеально
        filtered_df = df.filter(pl.col(name_column).cast(pl.String).is_in(existing_files_basenames))
        
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
            
            filtered_df.write_csv(output_csv_path)
            print(f"Отфильтрованный CSV-файл успешно сохранен: '{output_csv_path}'")
        except Exception as e:
            print(f"Ошибка при сохранении отфильтрованного CSV-файла '{output_csv_path}': {e}")

class NormOsc:
    # TODO: Подумать о том, что получается несколько "__init__"
    def __init__(self, norm_coef_file_path='norm_coef.csv'):
        if os.path.exists(norm_coef_file_path):
            self.norm_coef = pl.read_csv(norm_coef_file_path)
        else:
            self.norm_coef = None
                
    # TODO: подумать об унификации данной вещи, пока это локальная реализация
    # Пока прост скопировал из raw_to_csv
    def normalize_bus_signals(self, raw_df: pl.DataFrame, file_name: str, yes_prase: str = "YES", is_print_error: bool = False) -> Optional[pl.DataFrame]:
        """Нормализация аналоговых сигналов для каждой секции."""
        if self.norm_coef is None:
            return None

        norm_row = self.norm_coef.filter(pl.col("name") == file_name) # Поиск строки нормализации по имени файла
        if norm_row.is_empty() or yes_prase not in str(norm_row.get_column("norm")[0]): # Проверка наличия строки и разрешения на нормализацию
            if is_print_error:
                print(f"Предупреждение: {file_name} не найден в файле norm.csv или нормализация не разрешена.")
            return None

        new_cols = {}
        for bus in range(1, 9):
            # Номинальный ток
            col_name_ip = f"{bus}Ip_base"
            if col_name_ip in norm_row.columns:
                val = norm_row.get_column(col_name_ip)[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    nominal_current = 20 * float(val)
                    for phase in ['A', 'B', 'C']: # Нормализация токов
                        current_col_name = f'I | Bus-{bus} | phase: {phase}'
                        if current_col_name in raw_df.columns:
                            new_cols[current_col_name] = pl.col(current_col_name) / nominal_current

            # Номинальный ток I0
            col_name_iz = f"{bus}Iz_base"
            if col_name_iz in norm_row.columns:
                val = norm_row.get_column(col_name_iz)[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    nominal_current_I0 = 5 * float(val)
                    for phase in ['N']: # Нормализация тока нулевой последовательности
                        current_I0_col_name = f'I | Bus-{bus} | phase: {phase}'
                        if current_I0_col_name in raw_df.columns:
                            new_cols[current_I0_col_name] = pl.col(current_I0_col_name) / nominal_current_I0

            # Номинальное напряжение BB
            col_name_ub = f"{bus}Ub_base"
            if col_name_ub in norm_row.columns:
                val = norm_row.get_column(col_name_ub)[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    nominal_voltage_bb = 3 * float(val)
                    for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # Нормализация напряжений BusBar
                        voltage_bb_col_name = f'U | BusBar-{bus} | phase: {phase}'
                        if voltage_bb_col_name in raw_df.columns:
                            new_cols[voltage_bb_col_name] = pl.col(voltage_bb_col_name) / nominal_voltage_bb
                        
                        voltage_short_name = f'U{phase} BB'
                        if voltage_short_name in raw_df.columns:
                            new_cols[voltage_short_name] = pl.col(voltage_short_name) / nominal_voltage_bb

            # Номинальное напряжение CL
            col_name_uc = f"{bus}Uc_base"
            if col_name_uc in norm_row.columns:
                val = norm_row.get_column(col_name_uc)[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    nominal_voltage_cl = 3 * float(val)
                    for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # Нормализация напряжений CableLine
                        voltage_cl_col_name = f'U | CableLine-{bus} | phase: {phase}'
                        if voltage_cl_col_name in raw_df.columns:
                            new_cols[voltage_cl_col_name] = pl.col(voltage_cl_col_name) / nominal_voltage_cl
                        
                        voltage_short_name = f'U{phase} CL'
                        if voltage_short_name in raw_df.columns:
                            new_cols[voltage_short_name] = pl.col(voltage_short_name) / nominal_voltage_cl

        if new_cols:
            return raw_df.with_columns(**new_cols)
        return raw_df

            # TODO: Добавить дифференциальный ток
            
        return raw_df

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
