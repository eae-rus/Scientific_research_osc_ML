import os
import sys
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

from osc_tools.io.comtrade_parser import ComtradeParser
from osc_tools.ml import models as model
from torchinfo import summary
from ptflops import get_model_complexity_info
from osc_tools.core.constants import Features

class ComtradeProcessor(ComtradeParser):
    """Базовый класс для обработки данных Comtrade."""
    
    def __init__(self, norm_file_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.norm_csv = pd.read_csv(norm_file_path)
        self.FRAME_SIZE = 64
    
    def load_model(self, model_path: str):
        """Загрузка обученной модели."""
        self.model = torch.load(model_path, map_location=self.device).to(self.device)
    
    def normalize_signals(self, df_osc: pd.DataFrame, name_osc: str):
        """Нормализация токов и напряжений."""
        try:
            # Поддержка обоих форматов: 'Bus 2' (старый) и 'Bus-2' (новый)
            bus_part = name_osc.split("_")[1]
            if ' ' in bus_part:
                # Старый формат: "Bus 2"
                bus = int(bus_part.split(" ")[1])
            elif '-' in bus_part:
                # Новый формат: "Bus-2"
                bus = int(bus_part.split("-")[1])
            else:
                raise ValueError(f"Неизвестный формат имени шины: {bus_part}")
        except (ValueError, IndexError):
            print(f"Предупреждение: {name_osc} не найден в norm_osc_tuple.")
            return df_osc
        
        name_osc_i_bus = name_osc.split("_")[0]
        norm_osc_row = self.norm_csv[self.norm_csv["name"] == name_osc_i_bus]
        
        if norm_osc_row["norm"].values[0] != "ДА":
            return df_osc  # Пропускаем необученные наборы
        
        normalization_params = {
            "current": {
                "nominal": 20 * float(norm_osc_row[f"{bus}Ip_base"]),
                "features": Features.CURRENT
            },
            "voltage_bb": {
                "nominal": 3 * float(norm_osc_row[f"{bus}Ub_base"]),
                "features": Features.VOLTAGE_PHAZE_BB + [Features.VOLTAGE_ZERO_SEQ[0]]
            },
            "voltage_cl": {
                "nominal": 3 * float(norm_osc_row[f"{bus}Uc_base"]),
                "features": Features.VOLTAGE_PHAZE_CL + [Features.VOLTAGE_ZERO_SEQ[1]] + Features.VOLTAGE_LINE_CL
            }
        }
        
        for key, params in normalization_params.items():
            for feature in params["features"]:
                if feature in df_osc.columns:
                    df_osc[feature] = df_osc[feature] / params["nominal"]
        
        return df_osc

class MarkingUpOscillograms(ComtradeProcessor):
    """Класс для разметки осциллограмм."""

    def __init__(self, norm_file_path: str, device: str = 'cuda', batch_size: int = 10000):
        super().__init__(norm_file_path, device)
        self.FRAME_SIZE = 64
        self.threshold = 0.8
        self.batch_size = batch_size

    def search_events_in_comtrade(
        self,
        csv_name: str = 'marking_up_oscillograms/sorted_files.csv',
        ML_model_path: str = 'ML_model/trained_models/model_training.pt',
        norm_file_path: str = 'marking_up_oscillograms/norm_1600.csv'
    ):
        """Процесс разметки осциллограмм с использованием машинной модели."""
        self.load_model(ML_model_path)
        ml_for_answer = ["file_name", "bus"]
        for name in Features.TARGET:
            ml_for_answer.append(f"{name}_bool")
            ml_for_answer.append(f"{name}_count")
        sorted_files_df = pd.DataFrame(columns=ml_for_answer)

        # Установка типов данных для логических столбцов
        for name in Features.TARGET:
            sorted_files_df[f"{name}_bool"] = sorted_files_df[f"{name}_bool"].astype(bool)

        norm_osc_tuple = tuple(self.norm_csv["name"].values)
        raw_files = sorted([file[:-4] for file in os.listdir(self.raw_path) if 'cfg' in file])
        raw_files = [file for file in raw_files if file in norm_osc_tuple]

        with tqdm(total=len(raw_files), desc="Преобразование Comtrade в CSV") as pbar:
            for file in raw_files:
                df = self.create_one_df(os.path.join(self.raw_path, f"{file}.cfg"), f"{file}.cfg")
                if df.empty:
                    pbar.update(1)
                    continue

                names_osc = df["file_name"].unique()
                for name_osc in names_osc:
                    df_osc_one_bus = df[df["file_name"] == name_osc].copy()

                    # Инициализация DataFrame с нулевыми значениями
                    df_osc = pd.DataFrame(0, index=np.arange(len(df_osc_one_bus)), columns=Features.ALL)

                    # Заполнение реальными данными, если они есть
                    for feature in Features.ALL:
                        if feature in df_osc_one_bus.columns:
                            df_osc[feature] = df_osc_one_bus[feature].copy()

                    # Нормализация сигналов
                    df_osc = self.normalize_signals(df_osc, name_osc)

                    indexes = len(df_osc) - self.FRAME_SIZE + 1
                    df_osc.fillna(0, inplace=True)

                    # Инициализация предсказаний
                    predict_labels = {name: np.zeros(len(df_osc)) for name in Features.TARGET}

                    # Создание окон
                    windows = []
                    target_indices = []
                    for ind in range(indexes):
                        window = df_osc.iloc[ind:ind + self.FRAME_SIZE].values.astype(np.float32)
                        windows.append(window)
                        # Определение индекса для присвоения предсказания
                        target_idx = self.FRAME_SIZE - 8 + ind
                        if 0 <= target_idx < len(predict_labels[list(predict_labels.keys())[0]]):
                            target_indices.append(target_idx)
                        else:
                            target_indices.append(None)  # Индекс вне диапазона

                    # Преобразование окон в пакеты и передача их через модель
                    batch_size = self.batch_size
                    for i in range(0, len(windows), batch_size):
                        batch_windows = windows[i:i + batch_size]
                        batch_target_indices = target_indices[i:i + batch_size]

                        batch_tensor = torch.tensor(batch_windows).to(self.device).float()  # Форма: (batch, FRAME_SIZE, features)
                        #batch_tensor = batch_tensor.permute(0, 2, 1)  # Вам может потребоваться переставить оси (batch, features, FRAME_SIZE)
                        with torch.no_grad():
                            model_prediction = self.model(batch_tensor)

                        predictions = (model_prediction > self.threshold).int().cpu().numpy()

                        for j, name in enumerate(Features.TARGET):
                            for k, target_idx in enumerate(batch_target_indices):
                                if target_idx is not None:
                                    predict_labels[name][target_idx] = predictions[k][j]

                    # Применение двойного прохода (слева и справа)
                    left_predict_labels = {name: np.zeros(len(df_osc)) for name in Features.TARGET}
                    right_predict_labels = {name: np.zeros(len(df_osc)) for name in Features.TARGET}

                    for name in Features.TARGET:
                        # Слева направо
                        for i in range(len(predict_labels[name]) - 6):
                            window_sum = np.sum(predict_labels[name][i:i + 7]) > 4
                            if predict_labels[name][i] and (window_sum > 4):
                                left_predict_labels[name][i:i + 7] = 1

                        # Справа налево
                        for i in range(len(predict_labels[name]) - 6):
                            window_sum = np.sum(predict_labels[name][-i-8:-i-1])
                            if predict_labels[name][-i-1] and(window_sum > 4):
                                right_predict_labels[name][-i-8:-i-1] = 1

                        # Объединение предсказаний
                        predict_labels[name] = np.logical_and(left_predict_labels[name], right_predict_labels[name])

                    # Подсчет событий
                    count_events = {name: np.sum(predict_labels[name]) for name in Features.TARGET}

                    # Создание новой строки для DataFrame
                    events_predicted = {}
                    for name in Features.TARGET:
                        events_predicted[f"{name}_bool"] = count_events[name] > 0
                        events_predicted[f"{name}_count"] = count_events[name]

                    # безопасная попытка определить секцию
                    bus = 1
                    try:
                        # Поддержка обоих форматов: 'Bus 2' (старый) и 'Bus-2' (новый)
                        bus_part = name_osc.split("_")[1]
                        if ' ' in bus_part:
                            # Старый формат: "Bus 2"
                            bus = int(bus_part.split(" ")[1])
                        elif '-' in bus_part:
                            # Новый формат: "Bus-2"
                            bus = int(bus_part.split("-")[1])
                    except (ValueError, IndexError):
                        pass

                    new_row = pd.DataFrame({
                        "file_name": [name_osc.split("_")[0]],
                        "bus": [bus],
                        **{k: [v] for k, v in events_predicted.items()}
                    })

                    sorted_files_df = pd.concat([sorted_files_df, new_row], ignore_index=True)

                sorted_files_df.to_csv(csv_name, index=False)
                pbar.update(1)

class ComtradePredictionAndPlotting(ComtradeProcessor):
    """Класс для прогнозирования событий и построения графиков."""

    def __init__(
        self, 
        osc_name: str, 
        uses_bus: str, 
        strat_point: int, 
        end_point: int, 
        norm_file_path: str, 
        model_path: str = None, 
        use_model: bool = True,
        device: str = 'cuda', 
        f_networks: int = 50, 
        ADC_sampling_rate: int = 1600, 
        batch_size: int = 10000,
        font_size: int = 11,
        font_family: str = 'DejaVu Sans',
        language: str = 'ru'  # 'ru' или 'en'
    ):
        super().__init__(norm_file_path, device)
        
        # Параметры модели
        self.use_model = use_model
        if use_model:
            if model_path is None:
                raise ValueError("model_path должен быть указан, когда use_model=True")
            self.load_model(model_path)
        else:
            self.model = None
        
        # Параметры осциллограммы
        self.osc_name = osc_name
        self.uses_bus = uses_bus
        self.start_point = strat_point
        self.end_point = end_point
        self.FRAME_SIZE = 64
        self.threshold = 0.8
        self.font_size = font_size
        self.font_family = font_family
        self.language = language  # 'ru' или 'en'
        
        self.ml_signals, self.ml_operational_switching, self.ml_abnormal_event, self.ml_emergency_event = self.get_short_names_ml_signals()

        self.f_networks, self.ADC_sampling_rate = f_networks, ADC_sampling_rate
        self.batch_size = batch_size
        self.len_df = 0
        
        # Словари для переключения языка
        self._setup_labels()

    def _setup_labels(self):
        """Настройка текстовых меток в зависимости от языка."""
        if self.language == 'en':
            self.labels = {
                'title_prefix': 'Oscillogram:',
                'currents_label': 'Currents (A)',
                'voltages_label': 'Voltages (V)',
                'discrete_label': 'Discrete signals',
                'time_label': 'Time (ms)',
                'prediction_label': 'Prediction',
                'real_label': 'Real',
                'opr_swch': 'Operational switching',
                'abnorm_evnt': 'Abnormal event',
                'emerg_evnt': 'Emergency event'
            }
        else:  # 'ru' по умолчанию
            self.labels = {
                'title_prefix': 'Осциллограмма:',
                'currents_label': 'Токи (А)',
                'voltages_label': 'Напряжения (В)',
                'discrete_label': 'Дискретные сигналы',
                'time_label': 'Время (мс)',
                'prediction_label': 'Прогноз',
                'real_label': 'Реальный',
                'opr_swch': 'Операт. переключение',
                'abnorm_evnt': 'Аномалия',
                'emerg_evnt': 'Авария'
            }

    def predict(self):
        """Применение модели для прогнозирования на основе входных данных.
        Если use_model=False, возвращает нулевые предсказания и реальные метки."""
        df = self.create_one_df(os.path.join(self.raw_path, f"{self.osc_name}.cfg"), f"{self.osc_name}.cfg")
        if df.empty:
            return np.nan, np.nan, np.nan

        names_osc = df["file_name"].unique()
        try:
            name_osc = names_osc[int(self.uses_bus) - 1]
        except IndexError:
            print(f"Ошибка: Шина {self.uses_bus} не найдена в именах файлов.")
            return np.nan, np.nan, np.nan

        df_osc_one_bus = df[df["file_name"] == name_osc].copy()
        self.len_df = len(df_osc_one_bus)
        analog_signal = {name: np.zeros(self.len_df) for name in Features.ANALOG_SIGNALS_FOR_PLOT}

        # Инициализация DataFrame с нулевыми значениями
        df_osc = pd.DataFrame(0, index=np.arange(self.len_df), columns=Features.ALL)

        # Заполнение реальными данными, если они есть
        for feature in Features.ALL:
            if feature in df_osc_one_bus.columns:
                df_osc[feature] = df_osc_one_bus[feature].copy()
                if feature in Features.ANALOG_SIGNALS_FOR_PLOT:
                    analog_signal[feature] = df_osc_one_bus[feature].copy()

        # Нормализация сигналов
        df_osc = self.normalize_signals(df_osc, name_osc)

        # Здесь происходит получение реальной разметки
        df_osc_target = pd.DataFrame()
        for feature in self.ml_signals:
            if feature in df_osc_one_bus.columns:
                df_osc_target[feature] = df_osc_one_bus[feature].copy()
        real_labels = self.get_real_labels(df_osc_target)

        # Если модель не используется, возвращаем нулевые предсказания
        if not self.use_model:
            predict_labels = {name: np.zeros(self.len_df) for name in Features.TARGET}
            return analog_signal, predict_labels, real_labels

        # =========== ПРОЦЕСС РАБОТЫ С МОДЕЛЬЮ ===========
        indexes = self.len_df - self.FRAME_SIZE + 1
        df_osc.fillna(0, inplace=True)

        # Инициализация предсказаний
        predict_labels = {name: np.zeros(self.len_df) for name in Features.TARGET}

        # Создание окон
        windows = []
        target_indices = []
        for ind in range(indexes):
            window = df_osc.iloc[ind:ind + self.FRAME_SIZE].values.astype(np.float32)
            windows.append(window)
            # Определение индекса для присвоения предсказания
            target_idx = self.FRAME_SIZE - 8 + ind
            if 0 <= target_idx < len(predict_labels[list(predict_labels.keys())[0]]):
                target_indices.append(target_idx)
            else:
                target_indices.append(None)  # Индекс вне диапазона

        # Преобразование окон в пакеты и передача их через модель
        batch_size = self.batch_size
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_target_indices = target_indices[i:i + batch_size]

            batch_tensor = torch.tensor(batch_windows).to(self.device).float()  # Форма: (batch, FRAME_SIZE, features)
            #batch_tensor = batch_tensor.permute(0, 2, 1)  # Вам может потребоваться переставить оси (batch, features, FRAME_SIZE)
            with torch.no_grad():
                model_prediction = self.model(batch_tensor)

            predictions = (model_prediction > self.threshold).int().cpu().numpy()

            for j, name in enumerate(Features.TARGET):
                for k, target_idx in enumerate(batch_target_indices):
                    if target_idx is not None:
                        predict_labels[name][target_idx] = predictions[k][j]

        # Применение двойного прохода (слева и справа)
        left_predict_labels = {name: np.zeros(self.len_df) for name in Features.TARGET}
        right_predict_labels = {name: np.zeros(self.len_df) for name in Features.TARGET}

        for name in Features.TARGET:
            # Слева направо
            for i in range(self.len_df - 6):
                window_sum = np.sum(predict_labels[name][i:i + 7])
                if predict_labels[name][i] and (window_sum > 3):
                    left_predict_labels[name][i:i + 7] = 1

            # Справа налево
            for i in range(self.len_df - 6):
                window_sum = np.sum(predict_labels[name][-i-8:-i-1])
                if predict_labels[name][-i-1] and(window_sum > 3):
                    right_predict_labels[name][-i-8:-i-1] = 1

            # Объединение предсказаний
            predict_labels[name] = np.logical_and(left_predict_labels[name], right_predict_labels[name])

        # Возврат результатов
        return analog_signal, predict_labels, real_labels

    def get_real_labels(self, df: pd.DataFrame):
        """Получение реальной разметки событий на основе файла.
        
        Поддерживает оба формата имён сигналов:
        - Краткие имена: 'ML_1', 'ML_2', 'ML_3' (из self.ml_*) 
        - Полные имена: 'MLsignal_2_1', 'MLsignal_2_2', 'MLsignal_2_3' (из файла)
        """
        real_labels = {name: np.zeros(self.len_df) for name in Features.TARGET}
        
        # Проверяем, есть ли в файле полные имена (MLsignal_*) или краткие (ML_*)
        df_columns = set(df.columns)
        
        # opr_swch - операционные переключения (ML_1*)
        for name_column in self.ml_operational_switching:
            if name_column in df_columns:
                real_labels["opr_swch"] = np.logical_or(real_labels["opr_swch"], df[name_column].fillna(0))
        
        # Если краткие имена не найдены, ищем полные имена в файле
        if not any(col in df_columns for col in self.ml_operational_switching):
            for col in df_columns:
                if 'MLsignal' in col and '_1' in col and '_1_1' not in col and '_1_2' not in col:
                    # Это ML_1 (операционное переключение)
                    real_labels["opr_swch"] = np.logical_or(real_labels["opr_swch"], df[col].fillna(0))
        
        # abnorm_evnt - аномальные события (ML_2*)
        for name_column in self.ml_abnormal_event:
            if name_column in df_columns:
                real_labels["abnorm_evnt"] = np.logical_or(real_labels["abnorm_evnt"], df[name_column].fillna(0))
        
        # Если краткие имена не найдены, ищем полные имена
        if not any(col in df_columns for col in self.ml_abnormal_event):
            for col in df_columns:
                if 'MLsignal' in col and '_2' in col and '_2_1' not in col and '_2_2' not in col and '_2_3' not in col and '_2_4' not in col:
                    # Это ML_2 (аномалия без уточнения)
                    real_labels["abnorm_evnt"] = np.logical_or(real_labels["abnorm_evnt"], df[col].fillna(0))
                elif 'MLsignal' in col and ('_2_' in col):
                    # Это любой ML_2_* (более специфичные типы аномалий)
                    real_labels["abnorm_evnt"] = np.logical_or(real_labels["abnorm_evnt"], df[col].fillna(0))
        
        # emerg_evnt - аварийные события (ML_3*)
        for name_column in self.ml_emergency_event:
            if name_column in df_columns:
                real_labels["emerg_evnt"] = np.logical_or(real_labels["emerg_evnt"], df[name_column].fillna(0))
        
        # Если краткие имена не найдены, ищем полные имена
        if not any(col in df_columns for col in self.ml_emergency_event):
            for col in df_columns:
                if 'MLsignal' in col and '_3' in col:
                    # Это ML_3* (аварийные события)
                    real_labels["emerg_evnt"] = np.logical_or(real_labels["emerg_evnt"], df[col].fillna(0))
        
        return real_labels

    def plot_predictions_vs_real(
        self, 
        start: int, 
        end: int, 
        analog_signal: dict, 
        predict_labels: dict, 
        real_labels: dict
    ):
        """Визуализация осциллограмм прогнозов и реальной разметки.
        
        Поддерживает отображение как с предсказаниями модели, так и без них.
        Поддерживает русский и английский языки.
        """
        # Установка стиля шрифта
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['font.family'] = self.font_family
        
        time_range = np.arange(start, end) * 1000 / self.ADC_sampling_rate  # мс (период дискретизации = 1000 / ADC_sampling_rate)
        
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f'{self.labels["title_prefix"]} {self.osc_name}, Bus: {self.uses_bus}', 
                     fontsize=self.font_size + 2, fontweight='bold', fontfamily=self.font_family)
        fig.subplots_adjust(hspace=0.35)
        
        # Data Frame для сохранения всех данных
        data = pd.DataFrame({self.labels['time_label']: time_range})
        
        # ========== ПОСТРОЕНИЕ ГРАФИКОВ ТОКОВ ==========
        current_signals = [sig for sig in Features.CURRENT_FOR_PLOT if sig in analog_signal]
        for i, current in enumerate(current_signals):
            signal_data = analog_signal[current][start:end]
            color = ['#FFD700', '#00AA00', '#FF0000'][i % 3]
            axs[0].plot(time_range, signal_data, label=f'{current}', color=color, linewidth=1.5)
            data[current] = analog_signal[current][start:end]
        
        axs[0].legend(loc='upper right', fontsize=self.font_size, framealpha=0.9)
        axs[0].set_ylabel(self.labels['currents_label'], rotation=90, labelpad=12, fontsize=self.font_size, fontweight='bold')
        axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        axs[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axs[0].set_facecolor('#f9f9f9')
        for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
            label.set_fontsize(self.font_size)
        
        # ========== ПОСТРОЕНИЕ ГРАФИКОВ НАПРЯЖЕНИЙ ==========
        # Показываем напряжения BB и CL
        voltage_signals = [sig for sig in Features.VOLTAGE_FOR_PLOT + ['UA CL', 'UB CL', 'UC CL'] if sig in analog_signal]
        for i, voltage in enumerate(voltage_signals):
            signal_data = analog_signal[voltage][start:end]
            color = ['#FFD700', '#00AA00', '#FF0000'][i % 3]
            axs[1].plot(time_range, signal_data, label=f'{voltage}', color=color, linewidth=1.5)
            data[voltage] = analog_signal[voltage][start:end]
        
        axs[1].legend(loc='upper right', fontsize=self.font_size, framealpha=0.9)
        axs[1].set_ylabel(self.labels['voltages_label'], rotation=90, labelpad=12, fontsize=self.font_size, fontweight='bold')
        axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        axs[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axs[1].set_facecolor('#f9f9f9')
        for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
            label.set_fontsize(self.font_size)
        
        # ========== ПОСТРОЕНИЕ ДИСКРЕТНЫХ СИГНАЛОВ ==========
        label_names = Features.TARGET
        amp = [1, 2, 3]
        
        if self.use_model:
            # Режим с предсказаниями модели
            pred_colors = ['#ADD8E6', '#F08080', '#90EE90']  # lightblue, lightcoral, lightgreen
            real_colors = ['#0000FF', '#FF0000', '#008000']   # blue, red, green
            
            for i, label_name in enumerate(label_names):
                pred_positions = [amp[i] if lbl else np.nan for lbl in predict_labels[label_name][start:end]]
                real_positions = [-amp[i] if lbl else np.nan for lbl in real_labels[label_name][start:end]]
                
                label_text = self.labels.get(label_name, label_name)
                axs[2].scatter(time_range, pred_positions, color=pred_colors[i], marker='s', s=20, 
                             label=f'{self.labels["prediction_label"]} {label_text}', alpha=0.7)
                axs[2].scatter(time_range, real_positions, color=real_colors[i], marker='o', s=30, 
                             label=f'{self.labels["real_label"]} {label_text}', alpha=0.8, edgecolors='black', linewidth=0.5)
                axs[2].fill_between(time_range, 0, pred_positions, color=pred_colors[i], alpha=0.25, step='mid')
                axs[2].fill_between(time_range, 0, real_positions, color=real_colors[i], alpha=0.15, step='mid')
                
                data[f'{self.labels["prediction_label"]} {label_text}'] = pred_positions
                data[f'{self.labels["real_label"]} {label_text}'] = real_positions
            
            # Настройка третьего графика
            axs[2].set_ylim([-3.8, 3.8])
            axs[2].text(time_range[int(len(time_range) * 0.02)], 3.5, self.labels['prediction_label'], 
                       horizontalalignment='left', fontsize=self.font_size - 1, color='black', fontweight='bold')
            axs[2].text(time_range[int(len(time_range) * 0.02)], -3.5, self.labels['real_label'], 
                       horizontalalignment='left', fontsize=self.font_size - 1, color='black', fontweight='bold')
        else:
            # Режим без модели - только реальная разметка
            real_colors = ['#0000FF', '#FF0000', '#008000']   # blue, red, green
            
            for i, label_name in enumerate(label_names):
                real_positions = [amp[i] if lbl else np.nan for lbl in real_labels[label_name][start:end]]
                label_text = self.labels.get(label_name, label_name)
                
                axs[2].scatter(time_range, real_positions, color=real_colors[i], marker='o', s=30, 
                             label=label_text, alpha=0.8, edgecolors='black', linewidth=0.5)
                axs[2].fill_between(time_range, 0, real_positions, color=real_colors[i], alpha=0.2, step='mid')
                
                data[label_text] = real_positions
            
            # Настройка третьего графика
            axs[2].set_ylim([-3.8, 3.8])
        
        axs[2].legend(loc='upper right', fontsize=self.font_size - 1, framealpha=0.9)
        axs[2].set_ylabel(self.labels['discrete_label'], rotation=90, labelpad=12, fontsize=self.font_size, fontweight='bold')
        axs[2].set_xlabel(self.labels['time_label'], fontsize=self.font_size, fontweight='bold', labelpad=10)
        axs[2].axhline(0, color='black', linestyle='-', linewidth=1)
        axs[2].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axs[2].set_facecolor('#f9f9f9')
        for label in axs[2].get_xticklabels() + axs[2].get_yticklabels():
            label.set_fontsize(self.font_size)
        
        # Сохранение всех данных в CSV и графиков
        # Получаем текущую директорию скрипта для сохранения файлов
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        csv_path = os.path.join(output_dir, f'{self.osc_name}_bus_{self.uses_bus}.csv')
        png_path = os.path.join(output_dir, f'{self.osc_name}_bus_{self.uses_bus}.png')
        pdf_path = os.path.join(output_dir, f'{self.osc_name}_bus_{self.uses_bus}.pdf')
        
        data.to_csv(csv_path, index=False)
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.show()

    def process_and_plot(self):
        """Основная функция для обработки данных из Comtrade, прогнозирования и визуализации.
        
        Работает как в режиме с моделью, так и без неё.
        """
        analog_signal, predict_labels, real_labels = self.predict()
        if isinstance(analog_signal, float) and np.isnan(analog_signal):
            print("Прогноз вернул NaN. Пропуск построения графика.")
            return
        self.plot_predictions_vs_real(self.start_point, self.end_point, analog_signal, predict_labels, real_labels)
 
class CalcFlops():
    """Класс для вычисления FLOPS."""

    def __init__(self, device: str = 'cuda', batch_size: int = 1):
        super().__init__()
        self.FRAME_SIZE = 64
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def calc_FPLOPS(
        self,
        ML_model_path: str,
    ):
        """Процесс разметки осциллограмм с использованием машинной модели."""
        model = torch.load(ML_model_path, map_location=self.device).to(self.device)
        # Подсчёт FLOPS и параметров
        summary(model, input_size=(1, self.FRAME_SIZE, len(FeaturesForDataset.FEATURES)))
        
        print("-------------------------Другие параметры-------------------------")
        # Получение FLOPS и параметров
        with torch.cuda.device(self.device):
            macs, params = get_model_complexity_info(
                model, 
                (self.FRAME_SIZE, len(FeaturesForDataset.FEATURES)),
                as_strings=True,
                print_per_layer_stat=True
            )
        
        print("-------------------------Другие параметры-------------------------")
        print(f"FLOPS: {macs}")
        print(f"Параметры: {params}")
 
        
# Usage example
if __name__ == "__main__":
    # ===================== ПРИМЕР 1: Построение графиков С МОДЕЛЬЮ =====================
    # Раскомментируйте для использования:
    """
    comtrade_predictor = ComtradePredictionAndPlotting(
        osc_name='a3a1591bc548a7faf784728430499837',
        uses_bus='2',
        strat_point=1000,
        end_point=2000,
        norm_file_path='raw_data/norm_coef_all_v1.4.csv',
        model_path='marking_up_oscillograms/model/model_ConvMLP_ep100_tl0.1709_train18.7660.pt',
        use_model=True,
        font_size=11,
        font_family='DejaVu Sans'
    )
    comtrade_predictor.process_and_plot()
    """
    
    # ===================== ПРИМЕР 2: Построение графиков БЕЗ МОДЕЛИ (только разметка) =====================
    # Раскомментируйте для использования:
    comtrade_predictor = ComtradePredictionAndPlotting(
        osc_name='a3a1591bc548a7faf784728430499837',
        uses_bus='1',
        strat_point=1000,
        end_point=2000,
        norm_file_path='raw_data/norm_coef_all_v1.4.csv',
        model_path=None,  # Не требуется при use_model=False
        use_model=False,  # Отключаем использование модели
        font_size=11,
        font_family='DejaVu Sans',
        language='en'  # 'ru' для русского, 'en' для английского
    )
    comtrade_predictor.process_and_plot()
    
    # ===================== ПРИМЕР 3: Расчёт FLOPS (опционально) =====================
    # Раскомментируйте для использования:
    """
    сalcFlops = CalcFlops()
    сalcFlops.calc_FPLOPS(ML_model_path='marking_up_oscillograms/model/model_FFT_MLP_ep100_tl0.3389_train60.7187.pt')
    """
    
    # ===================== ПРИМЕР 4: Полная разметка осциллограмм (MarkingUpOscillograms) =====================
    # Раскомментируйте для использования:
    """
    marking_up_oscillograms = MarkingUpOscillograms(norm_file_path="marking_up_oscillograms/norm_1600_v2.1.csv")
    marking_up_oscillograms.search_events_in_comtrade(
        csv_name='marking_up_oscillograms/sorted_files.csv',
        ML_model_path="marking_up_oscillograms/model/model_FftKAN_ep100_tl0.1193_train79.6926.pt",
    )
    """