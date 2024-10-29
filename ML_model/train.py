import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import csv
import json
import time

from sklearn.metrics import hamming_loss, jaccard_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
import model as Model

# Добавлено для исключения лишних предупреждений о возможных будущих проблемах.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class FeaturesForDataset():
        CURRENT = ["IA", "IB", "IC"]

        VOLTAGE = ["UA BB", "UB BB", "UC BB", "UN BB",
                   "UA CL", "UB CL", "UC CL", "UN CL",
                   "UAB CL","UBC CL","UCA CL"]
        

        VOLTAGE_PHAZE = ["UA BB", "UB BB", "UC BB",
                         "UA CL", "UB CL", "UC CL"]
        VOLTAGE_LINE_NOMINAL = ["UN BB", "UN CL", "UAB CL", "UBC CL", "UCA CL"]
        
        VOLTAGE_PHAZE_BB = ["UA BB", "UB BB", "UC BB"]
        VOLTAGE_PHAZE_CL = ["UA CL", "UB CL", "UC CL"]

        VOLTAGE_LINE_CL = ["UAB CL", "UBC CL", "UCA CL"]
        
        ZERO_SIGNAL = ["UN BB", "UN CL"] # не хватает тока
        
        FEATURES = CURRENT.copy()
        FEATURES.extend(VOLTAGE)
        
        # TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        
        TARGET_WITH_FILENAME = ["file_name"]
        TARGET_WITH_FILENAME.extend(TARGET)

# TODO: Решить проблему различия __getitem__ в первой строке с "iloc" у CustomDataset_train и CustomDataset
class CustomDataset_train(Dataset):
    def __init__(self, dt: pd.DataFrame(), indexes: pd.DataFrame(), frame_size: int, target_position: int = None, 
                 apply_inversion: bool = False, apply_noise: bool = False, current_noise_level: float = 0.0004, voltage_noise_level: float = 0.0001,
                 apply_amplitude_scaling: bool = False, current_amplitude_factor: float = 5.0, voltage_amplitude_factor: float = 1.05,
                 apply_offset: bool = False, offset_range: tuple = (-0.001, 0.001),
                 apply_phase_shuffling: bool = False,
                 apply_delet_zero_signal: bool = False,
                 augmentation_probabilities: dict = None):
        """
        Initialize the dataset.

        Args:
            dt (pd.DataFrame()): The DataFrame containing the data.
            indexes (pd.DataFrame()): DataFrame with index positions to split the data.
            frame_size (int): The size of the selection window.
            target_position (int, optional): The position from which the target value is selected. 0 - means that it is taken from the first point. frame_size-1 - means that it is taken from the last point.
            apply_inversion (bool): Whether to apply signal inversion.
            apply_noise (bool): Whether to apply Gaussian noise.
            current_noise_level (float): Standard deviation of the Gaussian noise.
            voltage_noise_level (float): Standard deviation of the Gaussian noise.
            apply_amplitude_scaling (bool): Whether to apply amplitude scaling.
            current_amplitude_range (float): Scaling range for current signals.
            voltage_amplitude_range (float): Scaling range for voltage signals.
            apply_offset (bool): Whether to apply offset drift.
            offset_range (tuple): Range for the offset value.
            apply_phase_shuffling (bool): Whether to apply phase shuffling for current and voltage channels.
            augmentation_probabilities (dict, optional): Dictionary specifying the activation probabilities for each augmentation.
                Example: {"inversion": 0.5, "noise": 0.3, "scaling": 0.7, "offset": 0.4, "phase_shuffling": 0.2}.
                Default value is 0.5 for all augmentations if not provided.
        """
        self.data = dt
        self.indexes = indexes
        self.frame_size = frame_size

        # Параметры целевой позиции
        if target_position is not None:
            if 0 <= target_position < frame_size:
                self.target_position = target_position
            else:
                self.target_position = frame_size - 1
                print("Invalid target position. Target position should be in range of 0 to frame_size-1")
        else:
            self.target_position = frame_size - 1

        # Параметры для аугментации
        self.apply_inversion = apply_inversion
        self.apply_noise = apply_noise
        self.current_noise_level = current_noise_level
        self.voltage_noise_level = voltage_noise_level
        self.apply_amplitude_scaling = apply_amplitude_scaling
        self.current_amplitude_factor = current_amplitude_factor
        self.voltage_amplitude_factor = voltage_amplitude_factor
        self.apply_offset = apply_offset
        self.offset_range = offset_range
        self.apply_phase_shuffling = apply_phase_shuffling
        self.apply_delet_zero_signal = apply_delet_zero_signal

        # Вероятности активации аугментаций
        # По умолчанию вероятность для всех аугментаций = 0.5
        default_probabilities = {"inversion": 0.5, "noise": 0.5, "scaling": 0.5, "offset": 0.5, "phase_shuffling": 0.5, "delet_zero_singnal": 0.5}
        self.augmentation_probabilities = default_probabilities if augmentation_probabilities is None else augmentation_probabilities

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        # Получение начального индекса для окна
        start = self.indexes.loc[idx].name  
        if start + self.frame_size - 1 >= len(self.data):
            # Защита от выхода за диапазон. Такого быть не должно при обрезании массива, но оставил на всякий случай.
            return (None, None)
        
        # Извлечение окна данных
        sample = self.data.loc[start: start + self.frame_size - 1][FeaturesForDataset.FEATURES]
        x = torch.tensor(sample.to_numpy(dtype=np.float32), dtype=torch.float32)

        # === АУГМЕНТАЦИЯ ДАННЫХ === #
        
        # 1. Инверсия сигнала
        if self.apply_inversion and random.random() < self.augmentation_probabilities["inversion"]:
            x = x * -1

        # 2. Амплитудные искажения (раздельно для токов и напряжений)
        if self.apply_amplitude_scaling and random.random() < self.augmentation_probabilities["scaling"]:
            # Определение типа искажения: уменьшение, увеличение или без изменений
            scaling_type = np.random.choice(['decrease', 'increase', 'none'], p=[1/3, 1/3, 1/3])

            if scaling_type == 'decrease':
                current_scale_factor = np.random.uniform(1 / self.current_amplitude_factor, 1)
                voltage_scale_factor = np.random.uniform(1 / self.voltage_amplitude_factor, 1)
            elif scaling_type == 'increase':
                current_scale_factor = np.random.uniform(1, self.current_amplitude_factor)
                voltage_scale_factor = np.random.uniform(1, self.voltage_amplitude_factor)
            else:
                current_scale_factor = 1.0  # Без изменений
                voltage_scale_factor = 1.0  # Без изменений

            # Масштабирование токов (одним коэффициентом для всех фаз)
            x[:, [sample.columns.get_loc(col) for col in FeaturesForDataset.CURRENT]] *= current_scale_factor

            # Масштабирование напряжений (одним коэффициентом для всех фаз и линий)
            x[:, [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE]] *= voltage_scale_factor

        # 3. Добавление шума
        if self.apply_noise and random.random() < self.augmentation_probabilities["noise"]:
            # Создание шума для токов
            current_noise = torch.normal(
                mean=0,
                std=self.current_noise_level,
                size=(x.shape[0], len(FeaturesForDataset.VOLTAGE_PHAZE))
            )
            # Применение шума к колонкам токов
            x[:, [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_PHAZE]] += current_noise
            
            # Создание шума для напряжений
            voltage_phaze_noise = torch.normal(
                mean=0,
                std=self.voltage_noise_level,
                size=(x.shape[0], len(FeaturesForDataset.VOLTAGE_PHAZE))
            )
            voltage_line_noise = torch.normal(
                mean=0,
                std=self.voltage_noise_level * torch.sqrt(torch.tensor(3.)),
                size=(x.shape[0], len(FeaturesForDataset.VOLTAGE_LINE_NOMINAL))
            )
            # Применение шума к колонкам напряжений
            x[:, [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_PHAZE]] += voltage_phaze_noise
            x[:, [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_LINE_NOMINAL]] += voltage_line_noise

        # 4. Сдвиг значений (Offset)
        if self.apply_offset and random.random() < self.augmentation_probabilities["offset"]:
            offset = torch.tensor(np.random.uniform(self.offset_range[0], self.offset_range[1], size=x.shape[1]), dtype=torch.float32)
            x = x + offset

        # 5. Перетасовка фаз
        if self.apply_phase_shuffling and random.random() < self.augmentation_probabilities["phase_shuffling"]:
            # Генерация случайного сдвига фаз (0 - без изменений, 1 - сдвиг на одну фазу, 2 - сдвиг на две фазы)
            phase_shift = np.random.randint(0, 3)

            # Сдвиг фазовых токов (IA, IB, IC)
            current_indices = [sample.columns.get_loc(col) for col in FeaturesForDataset.CURRENT]
            x[:, current_indices] = torch.roll(x[:, current_indices], shifts=phase_shift, dims=1)

            # Сдвиг фазных напряжений
            phaze_BB_indices = [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_PHAZE_BB]
            x[:, phaze_BB_indices] = torch.roll(x[:, phaze_BB_indices], shifts=phase_shift, dims=1)

            phaze_CL_indices = [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_PHAZE_CL]
            x[:, phaze_CL_indices] = torch.roll(x[:, phaze_CL_indices], shifts=phase_shift, dims=1)

            # Сдвиг линейных напряжений (UAB CL, UBC CL, UCA CL)
            line_CL_indices = [sample.columns.get_loc(col) for col in FeaturesForDataset.VOLTAGE_LINE_CL]
            x[:, line_CL_indices] = torch.roll(x[:, line_CL_indices], shifts=phase_shift, dims=1)

        # 6. Удаление входов нулевой последовательности
        if self.apply_delet_zero_signal and random.random() < self.augmentation_probabilities["delet_zero_singnal"]:
            zeroSignal_indices = [sample.columns.get_loc(col) for col in FeaturesForDataset.ZERO_SIGNAL]
            x[:, zeroSignal_indices] = torch.zeros_like(x[:, zeroSignal_indices])

        # === ИЗВЛЕЧЕНИЕ ЦЕЛЕВОГО ЗНАЧЕНИЯ === #
        target_index = start + self.target_position
        target_sample = self.data.loc[target_index][FeaturesForDataset.TARGET]
        target = torch.tensor(target_sample, dtype=torch.float32)

        return x, target

class CustomDataset(Dataset):

    def __init__(self, dt: pd.DataFrame(), indexes: pd.DataFrame(), frame_size: int, target_position: int = None):
        """ Initialize the dataset.

        Args:
            dt (pd.DataFrame()): The DataFrame containing the data
            indexes (pd.DataFrame()): _description_
            frame_size (int): the size of the selection window
            target_position (int, optional): The position from which the target value is selected. 0 - means that it is taken from the first point. frame_size-1 - means that it is taken from the last point.
        """
        self.data = dt
        self.indexes = indexes
        self.frame_size = frame_size
        if (target_position is not None):
            if 0 <= target_position < frame_size:
                self.target_position = target_position
            else:
                self.target_position = frame_size
                print("Invalid target position. Target position should be in range of 0 to frame_size-1")
        else:
            self.target_position = frame_size-1

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        start = self.indexes.iloc[idx].name
        if start + self.frame_size - 1 >= len(self.data):
            # Защита от выхода за диапазон. Такого быть не должно при обрезании массива, но оставил на всякий случай.
            return (None, None)
        
        sample = self.data.loc[start : start + self.frame_size - 1][FeaturesForDataset.FEATURES]
        #sample = frame[FeaturesForDataset.FEATURES]
        x = torch.tensor(
            sample.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        target_index = start + self.target_position
        target_sample = self.data.loc[target_index][FeaturesForDataset.TARGET]
        target = torch.tensor(target_sample, dtype=torch.float32) # было torch.long
        return x, target


class FastBalancedBatchSampler(torch.utils.data.Sampler):
    """Быстрая реализация BalancedBatchSampler.
    Для каждой категории данных заранее создаются списки индексов, а затем выбираются батчи."""

    def __init__(self, datasets_by_class, batch_size_per_class, num_batches, shuffle=True):
        """ Инициализация
        Args:
            datasets_by_class (dict): Словарь, где ключ - имя категории, а значение - индексы для этой категории
            batch_size_per_class (int): Количество примеров на класс в одном батче
            num_batches (int): Общее количество батчей для генерации
            shuffle (bool): Перемешивание данных внутри батча
        """
        self.datasets_by_class = datasets_by_class
        self.batch_size_per_class = batch_size_per_class
        self.num_batches = num_batches
        self.shuffle = shuffle

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for class_name, indices in self.datasets_by_class.items():
                # Выбор случайных индексов, без повторений
                selected_indices = np.random.choice(indices, size=self.batch_size_per_class, replace=False)
                batch.extend(selected_indices)

            yield batch

class MultiLabelFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # Рассчитываем Focal Loss вручную
        pt = targets * outputs + (1 - targets) * (1 - outputs)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * (targets * torch.log(outputs + 1e-8) + (1 - targets) * torch.log(1 - outputs + 1e-8))
        
        return focal_loss.mean()

def seed_everything(seed: int = 42):
    """
    This function is used to maintain repeatability
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Сохранение статистики в CSV-файл
def save_stats_to_csv(epoch, batch_count, epoch_duration, train_loss, test_loss, mean_f1, mean_ba, hamming, jaccard, lr, f1_per_class, ba_per_class):
    """Сохраняем данные в CSV файл с заголовками."""
    # Путь к основному файлу статистики
    training_statistics_file = "ML_model/trained_models/training_statistics.csv"
    per_class_metrics_file = "ML_model/trained_models/per_class_metrics.csv"

    # Создаем или открываем файл для записи общей статистики
    with open(training_statistics_file, mode="a", newline='') as file:
        writer = csv.writer(file)

        # Записываем заголовок, если файл пустой
        if os.stat(training_statistics_file).st_size == 0:
            header = [
                "epoch", "batch_count", "epoch_duration",
                "train_loss", "test_loss", "mean_f1", "mean_ba",
                "hamming_loss", "jaccard_score", "learning_rate"
            ]
            writer.writerow(header)

        # Записываем строку с текущими значениями
        writer.writerow([
            epoch, batch_count, epoch_duration, train_loss, test_loss,
            mean_f1, mean_ba, hamming, jaccard, lr
        ])

    # Сохраняем метрики по каждому классу в отдельный CSV файл
    with open(per_class_metrics_file, mode="a", newline='') as file:
        writer = csv.writer(file)

        # Записываем заголовок, если файл пустой
        if os.stat(per_class_metrics_file).st_size == 0:
            header = ["epoch", "batch_count", "epoch_duration"] + [
                f"{feature}_f1" for feature in FeaturesForDataset.TARGET
            ] + [
                f"{feature}_ba" for feature in FeaturesForDataset.TARGET
            ]
            writer.writerow(header)

        # Формируем строку с метриками для текущей эпохи и батча
        row = [epoch, batch_count, epoch_duration] + f1_per_class + ba_per_class
        writer.writerow(row)

    # Сохранение статистики в JSON
    save_stats_to_json(
        "ML_model/trained_models/epoch_statistics.json", epoch, batch_count, epoch_duration,
        train_loss, test_loss,
        mean_f1, mean_ba, hamming, jaccard, lr,
        f1_per_class=f1, ba_per_class=ba
    )

def expand_array(arr, expand_left, expand_right):
    if expand_left < 0 or expand_right < 0 or (expand_left == 0 and expand_right == 0):
        return arr

    expanded = np.zeros_like(arr)  # создаем новый массив такой же длины, заполненный нулями
    n = len(arr)
        
    for i in range(n):
        if arr[i] == 1:
            # Определяем границы расширения
            left = max(0, i - expand_left)
            right = min(n, i + expand_right + 1)
            # Проставляем единицы в расширенных пределах
            expanded[left:right] = 1
    
    return expanded

def save_stats_to_json(filename, epoch, batch_count, epoch_duration, train_loss, test_loss, mean_f1, mean_ba, hamming, jaccard, lr, f1_per_class, ba_per_class):
    """Сохранение статистики в JSON формате."""
    # Создаем структуру данных
    data = {
        "epoch": epoch,
        "batch_count": batch_count,
        "epoch_duration": epoch_duration,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "mean_f1": mean_f1,
        "mean_ba": mean_ba,
        "hamming_loss": hamming,
        "jaccard_score": jaccard,
        "learning_rate": lr,
        "f1_scores_per_class": {feature: f1 for feature, f1 in zip(FeaturesForDataset.TARGET, f1_per_class)},
        "balanced_accuracy_per_class": {feature: ba for feature, ba in zip(FeaturesForDataset.TARGET, ba_per_class)}
    }

    # Записываем в файл
    with open(filename, "a") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    FRAME_SIZE = 64 # 64
    POINT_TARGET_SHIFT = 0
    BATCH_SIZE_PER_CLASS = 32  # Например, 128/4=32
    BATCH_SIZE_TRAIN = 4 * BATCH_SIZE_PER_CLASS # 4 - количество фич, сделать через Гипер Параметр
    NUM_BATCHES = 1000 # Количество генерируемых батчей
    BATCH_SIZE_TEST = 1024
    HIDDEN_SIZE = 40
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    L2_REGULARIZATION_COEFFICIENT = 0.001
    MAX_GRAD_NORM = 10
    SEED = 42
    print(f"{BATCH_SIZE_TRAIN=}")

    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    file_csv = "ML_model/datset_simpl v2.csv"
    # Create the folder if it doesn't exist
    folder_path = "/ML_model"
    os.makedirs(folder_path, exist_ok=True)
    file_with_target_frame_train = (
        f"ML_model/small_pqd_target_frame_{FRAME_SIZE}_train.csv"
    )
    file_with_target_frame_test = (
        f"ML_model/small_pqd_target_frame_{FRAME_SIZE}_test.csv"
    )

    if os.path.isfile(file_with_target_frame_train) and os.path.isfile(
        file_with_target_frame_test
    ):
        dt_train = pd.read_csv(file_with_target_frame_train)
        dt_test = pd.read_csv(file_with_target_frame_test)
    else:
        dt = pd.read_csv(file_csv)
        
        # TODO: Нормальизацию стоит делать отдельно. Здесь я её закладываю временно
        Inom, Unom = 5, 100
        for name in FeaturesForDataset.CURRENT:
            dt[name] = dt[name] / (Inom*20)
        for name in FeaturesForDataset.VOLTAGE:
            dt[name] = dt[name] / (Unom*3)
        
        with open('ML_model/test_files.json', 'r') as infile:
            data = json.load(infile)  
        files_to_test = data["test_files"]
        
        # Create a new list to store the full file names
        full_files_to_test = []
        for full_file_name in dt["file_name"].unique():
            for file_name in files_to_test:
                if file_name in full_file_name:
                    full_files_to_test.append(full_file_name)
        
        dt_test = dt[dt["file_name"].str.strip().isin(full_files_to_test)]
        dt_train = dt[~dt["file_name"].str.strip().isin(full_files_to_test)]

        std_scaler = StandardScaler()
        
        df_scaled_train = dt_train[FeaturesForDataset.FEATURES]
        df_scaled_test = dt_test[FeaturesForDataset.FEATURES]
        # TODO: массштабирование производится отдельно. Нормализация в адекватных данных не должна требоваться
        # так как знечения симметричны относительно 0. 
        df_scaled_train = pd.DataFrame(
            df_scaled_train,
            columns=df_scaled_train.columns,
            index=df_scaled_train.index,
        )
        df_scaled_test = pd.DataFrame(
            df_scaled_test,
            columns=df_scaled_test.columns,
            index=df_scaled_test.index,
        )

        dt_train = pd.concat(
            (dt_train.drop(FeaturesForDataset.FEATURES, axis=1), df_scaled_train), axis=1
        )
        dt_test = pd.concat(
            (dt_test.drop(FeaturesForDataset.FEATURES, axis=1), df_scaled_test), axis=1
        )

        dt_train.reset_index(drop=True, inplace=True)
        dt_test.reset_index(drop=True, inplace=True)

        target_series_train = dt_train[FeaturesForDataset.TARGET]
        target_series_test = dt_test[FeaturesForDataset.TARGET]

        # замена значений NaN на 0 в конце датафрейма
        for name in FeaturesForDataset.TARGET:
            dt_train[name] = dt_train[name].fillna(0)  
            dt_train[name] = dt_train[name].astype(int)
            dt_test[name] = dt_test[name].fillna(0)
            dt_test[name] = dt_test[name].astype(int)
        # Замена значений NaN на 0 для пустых ячеек
        for name in FeaturesForDataset.FEATURES:
            dt_train[name] = dt_train[name].fillna(0)  
            dt_train[name] = dt_train[name].astype(np.float32)
            dt_test[name] = dt_test[name].fillna(0)
            dt_test[name] = dt_test[name].astype(np.float32)

        dt_train.to_csv(file_with_target_frame_train, index=False)
        dt_test.to_csv(file_with_target_frame_test, index=False)
    
    
    files_train = dt_train["file_name"].unique()
    for file in files_train: # Непонятно почему не улучшает, надо отдельно проверить
        for target in FeaturesForDataset.TARGET:
            # Извлекаем строки, где file_name соответствует текущему файлу
            mask = dt_train["file_name"] == file
            # Обновляем нужный столбец после применения функции expand_array
            dt_train.loc[mask, target] = expand_array(dt_train.loc[mask, target].values, expand_left=0, expand_right=31)
    
    files_test = dt_test["file_name"].unique()
    for file in files_test: # Непонятно почему не улучшает, надо отдельно проверить
        for target in FeaturesForDataset.TARGET:
            # Извлекаем строки, где file_name соответствует текущему файлу
            mask = dt_test["file_name"] == file
            # Обновляем нужный столбец после применения функции expand_array
            dt_test.loc[mask, target] = expand_array(dt_test.loc[mask, target].values, expand_left=0, expand_right=31)
    
    # копия датафрейма для получения индексов начал фреймов для трейн
    dt_indexes_train = dt_train[FeaturesForDataset.TARGET_WITH_FILENAME]
    files_train = dt_indexes_train["file_name"].unique()
    train_indexes = pd.DataFrame()
    for file in files_train:
        # удаление последних FRAME_SIZE сэмплов в каждом файле
        df_file = dt_indexes_train[dt_indexes_train["file_name"] == file][
            :-FRAME_SIZE # Удаление последних FRAME_SIZE сэмплов в каждом файл, чтобы индексы не выходили за диапазон
        ]
        train_indexes = pd.concat((train_indexes, df_file))
    
    dt_train_opr_swch = train_indexes[train_indexes["opr_swch"] == 1]
    dt_train_abnorm_evnt = train_indexes[train_indexes["abnorm_evnt"] == 1]
    dt_train_emerg_evnt = train_indexes[train_indexes["emerg_evnt"] == 1]
    
    dt_train_no_event = train_indexes[train_indexes["opr_swch"] == 0]
    dt_train_no_event = dt_train_no_event[dt_train_no_event["abnorm_evnt"] == 0]
    dt_train_no_event = dt_train_no_event[dt_train_no_event["emerg_evnt"] == 0]
    
    datasets_by_class = {
        'opr_swch': list(dt_train_opr_swch.index),
        'abnorm_evnt': list(dt_train_abnorm_evnt.index),
        'emerg_evnt': list(dt_train_emerg_evnt.index),
        'no_event': list(dt_train_no_event.index)
    }

    # копия датафрейма для получения индексов начал фреймов для тест
    dt_indexes_test = dt_test[FeaturesForDataset.TARGET_WITH_FILENAME]
    files_test = dt_indexes_test["file_name"].unique()
    test_indexes = pd.DataFrame()
    for file in files_test:
        # удаление последних FRAME_SIZE сэмплов в каждом файле
        df_file = dt_indexes_test[dt_indexes_test["file_name"] == file][
            :-FRAME_SIZE # Удаление последних FRAME_SIZE сэмплов в каждом файл, чтобы индексы не выходили за диапазон
        ]
        test_indexes = pd.concat((test_indexes, df_file))

    # скорректировать точку, я сейчас задал принудительно 8 точек назад (четверть периода назад, или 4мс)
    train_dataset = CustomDataset_train(
        dt=dt_train, indexes=train_indexes,
        frame_size=FRAME_SIZE,  # Указываем размер окна
        target_position=FRAME_SIZE-1-POINT_TARGET_SHIFT,  # Целевая позиция – последний элемент окна
        apply_inversion=True, # Активируем рандомную инверсию сигнала
        apply_noise=True, current_noise_level=0, # Активируем добавление шума, но зануляем его по току
        apply_amplitude_scaling=True, # Активируем изменение масштаба
        apply_offset=False, # Активирует добавление рандомной постоянной составляющей
        apply_phase_shuffling=True,  # Активируем рандомнуюп перетасовку фаз
        apply_delet_zero_signal=True  # Активируем рандомное зануление входов нулевой последователности
    )
    
    test_dataset = CustomDataset(dt_test, test_indexes, FRAME_SIZE, FRAME_SIZE-1-POINT_TARGET_SHIFT)

    start_epoch = 0
    # !! создание новой !!
    name_model = "FFT_MLP_COMPLEX_v2" # "ConvMLP" # FftMLP , FftKAN, FFT_MLP_COMPLEX_v1, FFT_MLP_COMPLEX_v2, FFT_MLP_COMPLEX_v2
    # model = CONV_MLP_v2(
    # model = FFT_MLP(
    # model = FFT_MLP_KAN_v1(
    # model = FFT_MLP_KAN_v2(
    model = Model.CONV_AND_FFT_COMPLEX_v3(
        FRAME_SIZE,
        channel_num=len(FeaturesForDataset.FEATURES),
        hidden_size=HIDDEN_SIZE,
        output_size=len(FeaturesForDataset.TARGET),
        device=device,
    )
    
    model.to(device)
    # # !! Загрузка модели из файла !!
    # filename_model = "ML_model/trained_models/model_ep2_tl0.3498_train1432.3669.pt"
    # model = torch.load(filename_model)
    # start_epoch = int(filename_model.split("ep")[1].split("_")[0])
    # model.eval()  # Set the model to evaluation mode

    criterion = MultiLabelFocalLoss(gamma=3) # Пока это лучшая метрика. Можно будет поиграться с гамма.
    
    all_losses = []

    sampler = FastBalancedBatchSampler(datasets_by_class, batch_size_per_class=BATCH_SIZE_PER_CLASS, num_batches=NUM_BATCHES)
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST)

    # Инициализация DataFrame для статистики
    # Статистика по каждому классу и общей метрике
    epoch_statistics = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "f1_scores_per_class": [],
        "balanced_accuracy_per_class": [],
        "hamming_loss": [],
        "jaccard_score": [],
        "learning_rate": []
    }

    # Также можно создать CSV для хранения метрик каждой эпохи с разбивкой по классам
    per_class_metrics = {feature_name: {"f1_score": [], "balanced_accuracy": []} for feature_name in FeaturesForDataset.TARGET}


    current_lr = LEARNING_RATE
    batch_count = 0
    for epoch in range(start_epoch,EPOCHS):
        epoch_start_time = time.time()  # Начало отсчета времени для эпохи
        if (epoch % 10 == 0) and (epoch != 0):
            current_lr /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
        #optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=L2_REGULARIZATION_COEFFICIENT)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
        )
        with tqdm(train_dataloader, unit=" batch", mininterval=3) as t:
            loss_sum = 0.0
            for i, (batch, targets) in enumerate(t):
                if (batch == None or targets == None):
                    # TODO: Защита от необрезанных массивов. Но сюда не должно попадать.
                    continue
                
                batch_count += 1  # Счётчик батчей
                
                targets = targets.to(device)
                batch = batch.to(device)

                output = model(batch)
                loss = criterion(output, targets)
                loss_sum += loss.item()
                all_losses.append(loss.item())
                message = (
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"LR={current_lr:.3e} "
                    f"Train loss: {1000*(loss_sum / (i + 1)):.4f} "
                )
                t.set_postfix_str(s=message)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )
                optimizer.step()

            model.eval()

            # Сохранение статистики каждую эпоху
            # Выполняем оценку на тестовой выборке
            test_loss = 0.0
            true_labels, predicted_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, labels.to(device)).item()
                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(outputs.cpu().numpy())
            
            # Рассчитываем метрики
            numpy_labels = np.array(predicted_labels) # промежуточные преобразования для ускорения
            predicted_labels_tensor = torch.from_numpy(numpy_labels)
            predicted_labels = torch.where(predicted_labels_tensor >= 0.5, torch.tensor(1), torch.tensor(0))
            # Hamming Loss - чем меньше, тем лучше
            hamming = hamming_loss(true_labels, predicted_labels)
            # Jaccard Score - для многолейбловой задачи, 'samples' для подсчета по образцам - чем ближе к 1, тем лучше
            jaccard = jaccard_score(true_labels, predicted_labels, average='samples')
            
            # TODO: Разобраться с метриками и модернизировать их
            numpy_true_labels = np.array(true_labels) # промежуточные преобразования для ускорения
            true_labels_tensor = torch.from_numpy(numpy_true_labels)
            ba, f1 = [], []
            for k in range(len(FeaturesForDataset.TARGET)): # 'binary' для бинарной классификации на каждом классе
                true_binary = true_labels_tensor[:, k].flatten()
                pred_binary = predicted_labels[:, k].flatten()
                
                ba.append(balanced_accuracy_score(true_binary, pred_binary))
                f1.append(f1_score(true_binary, pred_binary, average='binary'))

            
            # Сохраняем данные
            # Рассчитываем метрики после каждой эпохи
            test_loss = test_loss / len(test_dataloader) # Средние потери на тестовой выборке
            mean_f1 = np.mean(f1)  # Средний F1-score по всем классам
            mean_ba = np.mean(ba)  # Средний Balanced Accuracy по всем классам
            
            # Время обучения текущей эпохи
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time  # Время в секундах

            # Сохраняем данные в CSV
            save_stats_to_csv(
                epoch, batch_count, epoch_duration, loss_sum / len(train_dataloader), test_loss,
                mean_f1, mean_ba, hamming, jaccard, current_lr,
                f1_per_class=f1, ba_per_class=ba
            )   


            # Сообщение для tqdm
            t.set_postfix_str(f"Batch: {batch_count}, Train loss: {loss_sum / (i + 1):.4f}, Test loss: {test_loss:.4f}, LR: {current_lr:.4e}")
            message_f1_ba = (
                             f"Prev. test loss: {1000*test_loss:.4f} "
                             f"F1 / BA: {', '.join([f'{signal_name}: {f1_score:.4f}/{ba_score:.4f}' for signal_name, f1_score, ba_score in zip(FeaturesForDataset.TARGET, f1, ba)])} "
                             )
            print(message_f1_ba)
            print(f"Hamming Loss: {hamming}, Jaccard Score: {jaccard}")
            
            torch.save(model, f"ML_model/trained_models/model_{name_model}_ep{epoch+1}_tl{test_loss:.4f}_train{loss_sum:.4f}.pt")
            model.train()
    pass
pass
