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
        CURRENT = ["I_pos_seq_mag", "I_pos_seq_angle"]

        VOLTAGE = ["V_pos_seq_mag", "V_pos_seq_angle"]
        
        FEATURES = CURRENT.copy()
        FEATURES.extend(VOLTAGE)
        
        TARGET_train = ["rPDR PS"]
        TARGET_WITH_FILENAME_train = ["file_name"]
        TARGET_WITH_FILENAME_train.extend(TARGET_train)
        
        TARGET_test = ["iPDR PS"]
        TARGET_WITH_FILENAME_test = ["file_name"]
        TARGET_WITH_FILENAME_test.extend(TARGET_test)

class CustomDataset(Dataset):

    def __init__(self, name_target: str, dt: pd.DataFrame(), indexes: pd.DataFrame(), frame_size: int, target_position: int = None):
        """ Initialize the dataset.

        Args:
            dt (pd.DataFrame()): The DataFrame containing the data
            indexes (pd.DataFrame()): _description_
            frame_size (int): the size of the selection window
            target_position (int, optional): The position from which the target value is selected. 0 - means that it is taken from the first point. frame_size-1 - means that it is taken from the last point.
        """
        self.data = dt
        self.name_target = name_target
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
        target_sample = self.data.loc[target_index][self.name_target]
        target = torch.tensor(target_sample, dtype=torch.float32) # было torch.long
        return x, target


class SingleOscillogramSampler(torch.utils.data.Sampler):
    """
    Сэмплер, который формирует батчи, где каждый батч содержит
    все данные из одной случайно выбранной осциллограммы (файла).
    """
    def __init__(self, all_files, file_to_indices_map, num_batches, shuffle=True):
        """
        Args:
            all_files (list): Список имен всех файлов для сэмплирования.
            file_to_indices_map (dict): Словарь {имя_файла: [индекс1, индекс2, ...]}
            num_batches (int): Общее количество батчей (осциллограмм) для генерации за эпоху.
            shuffle (bool): Перемешивать ли индексы внутри батча (обычно не нужно, т.к. порядок важен).
                             Но мы можем перемешивать *выбор* файлов.
        """
        self.all_files = all_files
        self.file_to_indices_map = file_to_indices_map
        self.num_batches = num_batches
        # self.shuffle = shuffle # Перемешивание внутри батча тут может нарушить временную структуру

        if not self.all_files:
            raise ValueError("List of files cannot be empty.")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Выбираем num_batches случайных имен файлов (с возможностью повторений)
        selected_files = np.random.choice(
            self.all_files,
            size=self.num_batches,
            replace=True # Разрешаем повторный выбор файлов внутри эпохи
        )

        for file_name in selected_files:
            # Получаем все индексы для выбранного файла
            batch_indices = self.file_to_indices_map[file_name]

            # Проверка на пустой батч (если файл пустой или не содержит нужных строк)
            if not batch_indices:
                print(f"Warning: File {file_name} resulted in an empty batch. Skipping.")
                continue

            # Перемешивание индексов внутри батча здесь обычно не делают для временных рядов.
            # if self.shuffle:
            #     random.shuffle(batch_indices)

            yield batch_indices # Возвращаем список индексов для этого батча (одной осциллограммы)

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
    FRAME_SIZE = 1 # 64
    NUM_BATCHES = 1000 # Количество генерируемых батчей
    BATCH_SIZE_TEST = 8192
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    L2_REGULARIZATION_COEFFICIENT = 0.001
    MAX_GRAD_NORM = 10
    SEED = 42

    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    file_csv = "ML_model/dataset_cut_out_PDR_norm_v2.csv"
    # Create the folder if it doesn't exist
    folder_path = "/ML_model"
    os.makedirs(folder_path, exist_ok=True)
    #file_with_target_frame_train = "ML_model/dataset_cut_out_PDR_norm_v2.csv"
    file_with_target_frame_train = "ML_model/dataset_iPDR_norm_v1.csv"
    file_with_target_frame_test = "ML_model/dataset_iPDR_norm_v1.csv"

    if os.path.isfile(file_with_target_frame_train) and os.path.isfile(file_with_target_frame_test):
        dt_train = pd.read_csv(file_with_target_frame_train)
        dt_train.replace({'True': 1.0, '1': 1.0, '1.0': 1.0, 
                          'False': 0.0, '0': 1.0, '0.0': 0.0,}, inplace=True)
        
        # TODO: добиться, чтобы не было этих ошибок в датасете
        # Проверим, есть ли NaN в интересующих столбцах
        columns_to_check = FeaturesForDataset.FEATURES
        # Проверка по столбцам
        nan_check = dt_train[columns_to_check].isna().sum()
        # Выведем только те столбцы, где есть хотя бы один NaN
        print("NaN values found in:")
        print(nan_check[nan_check > 0])
        
        
        dt_test = pd.read_csv(file_with_target_frame_test)
        dt_test.replace({'True': 1.0, '1': 1.0, '1.0': 1.0, 
                         'False': 0.0, '0': 1.0, '0.0': 0.0,}, inplace=True)
    else:
        pass
        # TODO: Написать если нужно будет.

    # --- Обновленная ЛОГИКА ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ОБУЧЕНИЯ ---
    print("Preparing train indices grouped by file...")
    # 1. Группируем индексы по файлам
    file_to_indices_map_train = dt_train.groupby('file_name').apply(lambda x: list(x.index))
    # 2. Получаем список всех имен файлов для обучения
    all_train_files = list(file_to_indices_map_train.keys())
    
    print(f"Total train files: {len(all_train_files)}")
    # --- КОНЕЦ ЛОГИКИ ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ОБУЧЕНИЯ ---

    # --- ЛОГИКА ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ТЕСТА ---
    print("Preparing test indices...")
    test_indexes = dt_test.index # Используем все строки для теста
    print(f"Total test data points: {len(test_indexes)}")
    # --- КОНЕЦ ЛОГИКИ ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ТЕСТА ---

    # В CustomDataset передаем ПОЛНЫЙ DataFrame
    train_dataset = CustomDataset(FeaturesForDataset.TARGET_train[0], dt_train, dt_train, FRAME_SIZE, 0)
    test_dataset = CustomDataset(FeaturesForDataset.TARGET_test[0], dt_test, dt_test.loc[test_indexes], FRAME_SIZE, 0)

    start_epoch = 0
    # !! создание новой !!
    name_model = "PDR_MLP_v1"
    model = Model.PDR_MLP_v1(
        FRAME_SIZE,
        channel_num=len(FeaturesForDataset.FEATURES),
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

    sampler = SingleOscillogramSampler(
        all_files=all_train_files,
        file_to_indices_map=file_to_indices_map_train,
        num_batches=NUM_BATCHES # Количество осциллограмм в эпохе
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

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
    per_class_metrics = {feature_name: {"f1_score": [], "balanced_accuracy": []} for feature_name in FeaturesForDataset.TARGET_test}


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
