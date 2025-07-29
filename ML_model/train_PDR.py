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
        CURRENT_1 = ["I_pos_seq_mag", "I_pos_seq_angle"]
        CURRENT_2 = ["I_neg_seq_mag", "I_neg_seq_angle"]

        VOLTAGE_1 = ["V_pos_seq_mag"] # V_pos_seq_angle - не нужно, так как от него расчёт
        VOLTAGE_2 = ["V_neg_seq_mag", "V_neg_seq_angle"]
        
        POWER_1 = ["P_pos_seq", "Q_pos_seq"]
        POWER_2 = ["P_neg_seq", "Q_neg_seq"]
        
        IMPEDACY = ["Z_pos_seq_mag", "Z_pos_seq_angle"]
        
        # Простейшая - модель 1
        FEATURES = CURRENT_1.copy()
        FEATURES.extend(VOLTAGE_1)
        
        # Простейшая с всеми сигналами - модель 2
        # FEATURES = CURRENT_1.copy()
        # FEATURES.extend(CURRENT_2)
        # FEATURES.extend(VOLTAGE_1)
        # FEATURES.extend(VOLTAGE_2)
        # FEATURES.extend(POWER_1)
        # FEATURES.extend(POWER_2)
        # FEATURES.extend(IMPEDACY)
        
        TARGET_train = ["rPDR PS"]
        TARGET_WITH_FILENAME_train = ["file_name"]
        TARGET_WITH_FILENAME_train.extend(TARGET_train)
        
        TARGET_test = ["iPDR PS"]
        TARGET_WITH_FILENAME_test = ["file_name"]
        TARGET_WITH_FILENAME_test.extend(TARGET_test)

class CustomDataset(Dataset):

    def __init__(self, name_target: str, dt: pd.DataFrame(), indexes: pd.DataFrame(), frame_size: int, target_position: int = None):
        """ Инициализация набора данных.

        Args:
            dt (pd.DataFrame()): DataFrame, содержащий данные
            indexes (pd.DataFrame()): _описание_
            frame_size (int): размер окна выбора
            target_position (int, optional): Позиция, из которой выбирается целевое значение. 0 - означает, что оно берется из первой точки. frame_size-1 - означает, что оно берется из последней точки.
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


class MultiFileRandomPointSampler(torch.utils.data.Sampler):
    """
    Сэмплер, который формирует батчи, выбирая K случайных точек
    из N случайно выбранных файлов.
    """
    def __init__(self, all_files, file_to_valid_start_indices_map,
                 num_batches_per_epoch, num_files_per_batch, num_samples_per_file):
        """
        Args:
            all_files (list): Список имен всех файлов для сэмплирования.
            file_to_valid_start_indices_map (dict): Словарь {имя_файла: [валидный_индекс1, ...]}
            num_batches_per_epoch (int): Количество батчей, генерируемых за эпоху.
            num_files_per_batch (int): N - сколько файлов выбирать для одного батча.
            num_samples_per_file (int): K - сколько точек сэмплировать из каждого файла.
        """
        self.all_files = all_files
        self.file_to_valid_start_indices_map = file_to_valid_start_indices_map
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_files_per_batch = num_files_per_batch
        self.num_samples_per_file = num_samples_per_file

        if not self.all_files:
            raise ValueError("Список файлов не может быть пустым.")
        if self.num_files_per_batch <= 0 or self.num_samples_per_file <= 0:
            raise ValueError("num_files_per_batch и num_samples_per_file должны быть положительными.")

    def __len__(self):
        # Длина сэмплера - это количество батчей за эпоху
        return self.num_batches_per_epoch

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            # 1. Выбрать N случайных файлов (можно с повторениями)
            selected_files = np.random.choice(
                self.all_files,
                size=self.num_files_per_batch,
                replace=True
            )

            batch_indices = []
            # 2. Для каждого выбранного файла взять K случайных точек
            for file_name in selected_files:
                valid_indices = self.file_to_valid_start_indices_map[file_name]
                num_valid = len(valid_indices)

                if num_valid == 0:
                    # Такого быть не должно, если мы фильтровали all_files, но на всякий случай
                    continue

                # Выбираем K индексов из списка валидных для этого файла
                # Если в файле меньше K точек, берем все с повторениями или без?
                # Берем с повторениями, чтобы размер батча был стабильным N*K
                sampled_indices_for_file = np.random.choice(
                    valid_indices,
                    size=self.num_samples_per_file,
                    replace=True # Разрешаем повторы, если num_valid < K
                ).tolist()
                batch_indices.extend(sampled_indices_for_file)

            if not batch_indices:
                # Если по какой-то причине батч пустой, пропускаем
                # (например, если все выбранные файлы оказались пустыми, что маловероятно)
                print("Предупреждение: Сгенерирован пустой батч. Пропускаем.")
                continue

            # Перемешиваем индексы внутри итогового батча (опционально, но может быть полезно)
            random.shuffle(batch_indices)
            yield batch_indices # Возвращаем объединенный список индексов для батча

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
    Эта функция используется для поддержания повторяемости
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
                f"{feature}_f1" for feature in FeaturesForDataset.TARGET_test
            ] + [
                f"{feature}_ba" for feature in FeaturesForDataset.TARGET_test
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
        "f1_scores_per_class": {feature: f1 for feature, f1 in zip(FeaturesForDataset.TARGET_test, f1_per_class)},
        "balanced_accuracy_per_class": {feature: ba for feature, ba in zip(FeaturesForDataset.TARGET_test, ba_per_class)}
    }

    # Записываем в файл
    with open(filename, "a") as file:
        json.dump(data, file, indent=4)

def compute_loss(criterion, outputs, targets):
    # Загружаем цели и прогнозы в одинаковой форме:
    if outputs.dim() == targets.dim() + 1:
        # выход [B,1], цели [B] -> подтягиваем цели
        targets = targets.unsqueeze(1)
    elif targets.dim() == outputs.dim() + 1:
        # редкий случай, наоборот — подтягиваем выход
        outputs = outputs.unsqueeze(1)

    return criterion(outputs, targets)

if __name__ == "__main__":
    FRAME_SIZE = 1 # 64
    # BATCH_SIZE_TRAIN = NUM_FILES_PER_BATCH * SAMPLES_PER_FILE
    NUM_FILES_PER_BATCH = 100 # N: Сколько файлов брать для одного батча
    SAMPLES_PER_FILE = 10  # K: Сколько точек брать из каждого файла
    NUM_TRAIN_BATCHES_PER_EPOCH = 1000 # Сколько таких N*K батчей делать за эпоху
    BATCH_SIZE_TEST = 8192
    EPOCHS = 100
    LEARNING_RATE = 1e-2
    L2_REGULARIZATION_COEFFICIENT = 0.001
    MAX_GRAD_NORM = 10
    SEED = 42

    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # file_csv = "ML_model/dataset_cut_out_PDR_norm_v2.csv"
    # Создаем папку, если она не существует
    folder_path = "/ML_model"
    os.makedirs(folder_path, exist_ok=True)
    #file_with_target_frame_train = "ML_model/dataset_cut_out_PDR_norm_v2.csv"
    file_with_target_frame_train = "ML_model/train_dataset_rPDR_norm_v1.csv"
    file_with_target_frame_test = "ML_model/test_dataset_iPDR_1600_norm_v1.csv"

    if os.path.isfile(file_with_target_frame_train) and os.path.isfile(file_with_target_frame_test):
        dt_train = pd.read_csv(file_with_target_frame_train)
        dt_train.replace({'True': 1.0, '1': 1.0, '1.0': 1.0, 
                          'False': 0.0, '0': 1.0, '0.0': 0.0,}, inplace=True)
        
        # TODO: добиться, чтобы не было этих ошибок в датасете
        # Проверим, есть ли NaN в интересующих столбцах
        nan_check = dt_train[FeaturesForDataset.FEATURES].isna().sum()
        # Выведем только те столбцы, где есть хотя бы один NaN
        print("Найдены значения NaN в:")
        print(nan_check[nan_check > 0])
        
        
        dt_test = pd.read_csv(file_with_target_frame_test)
        dt_test.replace({'True': 1.0, '1': 1.0, '1.0': 1.0, 
                         'False': 0.0, '0': 1.0, '0.0': 0.0,}, inplace=True)
    else:
        pass
        # TODO: Написать если нужно будет.

    # --- Обновленная ЛОГИКА ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ОБУЧЕНИЯ ---
    print("Подготовка индексов для обучения, сгруппированных по файлам (только допустимые начальные точки)...")
    file_to_valid_start_indices_map_train = {}
    grouped_train = dt_train.groupby('file_name')
    for file_name, group in grouped_train:
        # Находим последний возможный стартовый индекс для этого файла
        last_possible_start = group.index[-1] - FRAME_SIZE + 1
        # Отбираем только те индексы группы, которые могут быть началом окна
        valid_start_indices = group.index[group.index <= last_possible_start].tolist()
        if valid_start_indices: # Добавляем, только если есть валидные точки
            file_to_valid_start_indices_map_train[file_name] = valid_start_indices

    all_train_files = list(file_to_valid_start_indices_map_train.keys())
    # Удаляем файлы, у которых не оказалось валидных стартовых точек
    all_train_files = [f for f in all_train_files if file_to_valid_start_indices_map_train[f]] 

    print(f"Всего файлов для обучения с допустимыми начальными точками: {len(all_train_files)}")
    if not all_train_files:
        raise ValueError("Ни в одном из обучающих файлов недостаточно точек данных для заданного FRAME_SIZE.")
    # --- КОНЕЦ ЛОГИКИ ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ОБУЧЕНИЯ ---
    
    # ---> ПОДГОТОВКА ИНДЕКСОВ ДЛЯ ТЕСТА <---
    print("Подготовка индексов для теста (только допустимые начальные точки)...")
    # Находим последний возможный стартовый индекс для всего тестового датафрейма
    last_possible_start_test = dt_test.index[-1] - FRAME_SIZE + 1
    # Отбираем только те индексы dt_test, которые могут быть началом окна
    valid_start_indices_test = dt_test.index[dt_test.index <= last_possible_start_test]

    # Создаем DataFrame, содержащий эти валидные индексы.
    # CustomDataset использует .iloc[idx].name на этом DataFrame для получения стартовой точки.
    test_indexes_df = pd.DataFrame(index=valid_start_indices_test) # Индексы этого DF - это то, что нам нужно

    if valid_start_indices_test.empty:
        raise ValueError("В тестовом наборе данных нет допустимых начальных точек для заданного FRAME_SIZE.")
        
    print(f"Всего допустимых начальных точек в тестовых данных: {len(test_indexes_df)}")
    # ---> КОНЕЦ ПОДГОТОВКИ ИНДЕКСОВ ДЛЯ ТЕСТА <---

    # В CustomDataset передаем ПОЛНЫЙ DataFrame
    train_dataset = CustomDataset(FeaturesForDataset.TARGET_train[0], dt_train, dt_train, FRAME_SIZE, 0)
    test_dataset = CustomDataset(FeaturesForDataset.TARGET_test[0], dt_test, test_indexes_df, FRAME_SIZE, 0)

    start_epoch = 0
    # !! создание новой !!
    name_model = "PDR_MLP_v2" # PDR_MLP_v1_2 (модель та же) 
    # model = Model.PDR_MLP_v1(
    #     FRAME_SIZE,
    #     channel_num=len(FeaturesForDataset.FEATURES),
    #     device=device,
    # )
    model = Model.PDR_MLP_v2(
        input_features=len(FeaturesForDataset.FEATURES),
        block_neuron_config=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        device=device,
    )
    model.to(device)
    
    # !! Загрузка модели из файла !!
    # filename_model = "ML_model/trained_models/model_PDR_MLP_v2_ep6_vbl0.0072_train12.9364.pt"
    # model = torch.load(filename_model)
    # start_epoch = int(filename_model.split("ep")[1].split("_")[0])
    # model.eval()  # Установка модели в режим оценки

    # Расчет pos_weight (делать один раз перед циклом обучения)
    neg = (dt_train[FeaturesForDataset.TARGET_train[0]] == 0).sum()
    pos = (dt_train[FeaturesForDataset.TARGET_train[0]] == 1).sum()
    pos_weight_value = neg / pos if pos > 0 else 1.0 # Защита от деления на ноль
    print("Отношение 0 к 1 = ", pos_weight_value)
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device) # Создаем тензор
    
    criterion = MultiLabelFocalLoss(gamma=3) # Пока это лучшая метрика. Можно будет поиграться с гамма.
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    all_losses = []

    train_sampler = MultiFileRandomPointSampler(
        all_files=all_train_files,
        file_to_valid_start_indices_map=file_to_valid_start_indices_map_train,
        num_batches_per_epoch=NUM_TRAIN_BATCHES_PER_EPOCH,
        num_files_per_batch=NUM_FILES_PER_BATCH,
        num_samples_per_file=SAMPLES_PER_FILE
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True) # num_workers можно настроить
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0, pin_memory=True)

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
        if (epoch % 5 == 0) and (epoch != 0):
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
                loss = compute_loss(criterion, output, targets)
                loss_sum += loss.item()
                all_losses.append(loss.item())
                message = (
                    f"Эпоха {epoch+1}/{EPOCHS} "
                    f"LR={current_lr:.3e} "
                    f"Потери при обучении: {1000*(loss_sum / (i + 1)):.4f} "
                )
                t.set_postfix_str(s=message)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )
                optimizer.step()

            model.eval()
            test_loss = 0.0
            true_labels, predicted_labels = [], []
            f1 = [0.0] * len(FeaturesForDataset.TARGET_test) # Инициализация нулями
            ba = [0.0] * len(FeaturesForDataset.TARGET_test) # Инициализация нулями
            hamming = 1.0 # Худшее значение
            jaccard = 0.0 # Худшее значение

            try:
                # Получаем итератор и берем ОДИН батч для быстрой валидации
                test_iterator = iter(test_dataloader)
                val_inputs, val_labels = next(test_iterator)

                with torch.no_grad():
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device) # Переносим таргеты на device
                    
                    outputs = model(val_inputs)
                    test_loss = compute_loss(criterion, outputs, val_labels).item() # Считаем loss по батчу

                    # Преобразуем выходы в предсказания (0 или 1)
                    preds = torch.where(outputs >= 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

                    # Собираем метки для расчета метрик (переносим обратно на CPU)
                    true_labels_batch = val_labels.cpu().numpy()
                    predicted_labels_batch = preds.cpu().numpy()

                # Рассчитываем метрики ПО ЭТОМУ ОДНОМУ БАТЧУ
                hamming = hamming_loss(true_labels_batch, predicted_labels_batch)
                # jaccard = jaccard_score(true_labels_batch, predicted_labels_batch, average='samples')

                ba, f1 = [], []
                true_binary = true_labels_batch[:].flatten()
                pred_binary = predicted_labels_batch[:].flatten()
                # Добавим zero_division=0 для избежания предупреждений, если класс отсутствует в батче
                ba.append(balanced_accuracy_score(true_binary, pred_binary, adjusted=False)) # adjusted=False по умолчанию
                f1.append(f1_score(true_binary, pred_binary, average='binary', zero_division=0))

            except StopIteration:
                print("Предупреждение: Тестовый загрузчик данных пуст или меньше размера батча. Пропускаем валидацию.")
                test_loss = -1.0 # Индикация ошибки

            # Рассчитываем средние метрики (по одному батчу)
            mean_f1 = np.mean(f1)
            mean_ba = np.mean(ba)

            # Время обучения текущей эпохи
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # Сохраняем данные в CSV (теперь они основаны на одном батче)
            save_stats_to_csv(
                epoch, batch_count, epoch_duration, loss_sum / len(train_dataloader), test_loss, # test_loss теперь по батчу
                mean_f1, mean_ba, hamming, -1, current_lr,
                f1_per_class=f1, ba_per_class=ba
            )
            

            # Обновляем scheduler на основе test_loss батча
            # Важно: эта потеря может быть шумной! Возможно, лучше обновлять по средней по эпохе train_loss
            scheduler.step(test_loss if test_loss >= 0 else float('inf')) # Используем test_loss батча

            # Сообщение для tqdm (оставляем старое, т.к. train_loss считается по эпохе)
            t.set_postfix_str(f"Батч: {batch_count}, Потери при обучении: {loss_sum / len(train_dataloader):.4f}, Потери на валидационном батче: {test_loss:.4f}, LR: {current_lr:.4e}")
            # Вывод метрик, рассчитанных по батчу
            message_f1_ba = (
                            f"Потери на валидационном батче: {1000*test_loss:.4f} "
                            f"Val F1/BA: {', '.join([f'{signal_name}: {f1_score:.4f}/{ba_score:.4f}' for signal_name, f1_score, ba_score in zip(FeaturesForDataset.TARGET_test, f1, ba)])} "
                            )
            print(message_f1_ba)
            print(f"Val Hamming: {hamming:.4f}, Val Jaccard: {jaccard:.4f}")

            torch.save(model, f"ML_model/trained_models/model_{name_model}_ep{epoch+1}_vbl{test_loss:.4f}_train{loss_sum:.4f}.pt") # vbl = validation batch loss
            model.train() # Возвращаем модель в режим обучения
    pass
pass
