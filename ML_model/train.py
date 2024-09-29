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

from sklearn.metrics import hamming_loss, jaccard_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from model import CONV_MLP_v2, FFT_MLP, FFT_MLP_KAN_v1


class FeaturesForDataset():
        FEATURES_CURRENT = ["IA", "IB", "IC"]
        # FEATURES_CURRENT = ["IA", "IC"]
        FEATURES_VOLTAGE = ["UA BB", "UB BB", "UC BB", "UN BB",
                            "UA CL", "UB CL", "UC CL", "UN CL",
                            "UAB CL","UBC CL","UCA CL"]
        # FEATURES_VOLTAGE = ["UA BB", "UB BB", "UC BB", "UN BB"]
        FEATURES = FEATURES_CURRENT.copy()
        FEATURES.extend(FEATURES_VOLTAGE)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        
        FEATURES_TARGET_WITH_FILENAME = ["file_name"]
        FEATURES_TARGET_WITH_FILENAME.extend(FEATURES_TARGET)

class CustomDataset_train(Dataset):
    # TODO: Решить проблему различия __getitem__ в первой строке с "iloc"
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
        start = self.indexes.loc[idx].name # было iloc, но в данном случае это некорректно
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
        target_sample = self.data.loc[target_index][FeaturesForDataset.FEATURES_TARGET]
        target = torch.tensor(target_sample, dtype=torch.float32) # было torch.long
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
        target_sample = self.data.loc[target_index][FeaturesForDataset.FEATURES_TARGET]
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


if __name__ == "__main__":
    FRAME_SIZE = 64 # 64
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
        # удаляю осциллограммы с данными, которые плохо поддаются БПФ (резкие выбросы)
        # Не актуальны, сейчас другие имена
        files_to_remove = [
            "Осц_1_26_1",
            "Осц_1_27_1",
            "Осц_1_28_1",
            "Осц_1_41_1",
            "Осц_1_26_2",
            "Осц_1_27_2",
            "Осц_1_28_2",
            "Осц_1_41_2",
        ]
        dt = dt[~dt["file_name"].str.strip().isin(files_to_remove)]
        
        # TODO: Нормальизацию стоит делать отдельно. Здесь я её закладываю временно
        Inom, Unom = 5, 100
        for name in FeaturesForDataset.FEATURES_CURRENT:
            dt[name] = dt[name] / (Inom*20)
        for name in FeaturesForDataset.FEATURES_VOLTAGE:
            dt[name] = dt[name] / (Unom*3)
                
        files_to_test = [
            "a3a1591bc548a7faf784728430499837_Bus 1 _event N1",  # "Осц_1_2_1",
            "4d8cbe560afd5f7f50ee476b9651f95d_Bus 1 _event N1",  # "Осц_1_8_1",
            "5718d1a5fc834efcfc0cf98f19485db7_Bus 1 _event N1",  # "Осц_1_15_1",
            "a2321d3375dbade8cd9afecfc2571a99_Bus 1 _event N1",  # "Осц_1_25_1",
            "5e01b6ca41575c55ebd68978d6f3227c_Bus 1 _event N1",  # "Осц_1_38_1",
            "9a26f30ebb02b8dd74a65c8c33c1dbcd_Bus 1 _event N1",  # "Осц_1_42_1",
              
            "a3a1591bc548a7faf784728430499837_Bus 2 _event N1"  # "Осц_1_2_2",
            "4d8cbe560afd5f7f50ee476b9651f95d_Bus 2 _event N1",  # "Осц_1_8_2",
            "5718d1a5fc834efcfc0cf98f19485db7_Bus 2 _event N1",  # "Осц_1_15_2",
            "a2321d3375dbade8cd9afecfc2571a99_Bus 2 _event N1",  # "Осц_1_25_2",
            "5e01b6ca41575c55ebd68978d6f3227c_Bus 2 _event N1",  # "Осц_1_38_2",
            "9a26f30ebb02b8dd74a65c8c33c1dbcd_Bus 2 _event N1"  # "Осц_1_42_2",
            # TODO: Добавить другие осциллограммы или подумать над тем, как лучше разделить их
        ]
        dt_test = dt[dt["file_name"].str.strip().isin(files_to_test)]
        dt_train = dt[~dt["file_name"].str.strip().isin(files_to_test)]

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

        target_series_train = dt_train[FeaturesForDataset.FEATURES_TARGET]
        target_series_test = dt_test[FeaturesForDataset.FEATURES_TARGET]

        # замена значений NaN на 0 в конце датафрейма
        for name in FeaturesForDataset.FEATURES_TARGET:
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
    
    # копия датафрейма для получения индексов начал фреймов для трейн
    dt_indexes_train = dt_train[FeaturesForDataset.FEATURES_TARGET_WITH_FILENAME]
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
    dt_train_no_event = train_indexes[train_indexes[["opr_swch", "abnorm_evnt", "emerg_evnt"]].all(axis=1) == 0]
    datasets_by_class = {
        'opr_swch': list(dt_train_opr_swch.index),
        'abnorm_evnt': list(dt_train_abnorm_evnt.index),
        'emerg_evnt': list(dt_train_emerg_evnt.index),
        'no_event': list(dt_train_no_event.index)
    }

    # копия датафрейма для получения индексов начал фреймов для тест
    dt_indexes_test = dt_test[FeaturesForDataset.FEATURES_TARGET_WITH_FILENAME]
    files_test = dt_indexes_test["file_name"].unique()
    test_indexes = pd.DataFrame()
    for file in files_test:
        # удаление последних FRAME_SIZE сэмплов в каждом файле
        df_file = dt_indexes_test[dt_indexes_test["file_name"] == file][
            :-FRAME_SIZE # Удаление последних FRAME_SIZE сэмплов в каждом файл, чтобы индексы не выходили за диапазон
        ]
        test_indexes = pd.concat((test_indexes, df_file))

    # скорректировать точку, я сейчас задал принудительно 32 точки назад (период наза, или 20мс)
    train_dataset = CustomDataset_train(dt_train, train_indexes, FRAME_SIZE, FRAME_SIZE-8)
    test_dataset = CustomDataset(dt_test, test_indexes, FRAME_SIZE, FRAME_SIZE-8)

    start_epoch = 0
    # !! создание новой !!
    name_model = "ConvMLP" # FftMLP , FftKAN
    # model = CONV_MLP_v2(
    # model = FFT_MLP(
    model = FFT_MLP_KAN_v1(
        FRAME_SIZE,
        channel_num=len(FeaturesForDataset.FEATURES),
        hidden_size=HIDDEN_SIZE,
        output_size=len(FeaturesForDataset.FEATURES_TARGET),
        device=device,
    )
    
    model.to(device)
    # # !! Загрузка модели из файла !!
    # filename_model = "ML_model/trained_models/model_ep2_tl0.3498_train1432.3669.pt"
    # model = torch.load(filename_model)
    # start_epoch = int(filename_model.split("ep")[1].split("_")[0])
    # model.eval()  # Set the model to evaluation mode

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss() # Лучше для многоклассовой пересекающейся модели
    
    all_losses = []

    sampler = FastBalancedBatchSampler(datasets_by_class, batch_size_per_class=BATCH_SIZE_PER_CLASS, num_batches=NUM_BATCHES)
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST)

    current_lr = 2*LEARNING_RATE
    for epoch in range(start_epoch,EPOCHS):
        if epoch // 10 == 0:
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

                targets = targets.to(device)
                batch = batch.to(device)

                output = model(batch)
                loss = criterion(output, targets)
                loss_sum += loss.item()
                all_losses.append(loss.item())
                message = (
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"LR={current_lr:.3e} "
                    f"Train loss: {(loss_sum / (i + 1)):.4f} "
                )
                t.set_postfix_str(s=message)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )
                optimizer.step()

                # print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}")

            model.eval()
            true_labels = []
            predicted_labels = []
            loss_test = 0.0
            # FIXME: разобраться, почему не сходятся массивы меток. [12460, 172]
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    loss_test += criterion(
                        outputs, labels.to(device).squeeze()
                    ).item()
                    # _, predicted = torch.max(outputs.data, 1)
                    predicted = outputs.data
                    true_labels.extend(labels.numpy())
                    predicted_labels.extend(predicted.cpu().numpy())
                loss_test /= len(test_dataloader)

            numpy_labels = np.array(predicted_labels) # промежуточные преобразования для ускорения
            predicted_labels_tensor = torch.from_numpy(numpy_labels)
            predicted_labels = torch.where(predicted_labels_tensor >= 0.5, torch.tensor(1), torch.tensor(0))
            
            # TODO: Разобраться с метриками и модернизировать их
            numpy_true_labels = np.array(true_labels) # промежуточные преобразования для ускорения
            true_labels_tensor = torch.from_numpy(numpy_true_labels)
            ba, f1 = [], []
            for i in range(len(FeaturesForDataset.FEATURES_TARGET)): # 'binary' для бинарной классификации на каждом классе
                true_binary = true_labels_tensor[:, i].flatten()
                pred_binary = predicted_labels[:, i].flatten()
                
                ba.append(balanced_accuracy_score(true_binary, pred_binary))
                f1.append(f1_score(true_binary, pred_binary, average='binary'))
            
            message_f1_ba = (
                             f"Prev. test loss: {loss_test:.4f} "
                             f"F1 / BA: {', '.join([f'{signal_name}: {f1_score:.4f}/{ba_score:.4f}' for signal_name, f1_score, ba_score in zip(FeaturesForDataset.FEATURES_TARGET, f1, ba)])} "
                             )
            print(message_f1_ba)
            # Hamming Loss - чем меньше, тем лучше
            hl = hamming_loss(true_labels, predicted_labels)
            # Jaccard Score - для многолейбловой задачи, 'samples' для подсчета по образцам - чем ближе к 1, тем лучше
            js = jaccard_score(true_labels, predicted_labels, average='samples')
            print(f"Hamming Loss: {hl}, Jaccard Score: {js}")
            
            torch.save(model, f"ML_model/trained_models/model_{name_model}_ep{epoch+1}_tl{loss_test:.4f}_train{loss_sum:.4f}.pt")
            model.train()
    pass
pass
