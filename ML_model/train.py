import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import hamming_loss, jaccard_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from model import CONV_MLP, KAN_firrst


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
        
        # WEIGHT_IMPORTANCE_TARGET = {"normal": 0.1, "opr_swch": 1, "abnorm_evnt": 5, "emerg_evnt": 10}
        WEIGHT_IMPORTANCE_TARGET = {"opr_swch": 1, "abnorm_evnt": 3, "emerg_evnt": 5}

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
        # self.len_files = dict(
        #     self.data.groupby("file_name").count()["sample"].items()
        # )

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


class CustomSampler(torch.utils.data.Sampler):
    """Creates indexes to sample from sequentially spaced data
    with a given frame size from only one file"""

    def __init__(self, data_source, batch_size, frame_size, shuffle):
        self.data_source = data_source
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.shuffle = shuffle

        all_files = list(data_source.len_files.keys())
        start_indexes = []
        df = data_source.data
        for file in all_files:
            start_index_in_dataset = df.loc[(df["file_name"] == file)].index[0]
            len_selected_file = data_source.len_files[file]
            start_indexes.extend(
                list(
                    range(
                        start_index_in_dataset,
                        start_index_in_dataset
                        + len_selected_file
                        - frame_size
                        + 1,
                    )
                )
            )
        self.start_indexes = start_indexes
        self.length = len(start_indexes)

    def __iter__(self):
        for _ in range(self.length):
            batch = []
            for __ in range(self.batch_size):
                if len(self.start_indexes) > 0:
                    if self.shuffle:
                        selected_index = np.random.randint(
                            0, len(self.start_indexes)
                        )
                    else:
                        selected_index = 0
                    start_index = self.start_indexes.pop(selected_index)
                    batch.append(
                        np.arange(start_index, start_index + self.frame_size)
                    )
                else:
                    break
            if len(batch):
                yield np.array(batch)
            else:
                break

    def __len__(self):
        return self.length


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

    FRAME_SIZE = 32 # 64
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_TEST = 1024
    HIDDEN_SIZE = 40
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    MAX_GRAD_NORM = 10
    SEED = 42
    print(f"{BATCH_SIZE_TRAIN=}")

    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    file_csv = "ML_model/datset_simpl v1.csv"
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
        # TODO: массштабирование производится отдельно. Нормальизация в адекватных данных не должна требоваться
        # так как знечения симметричны относительно 0. 
        # Причём ранее числа тут преобразовывались в -1 / 0 / 1, что было странно очень.
        # std_scaler.fit(df_unscaled_train.to_numpy())
        # df_scaled_train = std_scaler.transform(df_unscaled_train.to_numpy())
        # df_scaled_test = std_scaler.transform(df_unscaled_test.to_numpy())
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

        # Пока без этого, я разделил на 4 разные столбца
        # записываем в target_frame значение, которого больше всего в окне FRAME_SIZE:
        # dt_train[FeaturesForDataset.FEATURES_TARGET] = (
        #     target_series_train.rolling(window=FRAME_SIZE, min_periods=1)
        #     .apply(lambda x: pd.Series(x).value_counts().idxmax(), raw=True)
        #     .shift(-FRAME_SIZE)
        # )
        # dt_test[FeaturesForDataset.FEATURES_TARGET] = (
        #     target_series_test.rolling(window=FRAME_SIZE, min_periods=1)
        #     .apply(lambda x: pd.Series(x).value_counts().idxmax(), raw=True)
        #     .shift(-FRAME_SIZE)
        # )

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

    ############ для быстрого тестирования кода !!!!!!!!!!
    # dt_train = dt_train.head(int(len(dt_train) * 0.1))
    # dt_test = dt_test.head(int(len(dt_test) * 0.1))
    ############ для быстрого тестирования кода !!!!!!!!!!
    
    
    # TODO: Добавить имена для всего класса
    # print("Train:")
    # print(dt_train[FeaturesForDataset.FEATURES_TARGET].value_counts())
    # print("Test:")
    # print(dt_test[FeaturesForDataset.FEATURES_TARGET].value_counts())

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

    labels_train = train_indexes[FeaturesForDataset.FEATURES_TARGET]
    
    # создание весов ценности для классов
    target_samples = {name: 0 for name in FeaturesForDataset.FEATURES_TARGET}
    all_samples = len(dt_train)
    class_weights = []
    for name in FeaturesForDataset.FEATURES_TARGET:
        target_samples[name] = dt_train[name].value_counts()[1]
        value = 1
        if target_samples[name] > 0:
            value = FeaturesForDataset.WEIGHT_IMPORTANCE_TARGET[name] * np.log10(all_samples / target_samples[name] + 1)
        class_weights.append(value)
    
    max_weight = max(class_weights)
    class_weights_normalized = [weight / max_weight for weight in class_weights]
    class_weights_tensor = torch.FloatTensor(class_weights_normalized).to(device)
    print(f"{class_weights_tensor = }")    

    # TODO: Не понимаю как это заставить работать для 4 столбцов    
    # class_weights = compute_class_weight(
    #     "balanced",
    #     classes=np.unique(labels_train.values.ravel()),  # Flatten the 2D array
    #     y=labels_train.values.ravel()  # Flatten the 2D array
    # )
    # class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    # print(f"{class_weights = }")

    train_dataset = CustomDataset(dt_train, train_indexes, FRAME_SIZE, int(FRAME_SIZE/2))
    test_dataset = CustomDataset(dt_test, test_indexes, FRAME_SIZE, int(FRAME_SIZE/2))

    start_epoch = 0
    # !! создание новой !!
    model = CONV_MLP(
    #model = KAN_firrst(
        FRAME_SIZE,
        channel_num=len(FeaturesForDataset.FEATURES),
        hidden_size=HIDDEN_SIZE,
        output_size=len(FeaturesForDataset.FEATURES_TARGET),
    )
    
    model.to(device)
    # !! Загрузка модели из файла !!
    # filename_model = "ML_model/trained_models/model_ep31_tl0.0061_train119.7887.pt"
    # model = torch.load(filename_model)
    # start_epoch = int(filename_model.split("ep")[1].split("_")[0])
    # model.eval()  # Set the model to evaluation mode

    #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
    )
    current_lr = LEARNING_RATE

    all_losses = []

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST)

    min_loss_test = 100
    # loss_test = 0.0
    # f1 = ""
    for epoch in range(start_epoch,EPOCHS):
        with tqdm(train_dataloader, unit=" batch", mininterval=3) as t:
            loss_sum = 0.0
            for i, (batch, targets) in enumerate(t):
                if (batch == None or targets == None):
                    # TODO: Защита от необрезанных массивов. Но сюда не должно попадать.
                    continue

                targets = targets.to(device)
                batch = batch.to(device)

                output = model(batch)

                # графики сигналов
                # index_in_batch = 0
                # plt.figure()
                # for j in range(5):
                #     plt.subplot(2, 3, j + 1)
                #     plt.plot(batch[index_in_batch, :, j].cpu().numpy())
                #     plt.gca().set_ylim(bottom=-2, top=2)
                # plt.show()

                loss = criterion(output, targets)
                loss_sum += loss.item()
                all_losses.append(loss.item())
                message = (
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"LR={current_lr:.3e} "
                    f"Train loss: {(loss_sum / (i + 1)):.4f} "
                )
                # if epoch > 0:
                #     message += (
                #         f"Prev. test loss: {loss_test:.4f} "
                #         #f"& F1: {', '.join([f'{score:.4f}' for score in f1])} "
                #         f"F1: {', '.join([f'{signal_name}: {score:.4f}' for signal_name, score in zip(FeaturesForDataset.FEATURES_TARGET, f1)])} "
                #         # f"& BA: {ba:.4f}" # TODO: Сделать ba мультимодальным, причём
                #     )
                t.set_postfix_str(s=message)

                # plt.plot(all_losses, marker="o", linestyle="-")
                # plt.grid(True)
                # plt.show()

                # smoothed plot
                # window_size = 20
                # smoothed_values = np.convolve(
                #     all_losses, np.ones(window_size) / window_size, mode="valid"
                # )
                # plt.figure()
                # plt.plot(smoothed_values, marker='o', linestyle='-')
                # plt.grid(True)
                # plt.show()

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

            torch.save(model, f"ML_model/trained_models/model_ep{epoch+1}_tl{loss_test:.4f}_train{loss_sum:.4f}.pt")
            if loss_test < min_loss_test:
                #torch.save(model, f"ML_model/trained_models/model_ep{epoch+1}_tl{loss_test:.4f}.pt")
                min_loss_test = loss_test
            model.train()
            # scheduler.step(loss_test) # constant LR
            current_lr = optimizer.param_groups[0]["lr"]
    pass
pass

# KAN_first model
# BATCH_SIZE_TRAIN = 128 
# LEARNING_RATE = 1e-04
# class_weights_tensor = tensor([0.1511, 0.3798, 1.0000], device='cuda:0')
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [03:02<00:00,  2.95 batch/s, Epoch 1/100 LR=1.000e-03 Train loss: 0.0424 ] 
# Prev. test loss: 0.0255 F1 / BA: opr_swch: 0.4188/0.8108, abnorm_evnt: 0.9885/0.9887, emerg_evnt: 0.3066/0.9017
# Hamming Loss: 0.13182457486254956, Jaccard Score: 0.577982355197545
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:34<00:00,  3.48 batch/s, Epoch 2/100 LR=1.000e-03 Train loss: 0.0179 ] 
# Prev. test loss: 0.0157 F1 / BA: opr_swch: 0.4473/0.8301, abnorm_evnt: 0.9867/0.9869, emerg_evnt: 0.6082/0.9483
# Hamming Loss: 0.10387418488684312, Jaccard Score: 0.6050249328730342
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.58 batch/s, Epoch 3/100 LR=1.000e-03 Train loss: 0.0140 ] 
# Prev. test loss: 0.0176 F1 / BA: opr_swch: 0.4249/0.8152, abnorm_evnt: 0.9872/0.9873, emerg_evnt: 0.6600/0.9704 
# Hamming Loss: 0.1113156885308784, Jaccard Score: 0.6113796189745556
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.59 batch/s, Epoch 4/100 LR=1.000e-03 Train loss: 0.0127 ]
# Prev. test loss: 0.0174 F1 / BA: opr_swch: 0.5237/0.8713, abnorm_evnt: 0.9341/0.9319, emerg_evnt: 0.6991/0.8962 
# Hamming Loss: 0.0950517836593786, Jaccard Score: 0.6095384221966501
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.58 batch/s, Epoch 5/100 LR=1.000e-03 Train loss: 0.0119 ] 
# Prev. test loss: 0.0226 F1 / BA: opr_swch: 0.4573/0.8362, abnorm_evnt: 0.9711/0.9712, emerg_evnt: 0.5346/0.8954 
# Hamming Loss: 0.10699399053829434, Jaccard Score: 0.6079657332821891
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.58 batch/s, Epoch 6/100 LR=1.000e-03 Train loss: 0.0118 ] 
# Prev. test loss: 0.0235 F1 / BA: opr_swch: 0.4098/0.8045, abnorm_evnt: 0.9695/0.9695, emerg_evnt: 0.5442/0.9077 
# Hamming Loss: 0.12655670630354174, Jaccard Score: 0.6088991177598773
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.59 batch/s, Epoch 7/100 LR=1.000e-03 Train loss: 0.0113 ] 
# Prev. test loss: 0.0228 F1 / BA: opr_swch: 0.4210/0.8127, abnorm_evnt: 0.9494/0.9485, emerg_evnt: 0.6063/0.9067 
# Hamming Loss: 0.12676128372330903, Jaccard Score: 0.6095000639304436
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.59 batch/s, Epoch 8/100 LR=1.000e-03 Train loss: 0.0112 ] 
# Prev. test loss: 0.0149 F1 / BA: opr_swch: 0.4242/0.8150, abnorm_evnt: 0.9720/0.9721, emerg_evnt: 0.4543/0.9753 
# Hamming Loss: 0.1255849635596471, Jaccard Score: 0.6128244470016622
# 100%|██████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.59 batch/s, Epoch 9/100 LR=1.000e-03 Train loss: 0.0113 ] 
# Prev. test loss: 0.0184 F1 / BA: opr_swch: 0.4786/0.8490, abnorm_evnt: 0.9613/0.9611, emerg_evnt: 0.4524/0.9732 
# Hamming Loss: 0.10929548651067639, Jaccard Score: 0.607543792353919
# 100%|█████████████████████████████████████████████████████████████████████████████| 537/537 [02:29<00:00,  3.59 batch/s, Epoch 10/100 LR=1.000e-03 Train loss: 0.0112 ] 
# Prev. test loss: 0.0137 F1 / BA: opr_swch: 0.4220/0.8136, abnorm_evnt: 0.9608/0.9605, emerg_evnt: 0.6247/0.9807 
# Hamming Loss: 0.12297660145761412, Jaccard Score: 0.6113156885308784
# 100%|█████████████████████████████████████████████████████████████████████████████| 537/537 [02:32<00:00,  3.53 batch/s, Epoch 11/100 LR=1.000e-03 Train loss: 0.0109 ]
# Prev. test loss: 0.0189 F1 / BA: opr_swch: 0.4169/0.8098, abnorm_evnt: 0.9572/0.9567, emerg_evnt: 0.5535/0.9514 
# Hamming Loss: 0.12832118654903465, Jaccard Score: 0.6096151387290628
# 100%|█████████████████████████████████████████████████████████████████████████████| 537/537 [02:32<00:00,  3.53 batch/s, Epoch 12/100 LR=1.000e-03 Train loss: 0.0109 ] 
# Prev. test loss: 0.0176 F1 / BA: opr_swch: 0.4161/0.8093, abnorm_evnt: 0.9635/0.9633, emerg_evnt: 0.5436/0.9743 
# Hamming Loss: 0.12742616033755275, Jaccard Score: 0.607825086306099
# 100%|█████████████████████████████████████████████████████████████████████████████| 537/537 [02:31<00:00,  3.55 batch/s, Epoch 13/100 LR=1.000e-03 Train loss: 0.0112 ] 
# Prev. test loss: 0.0181 F1 / BA: opr_swch: 0.4206/0.8130, abnorm_evnt: 0.9456/0.9444, emerg_evnt: 0.6314/0.9236 
# Hamming Loss: 0.12806546477432554, Jaccard Score: 0.6079018028385117