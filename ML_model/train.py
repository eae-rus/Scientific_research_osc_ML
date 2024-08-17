import os
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

from model import CONV_MLP


class CustomDataset(Dataset):

    def __init__(self, dt, indexes, frame_size):
        self.data = dt
        self.indexes = indexes
        self.frame_size = frame_size
        # self.len_files = dict(
        #     self.data.groupby("file_name").count()["sample"].items()
        # )

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        start = self.indexes.iloc[idx].name
        frame = self.data.loc[start : start + self.frame_size - 1]
        # TODO: вынести "фичи" во вне
        # features_current = ["IA", "IB", "IC"]
        features_current = ["IA", "IC"]
        # features_voltage = ["UA BB", "UB BB", "UC BB", "UN BB",
        #                     "UA CL", "UB CL", "UC CL", "UN CL",
        #                     "UAB CL","UBC CL","UCA CL"]
        features_voltage = ["UA BB", "UB BB", "UC BB", "UN BB"]
        features = features_current.copy()
        features.extend(features_voltage)
        sample = frame[features]
        features_target = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        x = torch.tensor(
            sample.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        target = self.indexes.iloc[idx][features_target]
        target = torch.tensor(target, dtype=torch.float32) # было torch.long
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

    FRAME_SIZE = 64
    BATCH_SIZE_TRAIN = 128# 128
    BATCH_SIZE_TEST = 1024
    HIDDEN_SIZE = 40
    EPOCHS = 100
    LEARNING_RATE = 1e-4
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
        # features_current = ["IA", "IB", "IC"]
        features_current = ["IA", "IC"]
        # features_voltage = ["UA BB", "UB BB", "UC BB", "UN BB",
        #                     "UA CL", "UB CL", "UC CL", "UN CL",
        #                     "UAB CL","UBC CL","UCA CL"]
        features_voltage = ["UA BB", "UB BB", "UC BB", "UN BB"]
        features = features_current.copy()
        features.extend(features_voltage)
        
        for name in features_current:
            dt[name] = dt[name] / (Inom*20)
        for name in features_voltage:
            dt[name] = dt[name] / (Unom*3)
                
        files_to_test = [
            "a3a1591bc548a7faf784728430499837_Bus 1 _event N1",  # "Осц_1_2_1",
            "4d8cbe560afd5f7f50ee476b9651f95d_Bus 1 _event N1",  # "Осц_1_8_1",
            "5718d1a5fc834efcfc0cf98f19485db7_Bus 1 _event N1",  # "Осц_1_15_1",
            # "a2321d3375dbade8cd9afecfc2571a99_Bus 1",  # "Осц_1_25_1",
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
        
        df_unscaled_train = dt_train[features]
        df_unscaled_test = dt_test[features]
        std_scaler.fit(df_unscaled_train.to_numpy())
        df_scaled_train = std_scaler.transform(df_unscaled_train.to_numpy())
        df_scaled_test = std_scaler.transform(df_unscaled_test.to_numpy())
        df_scaled_train = pd.DataFrame(
            df_scaled_train,
            columns=df_unscaled_train.columns,
            index=df_unscaled_train.index,
        )
        df_scaled_test = pd.DataFrame(
            df_scaled_test,
            columns=df_unscaled_test.columns,
            index=df_unscaled_test.index,
        )

        dt_train = pd.concat(
            (dt_train.drop(features, axis=1), df_scaled_train), axis=1
        )
        dt_test = pd.concat(
            (dt_test.drop(features, axis=1), df_scaled_test), axis=1
        )

        dt_train.reset_index(drop=True, inplace=True)
        dt_test.reset_index(drop=True, inplace=True)

        features_target = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        target_series_train = dt_train[features_target]
        target_series_test = dt_test[features_target]

        # Пока без этого, я разделил на 4 разные столбца
        # записываем в target_frame значение, которого больше всего в окне FRAME_SIZE:
        # dt_train[features_target] = (
        #     target_series_train.rolling(window=FRAME_SIZE, min_periods=1)
        #     .apply(lambda x: pd.Series(x).value_counts().idxmax(), raw=True)
        #     .shift(-FRAME_SIZE)
        # )
        # dt_test[features_target] = (
        #     target_series_test.rolling(window=FRAME_SIZE, min_periods=1)
        #     .apply(lambda x: pd.Series(x).value_counts().idxmax(), raw=True)
        #     .shift(-FRAME_SIZE)
        # )

        # замена значений NaN на 0 в конце датафрейма
        for name in features_target:
            dt_train[name] = dt_train[name].fillna(0)  
            dt_train[name] = dt_train[name].astype(int)
            dt_train[name] = dt_train[name].astype(np.float32)
            dt_test[name] = dt_test[name].fillna(0)
            dt_test[name] = dt_test[name].astype(int)
            dt_test[name] = dt_test[name].astype(np.float32)
        # Замена значений NaN на 0 для пустых ячеек
        for name in features:
            dt_train[name] = dt_train[name].fillna(0)  
            dt_train[name] = dt_train[name].astype(int)
            dt_test[name] = dt_test[name].fillna(0)
            dt_test[name] = dt_test[name].astype(int)

        dt_train.to_csv(file_with_target_frame_train, index=False)
        dt_test.to_csv(file_with_target_frame_test, index=False)

    ############ для быстрого тестирования кода !!!!!!!!!!
    # dt_train = dt_train.head(int(len(dt_train) * 0.1))
    # dt_test = dt_test.head(int(len(dt_test) * 0.1))
    ############ для быстрого тестирования кода !!!!!!!!!!
    
    
    # TODO: Добавить имена для всего класса
    features_target = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
    features_target_with_fileName = ["file_name", "opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
    print("Train:")
    print(dt_train[features_target].value_counts())
    print("Test:")
    print(dt_test[features_target].value_counts())

    # копия датафрейма для получения индексов начал фреймов для трейн
    dt_indexes_train = dt_train[features_target_with_fileName]
    files_train = dt_indexes_train["file_name"].unique()
    train_indexes = pd.DataFrame()
    for file in files_train:
        # удаление последних FRAME_SIZE сэмплов в каждом файле
        df_file = dt_indexes_train[dt_indexes_train["file_name"] == file][
            :-FRAME_SIZE
        ]
        #df_file = dt_indexes_train.loc[dt_indexes_train["file_name"] == file].iloc[:-FRAME_SIZE]
        train_indexes = pd.concat((train_indexes, df_file))

    # копия датафрейма для получения индексов начал фреймов для тест
    dt_indexes_test = dt_test[features_target_with_fileName]
    files_test = dt_indexes_test["file_name"].unique()
    test_indexes = pd.DataFrame()
    for file in files_test:
        # удаление последних FRAME_SIZE сэмплов в каждом файле
        df_file = dt_indexes_test[dt_indexes_test["file_name"] == file][
            :-FRAME_SIZE
        ]
        test_indexes = pd.concat((test_indexes, df_file))

    labels_train = train_indexes[features_target]
    
    # TODO: Не понимаю как это заставить работать для 4 столбцов
    # class_weights = compute_class_weight(
    #     "balanced",
    #     classes=np.unique(labels_train.values.ravel()),  # Flatten the 2D array
    #     y=labels_train.values.ravel()  # Flatten the 2D array
    # )
    # class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    # print(f"{class_weights = }")
    
    class_weights_tensor = torch.rand(len(features_target)).to(device)
    print(f"{class_weights_tensor = }")

    train_dataset = CustomDataset(dt_train, train_indexes, FRAME_SIZE)
    test_dataset = CustomDataset(dt_test, test_indexes, FRAME_SIZE)

    model = CONV_MLP(
        FRAME_SIZE,
        channel_num=6, # TODO: сделать назначаемой автоматически len(features)
        hidden_size=HIDDEN_SIZE,
        output_size=4,# Было len(np.unique(labels_train)), Но у меня 4 независимых столбца
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
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
    for epoch in range(EPOCHS):
        with tqdm(train_dataloader, unit=" batch", mininterval=3) as t:
            loss_sum = 0.0
            for i, (batch, targets) in enumerate(t):

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
                if epoch > 0:
                    message += (
                        f"Prev. test loss: {loss_test:.4f} "
                        f"& F1: {', '.join([f'{score:.4f}' for score in f1])} "
                        # f"& BA: {ba:.4f}" # TODO: Сделать ba мультимодальным, причём
                    )
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
            
            f1 = f1_score(true_labels, predicted_labels, average=None)
            
            # TODO: Сделать balanced_accuracy_score мультимодальным, причём, когда одновременно разные столбцы могут быть равны 1
            # ba = balanced_accuracy_score(true_labels, predicted_labels)
            # print(
            #     f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Test loss: {loss_test:.4f}, F1: {f1}, Balanced Accuracy: {ba}"
            # )
            if loss_test < min_loss_test:
                torch.save(model, f"model_ep{epoch+1}_tl{loss_test:.4f}.pt")
                min_loss_test = loss_test
            model.train()
            # scheduler.step(loss_test) # constant LR
            current_lr = optimizer.param_groups[0]["lr"]
    pass
pass
# torch.save(model, "model40_3.pt")
# model = torch.load("model40_3.pt")

# BATCH_SIZE_TRAIN = 128
# LEARNING_RATE = 1e-04
