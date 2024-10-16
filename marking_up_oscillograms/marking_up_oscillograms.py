import pandas as pd
import os
import sys
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from raw_to_csv.raw_to_csv import RawToCSV
import numpy as np
from ML_model import model # FIXME: It doesn't work, I had to copy it to a folder, it's temporary
# from ML_model.train import FeaturesForDataset # FIXME: fix the call


class FeaturesForDataset():
    
        FEATURES_CURRENT = ["IA", "IB", "IC"]
        FEATURES_VOLTAGE_BB = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
        FEATURES_VOLTAGE_CL = ['UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL']
        
        FEATURES = FEATURES_CURRENT.copy()
        FEATURES.extend(FEATURES_VOLTAGE_BB)
        FEATURES.extend(FEATURES_VOLTAGE_CL)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"]
        FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        
        CURRENT_FOR_PLOT = ["IA", "IB", "IC"]
        VOLTAGE_FOR_PLOT = ['UA BB', 'UB BB', 'UC BB']
        ANALOG_SIGNALS_FOR_PLOT = CURRENT_FOR_PLOT.copy()
        ANALOG_SIGNALS_FOR_PLOT.extend(VOLTAGE_FOR_PLOT)


class MarkingUpOscillograms(RawToCSV):
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self):
        super().__init__()
        self.FRAME_SIZE = 32 # TODO: receiving data from the model + protection by signal names is required (for the model to receive).
        pass

    def search_events_in_comtrade(self, csv_name: str ='marking_up_oscillograms/sorted_files.csv', 
                                  ML_model_path: str='ML_model/trained_models/model_training.pt',
                                  norm_file_path: str='marking_up_oscillograms/norm_1600.csv'):
        """
        The function processes Comtrade files using a pre-prepared neural model and generates a CSV file with markup.
        !!Note!!: A huge amount of customization and refinement is required, the function is at an early stage.

        Args:
            csv_name (str): The csv file name.
            ML_model_path (str): The ML model path.
            norm_file_path (str): The normalization file path.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        model = torch.load(ML_model_path)
        model = model.to(device)
        
        ml_for_answer = ["file_name", "bus"]
        for name in FeaturesForDataset.FEATURES_TARGET:
            ml_for_answer.append(name + "_bool")
            ml_for_answer.append(name + "_count")
        
        sorted_files_df = pd.DataFrame(columns=ml_for_answer)
        # Adding an indication of the data type for columns with Boolean values
        for name in FeaturesForDataset.FEATURES_TARGET:
            sorted_files_df[name + "_bool"] = sorted_files_df[name + "_bool"].astype(bool)
        
        norm_csv = pd.read_csv(norm_file_path)
        norm_osc_tuple = tuple(norm_csv["name"].values)
        
        raw_files = sorted([file[:-4] for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        raw_files = [file for file in raw_files if file in norm_osc_tuple]
        
        # TODO: To speed up the algorithms, for a VERY long time
        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file in raw_files:
                df = self.create_one_df(self.raw_path + file + ".cfg", file + ".cfg")
                if df.empty:
                    continue # Protection empty df
                names_osc = df["file_name"].unique()
                for name_osc in names_osc:
                    count_events = {name : 0 for name in FeaturesForDataset.FEATURES_TARGET}
                    df_osc_one_bus = df[df["file_name"] == name_osc].copy()
                    df_osc = pd.DataFrame(columns=FeaturesForDataset.FEATURES)
                    for name_cullum in FeaturesForDataset.FEATURES:
                        if name_cullum in df_osc_one_bus.columns:
                            df_osc[name_cullum] = df_osc_one_bus[name_cullum]
                    
                    # Normalization
                    # TODO: think about how to define a section without extracting it from the name.
                    # example: "04024d46359f94ebfd8123b8514a23fa_Bus 2 _event N1"
                    try:
                        bus = int((name_osc.split("_")[1]).split(" ")[1])
                    except ValueError:
                        # TODO: As a rule, this is "_Diff" - write the processing
                        print(f"Warning: {name_osc} not found in norm_osc_tuple.")
                        continue
                    name_osc_i_bus = name_osc.split("_")[0]
                    norm_osc_row = norm_csv[norm_csv["name"] == name_osc_i_bus]
                    if (norm_osc_row["norm"].values[0] != "ДА"):
                        continue # пропуск не подготовленных наборов
                    # TODO: Add secondary/primary processing
                    # TODO: Add secondary/primary processing
                    # Phase current
                    I_nom = 20 * float(norm_osc_row[f"{bus}Ip_base"])
                    for current in FeaturesForDataset.FEATURES_CURRENT:
                        df_osc[current] = df_osc[current] / I_nom
                    # TODO: add zero current transformers
                    # TODO: Add a check that the markup has been implemented and it is not raw
                    # Voltage BusBar
                    U_nom_BB = 3 * float(norm_osc_row[f"{bus}Ub_base"])
                    for votage_BB in FeaturesForDataset.FEATURES_VOLTAGE_BB:
                        df_osc[votage_BB] = df_osc[votage_BB] / U_nom_BB
                    # Voltage CableLine
                    U_nom_CL = 3 * float(norm_osc_row[f"{bus}Uc_base"])
                    for votage_CL in FeaturesForDataset.FEATURES_VOLTAGE_CL:
                        df_osc[votage_CL] = df_osc[votage_CL] / U_nom_CL
                    
                    indexes = len(df_osc) - self.FRAME_SIZE + 1
                    df_osc.fillna(0, inplace=True)
                    for ind in range(indexes):
                        df_window = df_osc.iloc[ind:ind+self.FRAME_SIZE]
                        df_window_tensor = torch.tensor(df_window.values.astype(np.float32)).unsqueeze(0).to(device).float()
                        model_prediction = model(df_window_tensor)
                        for k, name in enumerate(FeaturesForDataset.FEATURES_TARGET):
                            count_events[name] += int(model_prediction[0][k] > 0.5)
                    
                    # TODO: Improve the preservation of results and its processing
                    events_predicted = {}
                    for name in FeaturesForDataset.FEATURES_TARGET:
                        events_predicted[f"{name}_bool"] = count_events[name] > 0
                        events_predicted[f"{name}_count"] = count_events[name]
                        
                    new_row = pd.DataFrame({
                        "file_name": [name_osc_i_bus],
                        "bus": [bus],
                        **{k: [v] for k, v in events_predicted.items()}
                    }, index=[0])

                    sorted_files_df = pd.concat([sorted_files_df, new_row], ignore_index=True)
                    # I save each line so as not to lose the value in case of an error
                    # TODO:to do it through try catch
                    sorted_files_df.to_csv(csv_name)
                pbar.update(1)

class ComtradePredictionAndPlotting(RawToCSV):
    def __init__(self, model_path: str, osc_name: str, uses_bus: str, strat_point: int, end_pont: int, norm_file_path: str, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device).to(self.device)
        self.norm_csv = pd.read_csv(norm_file_path)
        self.osc_name = osc_name
        self.start_point, self.end_point = strat_point, end_pont
        self.uses_bus = uses_bus
        self.FRAME_SIZE = 64
        self.ml_signals, self.ml_operational_switching, self.ml_abnormal_event, self.ml_emergency_event = self.get_short_names_ml_signals()

        # TODO: receiving data from the model + protection by signal names is required (for the model to receive).


    def predict(self):
        """
        Применение модели для предсказания на основе входных данных.
        """
        df = self.create_one_df(self.raw_path + self.osc_name + ".cfg", self.osc_name + ".cfg")
        if df.empty:
            return np.nan # Protection empty df
        analog_signal = {name : np.zeros(len(df)) for name in FeaturesForDataset.ANALOG_SIGNALS_FOR_PLOT}
        
        names_osc = df["file_name"].unique()
        name_osc = names_osc[int(self.uses_bus)-1]
        
        df_osc_one_bus = df[df["file_name"] == name_osc].copy()
        df_osc = pd.DataFrame(columns=FeaturesForDataset.FEATURES)
        for name_cullum in FeaturesForDataset.FEATURES:
            if name_cullum in df_osc_one_bus.columns:
                df_osc[name_cullum] = df_osc_one_bus[name_cullum]
                ######################## analog_signal #########################
                if name_cullum in FeaturesForDataset.ANALOG_SIGNALS_FOR_PLOT:
                    analog_signal[name_cullum] = df_osc_one_bus[name_cullum]
                    
                
        ######################## real_labels #########################
        df_osc_target = pd.DataFrame(columns=self.ml_signals)
        for name_cullum in self.ml_signals:
            if name_cullum in df_osc_one_bus.columns:
                df_osc_target[name_cullum] = df_osc_one_bus[name_cullum].fillna(0)
        real_labels = self.get_real_labels(df_osc_target)
            
        ######################## predict_labels #########################
            
        # Normalization
        # TODO: think about how to define a section without extracting it from the name.
        # example: "04024d46359f94ebfd8123b8514a23fa_Bus 2 _event N1"
        try:
            bus = int((name_osc.split("_")[1]).split(" ")[1])
        except ValueError:
            # TODO: As a rule, this is "_Diff" - write the processing
            print(f"Warning: {name_osc} not found in norm_osc_tuple.")
            return (np.nan, np.nan, np.nan)

        name_osc_i_bus = name_osc.split("_")[0]
        norm_osc_row = self.norm_csv[self.norm_csv["name"] == name_osc_i_bus]
        if (norm_osc_row["norm"].values[0] != "ДА"):
            return (np.nan, np.nan, np.nan) # пропуск не подготовленных наборов
        
        # TODO: Add secondary/primary processing
        # TODO: Add secondary/primary processing
        # Phase current
        I_nom = 20 * float(norm_osc_row[f"{bus}Ip_base"])
        for current in FeaturesForDataset.FEATURES_CURRENT:
            df_osc[current] = df_osc[current] / I_nom
        # TODO: add zero current transformers
        # TODO: Add a check that the markup has been implemented and it is not raw
        # Voltage BusBar
        U_nom_BB = 3 * float(norm_osc_row[f"{bus}Ub_base"])
        for votage_BB in FeaturesForDataset.FEATURES_VOLTAGE_BB:
            df_osc[votage_BB] = df_osc[votage_BB] / U_nom_BB
        # Voltage CableLine
        U_nom_CL = 3 * float(norm_osc_row[f"{bus}Uc_base"])
        for votage_CL in FeaturesForDataset.FEATURES_VOLTAGE_CL:
            df_osc[votage_CL] = df_osc[votage_CL] / U_nom_CL
        
        indexes = len(df_osc) - self.FRAME_SIZE + 1
        df_osc.fillna(0, inplace=True)
        predict_labels = {name : np.zeros(len(df_osc)) for name in FeaturesForDataset.FEATURES_TARGET}
        for ind in range(indexes):
            df_window = df_osc.iloc[ind:ind+self.FRAME_SIZE]
            df_window_tensor = torch.tensor(df_window.values.astype(np.float32)).unsqueeze(0).to(self.device).float()
            model_prediction = self.model(df_window_tensor)
            for k, name in enumerate(FeaturesForDataset.FEATURES_TARGET):
                predict_labels[name][self.FRAME_SIZE - 8 + ind] = int(model_prediction[0][k] > 0.5)
            
        return (analog_signal, predict_labels, real_labels)

    def get_real_labels(self, df: pd):
        """
        Получение реальной разметки событий на основе файла.
        """

        real_labels = {name : np.zeros(len(df)) for name in FeaturesForDataset.FEATURES_TARGET}
        # FIXME: think about how to make it independent of the manual task.
        # opr_swch
        for name_cullum in self.ml_operational_switching:
            if name_cullum in df.columns:
                real_labels["opr_swch"] = np.logical_or(real_labels["opr_swch"], df[name_cullum].fillna(0))
                
        # abnorm_evnt
        for name_cullum in self.ml_abnormal_event:
            if name_cullum in df.columns:
                real_labels["abnorm_evnt"] = np.logical_or(real_labels["abnorm_evnt"], df[name_cullum].fillna(0))
                
        # emerg_evnt
        for name_cullum in self.ml_emergency_event:
            if name_cullum in df.columns:
                real_labels["emerg_evnt"] = np.logical_or(real_labels["emerg_evnt"], df[name_cullum].fillna(0))
        
        return real_labels
        
    def plot_predictions_vs_real(self, start: int, end: int, analog_signal: dict, predict_labels: dict, real_labels: dict):
        time_range = np.arange(start, end) * 20 / 32 # FIXME: это можно через параметры частоты дискретизации, лишь пока вручную
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # DataFrame to store all data
        data = pd.DataFrame({'Time': time_range})

        # Plot currents
        for i, current in enumerate(FeaturesForDataset.CURRENT_FOR_PLOT):
            signal_data = analog_signal[current][start:end]
            axs[0].plot(time_range, signal_data, label=f'{current}', color=['yellow', 'green', 'red'][i])
            data[f'{current}'] = [value for value in analog_signal[current][start:end]]
        axs[0].legend()
        axs[0].set_title('Currents')
        axs[0].axhline(0, color='black', linestyle='--', linewidth=1)  # Добавляем ось через 0

        # Plot voltages
        for i, voltage in enumerate(FeaturesForDataset.VOLTAGE_FOR_PLOT):
            signal_data = analog_signal[voltage][start:end]
            axs[1].plot(time_range, signal_data, label=f'{voltage}', color=['yellow', 'green', 'red'][i])
            data[f'{voltage}'] = [value for value in analog_signal[current][start:end]]
        axs[1].legend()
        axs[1].set_title('Voltages')
        axs[1].axhline(0, color='black', linestyle='--', linewidth=1)  # Добавляем ось через 0

        # Plot discrete signals
        label_names = FeaturesForDataset.FEATURES_TARGET
        amp = [1, 2, 3]
        pred_colors = ['lightblue', 'lightcoral', 'lightgreen']
        real_colors = ['blue', 'red', 'green']
        for i, label_name in enumerate(label_names):
            pred_positions = [amp[i] if lbl else np.nan for lbl in predict_labels[label_name][start:end]]
            real_positions = [-amp[i] if lbl else np.nan for lbl in real_labels[label_name][start:end]]
            axs[2].set_ylim([-3.5, 3.5]) # FIXME: manually designed for 3 classes
            axs[2].scatter(time_range, pred_positions, color=pred_colors[i], marker='+', label=f'Pred {label_name}')
            axs[2].scatter(time_range, real_positions, color=real_colors[i], marker='o', label=f'Real {label_name}')
            axs[2].fill_between(time_range, 0, pred_positions, color=pred_colors[i], alpha=0.3, step='mid')
            axs[2].fill_between(time_range, 0, real_positions, color=real_colors[i], alpha=0.1, step='mid')
            data[f'Pred {label_name}'] = pred_positions
            data[f'Real {label_name}'] = real_positions

        # Adding labels for prediction and real areas
        axs[2].text(time_range[int(len(time_range) * 0.5)], 3, 'Pred', horizontalalignment='center', fontsize=12, color='black')
        axs[2].text(time_range[int(len(time_range) * 0.5)], -3, 'Real', horizontalalignment='center', fontsize=12, color='black')
        
        axs[2].legend()
        axs[2].set_title('Discrete Signals')
        axs[2].axhline(0, color='black', linestyle='--', linewidth=1)  # Добавляем ось через 0

        # Save all data to CSV
        data.to_csv(f'marking_up_oscillograms/{self.osc_name}_bus_{self.uses_bus}.csv', index=False)

        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(f'marking_up_oscillograms/{self.osc_name}_bus_{self.uses_bus}.png')
        plt.show()
    
    def process_and_plot(self,):
        """
        Основная функция для обработки данных из Comtrade, предсказания и визуализации.
        """
        # 1. Загрузка Comtrade, получение предсказаний и реальной разметки
        analog_signal, predict_labels, real_labels = self.predict()

        # 2. Построение осциллограмм
        self.plot_predictions_vs_real(self.start_point, self.end_point, analog_signal, predict_labels, real_labels)

# Пример использования:

features_for_dataset = FeaturesForDataset()  # Класс с характеристиками
# model_ConvMLP_ep100_tl0.1709_train18.7660
# model_FftMLP_ep100_tl0.2305_train56.6066

# 5e01b6ca41575c55ebd68978d6f3227c
# a3a1591bc548a7faf784728430499837
comtrade_predictor = ComtradePredictionAndPlotting(model_path = 'marking_up_oscillograms/model/model_FftKAN_ep100_tl0.1193_train79.6926.pt', 
                                                   osc_name='a3a1591bc548a7faf784728430499837',
                                                   uses_bus = '1',
                                                   strat_point = 1000, end_pont = 2000,
                                                   norm_file_path = 'marking_up_oscillograms/norm_1600_v2.1.csv', 
                                                   device = "cuda" if torch.cuda.is_available() else "cpu")
comtrade_predictor.process_and_plot()
pass
                 
# markingUpOscillograms = MarkingUpOscillograms()
# markingUpOscillograms.search_events_in_comtrade(ML_model_path = "ML_model/trained_models/model/model_FftKAN_ep100_tl0.1193_train79.6926.pt")
