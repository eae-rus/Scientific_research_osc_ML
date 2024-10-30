import os
import sys
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

from raw_to_csv.raw_to_csv import RawToCSV
from ML_model import model # FIXME: It doesn't work, I had to copy it to a folder, it's temporary

class FeaturesForDataset:
    FEATURES_CURRENT = ["IA", "IB", "IC"]
    FEATURES_VOLTAGE_BB = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
    FEATURES_VOLTAGE_CL = ['UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL']
    FEATURES = FEATURES_CURRENT.copy()
    FEATURES.extend(FEATURES_VOLTAGE_BB)
    FEATURES.extend(FEATURES_VOLTAGE_CL)
    FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
    CURRENT_FOR_PLOT = ["IA", "IB", "IC"]
    VOLTAGE_FOR_PLOT = ['UA BB', 'UB BB', 'UC BB']
    ANALOG_SIGNALS_FOR_PLOT = CURRENT_FOR_PLOT.copy()
    ANALOG_SIGNALS_FOR_PLOT.extend(VOLTAGE_FOR_PLOT)

class ComtradeProcessor(RawToCSV):
    """Base class for Comtrade data processing."""
    
    def __init__(self, norm_file_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.norm_csv = pd.read_csv(norm_file_path)
        self.FRAME_SIZE = 64
    
    def load_model(self, model_path: str):
        """Loading the trained model."""
        self.model = torch.load(model_path, map_location=self.device).to(self.device)
    
    def normalize_signals(self, df_osc: pd.DataFrame, name_osc: str):
        """Normalization of currents and voltages."""
        try:
            bus = int(name_osc.split("_")[1].split(" ")[1])
        except ValueError:
            print(f"Warning: {name_osc} not found in norm_osc_tuple.")
            return df_osc
        
        name_osc_i_bus = name_osc.split("_")[0]
        norm_osc_row = self.norm_csv[self.norm_csv["name"] == name_osc_i_bus]
        
        if norm_osc_row["norm"].values[0] != "ДА":
            return df_osc  # Skipping untrained sets
        
        normalization_params = {
            "current": {
                "nominal": 20 * float(norm_osc_row[f"{bus}Ip_base"]),
                "features": FeaturesForDataset.FEATURES_CURRENT
            },
            "voltage_bb": {
                "nominal": 3 * float(norm_osc_row[f"{bus}Ub_base"]),
                "features": FeaturesForDataset.FEATURES_VOLTAGE_BB
            },
            "voltage_cl": {
                "nominal": 3 * float(norm_osc_row[f"{bus}Uc_base"]),
                "features": FeaturesForDataset.FEATURES_VOLTAGE_CL
            }
        }
        
        for key, params in normalization_params.items():
            for feature in params["features"]:
                if feature in df_osc.columns:
                    df_osc[feature] = df_osc[feature] / params["nominal"]
        
        return df_osc

class MarkingUpOscillograms(ComtradeProcessor):
    """Class for marking oscillograms."""

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
        """The process of marking oscillograms using a machine model."""
        self.load_model(ML_model_path)
        ml_for_answer = ["file_name", "bus"]
        for name in FeaturesForDataset.FEATURES_TARGET:
            ml_for_answer.append(f"{name}_bool")
            ml_for_answer.append(f"{name}_count")
        sorted_files_df = pd.DataFrame(columns=ml_for_answer)

        # Setting data types for Boolean columns
        for name in FeaturesForDataset.FEATURES_TARGET:
            sorted_files_df[f"{name}_bool"] = sorted_files_df[f"{name}_bool"].astype(bool)

        norm_osc_tuple = tuple(self.norm_csv["name"].values)
        raw_files = sorted([file[:-4] for file in os.listdir(self.raw_path) if 'cfg' in file])
        raw_files = [file for file in raw_files if file in norm_osc_tuple]

        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file in raw_files:
                df = self.create_one_df(os.path.join(self.raw_path, f"{file}.cfg"), f"{file}.cfg")
                if df.empty:
                    pbar.update(1)
                    continue

                names_osc = df["file_name"].unique()
                for name_osc in names_osc:
                    df_osc_one_bus = df[df["file_name"] == name_osc].copy()

                    # Initializing a DataFrame with zero values
                    df_osc = pd.DataFrame(0, index=np.arange(len(df_osc_one_bus)), columns=FeaturesForDataset.FEATURES)

                    # Filling in with real data, if present
                    for feature in FeaturesForDataset.FEATURES:
                        if feature in df_osc_one_bus.columns:
                            df_osc[feature] = df_osc_one_bus[feature].copy()

                    # Normalization of signals
                    df_osc = self.normalize_signals(df_osc, name_osc)

                    indexes = len(df_osc) - self.FRAME_SIZE + 1
                    df_osc.fillna(0, inplace=True)

                    # Initialization of predictions
                    predict_labels = {name: np.zeros(len(df_osc)) for name in FeaturesForDataset.FEATURES_TARGET}

                    # Creating windows
                    windows = []
                    target_indices = []
                    for ind in range(indexes):
                        window = df_osc.iloc[ind:ind + self.FRAME_SIZE].values.astype(np.float32)
                        windows.append(window)
                        # Defining an index for assigning a prediction
                        target_idx = self.FRAME_SIZE - 8 + ind
                        if 0 <= target_idx < len(predict_labels[list(predict_labels.keys())[0]]):
                            target_indices.append(target_idx)
                        else:
                            target_indices.append(None)  # Index out of range

                    # Convert windows to butches and pass them through the model
                    batch_size = self.batch_size
                    for i in range(0, len(windows), batch_size):
                        batch_windows = windows[i:i + batch_size]
                        batch_target_indices = target_indices[i:i + batch_size]

                        batch_tensor = torch.tensor(batch_windows).to(self.device).float()  # Shape: (batch, FRAME_SIZE, features)
                        #batch_tensor = batch_tensor.permute(0, 2, 1)  # You may need to rearrange the axes (batch, features, FRAME_SIZE)
                        with torch.no_grad():
                            model_prediction = self.model(batch_tensor)

                        predictions = (model_prediction > self.threshold).int().cpu().numpy()

                        for j, name in enumerate(FeaturesForDataset.FEATURES_TARGET):
                            for k, target_idx in enumerate(batch_target_indices):
                                if target_idx is not None:
                                    predict_labels[name][target_idx] = predictions[k][j]

                    # Double pass application (left and right)
                    left_predict_labels = {name: np.zeros(len(df_osc)) for name in FeaturesForDataset.FEATURES_TARGET}
                    right_predict_labels = {name: np.zeros(len(df_osc)) for name in FeaturesForDataset.FEATURES_TARGET}

                    for name in FeaturesForDataset.FEATURES_TARGET:
                        # From left to right
                        for i in range(len(predict_labels[name]) - 6):
                            window_sum = np.sum(predict_labels[name][i:i + 7]) > 4
                            if predict_labels[name][i] and (window_sum > 4):
                                left_predict_labels[name][i:i + 7] = 1

                        # From right to left
                        for i in range(len(predict_labels[name]) - 6):
                            window_sum = np.sum(predict_labels[name][-i-8:-i-1])
                            if predict_labels[name][-i-1] and(window_sum > 4):
                                right_predict_labels[name][-i-8:-i-1] = 1

                        # Объединение предсказаний
                        predict_labels[name] = np.logical_and(left_predict_labels[name], right_predict_labels[name])

                    # Counting events
                    count_events = {name: np.sum(predict_labels[name]) for name in FeaturesForDataset.FEATURES_TARGET}

                    # Creating a new row for the DataFrame
                    events_predicted = {}
                    for name in FeaturesForDataset.FEATURES_TARGET:
                        events_predicted[f"{name}_bool"] = count_events[name] > 0
                        events_predicted[f"{name}_count"] = count_events[name]

                    # secure attempt to define a section
                    bus = 1
                    try:
                        # Split the name of the waveform by the symbol "_" and extract the necessary parts
                        bus = int(name_osc.split("_")[1].split(" ")[1])
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
    """A class for predicting events and plotting."""

    def __init__(
        self, 
        model_path: str, 
        osc_name: str, 
        uses_bus: str, 
        strat_point: int, 
        end_point: int, 
        norm_file_path: str, 
        device: str = 'cuda', 
        f_networks: int = 50, 
        ADC_sampling_rate: int = 1600, 
        batch_size: int = 10000
    ):
        super().__init__(norm_file_path, device)
        self.load_model(model_path)
        self.osc_name = osc_name
        self.uses_bus = uses_bus
        self.start_point = strat_point
        self.end_point = end_point
        self.FRAME_SIZE = 64
        self.threshold = 0.8
        self.ml_signals, self.ml_operational_switching, self.ml_abnormal_event, self.ml_emergency_event = self.get_short_names_ml_signals()
        self.f_networks, self.ADC_sampling_rate = f_networks, ADC_sampling_rate
        self.batch_size = batch_size
        self.len_df = 0

    def predict(self):
        """Applying a model to predict based on input data."""
        df = self.create_one_df(os.path.join(self.raw_path, f"{self.osc_name}.cfg"), f"{self.osc_name}.cfg")
        if df.empty:
            return np.nan, np.nan, np.nan

        names_osc = df["file_name"].unique()
        try:
            name_osc = names_osc[int(self.uses_bus) - 1]
        except IndexError:
            print(f"Error: Bus {self.uses_bus} not found in file names.")
            return np.nan, np.nan, np.nan

        df_osc_one_bus = df[df["file_name"] == name_osc].copy()
        self.len_df = len(df_osc_one_bus)
        analog_signal = {name: np.zeros(self.len_df) for name in FeaturesForDataset.ANALOG_SIGNALS_FOR_PLOT}

        # Initializing a DataFrame with zero values
        df_osc = pd.DataFrame(0, index=np.arange(self.len_df), columns=FeaturesForDataset.FEATURES)

        # Filling in with real data, if present
        for feature in FeaturesForDataset.FEATURES:
            if feature in df_osc_one_bus.columns:
                df_osc[feature] = df_osc_one_bus[feature].copy()
                if feature in FeaturesForDataset.ANALOG_SIGNALS_FOR_PLOT:
                    analog_signal[feature] = df_osc_one_bus[feature].copy()

        # Normalization of signals
        df_osc = self.normalize_signals(df_osc, name_osc)

        indexes = self.len_df - self.FRAME_SIZE + 1
        df_osc.fillna(0, inplace=True)

        # Here is getting the real markup
        df_osc_target = pd.DataFrame()
        for feature in self.ml_signals:
            if feature in df_osc_one_bus.columns:
                df_osc_target[feature] = df_osc_one_bus[feature].copy()
        real_labels = self.get_real_labels(df_osc_target)

        # Initialization of predictions
        predict_labels = {name: np.zeros(self.len_df) for name in FeaturesForDataset.FEATURES_TARGET}

        # Creating windows
        windows = []
        target_indices = []
        for ind in range(indexes):
            window = df_osc.iloc[ind:ind + self.FRAME_SIZE].values.astype(np.float32)
            windows.append(window)
            # Defining an index for assigning a prediction
            target_idx = self.FRAME_SIZE - 8 + ind
            if 0 <= target_idx < len(predict_labels[list(predict_labels.keys())[0]]):
                target_indices.append(target_idx)
            else:
                target_indices.append(None)  # Index out of range

        # Convert windows to butches and pass them through the model
        batch_size = self.batch_size
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_target_indices = target_indices[i:i + batch_size]

            batch_tensor = torch.tensor(batch_windows).to(self.device).float()  # Shape: (batch, FRAME_SIZE, features)
            #batch_tensor = batch_tensor.permute(0, 2, 1)  # You may need to rearrange the axes (batch, features, FRAME_SIZE)
            with torch.no_grad():
                model_prediction = self.model(batch_tensor)

            predictions = (model_prediction > self.threshold).int().cpu().numpy()

            for j, name in enumerate(FeaturesForDataset.FEATURES_TARGET):
                for k, target_idx in enumerate(batch_target_indices):
                    if target_idx is not None:
                        predict_labels[name][target_idx] = predictions[k][j]

        # Double pass application (left and right)
        left_predict_labels = {name: np.zeros(self.len_df) for name in FeaturesForDataset.FEATURES_TARGET}
        right_predict_labels = {name: np.zeros(self.len_df) for name in FeaturesForDataset.FEATURES_TARGET}

        for name in FeaturesForDataset.FEATURES_TARGET:
            # From left to right
            for i in range(self.len_df - 6):
                window_sum = np.sum(predict_labels[name][i:i + 7])
                if predict_labels[name][i] and (window_sum > 3):
                    left_predict_labels[name][i:i + 7] = 1

            # From right to left
            for i in range(self.len_df - 6):
                window_sum = np.sum(predict_labels[name][-i-8:-i-1])
                if predict_labels[name][-i-1] and(window_sum > 3):
                    right_predict_labels[name][-i-8:-i-1] = 1

            # Combining predictions
            predict_labels[name] = np.logical_and(left_predict_labels[name], right_predict_labels[name])

        # Return of results
        return analog_signal, predict_labels, real_labels

    def get_real_labels(self, df: pd.DataFrame):
        """Getting real event markup based on a file."""
        real_labels = {name: np.zeros(self.len_df) for name in FeaturesForDataset.FEATURES_TARGET}
        
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

    def plot_predictions_vs_real(
        self, 
        start: int, 
        end: int, 
        analog_signal: dict, 
        predict_labels: dict, 
        real_labels: dict
    ):
        """Visualization of prediction waveforms and real markup."""
        time_range = np.arange(start, end) * 1000 / self.ADC_sampling_rate  # ms (sampling period = 1000 / ADC_sampling_rate)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.subplots_adjust(hspace=0.4)
        
        # Data Frame to save all data
        data = pd.DataFrame({'Time': time_range})
        
        # Plot currents
        for i, current in enumerate(FeaturesForDataset.CURRENT_FOR_PLOT):
            signal_data = analog_signal[current][start:end]
            axs[0].plot(time_range, signal_data, label=f'{current}', color=['yellow', 'green', 'red'][i])
            data[current] = analog_signal[current][start:end]
        axs[0].legend()
        axs[0].set_ylabel('Currents', rotation=90, labelpad=10, loc='center')
        axs[0].axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Plot voltages
        for i, voltage in enumerate(FeaturesForDataset.VOLTAGE_FOR_PLOT):
            signal_data = analog_signal[voltage][start:end]
            axs[1].plot(time_range, signal_data, label=f'{voltage}', color=['yellow', 'green', 'red'][i])
            data[voltage] = analog_signal[voltage][start:end]
        axs[1].legend()
        axs[1].set_ylabel('Voltages', rotation=90, labelpad=10, loc='center')
        axs[1].axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Plot discrete signals
        label_names = FeaturesForDataset.FEATURES_TARGET
        amp = [1, 2, 3]
        pred_colors = ['lightblue', 'lightcoral', 'lightgreen']
        real_colors = ['blue', 'red', 'green']
        
        for i, label_name in enumerate(label_names):
            pred_positions = [amp[i] if lbl else np.nan for lbl in predict_labels[label_name][start:end]]
            real_positions = [-amp[i] if lbl else np.nan for lbl in real_labels[label_name][start:end]]
            
            axs[2].scatter(time_range, pred_positions, color=pred_colors[i], marker='+', label=f'Pred {label_name}')
            axs[2].scatter(time_range, real_positions, color=real_colors[i], marker='o', label=f'Real {label_name}')
            axs[2].fill_between(time_range, 0, pred_positions, color=pred_colors[i], alpha=0.3, step='mid')
            axs[2].fill_between(time_range, 0, real_positions, color=real_colors[i], alpha=0.1, step='mid')
            
            data[f'Pred {label_name}'] = pred_positions
            data[f'Real {label_name}'] = real_positions
        
        # Setting up the third chart
        axs[2].set_ylim([-3.5, 3.5])
        axs[2].text(time_range[int(len(time_range) * 0.5)], 3, 'Pred', horizontalalignment='center', fontsize=12, color='black')
        axs[2].text(time_range[int(len(time_range) * 0.5)], -3, 'Real', horizontalalignment='center', fontsize=12, color='black')
        axs[2].legend()
        axs[2].set_ylabel('Discrete signals', rotation=90, labelpad=10, loc='center')
        axs[2].axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Saving all data in CSV
        data.to_csv(f'marking_up_oscillograms/{self.osc_name}_bus_{self.uses_bus}.csv', index=False)
        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig(f'marking_up_oscillograms/{self.osc_name}_bus_{self.uses_bus}.png', dpi=600)
        plt.show()

    def process_and_plot(self):
        """The main function for processing data from Comtrade, prediction and visualization."""
        analog_signal, predict_labels, real_labels = self.predict()
        if analog_signal is np.nan:
            print("Prediction returned NaN. Skipping plotting.")
            return
        self.plot_predictions_vs_real(self.start_point, self.end_point, analog_signal, predict_labels, real_labels)
        
# Usage example
if __name__ == "__main__":
    # Initializing classes
    # marking_up_oscillograms = MarkingUpOscillograms(norm_file_path="marking_up_oscillograms/norm_1600_v2.1.csv")
    # marking_up_oscillograms.search_events_in_comtrade(
    #     csv_name='marking_up_oscillograms/sorted_files.csv',
    #     ML_model_path="marking_up_oscillograms/model/model_FftKAN_ep100_tl0.1193_train79.6926.pt",
    # )
    
    
    # Initializing a class for prediction and plotting
    comtrade_predictor = ComtradePredictionAndPlotting(
        model_path='marking_up_oscillograms/model/model_ConvMLP_ep100_tl0.1709_train18.7660.pt',
        osc_name='a3a1591bc548a7faf784728430499837',
        uses_bus='2',
        strat_point=1000,
        end_point=2000,
        norm_file_path='marking_up_oscillograms/norm_1600_v2.1.csv'
    )
    # model_ConvMLP_ep100_tl0.1709_train18.7660
    # model_FftMLP_ep100_tl0.2305_train56.6066

    # 5e01b6ca41575c55ebd68978d6f3227c
    # a3a1591bc548a7faf784728430499837
    
    comtrade_predictor.process_and_plot()