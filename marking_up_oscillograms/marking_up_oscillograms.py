import pandas as pd
import os
import sys
from tqdm import tqdm
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from raw_to_csv.raw_to_csv import RawToCSV
import numpy as np
from ML_model import model # FIXME: It doesn't work, I had to copy it to a folder, it's temporary
# from ML_model.train import FeaturesForDataset # FIXME: fix the call


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


class MarkingUpOscillograms(RawToCSV):
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self):
        super().__init__()
        pass

    def search_events_in_comtrade(self, csv_name='sorted_files.csv', ML_model_path='ML_model/trained_models/model_training.pt',
                                  norm_file_path='marking_up_oscillograms/norm_1600.csv'):
        """
        ...

        Args:
            ...
        """
        sorted_files_df = pd.DataFrame()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        model = torch.load(ML_model_path)
        model = model.to(device)
        FRAME_SIZE = 32 # TODO: receiving data from the model + protection by signal names is required (for the model to receive).
        
        # the signals used by this model need to be checked!!
        ml_all = []
        ml_current = [ 'IA', 'IB', 'IC']
        #ml_current = [ 'IA', 'IC'] # for MLOps dataset
        ml_all.extend(ml_current)
        ml_votage_BB = ['UA BB', 'UB BB', 'UC BB', 'UN BB']
        ml_all.extend(ml_votage_BB)
        ml_votage_CL = ['UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL']
        #ml_votage_CL = [] # for MLOps dataset
        ml_all.extend(ml_votage_CL)
        
        norm_csv = pd.read_csv(norm_file_path)
        norm_osc_tuple = tuple(norm_csv["name"].values)
        
        raw_files = sorted([file[:-4] for file in os.listdir(self.raw_path)
                            if 'cfg' in file])
        raw_files = [file for file in raw_files if file in norm_osc_tuple]
        
        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file in raw_files:
                df = self.create_one_df(self.raw_path + file + ".cfg", file + ".cfg")
                names_osc = df["file_name"].unique()
                for name_osc in names_osc:
                    count_events = {name : 0 for name in FeaturesForDataset.FEATURES_TARGET}
                    df_osc_one_bus = df[df["file_name"] == name_osc].copy()
                    df_osc = pd.DataFrame(columns=ml_all)
                    for name_cullum in ml_all:
                        if name_cullum in df_osc_one_bus.columns:
                            df_osc[name_cullum] = df_osc_one_bus[name_cullum]
                    
                    # Normalization
                    # TODO: think about how to define a section without extracting it from the name.
                    # example: "04024d46359f94ebfd8123b8514a23fa_Bus 2 _event N1"
                    bus_name = int((name_osc.split("_")[1]).split(" ")[1])
                    name_osc_i_bus = name_osc.split("_")[0]
                    norm_osc_row = norm_csv[norm_csv["name"] == name_osc_i_bus]
                    # TODO: Add secondary/primary processing
                    # TODO: Add secondary/primary processing
                    # Phase current
                    I_nom = 20 * float(norm_osc_row[f"{bus_name}Ip_base"])
                    for current in ml_current:
                        df_osc[current] = df_osc[current] / I_nom
                    # TODO: add zero current transformers
                    # TODO: Add a check that the markup has been implemented and it is not raw
                    # Voltage BusBar
                    U_nom_BB = 3 * float(norm_osc_row[f"{bus_name}Ub_base"])
                    for votage_BB in ml_votage_BB:
                        df_osc[votage_BB] = df_osc[votage_BB] / U_nom_BB
                    # Voltage CableLine
                    U_nom_CL = 3 * float(norm_osc_row[f"{bus_name}Uc_base"])
                    for votage_CL in ml_votage_CL:
                        df_osc[votage_CL] = df_osc[votage_CL] / U_nom_CL
                    
                    indexes = len(df_osc) - FRAME_SIZE + 1
                    df_osc.fillna(0, inplace=True)
                    for ind in range(indexes):
                        df_window = df_osc.iloc[ind:ind+FRAME_SIZE]
                        df_window_tensor = torch.tensor(df_window.values.astype(np.float32)).unsqueeze(0).to(device).float()
                        model_prediction = model(df_window_tensor)
                        for k, name in enumerate(FeaturesForDataset.FEATURES_TARGET):
                            count_events[name] += int(model_prediction[0][k] > 0.5)
                            if int(model_prediction[0][0] > 0.5) > 0 and k == 0: # opr_swch
                                pass
                            if int(model_prediction[0][1] > 0.5) > 0 and k == 1: # abnorm_evnt
                                pass
                            if int(model_prediction[0][2] > 0.5) > 0 and k == 2: # emerg_evnt
                                pass
                    # TODO: !!ДОПИСАТЬ!! обработку и сохранение результатов
                    
markingUpOscillograms = MarkingUpOscillograms()
markingUpOscillograms.search_events_in_comtrade(ML_model_path = "ML_model/trained_models/model_ep1_tl0.0956_train309.1770.pt")
