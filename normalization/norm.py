import pandas as pd
import os
from tqdm.auto import tqdm
import comtrade


class NormOsc:
    def __init__(self,
                 raw_path,
                 f_rate,
                 stride=1
                 ):
        self.raw_path = raw_path
        self.window_size = f_rate // 50
        self.stride = stride
        self.df = pd.DataFrame(data=_result_columns)

        self.files = sorted([file for file in os.listdir(self.raw_path)
                            if "cfg" in file], reverse=False)
    
    def get_summary(self,):
        unread = []
        c = 0
        for file in tqdm(self.files):
            try:
                osc_df = comtrade.load_as_dataframe(self.raw_path + file)
            except Exception as ex:
                unread.append((file, ex))
                continue
            c += 1
            if c > 30:
                break
        with open('unread_files.txt', 'w') as f:
            for line in unread:
                f.write(f"{line}\n")


_result_columns = {'name': [], 'norm': [], 
                   '1Ub_PS': [], '1Ub_base': [], '1Ub_h1': [], '1Ub_hx': [], 
                   '1Uc_PS': [], '1Uc_base': [], '1Uc_h1': [], '1Uc_hx': [],
                   '1Ip_PS': [], '1Ip_base': [], '1Ip_h1': [], '1Ip_hx': [], 
                   '1Iz_PS': [], '1Iz_base': [], '1Iz_h1': [], '1Iz_hx': [],
                   '2Ub_PS': [], '2Ub_base': [], '2Ub_h1': [], '2Ub_hx': [], 
                   '2Uc_PS': [], '2Uc_base': [], '2Uc_h1': [], '2Uc_hx': [],
                   '2Ip_PS': [], '2Ip_base': [], '2Ip_h1': [], '2Ip_hx': [], 
                   '2Iz_PS': [], '2Iz_base': [], '2Iz_h1': [], '2Iz_hx': [],
                   '3Ub_PS': [], '3Ub_base': [], '3Ub_h1': [], '3Ub_hx': [],
                   '3Uc_PS': [], '3Uc_base': [], '3Uc_h1': [], '3Uc_hx': [],
                   '3Ip_PS': [], '3Ip_base': [], '3Ip_h1': [], '3Ip_hx': [],
                   '3Iz_PS': [], '3Iz_base': [], '3Iz_h1': [], '3Iz_hx': [],
                   '4Ub_PS': [], '4Ub_base': [], '4Ub_h1': [], '4Ub_hx': [],
                   '4Ip_PS': [], '4Ip_base': [], '4Ip_h1': [], '4Ip_hx': [],
                   '4Iz_PS': [], '4Iz_base': [], '4Iz_h1': [], '4Iz_hx': [],
                   '5Ip_PS': [], '5Ip_base': [], '5Ip_h1': [], '5Ip_hx': [],
                   '5Iz_PS': [], '5Iz_base': [], '5Iz_h1': [], '5Iz_hx': [],
                   '6Ip_PS': [], '6Ip_base': [], '6Ip_h1': [], '6Ip_hx': [],
                   '6Iz_PS': [], '6Iz_base': [], '6Iz_h1': [], '6Iz_hx': [],
                   'dId_PS': [], 'dId_base': [], 'dId_h1': []}
