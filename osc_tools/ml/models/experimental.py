import torch
import torch.nn as nn
from osc_tools.ml.kan_conv.KANLinear import KANLinear
from osc_tools.ml.layers.complex_ops import cLeakyReLU, cSigmoid
from osc_tools.ml.models.cnn import Conv_3
from osc_tools.ml.utils import fft_calc, fft_calc_abs_angle, create_signal_group, create_line_group
from osc_tools.core.constants import Features

class CONV_MLP_v2(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear((24+32)*14, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, 4*hidden_size),
            nn.LeakyReLU(),
            nn.Linear(4*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(CONV_MLP_v2, self).__init__()
        self.conv32 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=32, stride=16),
            nn.LeakyReLU(True),
        )
        self.conv3 = Conv_3()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        X_sum = torch.zeros(x.size(0), 24+32, self.channel_num, device=x.device)
        for i in range(self.channel_num):
            x_i = x[:, i:i+1, :]
            x1 = self.conv32(x_i)
            x1 = x1.reshape(x1.size(0), -1)
            x3 = self.conv3(x_i[:, :, 32:])
            x3 = x3.reshape(x3.size(0), -1)
            # Объединение тензоров по оси 0
            x_i = torch.cat((x1, x3), dim=1)
            X_sum[:,:, i] = x_i
            
        X_sum = X_sum.reshape(X_sum.size(0), -1)  # Выравнивание тензора до 2 измерений
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(X_sum)
        x_abnorm_evnt = self.fc_abnorm_evnt(X_sum)
        x_emerg_evnt = self.fc_emerg_evnt(X_sum)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x

class CONV_COMPLEX_v1(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear((24+32)*14, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            cSigmoid()
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(CONV_COMPLEX_v1, self).__init__()
        self.conv32 = nn.Sequential(
            nn.Conv1d(1,8, kernel_size=32, stride=16, dtype=torch.cfloat),
            cLeakyReLU()
        )
        self.conv3 = Conv_3(useComplex=True)

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        currents = [Features.CURRENT_INDICES["IA"], Features.CURRENT_INDICES["IB"], Features.CURRENT_INDICES["IC"], Features.CURRENT_INDICES["IN"]]
        voltages_bb = [Features.VOLTAGE_PHAZE_BB_INDICES["UA BB"], Features.VOLTAGE_PHAZE_BB_INDICES["UB BB"], Features.VOLTAGE_PHAZE_BB_INDICES["UC BB"], Features.VOLTAGE_PHAZE_BB_INDICES["UN BB"]]
        voltages_cl = [Features.VOLTAGE_PHAZE_CL_INDICES["UA CL"], Features.VOLTAGE_PHAZE_CL_INDICES["UB CL"], Features.VOLTAGE_PHAZE_CL_INDICES["UC CL"], Features.VOLTAGE_PHAZE_CL_INDICES["UN CL"]]
        
        x_g1 = create_signal_group(x, currents, voltages_bb, device = x.device)
        x_g2 = create_signal_group(x, currents, voltages_cl, device = x.device)
        
        ic_L = [
            [Features.CURRENT_INDICES["IA"], Features.CURRENT_INDICES["IB"]],
            [Features.CURRENT_INDICES["IB"], Features.CURRENT_INDICES["IC"]],
            [Features.CURRENT_INDICES["IC"], Features.CURRENT_INDICES["IA"]]
        ]
        voltages_line_bb = [Features.VOLTAGE_LINE_BB_INDICES["UAB BB"], Features.VOLTAGE_LINE_BB_INDICES["UBC BB"], Features.VOLTAGE_LINE_BB_INDICES["UCA BB"]]
        voltages_line_cl = [Features.VOLTAGE_LINE_CL_INDICES["UAB CL"], Features.VOLTAGE_LINE_CL_INDICES["UBC CL"], Features.VOLTAGE_LINE_CL_INDICES["UCA CL"]]

        x_g3 = create_line_group(x, ic_L, voltages_line_bb, device=x.device)
        x_g4 = create_line_group(x, ic_L, voltages_line_cl, device=x.device)

        x_new = torch.cat((x_g1, x_g2, x_g3, x_g4), dim=1)
        
        X_sum = torch.zeros(x.size(0), 24+32, self.channel_num, device=x.device, dtype=torch.cfloat)
        for i in range(4+4+3+3): # По количествам сигналов в группах
            x_i = x_new[:, i:i+1, :]
            x1 = self.conv32(x_i)
            x1 = x1.reshape(x1.size(0), -1)
            x3 = self.conv3(x_i[:, :, 32:])
            x3 = x3.reshape(x3.size(0), -1)
            # Объединение тензоров по оси 0
            x_i = torch.cat((x1, x3), dim=1)
            X_sum[:,:, i] = x_i
            
        X_sum = X_sum.reshape(X_sum.size(0), -1)  # Выравнивание тензора до 2 измерений
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(X_sum)
        x_abnorm_evnt = self.fc_abnorm_evnt(X_sum)
        x_emerg_evnt = self.fc_emerg_evnt(X_sum)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x


class FFT_MLP(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(FFT_MLP, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc = nn.Sequential(
            nn.Linear(4*9*14, 4*hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(4*hidden_size, 4*hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(4*hidden_size, 2*hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(2*hidden_size, hidden_size),
            nn.LeakyReLU(0.05),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        ## ТРЕБУЕТСЯ сделать независимые выходы для КАЖДОГО класса 
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = torch.cat(fft_calc_abs_angle(x, count_harmonic = 8), dim=1)
        # Объединение тензоров по оси 0
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(x)
        x_abnorm_evnt = self.fc_abnorm_evnt(x)
        x_emerg_evnt = self.fc_emerg_evnt(x)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x

class FFT_MLP_KAN_v1(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # nn.Linear(hidden_size * 7, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(FFT_MLP_KAN_v1, self).__init__()
        
        # TODO: Расписать # KANLayer
        self.kan1 = KANLinear(in_features=4*9*14,
                             out_features=2*hidden_size,
                             grid_size=10,
                             spline_order=3,
                             scale_noise=0.01,
                             scale_base=1,
                             scale_spline=1,
                             base_activation=nn.SiLU,
                             grid_eps=0.02,
                             grid_range=[0,1])
        self.kan2 = KANLinear(in_features=2*hidden_size,
                             out_features=4*hidden_size,
                             grid_size=10,
                             spline_order=3,
                             scale_noise=0.01,
                             scale_base=1,
                             scale_spline=1,
                             base_activation=nn.SiLU,
                             grid_eps=0.02,
                             grid_range=[0,1])
        self.kan3 = KANLinear(in_features=4*hidden_size,
                             out_features=2*hidden_size,
                             grid_size=10,
                             spline_order=3,
                             scale_noise=0.01,
                             scale_base=1,
                             scale_spline=1,
                             base_activation=nn.SiLU,
                             grid_eps=0.02,
                             grid_range=[0,1])
        self.kan4 = KANLinear(in_features=2*hidden_size,
                             out_features=hidden_size,
                             grid_size=10,
                             spline_order=3,
                             scale_noise=0.01,
                             scale_base=1,
                             scale_spline=1,
                             base_activation=nn.SiLU,
                             grid_eps=0.02,
                             grid_range=[0,1])
        
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = torch.cat(fft_calc_abs_angle(x, count_harmonic = 8), dim=1)
        # Объединение тензоров по оси 0
        x = x.reshape(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        x = self.kan4(x)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(x)
        x_abnorm_evnt = self.fc_abnorm_evnt(x)
        x_emerg_evnt = self.fc_emerg_evnt(x)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x

class FFT_MLP_COMPLEX_v1(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            cSigmoid(),
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(FFT_MLP_COMPLEX_v1, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc = nn.Sequential(
            nn.Linear(2*9*14, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 2*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(2*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)

        
    def forward(self, x):
        ## ТРЕБУЕТСЯ сделать независимые выходы для КАЖДОГО класса 
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = torch.cat(fft_calc(x, count_harmonic = 8), dim=-1)
        # Объединение тензоров по оси 0
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(x)
        x_abnorm_evnt = self.fc_abnorm_evnt(x)
        x_emerg_evnt = self.fc_emerg_evnt(x)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x
