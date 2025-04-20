import sys
import os
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from kan_convolutional.KANLinear import KANLinear, KANLinearComplex


# Требуется разобраться с базовой моделью KANLinear, у меня не работала норально.
# У них можно график строить заполениня и другие фичи
# from kan import *
# KAN

####################
# Общшие классы
####################

class cLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.05):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, input):
        amp = torch.abs(input) # TODO: наверное можно как-то сразу, надо изучить, "что есть" в этих числах
        # можно будет сделать порог сдвигаемым
        amp_LeakyReLU = torch._C._nn.leaky_relu_(amp, self.negative_slope)
        # позитив и негатив - для создания полуплоскости (можно будет сделать их адаптивными)
        angle = torch.angle(input)
        angle_LeakyReLU = torch._C._nn.leaky_relu_(angle, self.negative_slope)
        
        real = amp_LeakyReLU * torch.cos(angle_LeakyReLU)
        imag = amp_LeakyReLU * torch.sin(angle_LeakyReLU)
        return torch.complex(real, imag)

class cSigmoid(nn.Module):
    # TODO: На будущее стоит сделать новый орган, который будет адаптивно учитывать границы угла, а не "зону"
    # Актуально для выходной переменной
    
    def forward(self, input):
        return torch.sigmoid(input.real)

class cMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(cMaxPool1d, self).__init__()
        self.real_pool = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.imag_pool = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        # Разделение на действительную и мнимую части
        real = x.real
        imag = x.imag

        # Применение MaxPool1d отдельно к действительной и мнимой частям
        pooled_real = self.real_pool(real)
        pooled_imag = self.imag_pool(imag)

        # Объединение действительной и мнимой частей обратно в комплексное число
        return torch.complex(pooled_real, pooled_imag)

class Features():
        CURRENT = {"IA": 0, "IB": 1, "IC": 2, "IN": -1} # "-1" их пока нет в базе
        CURRENT_FOR_LINE = {"IA": 0, "IB": 1, "IC": 2}
        
        VOLTAGE_PHAZE_BB = {"UA BB" : 3, "UB BB" : 4, "UC BB" : 5, "UN BB" : 6}
        VOLTAGE_PHAZE_CL = {"UA CL" : 7, "UB CL" : 8, "UC CL" : 9, "UN CL" : 10}

        VOLTAGE_LINE_BB = {"UAB BB" : -1,"UBC BB" : -1,"UCA BB" : -1}
        VOLTAGE_LINE_CL = {"UAB CL" : 11,"UBC CL": 12,"UCA CL": 13}

def create_conv_block(in_channels, out_channels, maxPool_size = 2, kernel_size=3, stride=1, padding=1, padding_mode="circular", useComplex=False):
    if useComplex:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dtype=torch.cfloat)
        return nn.Sequential(conv, cLeakyReLU(), cMaxPool1d(kernel_size=maxPool_size, stride=maxPool_size))
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
    return nn.Sequential(conv, nn.LeakyReLU(True), nn.MaxPool1d(kernel_size=maxPool_size, stride=maxPool_size))

class Conv_3(nn.Module):
    def __init__(self, useComplex=False):
        super(Conv_3, self).__init__()
        # TODO: Добавить задание параметров
        self.layer = nn.Sequential(
            create_conv_block(1, 16, useComplex=useComplex), # 32*16 -> 16*16
            create_conv_block(16, 32, useComplex=useComplex), # 16*32 -> 8*32
            create_conv_block(32, 32, maxPool_size = 8, useComplex=useComplex) # 8*32 -> 1*32
        )

    def forward(self, x):
        # FIXME: Не задаётся "devise" и от этого падает при расчёте.
        return self.layer(x)

class cDropout1d(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Создаем маску зануления как тензор float/double
        mask = (torch.rand(x.shape[:-1], device=x.device) >= self.p).to(x.real.dtype)
        mask = mask.unsqueeze(-1)  # Добавляем измерение для применения к complex
        
        # Применяем маску одновременно к real и imag частям
        real = x.real * mask
        imag = x.imag * mask
        return torch.complex(real, imag)

####################
# Общшие функции
####################

def cubic_interpolate(tensor, output_size=64):
    """
    Интерполяция данных до фиксированного числа точек с использованием кубической интерполяции.
    
    Параметры:
    tensor: Tensor размера (Batch, Channels, Points)
    output_size: Требуемое количество точек (по умолчанию 64)
    
    Возвращает:
    Интерполированный тензор размера (Batch, Channels, output_size)
    """
    #batch_size, channels, input_points = tensor.shape
    
    # Применяем интерполяцию только по третьему измерению (Points)
    interpolated_tensor = F.interpolate(
        tensor, 
        size=(output_size,),  # Интерполируем по размеру точек (Points)
        mode='bicubic',  # Кубическая интерполяция
        align_corners=True
    )

    return interpolated_tensor

def fft_calc(input, count_harmonic=1):
        prev_signals = torch.fft.rfft(input[:, :, :32])
        current_signals = torch.fft.rfft(input[:, :, 32:])
        
        return (prev_signals[:, :, :count_harmonic+1], current_signals[:, :, :count_harmonic+1])

def fft_calc_abs_angle(input, count_harmonic=1):
        prev_signals = torch.fft.rfft(input[:, :, :32])
        current_signals = torch.fft.rfft(input[:, :, 32:])
        
        prev_signals = prev_signals[:, :, :count_harmonic+1]
        current_signals = current_signals[:, :, :count_harmonic+1]
        
        return (torch.abs(prev_signals), torch.abs(current_signals), torch.angle(current_signals), torch.angle(current_signals))

def create_signal_group(x, currents, voltages, device = "cpu"):
    group = torch.zeros(x.size(0), len(currents), x.size(2), device=device, dtype=torch.cfloat)
    for i, (ic, iv) in enumerate(zip(currents, voltages)):
        if ic != -1 and iv != -1:
            group[:, i, :] = x[:, ic, :] + 1j * x[:, iv, :]
        elif ic == -1:
            group[:, i, :] = 1j * x[:, iv, :]
        else:
            group[:, i, :] = x[:, ic, :]
    return group

def create_line_group(x, ic_L, voltages, device="cpu"):
        group = torch.zeros(x.size(0), len(ic_L), x.size(2), device=device, dtype=torch.cfloat)
        for i, (ic_pair, iv) in enumerate(zip(ic_L, voltages)):
            group[:, i, :] = x[:, ic_pair[0], :] - x[:, ic_pair[1], :] + 1j * x[:, iv, :]
        return group

#####################
# МОДЕЛИ
#####################

class PDR_MLP_v1(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, channel_num, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(channel_num, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        def forward(self, x):
            return self.layer(x)
    def __init__(
        self, frame_size, channel_num=4, device = None,
    ):
        self.channel_num = channel_num
        self.device = device
        super(PDR_MLP_v1, self).__init__()
        self.fc = self.Head_fc(self.channel_num, hidden_size=20)

    def forward(self, x):
        # x: (batch_size, *, *) — например (1178, 4, 1) или (1178, 1, 4)
        x = x.flatten(start_dim=1)  # → (batch_size, feature_dim), здесь feature_dim=4*1=4
        return self.fc(x)







#####################
# СТАРЫЕ
#####################


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
        X_sum = torch.zeros(x.size(0), 24+32, self.channel_num, device=self.device)
        for i in range(self.channel_num):
            x_i = x[:, i:i+1, :]
            x1 = self.conv32(x_i)
            x1 = x1.reshape(x1.size(0), -1)
            x3 = self.conv3(x_i[:, :, 32:])
            x3 = x3.reshape(x3.size(0), -1)
            # Concatenate tensors along axis 0
            x_i = torch.cat((x1, x3), dim=1)
            X_sum[:,:, i] = x_i
            
        X_sum = X_sum.reshape(X_sum.size(0), -1)  # Flatten the tensor to 2 dimensions
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
        
        currents = [Features.CURRENT["IA"], Features.CURRENT["IB"], Features.CURRENT["IC"], Features.CURRENT["IN"]]
        voltages_bb = [Features.VOLTAGE_PHAZE_BB["UA BB"], Features.VOLTAGE_PHAZE_BB["UB BB"], Features.VOLTAGE_PHAZE_BB["UC BB"], Features.VOLTAGE_PHAZE_BB["UN BB"]]
        voltages_cl = [Features.VOLTAGE_PHAZE_CL["UA CL"], Features.VOLTAGE_PHAZE_CL["UB CL"], Features.VOLTAGE_PHAZE_CL["UC CL"], Features.VOLTAGE_PHAZE_CL["UN CL"]]
        
        x_g1 = create_signal_group(x, currents, voltages_bb, device = self.device)
        x_g2 = create_signal_group(x, currents, voltages_cl, device = self.device)
        
        ic_L = [
            [Features.CURRENT["IA"], Features.CURRENT["IB"]],
            [Features.CURRENT["IB"], Features.CURRENT["IC"]],
            [Features.CURRENT["IC"], Features.CURRENT["IA"]]
        ]
        voltages_line_bb = [Features.VOLTAGE_LINE_BB["UAB BB"], Features.VOLTAGE_LINE_BB["UBC BB"], Features.VOLTAGE_LINE_BB["UCA BB"]]
        voltages_line_cl = [Features.VOLTAGE_LINE_CL["UAB CL"], Features.VOLTAGE_LINE_CL["UBC CL"], Features.VOLTAGE_LINE_CL["UCA CL"]]

        x_g3 = create_line_group(x, ic_L, voltages_line_bb, device=self.device)
        x_g4 = create_line_group(x, ic_L, voltages_line_cl, device=self.device)

        x_new = torch.cat((x_g1, x_g2, x_g3, x_g4), dim=1)
        
        X_sum = torch.zeros(x.size(0), 24+32, self.channel_num, device=self.device, dtype=torch.cfloat)
        for i in range(4+4+3+3): # По количествам сигналов в группах
            x_i = x_new[:, i:i+1, :]
            x1 = self.conv32(x_i)
            x1 = x1.reshape(x1.size(0), -1)
            x3 = self.conv3(x_i[:, :, 32:])
            x3 = x3.reshape(x3.size(0), -1)
            # Concatenate tensors along axis 0
            x_i = torch.cat((x1, x3), dim=1)
            X_sum[:,:, i] = x_i
            
        X_sum = X_sum.reshape(X_sum.size(0), -1)  # Flatten the tensor to 2 dimensions
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
        # Concatenate tensors along axis 0
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
        # Concatenate tensors along axis 0
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
        # Concatenate tensors along axis 0
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(x)
        x_abnorm_evnt = self.fc_abnorm_evnt(x)
        x_emerg_evnt = self.fc_emerg_evnt(x)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x

class FFT_MLP_COMPLEX_v2(nn.Module):
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
        super(FFT_MLP_COMPLEX_v2, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_level = nn.Sequential(
            nn.Linear(2*1*14, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 2*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(2*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(40*9, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        # Преобразование x так, чтобы каналы были последней размерностью
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Вычисляем FFT с количеством гармоник 8
        count_harmonic = 8
        x_previous, x_current = fft_calc(x, count_harmonic=count_harmonic)

        # Переменная для сохранения промежуточных расчетов
        x_level_1 = []

        # Обработка каждой гармоники по отдельности
        for i in range(count_harmonic+1):  # Пробегаем по каждой гармонике
            x1 = x_previous[:, :, i]  # Извлекаем i-ю гармонику
            x2 = x_current[:, :, i]  # Извлекаем i-ю гармонику
            
            # Объединяем в один тензор с последующим разворачиванием
            x_combined = torch.cat((x1, x2), dim=-1)  # Склеиваем по последней размерности
            x_combined = x_combined.reshape(x_combined.size(0), -1)  # Выравнивание
            x_combined = self.fc_level(x_combined)  # Пропускаем через fully connected сеть
            x_level_1.append(x_combined)  # Сохраняем результат

        # Объединяем все рассчитанные гармоники вдоль нового измерения
        x_level_1 = torch.stack(x_level_1, dim=1)

        # Последующий слой обработки, который будет принимать x_level_1
        # Здесь можно добавить новый слой, аналогичный вашему текущему fc
        # Для примера, допустим, что будет еще один слой, который объединяет результаты
        x_level_1 = x_level_1.reshape(x_level_1.size(0), -1)  # Выравнивание
        x_level_1 = self.fc(x_level_1)

        # Создаем выходы для каждого класса
        x_opr_swch = self.fc_opr_swch(x_level_1)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_level_1)
        x_emerg_evnt = self.fc_emerg_evnt(x_level_1)

        # Объединяем выходы в один тензор
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x

class FFT_MLP_COMPLEX_v3(nn.Module):
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
        super(FFT_MLP_COMPLEX_v3, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_level = nn.Sequential(
            nn.Linear(2*1*14, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(4*hidden_size, 2*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(2*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size//2, hidden_size//4, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(9*hidden_size//4, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        # Преобразование x так, чтобы каналы были последней размерностью
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Вычисляем FFT с количеством гармоник 8
        count_harmonic = 8
        x_previous, x_current = fft_calc(x, count_harmonic=count_harmonic)

        # Переменная для сохранения промежуточных расчетов
        x_level_1 = []

        # Обработка каждой гармоники по отдельности
        for i in range(count_harmonic+1):  # Пробегаем по каждой гармонике
            x1 = x_previous[:, :, i]  # Извлекаем i-ю гармонику
            x2 = x_current[:, :, i]  # Извлекаем i-ю гармонику
            
            # Объединяем в один тензор с последующим разворачиванием
            x_combined = torch.cat((x1, x2), dim=-1)  # Склеиваем по последней размерности
            x_combined = x_combined.reshape(x_combined.size(0), -1)  # Выравнивание
            x_combined = self.fc_level(x_combined)  # Пропускаем через fully connected сеть
            x_level_1.append(x_combined)  # Сохраняем результат

        # Объединяем все рассчитанные гармоники вдоль нового измерения
        x_level_1 = torch.stack(x_level_1, dim=1)

        # Последующий слой обработки, который будет принимать x_level_1
        # Здесь можно добавить новый слой, аналогичный вашему текущему fc
        # Для примера, допустим, что будет еще один слой, который объединяет результаты
        x_level_1 = x_level_1.reshape(x_level_1.size(0), -1)  # Выравнивание
        x_level_1 = self.fc(x_level_1)

        # Создаем выходы для каждого класса
        x_opr_swch = self.fc_opr_swch(x_level_1)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_level_1)
        x_emerg_evnt = self.fc_emerg_evnt(x_level_1)

        # Объединяем выходы в один тензор
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x

class FFT_MLP_COMPLEX_v4(nn.Module):
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
        self.harmonic_count = 17

        super(FFT_MLP_COMPLEX_v4, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        # Создаем массив слоев fc, по одному для каждой гармоники
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*1*14, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
            ) for _ in range(self.harmonic_count)  # Для каждой гармоники
        ])
        self.fc = nn.Sequential(
            nn.Linear(self.harmonic_count*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        # Преобразование x так, чтобы каналы были последней размерностью
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Вычисляем FFT с количеством гармоник
        x_previous, x_current = fft_calc(x, count_harmonic=self.harmonic_count-1)

        # Переменная для сохранения промежуточных расчетов
        x_level_1 = []

        # Обработка каждой гармоники по отдельности
        for i in range(self.harmonic_count):  # Пробегаем по каждой гармонике
            x1 = x_previous[:, :, i]  # Извлекаем i-ю гармонику
            x2 = x_current[:, :, i]  # Извлекаем i-ю гармонику
            
            # Объединяем в один тензор с последующим разворачиванием
            x_combined = torch.cat((x1, x2), dim=-1)  # Склеиваем по последней размерности
            x_combined = x_combined.reshape(x_combined.size(0), -1)  # Выравнивание
            x_combined = self.fc_layers[i](x_combined)  # Пропускаем через fully connected сеть
            x_level_1.append(x_combined)  # Сохраняем результат

        # Объединяем все рассчитанные гармоники вдоль нового измерения
        x_level_1 = torch.stack(x_level_1, dim=1)

        # Последующий слой обработки, который будет принимать x_level_1
        # Здесь можно добавить новый слой, аналогичный вашему текущему fc
        # Для примера, допустим, что будет еще один слой, который объединяет результаты
        x_level_1 = x_level_1.reshape(x_level_1.size(0), -1)  # Выравнивание
        x_level_1 = self.fc(x_level_1)

        # Создаем выходы для каждого класса
        x_opr_swch = self.fc_opr_swch(x_level_1)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_level_1)
        x_emerg_evnt = self.fc_emerg_evnt(x_level_1)

        # Объединяем выходы в один тензор
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x

class CONV_AND_FFT_COMPLEX_v1(nn.Module):
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
        self.harmonic_count = 3

        super(CONV_AND_FFT_COMPLEX_v1, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*1*14, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
            ) for _ in range(self.harmonic_count)  # Для каждой гармоники
        ])
        self.conv3 = Conv_3()
        
        self.fc = nn.Sequential(
            nn.Linear(self.harmonic_count*hidden_size + 32*14, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
        
    def forward(self, x):
        # Преобразование x так, чтобы каналы были последней размерностью
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Вычисляем FFT с количеством гармоник
        x_previous, x_current = fft_calc(x, count_harmonic=self.harmonic_count-1)

        # Переменная для сохранения промежуточных расчетов
        x_level_1 = []

        # Обработка каждой гармоники по отдельности
        for i in range(self.harmonic_count):  # Пробегаем по каждой гармонике
            x1 = x_previous[:, :, i]  # Извлекаем i-ю гармонику
            x2 = x_current[:, :, i]  # Извлекаем i-ю гармонику
            
            # Объединяем в один тензор с последующим разворачиванием
            x_combined = torch.cat((x1, x2), dim=-1)  # Склеиваем по последней размерности
            x_combined = x_combined.reshape(x_combined.size(0), -1)  # Выравнивание
            x_combined = self.fc_layers[i](x_combined)  # Пропускаем через fully connected сеть
            x_level_1.append(x_combined)  # Сохраняем результат
        
        # Объединяем все рассчитанные гармоники вдоль нового измерения
        x_level_1 = torch.stack(x_level_1, dim=1)

        # Последующий слой обработки, который будет принимать x_level_1
        # Здесь можно добавить новый слой, аналогичный вашему текущему fc
        # Для примера, допустим, что будет еще один слой, который объединяет результаты
        x_level_1 = x_level_1.reshape(x_level_1.size(0), -1)  # Выравнивание
        
        x_conv = torch.zeros(x.size(0), 32, self.channel_num, device=self.device)
        for i in range(self.channel_num):
            x_i = x[:, i:i+1, :]
            x3 = self.conv3(x_i[:, :, 32:])
            x3 = x3.reshape(x3.size(0), -1)
            x_conv[:,:, i] = x3
        
        x_conv = x_conv.reshape(x_conv.size(0), -1)  # Flatten the tensor to 2 dimensions
        # Преобразуем тензор x_conv в комплексный тензор
        x_conv_complex = x_conv.type(torch.complex64)
        
        # Concatenate tensors along axis 0
        x_sum = torch.cat((x_level_1, x_conv_complex), dim=1)
        x_sum = self.fc(x_sum)

        # Создаем выходы для каждого класса
        x_opr_swch = self.fc_opr_swch(x_sum)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_sum)
        x_emerg_evnt = self.fc_emerg_evnt(x_sum)

        # Объединяем выходы в один тензор
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x

class CONV_AND_FFT_COMPLEX_v2(nn.Module):
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
        self.harmonic_count = 3

        super(CONV_AND_FFT_COMPLEX_v2, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*1*14, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
            ) for _ in range(self.harmonic_count)  # Для каждой гармоники
        ])
        
        self.conv3 = Conv_3(useComplex=True)
        
        self.fc = nn.Sequential(
            nn.Linear(self.harmonic_count*hidden_size + 32*14, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
    
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        
        currents = [Features.CURRENT["IA"], Features.CURRENT["IB"], Features.CURRENT["IC"], Features.CURRENT["IN"]]
        voltages_bb = [Features.VOLTAGE_PHAZE_BB["UA BB"], Features.VOLTAGE_PHAZE_BB["UB BB"], Features.VOLTAGE_PHAZE_BB["UC BB"], Features.VOLTAGE_PHAZE_BB["UN BB"]]
        voltages_cl = [Features.VOLTAGE_PHAZE_CL["UA CL"], Features.VOLTAGE_PHAZE_CL["UB CL"], Features.VOLTAGE_PHAZE_CL["UC CL"], Features.VOLTAGE_PHAZE_CL["UN CL"]]
        
        x_g1 = create_signal_group(x, currents, voltages_bb, device=self.device)
        x_g2 = create_signal_group(x, currents, voltages_cl, device=self.device)
        
        ic_L = [
            [Features.CURRENT["IA"], Features.CURRENT["IB"]],
            [Features.CURRENT["IB"], Features.CURRENT["IC"]],
            [Features.CURRENT["IC"], Features.CURRENT["IA"]]
        ]
        voltages_line_bb = [Features.VOLTAGE_LINE_BB["UAB BB"], Features.VOLTAGE_LINE_BB["UBC BB"], Features.VOLTAGE_LINE_BB["UCA BB"]]
        voltages_line_cl = [Features.VOLTAGE_LINE_CL["UAB CL"], Features.VOLTAGE_LINE_CL["UBC CL"], Features.VOLTAGE_LINE_CL["UCA CL"]]

        x_g3 = create_line_group(x, ic_L, voltages_line_bb, device=self.device)
        x_g4 = create_line_group(x, ic_L, voltages_line_cl, device=self.device)

        x_new = torch.cat((x_g1, x_g2, x_g3, x_g4), dim=1)

        # Apply conv to each signal
        count_signal_group = 4+4+3+3 # g1+g2+g3+g4
        x_conv = x_conv = torch.stack([self.conv3(x_new[:, i:i+1, 32:]).reshape(x.size(0), -1) for i in range(count_signal_group)], dim=-1)
        x_conv = x_conv.reshape(x_conv.size(0), -1)
        
        
        # Process FFT harmonics
        x_previous, x_current = fft_calc(x, count_harmonic=self.harmonic_count-1)

        x_level_1 = [
            self.fc_layers[i](torch.cat((x_previous[:, :, i], x_current[:, :, i]), dim=-1).reshape(x.size(0), -1))
            for i in range(self.harmonic_count)
        ]

        x_level_1 = torch.stack(x_level_1, dim=1)
        x_level_1 = x_level_1.reshape(x_level_1.size(0), -1)
        
        x_sum = torch.cat((x_level_1, x_conv), dim=1)
        x_sum = self.fc(x_sum)

        # Create outputs for each class
        x_opr_swch = self.fc_opr_swch(x_sum)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_sum)
        x_emerg_evnt = self.fc_emerg_evnt(x_sum)

        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x

class CONV_AND_FFT_COMPLEX_v3(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(4*hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
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
        self.harmonic_count = 16

        super(CONV_AND_FFT_COMPLEX_v3, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*1*14, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat),
                cLeakyReLU(),
            ) for _ in range(self.harmonic_count)  # Для каждой гармоники
        ])
        
        self.conv32 = nn.Sequential(
            nn.Conv1d(1,8, kernel_size=32, stride=16, dtype=torch.cfloat),
            cLeakyReLU()
        )
        self.conv3 = Conv_3(useComplex=True)
        
        self.conv_layer = nn.Sequential(
            nn.Linear((24+32)*14, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(4*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        
        self.fft_layer = nn.Sequential(
            nn.Linear(self.harmonic_count*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(hidden_size, 4*hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(4*hidden_size, hidden_size, dtype=torch.cfloat),
            cLeakyReLU(),
            cDropout1d(0.1),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            cLeakyReLU(),
        )
        
        self.fc_opr_swch = self.Head_fc(hidden_size)
        self.fc_abnorm_evnt = self.Head_fc(hidden_size)
        self.fc_emerg_evnt = self.Head_fc(hidden_size)
    
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        
        currents = [Features.CURRENT["IA"], Features.CURRENT["IB"], Features.CURRENT["IC"], Features.CURRENT["IN"]]
        voltages_bb = [Features.VOLTAGE_PHAZE_BB["UA BB"], Features.VOLTAGE_PHAZE_BB["UB BB"], Features.VOLTAGE_PHAZE_BB["UC BB"], Features.VOLTAGE_PHAZE_BB["UN BB"]]
        voltages_cl = [Features.VOLTAGE_PHAZE_CL["UA CL"], Features.VOLTAGE_PHAZE_CL["UB CL"], Features.VOLTAGE_PHAZE_CL["UC CL"], Features.VOLTAGE_PHAZE_CL["UN CL"]]
        
        x_g1 = create_signal_group(x, currents, voltages_bb, device=self.device)
        x_g2 = create_signal_group(x, currents, voltages_cl, device=self.device)
        
        ic_L = [
            [Features.CURRENT["IA"], Features.CURRENT["IB"]],
            [Features.CURRENT["IB"], Features.CURRENT["IC"]],
            [Features.CURRENT["IC"], Features.CURRENT["IA"]]
        ]
        voltages_line_bb = [Features.VOLTAGE_LINE_BB["UAB BB"], Features.VOLTAGE_LINE_BB["UBC BB"], Features.VOLTAGE_LINE_BB["UCA BB"]]
        voltages_line_cl = [Features.VOLTAGE_LINE_CL["UAB CL"], Features.VOLTAGE_LINE_CL["UBC CL"], Features.VOLTAGE_LINE_CL["UCA CL"]]

        x_g3 = create_line_group(x, ic_L, voltages_line_bb, device=self.device)
        x_g4 = create_line_group(x, ic_L, voltages_line_cl, device=self.device)

        x_new = torch.cat((x_g1, x_g2, x_g3, x_g4), dim=1)

        # Apply conv to each signal
        count_signal_group = 4+4+3+3 # g1+g2+g3+g4
        x_conv_3 = torch.stack([self.conv3(x_new[:, i:i+1, 32:]).reshape(x.size(0), -1) for i in range(count_signal_group)], dim=-1)
        x_conv_3 = x_conv_3.reshape(x_conv_3.size(0), -1)
        x_conv_32 = torch.stack([self.conv32(x_new[:, i:i+1, :]).reshape(x.size(0), -1) for i in range(count_signal_group)], dim=-1)
        x_conv_32 = x_conv_32.reshape(x_conv_32.size(0), -1)
        
        x_conv = torch.cat((x_conv_3, x_conv_32), dim=1)
        x_conv = self.conv_layer(x_conv)

        
        # Process FFT harmonics
        x_previous, x_current = fft_calc(x, count_harmonic=self.harmonic_count-1)
        
        x_fft = torch.stack([self.fc_layers[i](torch.cat((x_previous[:, :, i], x_current[:, :, i]), dim=-1).reshape(x.size(0), -1)) for i in range(self.harmonic_count)], dim=1)
        x_fft = x_fft.reshape(x_fft.size(0), -1)
        
        x_fft = self.fft_layer(x_fft)

        x_sum = torch.cat((x_fft, x_conv), dim=1)

        # Create outputs for each class
        x_opr_swch = self.fc_opr_swch(x_sum)
        x_abnorm_evnt = self.fc_abnorm_evnt(x_sum)
        x_emerg_evnt = self.fc_emerg_evnt(x_sum)

        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)

        return x


class FFT_MLP_KAN_v2(nn.Module):
    class Head_fc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat), # nn.Linear(hidden_size * 7, hidden_size),
            cLeakyReLU(),
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
        super(FFT_MLP_KAN_v2, self).__init__()
        
        # TODO: Расписать # KANLayer
        self.kan1 = KANLinearComplex(in_features=2*9*14,
                                     out_features=2*hidden_size,
                                     grid_size=10,
                                     spline_order=3,
                                     scale_noise=0.01,
                                     scale_base=1,
                                     scale_spline=1,
                                     base_activation=nn.SiLU,
                                     grid_eps=0.02,
                                     grid_range=[0,1])
        self.kan2 = KANLinearComplex(in_features=2*hidden_size,
                                     out_features=4*hidden_size,
                                     grid_size=10,
                                     spline_order=3,
                                     scale_noise=0.01,
                                     scale_base=1,
                                     scale_spline=1,
                                     base_activation=nn.SiLU,
                                     grid_eps=0.02,
                                     grid_range=[0,1])
        self.kan3 = KANLinearComplex(in_features=4*hidden_size,
                                     out_features=2*hidden_size,
                                     grid_size=10,
                                     spline_order=3,
                                     scale_noise=0.01,
                                     scale_base=1,
                                     scale_spline=1,
                                     base_activation=nn.SiLU,
                                     grid_eps=0.02,
                                     grid_range=[0,1])
        self.kan4 = KANLinearComplex(in_features=2*hidden_size,
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
        x = torch.cat(fft_calc(x, count_harmonic = 8), dim=-1) 
        # Concatenate tensors along axis 0
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

if __name__ == "__main__":
    print(CONV_MLP_v2())
