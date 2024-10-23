import sys
import os
import torch as torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)
from kan_convolutional.KANLinear import KANLinear, KANLinearComplex


# Требуется разобраться с базовой моделью KANLinear, у меня не работала норально.
# У них можно график строить заполениня и другие фичи
# from kan import *
# KAN

    
class CONV_MLP_v2(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(CONV_MLP_v2, self).__init__()
        self.conv32 = nn.Sequential(
            nn.Conv1d(
                1,
                8,
                kernel_size=32,
                stride=16,
            ),
            nn.LeakyReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                1,
                2*8,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2), # 32*16 -> 16*16
            nn.Conv1d(
                16,
                4*8,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16*32 -> 8*32
            nn.Conv1d(
                4*8,
                4*8,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=8), # 8*32 -> 1*32
        )

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc_opr_swch = nn.Sequential(
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
        self.fc_abnorm_evnt = nn.Sequential(
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
        self.fc_emerg_evnt = nn.Sequential(
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


class FFT_MLP(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
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
        self.fc_opr_swch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        self.fc_abnorm_evnt = nn.Sequential(

            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        self.fc_emerg_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )

    def fft_calc(self, input, count_harmonic=1):
        # input - тензор формы (batch_size, channel_num, frame_size)
        batch_size, channel_num, frame_size = input.size()
        
        # Вычисление FFT для каждого канала
        fft_result_previous = torch.fft.rfft(input[:,:,:32])  # Одностороннее комплексное преобразование Фурье (работает с реальными числами)
        fft_result_current = torch.fft.rfft(input[:,:,32:]) 

        # Ограничиваем количество гармоник до count_harmonic + 1
        fft_result_previous = fft_result_previous[:, :, :count_harmonic+1]
        fft_result_current = fft_result_current[:, :, :count_harmonic+1]

        # Извлечение амплитуды и фазы только для выбранных гармоник
        fft_amplitude_previous = torch.abs(fft_result_previous)  # Амплитуда
        fft_amplitude_current = torch.abs(fft_result_current)
        fft_phase_previous = torch.angle(fft_result_current)  # Фаза
        fft_phase_current = torch.angle(fft_result_current)

        # Объединяем амплитуду и фазу в один тензор с последующим разворачиванием
        # Амплитуда и фаза будут идти последовательно по последней размерности
        # Например, (batch_size, channel_num, count_harmonic+1, 2) -> (batch_size, channel_num, (count_harmonic+1)*2)
        fft_combined = torch.cat((fft_amplitude_previous, fft_phase_previous, fft_amplitude_current, fft_phase_current), dim=-1)  # Склеиваем по последней размерности

        # Изменение формы результирующего тензора, чтобы убрать лишние измерения
        # (batch_size, channel_num, count_harmonic+1, 2) -> (batch_size, channel_num, (count_harmonic+1)*2)
        # fft_combined = fft_combined.view(batch_size, channel_num, -1)
        
        return fft_combined
        
    def forward(self, x):
        ## ТРЕБУЕТСЯ сделать независимые выходы для КАЖДОГО класса 
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.fft_calc(x, count_harmonic = 8)
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
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
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
        
        self.fc_opr_swch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # nn.Linear(hidden_size * 7, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        self.fc_abnorm_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # nn.Linear(hidden_size * 7, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )
        self.fc_emerg_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # nn.Linear(hidden_size * 7, hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(),
        )

    def fft_calc(self, input, count_harmonic=1, device = None):
        # input - тензор формы (batch_size, channel_num, frame_size)
        batch_size, channel_num, frame_size = input.size()
        
        # Вычисление FFT для каждого канала
        fft_result_previous = torch.fft.rfft(input[:,:,:32])  # Одностороннее комплексное преобразование Фурье (работает с реальными числами)
        fft_result_current = torch.fft.rfft(input[:,:,32:]) 

        # Ограничиваем количество гармоник до count_harmonic + 1
        fft_result_previous = fft_result_previous[:, :, :count_harmonic+1]
        fft_result_current = fft_result_current[:, :, :count_harmonic+1]

        # Извлечение амплитуды и фазы только для выбранных гармоник
        fft_amplitude_previous = torch.abs(fft_result_previous)  # Амплитуда
        fft_amplitude_current = torch.abs(fft_result_current)
        fft_phase_previous = torch.angle(fft_result_current)  # Фаза
        fft_phase_current = torch.angle(fft_result_current)

        # Объединяем амплитуду и фазу в один тензор с последующим разворачиванием
        # Амплитуда и фаза будут идти последовательно по последней размерности
        # Например, (batch_size, channel_num, count_harmonic+1, 2) -> (batch_size, channel_num, (count_harmonic+1)*2)
        fft_combined = torch.cat((fft_amplitude_previous, fft_phase_previous, fft_amplitude_current, fft_phase_current), dim=-1)  # Склеиваем по последней размерности

        # Изменение формы результирующего тензора, чтобы убрать лишние измерения
        # (batch_size, channel_num, count_harmonic+1, 2) -> (batch_size, channel_num, (count_harmonic+1)*2)
        # fft_combined = fft_combined.view(batch_size, channel_num, -1)
        
        return fft_combined
        
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.fft_calc(x, count_harmonic = 8)
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

class ComplexLeakyReLU(nn.Module):
    def forward(self, input, negative_slope = 0.05):
        amp = torch.abs(input) # TODO: наверное можно как-то сразу, надо изучить, "что есть" в этих числах
        # можно будет сделать порог сдвигаемым
        amp_LeakyReLU = torch._C._nn.leaky_relu_(amp, negative_slope)
        # позитив и негатив - для создания полуплоскости (можно будет сделать их адаптивными)
        angle = torch.angle(input)
        angle_LeakyReLU = torch._C._nn.leaky_relu_(angle, negative_slope)
        
        real = amp_LeakyReLU * torch.cos(angle_LeakyReLU)
        imag = amp_LeakyReLU * torch.sin(angle_LeakyReLU)
        return torch.complex(real, imag)

class ComplexSigmoid(nn.Module):
    # TODO: На будущее стоит сделать новый орган, который будет адаптивно учитывать границы угла, а не "зону"
    # Актуально для выходной переменной
    
    def forward(self, input):
        return torch.sigmoid(input.real)

class FFT_MLP_COMPLEX_v1(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.device = device
        super(FFT_MLP_COMPLEX_v1, self).__init__()

        # TODO: исправить расчётывание выходного размера после свёрток
        # пока что задаю принудительно 128*16
        self.fc = nn.Sequential(
            nn.Linear(2*9*14, 4*hidden_size, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(4*hidden_size, 4*hidden_size, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(4*hidden_size, 2*hidden_size, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(2*hidden_size, hidden_size, dtype=torch.cfloat),
            ComplexLeakyReLU(),
        )
        self.fc_opr_swch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )
        self.fc_abnorm_evnt = nn.Sequential(

            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )
        self.fc_emerg_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )

    def fft_calc(self, input, count_harmonic=1):
        # input - тензор формы (batch_size, channel_num, frame_size)
        batch_size, channel_num, frame_size = input.size()
        
        # Вычисление FFT для каждого канала
        fft_result_previous = torch.fft.rfft(input[:,:,:32])  # Одностороннее комплексное преобразование Фурье (работает с реальными числами)
        fft_result_current = torch.fft.rfft(input[:,:,32:]) 

        # Ограничиваем количество гармоник до count_harmonic + 1
        fft_result_previous = fft_result_previous[:, :, :count_harmonic+1]
        fft_result_current = fft_result_current[:, :, :count_harmonic+1]

        # Объединяем в один тензор с последующим разворачиванием
        fft_combined = torch.cat((fft_result_previous, fft_result_current), dim=-1)  # Склеиваем по последней размерности
        
        return fft_combined
        
    def forward(self, x):
        ## ТРЕБУЕТСЯ сделать независимые выходы для КАЖДОГО класса 
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.fft_calc(x, count_harmonic = 8)
        # Concatenate tensors along axis 0
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        # FEATURES_TARGET = ["opr_swch", "abnorm_evnt", "emerg_evnt"]
        x_opr_swch = self.fc_opr_swch(x)
        x_abnorm_evnt = self.fc_abnorm_evnt(x)
        x_emerg_evnt = self.fc_emerg_evnt(x)
        x = torch.cat((x_opr_swch, x_abnorm_evnt, x_emerg_evnt), dim=1)
        return x

class FFT_MLP_KAN_v2(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=40, output_size=4, device = None,
    ):
        # TODO: разобраться в схеме обработки сигналов
        # TODO: Подумать о создании более сложной схемы (2D свёртки, чтобы одновременно и I и U)
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
        
        self.fc_opr_swch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat), # nn.Linear(hidden_size * 7, hidden_size),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )
        self.fc_abnorm_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat), # nn.Linear(hidden_size * 7, hidden_size),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )
        self.fc_emerg_evnt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.cfloat), # nn.Linear(hidden_size * 7, hidden_size),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2, dtype=torch.cfloat),
            ComplexLeakyReLU(),
            nn.Linear(hidden_size//2, 1, dtype=torch.cfloat),
            ComplexSigmoid(),
        )

    def fft_calc(self, input, count_harmonic=1):
        # input - тензор формы (batch_size, channel_num, frame_size)
        batch_size, channel_num, frame_size = input.size()
        
        # Вычисление FFT для каждого канала
        fft_result_previous = torch.fft.rfft(input[:,:,:32])  # Одностороннее комплексное преобразование Фурье (работает с реальными числами)
        fft_result_current = torch.fft.rfft(input[:,:,32:]) 

        # Ограничиваем количество гармоник до count_harmonic + 1
        fft_result_previous = fft_result_previous[:, :, :count_harmonic+1]
        fft_result_current = fft_result_current[:, :, :count_harmonic+1]

        # Объединяем в один тензор с последующим разворачиванием
        fft_combined = torch.cat((fft_result_previous, fft_result_current), dim=-1)  # Склеиваем по последней размерности
        
        return fft_combined
        
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.fft_calc(x, count_harmonic = 8)
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
