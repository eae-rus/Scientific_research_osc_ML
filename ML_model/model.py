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

class PDRBlock(nn.Module):
    """
    Универсальный блок PDR с условно активными параллельными ветками:
    обычной, min, сравнения, умножения и деления.
    Активация выбирается параметром. Улучшена защита деления.
    """
    def __init__(self,
                 input_size,
                 base_neurons,
                 coeff_regular=4,
                 coeff_min=2,
                 coeff_compare=2,
                 coeff_mul=1,
                 coeff_div=1,
                 activation_type='leaky_relu', # 'leaky_relu' или 'sigmoid'
                 division_epsilon=1e-8):
        super().__init__()

        # --- Выбор функции активации ---
        if activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Неизвестный тип активации: {activation_type}")

        self.register_buffer('division_epsilon', torch.tensor(division_epsilon))
        current_output_size = 0 # Накапливаем размер активных веток

        # --- Ветка 1: Обычная ---
        self.regular_size = base_neurons * coeff_regular
        if self.regular_size > 0:
            self.lin_regular = nn.Linear(input_size, self.regular_size)
            current_output_size += self.regular_size
        else:
            self.lin_regular = None

        # --- Ветка 2: Min ---
        self.min_size = base_neurons * coeff_min
        if self.min_size > 0:
            self.lin_min1 = nn.Linear(input_size, self.min_size)
            self.lin_min2 = nn.Linear(input_size, self.min_size)
            current_output_size += self.min_size
        else:
            self.lin_min1, self.lin_min2 = None, None

        # --- Ветка 3: Сравнение ---
        self.compare_size = base_neurons * coeff_compare
        if self.compare_size > 0:
            self.lin_compare_signal = nn.Linear(input_size, self.compare_size)
            self.lin_compare_threshold = nn.Linear(input_size, self.compare_size)
            self.comparison_steepness = nn.Parameter(torch.tensor(10.0))
            current_output_size += self.compare_size
        else:
            self.lin_compare_signal, self.lin_compare_threshold = None, None
            self.comparison_steepness = None # Явно указываем отсутствие

        # --- Ветка 4: Умножение ---
        self.mul_size = base_neurons * coeff_mul
        if self.mul_size > 0:
            self.lin_mul1 = nn.Linear(input_size, self.mul_size)
            self.lin_mul2 = nn.Linear(input_size, self.mul_size)
            current_output_size += self.mul_size
        else:
            self.lin_mul1, self.lin_mul2 = None, None

        # --- Ветка 5: Деление ---
        self.div_size = base_neurons * coeff_div
        if self.div_size > 0:
            self.lin_div_numerator = nn.Linear(input_size, self.div_size)
            self.lin_div_denominator = nn.Linear(input_size, self.div_size)
            current_output_size += self.div_size
        else:
            self.lin_div_numerator, self.lin_div_denominator = None, None

        # Общая фактическая выходная размерность блока
        self.output_size = current_output_size

    def forward(self, x):
        outputs_to_cat = []

        # Ветка 1: Обычная
        if self.lin_regular:
            out_regular = self.activation(self.lin_regular(x))
            outputs_to_cat.append(out_regular)

        # Ветка 2: Min
        if self.lin_min1:
            out_min = self.activation(torch.min(self.lin_min1(x), self.lin_min2(x)))
            outputs_to_cat.append(out_min)

        # Ветка 3: Сравнение
        if self.lin_compare_signal:
            signals = self.lin_compare_signal(x)
            thresholds = self.lin_compare_threshold(x)
            # Аппроксимация сравнения (выход всегда в [0, 1] из-за sigmoid)
            out_compare = torch.sigmoid(self.comparison_steepness * (signals - thresholds))
            outputs_to_cat.append(out_compare)

        # Ветка 4: Умножение
        if self.lin_mul1:
            # TODO: разобраться почему из-за неё уходит в "nan" при использовании функций "leaky_relu" с длинными цепочками
            out_mul = torch.sigmoid(self.lin_mul1(x) * self.lin_mul2(x))
            outputs_to_cat.append(out_mul)

        # Ветка 5: Деление
        if self.lin_div_numerator:
            numerator = self.lin_div_numerator(x)
            denominator = self.lin_div_denominator(x)
            # Используем модуль знаменателя для защиты
            # TODO: разобраться почему из-за неё уходит в "nan" при использовании функций "leaky_relu" с длинными цепочками
            out_div = torch.sigmoid(numerator / (torch.abs(denominator) + self.division_epsilon))
            outputs_to_cat.append(out_div)

        # Если вдруг ни одна ветка не активна (хотя это странно)
        if not outputs_to_cat:
             # Можно вернуть тензор нулей нужного размера или бросить ошибку
             # Вернем тензор нулей с размером 0 в последней дименсии, чтобы cat не упал
             # Хотя лучше настроить коэффициенты так, чтобы хоть одна ветка была > 0
             print(f"Warning: PDRBlock с input_size={x.shape[-1]} не имеет активных веток!")
             return torch.empty((*x.shape[:-1], 0), device=x.device, dtype=x.dtype)

        # Конкатенация выходов активных веток
        return torch.cat(outputs_to_cat, dim=-1)

class PDR_MLP_v2(nn.Module):
    """
    Модульная MLP на основе PDRBlock с опциональными skip-соединениями.
    Конфигурация слоев задается списком базовых нейронов.
    Функция активации и активные ветки в блоках настраиваются.
    """
    def __init__(self,
                 input_features,
                 block_neuron_config=[3, 2, 1], # Список базовых нейронов для каждого блока
                 activation_type='leaky_relu',   # Тип активации для всех блоков
                 use_skip_connection=True,
                 # Коэффициенты передаются в каждый блок, управление активностью веток через них
                 coeff_regular=4, coeff_min=2, coeff_compare=2, coeff_mul=1, coeff_div=1,
                 device=None):
        super().__init__()
        self.use_skip_connection = use_skip_connection
        self.device = device
        self.expected_input_features = input_features
        self.num_blocks = len(block_neuron_config) # TODO: Решить как обходить проблему разного количества слоёв для скипа.

        self.blocks = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        current_size = input_features
        block_output_sizes = []

        # Создаем блоки на основе конфигурации
        for i, base_neurons in enumerate(block_neuron_config):
            # Проверяем, есть ли смысл создавать блок
            # (сумма размеров всех потенциальных веток > 0)
            potential_size = base_neurons * (coeff_regular + coeff_min + coeff_compare + coeff_mul + coeff_div)
            if potential_size <= 0 and base_neurons <=0: # Добавил проверку base_neurons на всякий случай
                print(f"Info: Пропуск создания блока {i}, т.к. base_neurons={base_neurons} или все коэффициенты <= 0.")
                self.num_blocks -= 1 # Уменьшаем фактическое число блоков
                continue # Не создаем этот блок

            block = PDRBlock(input_size=current_size,
                             base_neurons=base_neurons,
                             coeff_regular=coeff_regular,
                             coeff_min=coeff_min,
                             coeff_compare=coeff_compare,
                             coeff_mul=coeff_mul,
                             coeff_div=coeff_div,
                             activation_type=activation_type) # Передаем тип активации

            # Проверка, что блок реально что-то выводит
            if block.output_size <= 0:
                 print(f"Warning: Созданный блок {i} имеет output_size=0. Проверьте коэффициенты.")
                 # Можно его не добавлять, но это может сломать skip connections
                 # Пока оставляем, но надо следить за конфигурацией
            
            self.blocks.append(block)
            block_output_sizes.append(block.output_size)
            current_size = block.output_size

        # Настраиваем skip-соединения на основе ФАКТИЧЕСКИ созданных блоков
        if self.use_skip_connection:
            # Важно: block_output_sizes теперь может быть короче, чем len(block_neuron_config)
            actual_num_blocks = len(self.blocks)
            for i in range(actual_num_blocks):
                if i >= 2: # Skip от i-2 к входу i (т.е. после выхода i-1)
                    skip_source_size = block_output_sizes[i-2]
                    skip_target_size = self.blocks[i-1].output_size # == block_output_sizes[i-1]
                    if skip_source_size != skip_target_size:
                        # Добавляем проверку, что skip_target_size > 0, иначе Linear не создать
                        if skip_target_size > 0:
                            proj = nn.Linear(skip_source_size, skip_target_size)
                        else:
                             # Это странная ситуация, пропускаем проекцию
                             print(f"Warning: Skip target size для блока {i} равен 0. Проекция не создана.")
                             proj = None # или nn.Identity(), если source_size=0
                    elif skip_source_size == 0 and skip_target_size == 0:
                        proj = nn.Identity() # Для случая 0 -> 0
                    elif skip_source_size > 0: # Размеры равны и не нулевые
                        proj = nn.Identity()
                    else: # source=0, target>0 - нельзя сделать Identity
                        print(f"Warning: Skip source size=0, target size>0 для блока {i}. Проекция не создана.")
                        proj = None

                    self.skip_projs.append(proj)
                else:
                     self.skip_projs.append(None) # Для первых двух блоков проекции нет

        # Финальный слой
        # current_size теперь содержит размер выхода последнего *фактически созданного* блока
        if current_size > 0:
             self.output_layer = nn.Linear(current_size, 1)
        else:
             # Если после всех блоков размер 0, сеть не может работать
             print("Error: Выходной размер сети после всех блоков равен 0. Проверьте конфигурацию.")
             self.output_layer = None # Или создать фиктивный слой, который не будет использоваться

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.expected_input_features:
             raise ValueError(f"Input dimension mismatch! Expected {self.expected_input_features}, got {x.shape[1]}")
        if self.device:
             x = x.to(self.device)

        block_outputs = []
        current_input = x
        actual_block_index = 0 # Индекс для skip_projs

        for i, block in enumerate(self.blocks): # Итерируемся по фактически созданным блокам
            block_input = current_input
            # Применяем skip-соединение (если активно и возможно)
            if self.use_skip_connection and i >= 2:
                 # Используем actual_block_index для доступа к skip_projs
                skip_proj = self.skip_projs[actual_block_index]
                if skip_proj is not None: # Проверяем, что проекция была создана
                    skip_source_output = block_outputs[i-2] # Доступ к выходам по индексу i
                    skip_projected = skip_proj(skip_source_output)
                    block_input = block_input + skip_projected
                actual_block_index += 1 # Увеличиваем индекс только если i >= 2

            block_output = block(block_input)
            block_outputs.append(block_output)
            current_input = block_output

        # Финальный слой
        final_block_output = current_input
        if self.output_layer is not None: # Проверка, что слой был создан
            output = self.output_layer(final_block_output)
            output = self.output_activation(output)
        else:
            # Если финальный слой не создан, вернуть что-то осмысленное или ошибку
            print("Error: Финальный слой не инициализирован.")
            # Можно вернуть тензор нулей или NaN, чтобы показать проблему
            output = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype) * torch.nan

        return output

class PDR_MLP_v3(nn.Module):
    """
    Модель на основе PDR_MLP_v2.
    Но на вход она принимает не только точечные значения тока/напряжения и их же из памяти,
    а весь массив этих точек, например за 2 периода + более редкие за 10 периодов.
    А затем по ним проходится нейроннка на основе свёрточной и выдаёт значения уже PDR_MLP_v2.
    """
    pass







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
