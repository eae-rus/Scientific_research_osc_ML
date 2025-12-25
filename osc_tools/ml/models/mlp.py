import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.blocks import PDRBlock

class PDR_MLP_v1(BaseModel):
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
        super().__init__()
        self.channel_num = channel_num
        self.device = device
        self.fc = self.Head_fc(self.channel_num, hidden_size=20)

    def forward(self, x):
        # x: (batch_size, *, *) — например (1178, 4, 1) или (1178, 1, 4)
        x = x.flatten(start_dim=1)  # → (batch_size, feature_dim), здесь feature_dim=4*1=4
        return self.fc(x)

class PDR_MLP_v2(BaseModel):
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
                print(f"Информация: Пропуск создания блока {i}, т.к. base_neurons={base_neurons} или все коэффициенты <= 0.")
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
                 print(f"Предупреждение: Созданный блок {i} имеет output_size=0. Проверьте коэффициенты.")
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
                             print(f"Предупреждение: Skip target size для блока {i} равен 0. Проекция не создана.")
                             proj = None # или nn.Identity(), если source_size=0
                    elif skip_source_size == 0 and skip_target_size == 0:
                        proj = nn.Identity() # Для случая 0 -> 0
                    elif skip_source_size > 0: # Размеры равны и не нулевые
                        proj = nn.Identity()
                    else: # source=0, target>0 - нельзя сделать Identity
                        print(f"Предупреждение: Skip source size=0, target size>0 для блока {i}. Проекция не создана.")
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
             print("Ошибка: Выходной размер сети после всех блоков равен 0. Проверьте конфигурацию.")
             self.output_layer = None # Или создать фиктивный слой, который не будет использоваться

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.expected_input_features:
             raise ValueError(f"Несоответствие размерности входа! Ожидалось {self.expected_input_features}, получено {x.shape[1]}")
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
            print("Ошибка: Финальный слой не инициализирован.")
            return torch.zeros(x.shape[0], 1, device=x.device)
        
        return output

class PDR_MLP_v3(nn.Module):
    """
    Модель на основе PDR_MLP_v2.
    Но на вход она принимает не только точечные значения тока/напряжения и их же из памяти,
    а весь массив этих точек, например за 2 периода + более редкие за 10 периодов.
    А затем по ним проходится нейроннка на основе свёрточной и выдаёт значения уже PDR_MLP_v2.
    """
    pass
