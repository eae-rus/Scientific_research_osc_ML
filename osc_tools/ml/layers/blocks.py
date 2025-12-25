import torch
import torch.nn as nn

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
             print(f"Предупреждение: PDRBlock с input_size={x.shape[-1]} не имеет активных веток!")
             return torch.empty((*x.shape[:-1], 0), device=x.device, dtype=x.dtype)

        # Конкатенация выходов активных веток
        return torch.cat(outputs_to_cat, dim=-1)
