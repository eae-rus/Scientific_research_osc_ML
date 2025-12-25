import torch
import torch.nn as nn

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
