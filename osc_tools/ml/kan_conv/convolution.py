#Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: Union[List[List[float]], np.ndarray], #но как тензоры torch. Kernel side предполагает, что ядро квадратное
             kernel, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Выполняет двумерную свертку с ядром по матрице, используя заданные шаг, расширение и заполнение по осям.

    Аргументы:
        matrix (batch_size, colors, n, m]): двумерная матрица для свертки.
        kernel (function]): двумерная матрица нечетной формы (например, 3x3, 5x5, 13x9 и т. д.).
        stride (Tuple[int, int], optional): кортеж шага по осям. С шагом `(r, c)` мы перемещаемся на `r` пикселей по строкам и на `c` пикселей по столбцам на каждой итерации. По умолчанию (1, 1).
        dilation (Tuple[int, int], optional): кортеж расширения по осям. С расширением `(r, c)` мы удаляем соседние пиксели в ядре на `r` по строкам и `c` по столбцам. По умолчанию (1, 1).
        padding (Tuple[int, int], optional): кортеж с количеством строк и столбцов для заполнения. По умолчанию (0, 0).

    Возвращает:
        np.ndarray: двумерная карта признаков, т. е. матрица после свертки.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = torch.zeros((batch_size,n_channels,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)


    for channel in range(n_channels):
        #print(matrix[:,channel,:,:].unsqueeze(1).shape)
        conv_groups = unfold(matrix[:,channel,:,:].unsqueeze(1)).transpose(1, 2)
        #print("conv",conv_groups.shape)
        for k in range(batch_size):
            matrix_out[k,channel,:,:] = kernel.forward(conv_groups[k,:,:]).reshape((h_out,w_out))
    return matrix_out

def multiple_convs_kan_conv2d(matrix, #но как тензоры torch. Kernel side предполагает, что ядро квадратное
             kernels, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Выполняет двумерную свертку с ядром по матрице, используя заданные шаг, расширение и заполнение по осям.

    Аргументы:
        matrix (batch_size, colors, n, m]): двумерная матрица для свертки.
        kernel (function]): двумерная матрица нечетной формы (например, 3x3, 5x5, 13x9 и т. д.).
        stride (Tuple[int, int], optional): кортеж шага по осям. С шагом `(r, c)` мы перемещаемся на `r` пикселей по строкам и на `c` пикселей по столбцам на каждой итерации. По умолчанию (1, 1).
        dilation (Tuple[int, int], optional): кортеж расширения по осям. С расширением `(r, c)` мы удаляем соседние пиксели в ядре на `r` по строкам и `c` по столбцам. По умолчанию (1, 1).
        padding (Tuple[int, int], optional): кортеж с количеством строк и столбцов для заполнения. По умолчанию (0, 0).

    Возвращает:
        np.ndarray: двумерная карта признаков, т. е. матрица после свертки.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,n_channels*n_convs,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:,kern  + channel*n_convs,:,:] = kernels[kern].conv.forward(conv_groups[:,channel,:,:].flatten(0,1)).reshape((batch_size,h_out,w_out))
    return matrix_out


def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Добавляет заполнение в матрицу.

    Аргументы:
        matrix (np.ndarray): матрица, которую необходимо дополнить. Тип — List[List[float]], приведенный к np.ndarray.
        padding (Tuple[int, int]): кортеж с количеством строк и столбцов для заполнения. С заполнением `(r, c)` мы добавляем `r` строк сверху и снизу и `c` столбцов слева и справа от матрицы.

    Возвращает:
        np.ndarray: дополненная матрица с формой `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix
