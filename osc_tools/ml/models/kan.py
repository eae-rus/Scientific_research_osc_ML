import torch
import torch.nn as nn
import torch.nn.functional as F
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.kan_layers import KANLinear, KANConv1d
from osc_tools.ml.kan_conv.arithmetic import MultiplicationLayer, DivisionLayer
from osc_tools.ml.kan_conv.modern_wrappers import build_kan_linear


class ComplexPairDropout(nn.Module):
    """Dropout для представления [амплитуда, фаза], применяемый согласованно к обеим компонентам."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or not self.training:
            return amp, phase

        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(amp) < keep_prob).to(amp.dtype) / keep_prob
        return amp * mask, phase * mask


class ComplexMultiplicationLayer(nn.Module):
    """Умножение комплексных величин в полярной форме по паре [A, φ]."""

    def __init__(self, phase_bias_b: float = 0.0):
        super().__init__()
        self.phase_bias_b = float(phase_bias_b)

    @staticmethod
    def _split_amp_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0::2, ...], x[:, 1::2, ...]

    @staticmethod
    def _stack_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        out_shape = list(amp.shape)
        out_shape[1] = amp.shape[1] * 2
        out = torch.empty(out_shape, device=amp.device, dtype=amp.dtype)
        out[:, 0::2, ...] = amp
        out[:, 1::2, ...] = phase
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c % 4 != 0:
            raise ValueError(f"ComplexMultiplicationLayer ожидает число каналов кратное 4, получено {c}")

        amp, phase = self._split_amp_phase(x)
        n_complex = amp.shape[1]
        half = n_complex // 2

        amp_i, amp_u = amp[:, :half, ...], amp[:, half:, ...]
        phase_i, phase_u = phase[:, :half, ...], phase[:, half:, ...]

        amp_out = amp_i * amp_u
        phase_out = phase_i + phase_u + self.phase_bias_b
        return self._stack_amp_phase(amp_out, phase_out)


class ComplexDivisionLayer(nn.Module):
    """Деление комплексных величин в полярной форме по паре [A, φ]."""

    def __init__(self, epsilon: float = 1e-6, phase_bias_b: float = 0.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.phase_bias_b = float(phase_bias_b)

    @staticmethod
    def _split_amp_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0::2, ...], x[:, 1::2, ...]

    @staticmethod
    def _stack_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        out_shape = list(amp.shape)
        out_shape[1] = amp.shape[1] * 2
        out = torch.empty(out_shape, device=amp.device, dtype=amp.dtype)
        out[:, 0::2, ...] = amp
        out[:, 1::2, ...] = phase
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c % 4 != 0:
            raise ValueError(f"ComplexDivisionLayer ожидает число каналов кратное 4, получено {c}")

        amp, phase = self._split_amp_phase(x)
        n_complex = amp.shape[1]
        half = n_complex // 2

        amp_i, amp_u = amp[:, :half, ...], amp[:, half:, ...]
        phase_i, phase_u = phase[:, :half, ...], phase[:, half:, ...]

        amp_out = amp_i / amp_u.clamp_min(self.epsilon)
        phase_out = phase_i - phase_u + self.phase_bias_b
        return self._stack_amp_phase(amp_out, phase_out)

class SimpleKAN(BaseModel):
    """
    Простая полносвязная сеть на основе KAN (Kolmogorov-Arnold Network).
    Аналог SimpleMLP, но с использованием KANLinear слоев.
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, grid_size=5, spline_order=3, dropout=0.0, base_activation=torch.nn.SiLU):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(
                KANLinear(
                    in_features=prev_size, 
                    out_features=size,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
            
        # Выходной слой
        layers.append(
            KANLinear(
                in_features=prev_size, 
                out_features=output_size,
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation
            )
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)

class SafeMaxPool1d(nn.Module):
    """Pooling layer that handles small input sizes gracefully."""
    def __init__(self, kernel_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        if x.shape[-1] < self.pool.kernel_size:
            return x
        return self.pool(x)


class RelayKANBranch(nn.Module):
    """KAN-ветка для раздельной обработки амплитуды, фазы или relay-маски."""

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        allow_pooling: bool = True,
    ):
        super().__init__()

        if not channels:
            raise ValueError("RelayKANBranch требует непустой список channels")

        layers = []
        curr_channels = in_channels

        for i, out_channels in enumerate(channels):
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            curr_stride = stride if i == 0 else 1

            layers.append(
                KANConv1d(
                    curr_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=curr_stride,
                    padding=kernel_size // 2,
                    grid_size=curr_grid,
                    spline_order=spline_order,
                    base_activation=base_activation,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))

            if allow_pooling and pool_every > 0 and (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            curr_channels = out_channels

        self.net = nn.Sequential(*layers)
        self.out_channels = curr_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ConvKAN(BaseModel):
    """
    Сверточная сеть на основе KAN (Convolutional KAN) с гибкой архитектурой.
    Поддерживает произвольное количество слоев через список channels.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [8, 16, 32], 
                 kernel_size: int = 3, stride: int = 1, grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.2, pool_every: int = 1, base_activation=torch.nn.SiLU):
        super().__init__()
        
        layers = []
        curr_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            # KAN Convolutional block
            # grid_size может быть списком или числом. Если список - берем по индексу.
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            
            # Применяем stride только к первому слою
            s = stride if i == 0 else 1
            
            layers.append(
                KANConv1d(
                    curr_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=s,
                    padding=kernel_size//2, 
                    grid_size=curr_grid, 
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )
            layers.append(nn.BatchNorm1d(out_channels)) # Нормализация для стабильности
            
            # Pooling
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            curr_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            KANLinear(curr_channels, curr_channels // 2, grid_size=grid_size[0] if isinstance(grid_size, list) else grid_size, base_activation=base_activation),
            KANLinear(curr_channels // 2, num_classes, grid_size=grid_size[0] if isinstance(grid_size, list) else grid_size, base_activation=base_activation)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class PhysicsKAN(BaseModel):
    """
    KAN модель с физически интерпретируемыми слоями (умножение/деление).
    Принимает на вход [Currents, Voltages].
    Вычисляет Power (I*U) и Admittance (I/U), объединяет с исходными сигналами
    и подает в ConvKAN.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [8, 16, 32], 
                 kernel_size: int = 3, stride: int = 1, grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.2, pool_every: int = 1, base_activation=torch.nn.SiLU,
                 use_mlp: bool = False, input_size: int = 64): # use_mlp для snapshot
        super().__init__()
        
        if in_channels % 2 != 0:
            raise ValueError(f"PhysicsKAN requires even number of input channels (I, U pairs), got {in_channels}")
            
        self.mult = MultiplicationLayer()
        self.div = DivisionLayer()
        
        # Нормализация для физических слоев
        half_channels = in_channels // 2
        self.bn_mult = nn.BatchNorm1d(half_channels)
        self.bn_div = nn.BatchNorm1d(half_channels)

        self.use_mlp = use_mlp
        
        # Вход для ConvKAN: Original (C) + Mult (C/2) + Div (C/2) = 2 * C
        conv_in_channels = in_channels + (in_channels // 2) * 2
        
        if self.use_mlp:
             # Для MLP режима (snapshot c малым кол-вом точек или без временной структуры)
             # Вход: conv_in_channels * (input_size/in_channels)? Нет, вход уже будет развернут?
             # input_size здесь - это кол-во временных точек * каналов.
             # Но мы делаем feature engineering ДО flatten.
             # Поэтому нам надо знать длину временного ряда (pts).
             
             # Если мы получаем (B, C, T), то после mult/div будет (B, 2C, T).
             # Потом flatten -> (B, 2C*T).
             # И подаем в SimpleKAN.
             
            # Определяем размер входа для MLP на основе доступных признаков
            pts = input_size // in_channels
            mlp_input_size = conv_in_channels * pts
             
            # Масштабируем размеры скрытых слоев из конфигурации каналов для сохранения относительной сложности
            hidden_sizes = [h * 4 for h in channels]
             
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation
             )
        else:
            self.processing_net = ConvKAN(
                in_channels=conv_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation
            )

    def forward(self, x):
        # x: [Batch, Channels, Length]
        # Предполагается, что каналы упорядочены так, что первая половина - это I, вторая - U (или наоборот).
        # MultiplicationLayer делает x[:half] * x[half:]
        
        s = self.mult(x) # Power-like features
        s = self.bn_mult(s)

        z = self.div(x)  # Impedance-like features
        z = self.bn_div(z)
        
        # Concatenate along channel dimension
        x_combined = torch.cat([x, s, z], dim=1)
        
        return self.processing_net(x_combined)


class PhysicsKANConditional(BaseModel):
    """
    PhysicsKAN с последовательными головами:
    - Голова 1: Target_Normal (0/1)
    - Голова 2: Target_ML_1 (0/1), получает доп. вход от головы 1
    - Голова 3: Target_ML_3 (0/1), получает доп. вход от головы 1
    - Голова 4: Target_ML_2 (0/1), получает доп. вход от головы 1 и головы 3
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 4,
        channels: list = [16, 32, 64],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        kan_backend: str = 'baseline'
    ):
        super().__init__()

        if num_classes != 4:
            raise ValueError(f"PhysicsKANConditional требует num_classes=4, получено {num_classes}")

        if in_channels % 2 != 0:
            raise ValueError(f"PhysicsKANConditional требует чётное число каналов (I, U пары), получено {in_channels}")

        if use_mlp:
            raise ValueError("PhysicsKANConditional пока не поддерживает use_mlp=True (snapshot режим)")

        self.kan_backend = kan_backend

        self.mult = MultiplicationLayer()
        self.div = DivisionLayer()

        half_channels = in_channels // 2
        self.bn_mult = nn.BatchNorm1d(half_channels)
        self.bn_div = nn.BatchNorm1d(half_channels)

        # Вход для ConvKAN: Original (C) + Mult (C/2) + Div (C/2) = 2 * C
        conv_in_channels = in_channels + (in_channels // 2) * 2

        # Feature extractor (аналог ConvKAN, но без финального классификатора)
        layers = []
        curr_channels = conv_in_channels
        for i, out_channels in enumerate(channels):
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            s = stride if i == 0 else 1
            layers.append(
                KANConv1d(
                    curr_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=s,
                    padding=kernel_size // 2,
                    grid_size=curr_grid,
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Размер скрытого пространства
        feat_dim = curr_channels
        head_hidden = max(4, feat_dim // 2)
        grid_head = grid_size[0] if isinstance(grid_size, list) else grid_size

        # Голова 1: Target_Normal
        self.head_normal = nn.Sequential(
            build_kan_linear(
                backend=self.kan_backend,
                in_features=feat_dim,
                out_features=head_hidden,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            ),
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_hidden,
                out_features=1,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            )
        )

        # Головы 2 и 3: получают +1 признак от головы 1
        head_in_dim = feat_dim + 1
        self.head_ml1 = nn.Sequential(
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_in_dim,
                out_features=head_hidden,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            ),
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_hidden,
                out_features=1,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            )
        )
        self.head_ml3 = nn.Sequential(
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_in_dim,
                out_features=head_hidden,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            ),
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_hidden,
                out_features=1,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            )
        )

        # Голова 4: получает +2 признака (Normal + ML_3)
        head_in_dim_ml2 = feat_dim + 2
        self.head_ml2 = nn.Sequential(
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_in_dim_ml2,
                out_features=head_hidden,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            ),
            build_kan_linear(
                backend=self.kan_backend,
                in_features=head_hidden,
                out_features=1,
                grid_size=grid_head,
                spline_order=spline_order,
                base_activation=base_activation,
            )
        )

    def forward(self, x):
        # Физические преобразования
        s = self.mult(x)
        s = self.bn_mult(s)

        z = self.div(x)
        z = self.bn_div(z)

        x_combined = torch.cat([x, s, z], dim=1)

        feats = self.features(x_combined)
        feats = self.pool(feats)
        feats = feats.flatten(1)

        normal_logit = self.head_normal(feats).squeeze(1)
        normal_prob = torch.sigmoid(normal_logit).unsqueeze(1)

        head_input = torch.cat([feats, normal_prob], dim=1)
        ml1_logit = self.head_ml1(head_input).squeeze(1)
        ml3_logit = self.head_ml3(head_input).squeeze(1)

        ml3_prob = torch.sigmoid(ml3_logit).unsqueeze(1)
        head_input_ml2 = torch.cat([feats, normal_prob, ml3_prob], dim=1)
        ml2_logit = self.head_ml2(head_input_ml2).squeeze(1)

        # Возвращаем 4 выхода: normal, ml1, ml2, ml3
        return torch.stack([normal_logit, ml1_logit, ml2_logit, ml3_logit], dim=1)


class cPhysicsKAN(BaseModel):
    """
    Комплексная PhysicsKAN в полярной форме.

    Ожидает вход с чётным числом каналов, где:
    - чётные индексы (0, 2, 4, ...) — амплитуды;
    - нечётные индексы (1, 3, 5, ...) — фазы.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        phase_bias_b: float = 0.0,
        epsilon: float = 1e-6,
        kan_backend: str = 'baseline',
    ):
        super().__init__()

        if in_channels % 2 != 0:
            raise ValueError(
                f"cPhysicsKAN требует чётное число входных каналов (амплитуда/фаза), получено {in_channels}"
            )

        if in_channels % 4 != 0:
            raise ValueError(
                f"cPhysicsKAN требует число каналов кратное 4: [амплитуда/фаза] и пары I/U, получено {in_channels}"
            )

        self.epsilon = float(epsilon)
        self.phase_bias_b = float(phase_bias_b)
        self.kan_backend = kan_backend
        self.use_mlp = use_mlp
        self.dropout = ComplexPairDropout(dropout)

        self.mult = ComplexMultiplicationLayer(phase_bias_b=phase_bias_b)
        self.div = ComplexDivisionLayer(epsilon=epsilon, phase_bias_b=phase_bias_b)

        # s и z имеют C/2 каналов, из них амплитудных C/4
        self.bn_mult_amp = nn.BatchNorm1d(in_channels // 4)
        self.bn_div_amp = nn.BatchNorm1d(in_channels // 4)

        # Вход для ConvKAN/SimpleKAN: Original (C) + Mult (C/2) + Div (C/2) = 2 * C
        proc_in_channels = in_channels + (in_channels // 2) * 2

        if self.use_mlp:
            pts = input_size // in_channels
            mlp_input_size = proc_in_channels * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation
            )
        else:
            self.processing_net = ConvKAN(
                in_channels=proc_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation
            )

    @staticmethod
    def _split_amp_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = x[:, 0::2, :]
        phase = x[:, 1::2, :]
        return amp, phase

    @staticmethod
    def _stack_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        batch, channels, length = amp.shape
        out = torch.empty(batch, channels * 2, length, device=amp.device, dtype=amp.dtype)
        out[:, 0::2, :] = amp
        out[:, 1::2, :] = phase
        return out

    def _amp_norm_only(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        amp, phase = self._split_amp_phase(x)
        amp = bn(amp)
        amp, phase = self.dropout(amp, phase)
        return self._stack_amp_phase(amp, phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"cPhysicsKAN ожидает вход размерности [B, C, T], получено {tuple(x.shape)}")

        if x.shape[1] % 4 != 0:
            raise ValueError(f"cPhysicsKAN ожидает число каналов кратное 4, получено {x.shape[1]}")

        s = self.mult(x)
        s = self._amp_norm_only(s, self.bn_mult_amp)

        z = self.div(x)
        z = self._amp_norm_only(z, self.bn_div_amp)

        x_combined = torch.cat([x, s, z], dim=1)
        return self.processing_net(x_combined)


class rPhysicsKAN(BaseModel):
    """
    Релейная версия PhysicsKAN в полярной форме.

    Пайплайн:
    1. Создаёт дополнительные комплексные признаки через умножение и деление.
    2. Разделяет амплитуды и фазы на отдельные KAN-ветки.
    3. Формирует дополнительную relay-ветку из фаз, которая маскирует амплитуды значением в диапазоне [0, 1].
    4. Объединяет gated-амплитуды и обработанные фазы перед классификацией.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        phase_bias_b: float = 0.0,
        epsilon: float = 1e-6,
        kan_backend: str = 'baseline',
    ):
        super().__init__()

        if in_channels % 2 != 0:
            raise ValueError(
                f"rPhysicsKAN требует чётное число входных каналов (амплитуда/фаза), получено {in_channels}"
            )

        if in_channels % 4 != 0:
            raise ValueError(
                f"rPhysicsKAN требует число каналов кратное 4: [амплитуда/фаза] и пары I/U, получено {in_channels}"
            )

        self.phase_bias_b = float(phase_bias_b)
        self.epsilon = float(epsilon)
        self.kan_backend = kan_backend
        self.use_mlp = use_mlp
        self.dropout = ComplexPairDropout(dropout)

        self.mult = ComplexMultiplicationLayer(phase_bias_b=phase_bias_b)
        self.div = ComplexDivisionLayer(epsilon=epsilon, phase_bias_b=phase_bias_b)

        proc_in_channels = in_channels + (in_channels // 2) * 2
        relay_in_channels = proc_in_channels // 2
        allow_pooling = not use_mlp

        self.amp_input_bn = nn.BatchNorm1d(relay_in_channels)
        self.phase_input_bn = nn.BatchNorm1d(relay_in_channels)

        self.amp_branch = RelayKANBranch(
            in_channels=relay_in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
            pool_every=pool_every,
            base_activation=base_activation,
            allow_pooling=allow_pooling,
        )
        self.phase_branch = RelayKANBranch(
            in_channels=relay_in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
            pool_every=pool_every,
            base_activation=base_activation,
            allow_pooling=allow_pooling,
        )
        self.gate_branch = RelayKANBranch(
            in_channels=relay_in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
            pool_every=pool_every,
            base_activation=base_activation,
            allow_pooling=allow_pooling,
        )

        relay_out_channels = self.amp_branch.out_channels
        grid_head = grid_size[0] if isinstance(grid_size, list) else grid_size

        if self.use_mlp:
            pts = input_size // in_channels
            mlp_input_size = relay_out_channels * 2 * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation,
            )
        else:
            head_input_size = relay_out_channels * 2
            head_hidden = max(4, head_input_size // 2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.processing_net = nn.Sequential(
                build_kan_linear(
                    backend=self.kan_backend,
                    in_features=head_input_size,
                    out_features=head_hidden,
                    grid_size=grid_head,
                    spline_order=spline_order,
                    base_activation=base_activation,
                ),
                build_kan_linear(
                    backend=self.kan_backend,
                    in_features=head_hidden,
                    out_features=num_classes,
                    grid_size=grid_head,
                    spline_order=spline_order,
                    base_activation=base_activation,
                ),
            )

    @staticmethod
    def _split_amp_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = x[:, 0::2, :]
        phase = x[:, 1::2, :]
        return amp, phase

    @staticmethod
    def _stack_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        batch, channels, length = amp.shape
        out = torch.empty(batch, channels * 2, length, device=amp.device, dtype=amp.dtype)
        out[:, 0::2, :] = amp
        out[:, 1::2, :] = phase
        return out

    @staticmethod
    def _apply_phase_relay(amp: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp_safe = F.softplus(amp)
        gate = torch.sigmoid(gate_logits)
        return amp_safe * gate, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"rPhysicsKAN ожидает вход размерности [B, C, T], получено {tuple(x.shape)}")

        if x.shape[1] % 4 != 0:
            raise ValueError(f"rPhysicsKAN ожидает число каналов кратное 4, получено {x.shape[1]}")

        # Физические признаки строятся до KAN-веток и служат дополнительным источником информации.
        s = self.mult(x)
        z = self.div(x)

        x_features = torch.cat([x, s, z], dim=1)
        amp, phase = self._split_amp_phase(x_features)

        amp = F.softplus(self.amp_input_bn(amp))
        phase = self.phase_input_bn(phase)

        amp_features = self.amp_branch(amp)
        phase_features = self.phase_branch(phase)
        gate_logits = self.gate_branch(phase)

        gated_amp, _ = self._apply_phase_relay(amp_features, gate_logits)
        gated_amp, phase_features = self.dropout(gated_amp, phase_features)

        relay_features = self._stack_amp_phase(gated_amp, phase_features)

        if self.use_mlp:
            return self.processing_net(relay_features)

        relay_features = self.pool(relay_features).flatten(1)
        return self.processing_net(relay_features)


# ============================================================================
# Версия 2: физические/релейные блоки на глубоких слоях
# ----------------------------------------------------------------------------
# Идея: в исходных Physics-моделях физика (умножение/деление) и релейный орган
# применяются ТОЛЬКО на входе (stem). Здесь исследуется гипотеза о пользе их
# применения и на последующих (скрытых) слоях. Блоки спроектированы как
# резидуальные добавки с малым начальным масштабом, чтобы не разрушать
# основной поток признаков и не ломать обучение.
# ============================================================================


def _finite_tanh(x: torch.Tensor) -> torch.Tensor:
    """Ограничить выбросы без изменения знака и убрать нечисловые значения."""
    return torch.tanh(torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0))


def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
    """Вернуть углы к главному диапазону [-pi, pi] без потери периодичности."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class PhysicsInteractionBlock(nn.Module):
    """
    Резидуальный физический блок для скрытых слоёв (ограниченное число взаимодействий).

    В отличие от stem, где физика применяется ко всем парам каналов, здесь
    вычисляется лишь `n_interactions` физических взаимодействий. Все каналы
    линейно проецируются в 2k операндов (a_1..a_k, b_1..b_k) свёрткой 1x1, что
    позволяет взаимодействовать произвольным (в т.ч. межсигнальным) комбинациям,
    а не только соседним каналам. Для каждой пары считается произведение
    (аналог мощности) и отношение (аналог проводимости). Как в Physical
    KAN-Transformer, делительная ветвь и физические признаки проходят tanh-
    сжатие: это сохраняет знак и порядок, но не даёт малым знаменателям
    разнести KAN-сетку за рабочий диапазон. Результат смешивается KAN-свёрткой
    1x1 обратно в C каналов, дополнительно нормируется/ограничивается и
    добавляется к потоку с обучаемым масштабом. Сохраняет форму [B, C, T].

    Args:
        channels: число каналов входа.
        n_interactions: число физических взаимодействий k (операндных пар).
        grid_size: размер сетки для KAN-свёртки.
        spline_order: порядок сплайна KAN.
        epsilon: защита от деления на ноль (с сохранением знака).
        init_scale: начальный масштаб резидуальной добавки.
        base_activation: базовая активация KAN.
    """

    def __init__(
        self,
        channels: int,
        n_interactions: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        epsilon: float = 1e-4,
        init_scale: float = 0.1,
        base_activation=torch.nn.SiLU,
    ):
        super().__init__()
        k = max(1, int(n_interactions))
        self.k = k
        self.epsilon = float(epsilon)
        grid = grid_size[0] if isinstance(grid_size, list) else grid_size

        # Линейная проекция всех каналов в 2k операндов (a_1..a_k, b_1..b_k).
        self.operand_proj = nn.Conv1d(channels, 2 * k, kernel_size=1)
        # Признаки физики: mult (k) + div (k) = 2k
        self.bn = nn.BatchNorm1d(2 * k)
        self.mix = KANConv1d(
            2 * k,
            channels,
            kernel_size=1,
            grid_size=grid,
            spline_order=spline_order,
            base_activation=base_activation,
        )
        self.delta_bn = nn.BatchNorm1d(channels)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_phys = _finite_tanh(x)
        op = self.operand_proj(x_phys)
        op = torch.nan_to_num(op, nan=0.0, posinf=20.0, neginf=-20.0)
        a = op[:, : self.k, :]
        b = op[:, self.k:, :]

        mult = _finite_tanh(a * b)

        # Безопасное деление с сохранением знака знаменателя.
        sign = torch.sign(b)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denom = torch.where(b.abs() < self.epsilon, sign * self.epsilon, b)
        div = _finite_tanh(a / denom)

        phys = torch.cat([mult, div], dim=1)
        phys = _finite_tanh(self.bn(phys))
        delta = self.mix(phys)
        delta = _finite_tanh(self.delta_bn(delta))
        return x + self.scale * delta


class ComplexPhysicsInteractionBlock(nn.Module):
    """
    Комплексный (полярный) физический блок для скрытых слоёв (ограниченный k).

    Латентный поток трактуется как чередующиеся [амплитуда, фаза]. Все каналы
    проецируются в k комплексных операндных пар (a, b): отдельные проекции для
    амплитуд (положительны через softplus) и фаз. Считаются комплексные
    умножение и деление в полярной форме: амплитуды перемножаются/делятся, фазы
    складываются/вычитаются. Амплитудные признаки проходят tanh-сжатие, а фазы
    возвращаются в главный диапазон [-pi, pi], чтобы скрытый физический путь не
    создавал экстремальные значения на validation. Результат смешивается KAN-
    свёрткой 1x1 обратно в C каналов, нормируется/ограничивается и добавляется
    к потоку с обучаемым масштабом. Требует чётного числа каналов.

    Args:
        channels: число каналов входа (должно быть чётным: пары [A, φ]).
        n_interactions: число комплексных взаимодействий k.
        grid_size: размер сетки для KAN-свёртки.
        spline_order: порядок сплайна KAN.
        epsilon: защита от деления на ноль по амплитуде.
        init_scale: начальный масштаб резидуальной добавки.
        base_activation: базовая активация KAN.
    """

    def __init__(
        self,
        channels: int,
        n_interactions: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        epsilon: float = 1e-4,
        init_scale: float = 0.1,
        base_activation=torch.nn.SiLU,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                f"ComplexPhysicsInteractionBlock ожидает чётное число каналов, получено {channels}"
            )
        k = max(1, int(n_interactions))
        self.k = k
        self.epsilon = float(epsilon)
        grid = grid_size[0] if isinstance(grid_size, list) else grid_size

        # Проекции амплитуд (>0) и фаз операндов (a_1..a_k, b_1..b_k).
        self.amp_proj = nn.Conv1d(channels, 2 * k, kernel_size=1)
        self.phase_proj = nn.Conv1d(channels, 2 * k, kernel_size=1)
        self.softplus = nn.Softplus()
        # Нормируем только амплитуды mult|div (2k).
        self.bn_amp = nn.BatchNorm1d(2 * k)
        # Смешиваем 4k каналов (амплитуды + фазы mult/div) обратно в channels.
        self.mix = KANConv1d(
            4 * k,
            channels,
            kernel_size=1,
            grid_size=grid,
            spline_order=spline_order,
            base_activation=base_activation,
        )
        self.delta_bn = nn.BatchNorm1d(channels)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_phys = _finite_tanh(x)
        amps = _finite_tanh(self.softplus(self.amp_proj(x_phys)))  # [0, 1)
        phs = _wrap_phase(self.phase_proj(x_phys))

        a_amp, b_amp = amps[:, : self.k, :], amps[:, self.k:, :]
        a_ph, b_ph = phs[:, : self.k, :], phs[:, self.k:, :]

        # Комплексное умножение: |S| = |a|·|b|, ∠S = ∠a + ∠b
        mul_amp = _finite_tanh(a_amp * b_amp)
        mul_ph = _wrap_phase(a_ph + b_ph)
        # Комплексное деление: |Y| = |a|/|b|, ∠Y = ∠a − ∠b
        div_amp = _finite_tanh(a_amp / b_amp.clamp_min(self.epsilon))
        div_ph = _wrap_phase(a_ph - b_ph)

        amp_feats = _finite_tanh(self.bn_amp(torch.cat([mul_amp, div_amp], dim=1)))
        ph_feats = torch.cat([mul_ph, div_ph], dim=1)
        phys = torch.cat([amp_feats, ph_feats], dim=1)
        delta = self.mix(phys)
        delta = _finite_tanh(self.delta_bn(delta))
        return x + self.scale * delta


class RelayGateBlock(nn.Module):
    """
    Релейный орган, применяемый к произвольному скрытому представлению.

    Вычисляет вентиль (gate) из самого потока признаков и мягко модулирует
    его — аналог направленного/амплитудного реле в РЗА, но работающий не
    только на входе, а на любом слое сети. Резидуальная форма
    `out = x * (1 + scale * (gate - 0.5))` при малом init_scale почти не
    искажает поток в начале обучения.

    Args:
        channels: число каналов входа.
        kernel_size: размер ядра KAN-свёртки вентиля.
        grid_size: размер сетки KAN.
        spline_order: порядок сплайна KAN.
        init_scale: начальный масштаб модуляции.
        base_activation: базовая активация KAN.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        init_scale: float = 0.1,
        base_activation=torch.nn.SiLU,
    ):
        super().__init__()
        grid = grid_size[0] if isinstance(grid_size, list) else grid_size
        self.gate_conv = KANConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            grid_size=grid,
            spline_order=spline_order,
            base_activation=base_activation,
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(x))
        return x * (1.0 + self.scale * (gate - 0.5))


class _BlockConvKAN(nn.Module):
    """
    ConvKAN-подобный backbone с возможностью вставки физического и/или релейного
    блока после каждого сверточного слоя согласно стратегии `placement`.

    placement:
        'all'  — блоки вставляются после каждого conv-слоя;
        'last' — только после последнего conv-слоя;
        'none' — блоки не вставляются (эквивалент обычного ConvKAN).

    Число физических взаимодействий ограничено и зависит от позиции:
        - на ПЕРВОМ слое со вставкой:  round(first_interaction_ratio * out_c);
        - на последующих:              min(deep_interactions, out_c).

    Опционально применяет head_relay перед глобальным пулингом
    (финальное реле «на выходе», как в реальной РЗА).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list,
        kernel_size: int,
        stride: int,
        grid_size,
        spline_order: int,
        dropout: float,
        pool_every: int,
        base_activation,
        physics_block_builder=None,
        relay_block_builder=None,
        placement: str = "all",
        head_relay_builder=None,
        first_interaction_ratio: float = 0.25,
        deep_interactions: int = 4,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        curr = in_channels
        n = len(channels)
        n_inserted = 0

        for i, out_c in enumerate(channels):
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            s = stride if i == 0 else 1

            conv = KANConv1d(
                curr,
                out_c,
                kernel_size=kernel_size,
                stride=s,
                padding=kernel_size // 2,
                grid_size=curr_grid,
                spline_order=spline_order,
                base_activation=base_activation,
            )
            bn = nn.BatchNorm1d(out_c)

            insert = (placement == "all") or (placement == "last" and i == n - 1)
            phys_block = nn.Identity()
            relay_block = nn.Identity()
            if insert:
                if physics_block_builder is not None:
                    if n_inserted == 0:
                        n_inter = max(1, round(first_interaction_ratio * out_c))
                    else:
                        n_inter = max(1, min(deep_interactions, out_c))
                    phys_block = physics_block_builder(out_c, curr_grid, n_inter)
                if relay_block_builder is not None:
                    relay_block = relay_block_builder(out_c, curr_grid)
                n_inserted += 1

            pool = SafeMaxPool1d(2) if (pool_every > 0 and (i + 1) % pool_every == 0) else nn.Identity()
            drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            self.stages.append(
                nn.ModuleDict(
                    {
                        "conv": conv,
                        "bn": bn,
                        "physics": phys_block,
                        "relay": relay_block,
                        "pool": pool,
                        "drop": drop,
                    }
                )
            )
            curr = out_c

        self.head_relay = head_relay_builder(curr) if head_relay_builder is not None else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)

        g0 = grid_size[0] if isinstance(grid_size, list) else grid_size
        self.classifier = nn.Sequential(
            KANLinear(curr, curr // 2, grid_size=g0, base_activation=base_activation),
            KANLinear(curr // 2, num_classes, grid_size=g0, base_activation=base_activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for st in self.stages:
            x = st["conv"](x)
            x = st["bn"](x)
            x = st["physics"](x)
            x = st["relay"](x)
            x = st["pool"](x)
            x = st["drop"](x)
        x = self.head_relay(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class PhysicsKANv2(BaseModel):
    """
    PhysicsKAN с физическими блоками на глубоких слоях (версия 2).

    На входе, как и в PhysicsKAN, вычисляются Power (I*U) и Admittance (I/U)
    и объединяются с исходными сигналами. Далее backbone дополнительно
    содержит резидуальные PhysicsInteractionBlock по стратегии
    `physics_placement` (по умолчанию 'all' — на каждом слое), что позволяет
    проверить гипотезу о пользе физики не только на stem.

    Совместима по конструктору с PhysicsKAN (включая режим use_mlp для snapshot).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        physics_placement: str = "all",
        block_init_scale: float = 0.1,
        first_interaction_ratio: float = 0.25,
        deep_interactions: int = 4,
    ):
        super().__init__()

        if in_channels % 2 != 0:
            raise ValueError(
                f"PhysicsKANv2 requires even number of input channels (I, U pairs), got {in_channels}"
            )

        self.mult = MultiplicationLayer()
        self.div = DivisionLayer()

        half_channels = in_channels // 2
        self.bn_mult = nn.BatchNorm1d(half_channels)
        self.bn_div = nn.BatchNorm1d(half_channels)

        self.use_mlp = use_mlp
        conv_in_channels = in_channels + half_channels * 2  # = 2 * in_channels

        if self.use_mlp:
            # Для snapshot/MLP-режима глубокие блоки не применимы (нет conv-слоёв),
            # поведение совпадает с PhysicsKAN.
            pts = input_size // in_channels
            mlp_input_size = conv_in_channels * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation,
            )
        else:
            def physics_block_builder(ch: int, grid, n_inter: int):
                return PhysicsInteractionBlock(
                    ch,
                    n_interactions=n_inter,
                    grid_size=grid,
                    spline_order=spline_order,
                    init_scale=block_init_scale,
                    base_activation=base_activation,
                )

            self.processing_net = _BlockConvKAN(
                in_channels=conv_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation,
                physics_block_builder=physics_block_builder,
                placement=physics_placement,
                first_interaction_ratio=first_interaction_ratio,
                deep_interactions=deep_interactions,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = _finite_tanh(self.bn_mult(_finite_tanh(self.mult(x))))
        z = _finite_tanh(self.bn_div(_finite_tanh(self.div(x))))
        x_combined = torch.cat([x, s, z], dim=1)
        return self.processing_net(x_combined)


class _ComplexStemMixin:
    """
    Общий комплексный (полярный) stem для cPhysicsKANv2 / rPhysicsKANv2.

    Воспроизводит логику cPhysicsKAN: чётные каналы — амплитуды, нечётные — фазы;
    физический слой считает комплексные умножение/деление, нормирует ТОЛЬКО
    амплитуды и согласованно применяет ComplexPairDropout. Требует число каналов,
    кратное 4 (пары [A, φ] и пары I/U).
    """

    @staticmethod
    def _split_amp_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0::2, :], x[:, 1::2, :]

    @staticmethod
    def _stack_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        batch, channels, length = amp.shape
        out = torch.empty(batch, channels * 2, length, device=amp.device, dtype=amp.dtype)
        out[:, 0::2, :] = amp
        out[:, 1::2, :] = phase
        return out

    def _amp_norm_only(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        amp, phase = self._split_amp_phase(x)
        amp = _finite_tanh(bn(_finite_tanh(amp)))
        phase = _wrap_phase(phase)
        amp, phase = self.dropout(amp, phase)
        return self._stack_amp_phase(amp, phase)

    def _build_complex_stem(
        self,
        in_channels: int,
        phase_bias_b: float,
        epsilon: float,
        dropout_p: float,
    ) -> None:
        self.dropout = ComplexPairDropout(dropout_p)
        self.mult = ComplexMultiplicationLayer(phase_bias_b=phase_bias_b)
        self.div = ComplexDivisionLayer(epsilon=epsilon, phase_bias_b=phase_bias_b)
        self.bn_mult_amp = nn.BatchNorm1d(in_channels // 4)
        self.bn_div_amp = nn.BatchNorm1d(in_channels // 4)

    def _complex_stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Ожидается вход [B, C, T], получено {tuple(x.shape)}")
        if x.shape[1] % 4 != 0:
            raise ValueError(f"Ожидается число каналов кратное 4, получено {x.shape[1]}")
        s = self._amp_norm_only(self.mult(x), self.bn_mult_amp)
        z = self._amp_norm_only(self.div(x), self.bn_div_amp)
        return torch.cat([x, s, z], dim=1)


class cPhysicsKANv2(BaseModel, _ComplexStemMixin):
    """
    Комплексная PhysicsKAN с физическими блоками на глубоких слоях (версия 2).

    То же, что и PhysicsKANv2, но в полярной (комплексной) плоскости: на входе —
    комплексный физический stem (как в cPhysicsKAN), а в backbone дополнительно
    размещаются резидуальные ComplexPhysicsInteractionBlock (ограниченное число
    взаимодействий) по стратегии `physics_placement`. Предназначена для валидного
    сравнения «cPhysicsKAN (физика только на stem) vs cPhysicsKANv2 (физика и
    глубже)». Требует число каналов, кратное 4.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        phase_bias_b: float = 0.0,
        epsilon: float = 1e-6,
        physics_placement: str = "all",
        block_init_scale: float = 0.1,
        first_interaction_ratio: float = 0.25,
        deep_interactions: int = 4,
    ):
        super().__init__()

        if in_channels % 4 != 0:
            raise ValueError(
                f"cPhysicsKANv2 требует число каналов кратное 4 (пары [A, φ] и I/U), получено {in_channels}"
            )

        self._build_complex_stem(in_channels, phase_bias_b, epsilon, dropout)
        self.use_mlp = use_mlp
        proc_in_channels = in_channels + (in_channels // 2) * 2  # = 2 * in_channels

        if self.use_mlp:
            pts = input_size // in_channels
            mlp_input_size = proc_in_channels * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation,
            )
        else:
            def physics_block_builder(ch: int, grid, n_inter: int):
                return ComplexPhysicsInteractionBlock(
                    ch,
                    n_interactions=n_inter,
                    grid_size=grid,
                    spline_order=spline_order,
                    init_scale=block_init_scale,
                    base_activation=base_activation,
                )

            self.processing_net = _BlockConvKAN(
                in_channels=proc_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation,
                physics_block_builder=physics_block_builder,
                placement=physics_placement,
                first_interaction_ratio=first_interaction_ratio,
                deep_interactions=deep_interactions,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processing_net(self._complex_stem_forward(x))


class rPhysicsKANv2(BaseModel, _ComplexStemMixin):
    """
    Релейная комплексная PhysicsKAN с органами на глубоких слоях (версия 2).

    Построена на том же комплексном stem и глубоких ComplexPhysicsInteractionBlock,
    что и cPhysicsKANv2, но дополнительно после каждого вставляемого слоя
    размещается релейный орган RelayGateBlock, а также (опционально) финальное
    реле «на выходе» (`relay_at_head=True`) — по аналогии с расположением органа
    в конце тракта реальной РЗА. Это даёт валидную вложенную линейку сравнения:
    cPhysicsKANv2 (комплексная физика глубоко) → rPhysicsKANv2 (то же + реле),
    изолируя именно вклад релейного механизма. Требует число каналов, кратное 4.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        phase_bias_b: float = 0.0,
        epsilon: float = 1e-6,
        physics_placement: str = "all",
        relay_at_head: bool = True,
        block_init_scale: float = 0.1,
        first_interaction_ratio: float = 0.25,
        deep_interactions: int = 4,
    ):
        super().__init__()

        if in_channels % 4 != 0:
            raise ValueError(
                f"rPhysicsKANv2 требует число каналов кратное 4 (пары [A, φ] и I/U), получено {in_channels}"
            )

        self._build_complex_stem(in_channels, phase_bias_b, epsilon, dropout)
        self.use_mlp = use_mlp
        proc_in_channels = in_channels + (in_channels // 2) * 2  # = 2 * in_channels

        if self.use_mlp:
            pts = input_size // in_channels
            mlp_input_size = proc_in_channels * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation,
            )
        else:
            def physics_block_builder(ch: int, grid, n_inter: int):
                return ComplexPhysicsInteractionBlock(
                    ch,
                    n_interactions=n_inter,
                    grid_size=grid,
                    spline_order=spline_order,
                    init_scale=block_init_scale,
                    base_activation=base_activation,
                )

            def relay_block_builder(ch: int, grid):
                return RelayGateBlock(
                    ch,
                    kernel_size=kernel_size,
                    grid_size=grid,
                    spline_order=spline_order,
                    init_scale=block_init_scale,
                    base_activation=base_activation,
                )

            head_builder = None
            if relay_at_head:
                def head_builder(ch: int):
                    return RelayGateBlock(
                        ch,
                        kernel_size=kernel_size,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        init_scale=block_init_scale,
                        base_activation=base_activation,
                    )

            self.processing_net = _BlockConvKAN(
                in_channels=proc_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation,
                physics_block_builder=physics_block_builder,
                relay_block_builder=relay_block_builder,
                placement=physics_placement,
                head_relay_builder=head_builder,
                first_interaction_ratio=first_interaction_ratio,
                deep_interactions=deep_interactions,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processing_net(self._complex_stem_forward(x))


class rKANv2(BaseModel, _ComplexStemMixin):
    """
    Релейная KAN с глубокими релейными блоками, но БЕЗ глубоких физических блоков.

    Промежуточная модель для абляционного исследования. Изолирует вклад
    глубокого релейного механизма отдельно от глубокой физики:
      - cPhysicsKANv2 = complex stem + deep physics (без реле)
      - rKANv2        = complex stem + deep relay   (без глубокой физики) ← ЭТА
      - rPhysicsKANv2 = complex stem + deep physics + deep relay (полная)

    Архитектура: комплексный stem (умножение/деление на входе) + _BlockConvKAN
    с RelayGateBlock после каждого сверточного слоя + финальное реле перед
    классификатором. Физические блоки (ComplexPhysicsInteractionBlock) НЕ
    вставляются на глубоких слоях — только релейные.

    Требует число каналов, кратное 4 (пары [A, φ] и пары I/U).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [8, 16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        pool_every: int = 1,
        base_activation=torch.nn.SiLU,
        use_mlp: bool = False,
        input_size: int = 64,
        phase_bias_b: float = 0.0,
        epsilon: float = 1e-6,
        relay_at_head: bool = True,
        block_init_scale: float = 0.1,
    ):
        super().__init__()

        if in_channels % 4 != 0:
            raise ValueError(
                f"rKANv2 требует число каналов кратное 4 (пары [A, φ] и I/U), получено {in_channels}"
            )

        self._build_complex_stem(in_channels, phase_bias_b, epsilon, dropout)
        self.use_mlp = use_mlp
        proc_in_channels = in_channels + (in_channels // 2) * 2  # = 2 * in_channels

        if self.use_mlp:
            pts = input_size // in_channels
            mlp_input_size = proc_in_channels * pts
            hidden_sizes = [h * 4 for h in channels]
            self.processing_net = SimpleKAN(
                input_size=mlp_input_size,
                hidden_sizes=hidden_sizes,
                output_size=num_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                base_activation=base_activation,
            )
        else:
            def relay_block_builder(ch: int, grid):
                return RelayGateBlock(
                    ch,
                    kernel_size=kernel_size,
                    grid_size=grid,
                    spline_order=spline_order,
                    init_scale=block_init_scale,
                    base_activation=base_activation,
                )

            head_builder = None
            if relay_at_head:
                def head_builder(ch: int):
                    return RelayGateBlock(
                        ch,
                        kernel_size=kernel_size,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        init_scale=block_init_scale,
                        base_activation=base_activation,
                    )

            self.processing_net = _BlockConvKAN(
                in_channels=proc_in_channels,
                num_classes=num_classes,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout,
                pool_every=pool_every,
                base_activation=base_activation,
                physics_block_builder=None,  # НЕТ глубокой физики
                relay_block_builder=relay_block_builder,
                placement="all",
                head_relay_builder=head_builder,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processing_net(self._complex_stem_forward(x))

