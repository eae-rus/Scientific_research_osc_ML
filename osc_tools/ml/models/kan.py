import torch
import torch.nn as nn
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
