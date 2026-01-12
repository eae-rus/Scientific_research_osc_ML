import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.models.cnn import ResNet1D
from osc_tools.ml.models.baseline import SimpleCNN, SafeMaxPool1d
from osc_tools.ml.layers.kan_layers import KANConv1d

# --- Helpers ---

class GroupedKANConv1d(nn.Module):
    """
    Обертка для KANConv1d, реализующая групповую свертку (groups > 1).
    Разбивает вход на логические группы и обрабатывает их независимыми KANConv1d слоями.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, **kwargs):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"In_channels {in_channels} must be divisible by groups {groups}")
        if out_channels % groups != 0:
            raise ValueError(f"Out_channels {out_channels} must be divisible by groups {groups}")
        
        self.groups = groups
        self.convs = nn.ModuleList([
            KANConv1d(
                in_channels // groups, 
                out_channels // groups, 
                kernel_size=kernel_size, 
                **kwargs
            ) for _ in range(groups)
        ])

    def forward(self, x):
        # x: (Batch, Channels, Length)
        chunks = torch.chunk(x, self.groups, dim=1)
        outs = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)

class HierarchicalStem(nn.Module):
    """
    Иерархический ствол (Stem) для последовательной обработки и объединения сигналов.
    Позволяет настраивать глубину этапов Independent и Grouped.
    
    Архитектура:
    1. Independent Signal Stage: Каждый входной сигнал обрабатывается независимо.
    2. Physical Group Stage: Объединение в группы (Токи, Напряжения).
    """
    def __init__(self, in_channels, base_filters, kernel_size=3, stride=1, use_kan=False, 
                 independent_layers=2, grouped_layers=2, **kwargs):
        super().__init__()
        
        # --- Stage 1: Independent Signals ---
        # Мы хотим обрабатывать каждый из `in_channels` отдельно.
        # groups = in_channels.
        
        # Гарантируем кратность для групп
        if base_filters % in_channels != 0:
            base_filters = ((base_filters // in_channels) + 1) * in_channels
            
        stage1_out = base_filters
        
        if use_kan:
            ConvType = GroupedKANConv1d
            conv_kwargs = kwargs
            def act_layer(): return nn.Identity()
            def bn_layer(ch): return nn.Identity()
        else:
            ConvType = nn.Conv1d
            conv_kwargs = {'padding': kernel_size//2, 'bias': False}
            def act_layer(): return nn.SiLU()
            def bn_layer(ch): return nn.BatchNorm1d(ch)
        
        # Строим слои Independent Stage
        s1_layers_list = []
        curr = in_channels
        
        for i in range(independent_layers):
            s = stride if i == 0 else 1
            # Первый слой повышает размерность до stage1_out, остальные сохраняют
            out = stage1_out 
            
            s1_layers_list.append(
                ConvType(curr, out, kernel_size, stride=s, groups=in_channels, **conv_kwargs)
            )
            s1_layers_list.append(bn_layer(out))
            s1_layers_list.append(act_layer())
            curr = out
            
        self.s1 = nn.Sequential(*s1_layers_list)
        
        # --- Stage 2: Physical Groups (Currents / Voltages) ---
        # Предполагаем, что входные данные упорядочены: [Currents..., Voltages...]
        # Делим на 2 группы.
        
        groups_stage2 = 2
        stage2_out = stage1_out * 2 # Расширяем кол-во фильтров при объединении
        
        s2_layers_list = []
        curr_s2 = curr # Output of stage 1
        
        for i in range(grouped_layers):
            # Первый слой этого этапа расширяет каналы и меняет группы
            # Остальные сохраняют размерность
            out_s2 = stage2_out
            
            # Для первого слоя группы меняются с in_channels на 2
            # Переход: groups=in_channels -> groups=2
            # Логически, чтобы объединить каналы, мы должны уменьшить groups.
            # Если мы делаем Conv1D(..., groups=2), PyTorch сам разобьет вход на 2 больших куска 
            # (первую половину каналов и вторую) и свернет их.
            # Это как раз то что нужно: половина = токи, половина = напряжения.
            
            grp = groups_stage2 if i == 0 else groups_stage2
            
            s2_layers_list.append(
                ConvType(curr_s2, out_s2, kernel_size, stride=1, groups=grp, **conv_kwargs)
            )
            s2_layers_list.append(bn_layer(out_s2))
            s2_layers_list.append(act_layer())
            curr_s2 = out_s2
            
        self.s2 = nn.Sequential(*s2_layers_list)
        
        self.out_channels = stage2_out

    def forward(self, x):
        if len(self.s1) > 0:
            x = self.s1(x)
        if len(self.s2) > 0:
            x = self.s2(x)
        return x

# --- Models ---

class HierarchicalCNN(nn.Module):
    """
    CNN с иерархической обработкой сигналов (Hierarchical Signal Processing).
    Позволяет настраивать глубину каждого этапа.
    
    Параметры:
    - stem_config: dict с ключами 'independent_layers', 'grouped_layers' (по умолчанию 1, 1 для 'light' complexity)
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [32, 64, 128], 
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.2, use_bn: bool = True,
                 pool_every: int = 1, stem_config: dict = None):
        super().__init__()
        
        if stem_config is None:
            # Поведение по умолчанию (2+2), если конфигурация не задана
            stem_config = {'independent_layers': 2, 'grouped_layers': 2}
            
        # Stem
        base_filters = channels[0]
        self.stem = HierarchicalStem(
            in_channels, base_filters, kernel_size, stride, 
            use_kan=False, 
            **stem_config
        )
        
        # Backbone
        layers = []
        curr_in = self.stem.out_channels
        
        backbone_channels = channels[1:] if len(channels) > 1 else []
        
        for i, out_c in enumerate(backbone_channels):
            layers.append(nn.Conv1d(curr_in, out_c, kernel_size, padding=kernel_size//2, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.SiLU())
            
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_in = out_c
            
        self.features = nn.Sequential(
            self.stem,
            *layers
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(curr_in, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class HierarchicalKAN(nn.Module):
    """
    KAN версия иерархической сети.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [16, 32], 
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.2, 
                 grid_size=5, spline_order=3, pool_every: int = 1, stem_config: dict = None):
        super().__init__()
        
        if stem_config is None:
             stem_config = {'independent_layers': 2, 'grouped_layers': 2}
        
        base_filters = channels[0]
        self.stem = HierarchicalStem(
            in_channels, base_filters, kernel_size, stride, 
            use_kan=True, 
            grid_size=grid_size, spline_order=spline_order,
            **stem_config
        )
        
        layers = []
        curr_in = self.stem.out_channels
        backbone_channels = channels[1:] if len(channels) > 1 else []
        
        for i, out_c in enumerate(backbone_channels):
            layers.append(
                KANConv1d(
                    curr_in, out_c, kernel_size, padding=kernel_size//2, 
                    grid_size=grid_size, spline_order=spline_order
                )
            )
            layers.append(nn.BatchNorm1d(out_c)) 
            
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_in = out_c
            
        self.features = nn.Sequential(
            self.stem,
            *layers
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(curr_in, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class HierarchicalMLP(HierarchicalCNN):
    """
    Полносвязная (MLP) версия иерархической сети.
    Реализована через свертки 1x1 (Pointwise Convolution).
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [64, 128], 
                 dropout: float = 0.2, use_bn: bool = True, stem_config: dict = None):
        if stem_config is None:
            # Для MLP "Simple" может быть 1 indep, 1 grouped
            stem_config = {'independent_layers': 1, 'grouped_layers': 1}
            
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_size=1, # Ключевой параметр для MLP (свертка 1x1)
            stride=1,
            dropout=dropout,
            use_bn=use_bn,
            pool_every=999, # Отключаем пулинг внутри backbone
            stem_config=stem_config
        )
