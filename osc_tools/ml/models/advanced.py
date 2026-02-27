import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.models.cnn import ResNet1D, ResBlock1D, SafeMaxPool1d
from osc_tools.ml.models.baseline import SimpleCNN
from osc_tools.ml.models.kan import SimpleKAN, PhysicsKAN
from osc_tools.ml.layers.kan_layers import KANConv1d
from osc_tools.ml.layers.temporal_pooling import TemporalPooling

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
    
    Параметры:
    - expand_stage2: Если False, stage2 сохраняет кол-во каналов (для лёгких моделей).
                     Если True, удваивает каналы (старое поведение для CNN/ResNet).
    """
    def __init__(self, in_channels, base_filters, kernel_size=3, stride=1, use_kan=False, 
                 independent_layers=2, grouped_layers=2, expand_stage2=True, **kwargs):
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
            # Обеспечиваем сохранение длины (padding), если не задано иное
            conv_kwargs = kwargs.copy()
            if 'padding' not in conv_kwargs:
                conv_kwargs['padding'] = kernel_size // 2
                
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
        # expand_stage2=True: удваиваем каналы (для CNN/ResNet)
        # expand_stage2=False: сохраняем размерность (для SimpleKAN чтобы не раздувать вход)
        stage2_out = stage1_out * 2 if expand_stage2 else stage1_out
        
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
                 pool_every: int = 1, stem_config: dict = None,
                 pooling_strategy: str = "global_avg"):
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
        
        self.global_pool = TemporalPooling(channels=curr_in, strategy=pooling_strategy)
        classifier_input = curr_in * self.global_pool.output_scale
        self.classifier = nn.Linear(classifier_input, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class HierarchicalConvKAN(nn.Module):
    """
    Hierarchical Signal Processing + KAN Convolutional Backbone.
    KAN-версия иерархической сети (свёрточная).
    
    Оптимизации (v2):
    - Stem использует обычные Conv1d для скорости
    - KAN слои только в backbone
    - expand_stage2=True для свёрточных сетей (сохраняем оригинальную логику)
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [16, 32], 
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.2, 
                 grid_size=5, spline_order=3, pool_every: int = 1, stem_config: dict = None,
                 pooling_strategy: str = "global_avg"):
        super().__init__(
        )
        
        if stem_config is None:
             stem_config = {'independent_layers': 2, 'grouped_layers': 2}
        
        base_filters = channels[0]
        self.stem = HierarchicalStem(
            in_channels, base_filters, kernel_size, stride, 
            use_kan=False,  # Обычные Conv1d для скорости!
            expand_stage2=True,  # Для Conv сетей раздуваем каналы (больше capacity)
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
        
        self.global_pool = TemporalPooling(channels=curr_in, strategy=pooling_strategy)
        classifier_input = curr_in * self.global_pool.output_scale
        self.classifier = nn.Linear(classifier_input, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
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

class HierarchicalResNet(nn.Module):
    """
    ResNet с иерархическим стволом.
    Заменяет стандартный Conv1 слой ResNet на HierarchicalStem.
    """
    def __init__(self, in_channels: int, num_classes: int, layers=[2, 2, 2, 2], base_filters=64, 
                 stem_config: dict = None, pooling_strategy: str = "global_avg", **kwargs):
        super().__init__()
        
        if stem_config is None:
            stem_config = {'independent_layers': 2, 'grouped_layers': 2}
            
        # Stem вместо conv1
        # Чтобы состыковаться с ResNet, stem.out_channels должен быть совместим или мы адаптируем inplanes
        self.stem = HierarchicalStem(
            in_channels, base_filters, stride=1, # ResNet обычно имеет stride=2 в начале, но здесь мы контролируем это
            use_kan=False, 
            **stem_config
        )
        
        self.inplanes = self.stem.out_channels
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = SafeMaxPool1d(kernel_size=3, stride=2) # Опционально, как в оригинале ResNet

        # Reuse ResNet structure logic
        self.layer1 = self._make_layer(base_filters, layers[0])
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        self.avgpool = TemporalPooling(channels=base_filters * 8, strategy=pooling_strategy)
        classifier_input = base_filters * 8 * self.avgpool.output_scale
        self.fc = nn.Linear(classifier_input, num_classes)
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        # Копия логики из ResNet1D
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(ResBlock1D(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        # x = self.maxpool(x) # Можно включить, если нужно сильное понижение размерности
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

class HierarchicalSimpleKAN(nn.Module):
    """
    Hierarchical Stem + Dense KAN Backbone.
    Аналог SimpleKAN, но с физически-обоснованным препроцессингом сигналов.
    
    Ключевые оптимизации (исправлено v3):
    - Stem использует ОБЫЧНЫЕ Conv1d (быстрые), а не KANConv1d
    - GlobalAvgPool сжимает временную ось перед KAN backbone
    - KAN слои только в backbone (где они дают интерпретируемость)
    - Это даёт сопоставимое с SimpleKAN количество параметров И скорость
    
    Параметры:
    - channels: список hidden_sizes для KAN backbone (как у SimpleKAN)
    - stem_config: {'independent_layers': N, 'grouped_layers': M}
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [64, 32], 
                 grid_size=5, spline_order=3, dropout=0.0, stem_config: dict = None, input_size=3200,
                 pooling_strategy: str = "global_avg"):
        super().__init__()
        
        if stem_config is None:
            stem_config = {'independent_layers': 1, 'grouped_layers': 1}
        
        # Stem с минимальным расширением каналов
        # base_filters = in_channels чтобы сохранить размерность
        stem_filters = in_channels
        
        self.stem = HierarchicalStem(
            in_channels, stem_filters, stride=1,
            use_kan=False,  # КРИТИЧНО: используем обычные Conv1d для скорости!
            expand_stage2=False,  # НЕ раздуваем каналы в stage2
            **stem_config
        )
        
        # TemporalPooling для сжатия временной оси
        # Вместо простого AdaptiveAvgPool1d(1) можно использовать attention
        self.global_pool = TemporalPooling(channels=self.stem.out_channels, strategy=pooling_strategy)
        
        # Backbone работает с вектором размера stem.out_channels * output_scale
        kan_input_size = self.stem.out_channels * self.global_pool.output_scale
        
        self.backbone = SimpleKAN(
            input_size=kan_input_size, 
            hidden_sizes=channels, 
            output_size=num_classes,
            grid_size=grid_size, 
            spline_order=spline_order,
            dropout=dropout
        )

    def forward(self, x):
        x = self.stem(x)                  # (B, C', L)
        x = self.global_pool(x)           # (B, C') — уже без временной оси
        return self.backbone(x)

class HierarchicalPhysicsKAN(nn.Module):
    """
    Hierarchical Stem + PhysicsKAN Logic.
    Stem подготавливает "физические группы" признаков, которые затем обрабатываются через Mult/Div.
    
    Оптимизации (v2):
    - Stem не раздувает каналы (expand_stage2=False)
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [16, 32], 
                 grid_size=5, spline_order=3, dropout=0.2, stem_config: dict = None, input_size=None,
                 pooling_strategy: str = "global_avg", **kwargs):
        super().__init__()
        
        if stem_config is None:
            stem_config = {'independent_layers': 1, 'grouped_layers': 2}
            
        stem_filters = channels[0]
        
        self.stem = HierarchicalStem(
            in_channels, stem_filters, stride=1,
            use_kan=False,  # КРИТИЧНО: обычные Conv1d для скорости!
            expand_stage2=False,  # НЕ раздуваем каналы
            **stem_config
        )
        
        # PhysicsKAN ожидает четное кол-во каналов
        if self.stem.out_channels % 2 != 0:
            # Этого не должно быть по логике Stem, но на всякий случай
            raise ValueError("Stem output channels must be even for PhysicsKAN")
            
        # PhysicsKAN сам создаст ConvKAN или SimpleKAN внутри
        # Если передан input_size (для snapshot/MLP режима), нужно скорректировать его
        # т.к. PhysicsKAN будет думать, что это размер входа.
        # Но теперь входом является выход Stem.
        # Если это MLP режим (use_mlp=True внутри kwargs?), то input_size важен.
        
        adjusted_input_size = input_size
        if input_size is not None and kwargs.get('use_mlp', False):
             seq_len = input_size // in_channels
             adjusted_input_size = self.stem.out_channels * seq_len

        self.physics = PhysicsKAN(
            in_channels=self.stem.out_channels,
            num_classes=num_classes,
            channels=channels,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
            input_size=adjusted_input_size,
            pooling_strategy=pooling_strategy,
            **kwargs
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.physics(x)
        return x

