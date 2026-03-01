from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class PDR_MLP_v1(nn.Module):
    """Упрощённая совместимая реализация PDR MLP v1."""

    def __init__(
        self,
        frame_size: int | None = None,
        channel_num: int = 4,
        hidden_size: int = 20,
        device: torch.device | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.channel_num = int(channel_num)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(self.channel_num, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.LeakyReLU(),
            nn.Linear(max(1, hidden_size // 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.channel_num:
            if x.shape[1] > self.channel_num:
                x = x[:, : self.channel_num]
            else:
                pad = torch.zeros(x.shape[0], self.channel_num - x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        if self.device is not None:
            x = x.to(self.device)
        return self.net(x)


class PDR_MLP_v2(nn.Module):
    """Совместимая модульная MLP с атрибутом blocks для тестов и legacy-кода."""

    def __init__(
        self,
        input_features: int,
        block_neuron_config: Iterable[int] = (3, 2, 1),
        activation_type: str = 'leaky_relu',
        use_skip_connection: bool = True,
        coeff_regular: int = 4,
        coeff_min: int = 2,
        coeff_compare: int = 2,
        coeff_mul: int = 1,
        coeff_div: int = 1,
        device: torch.device | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.expected_input_features = int(input_features)
        self.device = device
        self.use_skip_connection = bool(use_skip_connection)

        act: nn.Module
        if activation_type == 'sigmoid':
            act = nn.Sigmoid()
        else:
            act = nn.LeakyReLU()

        scale = max(1, coeff_regular + coeff_min + coeff_compare + coeff_mul + coeff_div)
        self.blocks = nn.ModuleList()

        in_features = self.expected_input_features
        for base in block_neuron_config:
            base_int = max(1, int(base))
            out_features = max(1, base_int * scale)
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    act,
                )
            )
            in_features = out_features

        self.output_layer = nn.Linear(in_features, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.expected_input_features:
            raise ValueError(
                f'Несоответствие размерности входа: ожидалось {self.expected_input_features}, получено {x.shape[1]}'
            )

        if self.device is not None:
            x = x.to(self.device)

        history: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            out = block(x)
            if self.use_skip_connection and index >= 2 and history[index - 2].shape == out.shape:
                out = out + history[index - 2]
            history.append(out)
            x = out

        return self.output_activation(self.output_layer(x))


class PDR_MLP_v3(nn.Module):
    """Совместимая заглушка v3, основанная на PDR_MLP_v2."""

    def __init__(self, input_features: int, **kwargs: object) -> None:
        super().__init__()
        self.core = PDR_MLP_v2(input_features=input_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)
