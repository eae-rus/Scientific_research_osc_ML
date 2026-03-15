from __future__ import annotations

import torch
import torch.nn as nn


class PDRBlock(nn.Module):
    """Совместимый блок для PDR-сетей (минимальная реализация)."""

    def __init__(
        self,
        input_size: int,
        base_neurons: int,
        coeff_regular: int = 4,
        coeff_min: int = 2,
        coeff_compare: int = 2,
        coeff_mul: int = 1,
        coeff_div: int = 1,
        activation_type: str = 'leaky_relu',
        division_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.base_neurons = max(1, int(base_neurons))
        self.output_size = max(
            1,
            self.base_neurons
            * max(1, coeff_regular + coeff_min + coeff_compare + coeff_mul + coeff_div),
        )

        self.fc = nn.Linear(self.input_size, self.output_size)
        self.activation = nn.Sigmoid() if activation_type == 'sigmoid' else nn.LeakyReLU()
        self.register_buffer('division_epsilon', torch.tensor(division_epsilon, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))
