import pytest
import torch
from osc_tools.ml.models.kan import (
    SimpleKAN,
    ConvKAN,
    PhysicsKANConditional,
    cPhysicsKAN,
    rPhysicsKAN,
    ComplexMultiplicationLayer,
    ComplexDivisionLayer,
)

class TestKANModels:
    """
    Тесты для KAN моделей (SimpleKAN, ConvKAN).
    """

    def test_simple_kan_init(self):
        model = SimpleKAN(input_size=10, hidden_sizes=[20, 10], output_size=2)
        assert isinstance(model, SimpleKAN)
        
    def test_simple_kan_forward(self):
        batch_size = 5
        input_size = 10
        output_size = 2
        model = SimpleKAN(input_size=input_size, output_size=output_size)
        
        x = torch.randn(batch_size, input_size)
        y = model(x)
        
        assert y.shape == (batch_size, output_size)
        assert not torch.isnan(y).any()

    def test_conv_kan_init(self):
        model = ConvKAN(in_channels=3, num_classes=2)
        assert isinstance(model, ConvKAN)

    def test_conv_kan_forward(self):
        batch_size = 4
        in_channels = 3
        seq_len = 100
        num_classes = 2
        
        model = ConvKAN(in_channels=in_channels, num_classes=num_classes)
        
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_physics_kan_conditional_forward(self):
        """Smoke test для PhysicsKANConditional (проверка формы выхода)."""
        batch_size = 2
        in_channels = 8
        seq_len = 128
        model = PhysicsKANConditional(in_channels=in_channels, num_classes=4)

        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)

        assert y.shape == (batch_size, 4)
        assert not torch.isnan(y).any()

    def test_cphysics_kan_forward(self):
        """Smoke test для cPhysicsKAN (проверка формы выхода)."""
        batch_size = 2
        in_channels = 8  # [A1, φ1, A2, φ2, ...]
        seq_len = 128
        num_classes = 4

        model = cPhysicsKAN(in_channels=in_channels, num_classes=num_classes, channels=[4, 8])
        x = torch.randn(batch_size, in_channels, seq_len)
        x[:, 0::2, :] = x[:, 0::2, :].abs()  # амплитуды неотрицательные

        y = model(x)

        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_cphysics_kan_requires_even_channels(self):
        """cPhysicsKAN должен требовать чётное число каналов (амплитуда/фаза)."""
        with pytest.raises(ValueError, match="чётное число входных каналов"):
            cPhysicsKAN(in_channels=7, num_classes=4)

    def test_rphysics_kan_forward(self):
        """Smoke test для rPhysicsKAN (проверка формы выхода)."""
        batch_size = 2
        in_channels = 8
        seq_len = 128
        num_classes = 4

        model = rPhysicsKAN(in_channels=in_channels, num_classes=num_classes, channels=[4, 8])
        x = torch.randn(batch_size, in_channels, seq_len)
        x[:, 0::2, :] = x[:, 0::2, :].abs()

        y = model(x)

        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_rphysics_kan_requires_even_channels(self):
        """rPhysicsKAN должен требовать чётное число каналов (амплитуда/фаза)."""
        with pytest.raises(ValueError, match="чётное число входных каналов"):
            rPhysicsKAN(in_channels=7, num_classes=4)

    def test_rphysics_kan_phase_relay_contract(self):
        """Relay-маска должна ограничивать амплитуду диапазоном [0, 1] по gate-ветке."""
        amp = torch.tensor([[[2.0, 2.0]]])
        gate_logits = torch.tensor([[[-20.0, 20.0]]])

        gated_amp, gate = rPhysicsKAN._apply_phase_relay(amp, gate_logits)

        assert torch.all(gate >= 0.0)
        assert torch.all(gate <= 1.0)
        assert gated_amp[0, 0, 0] < 1e-6
        assert gated_amp[0, 0, 1] > 2.0
        assert torch.all(gated_amp >= 0.0)

    def test_cphysics_kan_complex_mul_div_contract(self):
        """Проверка контракта комплексных операций в полярной форме."""
        # Вход: 4 комплексных канала => 8 real-каналов [A1,φ1,A2,φ2,A3,φ3,A4,φ4]
        # Первые 2 комплексных канала делятся/умножаются на вторые 2.
        x = torch.tensor(
            [[
                [2.0], [0.3],
                [3.0], [0.5],
                [4.0], [1.2],
                [6.0], [1.5],
            ]]
        )

        mult = ComplexMultiplicationLayer()
        div = ComplexDivisionLayer(epsilon=1e-6)

        y_mul = mult(x)
        y_div = div(x)

        # mul:
        # (2,0.3)*(4,1.2) -> (8,1.5)
        # (3,0.5)*(6,1.5) -> (18,2.0)
        assert torch.allclose(y_mul[:, 0, :], torch.tensor([[8.0]]))
        assert torch.allclose(y_mul[:, 1, :], torch.tensor([[1.5]]))
        assert torch.allclose(y_mul[:, 2, :], torch.tensor([[18.0]]))
        assert torch.allclose(y_mul[:, 3, :], torch.tensor([[2.0]]))

        # div:
        # (2,0.3)/(4,1.2) -> (0.5,-0.9)
        # (3,0.5)/(6,1.5) -> (0.5,-1.0)
        assert torch.allclose(y_div[:, 0, :], torch.tensor([[0.5]]), atol=1e-6)
        assert torch.allclose(y_div[:, 1, :], torch.tensor([[-0.9]]), atol=1e-6)
        assert torch.allclose(y_div[:, 2, :], torch.tensor([[0.5]]), atol=1e-6)
        assert torch.allclose(y_div[:, 3, :], torch.tensor([[-1.0]]), atol=1e-6)
