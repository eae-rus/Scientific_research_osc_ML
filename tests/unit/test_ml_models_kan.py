import pytest
import torch
from osc_tools.ml.models.kan import (
    SimpleKAN,
    ConvKAN,
    PhysicsKANConditional,
    cPhysicsKAN,
    rPhysicsKAN,
    PhysicsKANv2,
    cPhysicsKANv2,
    rPhysicsKANv2,
    rKANv2,
    PhysicsInteractionBlock,
    ComplexPhysicsInteractionBlock,
    RelayGateBlock,
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


class TestKANv2Blocks:
    """Smoke/контрактные тесты для блоков и моделей версии 2 (физика/реле на глубоких слоях)."""

    def test_physics_interaction_block_shape_and_residual(self):
        """Блок сохраняет форму [B, C, T] и при scale=0 является тождественным."""
        block = PhysicsInteractionBlock(channels=8, grid_size=3)
        block.eval()
        x = torch.randn(2, 8, 16)

        y = block(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

        # При нулевом масштабе резидуальная добавка зануляется -> тождество.
        with torch.no_grad():
            block.scale.zero_()
        y0 = block(x)
        assert torch.allclose(y0, x, atol=1e-6)

    def test_physics_interaction_block_accepts_odd_channels(self):
        """Блок с проекцией операндов работает и при нечётном числе каналов."""
        block = PhysicsInteractionBlock(channels=7, n_interactions=2, grid_size=3)
        block.eval()
        x = torch.randn(2, 7, 16)
        y = block(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_physics_interaction_block_bounds_division_spikes(self):
        """Малый знаменатель в div-ветке не должен разносить резидуальную добавку."""
        block = PhysicsInteractionBlock(channels=4, n_interactions=1, grid_size=3)
        block.eval()
        with torch.no_grad():
            block.operand_proj.weight.zero_()
            block.operand_proj.bias.copy_(torch.tensor([1.0, 0.0]))

        x = torch.zeros(2, 4, 16)
        y = block(x)

        assert torch.isfinite(y).all()
        assert y.abs().max() <= block.scale.abs().item() + 1e-6

    def test_complex_physics_interaction_block_shape_and_residual(self):
        """Комплексный физический блок сохраняет форму и при scale=0 тождественен."""
        block = ComplexPhysicsInteractionBlock(channels=8, n_interactions=3, grid_size=3)
        block.eval()
        x = torch.randn(2, 8, 16)
        y = block(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

        with torch.no_grad():
            block.scale.zero_()
        y0 = block(x)
        assert torch.allclose(y0, x, atol=1e-6)

    def test_complex_physics_interaction_block_requires_even_channels(self):
        with pytest.raises(ValueError, match="чётное число каналов"):
            ComplexPhysicsInteractionBlock(channels=7)

    def test_complex_physics_interaction_block_bounds_division_spikes(self):
        """Полярная div-ветка ограничивает выбросы амплитуды при почти нулом знаменателе."""
        block = ComplexPhysicsInteractionBlock(channels=8, n_interactions=1, grid_size=3)
        block.eval()
        with torch.no_grad():
            block.amp_proj.weight.zero_()
            block.amp_proj.bias.copy_(torch.tensor([20.0, -20.0]))
            block.phase_proj.weight.zero_()
            block.phase_proj.bias.copy_(torch.tensor([100.0, -100.0]))

        x = torch.zeros(2, 8, 16)
        y = block(x)

        assert torch.isfinite(y).all()
        assert y.abs().max() <= block.scale.abs().item() + 1e-6

    def test_relay_gate_block_shape_and_residual(self):
        """Релейный орган сохраняет форму и при scale=0 не искажает поток."""
        block = RelayGateBlock(channels=8, grid_size=3)
        block.eval()
        x = torch.randn(2, 8, 16)

        y = block(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

        with torch.no_grad():
            block.scale.zero_()
        y0 = block(x)
        assert torch.allclose(y0, x, atol=1e-6)

    def test_physics_kan_v2_forward(self):
        """Smoke test для PhysicsKANv2 (физика на каждом слое)."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 128, 4
        model = PhysicsKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], physics_placement="all",
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_physics_kan_v2_placement_last(self):
        """PhysicsKANv2 с placement='last' тоже выдаёт корректную форму."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 64, 3
        model = PhysicsKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], physics_placement="last",
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_physics_kan_v2_requires_even_channels(self):
        with pytest.raises(ValueError, match="even number of input channels"):
            PhysicsKANv2(in_channels=7, num_classes=4)

    def test_physics_kan_v2_snapshot_mlp(self):
        """В режиме snapshot (use_mlp) PhysicsKANv2 работает как плоский KAN."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 8, 4
        model = PhysicsKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], use_mlp=True, input_size=in_channels * seq_len,
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_cphysics_kan_v2_forward(self):
        """Smoke test для cPhysicsKANv2 (комплексная физика на глубоких слоях)."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 128, 4
        model = cPhysicsKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], physics_placement="all",
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        x[:, 0::2, :] = x[:, 0::2, :].abs()
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_cphysics_kan_v2_requires_channels_multiple_of_4(self):
        """cPhysicsKANv2 требует число каналов кратное 4."""
        with pytest.raises(ValueError):
            cPhysicsKANv2(in_channels=6, num_classes=4)

    def test_rphysics_kan_v2_forward(self):
        """Smoke test для rPhysicsKANv2 (комплексная физика + реле на слоях и на выходе)."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 128, 4
        model = rPhysicsKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], physics_placement="all", relay_at_head=True,
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        x[:, 0::2, :] = x[:, 0::2, :].abs()
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_rphysics_kan_v2_requires_channels_multiple_of_4(self):
        with pytest.raises(ValueError):
            rPhysicsKANv2(in_channels=6, num_classes=4)

    # ===================== rKANv2 (relay-only deep) =====================

    def test_rkan_v2_forward(self):
        """Smoke test для rKANv2 (deep relay only, без глубокой физики)."""
        batch_size, in_channels, seq_len, num_classes = 2, 8, 128, 4
        model = rKANv2(
            in_channels=in_channels, num_classes=num_classes,
            channels=[4, 8], relay_at_head=True,
        )
        x = torch.randn(batch_size, in_channels, seq_len)
        x[:, 0::2, :] = x[:, 0::2, :].abs()
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()

    def test_rkan_v2_requires_channels_multiple_of_4(self):
        """rKANv2 требует число каналов кратное 4."""
        with pytest.raises(ValueError):
            rKANv2(in_channels=6, num_classes=4)

    def test_rkan_v2_no_physics_blocks(self):
        """rKANv2 не должна содержать ComplexPhysicsInteractionBlock на слоях."""
        model = rKANv2(in_channels=8, num_classes=4, channels=[4, 8])
        for st in model.processing_net.stages:
            assert isinstance(st["physics"], torch.nn.Identity), (
                "rKANv2 не должна иметь physics-блоков на глубоких слоях"
            )
            assert not isinstance(st["relay"], torch.nn.Identity), (
                "rKANv2 должна иметь relay-блоки на глубоких слоях"
            )
