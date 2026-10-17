import pytest
import torch

from osc_tools.ml.models.kan import (
    PhysicsKAN,
    PhysicsKANv2,
    cPhysicsKAN,
    cPhysicsKANv2,
    rPhysicsKAN,
    rPhysicsKANv2,
)
from scripts.evaluation._core.model_utils import (
    _create_model_from_config,
    _load_state_dict_safe,
)


def _small_phase2_config(model_name: str) -> dict:
    return {
        "model": {
            "name": model_name,
            "params": {
                "in_channels": 8,
                "num_classes": 4,
                "channels": [4, 8],
                "kernel_size": 3,
                "stride": 1,
                "grid_size": 3,
                "spline_order": 3,
                "dropout": 0.0,
                "pool_every": 1,
            },
        },
        "data": {
            "window_size": 32,
            "mode": "multilabel",
            "features": ["phase_polar"],
        },
    }


@pytest.mark.parametrize(
    ("model_cls", "required_keys", "required_prefixes", "forbidden_prefixes"),
    [
        (
            PhysicsKAN,
            ["bn_mult.weight", "bn_div.weight"],
            ["processing_net.features.0.kan_layer.", "processing_net.classifier.0."],
            ["features.0.", "classifier.out."],
        ),
        (
            cPhysicsKAN,
            ["bn_mult_amp.weight", "bn_div_amp.weight"],
            ["processing_net.features.0.kan_layer.", "processing_net.classifier.0."],
            ["features.0.", "classifier.out."],
        ),
        (
            rPhysicsKAN,
            ["amp_input_bn.weight", "phase_input_bn.weight"],
            [
                "amp_branch.net.0.kan_layer.",
                "phase_branch.net.0.kan_layer.",
                "gate_branch.net.0.kan_layer.",
                "processing_net.0.",
            ],
            ["features.0.", "classifier.out.", "classifier.amp_proj."],
        ),
    ],
)
def test_phase2_legacy_physics_models_keep_old_state_dict_contract(
    model_cls,
    required_keys,
    required_prefixes,
    forbidden_prefixes,
):
    model = model_cls(
        in_channels=8,
        num_classes=4,
        channels=[4, 8],
        grid_size=3,
        dropout=0.0,
    )
    keys = set(model.state_dict())

    for key in required_keys:
        assert key in keys

    for prefix in required_prefixes:
        assert any(key.startswith(prefix) for key in keys), prefix

    for prefix in forbidden_prefixes:
        assert not any(key.startswith(prefix) for key in keys), prefix


@pytest.mark.parametrize("model_cls", [PhysicsKAN, cPhysicsKAN, rPhysicsKAN])
def test_phase2_legacy_physics_models_forward_shape(model_cls):
    model = model_cls(
        in_channels=8,
        num_classes=4,
        channels=[4, 8],
        grid_size=3,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(2, 8, 32)
    x[:, 0::2, :] = x[:, 0::2, :].abs()

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 4)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    ("model_name", "legacy_cls", "v2_cls", "legacy_prefix"),
    [
        ("PhysicsKAN", PhysicsKAN, PhysicsKANv2, "processing_net.features.0.kan_layer."),
        ("cPhysicsKAN", cPhysicsKAN, cPhysicsKANv2, "processing_net.features.0.kan_layer."),
        ("rPhysicsKAN", rPhysicsKAN, rPhysicsKANv2, "amp_branch.net.0.kan_layer."),
    ],
)
def test_model_factory_keeps_phase2_names_on_legacy_classes(
    model_name,
    legacy_cls,
    v2_cls,
    legacy_prefix,
):
    model = _create_model_from_config(_small_phase2_config(model_name))

    assert isinstance(model, legacy_cls)
    assert not isinstance(model, v2_cls)
    assert any(key.startswith(legacy_prefix) for key in model.state_dict())


def test_safe_state_dict_loader_rejects_architecture_mismatch_before_eval():
    model = PhysicsKAN(
        in_channels=8,
        num_classes=4,
        channels=[4, 8],
        grid_size=3,
        dropout=0.0,
    )

    incompatible_state = {
        f"features.0.{key}": value.clone()
        for key, value in model.state_dict().items()
        if torch.is_tensor(value)
    }

    with pytest.raises(RuntimeError, match="State dict несовместим"):
        _load_state_dict_safe(
            model,
            {"model_state_dict": incompatible_state},
            exp_name="synthetic_old_phase2",
            tag="best_model",
        )


def test_safe_state_dict_loader_can_load_exact_legacy_state_dict():
    source = rPhysicsKAN(
        in_channels=8,
        num_classes=4,
        channels=[4, 8],
        grid_size=3,
        dropout=0.0,
    )
    target = rPhysicsKAN(
        in_channels=8,
        num_classes=4,
        channels=[4, 8],
        grid_size=3,
        dropout=0.0,
    )

    _load_state_dict_safe(
        target,
        {"model_state_dict": source.state_dict()},
        exp_name="synthetic_old_phase2",
        tag="best_model",
    )


@pytest.mark.parametrize(
    ("model_name", "foreign_key"),
    [
        ("PhysicsKAN", "features.0.conv.kan_layer.base_weight"),
        ("cPhysicsKAN", "features.0.conv_amp.kan_layer.base_weight"),
        ("rPhysicsKAN", "features.0.conv_gate.kan_layer.base_weight"),
    ],
)
def test_loader_rejects_intermediate_feature_backbone_checkpoints_for_legacy_names(
    model_name,
    foreign_key,
):
    model = _create_model_from_config(_small_phase2_config(model_name))
    assert model is not None

    foreign_checkpoint = {
        foreign_key: torch.randn(4, 12),
        "classifier.out.weight": torch.randn(4, 8),
        "classifier.out.bias": torch.randn(4),
    }

    with pytest.raises(RuntimeError, match="State dict несовместим"):
        _load_state_dict_safe(
            model,
            {"model_state_dict": foreign_checkpoint},
            exp_name=f"synthetic_{model_name}_wrong_arch",
            tag="best_model",
        )
