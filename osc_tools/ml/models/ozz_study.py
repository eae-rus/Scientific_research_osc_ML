"""V3 ablations and interchangeable temporal encoders for OZZ experiments."""

import math
from importlib import import_module
import torch
from torch import nn
from torch.nn import functional as F

from .phasor_hierarchical import KANFFN, PhysicalInjection
from .phasor_hierarchical_v3 import HierarchicalV3Config, PhasorHierarchicalKANV3


class VectorOnly(PhysicalInjection):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.vector = original.vector

    def forward(self, phi):
        return self.vector(phi.flatten(-3))


def matched_mlp(original, inputs, outputs):
    budget = sum(p.numel() for p in original.parameters())
    hidden = max(1, round((budget - outputs) / (inputs + outputs + 1)))
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class JointV3(PhasorHierarchicalKANV3):
    """Skip temporal encoding per phase; project all phases jointly."""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.pair, self.harmonic_mix, self.channel_id, self.mixer
        self.channel_blocks = nn.ModuleList()
        self.joint_projection = nn.Linear(cfg.harmonics * 6 * 6, cfg.width)

    def encode(self, z, state):
        # Retain the same physical features and missingness contract.
        from .phasor_hierarchical import physical_features

        visible = (state == 1) | (state == 2)
        safe = torch.where(visible.unsqueeze(-1), z.float(), 0.)
        phi = physical_features(safe, visible)
        amp = torch.sqrt(safe.square().sum(-1) + 1e-16)
        confidence = ((amp[..., :3] / self.cfg.current_noise - 1).clamp(0, 1) *
                      (amp[..., 3:] / self.cfg.voltage_noise - 1).clamp(0, 1))
        phi = torch.cat((phi[..., :3], phi[..., 3:5] * confidence.unsqueeze(-1), phi[..., 5:]), -1)
        x = torch.cat((safe, F.one_hot(state.long(), 4).float()), -1).flatten(-3)
        return self.joint_projection(x), safe, visible, phi

    def mix_channels(self, channels, available, index, mask, local_times):
        b, n, k = mask.shape
        return channels[:, index].reshape(b*n, k, self.cfg.width)


class TemporalEncoder(nn.Module):
    """Common dense readout for Transformer and a patch-token Transformer."""

    def __init__(self, cfg, patch=1, phases=6):
        super().__init__()
        self.cfg, self.patch = cfg, patch
        self.embed = nn.Linear(patch * cfg.harmonics * phases * 6, cfg.width)
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, 4 * cfg.width,
                                           cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.local_layers, enable_nested_tensor=False)
        self.decode = nn.Linear(cfg.width, patch * cfg.width) if patch > 1 else nn.Identity()

    def forward(self, z, state, times, valid, use_global=True):
        from .phasor_hierarchical import positional

        b, length = valid.shape
        x = torch.cat((z, F.one_hot(state.long(), 4).to(z.dtype)), -1).flatten(-3)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        pad = (-length) % self.patch
        x = self.embed(F.pad(x, (0, 0, 0, pad)).reshape(b, math.ceil(length / self.patch), -1))
        mask = F.pad(valid, (0, pad)).reshape(b, -1, self.patch).any(-1)
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        x = self.encoder(x + positional(times[:, ::self.patch], self.cfg.width), src_key_padding_mask=~mask)
        features = self.decode(x).reshape(b, -1, self.cfg.width)[:, :length]
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        return dict(token_features=features, local_features=features, valid_mask=valid)


class PatchTSTEncoder(nn.Module):
    """Channel-independent patch attention with a joint per-interval readout."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.shared = TemporalEncoder(cfg, patch=8, phases=1)
        self.project = nn.Linear(6 * cfg.width, cfg.width)

    def forward(self, z, state, times, valid, use_global=True):
        b, t, h, c, _ = z.shape
        # One stream per phase. All harmonics travel together in each patch.
        signal = torch.cat((z, F.one_hot(state.long(), 4).to(z.dtype)), -1)
        signal = signal.permute(0, 3, 1, 2, 4).reshape(b*c, t, h*6)
        patch = self.shared.patch
        pad = (-t) % patch
        phase_valid = ((state != 0).any(2) & valid[..., None]).transpose(1, 2).reshape(b*c, t)
        signal = signal.masked_fill(~phase_valid[..., None], 0)
        x = self.shared.embed(F.pad(signal, (0, 0, 0, pad)).reshape(b*c, -1, patch*h*6))
        mask = F.pad(phase_valid, (0, pad)).reshape(b*c, -1, patch).any(-1)
        mask[~mask.any(-1), 0] = True
        from .phasor_hierarchical import positional

        positions = positional(times[:, ::patch], self.cfg.width).repeat_interleave(c, dim=0)
        x = self.shared.encoder(x + positions, src_key_padding_mask=~mask)
        x = self.shared.decode(x).reshape(b*c, -1, self.cfg.width)[:, :t]
        x = x.masked_fill(~phase_valid[..., None], 0)
        features = self.project(x.reshape(b, c, t, -1).transpose(1, 2).flatten(-2))
        features = features.masked_fill(~valid[..., None], 0)
        return dict(token_features=features, local_features=features, valid_mask=valid)


class TimesNetEncoder(nn.Module):
    """Use the upstream TimesBlock with a dense per-interval readout."""

    def __init__(self, cfg):
        super().__init__()
        from types import SimpleNamespace
        from models.TimesNet import TimesBlock

        self.cfg = cfg
        self.embed = nn.Linear(cfg.harmonics * 36, cfg.width)
        self.norm = nn.LayerNorm(cfg.width)
        block_config = SimpleNamespace(seq_len=0, pred_len=0, top_k=3,
                                       d_model=cfg.width, d_ff=cfg.width * 2, num_kernels=3)
        self.blocks = nn.ModuleList(TimesBlock(block_config) for _ in range(3))

    def forward(self, z, state, times, valid, use_global=True):
        x = self.embed(torch.cat((z, F.one_hot(state.long(), 4).to(z.dtype)), -1).flatten(-3))
        if x.shape[1] > 3:
            for block in self.blocks:
                # Upstream TimesBlock uses this length in its period reshape.
                block.seq_len = x.shape[1]
                x = self.norm(block(x.masked_fill(~valid[..., None], 0).float()))
        x = x.masked_fill(~valid[..., None], 0)
        return dict(token_features=x, local_features=x, valid_mask=valid)


def build_encoder(variant, cfg=None, *, external_factory=None):
    cfg = cfg or HierarchicalV3Config()
    if variant in {"timesnet", "patchtst"} and external_factory is not None:
        module, separator, name = external_factory.partition(":")
        if not separator or not module or not name:
            raise ValueError("External adapter must be specified as module:factory")
        model = getattr(import_module(module), name)(cfg)
        if not isinstance(model, nn.Module):
            raise TypeError("External encoder factory must return an nn.Module")
        return model
    if variant == "timesnet":
        try:
            return TimesNetEncoder(cfg)
        except ModuleNotFoundError as exc:
            raise ImportError("TimesNet needs Time-Series-Library on PYTHONPATH") from exc
    if variant == "patchtst":
        return PatchTSTEncoder(cfg)
    if variant == "transformer":
        return TemporalEncoder(cfg)
    if variant == "patch_baseline":
        return TemporalEncoder(cfg, patch=8)
    if variant == "joint":
        return JointV3(cfg)
    if variant not in {"full", "mlp", "no_direction", "no_physics"}:
        raise ValueError(f"Unknown OZZ study encoder: {variant}")
    model = PhasorHierarchicalKANV3(cfg)
    if variant == "mlp":
        model.pair = matched_mlp(model.pair, 7, 16)
    for stage in (model.channel_blocks, model.local_blocks, model.global_blocks, model.fusion_blocks):
        for block in stage:
            if variant == "mlp" and isinstance(block.ffn, KANFFN):
                width = block.qkv.in_features
                block.ffn = matched_mlp(block.ffn, width, width)
            elif variant == "no_direction" and block.physics is not None:
                block.physics = VectorOnly(block.physics)
            elif variant == "no_physics":
                block.physics = None
    return model
