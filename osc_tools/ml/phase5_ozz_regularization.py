"""Training-only OZZ regularization; preserves all SSL parameter names/shapes."""
import torch
from osc_tools.ml.models.phasor_hierarchical import PhasorHierarchicalKAN


def smooth_binary_targets(target, error_probability=0.05):
    """Independent Bernoulli labels: 0 -> .05, 1 -> .95 (not softmax)."""
    if not 0 <= error_probability < .5:
        raise ValueError('Label error probability must be in [0, .5)')
    return target * (1 - 2 * error_probability) + error_probability


class OZZRegularizedBackbone(PhasorHierarchicalKAN):
    """Stochastic depth across complete residual blocks, shared by all tokens.

    CPU RNG chooses before checkpointing, so recomputation never resamples.
    Physical stem and fusion context projections are always evaluated.
    A retained block's entire residual increment is divided by survival rate.
    No extra parameters/buffers: old SSL state_dict loads with strict=True.
    """
    def __init__(self, cfg, drop_path_rate=0.):
        super().__init__(cfg)
        if not 0 <= drop_path_rate < 1:
            raise ValueError('drop_path_rate must be in [0, 1)')
        self.block_drop_rates = {}
        for stage in (self.channel_blocks, self.local_blocks, self.global_blocks, self.fusion_blocks):
            for index, block in enumerate(stage):
                self.block_drop_rates[id(block)] = drop_path_rate * (index + 1) / len(stage)

    def run_block(self, block, x, valid, phi=None):
        rate = self.block_drop_rates[id(block)] if self.training else 0.
        if rate == 0:
            return super().run_block(block, x, valid, phi)
        if torch.rand(()) < rate:
            return x * valid.unsqueeze(-1)
        result = super().run_block(block, x, valid, phi)
        return (x + (result - x) / (1 - rate)) * valid.unsqueeze(-1)
