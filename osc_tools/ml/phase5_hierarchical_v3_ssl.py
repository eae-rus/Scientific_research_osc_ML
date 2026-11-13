"""V3 SSL with exact Fourier support masking on contiguous raw-time gaps."""
import math

import numpy as np
import torch
from torch import nn

from osc_tools.ml.models.phasor_hierarchical_v3 import PhasorHierarchicalKANV3
from osc_tools.ml.phase5_hierarchical_data import BUCKETS, HierarchicalDataset
from osc_tools.ml.phase5_hierarchical_v2_features import phase_window


def masked_phase_window(raw, spp, rng, **kwargs):
    batch = phase_window(raw, np.zeros((len(raw), 4), dtype=np.bool_), spp, **kwargs)
    ends = batch["sample_ends"].numpy()  # exclusive, relative to the crop
    observed = batch["state"].numpy() == 1
    # Use the same rounded sample support as ExactFrequencyBuilder.
    widths = np.floor(np.array([1]*9 + [2,4,6,10])*spp + .5).astype(int)
    starts = ends[:, None] - widths
    hidden = np.zeros(observed.shape[:2], dtype=bool)
    target_times = np.zeros(len(ends), dtype=bool)
    # Approximately one gap per ten periods, spaced to retain visible context.
    block_samples = max(1, round(10*spp))
    for left in range(0, len(raw), block_samples):
        right = min(len(raw), left + block_samples)
        length = round(rng.uniform(1, 3)*spp)
        margin = round(spp)
        if right - left < length + 2*margin:
            continue
        begin = int(rng.integers(left + margin, right - length - margin + 1))
        end = begin + length
        hidden |= (ends[:, None] > begin) & (starts < end)
        target_times |= (ends > begin) & (ends <= end)
    mask = torch.from_numpy(hidden[:, :, None] & observed)
    selected = mask & torch.from_numpy(target_times[:, None, None])
    if not selected.any() or not (batch["state"].bool() & ~mask).any():
        return None
    return dict(z=batch["z"].masked_fill(mask[..., None], 0), target=batch["z"],
                state=batch["state"].masked_fill(mask, 3), selected=selected,
                times=batch["times"], valid=batch["valid"])


class HierarchicalV3SSL(PhasorHierarchicalKANV3):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.head = nn.Linear(cfg.width, cfg.harmonics*6*2)

    def forward(self, z, state, times, valid):
        features = super().forward(z, state, times, valid)
        for output, source in (("prediction", "token_features"), ("local_prediction", "local_features")):
            features[output] = self.head(features[source]).reshape_as(z)
        return features


class HierarchicalV3Dataset(HierarchicalDataset):
    def __getitem__(self,index):
        serial = (self.epoch*self.samples+index) if self.split=="train" else index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,serial]))
        # Exact 2:1 schedule, cyclic shuffled traversal of each split population.
        for attempt in range(32):
            ticket = serial+attempt*104729
            name = "french_rte" if ticket%3==2 else "open_ee"
            source = self.sources[name]
            draw = ticket//3 if name=="french_rte" else 2*(ticket//3)+ticket%3
            step = 7919
            while math.gcd(step,len(source)) != 1:
                step += 2
            record = (draw*step+self.seed)%len(source)
            metadata = source.get_metadata(record)
            raw = source.load_signal(record)
            provenance = source.get_provenance(record).copy()
            fnet = float(metadata.get("f_network",50))
            spp = float(metadata.get("f_adc",float(metadata["spp"])*fnet))/fnet
            if spp < 8:
                continue
            available_blocks = min(int(len(raw.T)/(10*spp)),int(self.max_seconds*fnet/10))
            # Brief curriculum, all lengths remain eligible after epoch 2.
            cap = (5 if self.epoch==0 else 25 if self.epoch==1 else 150) if self.split=="train" else 150
            choices = [b for b in BUCKETS if b<=min(available_blocks,cap)]
            if not choices:
                continue
            blocks = int(rng.choice(choices))
            count = 80*blocks
            raw_length = int(np.floor(count*spp/8+.5))
            start = int(rng.integers(0,raw.shape[1]-raw_length+1))
            history_start = max(0, start - int(math.floor(10*spp+.5)))
            history_count = start - history_start
            segment = raw[:,history_start:start+raw_length].T.copy()
            if self.augment:
                # Deterministic CPU augmentation, independent of training RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(rng.integers(0,2**31)))
                    segment = self.augmenter(segment)
                if rng.random()<.33:
                    shift = int(rng.integers(1,3))
                    for indices in ([0,1,2],[4,5,6]):
                        segment[:,indices] = np.roll(segment[:,indices],shift,axis=1)
                        provenance[indices] = np.roll(provenance[indices],shift)
            batch = masked_phase_window(
                segment[history_count:], spp, rng,
                basis=str(metadata.get("voltage_basis", "phase")), provenance=provenance,
                history=segment[:history_count],
            )
            if batch is not None:
                batch["metadata"] = dict(source=name, record=record, start=start, blocks=blocks,
                                         seconds=10*blocks/fnet, serial=serial, attempt=attempt)
                return batch
        raise RuntimeError(f"No usable V3 window for index {index}")
