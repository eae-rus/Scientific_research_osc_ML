"""Deterministic, variable-length Phase 5 data for the new hierarchical model."""
from pathlib import Path
import math
import numpy as np
import torch
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.dataset_registry import create_source
from osc_tools.ml.phase5_splits import IndexedDatasetSource
from osc_tools.ml.phase5_ssl import apply_group_mask, SpectralMaskingConfig
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


DATA_CONTRACT = "phase5_hierarchical_b_reim_v1"
BUCKETS = (1,2,5,10,25,50,150)


class ExactFrequencyBuilder(SpectralFeatureBuilder):
    """Keep the established B layout/reference but use actual f_adc/f_network.

    The observation window is rounded to raw samples; the Fourier frequency
    is not rounded. No resampling, invented raw points, or angle interpolation.
    """
    def __init__(self, spp_actual):
        super().__init__(SpectralFeatureConfig("B"))
        self.spp_actual = spp_actual
        self.spp_rounded = int(math.floor(spp_actual+.5))

    def _phasors_prefix(self, raw, positions, window, count):
        periods = int(round(window/self.spp_rounded))
        true_window = int(math.floor(periods*self.spp_actual+.5))
        out = np.full((len(positions),8,count),np.nan+1j*np.nan,dtype=np.complex64)
        eligible = (positions>=true_window-1)&(positions<len(raw))
        ends=positions[eligible]; starts=ends-true_window+1
        if not len(ends): return out
        finite=np.isfinite(raw)
        counts=np.vstack((np.zeros((1,8),dtype=np.int64),np.cumsum(finite,axis=0)))
        valid=counts[ends+1]-counts[starts]==true_window
        safe=np.where(finite,raw,0).astype(np.float64)
        samples=np.arange(len(raw),dtype=np.float64)
        for harmonic in range(1,count+1):
            # Nyquist itself has no independent quadrature: leave it missing.
            if harmonic >= periods*self.spp_actual/2: continue
            omega=2*np.pi*harmonic/(periods*self.spp_actual)
            weighted=safe*np.exp(-1j*omega*samples)[:,None]
            prefix=np.vstack((np.zeros((1,8),dtype=np.complex128),np.cumsum(weighted,axis=0)))
            value=(prefix[ends+1]-prefix[starts])*np.exp(1j*omega*starts)[:,None]*2/true_window
            out[eligible,:,harmonic-1]=np.where(valid,value,np.nan+1j*np.nan)
        return out


def token_positions(start, count, samples_per_period):
    """Round every absolute grid position, rather than accumulating rounded stride.

    Retain actual sample timestamps. At SPP=12 strides alternate 2,1 samples.
    This is sample-grid quantization, not interpolation of wrapped angles.
    """
    if samples_per_period < 8:
        raise ValueError("At least eight raw samples per period required")
    return start+np.floor(np.arange(1,count+1)*samples_per_period/8+.5).astype(np.int64)-1


def build_local_features(segment,spp,blocks,voltage_basis,provenance):
    positions=token_positions(0,80*blocks,spp)
    builder=ExactFrequencyBuilder(spp)
    feature_parts,missing_parts=[],[]
    for block in range(blocks):
        left=int(math.floor(block*10*spp+.5))
        right=int(math.floor((block+1)*10*spp+.5))
        feature,miss,meta=builder.build(segment[left:right],int(math.floor(spp+.5)),
                   positions[block*80:(block+1)*80]-left,
                   voltage_basis=voltage_basis,channel_provenance=provenance,phasor_backend="prefix")
        feature_parts.append(feature); missing_parts.append(miss)
    return np.concatenate(feature_parts),np.concatenate(missing_parts),meta,positions


class HierarchicalDataset:
    def __init__(self, registry: Path, manifest, split="train", samples=20000,
                 seed=42, max_seconds=30., augment=True):
        self.sources = {name:IndexedDatasetSource(create_source(registry,name),
                        manifest["sources"][name]["splits"][split],split)
                        for name in ("open_ee","french_rte")}
        self.builder = SpectralFeatureBuilder(SpectralFeatureConfig("B"))
        self.samples,self.seed,self.max_seconds = samples,seed,max_seconds
        self.split,self.augment = split,augment and split=="train"
        self.epoch = 0
        # Reuse existing inversion/scaling. Phase rotation is applied below
        # together with its provenance permutation; missing channels stay NaN.
        self.augmenter = TimeSeriesAugmenter({"p_inversion":.5,"p_scaling":.2,
                          "p_phase_shuffling":0,"p_noise":0,"p_offset":0,"p_drop_channel":0})

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch):
        self.epoch = epoch

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
            # Strict local contract includes the Fourier receptive field:
            # a block must not read samples before its own ten-period window.
            segment = raw[:,start:start+raw_length].T.copy()
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
            features,missing,meta,positions=build_local_features(segment,spp,blocks,
                               str(metadata.get("voltage_basis","phase")),provenance)
            try:
                _,selected = apply_group_mask(features,missing,self.builder.schema,SpectralMaskingConfig(.25),int(rng.integers(0,2**31)))
            except ValueError:
                continue
            # Add a contiguous time span per ten-period block. Targets are
            # available observations only, never missing/low-history padding.
            for block in range(blocks):
                span = int(rng.integers(8,25))
                begin = block*80+int(rng.integers(0,81-span))
                selected[begin:begin+span] |= ~missing[begin:begin+span]
            polar = np.nan_to_num(features,nan=0,posinf=0,neginf=0).reshape(count,13,6,2)
            target = np.stack((polar[...,0]*np.cos(polar[...,1]),polar[...,0]*np.sin(polar[...,1])),-1)
            missing_pair = missing.reshape(count,13,6,2).any(-1)
            chosen = selected.reshape(count,13,6,2).all(-1)&~missing_pair
            state = np.broadcast_to(np.array(meta["feature_provenance"],dtype=np.int64).reshape(13,6,2)[...,0],(count,13,6)).copy()
            state[missing_pair] = 0
            state[chosen] = 3
            model_input = target.copy()
            model_input[(state==0)|(state==3)] = 0
            return {"z":torch.from_numpy(model_input),"target":torch.from_numpy(target),
                    "state":torch.from_numpy(state),"selected":torch.from_numpy(chosen),
                    "times":torch.tensor((positions+1)/spp,dtype=torch.float32),
                    "valid":torch.ones(count,dtype=torch.bool),
                    "metadata":{"source":name,"record":record,"start":start,"blocks":blocks,
                                "seconds":10*blocks/fnet,"spp_actual":spp,"spp_fft":int(math.floor(spp+.5)),
                                "serial":serial,"attempt":attempt}}
        raise RuntimeError(f"No usable hierarchical window for index {index}")


def collate_windows(samples):
    """Padding remains separate from missing and SSL-hidden states."""
    length = max(len(s["z"]) for s in samples)
    out = {}
    for key in ("z","target","state","selected","times","valid"):
        shape = (len(samples),length)+samples[0][key].shape[1:]
        out[key] = torch.zeros(shape,dtype=samples[0][key].dtype)
        for i,sample in enumerate(samples):
            out[key][i,:len(sample[key])] = sample[key]
    out["metadata"] = [s["metadata"] for s in samples]
    return out
