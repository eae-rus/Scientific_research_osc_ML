"""Offline record inference, using the exact trained Phase 5 feature contract.

No overlap voting: each token owns one 1/8-period interval. Fourier warmup
and unavailable samples remain uncovered, rather than becoming negative labels.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window
from osc_tools.ml.phase5_hierarchical_data import token_positions

ROOT = Path(__file__).resolve().parents[2]


def load_hierarchical(state, checkpoint, device):
    from scripts.phase5_experiments.run_phase5_hierarchical_ozz import OZZClassifier
    from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig
    from osc_tools.ml.phase5_ozz_regularization import OZZRegularizedBackbone
    config = dict(state['contract'])
    model = OZZClassifier(OZZRegularizedBackbone(HierarchicalConfig(**config['architecture'])))
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    digest = hashlib.sha256()
    # Fingerprint loaded weights, not a path which a live trainer may overwrite.
    for key, value in sorted(state['model'].items()):
        digest.update(key.encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
    for relative in ('osc_tools/ml/phase5_ozz_inference.py', 'osc_tools/ml/phase5_fourier.py',
                     'osc_tools/ml/phase5_hierarchical_ozz.py', 'osc_tools/ml/models/phasor_hierarchical.py',
                     'osc_tools/ml/phase5_hierarchical_data.py',
                     'scripts/phase5_experiments/run_phase5_hierarchical_ozz.py'):
        digest.update((ROOT / relative).read_bytes())
    config.update(model_family='hierarchical', model_type='PhasorHierarchicalKAN',
                  data_dir=config['rtds_dir'], num_classes=4, inference_sha256=digest.hexdigest(),
                  checkpoint=str(Path(checkpoint).resolve()), checkpoint_step=state['step'])
    return model, config


def prepare_chunks(raw, spp, max_blocks, basis="phase", provenance=None):
    """Pad only the final block with missing values; never invent measured zeros."""
    total_blocks = int(np.ceil(len(raw) / (10*spp)))
    for first in range(0, total_blocks, max_blocks):
        blocks = min(max_blocks, total_blocks-first)
        offset = int(np.floor(first*10*spp+.5))
        length = int(np.floor(blocks*10*spp+.5))
        part = raw[offset:offset+length]
        if not len(part):
            continue
        padded = np.full((length, 8), np.nan, np.float32)
        padded[:len(part)] = part
        batch = supervised_window(padded, np.zeros((length, 4), np.float32), spp, blocks, basis=basis, provenance=provenance)
        ends = token_positions(0, 80*blocks, spp)+1
        starts = np.r_[0, ends[:-1]]
        batch['valid'] &= torch.from_numpy(ends <= len(part))
        usable = (batch['valid'] & batch['supervised']).numpy()
        yield batch, starts+offset, ends+offset, usable


def prepare_record(raw, config, device, fs, f_network=50., basis="phase", provenance=None):
    """Return compact interval arrays; content-addressed cache is shared by plots/stats."""
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    spp = float(fs)/float(f_network)
    if spp < 8 or not np.isfinite(spp):
        raise ValueError('Phase 5 requires at least 8 samples per network period')
    max_seconds = min(30., float(config.get('max_seconds', 30.)))
    blocks = max(1, int(max_seconds*f_network/10))
    amp = device.type == 'cuda' and bool(config.get('amp_bf16', False))
    signature = dict(model=config['inference_sha256'], fs=float(fs), fnet=float(f_network),
                     blocks=blocks, amp=amp, device=device.type, torch_version=torch.__version__, basis=basis,
                     provenance=None if provenance is None else np.asarray(provenance).tolist())
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
    digest.update(raw.tobytes())
    cache = ROOT/'reports/phase5/ozz_prediction_cache'/f'{digest.hexdigest()}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return cache, {k: saved[k] for k in saved.files}, None
    return cache, None, list(prepare_chunks(raw, spp, blocks, basis, provenance))


@torch.inference_mode()
def predict_record(model, raw, config, device, fs, f_network=50., basis="phase", provenance=None, prepared=None):
    cache, cached, chunks = (prepared if prepared is not None else
        prepare_record(raw, config, device, fs, f_network, basis, provenance))
    if cached is not None:
        return cached
    amp = device.type == "cuda" and bool(config.get("amp_bf16", False))
    probabilities, starts, ends, valid = [], [], [], []
    model.eval()
    for batch, left, right, usable in chunks:
        data = {k: (v.unsqueeze(0).pin_memory().to(device, non_blocking=True)
                    if device.type == "cuda" else v.unsqueeze(0)) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(data)
        probability = logits[0].float().sigmoid().cpu().numpy()
        if not np.isfinite(probability[usable]).all():
            raise FloatingPointError('Non-finite OZZ probability on a usable interval; record was not cached')
        probabilities.append(probability)
        starts.append(left); ends.append(right); valid.append(usable)
    result = dict(probabilities=np.concatenate(probabilities) if probabilities else np.empty((0, 4), np.float32),
                  starts=np.concatenate(starts) if starts else np.empty(0, np.int64),
                  ends=np.concatenate(ends) if ends else np.empty(0, np.int64),
                  valid=np.concatenate(valid) if valid else np.empty(0, bool))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **result)
    os.replace(temporary, cache)
    return result


def mark_record(model, raw, config, device, fs, f_network=50.):
    result = predict_record(model, raw, config, device, fs, f_network)
    probs = np.full((len(raw), 4), np.nan, np.float32)
    coverage = np.zeros(len(raw), np.int32)
    for p, a, b, valid in zip(result['probabilities'], result['starts'], result['ends'], result['valid']):
        if valid:
            probs[a:b] = p
            coverage[a:b] = 1
    return probs, coverage
