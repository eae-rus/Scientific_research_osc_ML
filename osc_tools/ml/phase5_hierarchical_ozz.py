"""Supervised OZZ adapter for the unchanged Phase 5 SSL input contract."""
from collections import OrderedDict
from pathlib import Path
import re
import numpy as np
import polars as pl
import torch
from osc_tools.ml.phase5_hierarchical_data import build_local_features, BUCKETS
from osc_tools.ml.simulated_ozz_dataset import load_raw_csv, parse_filename
from osc_tools.data_management.no_ozz_filter import get_no_ozz_files
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from osc_tools.ml.dataset_registry import create_source


def supervised_window(raw, labels, spp, blocks, basis="phase", provenance=None):
    if provenance is None:
        provenance = np.isfinite(raw).any(0).astype(np.uint8)
    features, missing, meta, positions = build_local_features(raw, spp, blocks, basis, provenance)
    polar = np.nan_to_num(features, nan=0, posinf=0, neginf=0).reshape(-1, 13, 6, 2)
    z = np.stack((polar[..., 0]*np.cos(polar[..., 1]), polar[..., 0]*np.sin(polar[..., 1])), -1)
    state = np.broadcast_to(np.array(meta['feature_provenance']).reshape(13, 6, 2)[..., 0], z.shape[:-1]).copy()
    state[missing.reshape(-1, 13, 6, 2).any(-1)] = 0
    z[state == 0] = 0
    # Label each 1/8-period interval ending at this token, without a future shift.
    starts = np.r_[0, positions[:-1]+1]
    target = np.stack([labels[a:b+1].max(0) for a, b in zip(starts, positions)]).astype(np.float32)
    return dict(z=torch.from_numpy(z), state=torch.tensor(state, dtype=torch.long),
                times=torch.tensor((positions+1)/spp, dtype=torch.float32),
                valid=torch.ones(len(z), dtype=torch.bool), target=torch.from_numpy(target),
                supervised=torch.tensor((state != 0).any((1, 2))))


def normalized_real(frame, nominal, return_basis=False):
    """Legacy labeled CSV is in physical units; missing coefficients stay missing."""
    name = frame['file_name'][0]
    match = re.fullmatch(r'(.+)_Bus (\d+)(?:\s+_event N\d+)?', name)
    if not match or match[1] not in nominal:
        return None
    row = nominal[match[1]]
    if 'YES' not in str(row.get('norm', '')):
        return None
    bus = match[2]
    raw = np.full((len(frame), 8), np.nan, np.float32)
    for j, channel in enumerate(('IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN')):
        candidates = [channel] if j < 4 else [channel+' BB', channel+' CL']
        for column in candidates:
            if column not in frame.columns:
                continue
            values = frame[column].cast(pl.Float32, strict=False).to_numpy()
            field, factor = ('Ip', 20) if j < 3 else ('Iz', 5) if j == 3 else ('Ub' if column.endswith('BB') else 'Uc', 3)
            base = row.get(f'{bus}{field}_base')
            if base is None or not np.isfinite(float(base)) or float(base) <= 0:
                continue
            if np.isfinite(values).any():
                raw[:, j] = values / (factor*float(base))
                break
    basis = 'phase'
    if not np.isfinite(raw[:, 4:7]).any():
        for location, field in [('BB', 'Ub'), ('CL', 'Uc')]:
            columns = [f'U{pair} {location}' for pair in ('AB', 'BC', 'CA')]
            base = row.get(f'{bus}{field}_base')
            if not all(c in frame.columns for c in columns) or base is None or not np.isfinite(float(base)) or float(base) <= 0:
                continue
            values = frame.select(pl.col(c).cast(pl.Float32, strict=False) for c in columns).to_numpy()
            if np.isfinite(values).any():
                raw[:, 4:7] = values/(3*float(base))
                basis = 'line'
                break
    if not (np.isfinite(raw[:, :3]).any() or np.isfinite(raw[:, 4:7]).any()):
        return None
    return (raw, basis) if return_basis else raw


def prepare_records(root, config, ssl_manifest):
    directory = Path(config['rtds_dir'])
    paths = sorted(p.name for p in directory.glob('*.csv') if parse_filename(p))
    if not paths:
        raise FileNotFoundError(f'No RTDS CSV files: {directory}')
    train, val, _, _ = stratified_sim_ozz_split(paths, seed=config['seed'])
    records = {'train': [], 'validation': []}
    for split, names in [('train', train), ('validation', val)]:
        for name in names:
            stat = (directory/name).stat()
            records[split].append(dict(kind='rtds', path=str(directory/name), group=parse_filename(name)['x']-1,
                                      bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    if config['use_real_no_ozz']:
        frame = pl.read_csv(config['real_csv'], infer_schema_length=5000)
        names = set(get_no_ozz_files(frame)[0])
        nominal = {r['name']: r for r in pl.read_csv(config['norm_csv'], infer_schema_length=None).to_dicts()}
        source = create_source(root/'data/phase5/datasets_registry.json', 'open_ee')
        ssl_groups = {}
        for split, indices in ssl_manifest['sources']['open_ee']['splits'].items():
            for i in indices:
                key = source.get_metadata(i)['file_name'].split('_Bus ')[0]
                ssl_groups.setdefault(key, set()).add(split)
        # Group all buses of one recording. Preserve assignments already used by SSL.
        groups = sorted({n.split('_Bus ')[0] for n in names})
        rng = np.random.default_rng(config['seed'])
        rng.shuffle(groups)
        assignment = {g: ('validation' if i < max(1, len(groups)//5) else 'train') for i, g in enumerate(groups)}
        assignment.update({g: ('holdout' if 'holdout' in splits else 'train' if 'train' in splits else 'validation')
                           for g, splits in ssl_groups.items()})
        skipped = []
        for part in frame.filter(pl.col('file_name').is_in(list(names))).partition_by('file_name', maintain_order=True):
            name = part['file_name'][0]
            split = assignment[name.split('_Bus ')[0]]
            if split not in records:
                continue
            part = part.sort('sample')
            adapted = normalized_real(part, nominal, return_basis=True)
            if adapted is None:
                skipped.append(name)
                continue
            raw, basis = adapted
            # Do not concatenate disconnected labeled ranges into an artificial waveform.
            samples = part['sample'].to_numpy()
            bounds = np.r_[0, np.flatnonzero(np.diff(samples) != 1)+1, len(part)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b-a >= 320:  # This legacy CSV has the established 32 samples/period.
                    records[split].append(dict(kind='real', name=name, start=int(samples[a]), raw=raw[a:b].copy(), basis=basis, group=4))
        print(f'Real no-OZZ: skipped {len(skipped)} files without usable nominal coefficients/channels.', flush=True)
    manifest = {'excluded_real_files': skipped if config['use_real_no_ozz'] else []}
    for split, items in records.items():
        manifest[split] = [{k: v for k, v in r.items() if k != 'raw'} for r in items]
        counts = [sum(r['group'] == g for r in items) for g in range(5 if config['use_real_no_ozz'] else 4)]
        if not all(counts):
            raise ValueError(f'{split}: empty data group, counts={counts}')
        print(f'{split}: RTDS types + real no-OZZ counts = {counts}', flush=True)
    return records, manifest


class OZZDataset:
    def __init__(self, records, samples, seed, max_seconds, train):
        self.groups = [[r for r in records if r['group'] == g] for g in sorted({r['group'] for r in records})]
        self.samples, self.seed, self.max_seconds, self.train = samples, seed, max_seconds, train
        self.epoch = 0
        self.cache = OrderedDict()

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if self.train else 0, index]))
        group = self.groups[index % len(self.groups)]
        record = group[int(rng.integers(len(group)))]
        if record['kind'] == 'real':
            raw, spp = record['raw'], 32.
            labels = np.zeros((len(raw), 4), np.float32)
        else:
            path = record['path']
            data = self.cache.pop(path, None)
            if data is None:
                data = load_raw_csv(path)
            self.cache[path] = data
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            raw, labels, spp = data['raw'], data['targets'], 1/(50*data['dt'])
        available = min(int(len(raw)/(10*spp)), int(self.max_seconds*5))
        choices = [b for b in BUCKETS if b <= available]
        if not choices:
            raise ValueError(f'Record shorter than 10 periods: {record.get("path", record.get("name"))}')
        blocks = int(rng.choice(choices))
        length = int(np.floor(blocks*10*spp+.5))
        start = int(rng.integers(len(raw)-length+1))
        segment = raw[start:start+length].copy()
        if self.train:
            if rng.random() < .5:
                segment *= -1
            if rng.random() < .2:
                segment *= rng.uniform(.8, 1.2)
            if rng.random() < .33:
                shift = int(rng.integers(1, 3))
                for cols in ([0, 1, 2], [4, 5, 6]):
                    segment[:, cols] = np.roll(segment[:, cols], shift, axis=1)
        return supervised_window(segment, labels[start:start+length], spp, blocks, basis=record.get('basis', 'phase'))
