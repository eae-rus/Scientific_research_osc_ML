"""Supervised OZZ adapter for the unchanged Phase 5 SSL input contract."""
from collections import OrderedDict
from pathlib import Path
import csv
import hashlib
import math
import re
import numpy as np
import polars as pl
import torch
from osc_tools.ml.phase5_hierarchical_data import build_local_features, BUCKETS
from osc_tools.ml.simulated_ozz_dataset import load_raw_csv, parse_filename
from osc_tools.data_management.sim_ozz_split import stratified_sim_ozz_split
from osc_tools.ml.dataset_registry import create_source


def select_real_background(file_names, report_path):
    """User policy: exclude any name containing a report ID, independent of ML labels.

    Both verified '+' and '-' entries belong to the reserved expert dataset.
    Matching the entire name also excludes all buses/event crops of a recording.
    """
    report_path = Path(report_path)
    with report_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        if 'filename' not in (reader.fieldnames or []):
            raise ValueError('Expert report must contain filename')
        known = {Path(row['filename'].strip()).stem.casefold() for row in reader if row['filename'].strip()}
    if not known:
        raise ValueError('Empty expert exclusion list')
    excluded = {name: sorted(key for key in known if key in name.casefold()) for name in sorted(set(file_names))}
    excluded = {name: matches for name, matches in excluded.items() if matches}
    selected = sorted(set(file_names) - excluded.keys())
    return selected, dict(policy='report_identifier_containment_all_other_records_negative',
                         report=str(report_path.resolve()), report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         report_identifiers=len(known), input_files=len(set(file_names)),
                         selected_files=len(selected), excluded_files=excluded)


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
    manifest = {}
    if config['use_real_no_ozz']:
        background, audit = prepare_unlabeled_background(root, config, ssl_manifest)
        for split in records:
            records[split].extend(background[split])
        manifest['real_background'] = audit
    for split, items in records.items():
        manifest[split] = [{k: v for k, v in r.items() if k != 'raw'} for r in items]
        counts = [sum(r['group'] == g for r in items) for g in range(5 if config['use_real_no_ozz'] else 4)]
        if not all(counts):
            raise ValueError(f'{split}: empty data group, counts={counts}')
        print(f'{split}: RTDS types + real no-OZZ counts = {counts}', flush=True)
    return records, manifest


def prepare_unlabeled_background(root, config, ssl_manifest):
    registry = Path(config.get('registry', root/'data/phase5/datasets_registry.json')).resolve()
    source = create_source(registry, 'open_ee')
    metadata = [source.get_metadata(i) for i in range(len(source))]
    names, audit = select_real_background([m['file_name'] for m in metadata],
                  config.get('expert_report', root/'data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv'))
    eligible = set(names)
    # Preserve SSL separation and keep all buses/crops of a recording together.
    membership = {}
    for split, indices in ssl_manifest['sources']['open_ee']['splits'].items():
        for index in indices:
            key = metadata[index]['file_name'].casefold().split('_bus ')[0]
            membership.setdefault(key, set()).add(split)
    records = {'train': [], 'validation': []}
    skipped = []
    held_out = []
    for index, meta in enumerate(metadata):
        name = meta['file_name']
        if name not in eligible:
            continue
        splits = membership.get(name.casefold().split('_bus ')[0], set())
        # Never move SSL-training records into validation; reserve any mixed holdout group.
        split = 'holdout' if 'holdout' in splits else 'train' if 'train' in splits else 'validation' if 'validation' in splits else None
        if split is None:
            raise ValueError(f'Eligible Open_EE record absent from SSL splits: {name}')
        if split == 'holdout':
            held_out.append(index)
            continue
        fnet = float(meta['f_network'])
        spp = float(meta['f_adc']) / fnet
        if spp < 8 or int(meta['n_samples']) < int(np.floor(10*spp+.5)):
            skipped.append(index)
            continue
        if not meta.get('normalized'):
            raise ValueError(f'Unnormalized Open_EE record: {name}')
        records[split].append(dict(kind='open_ee', registry=str(registry), index=index,
                                   name=name, group=4, spp=spp, f_network=fnet,
                                   basis=meta.get('voltage_basis', 'phase')))
    audit.update(source_records=len(source), registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
                 source_manifest_sha256=hashlib.sha256(source.manifest_path.read_bytes()).hexdigest(),
                 holdout_indices=held_out, unusable_indices=skipped,
                 counts={key: len(value) for key, value in records.items()})
    print(f'Open_EE background: {len(source)} records; {len(audit["excluded_files"])} excluded names; '
          f'{audit["counts"]}; holdout={len(held_out)}, unusable={len(skipped)}', flush=True)
    return records, audit


class OZZDataset:
    def __init__(self, records, samples, seed, max_seconds, train, event_windows=False, background_repeats=1,
                 current_dropout_all=0., current_dropout_phase=0., current_zero_probability=.5,
                 current_evaluation=None):
        self.groups = [[r for r in records if r['group'] == g] for g in sorted({r['group'] for r in records})]
        self.schedule = [i for i, group in enumerate(self.groups) for _ in range(background_repeats if group[0]['group'] == 4 and train else 1)]
        self.samples, self.seed, self.max_seconds, self.train = samples, seed, max_seconds, train
        self.epoch = 0
        self.event_windows = event_windows
        self.current_dropout = (current_dropout_all, current_dropout_phase, current_zero_probability)
        if current_evaluation not in (None, "missing", "zero") or (train and current_evaluation):
            raise ValueError("Invalid current evaluation mode")
        self.current_evaluation = current_evaluation
        self.cache = OrderedDict()
        self.sources = {}

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if self.train else 0, index]))
        # Balanced groups, but visit every source record before repeating it.
        serial = (self.epoch*self.samples if self.train else 0) + index
        slot = serial % len(self.schedule)
        group_id = self.schedule[slot]
        group = self.groups[group_id]
        visit = (serial//len(self.schedule))*self.schedule.count(group_id) + self.schedule[:slot].count(group_id)
        stride = 7919
        while math.gcd(stride, len(group)) != 1:
            stride += 2
        record = group[(visit*stride+self.seed) % len(group)]
        note = ''
        provenance = None
        fnet = 50.
        if record['kind'] == 'open_ee':
            registry = record['registry']
            if registry not in self.sources:
                self.sources[registry] = create_source(Path(registry), 'open_ee')
            source = self.sources[registry]
            raw = source.load_signal(record['index']).T
            provenance = source.get_provenance(record['index']).copy()
            spp, fnet = record['spp'], record['f_network']
            labels = np.zeros((len(raw), 4), np.float32)
        elif record['kind'] == 'real':
            raw, spp = record['raw'], 32.
            labels = np.zeros((len(raw), 4), np.float32)
        else:
            path = record['path']
            data = self.cache.pop(path, None)
            if data is None:
                import time
                for attempt in range(3):
                    try:
                        data = load_raw_csv(path)
                        break
                    except OSError:
                        if attempt == 2: raise
                        time.sleep(.2*(attempt+1))
            self.cache[path] = data
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            raw, labels, spp = data['raw'], data['targets'], 1/(50*data['dt'])
        available = min(int(len(raw)/(10*spp)), int(self.max_seconds*fnet/10))
        choices = [b for b in BUCKETS if b <= available]
        if not choices:
            raise ValueError(f'Record shorter than 10 periods: {record.get("path", record.get("name"))}')
        blocks = int(rng.choice(choices))
        if self.train and self.event_windows and record['kind'] == 'rtds' and rng.random() < .25:
            blocks = available  # Largest whole ten-period window, up to max_seconds.
        length = int(np.floor(blocks*10*spp+.5))
        start = int(rng.integers(len(raw)-length+1))
        if self.train and self.event_windows and record['kind'] == 'rtds':
            from osc_tools.ml.phase5_ozz_balance import event_start, NoUsableEvent
            event_group = record['group']
            if not labels[:, event_group].any():
                present = np.flatnonzero(labels.any(0))
                if len(present):
                    event_group = int(rng.choice(present))
                    note = 'source_type_without_labels_using_present_type'
                elif data['ozz'].any():
                    raise ValueError(f'OZZ flag positive but all type labels absent: {path}')
                else:
                    event_group = None
                    note = 'simulation_without_event_used_as_negative'
            if event_group is not None:
                try:
                    start = event_start(labels, length, spp, event_group, rng)
                except NoUsableEvent:
                    # Retain true interval labels; never invent or erase an event.
                    note = 'event_unavailable_in_usable_intervals_random_window'
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
                    if provenance is not None:
                        provenance[cols] = np.roll(provenance[cols], shift)
        input_variant = 'full'
        if self.train:
            from osc_tools.ml.phase5_ozz_input_augmentation import augment_current_channels
            segment, provenance, input_variant = augment_current_channels(segment, provenance, rng, *self.current_dropout)
        elif self.current_evaluation:
            from osc_tools.ml.phase5_ozz_input_augmentation import augment_current_channels
            segment, provenance, input_variant = augment_current_channels(
                segment, provenance, rng, 1., 0., float(self.current_evaluation == 'zero'))
        batch = supervised_window(segment, labels[start:start+length], spp, blocks, basis=record.get('basis', 'phase'), provenance=provenance)
        batch['input_variant'] = input_variant
        if self.train and self.event_windows and record['kind'] == 'rtds':
            if labels.any() and not batch['target'][batch['supervised']].any():
                note = 'event_unavailable_in_usable_intervals_random_window'
        batch['data_note'] = note
        batch['record_name'] = record.get('path', record.get('name', str(record.get('index'))))
        batch['source_group'] = torch.tensor(record['group'])
        return batch
