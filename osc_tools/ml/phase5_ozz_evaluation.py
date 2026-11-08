"""Bounded full-record Phase 5 evaluation used by the existing manual launchers."""
from pathlib import Path
import json
import numpy as np
from osc_tools.ml.phase5_ozz_inference import predict_record, mark_record


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def event_counts(pred, truth, valid, tolerance=0):
    """One-to-one overlap matching; unknown gaps do not imply event termination."""
    # A missing Fourier warmup is not evidence that an event ended.
    predicted = pred & valid
    for a, b in runs(~valid):
        if a > 0 and b < len(valid) and predicted[a-1] and predicted[b]:
            predicted[a:b] = True
    detected = runs(predicted)
    actual = [(a, b) for a, b in runs(truth) if valid[a:b].any()]
    tp = fn = short_hit = short_total = 0
    used = set()
    for start, end in actual:
        short = end-start == 1
        short_total += short
        match = next((j for j, (p, q) in enumerate(detected)
                      if j not in used and p < end+tolerance and q > start-tolerance), None)
        if match is None:
            fn += 1
        else:
            used.add(match); tp += 1; short_hit += short
    return np.array([tp, len(detected)-len(used), fn, short_hit, short_total], np.int64)


def threshold_report(records, thresholds=(.2, .3, .4, .5, .6, .7, .8, .9)):
    from osc_tools.ml.phase5_ozz_balance import average_precision_metrics
    report = {}
    for threshold in thresholds:
        counts = np.zeros((3, 4), np.int64)
        events = np.zeros((5, 4), np.int64)
        tolerant = np.zeros((5, 4), np.int64)
        covered = np.zeros((4, 4), np.int64)
        file_hits = np.zeros((2, 4), np.int64)
        background_fp = background_n = background_files_fp = background_files = 0
        weighted_fp = weighted_files = weighted_interval_fp = weighted_intervals = 0.
        absent_fp = np.zeros(4, np.int64)
        for record in records:
            p, y, v = record['p'], record['y'].astype(bool), record['v']
            pred = p >= threshold
            counts += np.stack(((pred & y & v[:, None]).sum(0),
                                (pred & ~y & v[:, None]).sum(0),
                                (~pred & y & v[:, None]).sum(0)))
            for c in range(4):
                for a, b in runs(y[:, c]):
                    if v[a:b].any():
                        hit = bool((pred[a:b, c] & v[a:b]).any())
                        covered[0, c] += hit; covered[1, c] += 1
                        if b-a == 1:
                            covered[2, c] += hit; covered[3, c] += 1
                if y[v, c].any():
                    file_hits[0, c] += int(pred[v, c].any()); file_hits[1, c] += 1
                events[:, c] += event_counts(pred[:, c], y[:, c], v)
                tolerant[:, c] += event_counts(pred[:, c], y[:, c], v, tolerance=1)
                if not y[v, c].any():
                    absent_fp[c] += int(pred[v, c].any())
            if record['group'] == 4 and v.any():
                weight = record.get('weight', 1.)
                weighted_files += weight; weighted_fp += weight*int(pred[v].any())
                weighted_intervals += weight*int(v.sum())
                weighted_interval_fp += weight*int(pred[v].any(-1).sum())
                background_files += 1
                background_files_fp += int(pred[v].any())
                background_fp += int(pred[v].any(-1).sum()); background_n += int(v.sum())
        tp, fp, fn = counts
        def ratio(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0).tolist()
        report[str(threshold)] = dict(f1=ratio(2*tp, 2*tp+fp+fn), precision=ratio(tp, tp+fp),
            recall=ratio(tp, tp+fn), interval_fp=fp.tolist(),
            event_any_overlap_recall=ratio(covered[0], covered[1]),
            single_token_overlap_hits=covered[2].tolist(),
            positive_file_recall=ratio(file_hits[0], file_hits[1]),
            event_recall=ratio(events[0], events[0]+events[2]), event_fp=events[1].tolist(),
            event_recall_tolerance_1=ratio(tolerant[0], tolerant[0]+tolerant[2]),
            event_fp_tolerance_1=tolerant[1].tolist(),
            single_token_events=events[4].tolist(), single_token_hits=events[3].tolist(),
            absent_class_files_fp=absent_fp.tolist(), background_fp_intervals=background_fp,
            background_intervals=background_n, background_fp_files=background_files_fp,
            background_files=background_files,
            background_population_weighted_file_fpr=weighted_fp/weighted_files if weighted_files else None,
            background_population_weighted_interval_fpr=weighted_interval_fp/weighted_intervals if weighted_intervals else None)
    p = np.concatenate([r['p'][r['v']] for r in records])
    y = np.concatenate([r['y'][r['v']] for r in records])
    # AP only depends on ordering; probabilities are legitimate ranking scores.
    ap = average_precision_metrics(p, y)
    unobserved = [sum(not r['v'][a:b].any() for r in records
                      for a, b in runs(r['y'][:, c] > .5)) for c in range(4)]
    return dict(thresholds=report, **ap, unobserved_events_per_class=unobserved, threshold_policy='diagnostic_only_keep_0.5',
                event_policy='one_to_one_overlap; bridge only unknown gaps between positive predictions; exclude wholly unobserved events; separate tolerance +/-1 token; no duration filter')


def select_records(checkpoint, per_class=240, background=1200):
    from osc_tools.ml.dataset_registry import create_source
    manifest = json.loads((Path(checkpoint).parent/'split.json').read_text(encoding='utf-8'))
    selected = []
    for group in range(5):
        rng = np.random.default_rng(42+group)
        pool = sorted([r for r in manifest['validation'] if r['group'] == group],
                      key=lambda r: r.get('path', r.get('name', '')))
        if group != 4:
            selected.extend(pool[int(i)] for i in rng.permutation(len(pool))[:per_class])
            continue
        if not pool or background == 0:
            continue
        source = create_source(Path(pool[0]['registry']), 'open_ee')
        strata = {}
        for record in pool:
            meta = source.get_metadata(record['index'])
            seconds = meta['n_samples']/(record['spp']*record['f_network'])
            # Broad physical categories, independent of model outputs and labels.
            key = f"duration_{np.searchsorted([.5, 2, 10, 30], seconds)}_spp_{np.searchsorted([32, 128], record['spp'])}"
            strata.setdefault(key, []).append(record)
        budget = min(background, len(pool))
        allocation = {key: 0 for key in strata}
        # Diminishing allocation: smooth population imbalance without excluding common strata.
        for _ in range(budget):
            key = max((k for k in strata if allocation[k] < len(strata[k])),
                      key=lambda k: np.sqrt(len(strata[k]))/(allocation[k]+1))
            allocation[key] += 1
        for key, members in sorted(strata.items()):
            count = allocation[key]
            for i in rng.permutation(len(members))[:count]:
                selected.append(dict(members[int(i)], evaluation_stratum=key,
                                     population_weight=len(members)/count,
                                     stratum_population=len(members), stratum_selected=count))
    return selected


def evaluate_records(model, config, device, checkpoint, per_class=240, background=1200,
                     output=None, plots=False, workers=2):
    from tqdm import tqdm
    from osc_tools.ml.simulated_ozz_dataset import load_raw_csv
    from osc_tools.ml.dataset_registry import create_source
    from osc_tools.ml.phase5_ozz_inference import ROOT
    output = Path(output) if output else ROOT/'reports/phase5/hierarchical_ozz_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    import hashlib
    manifest = json.loads((Path(checkpoint).parent/'split.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != config['split_sha256']:
        raise ValueError('Evaluation split does not match checkpoint contract')
    selected = select_records(checkpoint, per_class, background)
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    import threading
    from osc_tools.ml.phase5_ozz_inference import prepare_record
    local = threading.local()
    def prepare_input(record):
        if not hasattr(local, 'sources'):
            local.sources = {}
        provenance = None
        if record['kind'] == 'rtds':
            stat = Path(record['path']).stat()
            if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
                raise ValueError(f"RTDS file changed since training split: {record['path']}")
            data = load_raw_csv(record['path'])
            raw, labels, fs, fnet = data['raw'], data['targets'], 1/data['dt'], 50.
        else:
            registry = record['registry']
            if registry not in local.sources:
                local.sources[registry] = create_source(Path(registry), 'open_ee')
            source = local.sources[registry]
            raw = source.load_signal(record['index']).T.copy()
            provenance = source.get_provenance(record['index']).copy()
            raw[:, provenance == 0] = np.nan
            fnet = record['f_network']; fs = record['spp']*fnet
            labels = np.zeros((len(raw), 4), np.float32)
        original_labels = labels
        if config.get('label_tail_periods', 0):
            from osc_tools.ml.phase5_ozz_features import cached_expanded_labels
            labels = cached_expanded_labels(labels, fs/fnet)
        prepared = prepare_record(raw, config, device, fs, fnet,
                                  record.get('basis', 'phase'), provenance)
        return raw, labels, fs, fnet, prepared, original_labels

    def prefetch():
        # Bounded queue: CPU Fourier/IO overlaps GPU, without buffering the whole dataset.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            pending = deque()
            iterator = iter(selected)
            for record in iterator:
                pending.append((record, executor.submit(prepare_input, record)))
                if len(pending) >= max(1, workers)+1:
                    current, future = pending.popleft()
                    yield current, future.result()
            while pending:
                current, future = pending.popleft()
                yield current, future.result()

    results = []
    for record, (raw, labels, fs, fnet, prepared, original_labels) in tqdm(prefetch(), total=len(selected), desc='OZZ full records', unit='record'):
        pred = predict_record(model, raw, config, device, fs, fnet, prepared=prepared)
        targets = np.zeros_like(pred['probabilities'])
        original_targets = np.zeros_like(targets)
        count = int(np.searchsorted(pred['starts'], len(labels)))
        if count:
            starts = pred['starts'][:count]
            targets[:count] = np.maximum.reduceat(labels, starts, axis=0)
            original_targets[:count] = np.maximum.reduceat(original_labels, starts, axis=0)
        results.append(dict(p=pred['probabilities'], y=targets, y_original=original_targets, v=pred['valid'], group=record['group'], weight=record.get('population_weight', 1.)))
        if plots and record['kind'] == 'rtds':
            from scripts.phase4_experiments.sim_ozz.plot_sim_ozz_marking import plot_sim_ozz_marking
            raw_display = load_raw_csv(record['path'], normalize=False)['raw']
            p, coverage = mark_record(model, raw, config, device, fs, fnet)
            plot_sim_ozz_marking(output/(Path(record['path']).stem+'.png'), raw_display,
                p, coverage, original_labels, fs, title=Path(record['path']).stem, threshold=.5)
    report = threshold_report(results)
    report['original_labels'] = threshold_report([dict(r, y=r['y_original']) for r in results])
    simulated = [r for r in results if r['group'] < 4]
    if simulated:
        report['simulated_only'] = threshold_report(simulated)
    report.update(checkpoint=config['checkpoint'], checkpoint_step=config['checkpoint_step'],
                  inference_sha256=config['inference_sha256'], records=selected,
                  model_contract=config, selection_seed=42,
                  evaluation_contract=dict(input='full recording; chunks <=30 seconds',
                      output='1/8 network period', labels=('one-period forward union, then any positive sample in interval'
                          if config.get('label_tail_periods', 0) else 'any positive raw sample in interval'),
                      missing='excluded from metrics, never counted as negatives',
                      classes=['Stable', 'Petersen', 'PetersSlepyan', 'Belyakov'],
                      real_expert_labels='file-level OZZ presence; no verified subtype/timing truth'),
                  usable_tokens=sum(int(r['v'].sum()) for r in results),
                  total_tokens=sum(len(r['v']) for r in results),
                  background_sampling='sqrt allocation by duration and sample density; inverse inclusion weights',
                  hard_background_records=[dict(record=record, max_probability=float(result['p'][result['v']].max()))
                      for record, result in zip(selected, results) if record['group'] == 4 and result['v'].any()
                      and result['p'][result['v']].max() >= .2])
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_threshold_summary(report, output/'threshold_summary.png')
    np.savez_compressed(output/'predictions.npz', **{f'{key}{i}': r[key]
        for i, r in enumerate(results) for key in ('p', 'y', 'y_original', 'v', 'group', 'weight')})
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'hard_background_records', 'simulated_only', 'original_labels')}, ensure_ascii=False, indent=2))
    return report


def plot_threshold_summary(report, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 8)); FigureCanvasAgg(fig)
    values = list(report['thresholds'].values())
    x = [float(t) for t in report['thresholds']]
    for c, ax in enumerate(fig.subplots(2, 2).flat):
        for key in ('precision', 'recall', 'f1', 'event_any_overlap_recall'):
            ax.plot(x, [r[key][c] for r in values], label=key)
        ax.set(title=('Stable', 'Petersen', 'Peters-Slepyan', 'Belyakov')[c],
               xlabel='Threshold', ylim=(0, 1.02))
        ax.axvline(.5, color='grey', linestyle=':'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140)
