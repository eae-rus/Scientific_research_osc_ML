"""Phase 5 hierarchical OZZ engine; manual settings live in the existing SimOZZ launcher."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import signal
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig, PhasorHierarchicalKAN
from osc_tools.ml.phase5_hierarchical_ozz import OZZDataset, prepare_records
from scripts.phase5_experiments.run_phase5_hierarchical import atomic_save, rng_state, restore_rng, worker_init
from scripts.phase5_experiments.progress import TrainingProgress

ROOT = Path(__file__).resolve().parents[2]


class OZZClassifier(nn.Module):
    """Read the final global-context features without changing the SSL source/hash."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.cfg.width, 4)
        for head in (backbone.head, backbone.local_head):
            head.requires_grad_(False)

    def forward(self, batch):
        features = []
        handle = self.backbone.head.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            self.backbone(batch['z'], batch['state'], batch['times'], batch['valid'])
        finally:
            handle.remove()
        b, t = batch['valid'].shape
        return self.classifier(features[0]).reshape(b, -1, 4)[:, :t]


def loader(dataset, workers, cursor=0):
    return DataLoader(dataset, batch_size=1, sampler=range(cursor, len(dataset)), num_workers=workers,
                      worker_init_fn=worker_init, generator=torch.Generator().manual_seed(771))


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def metrics_from_counts(tp, fp, fn, tn):
    f1 = 2*tp/(2*tp+fp+fn).clamp_min(1)
    return {'f1_per_class': f1.tolist(), 'macro_f1': f1.mean().item(),
            'precision_per_class': (tp/(tp+fp).clamp_min(1)).tolist(),
            'recall_per_class': (tp/(tp+fn).clamp_min(1)).tolist(),
            'specificity_per_class': (tn/(tn+fp).clamp_min(1)).tolist()}


def evaluate(model, dataset, device, workers, amp, progress):
    model.eval()
    counts = torch.zeros(4, 4, device=device)
    total, count = 0., 0
    progress.begin('OZZ validation', len(dataset))
    with torch.no_grad():
        for i, batch in enumerate(loader(dataset, workers)):
            batch = to_device(batch, device)
            mask = batch['supervised'] & batch['valid']
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
            logits, target = logits[mask].float(), batch['target'][mask]
            total += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum').item()
            count += target.numel()
            pred, truth = logits >= 0, target > .5
            counts += torch.stack(((pred & truth).sum(0), (pred & ~truth).sum(0),
                                   (~pred & truth).sum(0), (~pred & ~truth).sum(0)))
            progress.update(i+1, {'loss': total/max(count, 1)})
    return {'loss': total/max(count, 1), **metrics_from_counts(*counts.cpu())}


def export_curves(output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = [json.loads(s) for s in (output/'training_log.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['kind'] == 'validation']
    fig = Figure(figsize=(10, 4)); FigureCanvasAgg(fig)
    ax, bx = fig.subplots(1, 2)
    x = [r['epoch']+1 for r in rows]
    ax.plot(x, [r['loss'] for r in rows]); ax.set(xlabel='Epoch', ylabel='Validation BCE')
    for i, name in enumerate(('Stable', 'Petersen', 'PetersSlepian', 'Beliakov')):
        bx.plot(x, [r['f1_per_class'][i] for r in rows], label=name)
    bx.legend(); bx.set(xlabel='Epoch', ylabel='Validation F1', ylim=(0, 1))
    for a in (ax, bx): a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output/'training_curves.png', dpi=140)


def run(config):
    config = dict(config)
    output = Path(config['output']).resolve()
    latest = output/'latest_checkpoint.pt'
    resume = config['resume']
    if (output/'STOP').exists():
        raise ValueError('Удалите STOP перед продолжением.')
    if resume and not latest.exists():
        raise FileNotFoundError(latest)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('Каталог не пуст. Выберите новый output или resume=True.')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    state = torch.load(latest if resume else config['ssl_checkpoint'], map_location='cpu', weights_only=False)
    architecture = state['contract']['architecture']
    backbone = PhasorHierarchicalKAN(HierarchicalConfig(**architecture))
    if not resume:
        if state['contract'].get('data_contract') != 'phase5_hierarchical_b_reim_v1':
            raise ValueError('Нужен checkpoint иерархического SSL Phase 5.')
        backbone.load_state_dict(state['model'], strict=True)
    model = OZZClassifier(backbone).to(device)
    manifest_path = ROOT/'data/phase5/research_strict_splits.json'
    records, manifest = prepare_records(ROOT, config, json.loads(manifest_path.read_text(encoding='utf-8')))
    contract = {k: v for k, v in config.items() if k not in ('resume', 'epochs', 'workers', 'save_every', 'console_interval', 'ssl_checkpoint')}
    contract.update(architecture=architecture, recipe='hierarchical_ozz_v1', amp_bf16=amp,
                    split_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                    ssl_split_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    code_paths = [Path(__file__), ROOT/'osc_tools/ml/phase5_hierarchical_ozz.py',
                  ROOT/'osc_tools/ml/phase5_hierarchical_data.py', ROOT/'osc_tools/ml/models/phasor_hierarchical.py']
    contract['implementation_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in code_paths)).hexdigest()
    for key in ('real_csv', 'norm_csv'):
        if config['use_real_no_ozz']:
            contract[key+'_sha256'] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    if resume and contract != state['contract']:
        raise ValueError('Параметры/данные дообучения не совпадают с checkpoint.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance = state.get('ssl_origin') if resume else {'path': str(config['ssl_checkpoint']), 'step': state['step'], 'epoch': state['epoch']}
    train = OZZDataset(records['train'], config['samples_per_epoch'], config['seed'], config['max_seconds'], True)
    val = OZZDataset(records['validation'], config['validation_samples'], config['seed']+100000, config['max_seconds'], False)
    opt = torch.optim.AdamW([{'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': config['backbone_lr']},
                            {'params': model.classifier.parameters(), 'lr': config['head_lr']}], weight_decay=1e-4)
    epoch, cursor, step, best = 0, 0, 0, float('inf')
    if resume:
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        epoch, cursor, step, best = (state[k] for k in ('epoch', 'cursor', 'step', 'best'))
        restore_rng(state['rng'])
    del state
    progress = TrainingProgress(output, interval=config['console_interval'])
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        progress.message('Остановка: завершаю накопление, затем сохраняю checkpoint.')
    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def save(path=latest):
        atomic_save(dict(contract=contract, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                         epoch=epoch, cursor=cursor, step=step, best=best, ssl_origin=provenance), path)
    def log(row):
        with (output/'training_log.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
    progress.message(f'OZZ Phase 5 | {device} | SSL step {provenance["step"]} | {output}\n'
                     '4 типа ОЗЗ; все четыре выхода 0 = не ОЗЗ. Ctrl+C или STOP: сохранить и остановиться.')
    try:
        while epoch < config['epochs']:
            train.epoch = epoch
            model.train(); opt.zero_grad(set_to_none=True)
            tokens, targets, loss_sum = 0, 0, 0.
            progress.begin(f'OZZ epoch {epoch+1}/{config["epochs"]}', len(train), initial=cursor)
            for batch in loader(train, config['workers'], cursor):
                batch = to_device(batch, device)
                mask = batch['valid'] & batch['supervised']
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(batch)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits[mask].float(), batch['target'][mask], reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite OZZ loss')
                (loss/4096).backward()
                tokens += batch['valid'].sum().item(); targets += mask.sum().item()*4
                loss_sum += loss.item(); cursor += 1
                if tokens < config['target_tokens'] and cursor < len(train):
                    continue
                for p in model.parameters():
                    if p.grad is not None: p.grad.mul_(4096/max(targets, 1))
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                row = dict(kind='train', epoch=epoch, cursor=cursor, step=step, loss=loss_sum/max(targets, 1), grad_norm=grad)
                log(row); progress.update(cursor, {'loss': row['loss'], 'step': step, 'grad_norm': grad, 'lr': config['backbone_lr']})
                tokens, targets, loss_sum = 0, 0, 0.
                if step % config['save_every'] == 0: save()
                if stop[0] or (output/'STOP').exists():
                    save(); progress.update(cursor, force=True, status='paused'); return
            result = evaluate(model, val, device, config['workers'], amp, progress)
            log(dict(kind='validation', epoch=epoch, step=step, **result))
            progress.message(f'Epoch {epoch+1}: val BCE={result["loss"]:.6f}, macro F1={result["macro_f1"]:.4f}; F1={result["f1_per_class"]}')
            epoch += 1; cursor = 0
            improved = result['loss'] < best
            best = min(best, result['loss'])
            save()
            if improved: save(output/'best_model.pt')
            preserved_rng = rng_state()
            export_curves(output)
            restore_rng(preserved_rng)
            if stop[0] or (output/'STOP').exists(): return
        progress.update(len(val), force=True, status='completed')
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)
