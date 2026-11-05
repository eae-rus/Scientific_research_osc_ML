"""Phase 5 hierarchical OZZ engine; manual settings live in the existing SimOZZ launcher."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random
import signal
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig, PhasorHierarchicalKAN
from osc_tools.ml.phase5_hierarchical_ozz import OZZDataset, prepare_records
from osc_tools.ml.phase5_ozz_regularization import OZZRegularizedBackbone, smooth_binary_targets
from scripts.phase5_experiments.run_phase5_hierarchical import atomic_save, rng_state, restore_rng, worker_init
from scripts.phase5_experiments.progress import TrainingProgress
from scripts.phase5_experiments.runtime import activation_policy, scale_gradients, fit_execution_to_memory

from osc_tools.ml.phase5_ozz_balance import estimate_balance, balanced_window_loss, average_precision_metrics, transition_weights

ROOT = Path(__file__).resolve().parents[2]


class DataLoadingFailure(RuntimeError):
    pass


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
                      pin_memory=torch.cuda.is_available(), timeout=120 if workers else 0, worker_init_fn=worker_init, generator=torch.Generator().manual_seed(771))


def guarded_batches(dataset, workers, initial):
    try:
        iterator = iter(loader(dataset, workers, initial))
    except Exception as exc:
        raise DataLoadingFailure(str(exc)) from exc
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        except Exception as exc:
            raise DataLoadingFailure(str(exc)) from exc
        yield batch

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


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
    real_fp, real_count = 0, 0
    real_window_fp, real_windows = 0, 0
    all_logits, all_targets = [], []
    progress.begin('OZZ validation', len(dataset))
    with torch.no_grad():
        for i, batch in enumerate(guarded_batches(dataset, workers, 0)):
            batch = to_device(batch, device)
            mask = batch['supervised'] & batch['valid']
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
            logits, target = logits[mask].float(), batch['target'][mask]
            total += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum').item()
            count += target.numel()
            all_logits.append(logits.cpu().numpy()); all_targets.append(target.cpu().numpy())
            pred, truth = logits >= 0, target > .5
            if batch['source_group'].item() == 4:
                real_fp += pred.any(-1).sum().item()
                real_count += len(pred)
                if len(pred):
                    real_window_fp += int(pred.any())
                    real_windows += 1
            counts += torch.stack(((pred & truth).sum(0), (pred & ~truth).sum(0),
                                   (~pred & truth).sum(0), (~pred & ~truth).sum(0)))
            progress.update(i+1, {'loss': total/max(count, 1)})
    progress.update(len(dataset), force=True)
    return {'loss': total/max(count, 1), **metrics_from_counts(*counts.cpu()),
            **average_precision_metrics(np.concatenate(all_logits), np.concatenate(all_targets)),
            'real_no_ozz_false_positive_rate': real_fp/real_count if real_count else None,
            'real_no_ozz_window_false_positive_rate': real_window_fp/real_windows if real_windows else None,
            'real_no_ozz_windows': real_windows, 'real_no_ozz_intervals': real_count}


def export_curves(output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = [json.loads(s) for s in (output/'training_log.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['kind'] == 'validation']
    fig = Figure(figsize=(15, 4)); FigureCanvasAgg(fig)
    ax, bx, cx = fig.subplots(1, 3)
    x = [r['epoch']+1 for r in rows]
    ax.plot(x, [r['loss'] for r in rows]); ax.set(xlabel='Epoch', ylabel='Validation BCE')
    for i, name in enumerate(('Stable', 'Petersen', 'PetersSlepian', 'Beliakov')):
        bx.plot(x, [r['f1_per_class'][i] for r in rows], label=name)
    bx.legend(); bx.set(xlabel='Epoch', ylabel='Validation F1', ylim=(0, 1))
    for i, name in enumerate(('Stable', 'Petersen', 'PetersSlepian', 'Beliakov')):
        cx.plot(x, [(r.get('average_precision_per_class') or [None]*4)[i] for r in rows], label=name)
    cx.legend(); cx.set(xlabel='Epoch', ylabel='Average precision (PR)', ylim=(0, 1))
    for a in (ax, bx, cx): a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output/'training_curves.png', dpi=140)


def validate_resume_contract(saved, contract, allow_upgrade=False):
    if contract != saved:
        old = dict(saved)
        known_old = old.get('implementation_sha256') in ('ab4098ec60d58712f4b33100df327c6acd7baed4510f06fa5a9641921d9b6d3a',
            '5175d524c3260da09135ca73d32b103bad684ed758f98ad3c15ea9c64bdce235',
            'd49eba6522a0f609bb464f086486f13df21f9d976c23ce76fd17fd25a12ef64d',
            '60de3a4bbe9e4320e5cc60b79a5fea7f4c3c83460ee0c81fd9931c35a4974b62',
            '0960cc18c653fef74e82e4a322a31c4a629fb703feaee541021a61d36e14e9e2')
        if saved.get('implementation_sha256') == 'ab4098ec60d58712f4b33100df327c6acd7baed4510f06fa5a9641921d9b6d3a':
            old.update({k: contract[k] for k in ('balanced_events', 'balance_samples', 'positive_weight_cap', 'focal_gamma')})
        for key in ('lr_schedule', 'lr_decay_start', 'lr_decay_end', 'lr_min_ratio'):
            if key not in saved and key in contract:
                old[key] = contract[key]
        old['implementation_sha256'] = contract['implementation_sha256']
        if 'boundary_weight' in contract:
            old['boundary_weight'] = contract['boundary_weight']
        if 'background_repeats' in contract:
            old['background_repeats'] = contract['background_repeats']
        if not (allow_upgrade and contract['balanced_events'] and known_old and old == contract):
            raise ValueError('Параметры/данные не совпадают; переход со старого OZZ требует allow_balance_upgrade=True.')
        return True
    return False


def learning_rate_factor(config, epoch_progress):
    """Stateless schedule: epoch/cursor and the checkpoint contract define LR."""
    if config['lr_schedule'] == 'constant':
        return 1.
    start, end = config['lr_decay_start'], config['lr_decay_end']
    t = min(1., max(0., (epoch_progress-start)/(end-start)))
    floor = config['lr_min_ratio']
    return floor + (1-floor)*.5*(1+math.cos(math.pi*t))


def run(config):
    config = dict(config)
    config.setdefault('lr_schedule', 'constant')
    config.setdefault('lr_decay_start', 150.)
    config.setdefault('lr_decay_end', 300.)
    config.setdefault('lr_min_ratio', .1)
    if config['lr_schedule'] not in ('constant', 'flat_cosine'):
        raise ValueError('Unknown learning-rate schedule')
    if not (0 <= config['lr_decay_start'] < config['lr_decay_end'] and 0 < config['lr_min_ratio'] <= 1):
        raise ValueError('Invalid cosine schedule bounds')
    config.setdefault('label_error_probability', 0.)
    config.setdefault('drop_path_rate', 0.)
    config.setdefault('balanced_events', False)
    config.setdefault('balance_samples', 400)
    config.setdefault('positive_weight_cap', 100.)
    config.setdefault('focal_gamma', 0.)
    config.setdefault('boundary_weight', 1.)
    if not 0 < config['boundary_weight'] <= 1:
        raise ValueError('boundary_weight must be in (0, 1]')
    config.setdefault('background_repeats', 1)
    if config['background_repeats'] not in (1, 2, 3):
        raise ValueError('background_repeats must be 1, 2 or 3')
    if config['focal_gamma'] < 0 or config['positive_weight_cap'] < 1 or config['balance_samples'] < 1:
        raise ValueError('Invalid OZZ balance/focal settings')
    no_checkpoint_max_tokens = config.pop('no_checkpoint_max_tokens', 0)
    if no_checkpoint_max_tokens < 0:
        raise ValueError('no_checkpoint_max_tokens must be nonnegative')
    allow_upgrade = config.pop('allow_balance_upgrade', False)
    smooth_binary_targets(torch.tensor(0.), config['label_error_probability'])
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
    backbone = OZZRegularizedBackbone(HierarchicalConfig(**architecture), config['drop_path_rate'])
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
                  ROOT/'osc_tools/ml/phase5_hierarchical_data.py', ROOT/'osc_tools/ml/models/phasor_hierarchical.py',
                  ROOT/'osc_tools/ml/phase5_ozz_regularization.py', ROOT/'osc_tools/ml/phase5_fourier.py', ROOT/'osc_tools/ml/phase5_ozz_balance.py']
    contract['implementation_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in code_paths)).hexdigest()
    upgraded = validate_resume_contract(state['contract'], contract, allow_upgrade) if resume else False
    output.mkdir(parents=True, exist_ok=True)
    if upgraded:
        backup = output/f'before_balance_step_{state["step"]}.pt'
        if not backup.exists():
            atomic_save(state, backup)
    (output/'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance = state.get('ssl_origin') if resume else {'path': str(config['ssl_checkpoint']), 'step': state['step'], 'epoch': state['epoch']}
    train = OZZDataset(records['train'], config['samples_per_epoch'], config['seed'], config['max_seconds'], True, event_windows=config['balanced_events'], background_repeats=config['background_repeats'])
    val = OZZDataset(records['validation'], config['validation_samples'], config['seed']+100000, config['max_seconds'], False)
    opt = torch.optim.AdamW([{'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': config['backbone_lr']},
                            {'params': model.classifier.parameters(), 'lr': config['head_lr']}], weight_decay=1e-4)
    epoch, cursor, step, best = 0, 0, 0, float('inf')
    if resume:
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        epoch, cursor, step, best = (state[k] for k in ('epoch', 'cursor', 'step', 'best'))
        restore_rng(state['rng'])
    best_f1 = state.get('best_f1', -1.) if resume else -1.
    balance = state.get('balance') if resume else None
    if config['balanced_events'] and balance is None:
        print('Оценка весов по обучающим окнам (без validation)...', flush=True)
        reference = OZZDataset(records['train'], config['samples_per_epoch'], config['seed'], config['max_seconds'], True, event_windows=config['balanced_events'])
        balance = estimate_balance(reference, config['balance_samples'], config['positive_weight_cap'])
    if balance is not None:
        print(f'Class balance: {balance}', flush=True)
        (output/'class_balance.json').write_text(json.dumps(balance, indent=2), encoding='utf-8')
    history = state.get('balance_history', []) if resume else []
    if upgraded:
        history.append(dict(epoch=epoch, cursor=cursor, step=step, previous_contract=state['contract']))
        if not state.get('balance'):
            best_f1 = -1.
    del state
    execution = fit_execution_to_memory(1, max(80,no_checkpoint_max_tokens), no_checkpoint_max_tokens, device=device)
    (output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    progress = TrainingProgress(output, interval=config['console_interval'])
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        progress.message('Остановка: завершаю накопление, затем сохраняю checkpoint.')
    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def save(path=latest):
        atomic_save(dict(contract=contract, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                         epoch=epoch, cursor=cursor, step=step, best=best, best_f1=best_f1, balance=balance,
                         balance_history=history, ssl_origin=provenance), path)
    def log(row):
        with (output/'training_log.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
    progress.message(f'OZZ Phase 5 | {device} | SSL step {provenance["step"]} | {output}\n'
                     '4 типа ОЗЗ; все четыре выхода 0 = не ОЗЗ. Ctrl+C или STOP: сохранить и остановиться.')
    safe_cursor, safe_rng = cursor, rng_state()
    optimizer_in_progress = False
    reported = set()
    try:
        if epoch >= config['epochs']:
            progress.message('Все заданные эпохи уже завершены. Увеличьте EPOCHS для продолжения.')
            return
        if not latest.exists() or upgraded:
            save()  # Recoverable state even if the very first AdamW update fails.
        while epoch < config['epochs']:
            train.epoch = epoch
            model.train(); opt.zero_grad(set_to_none=True)
            tokens, targets, loss_sum = 0, 0, 0.
            progress.begin(f'OZZ epoch {epoch+1}/{config["epochs"]}', len(train), initial=cursor)
            for batch in guarded_batches(train, config['workers'], cursor):
                note = batch.get('data_note', [''])[0]
                name = batch.get('record_name', [''])[0]
                if note and (name, note) not in reported:
                    reported.add((name, note))
                    log(dict(kind='data_notice', epoch=epoch, cursor=cursor, record=name, reason=note))
                    progress.message(f'Данные: {note}: {name}')
                cpu_mask = batch['valid'] & batch['supervised']
                window_tokens = int(batch['valid'].sum())
                window_targets = int(cpu_mask.any()) if balance is not None else int(cpu_mask.sum())*4
                batch = to_device(batch, device)
                mask = batch['valid'] & batch['supervised']
                with activation_policy(backbone, batch['z'].shape[1], execution['no_checkpoint_max_tokens']):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        logits = model(batch)
                        target = smooth_binary_targets(batch['target'][mask], config['label_error_probability'])
                        if balance is not None:
                            loss = balanced_window_loss(logits[mask].float(), batch['target'][mask],
                                                        config['label_error_probability'], balance, config['focal_gamma'],
                                                        transition_weights(batch['target'], mask, config['boundary_weight'])[mask])
                        else:
                            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask].float(), target, reduction='sum')
                    (loss/4096).backward()
                tokens += window_tokens; targets += window_targets
                loss_sum += loss.detach(); cursor += 1
                if tokens < config['target_tokens'] and cursor < len(train):
                    continue
                if targets == 0:
                    opt.zero_grad(set_to_none=True)
                    tokens, loss_sum = 0, 0.
                    safe_cursor, safe_rng = cursor, rng_state()
                    if stop[0] or (output/'STOP').exists():
                        save(); progress.update(cursor, force=True, status='paused'); return
                    continue
                loss_sum = float(loss_sum)
                if not math.isfinite(loss_sum):
                    raise FloatingPointError('Non-finite OZZ loss')
                scale_gradients(model, 4096/max(targets, 1))
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
                factor = learning_rate_factor(config, epoch + cursor/len(train))
                for group, base in zip(opt.param_groups, (config['backbone_lr'], config['head_lr'])):
                    group['lr'] = base*factor
                optimizer_in_progress = True
                opt.step()
                optimizer_in_progress = False
                step += 1
                safe_cursor, safe_rng = cursor, rng_state()
                opt.zero_grad(set_to_none=True)
                row = dict(kind='train', epoch=epoch, cursor=cursor, step=step, loss=loss_sum/max(targets, 1), grad_norm=grad,
                           backbone_lr=opt.param_groups[0]['lr'], head_lr=opt.param_groups[1]['lr'])
                log(row); progress.update(cursor, {'loss': row['loss'], 'step': step, 'grad_norm': grad, 'lr': opt.param_groups[0]['lr']})
                tokens, targets, loss_sum = 0, 0, 0.
                if step % config['save_every'] == 0: save()
                if stop[0] or (output/'STOP').exists():
                    save(); progress.update(cursor, force=True, status='paused'); return
            progress.update(cursor, force=True)
            result = evaluate(model, val, device, config['workers'], amp, progress)
            log(dict(kind='validation', epoch=epoch, step=step, **result))
            progress.message(f'Epoch {epoch+1}: val BCE={result["loss"]:.6f}, macro F1={result["macro_f1"]:.4f}; F1={result["f1_per_class"]}; AP={result["average_precision_per_class"]}; real no-OZZ FPR={result["real_no_ozz_false_positive_rate"]}; window FPR={result["real_no_ozz_window_false_positive_rate"]}')
            epoch += 1; cursor = 0
            safe_cursor, safe_rng = cursor, rng_state()
            improved = result['loss'] < best
            best = min(best, result['loss'])
            improved_f1 = result['macro_f1'] > best_f1
            best_f1 = max(best_f1, result['macro_f1'])
            save()
            if improved_f1: save(output/'best_f1_model.pt')
            if improved: save(output/'best_model.pt')
            preserved_rng = rng_state()
            export_curves(output)
            restore_rng(preserved_rng)
            if stop[0] or (output/'STOP').exists(): return
        progress.update(len(val), force=True, status='completed')
    except Exception as exc:
        if optimizer_in_progress:
            # AdamW may already have changed some tensors. Never overwrite a
            # good checkpoint with an incomplete optimizer update.
            progress.message('Ошибка внутри AdamW: сохранённый checkpoint оставлен без изменений.')
            log(dict(kind='optimizer_error', epoch=epoch, cursor=cursor, step=step, error=str(exc)))
            progress.update(cursor, force=True, status='paused_optimizer_error')
            return
        # Parameters/AdamW still belong to the last complete optimizer step.
        # Discard partial accumulated gradients and replay its windows on resume.
        opt.zero_grad(set_to_none=True)
        cursor = safe_cursor
        restore_rng(safe_rng)
        save()
        kind = 'data_error' if isinstance(exc, DataLoadingFailure) else 'training_error'
        log(dict(kind=kind, epoch=epoch, cursor=cursor, step=step, error=repr(exc)))
        progress.message(f'{kind}: {exc}\nСохранён шаг {step}, cursor={cursor}. Подробности: training_log.jsonl')
        progress.update(cursor, force=True, status='paused_'+kind)
        return
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)
