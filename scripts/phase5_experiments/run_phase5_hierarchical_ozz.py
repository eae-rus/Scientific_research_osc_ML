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
from osc_tools.ml.models.phasor_hierarchical_v2 import HierarchicalV2Config, PhasorHierarchicalKANV2
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
    """Four independent sigmoid logits; generation 2 also exposes a supervised local head."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.cfg.width, 4)
        for name in ('head','local_head'):
            if hasattr(backbone,name): getattr(backbone,name).requires_grad_(False)

    def forward(self, batch, return_aux=False, use_global=True):
        if self.backbone.cfg.architecture_id == 'phasor_hierarchical_v2':
            result=self.backbone(batch['z'],batch['state'],batch['times'],batch['valid'],use_global=use_global)
            logits=self.classifier(result['token_features'])
            return dict(logits=logits,local_logits=self.classifier(result['local_features'])) if return_aux else logits
        features = []
        handle = self.backbone.head.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            self.backbone(batch['z'], batch['state'], batch['times'], batch['valid'])
        finally:
            handle.remove()
        b, t = batch['valid'].shape
        logits=self.classifier(features[0]).reshape(b, -1, 4)[:, :t]
        return dict(logits=logits,local_logits=logits) if return_aux else logits


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
    local_counts=torch.zeros_like(counts)
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
                predictions=model(batch,return_aux=True)
                logits=predictions['logits']
                local=predictions['local_logits'][mask].float()
            logits, target = logits[mask].float(), batch['target'][mask]
            total += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum').item()
            count += target.numel()
            all_logits.append(logits.cpu().numpy()); all_targets.append(target.cpu().numpy())
            pred, truth = logits >= 0, target > .5
            lp=local>=0
            local_counts+=torch.stack(((lp&truth).sum(0),(lp&~truth).sum(0),(~lp&truth).sum(0),(~lp&~truth).sum(0)))
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
    local_metrics=metrics_from_counts(*local_counts.cpu())
    return {'loss': total/max(count, 1), 'local_macro_f1':local_metrics['macro_f1'],
            'local_f1_per_class':local_metrics['f1_per_class'], **metrics_from_counts(*counts.cpu()),
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


def validate_resume_contract(saved, contract):
    if saved != contract:
        raise ValueError('OZZ contract changed. Start generation 2 from scratch in a new output directory.')
    return False


def learning_rate_factor(config, epoch_progress):
    """One warmup and one cosine decay, determined by the saved training clock."""
    warmup = config['lr_warmup_epochs']
    if epoch_progress < warmup:
        t = max(0., epoch_progress/warmup)
        return config['lr_initial_ratio']+(1-config['lr_initial_ratio'])*t
    t = min(1., max(0., (epoch_progress-warmup)/(config['lr_total_epochs']-warmup)))
    return config['lr_min_ratio']+(1-config['lr_min_ratio'])*.5*(1+math.cos(math.pi*t))


def run(config):
    config = dict(config)
    config.setdefault('lr_warmup_epochs', 10.)
    config.setdefault('lr_total_epochs', float(config['epochs']))
    config.setdefault('lr_initial_ratio', .05)
    config.setdefault('lr_min_ratio', .05)
    if not (0 < config['lr_warmup_epochs'] < config['lr_total_epochs']
            and 0 < config['lr_initial_ratio'] <= 1 and 0 < config['lr_min_ratio'] <= 1):
        raise ValueError('Invalid warmup/cosine settings')
    config.setdefault('feature_mode', 'hierarchical_v2')
    config.setdefault('label_tail_periods', 1)
    config.setdefault('synthetic_context', True)
    config.setdefault('synthetic_normal_probability', .15)
    if not 0 <= config['synthetic_normal_probability'] <= 1: raise ValueError('Invalid synthetic normal probability')
    config.setdefault('local_loss_weight', .25)
    if not 0 <= config['local_loss_weight'] <= 1: raise ValueError('Invalid local loss weight')
    config.setdefault('voltage_dropout_phase', .03)
    config.setdefault('background_dropout_multiplier', 1.4)
    if config['feature_mode'] != 'hierarchical_v2' or config['label_tail_periods'] != 1:
        raise ValueError('Current OZZ recipe requires continuous history and a one-period label tail')
    config.setdefault('validation_current_samples', 0)
    if config['validation_current_samples'] < 0:
        raise ValueError('validation_current_samples must be nonnegative')
    config.setdefault('current_dropout_all', 0.)
    config.setdefault('current_dropout_phase', 0.)
    config.setdefault('current_zero_probability', .5)
    multiplier = config['background_dropout_multiplier']
    if not (multiplier >= 1 and 0 <= config['voltage_dropout_phase']*multiplier <= 1
            and (config['current_dropout_all']+config['current_dropout_phase'])*multiplier <= 1):
        raise ValueError('Invalid background/voltage dropout probabilities')
    from osc_tools.ml.phase5_ozz_input_augmentation import augment_current_channels
    augment_current_channels(np.ones((1, 8)), None, np.random.default_rng(0),
        config['current_dropout_all'], config['current_dropout_phase'], config['current_zero_probability'])
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
    state = torch.load(latest,map_location='cpu',weights_only=False) if resume else {}
    architecture = config.pop('architecture', asdict(HierarchicalV2Config(drop_path_rate=config['drop_path_rate'])))
    backbone = PhasorHierarchicalKANV2(HierarchicalV2Config(**architecture))
    model = OZZClassifier(backbone).to(device)
    manifest_path = ROOT/'data/phase5/research_strict_splits.json'
    records, manifest = prepare_records(ROOT, config, json.loads(manifest_path.read_text(encoding='utf-8')))
    contract = {k: v for k, v in config.items() if k not in ('resume', 'epochs', 'workers', 'save_every', 'console_interval', 'ssl_checkpoint')}
    contract.update(architecture=architecture, recipe='hierarchical_ozz_v2', amp_bf16=amp,
                    split_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                    ssl_split_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    code_paths = [Path(__file__), ROOT/'osc_tools/ml/phase5_hierarchical_ozz.py',
                  ROOT/'osc_tools/ml/phase5_hierarchical_data.py', ROOT/'osc_tools/ml/models/phasor_hierarchical.py',
                  ROOT/'osc_tools/ml/phase5_ozz_regularization.py', ROOT/'osc_tools/ml/phase5_fourier.py', ROOT/'osc_tools/ml/phase5_ozz_balance.py', ROOT/'osc_tools/ml/phase5_ozz_input_augmentation.py', ROOT/'osc_tools/ml/phase5_ozz_features.py', ROOT/'osc_tools/ml/phase5_hierarchical_v2_features.py',
                  ROOT/'osc_tools/ml/phase5_ozz_synthesis.py', ROOT/'osc_tools/ml/models/phasor_hierarchical_v2.py']
    contract['implementation_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in code_paths)).hexdigest()
    if resume:
        validate_resume_contract(state['contract'], contract)
    output.mkdir(parents=True, exist_ok=True)
    (output/'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance = state.get('initialization',dict(kind='random',seed=config['seed']))
    def dataset(items,samples,seed,train,current_evaluation=None):
        return OZZDataset(items,samples,seed,config['max_seconds'],train,
            event_windows=config['balanced_events'],background_repeats=config['background_repeats'],
            current_dropout_all=config['current_dropout_all'],current_dropout_phase=config['current_dropout_phase'],
            current_zero_probability=config['current_zero_probability'],feature_mode=config['feature_mode'],label_tail_periods=1,
            voltage_dropout_phase=config['voltage_dropout_phase'],background_dropout_multiplier=config['background_dropout_multiplier'],
            synthetic_context=config['synthetic_context'],current_evaluation=current_evaluation,
            synthetic_normal_probability=config['synthetic_normal_probability'])
    train=dataset(records['train'],config['samples_per_epoch'],config['seed'],True)
    val=dataset(records['validation'],config['validation_samples'],config['seed']+100000,False)
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
        reference = dataset(records['train'],config['samples_per_epoch'],config['seed'],True)
        balance = estimate_balance(reference, config['balance_samples'], config['positive_weight_cap'])
    if balance is not None:
        print(f'Class balance: {balance}', flush=True)
        (output/'class_balance.json').write_text(json.dumps(balance, indent=2), encoding='utf-8')
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
                         initialization=provenance), path)
    def log(row):
        with (output/'training_log.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
    progress.message(f'OZZ Phase 5 | {device} | random initialization | {output}\n'
                     '4 типа ОЗЗ; все четыре выхода 0 = не ОЗЗ. Ctrl+C или STOP: сохранить и остановиться.')
    safe_cursor, safe_rng = cursor, rng_state()
    optimizer_in_progress = False
    reported = set()
    try:
        if epoch >= config['epochs']:
            progress.message('Все заданные эпохи уже завершены. Увеличьте EPOCHS для продолжения.')
            return
        if not latest.exists():
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
                        prediction = model(batch,return_aux=True)
                        logits=prediction['logits']
                        target = smooth_binary_targets(batch['target'][mask], config['label_error_probability'])
                        if balance is not None:
                            loss = balanced_window_loss(logits[mask].float(), batch['target'][mask],
                                                        config['label_error_probability'], balance, config['focal_gamma'],
                                                        transition_weights(batch['target'], mask, config['boundary_weight'])[mask])
                            local_loss = balanced_window_loss(prediction['local_logits'][mask].float(),batch['target'][mask],
                                config['label_error_probability'],balance,config['focal_gamma'],
                                transition_weights(batch['target'],mask,config['boundary_weight'])[mask])
                            loss=(loss+config['local_loss_weight']*local_loss)/(1+config['local_loss_weight'])
                        else:
                            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask].float(), target, reduction='sum')
                            local_loss=nn.functional.binary_cross_entropy_with_logits(prediction['local_logits'][mask].float(),target,reduction='sum')
                            loss=(loss+config['local_loss_weight']*local_loss)/(1+config['local_loss_weight'])
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
            progress.message(f'Epoch {epoch+1}: val BCE={result["loss"]:.6f}, macro F1={result["macro_f1"]:.4f}; F1={result["f1_per_class"]}; AP={result["average_precision_per_class"]}; local F1={result['local_macro_f1']:.4f}; real no-OZZ FPR={result["real_no_ozz_false_positive_rate"]}; window FPR={result["real_no_ozz_window_false_positive_rate"]}')
            if config['validation_current_samples']:
                for variant in ('missing', 'zero'):
                    probe = dataset(records['validation'],config['validation_current_samples'],config['seed']+100000,False,variant)
                    robustness = evaluate(model, probe, device, config['workers'], amp, progress)
                    log(dict(kind='validation_current_'+variant, epoch=epoch, step=step, **robustness))
                    progress.message(f'Without currents ({variant}): F1={robustness["f1_per_class"]}; AP={robustness["average_precision_per_class"]}')
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
        progress.update(progress.total, force=True, status='completed')
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
