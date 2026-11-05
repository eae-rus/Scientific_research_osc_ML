"""Independent deep Phase 5 SSL runner. See PHASE_5_HIERARCHICAL_RUNBOOK.md."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig, PhasorHierarchicalKAN, vector_loss
from osc_tools.ml.phase5_ozz_regularization import OZZRegularizedBackbone
from osc_tools.ml import phase5_ozz_regularization as regularization_impl
from osc_tools.ml.phase5_hierarchical_data import DATA_CONTRACT, HierarchicalDataset, collate_windows
from scripts.phase5_experiments.run_phase5_pretrain import _load_splits
from scripts.phase5_experiments.progress import TrainingProgress, export_hierarchical_curves
from scripts.phase5_experiments.runtime import (activation_policy, scale_gradients, effective_batches,
    fit_execution_to_memory, backward_with_oom_retry, TrainingMemoryError, timed_batches)


def model_data_hash():
    return hashlib.sha256(b"".join((ROOT/path).read_bytes() for path in (
        "osc_tools/ml/models/phasor_hierarchical.py","osc_tools/ml/phase5_hierarchical_data.py",
        "osc_tools/ml/phase5_fourier.py"))).hexdigest()


def regularization_contract(rate):
    if not 0 <= rate < 1:
        raise ValueError('drop_path_rate must be in [0, 1)')
    return dict(kind='stage_linear_whole_block_v1', drop_path_rate=rate,
                implementation_sha256=hashlib.sha256(
                    Path(regularization_impl.__file__).read_bytes()).hexdigest())


def assert_resume_contract(saved,current,*,allow_batch_change=False):
    """Allow the verified console-only migration, never arbitrary hash bypass.

    New UI/manual settings no longer invalidate the numerical training recipe.
    Model/data bytes and all numerical settings still have to match exactly.
    """
    if saved==current: return False
    # Explicit state-compatible FP32 stem migration (not a generic hash bypass).
    if (current.get('model_data_sha256') == 'e4c68c096a4d1f446692ae36ac5dde90cbddbe47eca232a3041371035a6e781a'
            and saved.get('model_data_sha256') in (
                '9ba94e2e2c68fc6d9593eb69161bf6b543135a65b2a6824336326e6b5a017e8a',
                '8858f68cfeacd482ea92fc1dc8ddfa79930141bbca29c461426b997e57789776')):
        migrated = dict(saved, model_data_sha256=current['model_data_sha256'])
        assert_resume_contract(migrated, current, allow_batch_change=allow_batch_change)
        return True

    if allow_batch_change and saved.get('target_tokens') != current.get('target_tokens'):
        if not isinstance(current.get('target_tokens'), int) or current['target_tokens'] <= 0:
            raise ValueError('target_tokens must be a positive integer')
        migrated = dict(saved, target_tokens=current['target_tokens'])
        assert_resume_contract(migrated, current)
        return True
    # Verified bit-identical batched Fourier implementation, not a generic hash bypass.
    if (saved.get('model_data_sha256') == '8858f68cfeacd482ea92fc1dc8ddfa79930141bbca29c461426b997e57789776'
            and current.get('model_data_sha256') == '9ba94e2e2c68fc6d9593eb69161bf6b543135a65b2a6824336326e6b5a017e8a'):
        migrated = dict(saved, model_data_sha256=current['model_data_sha256'])
        assert_resume_contract(migrated, current)
        return True
    # Explicit supported transition: old SSL -> stochastic depth, without
    # relaxing any architecture/data/optimizer constraints. Subsequent resumes
    # must keep the recorded rate and implementation unchanged.
    if 'regularization' not in saved and 'regularization' in current:
        reg = current['regularization']
        if reg != regularization_contract(reg['drop_path_rate']):
            raise ValueError('Unknown stochastic-depth implementation')
        previous_recipe = dict(current)
        previous_recipe.pop('regularization')
        assert_resume_contract(saved, previous_recipe)
        return True
    legacy_hash="0341f4e14045b0d6c4089a89755635864799413eff5f443b41f9a7d15e43cf39"
    verified_data="8858f68cfeacd482ea92fc1dc8ddfa79930141bbca29c461426b997e57789776"
    migrated=dict(saved)
    old_hash=migrated.pop("implementation_sha256",None)
    if old_hash==legacy_hash and current.get("model_data_sha256")==verified_data:
        migrated.update(model_data_sha256=verified_data,training_recipe="hierarchical_ssl_v1")
        if migrated==current: return True
    raise ValueError("Resume contract mismatch: model/data/numerical settings differ; use a new run")


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(state,path):
    temp = path.with_suffix(".tmp")
    torch.save(state,temp)
    temp.replace(path)


def worker_init(_worker_id):
    # The main process owns graceful Ctrl+C; workers must finish their reads.
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def loader(dataset,start,workers,seed):
    # Dedicated generator avoids consuming model/dropout RNG when recreating
    # the DataLoader iterator on resume. Cursor tracks consumed, not prefetched.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=1,sampler=range(start,len(dataset)),num_workers=workers,
                      collate_fn=collate_windows,generator=generator,worker_init_fn=worker_init,
                      pin_memory=torch.cuda.is_available())


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def losses(model,batch):
    out = model(batch["z"],batch["state"],batch["times"],batch["valid"])
    total,count = vector_loss(out["prediction"],batch["target"],batch["selected"])
    local,_ = vector_loss(out["local_prediction"],batch["target"],batch["selected"])
    return total+.25*local,count


@torch.no_grad()
def evaluate(model,dataset,device,amp,progress=None):
    model.eval()
    stats = {}
    if progress: progress.begin("Validation",len(dataset))
    for index,batch in enumerate(loader(dataset,0,0,1234),1):
        batch = to_device(batch,device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
            total,count = losses(model,batch)
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite validation loss")
        meta = batch["metadata"][0]
        for key in ("all",meta["source"],f"blocks_{meta['blocks']}"):
            acc = stats.setdefault(key,[0.,0])
            acc[0] += total.item(); acc[1] += count.item()
        if progress: progress.update(index,force=index==len(dataset),status="validation")
    model.train()
    return {key:value[0]/max(1,value[1]) for key,value in stats.items()}


def synthetic_batch(tokens,device):
    z = torch.randn(1,tokens,13,6,2,device=device)*.1
    selected = torch.rand(1,tokens,13,6,device=device)<.3
    state = torch.ones_like(selected,dtype=torch.long)
    state[selected] = 3
    return {"z":z.masked_fill(selected.unsqueeze(-1),0),"target":z,"selected":selected,
            "state":state,"times":torch.arange(1,tokens+1,device=device)[None,:]/8,
            "valid":torch.ones(1,tokens,device=device,dtype=torch.bool)}


def profile(args,model,device,amp):
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.train()
    records = []
    for tokens in args.profile_tokens:
        opt.zero_grad(set_to_none=True)
        if device.type=="cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.perf_counter()
        batch = synthetic_batch(tokens,device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
            total,count = losses(model,batch)
            loss = total/count.clamp_min(1)
        loss.backward()
        finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        if not finite:
            raise RuntimeError("Nonfinite profile gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        opt.step()
        if device.type=="cuda": torch.cuda.synchronize()
        row = {"tokens":tokens,"seconds":time.perf_counter()-started,"loss":loss.item(),
               "finite_gradients":finite,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0,
               "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20 if device.type=="cuda" else 0}
        records.append(row); print(json.dumps(row),flush=True)
        del batch,total,count,loss
    destination = ROOT/"reports/phase5/hierarchical_v1_profile.json"
    destination.write_text(json.dumps({"model":asdict(model.cfg),"parameters":sum(p.numel() for p in model.parameters()),
                           "gpu":torch.cuda.get_device_name() if device.type=="cuda" else "cpu","amp_bf16":amp,
                           "torch":torch.__version__,"measurements":records},indent=2),encoding="utf-8")


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    amp = device.type=="cuda" and torch.cuda.is_bf16_supported() and not args.fp32
    cfg = HierarchicalConfig()
    drop_path_rate = getattr(args, 'drop_path_rate', 0.1)
    regularization = regularization_contract(drop_path_rate)
    model = OZZRegularizedBackbone(cfg, drop_path_rate).to(device)
    if args.profile:
        profile(args,model,device,amp); return
    output = args.output.resolve()
    latest = output/"latest_checkpoint.pt"
    if (output/"STOP").exists(): raise ValueError("Remove the STOP file before starting/resuming")
    if args.resume and not latest.exists(): raise FileNotFoundError(latest)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty: use --resume or a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    registry = ROOT/"data/phase5/datasets_registry.json"
    manifest = _load_splits(ROOT/"data/phase5/research_strict_splits.json")
    contract = {"architecture":asdict(cfg),"data_contract":DATA_CONTRACT,"split_sha256":manifest["sha256"],
                "model_data_sha256":model_data_hash(),"training_recipe":"hierarchical_ssl_v1",
                "registry_sha256":hashlib.sha256(registry.read_bytes()).hexdigest(),"seed":args.seed,
                "samples_per_epoch":args.samples_per_epoch,"validation_samples":args.validation_samples,
                "target_tokens":args.target_tokens,"max_seconds":args.max_seconds,"lr":args.lr,
                "amp_bf16":amp,"loss":"radial_vector_huber_0.1_local_0.25","protocol":"research_strict",
                "regularization":regularization}
    train = HierarchicalDataset(registry,manifest,samples=args.samples_per_epoch,seed=args.seed,max_seconds=args.max_seconds)
    val = HierarchicalDataset(registry,manifest,split="validation",samples=args.validation_samples,seed=args.seed+100000,
                              max_seconds=args.max_seconds,augment=False)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    def multiplier(step):
        return min((step+1)/200,1)*(.05+.95*(1+math.cos(math.pi*(max(step-200,0)%4000)/4000))/2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    scaler = torch.amp.GradScaler("cuda",enabled=False) # BF16 needs no dynamic scaling; state retained.
    epoch,cursor,step,best = 0,0,0,float("inf")
    saved_execution = None
    regularization_history = []
    batch_history = []
    if args.resume:
        state = torch.load(latest,map_location="cpu",weights_only=False) # own trusted checkpoint only
        migrated=assert_resume_contract(state["contract"],contract,
                                        allow_batch_change=getattr(args,'allow_batch_change',False))
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        epoch,cursor,step,best = state["epoch"],state["cursor"],state["step"],state["best"]
        restore_rng(state["rng"])
        saved_execution = state.get("execution")
        regularization_history = state.get('regularization_history', [])
        batch_history = state.get('batch_history', [])
        if state['contract']['target_tokens'] != args.target_tokens:
            batch_history.append(dict(epoch=epoch,cursor=cursor,step=step,
                                      previous_target_tokens=state['contract']['target_tokens'],
                                      target_tokens=args.target_tokens))
        if 'regularization' not in state['contract']:
            regularization_history.append(dict(epoch=epoch, cursor=cursor, step=step,
                                               previous_drop_path_rate=0., regularization=regularization))
        if migrated: print("Совместимый переход рецепта: веса, optimizer, scheduler и RNG восстановлены; новый режим записан в checkpoint.",flush=True)
    (output/"config.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
    requested = dict(max_windows=getattr(args,"max_windows_per_forward",8),
                     max_padded_tokens=getattr(args,"max_padded_tokens",4000),
                     no_checkpoint_max_tokens=getattr(args,"no_checkpoint_max_tokens",4000),
                     reserve_mib=getattr(args,"gpu_reserve_mib",1024))
    policy = fit_execution_to_memory(**requested,device=device)
    if saved_execution and saved_execution.get("requested") == requested:
        # Retain an OOM fallback across save/resume. A smaller available budget
        # may lower limits further, but never silently undo a saved fallback.
        for key in policy:
            policy[key] = min(policy[key], saved_execution["effective"][key])
    if getattr(args,'pack_by_length',False):
        policy['sort_by_length'] = True
    execution = {"version":"packed_v1","requested":requested,"effective":policy,
                 "workers":args.workers,"torch":torch.__version__,"cpu_threads":torch.get_num_threads()}
    def record_execution():
        (output/"execution.json").write_text(json.dumps(execution,indent=2),encoding="utf-8")
    record_execution()
    stop = [False]
    progress=TrainingProgress(output,interval=getattr(args,"console_interval",2.))
    progress.message(f"PhasorHierarchicalKAN | {device} | {'BF16' if amp else 'FP32'} | "
                     f"эпоха {epoch+1}/{args.epochs}, окно {cursor:,}/{len(train):,}, шаг {step:,}\n"
                     f"Результаты: {output}\n"
                     f"До {policy['max_windows']} окон за проход; накопление {args.target_tokens} токенов; workers={args.workers}. "
                     "Ctrl+C: сохранить и остановиться. Подробности — training_log.jsonl.")
    progress.message(f"Предел упаковки: {policy['max_padded_tokens']} токенов с padding; "
                     f"без пересчёта: {policy['no_checkpoint_max_tokens']}; запас VRAM: {requested['reserve_mib']} МиБ.")
    progress.message(f"Stochastic depth: до {drop_path_rate:.0%} внутри каждой стадии; validation — вся глубина.")
    if args.resume and (not saved_execution or saved_execution.get("effective") != policy):
        progress.message("Режим исполнения изменён: границы шагов сохранены, но упаковка может изменить dropout/округления.")
    def request_stop(signum,frame):
        stop[0]=True
        progress.message("Остановка запрошена: завершаю накопление и сохраняю latest.")
    signal.signal(signal.SIGINT,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    last_saved = [epoch,cursor,step]
    def save(path=latest):
        atomic_save({"contract":contract,"model":model.state_dict(),"optimizer":opt.state_dict(),
                     "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),
                     "epoch":epoch,"cursor":cursor,"step":step,"best":best,"execution":execution,
                     "regularization_history":regularization_history,"batch_history":batch_history},path)
        if path == latest: last_saved[:] = [epoch,cursor,step]
        record_execution()
    def log(row):
        with (output/"training_log.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        if row['kind']=='train':
            progress.update(row['cursor'],{k:row[k] for k in ('loss','lr','grad_norm','step','gpu_peak_allocated_mib','gpu_reserved_mib','gpu_free_mib') if k in row})
        else:
            metrics=row['metrics']
            progress.message(f"Эпоха {row['epoch']+1}: val={metrics['all']:.6f} | "
                             f"Open_EE={metrics.get('open_ee',float('nan')):.6f} | "
                             f"French={metrics.get('french_rte',float('nan')):.6f} | "
                             f"лучшее={min(best,metrics['all']):.6f}"+
                             ("  [новое лучшее]" if metrics['all']<best else ""))
    model.train()
    if epoch>=args.epochs:
        progress.message("Заданный предел эпох уже достигнут. Для продолжения увеличьте EPOCHS.")
        return
    # A valid rollback checkpoint exists even if the first AdamW update runs out of memory.
    if not latest.exists(): save()
    while epoch<args.epochs:
        progress.begin(f"Эпоха {epoch+1}/{args.epochs}",len(train),cursor)
        train.set_epoch(epoch)
        started = time.perf_counter()
        batches = effective_batches(loader(train,cursor,args.workers,args.seed+epoch),args.target_tokens)
        for windows, data_wait_seconds in timed_batches(batches):
            count_acc = sum(int(np.count_nonzero(w["selected"].numpy())) for w in windows)
            tokens_acc = sum(int(np.count_nonzero(w["valid"].numpy())) for w in windows)
            if count_acc == 0: raise RuntimeError(f"No targets at epoch={epoch}, cursor={cursor}")
            before_batch_rng = rng_state()
            failed = False
            try:
                result = backward_with_oom_retry(model,windows,device,amp,policy,losses,
                                                 rng_state,restore_rng,progress.message)
                scale_gradients(model,4096/count_acc)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            except (TrainingMemoryError, torch.OutOfMemoryError):
                failed = True
            if failed:
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda": torch.cuda.empty_cache()
                restore_rng(before_batch_rng)
                save()  # Current weights are valid; cursor still points before this batch.
                progress.update(cursor,{"step":step},force=True,status="paused")
                progress.message(f"Не хватает VRAM даже после уменьшения нагрузки. Сохранена позиция перед шагом: {latest}")
                return
            # AdamW can modify parameters before an OOM. Never overwrite latest
            # with such a partially updated state; roll back to the on-disk snapshot.
            optimizer_failed = False
            try:
                opt.step()
            except torch.OutOfMemoryError:
                optimizer_failed = True
            if optimizer_failed:
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda": torch.cuda.empty_cache()
                progress.begin(f"Эпоха {last_saved[0]+1}/{args.epochs}",len(train),last_saved[1])
                progress.update(last_saved[1],{"step":last_saved[2]},force=True,status="paused")
                progress.message(f"OOM внутри AdamW. Частичный шаг не сохранён; продолжать с checkpoint step={last_saved[2]}: {latest}")
                return
            scheduler.step(); step += 1; cursor += len(windows)
            opt.zero_grad(set_to_none=True)
            row = {"kind":"train","epoch":epoch,"cursor":cursor,"step":step,"loss":result['loss_sum']/count_acc,
                   "tokens":tokens_acc,"targets":count_acc,"grad_norm":float(grad),"lr":opt.param_groups[0]["lr"],
                   "seconds":time.perf_counter()-started,"windows":[w['metadata'][0] for w in windows],
                   "packs":result['packs'],"padded_tokens":result['padded_tokens'],"oom_retries":result['oom_retries'],
                   "data_wait_seconds":data_wait_seconds}
            if device.type == "cuda":
                row.update(gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                           gpu_reserved_mib=torch.cuda.memory_reserved()/2**20,
                           gpu_free_mib=torch.cuda.mem_get_info()[0]/2**20)
            log(row); started=time.perf_counter()
            if step%args.save_every==0: save()
            if stop[0] or (output/"STOP").exists() or (args.max_steps and step>=args.max_steps):
                save()
                progress.update(cursor,{"step":step},force=True,status="paused")
                progress.message(f"Сохранено: {latest}\nОстановлено: эпоха {epoch+1}, окно {cursor}, шаг {step}.")
                return
        # Validation is fixed across epochs; preserve training RNG around it.
        progress.update(cursor,{"step":step},force=True)
        saved_rng = rng_state()
        validation_failed = False
        try:
            metrics = evaluate(model,val,device,amp,progress)
        except torch.OutOfMemoryError:
            validation_failed = True
        restore_rng(saved_rng)
        if validation_failed:
            if device.type == "cuda": torch.cuda.empty_cache()
            save()
            progress.begin(f"Эпоха {epoch+1}/{args.epochs}",len(train),cursor)
            progress.update(cursor,{"step":step},force=True,status="paused")
            progress.message("Не хватает VRAM на validation. Обученные веса сохранены; продолжение повторит validation.")
            return
        log({"kind":"validation","epoch":epoch,"step":step,"metrics":metrics})
        improved = metrics["all"]<best
        best = min(best,metrics["all"])
        epoch += 1; cursor=0
        save()
        if improved: save(output/"best_model.pt")
        presentation_rng=rng_state()
        export_hierarchical_curves(output)
        restore_rng(presentation_rng)
        if stop[0] or (output/"STOP").exists():
            progress.update(len(val),{"step":step},force=True,status="paused")
            progress.message(f"Сохранено: {latest}; остановлено после validation.")
            return
    progress.update(len(val),{"step":step},force=True,status="completed")
    progress.message("Обучение завершено.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/phase5/phasor_hierarchical_v1")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--samples-per-epoch",type=int,default=20000)
    parser.add_argument("--validation-samples",type=int,default=64)
    parser.add_argument("--target-tokens",type=int,default=640)
    parser.add_argument("--allow-batch-change",action="store_true",
                        help="Explicitly allow changing target_tokens on resume; keep AdamW/scheduler state")
    parser.add_argument("--max-seconds",type=float,default=30)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--drop-path-rate",type=float,default=0.1,
                        help="Maximum whole residual block skip probability during training")
    parser.add_argument("--max-windows-per-forward",type=int,default=8)
    parser.add_argument("--pack-by-length",action="store_true",
                        help="Group windows by length within each effective batch, retaining its membership")
    parser.add_argument("--max-padded-tokens",type=int,default=4000)
    parser.add_argument("--gpu-reserve-mib",type=int,default=1024)
    parser.add_argument("--no-checkpoint-max-tokens",type=int,default=4000,
                        help="Keep activations up to this length; 0 restores original recomputation")
    parser.add_argument("--save-every",type=int,default=500)
    parser.add_argument("--max-steps",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--fp32",action="store_true")
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--profile-tokens",type=int,nargs="+",default=[80,800,12000])
    parser.add_argument("--console-interval",type=float,default=2.)
    return parser


def main():
    parser=build_parser()
    args = parser.parse_args()
    if min(args.epochs,args.samples_per_epoch,args.validation_samples,args.target_tokens,args.save_every)<=0 or args.max_seconds<.2:
        parser.error("Counts must be positive and max-seconds >= 0.2")
    run(args)


if __name__=="__main__":
    main()
