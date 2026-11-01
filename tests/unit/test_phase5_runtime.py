"""Window packing and transactional OOM recovery for hierarchical SSL."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from scripts.phase5_experiments import runtime
from scripts.phase5_experiments.run_phase5_hierarchical import synthetic_batch, rng_state, restore_rng, losses
from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig, PhasorHierarchicalKAN


def window(tokens, index=0):
    batch = synthetic_batch(tokens, torch.device('cpu'))
    batch['metadata'] = [dict(serial=index, source='open_ee', blocks=1)]
    return batch


def test_packing_preserves_effective_boundaries_order_and_masks():
    windows = [window(n, i) for i, n in enumerate((80,160,400,80,80,12000,80))]
    groups = list(runtime.effective_batches(windows, 640))
    assert [len(g) for g in groups] == [3,3,1]
    packs = [p for g in groups for p in runtime.window_packs(g,8,4000)]
    assert [w['metadata'][0]['serial'] for p in packs for w in p] == list(range(7))
    assert all(len(p)==1 or len(p)*max(w['z'].shape[1] for w in p)<=4000 for p in packs)
    result = runtime.pack_windows(groups[0])
    assert result['z'].shape == (3,400,13,6,2)
    for i, w in enumerate(groups[0]):
        n = w['z'].shape[1]
        torch.testing.assert_close(result['z'][i,:n],w['z'][0],rtol=0,atol=0)
        assert not result['valid'][i,n:].any() and not result['selected'][i,n:].any()
    assert result['selected'].sum() == sum(w['selected'].sum() for w in groups[0])


def test_packed_predictions_and_weighted_gradients_match_without_dropout():
    cfg = replace(HierarchicalConfig(),channel_width=8,width=16,channel_layers=1,
                  local_layers=1,global_layers=1,fusion_layers=1,block_tokens=4,dropout=0)
    model = PhasorHierarchicalKAN(cfg).eval()
    windows = [window(7),window(12)]
    ref = [model(w['z'],w['state'],w['times'],w['valid'])['prediction'].detach() for w in windows]
    reference_loss = sum(losses(model,w)[0] for w in windows)
    reference_loss.backward()
    grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    batch = runtime.pack_windows(windows)
    actual = model(batch['z'],batch['state'],batch['times'],batch['valid'])['prediction']
    for i,w in enumerate(windows):
        torch.testing.assert_close(ref[i][0],actual[i,:w['z'].shape[1]],rtol=1e-5,atol=1e-6)
    packed_loss = losses(model,batch)[0]
    torch.testing.assert_close(reference_loss,packed_loss,rtol=1e-6,atol=1e-5)
    packed_loss.backward()
    for n,p in model.named_parameters():
        if n in grads: torch.testing.assert_close(grads[n],p.grad,rtol=1e-4,atol=2e-5)
    # Changing the other example cannot affect the first example's output.
    batch['z'][1] *= 100
    changed = model(batch['z'],batch['state'],batch['times'],batch['valid'])['prediction']
    torch.testing.assert_close(actual[0],changed[0],rtol=0,atol=0)


def test_oom_retry_discards_partial_gradients_and_rewinds_rng(monkeypatch):
    model=torch.nn.Linear(2,1); seen=[]
    def fake(*args):
        seen.append(torch.get_rng_state())
        torch.rand(11)
        for p in model.parameters(): p.grad=torch.ones_like(p)*(10 if len(seen)==1 else 1)
        if len(seen)==1: raise torch.OutOfMemoryError('injected after partial backward')
        return dict(loss_sum=2.,packs=1,padded_tokens=80)
    monkeypatch.setattr(runtime,'_backward_packs',fake)
    policy=dict(max_windows=8,max_padded_tokens=4000,no_checkpoint_max_tokens=4000)
    result=runtime.backward_with_oom_retry(model,[],torch.device('cpu'),False,policy,None,rng_state,restore_rng)
    assert result['oom_retries']==1 and policy['no_checkpoint_max_tokens']==0
    assert torch.equal(seen[0],seen[1])
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())


def test_terminal_oom_restores_prebatch_state(monkeypatch):
    model=torch.nn.Linear(2,1); before=rng_state()
    weights={n:p.clone() for n,p in model.named_parameters()}
    def fail(*args):
        torch.rand(9)
        for p in model.parameters(): p.grad=torch.ones_like(p)
        raise torch.OutOfMemoryError('injected')
    monkeypatch.setattr(runtime,'_backward_packs',fail)
    policy=dict(max_windows=4,max_padded_tokens=4000,no_checkpoint_max_tokens=4000)
    with pytest.raises(runtime.TrainingMemoryError):
        runtime.backward_with_oom_retry(model,[],torch.device('cpu'),False,policy,None,rng_state,restore_rng)
    assert policy['max_windows']==1 and policy['no_checkpoint_max_tokens']==0
    assert torch.equal(before['torch'],torch.get_rng_state())
    for name,p in model.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(weights[name],p,rtol=0,atol=0)


@pytest.mark.parametrize('where',['backward','optimizer','validation'])
def test_runner_preserves_valid_checkpoint_on_memory_failure(tmp_path,monkeypatch,where):
    from scripts.phase5_experiments import run_phase5_hierarchical as engine
    cfg=replace(HierarchicalConfig(),channel_width=8,width=16,channel_layers=1,
                local_layers=1,global_layers=1,fusion_layers=1,dropout=0)
    monkeypatch.setattr(engine,'HierarchicalConfig',lambda:cfg)
    monkeypatch.setattr(engine,'ROOT',tmp_path)
    monkeypatch.setattr(engine,'model_data_hash',lambda:'fixture')
    monkeypatch.setattr(engine,'_load_splits',lambda p:{'sha256':'fixture'})
    monkeypatch.setattr(engine.signal,'signal',lambda *a:None)
    directory=tmp_path/'data/phase5';directory.mkdir(parents=True)
    (directory/'datasets_registry.json').write_text('{}')
    class Dataset:
        def __init__(self,*a,samples=2,**kw):
            self.samples=samples; self.batch=window(80)
        def __len__(self):return self.samples
        def set_epoch(self,epoch):pass
        def __getitem__(self,i):return {k:v[0] if isinstance(v,torch.Tensor) else v[0] for k,v in self.batch.items()}
    monkeypatch.setattr(engine,'HierarchicalDataset',Dataset)
    if where=='backward':
        monkeypatch.setattr(runtime,'_backward_packs',lambda *a:(_ for _ in ()).throw(torch.OutOfMemoryError('injected')))
    elif where=='optimizer':
        original=torch.optim.AdamW.step
        def broken_step(self,*a,**kw):
            original(self,*a,**kw)
            raise torch.OutOfMemoryError('injected AFTER parameter update')
        monkeypatch.setattr(torch.optim.AdamW,'step',broken_step)
    else:
        monkeypatch.setattr(engine,'evaluate',lambda *a:(_ for _ in ()).throw(torch.OutOfMemoryError('injected')))
    snapshots=[]; original_save=engine.atomic_save
    def save(state,path):
        snapshots.append((state['step'],state['cursor'],{n:p.clone() for n,p in state['model'].items()}))
        original_save(state,path)
    monkeypatch.setattr(engine,'atomic_save',save)
    args=engine.build_parser().parse_args(['--output',str(tmp_path/'run'),'--device','cpu','--workers','0',
        '--epochs','1','--samples-per-epoch','2','--validation-samples','1','--target-tokens','160'])
    engine.run(args)
    saved=torch.load(tmp_path/'run/latest_checkpoint.pt',weights_only=False)
    if where in ('backward','optimizer'):
        assert saved['step']==0 and saved['cursor']==0
        for key in saved['model']:torch.testing.assert_close(saved['model'][key],snapshots[0][2][key],rtol=0,atol=0)
    else:
        assert saved['step']==1 and saved['cursor']==2 and saved['epoch']==0
    if where=='optimizer': assert len(snapshots)==1  # Never overwrite with a partial update.


def test_memory_budget_accounts_for_reusable_cache_and_reserve(monkeypatch):
    mib=2**20
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1500*mib,8192*mib))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:1000*mib)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:500*mib)
    policy=runtime.fit_execution_to_memory(8,4000,4000,1024,torch.device('cuda'))
    # 1500 free + 500 reusable - 1024 reserve -> 976 MiB; 0.75 MiB/token.
    assert policy==dict(max_windows=8,max_padded_tokens=1280,no_checkpoint_max_tokens=1280)
    low=runtime.fit_execution_to_memory(8,4000,4000,3000,torch.device('cuda'))
    assert low['max_padded_tokens']==80 and low['no_checkpoint_max_tokens']==0


def test_cpu_execution_limits_do_not_need_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a: (_ for _ in ()).throw(AssertionError('CUDA queried')))
    assert runtime.fit_execution_to_memory(4,2000,800,1024,torch.device('cpu'))==dict(max_windows=4,max_padded_tokens=2000,no_checkpoint_max_tokens=800)
    with pytest.raises(ValueError):runtime.fit_execution_to_memory(max_windows=0)


def test_ssl_stochastic_depth_migration_keeps_other_contract_checks():
    from scripts.phase5_experiments import run_phase5_hierarchical as engine
    old = dict(model_data_sha256='same', lr=.0002)
    new = dict(old, regularization=engine.regularization_contract(.1))
    assert engine.assert_resume_contract(old, new)
    assert not engine.assert_resume_contract(new, new)
    with pytest.raises(ValueError): engine.assert_resume_contract(old, dict(new, lr=.001))
    with pytest.raises(ValueError): engine.assert_resume_contract(new, dict(old, regularization=engine.regularization_contract(.2)))


def test_batch_size_transition_requires_explicit_flag_and_preserves_other_checks():
    from scripts.phase5_experiments.run_phase5_hierarchical import assert_resume_contract
    old = dict(target_tokens=640, lr=.0002)
    new = dict(old, target_tokens=1280)
    with pytest.raises(ValueError): assert_resume_contract(old, new)
    assert assert_resume_contract(old, new, allow_batch_change=True)
    with pytest.raises(ValueError): assert_resume_contract(old, dict(new, lr=.001), allow_batch_change=True)


def test_ssl_stochastic_depth_exact_resume(tmp_path, monkeypatch):
    from scripts.phase5_experiments import run_phase5_hierarchical as engine
    cfg=replace(HierarchicalConfig(),channel_width=8,width=16,channel_layers=2,
                local_layers=2,global_layers=2,fusion_layers=2,dropout=.05)
    monkeypatch.setattr(engine,'HierarchicalConfig',lambda:cfg)
    monkeypatch.setattr(engine.signal,'signal',lambda *a:None)
    monkeypatch.setattr(engine,'export_hierarchical_curves',lambda *a:None)
    class Dataset:
        def __init__(self,*a,samples=4,**kw):
            self.samples=samples
            with torch.random.fork_rng():
                torch.manual_seed(711); self.batch=window(80)
        def __len__(self):return self.samples
        def set_epoch(self,epoch):pass
        def __getitem__(self,i):return {k:v[0] if isinstance(v,torch.Tensor) else v[0] for k,v in self.batch.items()}
    monkeypatch.setattr(engine,'HierarchicalDataset',Dataset)
    def run(name,resume=False,max_steps=0):
        args=engine.build_parser().parse_args(['--output',str(tmp_path/name),'--device','cpu','--workers','0',
            '--epochs','1','--samples-per-epoch','6','--validation-samples','1','--target-tokens','160',
            '--drop-path-rate','.3','--no-checkpoint-max-tokens','0'])
        args.resume=resume;args.max_steps=max_steps
        engine.run(args)
        return torch.load(tmp_path/name/'latest_checkpoint.pt',weights_only=False)
    full=run('continuous')
    run('resumed',max_steps=1)
    resumed=run('resumed',resume=True)
    assert (full['step'],full['best']) == (resumed['step'],resumed['best'])
    for key in full['model']:
        torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
