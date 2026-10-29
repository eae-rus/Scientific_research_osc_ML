"""Physical, locality, missing-data and time contracts of the new version."""
from dataclasses import replace
import numpy as np
import torch
import pytest
from osc_tools.ml.models.phasor_hierarchical import (
    HierarchicalConfig, PhasorHierarchicalKAN, physical_features, vector_loss,
)
from osc_tools.ml.phase5_hierarchical_data import token_positions, collate_windows, ExactFrequencyBuilder, build_local_features


def small_model():
    torch.manual_seed(7)
    cfg = replace(HierarchicalConfig(),channel_width=8,width=16,channel_layers=1,
                  local_layers=1,global_layers=1,fusion_layers=1,dropout=0,block_tokens=4)
    return PhasorHierarchicalKAN(cfg)


def inputs(t=8):
    torch.manual_seed(9)
    return (torch.randn(1,t,13,6,2)*.1,torch.ones(1,t,13,6,dtype=torch.long),
            torch.arange(t)[None,:].float()/8,torch.ones(1,t,dtype=torch.bool))


def test_physical_edges_and_power_direction():
    z = torch.zeros(1,1,13,6,2)
    z[...,0,0]=1; z[...,3,0]=2
    visible = torch.ones(z.shape[:-1],dtype=torch.bool)
    p = physical_features(z,visible)
    assert torch.allclose(p[...,0,0],torch.tensor(np.log(2),dtype=p.dtype))
    assert (p[...,0,3]>.99).all()
    z[...,0,:] *= -1
    reverse = physical_features(z,visible)
    assert (reverse[...,0,3]<-.99).all()
    # One harmonic affects no other physical harmonic or sequence.
    z2 = z.clone(); z2[:,:,5,3,1]=.7
    delta = physical_features(z2,visible)-reverse
    assert delta[:,:,5,0].abs().sum()>0
    delta[:,:,5,0]=0
    assert delta.abs().max()==0
    visible[...,0]=False
    assert physical_features(z2,visible)[...,0,:].abs().max()==0


def test_channel_stem_independence_and_hidden_values():
    model = small_model()
    z,state,_,_ = inputs()
    z.requires_grad_()
    h,_,_ = model.encode_channels(z,state)
    h[...,0,:].sum().backward()
    assert z.grad[...,1:,:].abs().max()==0
    state[:,:,2,0]=3
    h1 = model.encode_channels(z.detach(),state)[0]
    altered = z.detach().clone(); altered[:,:,2,0]=1e6
    h2 = model.encode_channels(altered,state)[0]
    assert torch.equal(h1,h2)


def test_local_context_isolation_and_global_context_access():
    model = small_model().eval()
    z,s,t,v = inputs()
    with torch.no_grad():
        first = model(z,s,t,v)
        changed = z.clone(); changed[:,4:]*=3
        second = model(changed,s,t,v)
    assert torch.equal(first["local_features"][:,:4],second["local_features"][:,:4])
    assert not torch.allclose(first["prediction"][:,:4],second["prediction"][:,:4],atol=1e-8,rtol=1e-8)


def test_padding_invariance_and_finite_gradients():
    model = small_model().eval()
    z,s,t,v = inputs(5)
    first = model(z,s,t,v)["prediction"]
    zpad=torch.cat((z,torch.zeros(1,7,13,6,2)),1)
    spad=torch.cat((s,torch.zeros(1,7,13,6,dtype=torch.long)),1)
    tpad=torch.cat((t,torch.zeros(1,7)),1)
    vpad=torch.cat((v,torch.zeros(1,7,dtype=torch.bool)),1)
    other = model(zpad,spad,tpad,vpad)["prediction"][:,:5]
    assert torch.allclose(first,other,atol=2e-6)
    model.train()
    out = model(z,s,t,v)["prediction"]
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.local_blocks[0].physics.mta.grad is not None


def test_vector_loss_rotation_and_zero():
    a,b = torch.randn(3,2),torch.randn(3,2)
    angle=.7
    rot=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=a.dtype)
    mask=torch.tensor([True,False,True])
    l,n=vector_loss(a,b,mask)
    rotated,_=vector_loss(a@rot,b@rot,mask)
    assert torch.allclose(l,rotated,atol=1e-6) and n==2
    same,_=vector_loss(a,a,mask)
    assert same==0


def test_absolute_grid_avoids_stride_drift():
    positions=token_positions(0,800,12)
    assert positions[-1]==1199
    assert set(np.diff(positions))=={1,2}
    ideal=np.arange(1,801)*12/8-1
    assert np.max(np.abs(positions-ideal))<=.5


def test_physical_rotation_invariance():
    z,_,_,_=inputs()
    visible=torch.ones(z.shape[:-1],dtype=torch.bool)
    angle=.41
    rot=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=z.dtype)
    assert torch.allclose(physical_features(z,visible),physical_features(z@rot,visible),atol=2e-5)


def test_noninteger_spp_uses_actual_fourier_frequency():
    spp=31.3
    builder=ExactFrequencyBuilder(spp)
    raw=np.tile(np.cos(2*np.pi*9*np.arange(1000)/spp)[:,None],(1,8))
    ends=np.array([400,700])
    actual=builder._phasors_prefix(raw,ends,31,9)
    for row,end in enumerate(ends):
        segment=raw[end-30:end+1,0]
        expected=2/31*np.sum(segment*np.exp(-1j*2*np.pi*9*np.arange(31)/spp))
        assert np.allclose(actual[row,0,8],expected,atol=1e-6)
    # At even-integer SPP, Nyquist is explicitly missing.
    nyquist=ExactFrequencyBuilder(12)._phasors_prefix(raw,ends,12,6)
    assert np.isnan(nyquist[...,5]).all()


def test_all_missing_and_padding_do_not_create_nan():
    model=small_model().train()
    z,state,t,valid=inputs(8)
    z.zero_(); state.zero_(); valid[:,4:]=False
    output=model(z,state,t,valid)
    assert torch.isfinite(output["prediction"]).all()
    output["prediction"].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_local_fourier_does_not_read_neighboring_raw_blocks():
    rng=np.random.default_rng(42)
    raw=rng.normal(size=(640,8)).astype(np.float32)
    provenance=np.ones(8,dtype=np.uint8)
    original,missing,_,_=build_local_features(raw,32,2,"phase",provenance)
    raw[:320]*=17
    changed,changed_missing,_,_=build_local_features(raw,32,2,"phase",provenance)
    np.testing.assert_equal(original[80:],changed[80:])
    np.testing.assert_equal(missing,changed_missing)
    # lp10 unavailable until the last token in each block, never zero-target.
    assert missing[:79,-12:].all() and not missing[79,-12:].any()


def test_console_migration_is_narrow_and_rejects_numerical_changes():
    from scripts.phase5_experiments.run_phase5_hierarchical import assert_resume_contract
    old={"implementation_sha256":"0341f4e14045b0d6c4089a89755635864799413eff5f443b41f9a7d15e43cf39","lr":.0002}
    new={"model_data_sha256":"8858f68cfeacd482ea92fc1dc8ddfa79930141bbca29c461426b997e57789776",
         "training_recipe":"hierarchical_ssl_v1","lr":.0002}
    assert assert_resume_contract(old,new)
    assert not assert_resume_contract(new,new)
    with pytest.raises(ValueError): assert_resume_contract(old,{**new,"lr":.0003})
    with pytest.raises(ValueError): assert_resume_contract(old,{**new,"model_data_sha256":"changed"})
    with pytest.raises(ValueError): assert_resume_contract({**old,"implementation_sha256":"unknown"},new)


def test_manual_entry_dispatches_current_run_without_training(monkeypatch):
    from scripts.phase5_experiments import run_phase5_pretrain as entry
    from scripts.phase5_experiments import run_phase5_hierarchical as engine
    captured=[]
    monkeypatch.setattr(engine,"run",lambda args:captured.append(args))
    entry.run_manual()
    assert len(captured)==1
    args=captured[0]
    assert args.resume and args.target_tokens==640 and args.validation_samples==64
    assert args.output.name=="phasor_hierarchical_v1"


def test_progress_nonterminal_throttles_and_keeps_machine_status(tmp_path):
    import io,json
    from scripts.phase5_experiments.progress import TrainingProgress
    stream=io.StringIO()
    progress=TrainingProgress(tmp_path,stream=stream)
    progress.begin("Epoch 1",100)
    progress.update(1,{"loss":.5,"step":1})
    progress.update(2,{"loss":.4,"step":2})
    progress.update(3,{"step":3},force=True,status="paused")
    assert len(stream.getvalue().splitlines())==2
    assert '{"kind"' not in stream.getvalue()
    status=json.loads((tmp_path/'progress.json').read_text(encoding='utf-8'))
    assert status['completed']==3 and status['status']=='paused'
