from dataclasses import replace
import numpy as np
import torch
import pytest
from osc_tools.ml.models.phasor_hierarchical_v2 import HierarchicalV2Config,PhasorHierarchicalKANV2


def test_previous_encoder_name_loads_with_unchanged_weights(tmp_path):
    from dataclasses import asdict
    from scripts.phase5_experiments.run_phase5_hierarchical_ozz import OZZClassifier
    from osc_tools.ml.phase5_ozz_inference import load_hierarchical
    model = OZZClassifier(PhasorHierarchicalKANV2(tiny()))
    architecture = asdict(tiny())
    architecture['architecture_id'] = 'phase_ozz_v2'
    state = dict(model=model.state_dict(), step=7, contract=dict(architecture=architecture,
        feature_mode='phase_v2', recipe='phase_ozz_v2', rtds_dir=str(tmp_path)))
    loaded, config = load_hierarchical(state, tmp_path/'old.pt', torch.device('cpu'))
    assert config['model_type'] == 'PhasorHierarchicalKANV2'
    assert config['feature_mode'] == 'hierarchical_v2'
    assert config['recipe'] == 'hierarchical_ozz_v2'
    assert state['contract']['architecture']['architecture_id'] == 'phase_ozz_v2'
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name])
from osc_tools.ml.phase5_hierarchical_v2_features import phase_window
from osc_tools.ml.phase5_ozz_synthesis import compose_context,UnsuitableEdges
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import OZZClassifier


def wave(spp=32., periods=25):
    t=np.arange(round(spp*periods))/spp*2*np.pi
    phases=np.array([0,-2*np.pi/3,2*np.pi/3])
    x=np.zeros((len(t),8),np.float32)
    x[:,:3]=.03*np.sin(t[:,None]+phases-.2)
    x[:,4:7]=.3*np.sin(t[:,None]+phases)
    return x


def test_reference_u_i_zero_and_missing_phase_does_not_hide_others():
    x=wave();y=np.zeros((len(x),4));full=phase_window(x,y,32.)
    assert (full['reference_source'][7:]==1).all()
    missing=x.copy();missing[:,4]=np.nan
    b=phase_window(missing,y,32.)
    assert not b['state'][:,:,3].any()
    assert b['state'][7:,0,4:].all() and b['state'][7:,0,:3].all()
    torch.testing.assert_close(full['z'][7:,0,:3],b['z'][7:,0,:3],atol=2e-6,rtol=2e-5)
    no_voltage=x.copy();no_voltage[:,4:7]=np.nan
    assert (phase_window(no_voltage,y,32.)['reference_source'][7:]==2).all()
    silent=x.copy();silent[:,:]=0
    assert not phase_window(silent,y,32.)['reference_source'].any()


def tiny():
    return HierarchicalV2Config(channel_width=8,channel_heads=2,width=16,heads=2,
        channel_layers=1,local_layers=1,global_layers=1,fusion_layers=1,dropout=0,drop_path_rate=0)


def test_overlap_merge_padding_masks_and_summary_tokens():
    cfg=tiny();model=PhasorHierarchicalKANV2(cfg).eval()
    x=wave(periods=16);y=np.zeros((len(x),4));batch=phase_window(x,y,32.)
    data={k:v[None] for k,v in batch.items()}
    result=model(data['z'],data['state'],data['times'],data['valid'])
    assert result['token_features'].shape==(1,128,16)
    assert result['global_features'].shape==(1,6,16) # windows 0,40,80; mean and max each
    # Changing masked padding must not change real output.
    z=torch.cat((data['z'],torch.randn(1,32,13,6,2)),1)
    state=torch.cat((data['state'],torch.ones(1,32,13,6,dtype=torch.long)),1)
    times=torch.cat((data['times'],torch.ones(1,32)*99),1)
    valid=torch.cat((data['valid'],torch.zeros(1,32,dtype=torch.bool)),1)
    other=model(z,state,times,valid)
    torch.testing.assert_close(other['token_features'][:,:128],result['token_features'],atol=2e-6,rtol=2e-5)
    assert torch.isfinite(other['token_features']).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_merge_constant_and_empty_input_are_finite(dtype):
    index=torch.arange(3)[:,None]*40+torch.arange(80)[None]
    windows=torch.ones(1,3,80,2,dtype=dtype);mask=torch.ones(1,3,80,dtype=torch.bool)
    torch.testing.assert_close(PhasorHierarchicalKANV2.merge(windows,mask,index,160),torch.ones(1,160,2))
    model=PhasorHierarchicalKANV2(tiny()).eval()
    r=model(torch.zeros(1,13,13,6,2),torch.zeros(1,13,13,6,dtype=torch.long),torch.zeros(1,13),torch.ones(1,13,dtype=torch.bool))
    assert not r['valid_mask'].any() and torch.isfinite(r['token_features']).all()


def test_global_and_local_heads_train_and_feature_api_exists():
    model=OZZClassifier(PhasorHierarchicalKANV2(tiny())).train()
    data={k:v[None] for k,v in phase_window(wave(periods=20),np.zeros((640,4)),32.).items()}
    r=model(data,return_aux=True)
    (r['logits'].square().mean()+.25*r['local_logits'].square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.global_blocks.parameters())
    assert model.backbone.gate.weight.grad is not None
    assert model(data,use_global=False).shape==(1,160,4)


@pytest.mark.parametrize('spp',[32.,769.2307692307693])
def test_insertions_reserve_half_seconds_and_only_one_source_period_is_blended(spp):
    x=wave(spp);y=np.zeros((len(x),4));y[int(5*spp):int(7*spp),1]=1
    n=round(30*spp*50)
    a,t,p=compose_context(x,y,spp,n,np.random.default_rng(4))
    assert len(p)==10 and a.shape==(n,8)
    assert all(left['reserve_end']<=right['reserve_start'] for left,right in zip(p,p[1:]))
    for r in p:
        assert 0<=r['source_crop_start']<=r['left_blend_source_end']<=spp
        assert len(x)-1-spp<=r['right_blend_source_start']<=r['source_crop_end']<=len(x)-1+1e-6
    assert t[:,1].any() and not t[:,[0,2,3]].any()
    normal,labels,placements=compose_context(x,y,spp,n,np.random.default_rng(4),include_events=False)
    assert not labels.any() and not placements
    assert np.isfinite(normal).all()


def test_fault_at_source_edge_is_rejected_not_smoothed():
    x=wave();y=np.zeros((len(x),4));y[-1,0]=1
    with pytest.raises(UnsuitableEdges):compose_context(x,y,32.,3200,np.random.default_rng(2))


def test_final_partial_token_is_kept():
    x=wave()[:323];y=np.zeros((len(x),4));y[-1,3]=1
    result=phase_window(x,y,32.)
    assert result['sample_ends'][-1]==323 and result['target'][-1,3]==1
    assert result['valid'][-1]


def test_synthetic_normal_context_does_not_trigger_source_label_contradiction(monkeypatch):
    import osc_tools.ml.phase5_hierarchical_ozz as module
    x=wave();y=np.zeros((len(x),4),np.float32);y[150:200,1]=1
    source=dict(raw=x,targets=y,dt=1/1600.,ozz=y.any(1))
    monkeypatch.setattr(module,'load_raw_csv',lambda *a,**k:source)
    records=[dict(kind='rtds',path='fixture.csv',group=1)]
    ds=module.OZZDataset(records,20,42,30.,True,event_windows=True,feature_mode='hierarchical_v2',
        label_tail_periods=1,synthetic_context=True,synthetic_normal_probability=1.)
    found=False
    for i in range(8):
        batch=ds[i]
        if batch['data_note']=='synthetic_context_0_copies':
            assert not batch['target'].any() and batch['supervised'].any()
            found=True;break
    assert found


def test_real_comtrade_reader_keeps_single_available_phase(monkeypatch):
    import polars as pl
    from types import SimpleNamespace
    from pathlib import Path
    from scripts.phase4_experiments.real_ozz import inference_real_ozz as module
    n=240
    frame=pl.DataFrame({'I | Bus-1 | phase: A':np.ones(n), 'U | BusBar-1 | phase: B':np.ones(n)*30})
    record=SimpleNamespace(cfg=SimpleNamespace(sample_rates=[(1200,n)],frequency=50.),time=np.arange(n)/1200)
    monkeypatch.setattr(module,'ReadComtrade',lambda:SimpleNamespace(read_comtrade=lambda *a:(record,frame)))
    norm=SimpleNamespace(norm_coef=pl.DataFrame({'name':['fixture'],'norm':['YES'],'1Ip_base':[1.],'1Ub_base':[10.]}),normalize_bus_signals=lambda df,name:df)
    _,x,_,_,_=module.load_comtrade_raw(Path('fixture.cfg'),'1',norm,hierarchical=True,allow_partial_phases=True)
    assert x is not None and np.isfinite(x[:,0]).all() and np.isfinite(x[:,5]).all()
    assert np.isnan(x[:,[1,2,3,4,6,7]]).all()
