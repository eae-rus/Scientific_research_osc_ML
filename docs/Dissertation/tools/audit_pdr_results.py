"""Verify saved PDR evaluations, grids, counts and training coverage without inference."""
import hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
MODES=('snapshot_2','snapshot_5','sequence_1_8')

def checksum(data): return hashlib.sha256(data).hexdigest()

def main():
    sources=[]
    def preserve(rel):
        p=REPO/rel; data=p.read_bytes(); digest=checksum(data)
        snap=ROOT/'sources/PDR-05'/digest/p.name
        snap.parent.mkdir(parents=True,exist_ok=True)
        if snap.exists(): assert snap.read_bytes()==data
        else: snap.write_bytes(data)
        sources.append({'path':rel,'sha256':digest,'snapshot':snap.relative_to(REPO).as_posix()})
        return p
    def load(rel): return json.loads(preserve(rel).read_text(encoding='utf-8'))
    def confusion(target,prediction):
        return {key:int(np.sum(mask)) for key,mask in {
            'tp':(target==1)&(prediction==1),'tn':(target==0)&(prediction==0),
            'fp':(target==0)&(prediction==1),'fn':(target==1)&(prediction==0)}.items()}
    def scores(c):
        tp,tn,fp,fn=(c[k] for k in ('tp','tn','fp','fn'))
        f1p=2*tp/(2*tp+fp+fn);f1n=2*tn/(2*tn+fp+fn)
        mcc=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return (f1p+f1n)/2,mcc
    grids={}; results=[]; training=[]
    for mode in MODES:
        for stage,epochs in [('weak',300),('expert',100)]:
            stem=f'experiments/phase5/pdr_{stage}_{mode}_stride5'
            cfg=load(stem+'/config.json')
            rows=[json.loads(x) for x in preserve(stem+'/training_log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            assert cfg['epochs']==epochs and cfg['samples_per_epoch']==20000
            assert len(rows)==epochs and [r['epoch'] for r in rows]==list(range(1,epochs+1))
            chosen=max(rows,key=lambda r:r['selection_score'])['epoch']
            coverage=None
            if stage=='expert':
                coverage={group:sorted({r['train_group_coverage'][group]['record_coverage_fraction'] for r in rows}) for group in ('expert_open_ee','expert_french_rte')}
                assert all(v==[1.0] for v in coverage.values())
                assert cfg['initialization'].replace('\\','/').endswith(f'pdr_weak_{mode}_stride5/latest_checkpoint.pt')
            training.append({'mode':mode,'stage':stage,'epochs':epochs,'selected_epoch':chosen,'samples_per_epoch':20000,
                             'total_presentations':epochs*20000,'expert_record_coverage':coverage,
                             'config':stem+'/config.json','batch_size':cfg['batch_size'],
                             'learning_rate':cfg['learning_rate'],'weight_decay':cfg['weight_decay'],
                             'seed':cfg['seed'],'augmentation_probability':cfg['augmentation_probability']})
        for weights in ('weak_initial','weak_best','expert_best'):
            stem=f'experiments/phase5/pdr_expert_evaluation_v3/{mode}_{weights}'
            report=load(stem+'.json'); validation=report['splits']['validation']; arrays=[]
            cfg_path=Path(report['training_config'])
            assert checksum(cfg_path.read_bytes())==report['training_config_sha256']
            assert checksum(Path(report['checkpoint']).read_bytes())==report['checkpoint_sha256']
            for source in ('open_ee','french_rte'):
                p=preserve(stem+f'__validation__{source}.npz')
                with np.load(p,allow_pickle=False) as archive:
                    z={k:archive[k].copy() for k in archive.files}
                grid=[z[k] for k in ('record_id','window_idx','target')]
                if source in grids:
                    assert all(np.array_equal(x,y) for x,y in zip(grid,grids[source]))
                else:grids[source]=grid
                arrays.append(z)
            z={k:np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
            valid=z['target']!=-999
            counts=confusion(z['target'][valid],z['prediction'][valid])
            app=confusion(valid.astype(int),z['applicable'].astype(int))
            metrics=validation['overall']
            assert all(counts[k]==metrics[k] for k in counts)
            assert all(app[k]==metrics['applicability_'+k] for k in app)
            mf1,mcc=scores(counts);af1,_=scores(app)
            assert math.isclose(mf1,metrics['macro_f1_score'],abs_tol=1e-12)
            assert math.isclose(mcc,metrics['mcc'],abs_tol=1e-12)
            assert math.isclose(af1,metrics['applicability_macro_f1_score'],abs_tol=1e-12)
            joint=float(np.mean((valid==z['applicable']) & ((~valid)|(z['target']==z['prediction']))))
            reported_joint=sum(validation['by_source'][s]['joint_state_accuracy']*validation['grid_signature'][s]['points'] for s in ('open_ee','french_rte'))/len(valid)
            assert math.isclose(joint,reported_joint,abs_tol=1e-12)
            results.append({'mode':mode,'weights':weights,'json_path':stem+'.json',
                            'direction_counts':counts,'applicability_counts':app,
                            'n_direction_points':int(np.sum(valid)),'n_applicability_points':len(valid),
                            'macro_f1_direction':mf1,'mcc_direction':mcc,
                            'macro_f1_applicability':af1,'joint_state_accuracy':joint,
                            'grid_signature':validation['grid_signature'],
                            'checkpoint_hash_verified':True,'config_hash_verified':True})
    complete=[]
    expected={'expert':'9921949d8ffcbe1b1c527fbf18ebcc5aebc95b72325861ed6fdbe4917572f22e',
              'rtds':'787ff8648f48046d0ab6fa5912128f542a893d4dce4c8fe19ce5ac8d351a24c4'}
    for scope,count in [('expert',54),('rtds',100)]:
        prefix=f'data/phase5/pdr_engineering_review/{scope}/full/'
        progress=load(prefix+'progress.json'); summary_path=preserve(prefix+'summary.json')
        assert progress=={'status':'complete','completed':count,'total':count}
        digest=checksum(summary_path.read_bytes())
        assert digest==expected[scope]
        complete.append({'scope':scope,'records':count,'progress':progress,'summary_sha256':digest,'matches_Report_8_15':True})
    for rel in ['docs/phase_discription/PDR_analise/Report_8_phase5_status_and_training_coverage.md',
                'data/phase5/pdr_expert_labels_v1/summary.json',
                'data/phase5/pdr_analysis_v6/summary.json',
                'experiments/phase5/pdr_expert_evaluation_v3/paired_comparison_summary.json']:
        preserve(rel)
    report={'date':'2026-10-07','status':'passed_scoped_checks','sources':sources,'training':training,'evaluations':results,
            'engineering_completeness':complete,
            'limits':'Recomputed metrics from saved predictions, not rerun inference/training. Grid equality does not remove train/validation leakage or checkpoint-selection optimism. No bootstrap resampling rerun; H123 predictions and RTDS per-record outputs not audited here.'}
    (ROOT/'review/checks/C02-PDR-05-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'evaluations':len(results),'prediction_grids':'identical','training_epochs_verified':1200,
                      'full_expert_and_RTDS':'matches Report_8 hashes','sources_preserved':len(sources)},ensure_ascii=False))

if __name__=='__main__':main()
