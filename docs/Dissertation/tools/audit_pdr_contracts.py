"""Bounded, reproducible audit of public PDR settings and scalar conventions.

Does not import private plugins, train models, relabel archives or overwrite
experiments. Preserves the exact input files used for this audit by SHA-256.
"""
import ast, cmath, hashlib, json, math
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    paths = ['docs/article/PHASE_5_PDR_STATISTICAL_ARTICLE_DRAFT.md',
             'data/phase5/pdr_labels_v6/config.json',
             'osc_tools/pdr/pdr_signal_utils.py', 'osc_tools/pdr/study.py',
             'scripts/phase5_experiments/run_pdr_dataset_study.py',
             'scripts/phase5_experiments/analyze_pdr_dataset_study.py',
             'osc_tools/pdr/public_algorithms/basic_phase.py',
             'osc_tools/pdr/public_algorithms/basic_pos_seq.py',
             'osc_tools/pdr/public_algorithms/basic_phase_power.py',
             'osc_tools/pdr/public_algorithms/basic_pos_seq_power.py',
             'osc_tools/pdr/public_algorithms/advanced_literature_algorithms.py']
    sources = []
    for rel in paths:
        data = (REPO/rel).read_bytes(); checksum = sha(data)
        snapshot = ROOT/'sources/PDR-05'/checksum/Path(rel).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            assert snapshot.read_bytes() == data
        else:
            snapshot.write_bytes(data)
        sources.append({'path':rel,'sha256':checksum,
                        'snapshot':snapshot.relative_to(REPO).as_posix()})
    config = json.loads((REPO/paths[1]).read_text(encoding='utf-8'))
    settings = []
    for rel in paths[6:]:
        tree = ast.parse((REPO/rel).read_text(encoding='utf-8'))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            values = {}
            for n in cls.body:
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if isinstance(target, ast.Name) and target.id in ('algorithm_id','tunable_parameters'):
                            values[target.id] = ast.literal_eval(n.value)
            if not values: continue
            params = values['tunable_parameters'].copy()
            params['scale_profile'] = config['dataset_scale_profile']
            fingerprint = sha(json.dumps(params,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
            expected = config['parameter_fingerprints'][values['algorithm_id']]
            settings.append({'algorithm_id':values['algorithm_id'],'path':rel,
                             'class_line':cls.lineno,'params':params,
                             'parameter_fingerprint':fingerprint,
                             'stored_fingerprint':expected,'matches_run_settings':fingerprint==expected})
    assert len(settings)==8
    # Execute only the selected public scalar conversion function, not package
    # imports or plugin discovery. It has no I/O or experiment side effects.
    tree = ast.parse((REPO/paths[2]).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_thresholds_for_profile')
    namespace={'math':math,'Tuple':tuple}
    exec(compile(ast.Module(body=[function],type_ignores=[]),paths[2],'exec'),namespace)
    scale=namespace['scale_thresholds_for_profile']
    converted=scale(.05,.05,.05,'dataset_peak_phasor')
    expected=(.05*math.sqrt(2)/(3*math.sqrt(3)),.05*math.sqrt(2)/20,.05*2/(60*math.sqrt(3)))
    assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(converted,expected))
    comparisons=[]
    a=cmath.exp(2j*math.pi/3)
    for angle in (-135,-45,0,45,135):
        ua=1+0j; ub=a*a; uc=a; ia=cmath.exp(-1j*math.radians(angle)); ib=ia*a*a; ic=ia*a
        u1=(ua+a*ub+a*a*uc)/3; i1=(ia+a*ib+a*a*ic)/3
        phase=abs(ub-uc)/math.sqrt(3)*abs(ia)*math.cos(cmath.phase(ub-uc)-cmath.phase(ia)+math.pi/4)
        seq=(u1*(i1*cmath.exp(1j*math.pi/4)).conjugate()).real
        assert math.isclose(phase,seq,abs_tol=1e-12)
        comparisons.append({'lag_angle_deg':angle,'phase_moment':phase,'sequence_moment':seq})
    lag_results=[]
    for fs in (1200,1600,6400,20000):
        lag=max(1,int(round(fs*.001)))
        lag_results.append({'sampling_rate_hz':fs,'sample_lag':lag,
                            'two_sample_delay_ms':1000*lag/fs,
                            'five_sample_window_ms':4000*lag/fs,
                            'five_sample_estimate_delay_ms':2000*lag/fs})
    assert config['algorithm_ids'][0]=='adaptive_pdr_mir'
    checks={'scope':'C02-PDR-05-02b; metadata and scalar identities, not archive recomputation',
            'date':date.today().isoformat(),'sources':sources,'settings':settings,
            'all_public_settings_match_v6':all(x['matches_run_settings'] for x in settings),
            'scale_profile':config['dataset_scale_profile'],
            'rms_thresholds':{'U':.05,'I':.05,'P':.05},
            'converted_peak_internal':dict(zip(('U','I','P'),converted)),
            'balanced_phase_sequence_moments':comparisons,'quantized_sample_intervals':lag_results,
            'actual_bit_order':config['algorithm_ids'],
            'findings':[
              {'id':'PDR-M01','article_lines':[196,307,317],'kind':'description_scope','detail':'All-algorithm Fourier statement excludes raw-sample estimation exceptions.'},
              {'id':'PDR-M02','article_lines':[154,155,156,271],'kind':'scale_or_setting_mismatch','detail':'Current two power baselines have physical RMS p_thresh_pu=0.05, mapping to internal peak 0.0009622504486493763; source 0.0025/0.001 require explicit scale and version.'},
              {'id':'PDR-M03','article_lines':[433,435],'kind':'order_mismatch','detail':'100000000 is adaptive-only for config/CSV order, which differs from article table row order.'},
              {'id':'PDR-M04','article_lines':[313,335],'kind':'timing_scope','detail':'h=1ms is quantized to samples; batch study starts its end_indices after one full period. Minimal estimator delay is not complete pipeline response delay.'}
            ],
            'limits':'Parameter fingerprints do not establish identical implementation code at experiment time. Adaptive private internals are not inspected. Runtime defects, test labels and archived model outputs are outside these scalar checks.'}
    output=ROOT/'review/checks/C02-PDR-05-02.json'
    output.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'public_algorithms':len(settings),'settings_match':checks['all_public_settings_match_v6'],'P_internal':converted[2],'files_snapshotted':len(sources),'scalar_checks':'passed'},ensure_ascii=False))

if __name__=='__main__':
    main()
