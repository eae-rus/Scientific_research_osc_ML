"""Audit saved v6 population summaries and article tables; no signal inference."""
import csv
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = REPO / 'data/phase5/pdr_analysis_v6'
ALGS = ('adaptive_pdr_mir', 'phase_pdr_basic', 'pos_seq_pdr_basic',
        'phase_power_pdr_basic', 'pos_seq_power_pdr_basic',
        'pdr_sivokobylenko_2pt', 'pdr_sivokobylenko_5pt',
        'pdr_bmrz_q_assisted', 'pdr_bavr072_crosspol')
SOURCES = ('open_ee', 'french_rte')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--article', type=Path,
                        help='Article input; default is the immutable version used by this audit.')
    args = parser.parse_args()
    inputs = []

    def preserve(path):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        target = ROOT / 'sources/PDR-05' / digest / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == raw
        else:
            target.write_bytes(raw)
        inputs.append({'path': path.relative_to(REPO).as_posix(), 'sha256': digest,
                       'snapshot': target.relative_to(REPO).as_posix()})
        return target

    def rows(name):
        with preserve(DATA / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    accepted_sha = 'a600f076da6631de7e20937ec468827360e1300836050ce83ffc87230dbc2a55'
    accepted = ROOT / 'sources/PDR-05' / accepted_sha / 'PHASE_5_PDR_STATISTICAL_ARTICLE_DRAFT.md'
    article = args.article or accepted
    assert hashlib.sha256(article.read_bytes()).hexdigest() == accepted_sha, 'Article version changed; review dependent claims before accepting a new audit.'
    inputs.append({'path':'docs/article/PHASE_5_PDR_STATISTICAL_ARTICLE_DRAFT.md',
                   'sha256':accepted_sha,'snapshot':accepted.relative_to(REPO).as_posix()})
    code = preserve(REPO / 'scripts/phase5_experiments/analyze_pdr_dataset_study.py')
    review_code = preserve(REPO / 'scripts/phase5_experiments/review_pdr_analysis_results.py')
    text = article.read_text(encoding='utf-8')
    tables = {}
    for number in range(4, 10):
        start = text.index(f'Таблица {number}.')
        block = re.search(r'^\|.*(?:\n\|.*)*', text[start:], flags=re.MULTILINE).group(0)
        tables[number] = [[c.strip().strip('*') for c in line.strip().strip('|').split('|')]
                          for line in block.splitlines()[2:] if line.startswith('|')]

    cells = []
    def check(table, row, source, published, expected, locator):
        nums = [float(v.replace(',', '.')) for v in re.findall(r'\d+[,.]\d+', published)]
        if 'менее' in published:
            ok = expected[0] < 0.01
        else:
            assert len(nums) == len(expected), (table, row, source, published, expected)
            ok = all(abs(value - 100 * actual) <= .00500001 for value, actual in zip(nums, expected))
        cells.append({'table': table, 'row': row, 'source': source, 'published': published,
                      'expected_percent': [100 * x for x in expected], 'matches': ok,
                      'primary_locator': locator})

    pop = rows('review_population_summary.csv')
    eligible_pop = {r['source']: r for r in pop if r['population'] == 'pdr_eligible_2i2u'}
    for i, alg in enumerate(ALGS):
        for j, source in enumerate(SOURCES):
            check(4, alg, source, tables[4][i][j+1],
                  [float(eligible_pop[source][alg+'__record_mean_coverage'])],
                  f'review_population_summary.csv[population=pdr_eligible_2i2u,source={source}].{alg}__record_mean_coverage')

    boot = rows('review_algorithm_record_bootstrap.csv')
    for i, alg in enumerate(ALGS):
        for j, source in enumerate(SOURCES):
            r = next(r for r in boot if r['source'] == source and r['algorithm_id'] == alg)
            values = [float(r[k]) for k in ('mean_record_forward_fraction', 'bootstrap95_low', 'bootstrap95_high')]
            assert values[1] <= values[0] <= values[2]
            check(5, alg, source, tables[5][i][j+1], values,
                  f'review_algorithm_record_bootstrap.csv[source={source},algorithm_id={alg}]')

    current = rows('review_current_profiles.csv')
    bins = ('<0.01', '0.01-0.05', '0.05-0.20', '0.20-0.50', '0.50-1.00', '1.00-2.00', '>=2.00')
    fields = ('fraction_of_source', 'mean_disagreement_fraction') + tuple(a+'__mean_forward_fraction' for a in ALGS[:3])
    for i, interval in enumerate(bins):
        r = next(r for r in current if r['source']=='open_ee' and r['current_bin_physical_pu']==interval)
        for j, field in enumerate(fields):
            check(6, interval+':'+field, 'open_ee', tables[6][i][j+1], [float(r[field])],
                  f'review_current_profiles.csv[source=open_ee,current_bin_physical_pu={interval}].{field}')
    for source in SOURCES:
        group = [r for r in current if r['source']==source]
        assert sum(int(r['records']) for r in group) == int(eligible_pop[source]['records'])
        assert math.isclose(sum(float(r['fraction_of_source']) for r in group), 1)

    pairs = rows('pairwise_pointwise_agreement.csv')
    for r in pairs:
        tn, fp, fn, tp = [int(r[k]) for k in ('n_reverse_reverse','n_reverse_forward','n_forward_reverse','n_forward_forward')]
        n = tn+fp+fn+tp
        assert n == int(r['common_windows'])
        agreement = (tn+tp)/n
        chance = ((tn+fp)*(tn+fn)+(fn+tp)*(fp+tp))/n**2
        mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        for key, value in [('agreement', agreement), ('disagreement', 1-agreement),
                           ('cohen_kappa', (agreement-chance)/(1-chance)), ('mcc',mcc)]:
            assert math.isclose(float(r[key]), value, rel_tol=1e-10, abs_tol=1e-10), (key,r)
    pair_ids = [(ALGS[0],ALGS[i]) for i in (1,2,5,6,7,8)] + [(ALGS[1],ALGS[2]),(ALGS[1],ALGS[3]),(ALGS[2],ALGS[4])]
    for i, (left,right) in enumerate(pair_ids):
        for j, source in enumerate(SOURCES):
            r = next(r for r in pairs if (r['source'],r['left'],r['right'])==(source,left,right))
            check(7, left+':'+right, source, tables[7][i][j+1], [float(r['disagreement'])],
                  f'pairwise_pointwise_agreement.csv[source={source},left={left},right={right}].disagreement')

    patterns = rows('algorithm_state_patterns.csv')
    for source in SOURCES:
        group = [r for r in patterns if r['source']==source]
        denominator = sum(int(r['point_count']) for r in group)
        for r in group:
            assert math.isclose(int(r['point_count'])/denominator, float(r['point_fraction']), abs_tol=1e-12)
        for i, pattern in enumerate(('111111111','000000000','100000000','101011111','000001000')):
            r = next(r for r in group if r['state_pattern']==pattern)
            check(8, pattern, source, tables[8][i][SOURCES.index(source)+2], [float(r['point_fraction'])],
                  f'algorithm_state_patterns.csv[source={source},state_pattern={pattern}].point_fraction')
        consensus = sum(float(r['point_fraction']) for r in group if r['state_pattern'] in ('111111111','000000000'))
        check(8, 'consensus', source, tables[8][5][SOURCES.index(source)+2], [consensus],
              f'algorithm_state_patterns.csv[source={source},state_pattern in (111111111,000000000)]')

    groups = rows('transition_groups.csv')
    for i, label in enumerate(('0_stable','0_other_switch','1_2','3_10','11_30','31_plus')):
        for j, source in enumerate((*SOURCES,'all')):
            r = next(r for r in groups if (r['source'],r['transition_group'])==(source,label))
            check(9, label, source, tables[9][i][j+2], [float(r['fraction'])],
                  f'transition_groups.csv[source={source},transition_group={label}].fraction')

    # Independently rebuild the groups and three coverage averages from per-record rows.
    eligibility = rows('record_signal_eligibility.csv')
    eligible_ids = {(r['source'],r['record_id']) for r in eligibility if r['pdr_structurally_eligible'].lower() in ('true','1')}
    counts, groups_rebuilt = Counter(), defaultdict(Counter)
    accum = defaultdict(lambda: defaultdict(float))
    hash_forward = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    no_valid = Counter()
    path = preserve(DATA/'record_statistics.csv')
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if (r['source'],r['record_id']) not in eligible_ids:
                continue
            source = r['source']
            tr, total = int(r[ALGS[0]+'__transitions']), int(r['total_transitions'])
            assert total == sum(int(r[a+'__transitions']) for a in ALGS)
            label = ('0_other_switch' if tr==0 and total>0 else '0_stable' if tr==0
                     else '1_2' if tr<=2 else '3_10' if tr<=10 else '11_30' if tr<=30 else '31_plus')
            assert label == r['adaptive_transition_group']
            for scope in (source,'all'):
                counts[scope] += 1
                groups_rebuilt[scope][label] += 1
            if total==0 and all(int(r[a+'__valid_windows'])==0 for a in ALGS):
                no_valid[source] += 1
            dur, windows = float(r['duration_sec']), int(r['n_windows'])
            for a in ALGS:
                x = float(r[a+'__coverage_fraction'])
                stats = accum[(source,a)]
                stats['records'] += 1; stats['sum'] += x
                stats['duration'] += dur; stats['duration_coverage'] += dur*x
                stats['windows'] += windows; stats['point_coverage'] += windows*x
                value = r[a+'__forward_fraction']
                if value and math.isfinite(float(value)):
                    hash_forward[source][r['input_sha256']][a].append((float(value), x))
    assert counts['all']==48790 and counts['open_ee']==36737 and counts['french_rte']==12053
    for r in groups:
        assert groups_rebuilt[r['source']][r['transition_group']]==int(r['records'])
        assert math.isclose(int(r['records'])/counts[r['source']],float(r['fraction']),abs_tol=1e-12)
    weighting = rows('review_weighting_sensitivity.csv')
    weighted = []
    for (source,a), st in accum.items():
        values = {'record_mean':st['sum']/st['records'], 'duration_weighted_mean':st['duration_coverage']/st['duration'],
                  'point_weighted_mean':st['point_coverage']/st['windows']}
        r = next(r for r in weighting if r['population']=='pdr_eligible_2i2u' and r['source']==source and r['metric']==a+'__coverage_fraction')
        for k,v in values.items():
            assert math.isclose(v,float(r[k]),abs_tol=1e-10), (source,a,k,v,r[k])
        weighted.append({'source':source,'algorithm':a,**values})
    # The table-5 mean gives one vote per hash after averaging duplicate rows.
    for r in boot:
        source,a = r['source'],r['algorithm_id']
        values = [sum(v for v,c in group[a])/len(group[a]) for group in hash_forward[source].values()
                  if group[a] and sum(c for v,c in group[a])>0]
        assert len(values)==int(r['unique_input_hashes'])
        assert math.isclose(sum(values)/len(values),float(r['mean_record_forward_fraction']),abs_tol=1e-10)

    mismatches = [c for c in cells if not c['matches']]
    output = {'task':'C02-PDR-05-03-002','date':'2026-10-07','status':'AUDITED_WITH_CONFLICTS',
              'inputs':inputs,'table_checks':cells,'mismatches':mismatches,
              'records_recomputed':dict(counts),'transition_groups_recomputed':dict(groups_rebuilt),
              'zero_transitions_and_no_valid_points':dict(no_valid),'coverage_recomputed':weighted,
              'pairwise_rows_recomputed':len(pairs),
              'bootstrap':{'means_recomputed':18,'intervals':'saved bounds compared with article; resampling NOT rerun',
                           'generator_seed':20260818,'generator_repetitions':1000,'unit':'one vote per input_sha256 group'},
              'limits':['No raw-signal reconstruction or physical truth audit.',
                        'Tables 1–3 refer to the preceding protocol audit; threshold-scale conflicts remain open.',
                        'Article and manuscript unchanged; publication acceptance remains open.']}
    target = ROOT/'review/checks/C02-PDR-05-population.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cells':len(cells),'matched':len(cells)-len(mismatches),
                      'mismatches':mismatches,'records':dict(counts),'no_valid':dict(no_valid)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
