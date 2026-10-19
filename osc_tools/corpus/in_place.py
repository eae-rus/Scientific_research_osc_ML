"""Подготовка пользовательской рабочей копии без повторного копирования корпуса."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

from .agent_runtime import now, write_json
from .case_inventory import classify, sha256

RULE_VERSION = 'in_place_1'
GENERATED_DIRS = {'_phase6', '_phase6_extracted'}


def source_files(folder):
    """Перечислить первичные файлы, исключив собственные каталоги и ссылки."""
    result = []
    for directory, dirs, files in os.walk(folder,followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(n for n in dirs if n not in GENERATED_DIRS and not (current/n).is_symlink() and
                          (current/n).resolve().is_relative_to(folder))
        for name in sorted(files):
            path = current/name
            if (not path.is_symlink() and path.resolve().is_relative_to(folder) and
                name not in {'_phase6_provenance.json','_phase6_record_index.json'}):
                result.append((path,path.relative_to(folder).as_posix()))
    return result


def file_state(path):
    """Получить быструю сигнатуру для повторного прохода по доверенной рабочей копии."""
    info = path.stat()
    return {'size':info.st_size,'mtime_ns':info.st_mtime_ns}


def prepare_in_place(preparer):
    """Обновить индексы на месте; законченные неизменившиеся папки пропускать."""
    config, root, output = preparer.config, preparer.source, preparer.output
    output.mkdir(exist_ok=True)
    saved_inventory = output / 'original_inventory.json'
    seed_path = saved_inventory if saved_inventory.exists() else Path(config.seed_inventory) if config.seed_inventory else None
    seed = json.loads(seed_path.read_text(encoding='utf-8-sig')) if seed_path else {}
    registry = dict(seed.get('case_registry',{}))
    if len(set(registry.values())) != len(registry) or any(not re.fullmatch(r'case_\d{6}',v) for v in registry.values()):
        raise ValueError('Реестр содержит неверные или повторные ID')
    next_id = max((int(v[5:]) for v in registry.values()),default=0)+1
    cases = []
    documents = {'.doc','.docx','.dot','.dotx','.pdf','.rtf'}
    for year in sorted(root.iterdir()):
        if not year.is_dir() or year.is_symlink() or not re.fullmatch(r'\d{4}',year.name) or (config.years and year.name not in config.years):
            continue
        loose = {}
        for child in sorted(year.iterdir()):
            if child.is_symlink():
                continue
            if child.is_dir() and child.name not in GENERATED_DIRS:
                cases.append({'key':child.relative_to(root).as_posix(),'folder':child.relative_to(root).as_posix(),
                              'input':child,'files':source_files(child)})
            elif child.is_file() and child.suffix.lower() in documents and not child.name.startswith('~$'):
                loose.setdefault(child.stem,[]).append(child)
        for stem, files in sorted(loose.items()):
            folder = year.name+'/'+stem
            cases.append({'key':'__loose__/'+folder,'folder':folder,'input':None,'files':[(p,p.name) for p in files]})
    rules = {key:value for key,value in asdict(config).items() if key not in {'years','seed_inventory','verify_unchanged_hashes'}}
    rule_hash = hashlib.sha256(json.dumps([RULE_VERSION,rules],sort_keys=True).encode()).hexdigest()
    processed = skipped = source_checked = 0
    original_cases = []
    for case in cases:
        key = case['key']
        if key not in registry:
            if next_id > 999999:
                raise ValueError('Закончились постоянные ID')
            registry[key] = f'case_{next_id:06d}'
            next_id += 1
        cid = registry[key]
        metadata = output/'cases'/cid
        metadata.mkdir(parents=True,exist_ok=True)
        marker_path = metadata/'prepared.json'
        current = {logical:file_state(path) for path,logical in case['files']}
        try:
            previous = json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else {}
        except (ValueError,OSError):
            previous = {}
        can_skip = (previous.get('status') == 'prepared' and previous.get('source_root') == str(root) and
                    previous.get('rules_sha256') == rule_hash and previous.get('source_state') == current and
                    (metadata/'provenance.json').is_file())
        if can_skip:
            for row in previous['outputs']:
                path = Path(row['path'])
                if (not path.resolve().is_relative_to(root) or not path.is_file() or file_state(path) != row['state'] or
                    (config.verify_unchanged_hashes and sha256(path) != row['sha256'])):
                    can_skip = False
                    break
        if can_skip and config.verify_unchanged_hashes:
            for path,logical in case['files']:
                source_checked += 1
                if sha256(path) != previous['source_sha256'][logical]:
                    can_skip = False
                    break
        if can_skip:
            saved = json.loads((metadata/'provenance.json').read_text(encoding='utf-8'))
            preparer.cases.append(saved)
            preparer.archives += previous.get('archives',[])
            skipped += 1
            original_files = previous['source_sha256']
        else:
            marker_path.unlink(missing_ok=True)  # Незавершённая папка не имеет отметки готовности.
            original_files = {logical:sha256(path) for path,logical in case['files']}
            input_root = case['input'] or output/'loose'/cid/'input'
            input_root.mkdir(parents=True,exist_ok=True)
            entries, archive_start = [], len(preparer.archives)
            for path,logical in case['files']:
                origin = {'source_path':str(path),'source_relative_path':path.relative_to(root).as_posix()}
                if classify(path.name)['role'] == 'archive':
                    entries += preparer.archive(path,input_root,origin)
                elif path.name.lower() == 'thumbs.db' or path.suffix.lower() == '.waves' or path.name.startswith('~$'):
                    entries.append({'logical_path':logical,'origin':origin,'sha256':original_files[logical],
                        'size':path.stat().st_size,'preparation_status':'cache_or_office_temp_excluded'})
                else:
                    entries.append(preparer.store(path,input_root,logical,origin))
            if any(sha256(path) != original_files[logical] for path,logical in case['files']):
                raise ValueError('Первичные байты изменились во время подготовки')
            saved = {'case_id':cid,'source_folder':case['folder'],'input_root':str(input_root),
                     'metadata_root':str(metadata),'files':entries,'source_files_not_modified':True,'semantic_importance':'not_assessed'}
            write_json(metadata/'provenance.json',saved)
            preparer.cases.append(saved)
            outputs = [{'path':f['prepared_path'],'sha256':f['sha256'],'state':file_state(Path(f['prepared_path']))}
                       for f in entries if f.get('prepared_path')]
            archives = preparer.archives[archive_start:]
            complete = not any(a['status'] not in {'listed','volume_processed_via_first_part'} or any(
                m['status'] in {'extraction_failed','unsafe_or_encrypted_skipped','size_limit_skipped','depth_limit_skipped'} for m in a['members']) for a in archives)
            write_json(marker_path,{'status':'prepared' if complete else 'partial','source_root':str(root),
                'rules_sha256':rule_hash,'source_state':current,'source_sha256':original_files,'outputs':outputs,
                'archives':archives,'created_at':now(),'verification':'sha256' if config.verify_unchanged_hashes else 'size_mtime_on_resume'})
            if case['input']:
                pointer = case['input']/'_phase6'
                pointer.mkdir(exist_ok=True)
                write_json(pointer/'prepared.json',{'case_id':cid,'status':'prepared' if complete else 'partial',
                           'metadata':str(metadata),'originals_not_deleted':True})
            source_checked += len(original_files)
            processed += 1
        original_cases.append({'case_id':cid,'source_folder':case['folder'],
            'files':[{'path':logical,'sha256':original_files[logical],**classify(path.name)} for path,logical in case['files']]})
    index = {'source_root':str(root),'created_at':now(),'cases':preparer.cases}
    prepared = {'source_root':str(root),'mode':'in_place','case_registry':registry,'cases':[]}
    for case in preparer.cases:
        input_root = Path(case['input_root'])
        prepared['cases'].append({'case_id':case['case_id'],'source_folder':input_root.relative_to(root).as_posix(),
            'files_to_include':[Path(f['prepared_path']).relative_to(input_root).as_posix() for f in case['files'] if f.get('prepared_path')]})
    write_json(output/'original_inventory.json',{'source_root':str(root),'case_registry':registry,'cases':original_cases})
    write_json(output/'prepared_inventory.json',prepared)
    write_json(output/'provenance_index.json',index)
    write_json(output/'archive_manifest.json',preparer.archives)
    from .record_identity import build_record_index, write_legacy_hash_table
    records = build_record_index(index)
    prior_path = output/'record_index.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {'records':[]}
    previous_records = {(r['record_id'],r['case_id']):r for r in prior['records']}
    current_keys = set()
    for record in records['records']:
        key = (record['record_id'],record['case_id'])
        current_keys.add(key)
        record['current_input'] = True
        if key in previous_records:
            record['derivatives'] = previous_records[key].get('derivatives',[])
    for key,record in previous_records.items():
        if key not in current_keys:
            record['current_input'] = False
            record['snapshot_status'] = 'historical_or_outside_selected_years'
            records['records'].append(record)
    # Общий индекс хранит происхождение прежних результатов, даже если выбран
    # другой год или первичный файл изменён. Новая подготовка не обнуляет lineage.
    for case in preparer.cases:
        local = Path(case['metadata_root'])/'record_index.json'
        data = json.loads(local.read_text(encoding='utf-8'))
        data['records'] = [r for r in records['records'] if r['case_id'] == case['case_id']]
        write_json(local,data)
    write_json(output/'record_index.json',records)
    legacy = write_legacy_hash_table(records,output)
    summary = {'status':'partial' if any(json.loads((output/'cases'/c['case_id']/'prepared.json').read_text(encoding='utf-8'))['status'] != 'prepared' for c in preparer.cases) else 'prepared',
        'mode':'in_place','cases':len(cases),'cases_processed':processed,'cases_skipped':skipped,
        'source_files_hash_checked_this_pass':source_checked,'source_files_unchanged':True,
        'extracted_bytes':preparer.extracted_bytes,'archives':len(preparer.archives),'record_candidates':len(current_keys),
        'indexed_records_including_history':len(records['records']),
        'legacy_hash_table':legacy,'output':str(output),'created_at':now(),
        'resume_check':'sha256' if config.verify_unchanged_hashes else 'size_mtime', 'signals_read':False}
    write_json(output/'preparation_config.json',asdict(config))
    write_json(output/'summary.json',summary)
    return summary
