"""Связь исходных и производных осциллограмм с папками событий.

SHA256 определяет точные байты. ID записи включает все известные компоненты;
прежний MD5 DAT сохранён только как поисковый ключ, допускающий несколько событий.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

from .agent_runtime import now, write_json
from .case_inventory import sha256


def build_record_index(provenance):
    records = []
    for case in provenance['cases']:
        files = {f['logical_path'].casefold(): f for f in case['files'] if f.get('prepared_path')}
        used = set()
        for key, file in sorted(files.items()):
            path = Path(key)
            ext = path.suffix
            if key in used or (file.get('role') != 'recording_candidate' and ext not in {'.dex', '.osc'}):
                continue
            components = [file]
            kind = 'comtrade' if ext == '.cfg' else 'native'
            if ext == '.cff':
                kind = 'comtrade_single_file'
            if ext in {'.dex', '.osc'}:
                kind = 'unparsed_vendor_container'
            if ext == '.cfg':
                for suffix in ('.dat', '.hdr', '.inf'):
                    component = files.get(str(path.with_suffix(suffix)).replace('\\', '/'))
                    if component:
                        components.append(component)
            elif ext == '.do':
                components += [f for k, f in files.items() if Path(k).parent == path.parent and
                               Path(k).stem == path.stem and Path(k).suffix.startswith('.d0')]
            elif file.get('format') == 'ekra-ndr':
                group = re.match(r'(DR\d+)F[0-3]\.\d+', path.name, re.I).group(1)
                components += [f for k, f in files.items() if k != key and Path(k).parent == path.parent and
                    (re.fullmatch(re.escape(group) + r'F[0-3]\.\d+', Path(k).name, re.I) or Path(k).name.startswith('wconfig.'))]
            def component_slot(file):
                """Различать позиции многокомпонентного NDR независимо от имени группы."""
                name = Path(file['logical_path']).name
                return re.sub(r'^DR\d+', '', name, flags=re.I).lower() if file.get('format') == 'ekra-ndr' else Path(name).suffix.lower()
            identity = [(component_slot(f), f['sha256']) for f in components]
            digest = hashlib.sha256(json.dumps([kind, sorted(identity)], separators=(',', ':')).encode()).hexdigest()
            data = next((f for f in components if Path(f['logical_path']).suffix.lower() == '.dat'), None)
            records.append({'record_id': digest, 'case_id': case['case_id'], 'source_folder': case['source_folder'],
                'kind': kind, 'complete_known_pair': bool(data) if kind == 'comtrade' else None,
                'legacy_md5': (data or file)['md5'], 'legacy_hash_basis': 'DAT' if data else 'main_file',
                'components': components, 'derivatives': [], 'created_at': now()})
            used.update(f['logical_path'].casefold() for f in components)
        # Непривязанный DAT/D01 не объявляется полноценной записью.
        case_index = {'case_id': case['case_id'], 'records': [r for r in records if r['case_id'] == case['case_id']],
            'unpaired_components': [f for k, f in files.items() if k not in used and f.get('role') == 'recording_component']}
        metadata = Path(case.get('metadata_root',Path(case['input_root']).parent))
        write_json(metadata / 'record_index.json', case_index)
    return {'schema_version': '0.1', 'source_root': provenance['source_root'], 'records': records,
            'matching_policy': 'all exact matches retained; transformed bytes require registered lineage'}


def write_legacy_hash_table(index, output):
    """Выдать старый формат только для однозначных MD5, отдельно сохранить конфликты."""
    buckets = {}
    for record in index['records']:
        buckets.setdefault(record['legacy_md5'], []).append(record)
    compatible, conflicts = {}, {}
    for digest, matches in buckets.items():
        if len(matches) != 1:
            conflicts[digest] = [{'case_id': r['case_id'], 'record_id': r['record_id']} for r in matches]
            continue
        main = matches[0]['components'][0]
        compatible[digest] = [Path(main['logical_path']).name, main['prepared_path']]
    write_json(Path(output) / '_hash_table.json', compatible)
    write_json(Path(output) / 'legacy_hash_conflicts.json', conflicts)
    return {'unambiguous': len(compatible), 'conflicts': len(conflicts)}


def register_derivative(index_path, record_id, paths, operation, parameters=None, case_id=None):
    """Зарегистрировать результат обработки сразу, пока его происхождение известно."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding='utf-8'))
    matches = [r for r in index['records'] if r['record_id'] == record_id and (case_id is None or r['case_id'] == case_id)]
    if not matches or not operation.strip():
        raise ValueError('Нужны существующий ID записи и описание преобразования')
    derivative = {'operation': operation, 'parameters': parameters or {}, 'created_at': now(), 'files': []}
    for path in paths:
        path = Path(path).resolve(strict=True)
        derivative['files'].append({'name': path.name, 'path_at_registration': str(path), 'sha256': sha256(path),
                                    'size': path.stat().st_size})
    if not derivative['files']:
        raise ValueError('Пустой набор результатов')
    for match in matches:
        match['derivatives'].append(derivative)
    write_json(index_path, index)
    for cid in {m['case_id'] for m in matches}:
        local_path = index_path.parent / 'cases' / cid / 'record_index.json'
        if local_path.exists():
            local = json.loads(local_path.read_text(encoding='utf-8'))
            local['records'] = [r for r in index['records'] if r['case_id'] == cid]
            write_json(local_path, local)
    return derivative


def match_flat(index_path, flat_root):
    """Составить план возврата без записи и без угадывания по одному имени."""
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    hashes = {}
    for record in index['records']:
        for file in record['components']:
            hashes.setdefault(file['sha256'], []).append({'case_id': record['case_id'], 'record_id': record['record_id'],
                'source_folder': record['source_folder'], 'target_name': file['logical_path'], 'kind': 'original_bytes'})
        for i, derivative in enumerate(record['derivatives']):
            for file in derivative['files']:
                hashes.setdefault(file['sha256'], []).append({'case_id': record['case_id'], 'record_id': record['record_id'],
                    'source_folder': record['source_folder'],
                    'target_name': f'derived/{record["record_id"]}/{i}/{file["name"]}', 'kind': 'registered_derivative'})
    rows = []
    flat_root = Path(flat_root).resolve(strict=True)
    for path in sorted(flat_root.rglob('*')):
        if path.is_file() and not path.is_symlink():
            digest = sha256(path)
            matches = hashes.get(digest, [])
            signatures = {(m['case_id'], m['record_id']) for m in matches}
            rows.append({'path': str(path), 'sha256': digest, 'matches': matches,
                         'status': 'unknown' if not matches else 'matched' if len(signatures) == 1 else 'multiple_matches_review'})
    return {'source_root': index['source_root'], 'flat_root': str(flat_root), 'files': rows, 'created_at': now(), 'operation': 'plan_only'}


def restore_flat(plan, output_root, include_multiple=False):
    """Копировать совпавшие байты в новый каталог; исходные файлы не перемещать."""
    output = Path(output_root).resolve()
    source = Path(plan['source_root']).resolve()
    flat = Path(plan['flat_root']).resolve() if plan.get('flat_root') else None
    if (output.is_relative_to(source) or source.is_relative_to(output) or output.exists() or
        (flat and (output.is_relative_to(flat) or flat.is_relative_to(output)))):
        raise ValueError('Возврат выполняется в новый каталог вне исходного корпуса')
    output.mkdir(parents=True)
    copied = []
    for row in plan['files']:
        if row['status'] != 'matched' and not (include_multiple and row['status'] == 'multiple_matches_review'):
            continue
        original = Path(row['path'])
        if sha256(original) != row['sha256']:
            raise ValueError('Файл изменился после построения плана')
        for match in row['matches']:
            target = (output / match.get('source_folder', match['case_id']) / match['target_name']).resolve()
            if not target.is_relative_to(output):
                raise ValueError('Путь из индекса выходит за каталог результата')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != row['sha256']:
                raise FileExistsError(target)
            if not target.exists():
                shutil.copy2(original, target)
            copied.append(str(target))
    write_json(output / 'restore_manifest.json', {'files': copied, 'created_at': now(), 'multiple_matches_included': include_multiple})
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--flat-root', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    write_json(Path(args.plan), match_flat(args.index, args.flat_root))


if __name__ == '__main__':
    main()
