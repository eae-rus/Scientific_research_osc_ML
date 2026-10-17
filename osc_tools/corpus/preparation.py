"""Подготовка корня корпуса без ИИ и без изменения первичных файлов.

Исходники учитываются отдельно от рабочих копий. Побайтовые дубли используют
общее хранилище SHA256. Архивы извлекаются выборочно с границами размера/пути.
MD5 сохранён для совместимости с прежним SearchOscillograms, но не заменяет SHA256.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import threading
import zipfile

from .agent_runtime import now, prepare, write_json
from .case_inventory import classify, folder_inventory, inventory_corpus, normalization_plan, safe_member, sha256
from .document_reading import SEVEN_ZIP


@dataclass
class PreparationConfig:
    """Параметры одного повторяемого прохода; выход всегда вне исходного корня."""
    source_root: str
    output_root: str
    seed_inventory: str | None = None
    years: tuple[str, ...] = ()
    extract_archives: bool = True
    max_member_bytes: int = 128 * 1024 * 1024
    max_total_extracted_bytes: int = 512 * 1024 * 1024
    max_archive_members: int = 5000
    max_archive_depth: int = 1
    seven_zip: str | None = None
    write_source_indexes: bool = False
    blob_root: str | None = None


def md5(path):
    """Вычислить совместимый MD5 порциями, не загружая тяжёлый файл в память."""
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relevant(name):
    """Отбирать материалы по техническому типу, не приписывая им смысловую важность."""
    role = classify(name)['role']
    extension = Path(name).suffix.lower()
    return role in {'document', 'image', 'recording_candidate', 'recording_component', 'possible_companion', 'archive'} or extension in {
        '.txt', '.log', '.vsd', '.vsdx', '.evt', '.psl', '.datapak', '.dg4', '.mix',
        '.dex', '.osc', '.db', '.sqlite', '.mdb', '.dcf', '.all', '.ini'}


def member_safe(name):
    """Проверить архивный путь, в том числе особенности имён Windows."""
    parts = PurePosixPath(name.replace('\\', '/')).parts
    return safe_member(name) and all(not (':' in p or p.endswith((' ', '.')) or
        re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', p, re.I)) for p in parts)


class CorpusPreparer:
    """Собирать подготовленные папки и индексы происхождения в новом запуске."""
    def __init__(self, config):
        self.config = config
        self.source = Path(config.source_root).resolve(strict=True)
        self.output = Path(config.output_root).resolve()
        if not self.source.is_dir() or self.output.is_relative_to(self.source) or self.source.is_relative_to(self.output):
            raise ValueError('Выход должен быть отдельно от исходного корня')
        self.blob_root = Path(config.blob_root).resolve() if config.blob_root else self.output / 'blobs'
        if self.blob_root.is_relative_to(self.source) or self.source.is_relative_to(self.blob_root):
            raise ValueError('Хранилище blob должно быть вне исходного корня')
        if any(not re.fullmatch(r'\d{4}', year) for year in config.years):
            raise ValueError('Годы задаются четырьмя цифрами')
        if min(config.max_member_bytes, config.max_total_extracted_bytes, config.max_archive_members) <= 0:
            raise ValueError('Лимиты должны быть положительными')
        if config.max_archive_depth not in range(0, 4):
            raise ValueError('Допустимая глубина архивов 0–3')
        self.seven = config.seven_zip or shutil.which('7zz') or shutil.which('7z') or str(SEVEN_ZIP)
        self.extracted_bytes = 0
        self.blobs, self.cases, self.archives = {}, [], []

    def store(self, source, input_root, logical_path, origin):
        """Сохранить один blob и связать его с исходным логическим именем."""
        logical_path = logical_path.replace('\\', '/')
        if not member_safe(logical_path):
            raise ValueError('Небезопасное логическое имя')
        source = Path(source)
        digest = sha256(source)
        size = source.stat().st_size
        entry = {'logical_path': logical_path, 'sha256': digest, 'md5': md5(source), 'size': size,
                 'origin': origin, **classify(logical_path)}
        if size == 0:
            return {**entry, 'preparation_status': 'zero_byte_not_copied'}
        if source.suffix.lower() in {'.txt', '.log'} and size < 64 * 1024:
            raw = source.read_bytes()
            encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            try:
                decoded = raw.decode(encoding)
            except UnicodeDecodeError:
                decoded = raw.decode('cp1251', errors='replace')
            entry['text_empty'] = not decoded.strip(' \r\n\t\x00')
        blob = self.blob_root / digest[:2] / digest
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, blob)
            if sha256(blob) != digest:
                raise ValueError('Байты копии не совпали')
        elif sha256(blob) != digest:
            raise ValueError('Хранимый blob изменён; повторное использование запрещено')
        self.blobs[digest] = size
        target = input_root / logical_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise FileExistsError('Конфликт логического имени')
            method = 'existing_same_bytes'
        else:
            try:
                os.link(blob, target)
                method = 'hardlink_to_workspace_blob'
            except OSError:
                shutil.copy2(blob, target)
                method = 'copy_fallback'
        return {**entry, 'prepared_path': str(target), 'preparation_status': method,
                'storage_blob': str(blob), 'agent_content_read': False}

    def listing(self, path):
        """Получить список ZIP либо 7-Zip, без чтения содержимого элементов."""
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                return [{'name': f.filename, 'size': f.file_size, 'directory': f.is_dir(),
                         'encrypted': bool(f.flag_bits & 1), 'link': stat.S_ISLNK(f.external_attr >> 16)}
                        for f in archive.infolist()]
        completed = subprocess.run([self.seven, 'l', '-slt', '-ba', '-sccUTF-8', '--', str(path)],
                                   capture_output=True, timeout=60, check=True)
        items = []
        for block in re.split(r'\r?\n\s*\r?\n', completed.stdout.decode('utf-8-sig', errors='replace')):
            fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
            if 'Path' not in fields:
                continue
            items.append({'name': fields['Path'], 'size': int(fields.get('Size', '0')),
                'directory': fields.get('Folder') == '+' or fields.get('Attributes', '').startswith('D'),
                'encrypted': fields.get('Encrypted') == '+',
                'link': 'Symbolic Link' in fields or 'Hard Link' in fields})
        return items

    def extract_one(self, path, member, temporary):
        """Извлечь один явно названный элемент с лимитом фактически полученных байтов."""
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive, archive.open(member['name']) as stream, temporary.open('wb') as out:
                total = 0
                while chunk := stream.read(128 * 1024):
                    total += len(chunk)
                    if total > self.config.max_member_bytes or self.extracted_bytes + total > self.config.max_total_extracted_bytes:
                        raise ValueError('Превышен лимит извлечения')
                    out.write(chunk)
        else:
            process = subprocess.Popen([self.seven, 'e', '-so', '-bd', '-y', '-spd', '--', str(path), member['name']],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            total = 0
            timeout = threading.Timer(60, process.kill)
            timeout.start()
            try:
                with temporary.open('wb') as out:
                    while chunk := process.stdout.read(128 * 1024):
                        total += len(chunk)
                        if total > self.config.max_member_bytes or self.extracted_bytes + total > self.config.max_total_extracted_bytes:
                            raise ValueError('Превышен лимит извлечения')
                        out.write(chunk)
                if process.wait(timeout=60):
                    raise RuntimeError('7-Zip не смог извлечь элемент')
            finally:
                timeout.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if total != member['size']:
            raise ValueError('Размер извлечения не совпал со списком архива')
        self.extracted_bytes += total

    def archive(self, path, input_root, origin, depth=0):
        """Перечислить контейнер и извлечь выбранные безопасные элементы отдельно."""
        digest = sha256(path)
        audit = {'path': str(path), 'sha256': digest, 'origin': origin, 'depth': depth,
                 'status': 'pending', 'members': [], 'volume_completeness': 'not_verified'}
        self.archives.append(audit)
        entries = []
        try:
            if re.search(r'\.part(?!0*1\.)\d+\.rar$', path.name, re.I):
                audit['status'] = 'volume_processed_via_first_part'
                return entries
            members = self.listing(path)
            audit['total_members'] = len(members)
            if len(members) > self.config.max_archive_members:
                audit['status'] = 'member_limit_exceeded'
                return entries
            audit['status'] = 'listed'
            for i, member in enumerate(members):
                item = {**member, 'selected_by_type': relevant(member['name']), 'status': 'listed_not_read'}
                audit['members'].append(item)
                if member['directory']:
                    item['status'] = 'directory'
                    continue
                if not member_safe(member['name']) or member['link'] or member['encrypted']:
                    item['status'] = 'unsafe_or_encrypted_skipped'
                    continue
                if not self.config.extract_archives or not item['selected_by_type']:
                    continue
                if member['size'] > self.config.max_member_bytes or self.extracted_bytes + member['size'] > self.config.max_total_extracted_bytes:
                    item['status'] = 'size_limit_skipped'
                    continue
                temporary = self.output / 'temporary' / f'{digest}_{i}{Path(member["name"]).suffix}'
                temporary.parent.mkdir(exist_ok=True)
                try:
                    self.extract_one(path, member, temporary)
                    member_origin = {**origin, 'archive_sha256': digest, 'archive_path': str(path), 'member': member['name'],
                        'archive_chain': origin.get('archive_chain', []) + [{'sha256': digest, 'member': member['name']}]}
                    if classify(member['name'])['role'] == 'archive':
                        if depth < self.config.max_archive_depth:
                            entries += self.archive(temporary, input_root, member_origin, depth + 1)
                            item['status'] = 'nested_archive_processed'
                        else:
                            item['status'] = 'depth_limit_skipped'
                    else:
                        logical = f'_archives/{digest[:16]}/{member["name"]}'
                        saved = self.store(temporary, input_root, logical, member_origin)
                        entries.append(saved)
                        item.update(status='extracted', sha256=saved['sha256'], prepared_path=saved.get('prepared_path'))
                except Exception as exc:
                    item.update(status='extraction_failed', error=f'{type(exc).__name__}: {exc}')
                finally:
                    temporary.unlink(missing_ok=True)
        except Exception as exc:
            audit.update(status='listing_failed', error=f'{type(exc).__name__}: {exc}')
        return entries

    def run(self):
        """Подготовить существующие папки и группы одиночных документов корня года."""
        seed = json.loads(Path(self.config.seed_inventory).read_text(encoding='utf-8-sig')) if self.config.seed_inventory else {}
        registry = dict(seed.get('case_registry', {}))
        registry_aliases = dict(seed.get('registry_aliases', {}))
        # Старое резервирование рабочей копии и новый переносимый ключ — один ID,
        # но основной реестр допускает только один ключ на ID.
        for key, value in list(registry.items()):
            duplicates = [k for k, v in registry.items() if v == value]
            if len(duplicates) > 1:
                if not all(k.startswith(('__prepared__/', '__loose__/')) for k in duplicates):
                    raise ValueError('Разные исходные папки имеют один ID')
                canonical = next((k for k in duplicates if k.startswith('__loose__/')), duplicates[0])
                for duplicate in duplicates:
                    if duplicate != canonical:
                        registry_aliases[duplicate] = registry.pop(duplicate)
        inventory = inventory_corpus(self.source, registry)
        registry = inventory['case_registry']
        plan = normalization_plan(self.source)
        selected = lambda year: not self.config.years or year in self.config.years
        cases = [(c['case_id'], c['source_folder'], [(self.source / c['source_folder'] / f['path'], f['path'])
                 for f in c['files'] if 'sha256' in f]) for c in inventory['cases'] if selected(c['source_folder'].split('/')[0])]
        reserved = {}
        loose_cases = []
        for case in seed.get('prepared_cases', []):
            if case.get('primary_document'):
                original = Path(case['primary_document'])
                if original.is_relative_to(self.source):
                    reserved[original.relative_to(self.source).as_posix()] = case['case_id']
        for group in plan['groups']:
            if not selected(group['target'].split('/')[0]) or not all(classify(f['source'])['role'] == 'document' for f in group['files']):
                continue
            key = '__loose__/' + group['target']
            case_id = registry.get(key) or reserved.get(group['files'][0]['source'])
            if not case_id:
                case_id = f"case_{max((int(v[5:]) for v in registry.values()), default=0) + 1:06d}"
                registry[key] = case_id
            else:
                for old_key in [k for k,v in registry.items() if v == case_id and k != key]:
                    registry_aliases[old_key] = registry.pop(old_key)
                registry[key] = case_id
            cases.append((case_id, group['target'], [(self.source / f['source'], Path(f['source']).name) for f in group['files']]))
            loose_cases.append({'case_id': case_id, 'primary_document': str(self.source / group['files'][0]['source']),
                                'source_folder': group['target']})
        if len({c[0] for c in cases}) != len(cases):
            raise ValueError('Неоднозначный постоянный ID')
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / 'preparation_config.json', asdict(self.config))
        initial_hashes = {}
        for case_id, source_folder, files in cases:
            input_root = self.output / 'cases' / case_id / 'input'
            input_root.mkdir(parents=True)
            entries = []
            for path, logical in files:
                if path.is_symlink() or not path.resolve().is_relative_to(self.source):
                    raise ValueError('Исходная ссылка выходит за пределы корпуса')
                origin = {'source_path': str(path), 'source_relative_path': path.relative_to(self.source).as_posix()}
                own_index = path.name in {'_phase6_provenance.json', '_phase6_record_index.json'}
                if not own_index:
                    initial_hashes[str(path)] = sha256(path)
                if classify(path.name)['role'] == 'archive':
                    entries += self.archive(path, input_root, origin)
                elif path.name.casefold() == 'thumbs.db' or path.suffix.lower() == '.waves' or path.name.startswith('~$') or own_index:
                    entries.append({'logical_path': logical, 'origin': origin, 'sha256': sha256(path),
                                    'size': path.stat().st_size, 'preparation_status': 'cache_or_office_temp_excluded'})
                else:
                    entries.append(self.store(path, input_root, logical, origin))
            case = {'case_id': case_id, 'source_folder': source_folder, 'input_root': str(input_root), 'files': entries,
                    'source_files_not_modified': True, 'semantic_importance': 'not_assessed'}
            write_json(input_root.parent / 'provenance.json', case)
            self.cases.append(case)
            if self.config.write_source_indexes and (self.source / source_folder).is_dir():
                write_json(self.source / source_folder / '_phase6_provenance.json', case)
        if not all(sha256(Path(p)) == h for p, h in initial_hashes.items()):
            raise ValueError('Исходные байты изменились во время подготовки')
        prepared_inventory = {'schema_version': '0.2', 'source_root': str(self.output), 'case_registry': registry,
            'cases': [{'case_id': c['case_id'], 'source_folder': f'cases/{c["case_id"]}/input'} for c in self.cases]}
        write_json(self.output / 'original_inventory.json', {**inventory, 'case_registry': registry,
            'registry_aliases': registry_aliases, 'prepared_cases': loose_cases})
        write_json(self.output / 'prepared_inventory.json', prepared_inventory)
        write_json(self.output / 'archive_manifest.json', self.archives)
        write_json(self.output / 'normalization_plan.json', plan)
        index = {'schema_version': '0.1', 'source_root': str(self.source), 'created_at': now(), 'cases': self.cases}
        write_json(self.output / 'provenance_index.json', index)
        from .record_identity import build_record_index, write_legacy_hash_table
        records = build_record_index(index)
        write_json(self.output / 'record_index.json', records)
        legacy = write_legacy_hash_table(records, self.output)
        if self.config.write_source_indexes:
            for case in self.cases:
                folder = self.source / case['source_folder']
                if folder.is_dir():
                    write_json(folder / '_phase6_record_index.json', {'case_id': case['case_id'],
                        'records': [r for r in records['records'] if r['case_id'] == case['case_id']]})
        summary = {'status': 'prepared', 'cases': len(self.cases), 'unique_blobs': len(self.blobs),
                   'unique_blob_bytes': sum(self.blobs.values()), 'logical_bytes': sum(f.get('size', 0) for c in self.cases for f in c['files'] if f.get('prepared_path')),
                   'extracted_bytes': self.extracted_bytes, 'archives': len(self.archives), 'source_files_hash_checked': len(initial_hashes),
                   'source_files_unchanged': True, 'archive_issues': sum(a['status'] not in {'listed','volume_processed_via_first_part'} for a in self.archives),
                   'archive_member_issues': sum(m['status'] in {'extraction_failed','unsafe_or_encrypted_skipped','size_limit_skipped','depth_limit_skipped'}
                       for a in self.archives for m in a['members']),
                   'record_candidates': len(records['records']), 'output': str(self.output), 'created_at': now()}
        summary['legacy_hash_table'] = legacy
        summary['source_indexes_written'] = self.config.write_source_indexes
        write_json(self.output / 'summary.json', summary)
        return summary


def prepare_agents(prepared_root, instruction, vocabulary, case_ids=None, run_name='agent_run_001', reuse_existing=False):
    """Создать отдельные рабочие контексты агента поверх подготовленного набора."""
    root = Path(prepared_root).resolve(strict=True)
    inventory = json.loads((root / 'prepared_inventory.json').read_text(encoding='utf-8'))
    selected = set(case_ids) if case_ids else {c['case_id'] for c in inventory['cases']}
    if selected - {c['case_id'] for c in inventory['cases']}:
        raise ValueError('Запрошен ID, отсутствующий в подготовленном наборе')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('Некорректное имя запуска')
    # agent_runtime требует выход вне source_root; используем соседний каталог запусков.
    destination = root.parent / (root.name + '__' + run_name)
    contexts = []
    for case in inventory['cases']:
        if case['case_id'] not in selected:
            continue
        workspace = destination / case['case_id']
        if workspace.exists() and reuse_existing:
            manifest = json.loads((workspace / 'manifest.json').read_text(encoding='utf-8'))
            source = (root / case['source_folder']).resolve(strict=True)
            fresh = folder_inventory(source)
            signature = lambda files: sorted((f['path'], f.get('sha256'), f.get('size')) for f in files)
            if (Path(manifest['source_root']).resolve() != source or signature(manifest['files']) != signature(fresh['files']) or
                manifest['instruction_sha256'] != sha256(Path(instruction)) or manifest['vocabulary_sha256'] != sha256(Path(vocabulary))):
                raise ValueError('Вход/инструкция/словарь изменились: создайте новый run_name')
            contexts.append({'case_id': case['case_id'], 'workspace': str(workspace), 'status': 'existing_context'})
        else:
            contexts.append(prepare(root / 'prepared_inventory.json', case['case_id'], workspace, instruction, vocabulary))
    return contexts


def main():
    """Запустить конфигурируемую подготовку корня корпуса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    print(json.dumps(CorpusPreparer(PreparationConfig(**config)).run(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
