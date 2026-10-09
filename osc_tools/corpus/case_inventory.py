"""Учёт папок комплектов без изменения исходников и планы их подготовки.

Запуск из корня проекта:
  python -m osc_tools.corpus.case_inventory scan --root CORPUS --output OUT.json
  python -m osc_tools.corpus.case_inventory plan --root CORPUS --output PLAN.json
Обе команды только читают исходники: не перемещают файлы, не распаковывают
архивы, не анализируют сигналы и не вызывают языковую модель.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from osc_tools.io.format_detection import UnsupportedFormatError, detect_format

DOCUMENTS = {'.doc', '.docx', '.dot', '.dotx', '.pdf', '.rtf'}
IMAGES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.svg'}
ARCHIVES = {'.zip', '.7z', '.rar'}


def sha256(path: Path) -> str:
    """Вычислить хеш файла порциями, без загрузки всего содержимого в память."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def classify(name: str) -> dict:
    """Предположить роль файла по имени; его содержимое здесь не проверяется."""
    p = Path(name)
    ext = p.suffix.lower()
    if p.name.startswith('~$'):
        return {'role': 'temporary_office_file'}
    try:
        kind = detect_format(p)
        component = ext == '.dat' or bool(re.fullmatch(r'\.d0\d+', ext))
        return {'role': 'recording_component' if component else 'recording_candidate',
                'format': kind, 'read_status': 'not_read'}
    except UnsupportedFormatError:
        pass
    if ext in DOCUMENTS:
        return {'role': 'document', 'read_status': 'not_read'}
    if ext in IMAGES:
        return {'role': 'image', 'read_status': 'not_viewed'}
    if ext in ARCHIVES:
        return {'role': 'archive'}
    if ext in {'.hdr', '.inf', '.xml'} or p.name.lower().startswith('wconfig.'):
        return {'role': 'possible_companion'}
    return {'role': 'other'}


def safe_member(name: str) -> bool:
    """Проверить, не выходит ли имя элемента архива за относительный каталог."""
    value = name.replace('\\', '/')
    p = PurePosixPath(value)
    return bool(value) and not p.is_absolute() and '..' not in p.parts and not re.match(r'^[A-Za-z]:', value)


def archive_listing(path: Path, max_members: int = 2000) -> dict:
    """Получить только список элементов архива, без распаковки и чтения записей."""
    try:
        entries = []
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                info = archive.infolist()
                for item in info[:max_members]:
                    if item.is_dir():
                        continue
                    entries.append({'name': item.filename, 'size': item.file_size,
                        'safe_path': safe_member(item.filename),
                        'encrypted': bool(item.flag_bits & 1),
                        'symlink': stat.S_ISLNK(item.external_attr >> 16),
                        **classify(item.filename)})
        elif path.suffix.lower() == '.7z':
            import py7zr  # Необязательная библиотека; её отсутствие сохраняется в результате.
            with py7zr.SevenZipFile(path, mode='r') as archive:
                info = archive.list()
                for item in info[:max_members]:
                    if not item.is_directory:
                        entries.append({'name': item.filename, 'size': item.uncompressed,
                            'safe_path': safe_member(item.filename), **classify(item.filename)})
        else:
            import rarfile  # Чтение списка не означает распаковку содержимого.
            with rarfile.RarFile(path) as archive:
                info = archive.infolist()
                for item in info[:max_members]:
                    if not item.isdir():
                        entries.append({'name': item.filename, 'size': item.file_size,
                            'safe_path': safe_member(item.filename), **classify(item.filename)})
        return {'status': 'listed' if len(info) <= max_members else 'partially_listed',
                'members': entries, 'total_entries': len(info),
                'content_read': False, 'volume_completeness': 'not_verified'}
    except ImportError as exc:
        return {'status': 'reader_unavailable', 'error': str(exc), 'members': []}
    except Exception as exc:
        return {'status': 'listing_failed', 'error': f'{type(exc).__name__}: {exc}', 'members': []}


def folder_inventory(folder: Path, max_archive_members: int = 2000) -> dict:
    """Учесть файлы папки, кандидаты записей, дубли и ограничения обхода."""
    folder = folder.resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    entries, errors = [], []
    for directory, dirs, files in os.walk(folder, followlinks=False,
            onerror=lambda exc: errors.append(str(exc))):
        current = Path(directory)
        for name in list(dirs):
            if (current / name).is_symlink() or not (current/name).resolve().is_relative_to(folder):
                dirs.remove(name)
                errors.append(f'Skipped directory link: {(current/name).relative_to(folder)}')
        for name in sorted(files):
            path = current / name
            item = {'path': path.relative_to(folder).as_posix(), **classify(name)}
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(folder):
                    raise ValueError('File link is outside the inventory scope')
                item.update(size=path.stat().st_size, sha256=sha256(path))
                if item['role'] == 'archive':
                    item['listing'] = archive_listing(path, max_archive_members)
            except (OSError, ValueError) as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
            entries.append(item)
    entries.sort(key=lambda item: item['path'])
    hashes = defaultdict(list)
    for item in entries:
        if 'sha256' in item:
            hashes[item['sha256']].append(item['path'])
    names = {item['path'].casefold() for item in entries}
    for item in entries:
        p = Path(item['path'])
        if item.get('format') == 'comtrade' and p.suffix.lower() in {'.cfg', '.dat'}:
            companion = p.with_suffix('.dat' if p.suffix.lower() == '.cfg' else '.cfg')
            item['companion_status'] = 'found_by_name' if companion.as_posix().casefold() in names else 'missing_by_name'
        elif re.fullmatch(r'\.d0\d+', p.suffix.lower()):
            item['main_status'] = 'found_by_name' if p.with_suffix('.do').as_posix().casefold() in names else 'missing_by_name'
    visible = any(item['role'] in {'recording_candidate','recording_component'} for item in entries)
    uncertain = bool(errors) or any('error' in item for item in entries)
    for item in entries:
        if item['role'] == 'archive':
            listing = item.get('listing', {})
            uncertain |= listing.get('status') != 'listed'
            visible |= any(member['role'] in {'recording_candidate','recording_component'}
                for member in listing.get('members', []))
    return {'files': entries, 'physical_file_counts': dict(Counter(item['role'] for item in entries)),
            'recording_presence': 'candidates_found' if visible else ('unknown' if uncertain else 'none_detected'),
            'recording_content_read': False, 'images_viewed': False,
            'byte_duplicate_groups': [v for v in hashes.values() if len(v) > 1],
            'archive_loose_duplicates': 'not_verified_without_member_content_hashes',
            'errors': errors}


def inventory_corpus(root: Path, registry: dict | None = None) -> dict:
    """Обойти структуру «год → папка», сохраняя уже назначенные ID комплектов."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    registry = dict(registry or {})
    ids = list(registry.values())
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'case_\d{6}', x) for x in ids):
        raise ValueError('Invalid or duplicate case IDs in registry')
    next_id = max((int(x.removeprefix('case_')) for x in ids), default=0) + 1
    cases, loose, excluded = [], [], []
    for year in sorted(root.iterdir()):
        if not year.is_dir() or year.is_symlink() or not year.resolve().is_relative_to(root) or not re.fullmatch(r'\d{4}', year.name):
            excluded.append(year.name)
            continue
        for child in sorted(year.iterdir()):
            key = child.relative_to(root).as_posix()
            if child.is_symlink() or not child.resolve().is_relative_to(root):
                loose.append({'path': key, 'status': 'link_not_processed'})
            elif child.is_file():
                loose.append({'path': key, 'status': 'needs_folder_preparation', **classify(child.name)})
            elif child.is_dir():
                if key not in registry:
                    if next_id > 999999:
                        raise ValueError('Case ID space exhausted')
                    registry[key] = f'case_{next_id:06d}'
                    next_id += 1
                cases.append({'case_id': registry[key], 'source_folder': key,
                    'report_filename': registry[key]+'__report.md', **folder_inventory(child)})
    return {'artifact_type': 'case_inventory', 'schema_version': '0.1', 'source_root': str(root), 'case_registry': registry,
            'cases': cases, 'unprepared_year_files': loose, 'excluded_root_entries': excluded,
            'scope': 'Inventory only; not a document analysis or signal validation.'}


def normalization_plan(root: Path) -> dict:
    """Подготовить план размещения отдельных документов по папкам, без переноса."""
    root = root.resolve(strict=True)
    groups = []
    for year in sorted(root.iterdir()):
        if not year.is_dir() or year.is_symlink() or not year.resolve().is_relative_to(root) or not re.fullmatch(r'\d{4}', year.name):
            continue
        stems = defaultdict(list)
        for item in year.iterdir():
            if item.is_file() and not item.is_symlink():
                stems[item.stem].append(item)
        for stem, files in sorted(stems.items()):
            target = year / stem
            state = 'ready' if all(p.suffix.lower() in DOCUMENTS and not p.name.startswith('~$') for p in files) else 'needs_review'
            if target.exists():
                state = 'target_exists'
            elif not stem or stem.endswith((' ', '.')):
                # Windows может незаметно изменить имя папки с конечным пробелом или точкой.
                state = 'needs_review'
            groups.append({'target': target.relative_to(root).as_posix(), 'status': state,
                'files': [{'source': p.relative_to(root).as_posix(), 'size': p.stat().st_size,
                           'sha256': sha256(p)} for p in sorted(files)]})
    return {'artifact_type': 'normalization_plan', 'schema_version': '0.1', 'source_root': str(root), 'groups': groups,
            'applied': False, 'rule': 'Exact stems only; non-document groups require review.'}


def apply_normalization_plan(root: Path, plan: dict) -> None:
    """Применить явно переданный план после проверки путей и хешей.

    Команды CLI эту функцию не вызывают. При ошибке выполненные переносы
    откатываются; удаляются только созданные этим вызовом пустые папки.
    """
    root = root.resolve(strict=True)
    if str(root) != plan['source_root']:
        raise ValueError('Plan belongs to another corpus')
    operations, seen = [], set()
    for group in plan['groups']:
        if group['status'] != 'ready':
            continue
        target = (root/group['target']).resolve()
        if not target.is_relative_to(root) or target.exists() or target in seen or Path(group['target']).name.endswith((' ', '.')):
            raise ValueError('Unsafe, duplicate or occupied target')
        seen.add(target)
        if target.parent.parent != root or not re.fullmatch(r'\d{4}', target.parent.name):
            raise ValueError('Target is not a second-level case folder')
        for item in group['files']:
            source = root/item['source']
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError('Source link or out-of-scope path')
            if source.parent.resolve() != target.parent or source.suffix.lower() not in DOCUMENTS or source.stem != target.name or source.name.startswith('~$'):
                raise ValueError('Unexpected source in normalization plan')
            if source.stat().st_size != item['size'] or sha256(source) != item['sha256']:
                raise ValueError('Source changed after planning')
            if source in seen:
                raise ValueError('Repeated source')
            seen.add(source)
            operations.append((source, target/source.name))
    created, moved = [], []
    try:
        for source, destination in operations:
            if destination.parent not in created:
                destination.parent.mkdir()
                created.append(destination.parent)
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if not source.exists():
                destination.rename(source)
        for folder in reversed(created):
            folder.rmdir()  # Только созданные этим вызовом папки, пустые после отката.
        raise
    plan['applied'] = bool(moved)
    plan['applied_groups'] = [folder.relative_to(root).as_posix() for folder in created]


def main() -> None:
    """Запустить учёт или подготовку плана и сохранить JSON вне исходного корпуса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['scan', 'plan'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(root):
        parser.error('Keep generated outputs outside the source corpus')
    registry = None
    if output.exists():
        old = json.loads(output.read_text(encoding='utf-8'))
        if old.get('source_root') != str(root) or old.get('artifact_type') != ('case_inventory' if args.action == 'scan' else 'normalization_plan'):
            parser.error('Existing output belongs to another corpus')
        registry = old.get('case_registry')
    result = inventory_corpus(root, registry) if args.action == 'scan' else normalization_plan(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    if temporary.exists():
        parser.error('Temporary output already exists')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'output': str(output), 'cases': len(result.get('cases', [])),
        'unprepared_year_files': len(result.get('unprepared_year_files', [])),
        'normalization_groups': len(result.get('groups', []))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
