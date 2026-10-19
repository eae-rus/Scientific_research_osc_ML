"""Поиск и копирование записей с точными именами и выборочным чтением архивов."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from osc_tools.core.constants import TYPE_OSC
from osc_tools.io.archive_reader import ArchiveReader, ARCHIVE_SUFFIXES, safe_archive_member


def digest_file(path, algorithm='sha256'):
    """Вычислить хеш порциями, не загружая осциллограмму в память целиком."""
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def recording_type(name):
    """Распознать только расширение/полное имя; подстроки .do/.os не подходят."""
    path = Path(name)
    suffix = path.suffix.lower()
    fixed = {'.cfg':TYPE_OSC.COMTRADE_CFG_DAT, '.cff':TYPE_OSC.COMTRADE_CFF,
        '.brs':TYPE_OSC.BRESLER,'.dfr':TYPE_OSC.EKRA,'.bb':TYPE_OSC.BLACK_BOX,
        '.sg2':TYPE_OSC.RES_3,'.osc':TYPE_OSC.OSC,'.dex':TYPE_OSC.DEX,
        '.do':TYPE_OSC.PARMA,'.to':TYPE_OSC.PARMA_TO}
    if suffix in fixed:
        return fixed[suffix]
    if re.fullmatch(r'\.d0\d+',suffix):
        return TYPE_OSC.PARMA
    if re.fullmatch(r'\.t0\d+',suffix):
        return TYPE_OSC.PARMA_TO
    if re.fullmatch(r'\.os\d*',suffix):
        return TYPE_OSC.NEVA
    if re.fullmatch(r'DR\d+F[0-3]\.\d+',path.name,re.I):
        return TYPE_OSC.EKRA_NDR
    return None


class SafeOscillogramSearch:
    """Копировать кандидатов записей; не исполнять файлы и не анализировать каналы."""
    CFG_EXTENSION, DAT_EXTENSION = '.cfg', '.dat'

    def __init__(self, seven_zip=None, max_archive_depth=2, max_member_bytes=128*1024**2,
                 max_total_extracted_bytes=512*1024**2, max_archive_members=5000):
        if not 0 <= max_archive_depth <= 5:
            raise ValueError('Глубина архивов должна быть от 0 до 5')
        self.seven_zip, self.max_archive_depth = seven_zip, max_archive_depth
        self.limits = dict(max_member_bytes=max_member_bytes,max_total_bytes=max_total_extracted_bytes,max_members=max_archive_members)

    def copy_new_oscillograms(self, source_dir, dest_dir, copied_hashes=None, preserve_dir_structure=True,
                             use_hashes=True, types_to_copy=None, _new_copied_hashes=None,
                             _first_run=True, _path_temp=None, progress_callback=None,
                             stop_processing_fn=None, is_write_names=None, **kwargs):
        """Обойти каталог, сохранить совместимый MD5 и полный журнал групп/архивов.

        Пары CFG/DAT и DO/D01, группы NDR с WConfig копируются вместе. DEX —
        непрочитанный контейнер. EXE, SFX и неподходящие элементы не извлекаются.
        Вложенный каталог назначения исключается из обхода исходного дерева.
        """
        source, destination = Path(source_dir).resolve(strict=True), Path(dest_dir).resolve()
        if not source.is_dir() or source == destination or source.is_relative_to(destination):
            raise ValueError('Источник должен быть каталогом, а выход — отдельным каталогом')
        destination.mkdir(parents=True,exist_ok=True)
        hashes = copied_hashes if copied_hashes is not None else {}
        hash_path = destination/'_hash_table.json'
        if use_hashes and hash_path.exists():
            for key,value in json.loads(hash_path.read_text(encoding='utf-8')).items():
                hashes.setdefault(key,value)
        new_hashes = _new_copied_hashes if _new_copied_hashes is not None else {}
        selected = set(types_to_copy or TYPE_OSC)
        reader = ArchiveReader(self.seven_zip,**self.limits)
        manifest_path = destination / '_search_manifest.json'
        previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        records = {r['group_sha256']:r for r in previous.get('records',[])}
        events, count, handled = [], 0, set()

        def notify(message):
            """Сообщить прогресс, если вызывающая программа установила обработчик."""
            if progress_callback:
                progress_callback(message)

        def stopped():
            """Проверить явный сигнал остановки без обращения к пустому callback."""
            return bool(stop_processing_fn and stop_processing_fn())

        def copy_group(path, root, target_base, origin):
            """Связать соседние компоненты и перенести группу без перезаписи байтов."""
            nonlocal count
            kind = recording_type(path.name)
            if kind not in selected or (str(root),str(path)) in handled:
                return
            siblings = {p.name.casefold():p for p in path.parent.iterdir() if p.is_file() and not p.is_symlink()}
            main, components = path, [path]
            if kind == TYPE_OSC.COMTRADE_CFG_DAT:
                data = siblings.get(path.with_suffix('.dat').name.casefold())
                if data is None:
                    events.append({'path':origin,'status':'missing_dat'})
                    return
                components += [data]
                components += [siblings[p] for p in (path.with_suffix(s).name.casefold() for s in ('.hdr','.inf','.xml')) if p in siblings]
            elif kind in {TYPE_OSC.PARMA,TYPE_OSC.PARMA_TO}:
                extension, prefix = ('.do','.d0') if kind == TYPE_OSC.PARMA else ('.to','.t0')
                main = siblings.get(path.with_suffix(extension).name.casefold(),path)
                components = [p for p in siblings.values() if p.stem.casefold() == path.stem.casefold() and
                    (p.suffix.lower() == extension or re.fullmatch(re.escape(prefix)+r'\d+',p.suffix.lower()))]
            elif kind == TYPE_OSC.EKRA_NDR:
                group = re.match(r'(DR\d+)F',path.name,re.I).group(1)
                components = [p for p in siblings.values() if re.fullmatch(re.escape(group)+r'F[0-3]\.\d+',p.name,re.I) or p.name.lower().startswith('wconfig.')]
                main = sorted(components,key=lambda p:p.name.casefold())[0]
            components = sorted(components,key=lambda p:p.name.casefold())
            # Обход может встретить D01 раньше DO; источник MD5 должен указывать
            # на основной файл, а не на случайно первый компонент.
            origin = origin.replace('\\','/').rsplit('/',1)[0]+'/'+main.name
            handled.update((str(root),str(p)) for p in components)
            signature = [(p.name.casefold(),digest_file(p)) for p in components]
            slots = [(re.sub(r'^DR\d+','',p.name,flags=re.I).casefold() if kind == TYPE_OSC.EKRA_NDR else p.suffix.lower(),h)
                     for p,(_,h) in zip(components,signature)]
            group_hash = hashlib.sha256(json.dumps([kind.name,sorted(slots)],separators=(',',':')).encode()).hexdigest()
            prior = records.get(group_hash)
            if use_hashes and prior and all(Path(f['path']).is_file() and digest_file(f['path']) == f['sha256'] for f in prior['outputs']):
                if origin not in prior['origins']:
                    prior['origins'].append(origin)
                hashes.setdefault(prior['legacy_md5'],(prior.get('main_filename',main.name),prior['origins'][0]))
                return
            folder = target_base / kind.name
            if preserve_dir_structure:
                folder /= path.parent.relative_to(root)
            if kind == TYPE_OSC.EKRA_NDR or any((folder/p.name).exists() and digest_file(folder/p.name) != h for p,(_,h) in zip(components,signature)):
                folder /= group_hash[:16]
            folder.mkdir(parents=True,exist_ok=True)
            written, outputs = False, []
            for component,(_,digest) in zip(components,signature):
                target = folder / component.name
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Выход группы вышел за каталог назначения')
                if not target.exists():
                    shutil.copy2(component,target)
                    written = True
                if digest_file(target) != digest:
                    raise ValueError('Байты скопированного компонента не совпали')
                outputs.append({'path':str(target),'sha256':digest})
            basis = next(p for p in components if p.suffix.lower() == '.dat') if kind == TYPE_OSC.COMTRADE_CFG_DAT else main
            legacy = digest_file(basis,'md5')
            if use_hashes:
                hashes.setdefault(legacy,(main.name,origin))
                if written:
                    new_hashes.setdefault(legacy,(main.name,origin))
            records[group_hash] = {'group_sha256':group_hash,'type':kind.name,'legacy_md5':legacy,
                'origins':list(dict.fromkeys((prior or {}).get('origins',[]) + [origin])), 'outputs':outputs,
                'main_filename':main.name,
                'content_read':False,'container_not_parsed':kind == TYPE_OSC.DEX}
            count += int(written)

        def walk(root, target_base, origin_prefix=None, depth=0):
            """Обойти исходники либо явно выбранные вложения одного архива."""
            for directory, dirs, files in os.walk(root,followlinks=False):
                current = Path(directory)
                dirs[:] = sorted(d for d in dirs if not (current/d).is_symlink() and
                    (current/d).resolve().is_relative_to(root) and not (current/d).resolve().is_relative_to(destination))
                for name in sorted(files):
                    if stopped():
                        return
                    path = current/name
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        continue
                    origin = origin_prefix + '!/' + path.relative_to(root).as_posix() if origin_prefix else str(path)
                    if is_write_names:
                        notify('Обработка: ' + origin)
                    if recording_type(name) in selected:
                        copy_group(path,root,target_base,origin)
                    elif path.suffix.lower() in ARCHIVE_SUFFIXES:
                        if depth > self.max_archive_depth:
                            events.append({'path':origin,'status':'archive_depth_limit'})
                            continue
                        if re.search(r'\.part(?!0*1\.)\d+\.rar$',name,re.I):
                            events.append({'path':origin,'status':'volume_via_part1'})
                            continue
                        try:
                            members = reader.list(path)
                            temp_parent = destination / '_search_temp'
                            temp_parent.mkdir(exist_ok=True)
                            with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
                                temporary = Path(temporary).resolve()
                                assert temporary.is_relative_to(destination)
                                for member in members:
                                    member_name = member['name'].replace('\\','/')
                                    suffix = Path(member_name).suffix.lower()
                                    wanted = (recording_type(member_name) in selected or
                                              (suffix == '.dat' and TYPE_OSC.COMTRADE_CFG_DAT in selected) or
                                              suffix in {'.hdr','.inf','.xml'} or
                                              (Path(member_name).name.lower().startswith('wconfig.') and TYPE_OSC.EKRA_NDR in selected) or suffix in ARCHIVE_SUFFIXES)
                                    if not wanted or member['directory']:
                                        continue
                                    try:
                                        reader.extract(path,member,temporary)
                                    except Exception as exc:
                                        events.append({'path':origin+'!/'+member_name,'status':'member_skipped','error':str(exc)})
                                archive_target = target_base / ('archive_' + path.suffix[1:].lower()) / path.relative_to(root)
                                walk(temporary,archive_target,origin,depth+1)
                            events.append({'path':origin,'status':'archive_selected_members_processed'})
                        except Exception as exc:
                            events.append({'path':origin,'status':'archive_failed','error':str(exc)})
                            notify('Архив пропущен: ' + origin + ': ' + str(exc))
        notify('Начинается поиск кандидатов осциллограмм: '+str(source))
        walk(source,destination)
        status = 'stopped' if stopped() else 'completed_with_limits' if any(e['status'] in {'archive_failed','member_skipped','archive_depth_limit','missing_dat'} for e in events) else 'completed'
        self.last_result = {'status':status,'new_groups':count,'records':list(records.values()),'events':events,
                            'extracted_bytes':reader.bytes_read,'source_root':str(source),'signals_read':False}
        legacy_groups = {}
        for record in records.values():
            legacy_groups.setdefault(record['legacy_md5'],[]).append(record['group_sha256'])
        self.last_result['legacy_md5_multiple_groups'] = {h:groups for h,groups in legacy_groups.items() if len(groups)>1}
        manifest_path.write_text(json.dumps(self.last_result,ensure_ascii=False,indent=2),encoding='utf-8')
        if use_hashes:
            (destination/'_hash_table.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            (destination/('new_hash_table_'+stamp+'.json')).write_text(json.dumps(new_hashes,ensure_ascii=False,indent=2),encoding='utf-8')
        notify(f'Новых групп/контейнеров: {count}; статус: {status}')
        return count
