"""Сохранение готовых агентских черновиков отдельно от рабочих контекстов."""
import json
from pathlib import Path
import re
import shutil
from .agent_runtime import now, write_json
from .case_inventory import sha256


def export_reports(prepared_root, contexts, run_name, destination=None):
    """Копировать отчёт/качество/состав источников по истинному имени папки.

    Прежние редакции и существующие пользовательские правки не затираются.
    При изменении байтов рядом создаётся следующая редакция. Рабочий контекст
    после успешного экспорта можно архивировать отдельно от итоговых файлов.
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_name):
        raise ValueError('Некорректное имя прогона')
    root = Path(prepared_root).resolve(strict=True)
    provenance = json.loads((root/'provenance_index.json').read_text(encoding='utf-8'))
    folders = {c['case_id']:c for c in provenance['cases']}
    output = Path(destination).resolve() if destination else root/'reports'
    exported = []
    for context in contexts:
        workspace = Path(context['workspace']).resolve(strict=True)
        manifest = json.loads((workspace/'manifest.json').read_text(encoding='utf-8'))
        cid = manifest['case_id']
        reports = sorted((workspace/'reports').glob(f'revision_*/{cid}__report.md'))
        if not reports:
            continue
        source = reports[-1]
        folder = folders[cid]['source_folder']
        year, leaf = folder.split('/',1)
        if not re.fullmatch(r'\d{4}',year) or '/' in leaf or '\\' in leaf or leaf.rstrip(' .') in {'','..'}:
            raise ValueError('Некорректная исходная папка')
        base = output/run_name/year/leaf.rstrip(' .')
        if not base.resolve().is_relative_to(output):
            raise ValueError('Отчёт выходит за каталог результатов')
        base.mkdir(parents=True,exist_ok=True)
        source_digest = sha256(source)
        existing = list(base.glob('revision_*/report.md'))
        same = next((p for p in existing if sha256(p) == source_digest and
                     (p.parent/'quality.json').is_file() and sha256(p.parent/'quality.json') == sha256(source.parent/'quality.json') and
                     (p.parent/'sources.json').is_file() and sha256(p.parent/'sources.json') == sha256(workspace/'manifest.json')),None)
        if same:
            exported.append({'case_id':cid,'source_folder':folder,'report':str(same),'status':'already_exported'})
            continue
        numbers = [int(p.name.split('_')[-1]) for p in base.glob('revision_*') if re.fullmatch(r'revision_\d+',p.name)]
        revision = base/f'revision_{max(numbers,default=0)+1:04d}'
        revision.mkdir(exist_ok=False)
        shutil.copy2(source,revision/'report.md')
        shutil.copy2(source.parent/'quality.json',revision/'quality.json')
        shutil.copy2(workspace/'manifest.json',revision/'sources.json')
        write_json(revision/'provenance.json',folders[cid])
        checkpoint = workspace/'remote_checkpoint.json'
        run_metadata = {'case_id':cid,'source_folder':folder,'report_sha256':source_digest,'created_at':now(),
            'human_review':'pending','instruction_sha256':manifest['instruction_sha256'],'vocabulary_sha256':manifest['vocabulary_sha256']}
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            run_metadata.update(model=state.get('model'),usage=state.get('usage'),status=state.get('status'))
        write_json(revision/'run.json',run_metadata)
        exported.append({'case_id':cid,'source_folder':folder,'report':str(revision/'report.md'),'status':'exported'})
    if exported:
        (output/run_name).mkdir(parents=True,exist_ok=True)
        index_path = output/run_name/'catalog.json'
        previous = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else []
        merged = {row['report']:row for row in previous+exported}
        write_json(index_path,list(merged.values()))
    return exported
