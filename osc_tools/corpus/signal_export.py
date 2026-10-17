"""Экспорт полных записей производителей отдельным рабочим процессом.

Используется существующий OscFormats bridge. Временная сегментация, фильтрация,
прореживание и восстановление неполных записей не выполняются. COMTRADE в этой
итерации явно отложен до подключения существующего Python-загрузчика.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from .agent_runtime import now, write_json
from .case_inventory import sha256


def validate_payload(folder):
    """Проверить все отсчёты экспортированного потока без обрезки массивов."""
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
    count = metadata['total_samples']
    na, nd = len(metadata['analog_channels']), len(metadata['status_channels'])
    if metadata.get('protocol') != 1 or type(count) is not int or count <= 0 or na + nd == 0:
        raise ValueError('Некорректные метаданные экспорта')
    binary = folder / 'samples.bin'
    if binary.stat().st_size != count * (8 + na * 8 + nd):
        raise ValueError('Размер потока не соответствует числу всех отсчётов')
    first_time, last_time, previous = None, None, None
    with binary.open('rb') as stream:
        for channel in range(1 + na):
            remaining = count
            while remaining:
                amount = min(remaining, 65536)
                values = array('d')
                values.frombytes(stream.read(amount * 8))
                if sys.byteorder != 'little':
                    values.byteswap()
                for value in values:
                    if not math.isfinite(value):
                        raise ValueError('Нечисловой/бесконечный отсчёт')
                    if channel == 0:
                        if previous is not None and value <= previous:
                            raise ValueError('Время не возрастает строго')
                        if first_time is None:
                            first_time = value
                        previous = last_time = value
                remaining -= amount
        for _ in range(nd):
            remaining = count
            while remaining:
                amount = min(remaining, 65536)
                if any(b > 1 for b in stream.read(amount)):
                    raise ValueError('Дискретный канал содержит значения вне 0/1')
                remaining -= amount
    return {'total_samples': count, 'analog_channels': na, 'status_channels': nd,
            'time_start_s': first_time, 'time_end_s': last_time, 'duration_s': last_time - first_time,
            'samples_sha256': sha256(binary), 'metadata_sha256': sha256(folder / 'metadata.json'),
            'format': metadata['format'], 'sample_rates': metadata.get('sample_rates'),
            'station_name': metadata.get('station_name'), 'start_timestamp': metadata.get('start_timestamp'),
            'trigger_timestamp': metadata.get('trigger_timestamp'), 'all_samples_validated': True,
            'cropping': 'none', 'resampling': 'none', 'channel_filtering': 'none',
            'recovery_requested': False, 'calibration': 'A/B уже применены bridge; повторно не применять',
            'document_episode_link': 'not_verified'}


def export_workspace(workspace, output, bridge, timeout=120):
    """Экспортировать кандидаты текущей папки, сохраняя ошибки каждого входа."""
    workspace = Path(workspace).resolve(strict=True)
    manifest = json.loads((workspace / 'manifest.json').read_text(encoding='utf-8'))
    source = Path(manifest['source_root']).resolve(strict=True)
    corpus = Path(manifest['corpus_root']).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(corpus) or corpus.is_relative_to(output) or workspace.is_relative_to(output):
        raise ValueError('Каталог экспорта должен быть отдельно от исходников и рабочего входа')
    bridge = Path(bridge).resolve(strict=True)
    dotnet = shutil.which('dotnet') if bridge.suffix.lower() == '.dll' else None
    if bridge.suffix.lower() == '.dll' and not dotnet:
        raise RuntimeError('Для OscFormats.dll нужен .NET runtime')
    if not 1 <= timeout <= 3600:
        raise ValueError('Недопустимый таймаут')
    initial = {}
    for f in manifest['files']:
        if 'sha256' not in f:
            continue
        path = (source / f['path']).resolve(strict=True)
        if not path.is_relative_to(source) or sha256(path) != f['sha256']:
            raise ValueError('Источник изменился после подготовки')
        initial[f['path']] = f['sha256']
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for f in manifest['files']:
        if f['role'] != 'recording_candidate':
            continue
        entry = {'file_id': f['file_id'], 'source_path': f['path'], 'sha256': f.get('sha256'),
                 'candidate_format': f.get('format'), 'status': 'pending'}
        entries.append(entry)
        target = output / f['file_id']
        if f.get('format') == 'comtrade':
            entry.update(status='reader_not_connected', error='Требуется существующий Python load_oscillogram; зависимости здесь не подключены')
        else:
            try:
                target.mkdir()
                command = ([dotnet] if dotnet else []) + [str(bridge), f['format'], str(source / f['path']), str(target), 'false']
                if f.get('format') == 'ekra-ndr':
                    expected = 'wconfig' + Path(f['path']).suffix.lower()
                    siblings = [p for p in (source / f['path']).parent.iterdir() if p.name.lower() == expected]
                    if len(siblings) != 1:
                        raise FileNotFoundError('Нужен единственный соответствующий WConfig рядом с NDR')
                    command.append(str(siblings[0]))
                completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
                if completed.returncode:
                    raise RuntimeError(completed.stderr.decode('utf-8', errors='replace').strip())
                entry.update(validate_payload(target), status='exported_full_record', output=str(target))
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'progress.json', {'case_id': manifest['case_id'], 'files': entries, 'updated_at': now()})
    unchanged = all(sha256(source / p) == digest for p, digest in initial.items())
    if not unchanged:
        raise ValueError('Во время экспорта изменились исходные байты')
    duplicates = {}
    for entry in entries:
        if entry.get('samples_sha256'):
            duplicates.setdefault(entry['samples_sha256'], []).append(entry['file_id'])
    result = {'case_id': manifest['case_id'], 'created_at': now(), 'source_files_unchanged': True,
              'bridge': str(bridge), 'bridge_sha256': sha256(bridge), 'files': entries,
              'status': 'completed' if entries and all(e['status'] == 'exported_full_record' for e in entries) else 'partial',
              'same_sample_payload_groups': [v for v in duplicates.values() if len(v) > 1],
              'duplicate_groups_are_not_event_identity': True, 'cropping': 'none',
              'event_detection': 'not_performed', 'document_labels_transferred': False}
    write_json(output / 'signal_manifest.json', result)
    return result


def main():
    """Выполнить независимый от документации экспорт в новый каталог."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bridge', required=True)
    args = parser.parse_args()
    result = export_workspace(args.workspace, args.output, args.bridge)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
