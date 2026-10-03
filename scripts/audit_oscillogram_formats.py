"""Читает входные файлы осциллограмм и сохраняет результат проверки каждого файла.

Запуск из корня репозитория: python -m scripts.audit_oscillogram_formats ROOT
Разные части DFR/NDR могут соответствовать одной собранной записи.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from osc_tools.io.oscillogram import MissingCompanionError, detect_format, iter_oscillograms, load_oscillogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wconfig', type=Path)
    parser.add_argument('--recover-incomplete', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('reports/oscillogram_formats.json'))
    args = parser.parse_args()
    results = []
    for path in iter_oscillograms(args.root):
        kind = detect_format(path)
        entry = {'path': str(path.relative_to(args.root)), 'format': kind}
        try:
            # Конфигурация для .046 не должна автоматически применяться к .017.
            config = args.wconfig if args.wconfig and args.wconfig.suffix.lower() == path.suffix.lower() else None
            rec = load_oscillogram(path, wconfig=config,
                                   recover_incomplete=args.recover_incomplete and kind in ('neva', 'res3'))
            frame = rec.to_dataframe()
            entry.update(status='ok', samples=rec.total_samples,
                         analogs=rec.analog_count, digitals=rec.status_count,
                         columns=frame.width, recovery_requested=getattr(rec, 'recovery_requested', False))
        except MissingCompanionError as error:
            entry.update(status='missing_companion', error=str(error))
        except Exception as error:
            entry.update(status='error', error=f'{type(error).__name__}: {error}')
        results.append(entry)
    summary = Counter(f"{item['format']}:{item['status']}" for item in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': dict(summary), 'files': results},
                                     ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary), ensure_ascii=False))
    print(args.output)
    return int(any(item['status'] == 'error' for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
