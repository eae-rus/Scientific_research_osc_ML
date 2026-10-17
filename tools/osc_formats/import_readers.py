"""Обновляет зафиксированную копию парсеров; запускается с явным путём к APScilloscope.

Изменяются только пространства имён, видимость классов и неиспользуемые методы
экспорта COMTRADE. После обновления проверьте различия и повторите проверки
на наборе образцов.
"""
from pathlib import Path
import hashlib
import json
import re
import sys


def import_readers(root: Path):
    destination = Path(__file__).parent / 'Readers'
    destination.mkdir(exist_ok=True)
    manifest = {}
    for source in sorted((root / 'Helpers' / 'ExternalFormats').glob('*.cs')):
        original = source.read_bytes()
        text = original.decode('utf-8-sig')
        # Методы экспорта с телом в виде выражения не содержат вложенных инструкций.
        text = re.sub(r'    public static string ConvertToComtrade\([^\n]*\)\s*=>[^;]+;', '', text)
        # В BlackBox есть один метод экспорта с блочным телом; удаляем его целиком.
        match = re.search(r'    public static string ConvertToComtrade\([^\n]*\)\s*\{', text)
        if match:
            end, depth = match.end(), 1
            while depth:
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                end += 1
            text = text[:match.start()] + text[end:]
        if 'ConvertToComtrade(' in text or 'RecordWriter' in text:
            raise ValueError(f'Unrecognised export method in {source}')
        text = text.replace('APS.ExternalFormats', 'OscFormats.Native').replace('APS.Comtrade', 'OscFormats.Model')
        text = text.replace('public static class', 'internal static class')
        (destination / source.name).write_text(text, encoding='utf-8', newline='\n')
        manifest[source.name] = hashlib.sha256(original).hexdigest()
    (destination / 'upstream-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import_readers(Path(sys.argv[1]))
