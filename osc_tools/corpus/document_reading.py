"""Локальное извлечение источников с локаторами и явными границами чтения.

DOC: только основной текст по таблице фрагментов MS-DOC; вёрстка, текстовые
поля, рисунки и вложения не проверяются. Потоки Compound читает установленный
7-Zip. DOCX: абзацы и таблицы в порядке тела, ссылки на встроенные изображения.
PDF: только текстовый слой по страницам. Просмотр выполняется отдельно.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
V = '{urn:schemas-microsoft-com:vml}'
SEVEN_ZIP = Path(r'C:\Program Files\7-Zip\7z.exe')


def doc_piece_text(word: bytes, table: bytes) -> str:
    """Прочитать основной текст Word 97+ без анализа оформления и объектов."""
    if len(word) < 34 or struct.unpack_from('<H', word)[0] != 0xA5EC:
        raise ValueError('Не поддерживается сигнатура DOC')
    if struct.unpack_from('<H', word, 2)[0] < 0x00C1:
        raise ValueError('Версия DOC до Word 97 не поддерживается')
    flags = struct.unpack_from('<H', word, 10)[0]
    if flags & 0x8100:
        raise ValueError('Зашифрованный DOC не поддерживается')
    pos = 32
    csw = struct.unpack_from('<H', word, pos)[0]
    pos += 2 + csw * 2
    cslw = struct.unpack_from('<H', word, pos)[0]
    pos += 2
    if cslw < 4:
        raise ValueError('Не поддерживается версия FIB')
    ccp_text = struct.unpack_from('<I', word, pos + 12)[0]
    pos += cslw * 4
    count = struct.unpack_from('<H', word, pos)[0]
    pos += 2
    if count <= 33:
        raise ValueError('В FIB отсутствует указатель Clx')
    offset, length = struct.unpack_from('<II', word, pos + 33 * 8)
    clx = table[offset:offset + length]
    if len(clx) != length:
        raise ValueError('Clx выходит за границы потока')
    pos = 0
    while pos < len(clx) and clx[pos] == 1:
        pos += 3 + struct.unpack_from('<H', clx, pos + 1)[0]
    if pos >= len(clx) or clx[pos] != 2:
        raise ValueError('Таблица фрагментов не найдена')
    size = struct.unpack_from('<I', clx, pos + 1)[0]
    plc = clx[pos + 5:pos + 5 + size]
    if len(plc) != size or size < 4 or (size - 4) % 12:
        raise ValueError('Повреждена таблица фрагментов')
    n = (size - 4) // 12
    cps = struct.unpack_from('<' + 'I' * (n + 1), plc)
    if cps[0] != 0 or any(a > b for a, b in zip(cps, cps[1:])) or cps[-1] < ccp_text:
        raise ValueError('Некорректные позиции символов')
    chunks = []
    for i in range(n):
        length = max(0, min(cps[i + 1], ccp_text) - cps[i])
        if not length:
            continue
        fc = struct.unpack_from('<I', plc, 4 * (n + 1) + 8 * i + 2)[0]
        compressed = bool(fc & 0x40000000)
        offset = fc & 0x3FFFFFFF
        if compressed:
            offset //= 2
        raw = word[offset:offset + length * (1 if compressed else 2)]
        if len(raw) != length * (1 if compressed else 2):
            raise ValueError('Фрагмент выходит за границы WordDocument')
        # MS-DOC FcCompressed: однобайтовые символы имеют отображение Windows-1252.
        chunks.append(raw.decode('cp1252' if compressed else 'utf-16-le', errors='strict'))
    return ''.join(chunks)


def compound_stream(path: Path, name: str, seven_zip: Path = SEVEN_ZIP) -> bytes:
    """Прочитать один явно выбранный поток Compound в память, без распаковки."""
    result = subprocess.run([str(seven_zip), 'e', '-so', str(path.resolve()), name],
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'7-Zip: {result.returncode}: {result.stderr.decode("utf-8", errors="replace")}')
    return result.stdout


def read_doc(path: Path, seven_zip: Path = SEVEN_ZIP) -> dict:
    """Получить только основной текст DOC, сохранив маркеры ячеек и полей."""
    word = compound_stream(path, 'WordDocument', seven_zip)
    flags = struct.unpack_from('<H', word, 10)[0]
    table = compound_stream(path, '1Table' if flags & 0x0200 else '0Table', seven_zip)
    text = doc_piece_text(word, table)
    blocks = [{'locator': f'main/p{i}', 'text': line.replace('\x07', ' | ')
               .replace('\x01', '[объект]').replace('\x13', '[поле:')
               .replace('\x14', ' → ').replace('\x15', ']')}
              for i, line in enumerate(text.split('\r'), 1) if line.strip()]
    return {'method': 'MS-DOC Clx/PlcPcd; потоки Compound через 7-Zip', 'blocks': blocks,
            'limitations': ['Только основной текст; вёрстка, колонтитулы, текстовые поля, изображения, вложения и отметки оформления не проверены'],
            'images': [], 'main_text_characters': len(text)}


def read_docx(path: Path, media_dir: Path | None = None) -> dict:
    """Сохранить порядок абзацев/таблиц DOCX и привязку изображений к блокам."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        rels = ET.fromstring(z.read('word/_rels/document.xml.rels')) if 'word/_rels/document.xml.rels' in z.namelist() else []
        relationships = {r.get('Id'): r.attrib for r in rels}
        blocks, images, marked_runs, table_rows = [], [], [], []
        body = root.find(W + 'body')
        for b, node in enumerate(body, 1):
            tag = node.tag.removeprefix(W)
            if tag not in {'p', 'tbl'}:
                continue
            parts = []
            for child in node.iter():
                if child.tag == W + 't': parts.append(child.text or '')
                elif child.tag in {W + 'br', W + 'cr'}: parts.append('\n')
                elif child.tag == W + 'tab': parts.append('\t')
                elif child.tag == W + 'tc': parts.append(' | ')
                elif child.tag == W + 'tr': parts.append('\n')
            locator = f'body/{tag}{b}'
            for run in node.iter(W + 'r'):
                props = run.find(W + 'rPr')
                if props is not None:
                    markers = {}
                    for prop in ('b', 'u', 'strike', 'highlight'):
                        element = props.find(W + prop)
                        if element is not None:
                            markers[prop] = element.get(W + 'val', 'true')
                    if markers:
                        marked_runs.append({'locator': locator, 'text': ''.join(t.text or '' for t in run.iter(W + 't')), 'properties': markers})
            if tag == 'tbl':
                for row_number, row in enumerate(node.findall(W + 'tr'), 1):
                    table_rows.append({'locator': f'{locator}/row{row_number}',
                                       'cells': [''.join(t.text or '' for t in cell.iter(W + 't')) for cell in row.findall(W + 'tc')]})
            references = []
            for element in node.iter():
                rid = element.get(R + 'embed') or element.get(R + 'id') if element.tag in {A + 'blip', V + 'imagedata'} else None
                if not rid: continue
                relationship = relationships.get(rid, {})
                target = relationship.get('Target', '').replace('\\', '/')
                info = {'relationship_id': rid, 'target': target, 'locator': locator, 'status': 'not_viewed'}
                if relationship.get('TargetMode') == 'External':
                    info['limitation'] = 'Внешняя ссылка не загружалась'
                else:
                    member = PurePosixPath('word') / target
                    if '..' in member.parts or member.is_absolute() or ':' in target:
                        info['limitation'] = 'Небезопасный путь не извлекался'
                    elif str(member) in z.namelist() and media_dir:
                        media_dir.mkdir(parents=True, exist_ok=True)
                        out = media_dir / f'{rid}__{member.name}'
                        if out.exists(): raise FileExistsError(out)
                        out.write_bytes(z.read(str(member)))
                        info['extracted_path'] = str(out.resolve())
                references.append(rid)
                images.append(info)
            blocks.append({'locator': locator, 'text': ''.join(parts).strip(), 'image_references': references})
        return {'method': 'OOXML: тело, абзацы, таблицы, relationship изображений', 'blocks': blocks,
                'images': images, 'marked_runs': marked_runs, 'table_rows': table_rows,
                'limitations': ['Вёрстка DOCX не рендерилась; колонтитулы, комментарии и история исправлений не разобраны']}


def read_pdf(path: Path) -> dict:
    """Извлечь текст PDF по страницам, не приписывая этому визуальный просмотр."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    blocks = [{'locator': f'page/{i}', 'text': page.extract_text() or ''}
              for i, page in enumerate(reader.pages, 1)]
    return {'method': 'pypdf: текстовый слой по страницам', 'blocks': blocks, 'images': [],
            'page_count': len(blocks), 'empty_text_pages': [b['locator'] for b in blocks if not b['text'].strip()],
            'limitations': ['Текстовый слой не подтверждает чтение скана и изображений; просмотр страниц учитывается отдельно']}


def read_text(path: Path, encoding: str | None = None) -> dict:
    """Декодировать TXT/LOG с BOM либо явно сохранить варианты русской кодировки."""
    raw = path.read_bytes()
    if encoding is not None:
        pass  # Явный выбор после смысловой проверки имеет приоритет над эвристикой.
    elif raw.startswith((b'\xff\xfe', b'\xfe\xff')): encoding = 'utf-16'
    else:
        try: raw.decode('utf-8-sig'); encoding = 'utf-8-sig'
        except UnicodeDecodeError: encoding = 'cp1251'
    value = raw.decode(encoding)
    result = {'method': f'декодирование {encoding}; читаемость проверяется анализатором',
              'encoding': encoding, 'blocks': [{'locator': f'line/{i}', 'text': line} for i, line in enumerate(value.splitlines(), 1)],
              'limitations': [], 'images': []}
    if encoding == 'cp1251':
        result['alternative_cp866'] = raw.decode('cp866', errors='replace')
        result['limitations'].append('Однобайтовая кодировка требует смысловой проверки с cp866')
    return result


def read_source(path: Path, source_id: str, media_dir: Path | None = None) -> dict:
    """Вернуть извлечение первичного источника с ID, хешем, методом и локаторами."""
    path = path.resolve(strict=True)
    readers = {'.doc': read_doc, '.pdf': read_pdf, '.txt': read_text, '.log': read_text}
    result = read_docx(path, media_dir) if path.suffix.lower() == '.docx' else readers[path.suffix.lower()](path)
    return {'source_id': source_id, 'source_path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **result}
