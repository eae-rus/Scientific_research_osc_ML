"""Read-only source inventory. Does not convert or rewrite source documents."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/Dissertation/planning/source_inventory.json'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def inspect(path):
    result = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix.lower() != '.docx':
        return result
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        body = ET.fromstring(archive.read('word/document.xml'))
        result['body_counts'] = {name: len(list(body.iter(W + tag))) for name, tag in
                                 [('paragraphs', 'p'), ('tables', 'tbl'), ('insertions', 'ins'),
                                  ('deletions', 'del'), ('comment_starts', 'commentRangeStart'),
                                  ('comment_ends', 'commentRangeEnd'), ('comment_refs', 'commentReference')]}
        result['body_counts']['math_objects'] = len(list(body.iter(
            '{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')))
        result['media_files'] = len([n for n in names if n.startswith('word/media/')])
        result['comment_parts'] = [n for n in names if n.startswith('word/comments')]
        result['comments'] = []
        if 'word/comments.xml' in names:
            comments = ET.fromstring(archive.read('word/comments.xml'))
            for c in comments.findall(W + 'comment'):
                result['comments'].append({'id': c.get(W + 'id'), 'author': c.get(W + 'author'),
                    'date': c.get(W + 'date'), 'text': ''.join(c.itertext())})
        # A structural orientation only; these are not rendered page locators.
        result['headings'] = []
        result['opening_paragraphs'] = []
        for index, p in enumerate(body.iter(W + 'p')):
            text = ''.join(t.text or '' for t in p.iter(W + 't'))
            if text.strip() and len(result['opening_paragraphs']) < 12:
                result['opening_paragraphs'].append(text)
            props = p.find(W + 'pPr')
            style = props.find(W + 'pStyle') if props is not None else None
            level = props.find(W + 'outlineLvl') if props is not None else None
            sid = style.get(W + 'val', '') if style is not None else ''
            if level is not None or 'heading' in sid.lower() or sid in ['1', '2', '3']:
                result['headings'].append({'paragraph_index': index, 'style': sid, 'text': text})
    return result


def main():
    article = ROOT / 'docs/article'
    paths = list(article.glob('*.docx'))
    paths += list((ROOT / 'docs/Dissertation/Архивное').glob('*'))
    paths += list((article / '(Энергоинновации) Датасет+Статистика ОЗЗ').glob('*.docx'))
    paths += [article / 'Engineering_Optimal_Feature_Spaces_OJIES/Engineering_Optimal_Feature_Spaces_OJIES v1.2.tex',
              article / '(Scientific_Data) Подготовка датасета/Для overleaf/Article.tex',
              article / 'PHASE_5_PDR_STATISTICAL_ARTICLE_DRAFT.md',
              ROOT / 'docs/Dissertation/Мысли о ведении документации.txt']
    payload = {'schema_version': 1,
        'scope': 'Selected local sources only; structural audit, not migration or visual review. Comment texts are preliminary; anchors and threads have not been exported.',
        'sources': [inspect(p) for p in sorted(set(paths)) if p.is_file()]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in payload['sources']:
        print(json.dumps({k: v for k, v in row.items() if k not in ['comments', 'headings', 'opening_paragraphs']}, ensure_ascii=False))
    for row in payload['sources']:
        if '/Архивное/' in row['path'] and row['path'].endswith('.docx'):
            print(json.dumps({k: row[k] for k in ['path', 'headings', 'opening_paragraphs']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
