"""Preserve author notes verbatim and attach explicitly reviewed status markers.

No decisions are inferred from a note's content. Callers supply dispositions
in review/issues/author_note_dispositions.json; unknown notes stay OPEN.
"""
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MARKER = re.compile(r' \*\*\[AA\d+: [^\n]*?\]\*\*')

def scan(text):
    result = []
    for match in re.finditer(r'\(Примечание автора(?:[:,])', text):
        # The author's numbered appendix ends with a standalone closing
        # parenthesis; list punctuation is intentionally not balanced.
        if text.startswith('(Примечание автора, Приложение', match.start()):
            closing = re.search(r'^\)(?: \*\*\[AA\d+: [^\n]*?\]\*\*)?[ \t]*$', text[match.end():], re.MULTILINE)
            if closing is None:
                raise ValueError('Missing standalone end of author appendix')
            end = match.end() + closing.start() + 1
            result.append((match.start(), end, text[match.start():end]))
            continue
        depth, end = 0, None
        for i in range(match.start(), len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                # A numbered list such as "1)" or "4.1)" inside a long
                # author note is not the closing delimiter of that note.
                prefix = text[text.rfind('\n', match.start(), i) + 1:i]
                if re.fullmatch(r'\s*\d+(?:\.\d+)*', prefix):
                    continue
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError(f'Unclosed author note at character {match.start()}')
        raw = text[match.start():end]
        if raw == '(Примечание автора: ТЕКСТ)':
            continue
        result.append((match.start(), end, raw))
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    files = [ROOT/'planning/integration/rewrite_proposals/ch03.md',
             ROOT/'planning/integration/rewrite_proposals/ch05.md',
             ROOT/'planning/handoffs/D00.md', ROOT/'planning/questions.md']
    registry_path = ROOT/'review/issues/author_annotations.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'schema_version':1,'notes':[]}
    disposition_path = ROOT/'review/issues/author_note_dispositions.json'
    dispositions = json.loads(disposition_path.read_text(encoding='utf-8')) if disposition_path.exists() else {}
    existing = {(n['file'],n['text_sha256']):n for n in registry['notes']}
    prepared = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        notes = scan(text)
        edits = []
        for index, (start,end,raw) in enumerate(notes,1):
            rel = path.relative_to(REPO).as_posix()
            sha = hashlib.sha256(raw.encode('utf-8')).hexdigest()
            record = existing.get((rel,sha))
            if record is None:
                record = {'id':f'AA{len(registry["notes"])+1:03d}','file':rel,'original_text':raw,'text_sha256':sha,'status':'OPEN','decision':'Not reviewed yet.'}
                registry['notes'].append(record)
                existing[(rel,sha)] = record
            key = f'{path.relative_to(ROOT).as_posix()}#{index}'
            if key in dispositions:
                record.update(dispositions[key])
            record['last_seen_note_index'] = index
            record['last_seen_file_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
            label = record.get('display','прочитано; решение открыто' if record['status']!='OPEN' else 'новое; не рассмотрено')
            old = MARKER.match(text,end)
            edits.append((end,old.end() if old else end,f' **[{record["id"]}: {label}]**'))
            print(record['id'],key,record['status'])
        changed = text
        for start,end,replacement in reversed(edits):
            changed = changed[:start]+replacement+changed[end:]
        assert [q[2] for q in scan(text)] == [q[2] for q in scan(changed)],path
        prepared.append((path,changed))
    if args.apply:
        registry['updated']='2026-10-06'
        registry['policy']='Original text retained; status means reviewed disposition, not automatic completion of related scientific work. Never silently delete notes.'
        registry_path.parent.mkdir(parents=True,exist_ok=True)
        registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for path,text in prepared:
            path.write_text(text,encoding='utf-8')

if __name__=='__main__':
    main()
