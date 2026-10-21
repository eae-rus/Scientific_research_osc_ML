"""Preserve author notes verbatim and attach explicitly reviewed status markers.

No decisions are inferred from a note's content. Callers supply dispositions
in review/issues/author_note_dispositions.json; unknown notes stay OPEN.
"""
import argparse, hashlib, json, re
from datetime import date
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


def group_author_replies(text):
    """Keep a reply beside an existing AA note in that note's history.

    The explicit authorreviewnote anchor identifies the parent; repeated
    reply wording in other blocks never merges unrelated comments.
    scan() remains lossless and returns every original note and reply.
    """
    all_notes = scan(text)
    reply_starts, replies = set(), {}
    for block in re.finditer(r'\\iffalse\b.*?\\fi\b\s*\\authorreviewnote\{(AA\d+)\}', text, re.S):
        contained = [n for n in all_notes if block.start() <= n[0] < n[1] <= block.end()]
        if len(contained) > 1:
            replies[contained[0][0]] = [n[2] for n in contained[1:]]
            reply_starts.update(n[0] for n in contained[1:])
    return [n for n in all_notes if n[0] not in reply_starts], replies

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--review-tex', action='store_true',
                    help='Render existing registry replies for the annotated PDF.')
    args = ap.parse_args()
    files = [ROOT/'planning/integration/rewrite_proposals/ch03.md',
             ROOT/'planning/integration/rewrite_proposals/ch05.md',
             ROOT/'planning/handoffs/D00.md', ROOT/'planning/questions.md',
             ROOT/'planning/sources/PDR-05/limitations_and_comments.md',
             ROOT/'manuscript/chapters/ch03_dataset_preparation/sec10_earth_fault_subset.tex',
             ROOT/'manuscript/chapters/ch04_statistical_analysis/sec02_physkan_architectures.tex',
             ROOT/'manuscript/chapters/ch04_statistical_analysis/sec01_evaluation_methodology.tex',
             ROOT/'manuscript/chapters/ch04_statistical_analysis/sec03_signal_representations.tex',
             ROOT/'manuscript/chapters/ch04_statistical_analysis/sec04_training_protocol.tex',
             ROOT/'manuscript/chapters/ch05_ml_adaptive_algorithms/chapter.tex']
    registry_path = ROOT/'review/issues/author_annotations.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'schema_version':1,'notes':[]}
    disposition_path = ROOT/'review/issues/author_note_dispositions.json'
    dispositions = json.loads(disposition_path.read_text(encoding='utf-8')) if disposition_path.exists() else {}
    existing = {(n['file'],n['text_sha256']):n for n in registry['notes']}
    prepared = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        notes, author_replies = group_author_replies(text)
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
            disposition = dispositions.get(key)
            # A new note inserted earlier must not inherit the old note's decision.
            if disposition and disposition.get('text_sha256') == sha:
                record.update(disposition)
            if start in author_replies:
                history = record.setdefault('author_replies', [])
                for reply in author_replies[start]:
                    reply_sha = hashlib.sha256(reply.encode('utf-8')).hexdigest()
                    if not any(r['sha256'] == reply_sha for r in history):
                        history.append({'text': reply, 'sha256': reply_sha})
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
        registry['updated']=date.today().isoformat()
        registry['policy']='Original text retained; status means reviewed disposition, not automatic completion of related scientific work. Never silently delete notes.'
        registry_path.parent.mkdir(parents=True,exist_ok=True)
        for path,text in prepared:
            path.write_text(text,encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rel = path.relative_to(REPO).as_posix()
            for record in registry['notes']:
                if record['file'] == rel:
                    record['last_seen_file_sha256'] = digest
        registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.review_tex:
        escapes = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
                   '%': r'\%', '&': r'\&', '#': r'\#', '_': r'\_',
                   '$': r'\$', '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
        def tex(value):
            return ''.join(escapes.get(c,c) for c in value)
        states = {'implemented': 'Правка внесена, ожидает проверки автора',
                  'partial': 'Выполнено частично, замечание открыто',
                  'deferred': 'Отложено, замечание открыто',
                  'not_required': 'Ответ записан, ожидает проверки автора'}
        chunks = ['% Generated from the existing author-note registry. Do not edit.']
        for record in registry['notes']:
            if record.get('suppress_in_review') and record.get('author_acceptance') == 'accepted':
                chunks.append(r'\expandafter\def\csname authornote'+record['id']+r'\endcsname{}')
                continue
            chunks.append(r'\expandafter\def\csname authorvisible'+record['id']+r'\endcsname{1}')
            state = states.get(record.get('implementation_status'), 'Исполнение отдельно не проверено; замечание открыто')
            reply = record.get('response_ru', 'Подробная повторная проверка этого примечания в данной партии не выполнялась.')
            target = record.get('section_label')
            accepted = record.get('author_acceptance') == 'accepted_current_plan'
            if accepted:
                state = 'Ок. Автор принял текущий план'
            location = (r'\par Место правки: раздел~\ref{'+target+'}.') if target else ''
            acceptance = ('Подтверждение относится только к текущему плану главы.' if accepted else
                          'Автоматического закрытия нет. Ожидается ответ автора.')
            chunks.append(r'\expandafter\def\csname authornote'+record['id']+r'\endcsname{'+
                          r'\noindent\textbf{'+record['id']+'. '+tex(state)+r'.}\par\nopagebreak[4] '+
                          tex(record['original_text'])+
                          ''.join(r'\par\textbf{Дополнение автора.} '+tex(r['text']) for r in record.get('author_replies', []))+
                          r'\par\textbf{Ответ агента.} '+tex(reply)+
                          location+r'\par\textit{'+tex(acceptance)+'}}')
        output = ROOT/'build/reading/author_notes.tex'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text('\n'.join(chunks)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
