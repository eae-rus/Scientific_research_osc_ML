"""Read-only checks of immutable research snapshots and OOXML exports."""
import hashlib, json, zipfile
from pathlib import Path
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    preparation = json.loads((ROOT/'build/reading/preparation.json').read_text(encoding='utf-8'))
    checked, comments, revisions, parts = [], 0, 0, 0
    for packet in preparation:
        for record in packet['records']:
            original = Path(record['source'])
            snapshot = REPO/record['snapshot']
            assert sha(original.read_bytes()) == record['sha256'], original
            assert original.read_bytes() == snapshot.read_bytes(), snapshot
            if 'raw_directory' in record:
                raw = REPO/record['raw_directory']
                manifest = json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
                exported = json.loads((raw/'comments.json').read_text(encoding='utf-8'))
                with zipfile.ZipFile(original) as archive:
                    expected = [n for n in archive.namelist() if n.endswith(('.xml', '.rels'))]
                    assert set(expected) == {p['path'] for p in manifest['preserved_parts']}
                    for part in manifest['preserved_parts']:
                        data = archive.read(part['path'])
                        assert data == (raw/'parts'/part['path']).read_bytes()
                        assert sha(data) == part['sha256']
                        parts += 1
                    ids = []
                    if 'word/comments.xml' in archive.namelist():
                        root = etree.fromstring(archive.read('word/comments.xml'))
                        ids = root.xpath('./w:comment/@w:id', namespaces=NS)
                    assert ids == [c['original_comment_id'] for c in exported]
                    assert len(exported) == manifest['comment_count']
                comments += len(exported)
                revisions += manifest['revision_count']
            checked.append({'source_id':packet['source_id'], 'sha256':record['sha256'], 'path':record['source']})
    report = {'date':'2026-10-05','status':'passed','files':len(checked),
              'comments':comments,'revisions':revisions,'xml_and_relationship_parts':parts,
              'checked':checked,
              'limits':'Byte preservation and comment ID coverage; no scientific or visual acceptance.'}
    target = ROOT/'build/reading/verification.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checked'},ensure_ascii=False))

if __name__ == '__main__':
    main()
