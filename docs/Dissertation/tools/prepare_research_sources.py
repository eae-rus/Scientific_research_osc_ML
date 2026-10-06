"""Immutable primary snapshots and lossless OOXML review parts for C02/C03.

Run from repository root with bundled Python. No source mutation or automatic
acceptance of tracked changes. Existing output with differing bytes is an error.
"""
from pathlib import Path
import argparse,hashlib,json,zipfile,shutil,posixpath
from lxml import etree
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W='{'+NS['w']+'}'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=data:raise ValueError(f'Immutable output differs: {p}')
 else:p.write_bytes(data)
def jbytes(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def snapshot(sid,path):
 path=Path(path);sha=digest(path);target=ROOT/'sources'/sid/sha/path.name
 save(target,path.read_bytes())
 save(target.parent/'origin.json',jbytes({'source_id':sid,'source_path':path.as_posix(),'sha256':sha,'snapshot_path':target.relative_to(REPO).as_posix(),'date':'2026-10-05'}))
 return target
def text_of(el):
 return ''.join(x if isinstance(x,str) else '\t' if x.tag==W+'tab' else '\n'
                for x in el.xpath('.//w:t/text() | .//w:delText/text() | .//w:tab | .//w:br',namespaces=NS))
def extract_docx(sid,path):
 path=Path(path);sha=digest(path);target=ROOT/'review/raw'/sid/sha
 with zipfile.ZipFile(path) as z:
  names=z.namelist();selected=[n for n in names if n.endswith('.xml') or n.endswith('.rels')]
  # Preserve all XML and relationships, including unknown/future metadata.
  for n in selected:save(target/'parts'/n,z.read(n))
  # All linked/non-XML attachments remain available in the immutable DOCX.
  paragraphs=[];anchors={};revisions=[]
  for part in selected:
   if not part.endswith('.xml'):continue
   try:tree=etree.fromstring(z.read(part))
   except etree.XMLSyntaxError:continue
   story=part.startswith('word/') and part.split('/')[-1].startswith(('document','footnotes','endnotes','header','footer'))
   pnums={}
   if story:
    for i,para in enumerate(tree.xpath('//w:p',namespaces=NS),1):
     loc=para.getroottree().getpath(para);pnums[para]=i
     paragraphs.append({'story_part':part,'paragraph_index':i,'locator':loc,'style':para.xpath('string(./w:pPr/w:pStyle/@w:val)',namespaces=NS),'text':text_of(para)})
    for node in tree.iter():
     if node.tag not in (W+'commentRangeStart',W+'commentRangeEnd',W+'commentReference'):continue
     cid=node.get(W+'id');parent=node
     while parent is not None and parent.tag!=W+'p':parent=parent.getparent()
     anchors.setdefault(cid,[]).append({'kind':node.tag.split('}')[1],'story_part':part,'locator':node.getroottree().getpath(node),'paragraph_index':pnums.get(parent),'paragraph_text':text_of(parent) if parent is not None else None})
   for node in tree.iter():
    if node.tag in (W+'ins',W+'del',W+'moveFrom',W+'moveTo'):
     revisions.append({'story_part':part,'kind':node.tag.split('}')[1],'locator':node.getroottree().getpath(node),'attributes':dict(node.attrib),'text':text_of(node)})
  comments=[]
  if 'word/comments.xml' in names:
   tree=etree.fromstring(z.read('word/comments.xml'))
   for c in tree.xpath('./w:comment',namespaces=NS):
    cid=c.get(W+'id');comments.append({'original_comment_id':cid,'attributes':dict(c.attrib),'paragraphs':[text_of(q) for q in c.xpath('./w:p',namespaces=NS)],'text':'\n'.join(text_of(q) for q in c.xpath('./w:p',namespaces=NS)),'anchors':anchors.get(cid,[]),'decision':'not_reviewed'})
  manifest={'source_id':sid,'source_sha256':sha,'source_path':path.as_posix(),'preserved_parts':[{'path':n,'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in selected],'non_xml_preserved_in_docx':[n for n in names if n not in selected],'paragraph_count':len(paragraphs),'comment_count':len(comments),'revision_count':len(revisions),'semantics':'w:t and w:delText exported; no tracked changes accepted or rejected; paragraph context is not an exact text range'}
 save(target/'manifest.json',jbytes(manifest));save(target/'comments.json',jbytes(comments));save(target/'revisions.json',jbytes(revisions))
 save(target/'paragraphs.json',jbytes(paragraphs))
 return paragraphs,comments,manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-id',action='append');args=ap.parse_args()
 m=json.loads((ROOT/'planning/sources/manifest.json').read_text(encoding='utf-8'))
 build=ROOT/'build/reading';build.mkdir(parents=True,exist_ok=True)
 summary=[]
 for s in m['sources']:
  sid=s['source_id']
  if args.source_id and sid not in args.source_id:continue
  primary=Path(s['main']);expected=m['files'][s['main']]['sha256']
  if digest(primary)!=expected:raise ValueError(f'Changed source version {sid}')
  inputs=list(dict.fromkeys([s['main']]+s.get('comment_inputs',[])))
  if sid=='DATA-SD':inputs.append((REPO/'docs/article/(Scientific_Data) Подготовка датасета/Для overleaf/Article.tex').as_posix())
  if sid in ('DATA-OZZ','STAT-OZZ'):
   inputs += [k for k in s['files'] if k.endswith('.docx')]
  records=[]
  for key in dict.fromkeys(inputs):
   q=Path(key);snap=snapshot(sid,q);record={'source':q.as_posix(),'sha256':digest(q),'snapshot':snap.relative_to(REPO).as_posix()}
   if q.suffix.lower()=='.docx':
    paras,comments,raw=extract_docx(sid,q);record.update(raw_directory=(ROOT/'review/raw'/sid/digest(q)).relative_to(REPO).as_posix(),comments=len(comments),paragraphs=len(paras),revisions=raw['revision_count'])
    dump='\n'.join(f'[{a["story_part"]}:P{a["paragraph_index"]:04d}; {a["style"]}] {a["text"]}' for a in paras)
    (build/f'{sid}-{digest(q)[:8]}.txt').write_text(dump,encoding='utf-8')
   elif q.suffix.lower()=='.pdf':
    pdf=PdfReader(q);dump='\n'.join(f'\n=== PAGE {i+1} ===\n'+(page.extract_text() or '') for i,page in enumerate(pdf.pages))
    (build/f'{sid}.txt').write_text(dump,encoding='utf-8');record['pages']=len(pdf.pages)
   else:
    dump='\n'.join(f'L{i:04d}: {line}' for i,line in enumerate(q.read_text(encoding='utf-8-sig').splitlines(),1))
    (build/f'{sid}-{q.suffix[1:]}.txt').write_text(dump,encoding='utf-8')
   records.append(record)
  summary.append({'source_id':sid,'records':records})
 (build/'preparation.json').write_bytes(jbytes(summary))
 for s in summary:print(s['source_id'],[(Path(q['source']).name,q.get('comments'),q.get('paragraphs'),q.get('pages')) for q in s['records']])
if __name__=='__main__':main()
