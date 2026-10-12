#!/usr/bin/env python3
"""Rebuild the literature catalogue; stdlib only. Never discard unresolved observations.

The canonical source is the union of input observations and explicitly enumerated
repository bibliographies/assets. An arXiv lookup verifies metadata, NOT scientific
results. Existing paper PDFs are indexed in place and never copied or relicensed.
"""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, difflib, hashlib, html, io, json
import os, re, sqlite3, subprocess, sys, time, unicodedata
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

VERSION = '1.1.0'
ARXIV = re.compile(r'(?<![\d.])((?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v\d+)?)(?!\d)', re.I)
NEWID = re.compile(r'^\d{4}\.\d{4,5}$')
PAPER_KINDS = {'paper','preprint','article','inproceedings','conference','book','thesis','misc','repository-paper','unresolved-paper'}
REPORT = Path('deliverables/neural-memory/reports/2026-10-06-sleep-time-compute/source/sources/references.json')

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9가-힣]+','',unicodedata.normalize('NFKC',s).casefold())

def aid(s: str) -> str:
    m=ARXIV.search(str(s or ''))
    return re.sub(r'v\d+$','',m.group(1)) if m else ''

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def read_json(path: Path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default

def date_value(value: str) -> tuple[str,str]:
    """Partial dates remain partial. Never infer a date from an arXiv number."""
    s=str(value or '').strip()
    m=re.search(r'(?<!\d)((?:19|20)\d{2})(?:[-./](\d{2})(?:[-./](\d{2}))?)?(?!\d)',s)
    if not m: return '', 'unknown'
    y,mo,day=m.groups()
    try:
        if day: dt.date(int(y),int(mo),int(day)); return f'{y}-{mo}-{day}','day'
        if mo and 1<=int(mo)<=12: return f'{y}-{mo}','month'
        if not mo: return y,'year'
    except ValueError: pass
    return '', 'unknown'

def sort_key(p):
    d=p['publication_date'] or ''
    return (not bool(d),d+'-01'*(2-d.count('-')) if d else '9999-12-31',p['title'].casefold(),p['id'])

def same_title(expected: str, actual: str) -> bool:
    x,y=norm(expected),norm(actual)
    if not x or not y: return False
    if x==y: return True
    if len(x)>=7 and (x in y or y in x): return True
    ex=norm(expected.split(':')[0]); ac=norm(actual.split(':')[0])
    if len(ex)>=3 and ex==ac: return True
    return difflib.SequenceMatcher(None,x,y).ratio()>=0.83

class Importer:
    def __init__(self,root:Path,lit:Path):
        self.root,self.lit=root,lit
        self.rows=[]; self.sources={}; self.assets=[]; self.warnings=[]; self.manifest=read_json(lit/'inputs/source_manifest.json',{})
    def source(self,key,path='',kind='repository',date='',expected=None):
        if key not in self.sources:
            p=self.root/path if path else None
            self.sources[key]={'id':key,'path':path,'kind':kind,'source_date':date,'input_count':0,
                               'sha256':sha(p.read_bytes()) if p and p.is_file() else None,
                               'expected_count':expected}
        return key
    def add(self,source,**kw):
        self.source(source)
        row={k:'' for k in ['title','alias','reported_published','reported_updated','reported_arxiv','reported_url','topics','note','canonical_arxiv','source_date']}
        row.update(kw); row['source_id']=source; row.setdefault('kind','paper')
        self.sources[source]['input_count']+=1
        row.setdefault('observation_id',source+':'+str(self.sources[source]['input_count']))
        row['kind']=row.get('kind') or 'paper'; row['scope']=row.get('scope') or ('task' if source.startswith(('weekly-','initial-','report-','legacy-','precursor-')) else 'repository_foundation')
        self.rows.append(row)
    def load_task(self):
        path=self.lit/'inputs/task_bibliography.tsv'
        with path.open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
        for row in rows:
            source=row.pop('source_id'); self.source(source,kind='recovered-task',date=row.get('source_date',''))
            self.add(source,**row)
        self.sources['task-input-file']={'id':'task-input-file','path':str(path.relative_to(self.root)),'kind':'input-container','source_date':'','input_count':0,'sha256':sha(path.read_bytes()),'expected_count':len(rows)}
        snapshot=read_json(self.lit/'inputs/prior_report_references.json',[])
        repo_ref=read_json(self.root/REPORT,[])
        if repo_ref and snapshot != repo_ref:
            self.warnings.append('The preserved report reference snapshot differs from the current repository copy; both are imported.')
            for r in repo_ref: self.add('repository-report-current',title=r.get('title',''),alias=r.get('key',''),reported_arxiv=r.get('arxiv','') or '',reported_published=r.get('date',''),reported_updated=r.get('updated',''),reported_url=r.get('url',''),kind=r.get('type','paper'),note=r.get('note',''))
        elif snapshot and not repo_ref:
            self.warnings.append('The original report reference path is unavailable in this checkout; preserved reference snapshot was used.')
    def load_corpus(self):
        path=Path('research/ttt-efficient-corpus.json')
        corpus=read_json(self.root/path,[])
        if not corpus: self.warnings.append('TTT corpus unavailable in this checkout.')
        self.source('repository-ttt-corpus',str(path),'curated-corpus',expected=len(corpus))
        for r in corpus:
            self.add('repository-ttt-corpus',title=r.get('name',''),alias=r.get('id',''),reported_arxiv=r.get('arxiv',''),reported_published=r.get('date',''),topics=','.join(filter(None,[r.get('axis',''),r.get('family','')])),note=r.get('claim',''),scope='repository_foundation')
    def load_bibliographies(self):
        paths=sorted(set(list(self.root.glob('**/*bibliography*.md'))+list(self.root.glob('**/*references*.bib'))+list(self.root.glob('**/*.bib'))))
        for path in paths:
            rel=path.relative_to(self.root)
            if any(x in rel.parts for x in ('.git','literature','node_modules','.venv')): continue
            text=path.read_text(encoding='utf-8',errors='replace')
            source='bibliography:'+str(rel); self.source(source,str(rel),'authored-bibliography')
            if path.suffix=='.bib':
                for key,block in bib_entries(text):
                    field=lambda name: bib_field(block,name)
                    title=field('title') or key
                    arxiv=aid(field('eprint') or field('url'))
                    url=field('url') or ('https://doi.org/'+field('doi') if field('doi') else '')
                    self.add(source,title=title,alias=key,reported_arxiv=arxiv,reported_url=url,reported_published=field('date') or field('year'),authors=field('author'),note='Imported BibTeX entry; original review claims were not independently reproduced.',doi=field('doi'))
                continue
            blocks=re.split(r'\n\s*\n|\n(?=\s*[-+]\s+)',text)
            seen=set()
            for i,block in enumerate(blocks):
                ids=list(dict.fromkeys(aid(m) for m in ARXIV.findall(block)))
                if len(ids)>1 and '|' in block and len(block.splitlines())>1:
                    blocks.extend([x for x in block.splitlines() if ARXIV.search(x)])
                    continue
                titled=[' '.join(t.split()) for t in re.findall(r'(?<!\*)\*([^*]{8,})\*(?!\*)',block)]
                for arxiv in ids:
                    if (arxiv,block) in seen: continue
                    seen.add((arxiv,block))
                    title=titled[0].rstrip('.') if titled else ''
                    if not title: title='arXiv:'+arxiv
                    dates=re.findall(r'(?<!\d)(?:19|20)\d{2}-\d{2}(?:-\d{2})?',block)
                    pub=dates[0] if dates and not re.search(r'\bv[2-9]\b',block[:block.find(dates[0])]) else ''
                    self.add(source,title=title,reported_arxiv=arxiv,reported_published=pub,reported_url='https://arxiv.org/abs/'+arxiv,note=block.strip(),source_locator=f'paragraph:{i+1}')
                if not ids and titled:
                    title=titled[0].rstrip('.')
                    doi_m=re.search(r'(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s<>]+)',block,re.I)
                    url_m=re.search(r'https?://[^\s<>]+',block)
                    if not (doi_m or re.search(r'\b(?:19|20)\d{2}\b',block)): continue
                    self.add(source,title=title,doi=doi_m.group(1).rstrip('.,)*') if doi_m else '',reported_url=url_m.group(0).rstrip('.,)*') if url_m else '',reported_published=date_value(block)[0],note=block.strip())
    def load_authored_mentions(self):
        # Authored citations only, not references inside downloaded papers or translations.
        pattern=re.compile(r'(?:arxiv\.org/(?:abs|pdf|html)/|arxiv\s*:\s*)(\d{4}\.\d{4,5}(?:v\d+)?)',re.I)
        for folder in ['research','study-kr','study-kr-stc','notes','docs','dossier','claims','audit','asi-blueprint','multiarch']:
            base=self.root/folder
            if not base.is_dir(): continue
            for path in sorted(base.rglob('*.md')):
                if 'bibliography' in path.name.lower() or path.name=='ttt-efficient-corpus.md': continue
                rel=str(path.relative_to(self.root))
                text=path.read_text(encoding='utf-8',errors='replace')
                matches={aid(m.group(1)):text.count('\n',0,m.start())+1 for m in pattern.finditer(text)}
                if not matches: continue
                source='authored-citations:'+rel;self.source(source,rel,'authored-citation-discovery')
                for a,line in sorted(matches.items()):
                    self.add(source,title='arXiv:'+a,reported_arxiv=a,reported_url='https://arxiv.org/abs/'+a,source_locator=f'line:{line}',note='Identifier explicitly cited in an authored repository document. This occurrence is not a full-text-review assertion.',scope='repository_mention')
    def load_assets(self):
        base=self.root/'papers'
        if not base.is_dir():
            self.warnings.append('papers/ unavailable; original-PDF inventory not checked in this checkout.'); return
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in ('.pdf','.txt'): continue
            rel=str(p.relative_to(self.root)); a=aid(p.name)
            asset={'path':rel,'arxiv_id':a,'type':p.suffix[1:],'size_bytes':p.stat().st_size,'sha256':None,'rights':'existing_repository_asset_not_relicensed','identifier_basis':'filename_arxiv_id' if a else 'filename_only','content_identity_verified':False}
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            asset['sha256']=h.hexdigest(); self.assets.append(asset)
            if p.suffix.lower()!='.pdf': continue
            title='arXiv:'+a if a else p.stem.replace('_',' ')
            txt=p.with_suffix('.txt')
            if txt.exists():
                head=txt.read_text(encoding='utf-8',errors='replace')[:1600]
                lines=[l.strip() for l in head.splitlines() if l.strip()]
                usable=[l for l in lines[:18] if len(l)>12 and not re.search(r'arxiv:|https?://|@|^\d|^abstract|^published|^under review',l,re.I)]
                if usable: title=usable[0]
            self.add('repository-pdf-assets',title=title,reported_arxiv=a,reported_url='https://arxiv.org/abs/'+a if a else '',note='An original PDF already exists in papers/. Header text is only a fallback label, not identity verification.',source_locator=rel,scope='repository_asset')
    def load(self):
        self.load_task(); self.load_corpus(); self.load_bibliographies(); self.load_authored_mentions(); self.load_assets()
        ids=[r['observation_id'] for r in self.rows]
        if len(set(ids))!=len(ids): raise ValueError('Duplicate observation IDs')
        return self

def bib_entries(text):
    pos=0
    while True:
        m=re.search(r'@([A-Za-z]+)\s*\{\s*([^,]+),',text[pos:])
        if not m: return
        start=pos+m.end(); level=1; end=start
        while end<len(text) and level:
            if text[end]=='{' and (end==0 or text[end-1]!='\\'): level+=1
            elif text[end]=='}' and (end==0 or text[end-1]!='\\'): level-=1
            end+=1
        if level: return
        yield m.group(2).strip(),text[start:end-1]
        pos=end

def bib_field(block,name):
    m=re.search(r'\b'+name+r'\s*=\s*',block,re.I)
    if not m: return ''
    i=m.end()
    if i>=len(block): return ''
    c=block[i]
    if c=='{':
        level=1;j=i+1
        while j<len(block) and level:
            if block[j]=='{' and block[j-1]!='\\': level+=1
            if block[j]=='}' and block[j-1]!='\\': level-=1
            j+=1
        return block[i+1:j-1].replace('{','').replace('}','').strip()
    if c=='"':
        j=i+1
        while j<len(block) and not (block[j]=='"' and block[j-1]!='\\'): j+=1
        return block[i+1:j].strip()
    return block[i:].split(',')[0].strip()

NS={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
def get_url(url,timeout=25):
    req=Request(url,headers={'User-Agent':'neural-memory-study-catalog/1.0 (bibliographic audit; https://github.com/jimmylegendary/neural-memory-study)'})
    with urlopen(req,timeout=timeout) as res:
        data=res.read(8*1024*1024)
        return data,res.geturl()

def parse_atom(data,checked):
    root=ET.fromstring(data); out={}
    for entry in root.findall('a:entry',NS):
        text=lambda k: ' '.join((entry.findtext(k,default='',namespaces=NS)).split())
        rawid=text('a:id'); a=aid(rawid)
        if not a: continue
        ver=re.search(r'v(\d+)$',rawid)
        out[a]={'arxiv_id':a,'title':text('a:title'),'summary':text('a:summary'),'authors':[x.findtext('a:name',default='',namespaces=NS) for x in entry.findall('a:author',NS)],'published':text('a:published'),'updated':text('a:updated'),'latest_version':int(ver.group(1)) if ver else None,'doi':text('x:doi'),'journal_ref':text('x:journal_ref'),'comment':text('x:comment'),'categories':[x.attrib.get('term','') for x in entry.findall('a:category',NS)],'url':'https://arxiv.org/abs/'+a,'checked_at':checked,'method':'arxiv_atom_metadata','status':'ok'}
    return out

def parse_abs(data,a,checked):
    text=data.decode('utf-8',errors='replace')
    metas=collections.defaultdict(list)
    for tag in re.findall(r'<meta\b[^>]+>',text,re.I):
        attrs=dict((k.lower(),html.unescape(v)) for k,_,v in re.findall(r'([\w:-]+)\s*=\s*([\"\'])(.*?)\2',tag,re.S))
        if attrs.get('name') and 'content' in attrs: metas[attrs['name'].lower()].append(attrs['content'])
    if not metas.get('citation_title'): raise ValueError('No citation_title in arXiv abstract page')
    title=metas['citation_title'][0]
    published=metas.get('citation_date',[''])[0].replace('/','-')
    versions=[]
    flat=html.unescape(re.sub('<[^>]+>',' ',text))
    for ver,datestr in re.findall(r'\[v(\d+)\]\s*((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+UTC)',flat):
        try: date=dt.datetime.strptime(datestr,'%a, %d %b %Y %H:%M:%S UTC').isoformat()+'Z'
        except ValueError: continue
        versions.append({'version':int(ver),'date':date})
    versions.sort(key=lambda x:x['version'])
    if versions: published=versions[0]['date']
    return {'arxiv_id':a,'title':title,'authors':metas.get('citation_author',[]),'published':published,'updated':versions[-1]['date'] if versions else '', 'latest_version':versions[-1]['version'] if versions else None,'versions':versions,'doi':metas.get('citation_doi',[''])[0],'url':'https://arxiv.org/abs/'+a,'checked_at':checked,'method':'arxiv_abstract_metadata','status':'ok'}

def enrich(ids,cache,checked,deadline_seconds=420):
    errors=[]; start=time.monotonic(); todo=[a for a in sorted(ids) if cache.get(a,{}).get('checked_at')!=checked or cache.get(a,{}).get('status')!='ok' or cache.get(a,{}).get('method')=='arxiv_search_metadata' or 'summary' not in cache.get(a,{})]
    for i in range(0,len(todo),20):
        if time.monotonic()-start>deadline_seconds: break
        batch=todo[i:i+20]; url='https://export.arxiv.org/api/query?'+urlencode({'id_list':','.join(batch),'max_results':len(batch)})
        try:
            data,_=get_url(url); fresh=parse_atom(data,checked)
            for m in fresh.values():m['response_sha256']=sha(data);m['retrieved_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            cache.update(fresh)
            for a in batch:
                if a not in fresh: errors.append({'arxiv_id':a,'error':'not_returned_by_api','url':url})
            print(f'arXiv metadata {i+len(batch)}/{len(todo)}: {len(fresh)} returned',flush=True)
        except Exception as exc:
            errors.append({'batch':batch,'url':url,'error':str(exc)[:300]}); print('arXiv API:',str(exc)[:100],flush=True)
        time.sleep(3.1)
    missing=[a for a in sorted(ids) if cache.get(a,{}).get('status')!='ok']
    for a in missing:
        if time.monotonic()-start>deadline_seconds: break
        url='https://arxiv.org/abs/'+a
        try:
            data,_=get_url(url,timeout=12); cache[a]=parse_abs(data,a,checked);cache[a]['response_sha256']=sha(data);cache[a]['retrieved_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        except Exception as exc: errors.append({'arxiv_id':a,'url':url,'error':str(exc)[:300]})
        time.sleep(3.1)
    return cache,errors

def resolve_rows(rows,resolutions,refs):
    by_alias={norm(k):v for k,v in resolutions.get('aliases',{}).items()}
    for r in refs:
        a=aid(r.get('arxiv',''))
        if a:
            by_alias.setdefault(norm(r['title']),a); by_alias.setdefault(norm(r['key']),a)
    corrections=[]
    for row in rows:
        original=aid(row.get('reported_arxiv',''))
        keys=([norm(row.get('alias',''))] if row.get('scope')=='task' else [])+[norm(row.get('title',''))]
        row['resolved_url']=next((v for k,v in resolutions.get('urls',{}).items() if norm(k) in keys),'')
        selected=row.get('canonical_arxiv') or next((by_alias[x] for x in keys if x and x in by_alias),'') or original
        row['canonical_arxiv']=aid(selected)
        if original and row['canonical_arxiv'] and original!=row['canonical_arxiv']:
            corrections.append({'observation_id':row['observation_id'],'title':row['title'],'reported_arxiv':original,'resolved_arxiv':row['canonical_arxiv'],'basis':'explicit_title_or_alias_resolution','checked_at':resolutions.get('checked_at','')})
    return corrections

def resolve_title_queries(rows,cache,checked):
    seen=set();log=[]
    for row in rows:
        title=row['title']
        if row.get('canonical_arxiv') or row.get('scope')!='task' or row['kind'] not in PAPER_KINDS or len(title)<16 or norm(title) in seen: continue
        seen.add(norm(title))
        if len(seen)>12: break
        query='ti:"'+re.sub(r'[^\w -]',' ',title)+'"'
        url='https://export.arxiv.org/api/query?'+urlencode({'search_query':query,'max_results':5})
        try:
            data,_=get_url(url,timeout=12);found=parse_atom(data,checked)
            candidates=[(a,m) for a,m in found.items() if difflib.SequenceMatcher(None,norm(title),norm(m['title'])).ratio()>=0.92]
            if len(candidates)==1:
                a,m=candidates[0];cache[a]=m
                for r in rows:
                    if not r.get('canonical_arxiv') and (norm(r['title'])==norm(title) or (r.get('alias') and r['alias']==row.get('alias'))):r['canonical_arxiv']=a
                log.append({'title':title,'arxiv_id':a,'method':'unique_primary_title_match'})
            else:log.append({'title':title,'method':'primary_title_lookup','status':'ambiguous_or_not_found'})
        except Exception as exc:log.append({'title':title,'method':'primary_title_lookup','status':'lookup_error','error':str(exc)[:200]})
        time.sleep(3.1)
    return log

def build_publications(rows,cache,assets,resolutions):
    groups=collections.defaultdict(list)
    doi_alias={norm(k):v for k,v in resolutions.get('dois',{}).items()}
    for row in rows:
        a=row['canonical_arxiv']; doi=row.get('doi','') or doi_alias.get(norm(row['title']),'')
        kind='paper' if row['kind'] in PAPER_KINDS else 'other'
        if a: pid='arxiv:'+a
        elif doi: pid='doi:'+doi.casefold()
        else: pid=kind+':'+hashlib.sha256(norm(row['title']).encode()).hexdigest()[:16]
        row['publication_id']=pid; groups[pid].append(row)
    pubs=[]
    for pid,obs in groups.items():
        a=obs[0]['canonical_arxiv']; m=cache.get(a,{})
        aliases=sorted(set(r['alias'] for r in obs if r.get('alias')))
        titles=sorted(set(r['title'] for r in obs if r.get('title')))
        informative=sorted(set(r['title'] for r in obs if r.get('title') and not r['title'].lower().startswith('arxiv:') and r.get('scope')!='repository_asset'))
        title=max(informative or titles,key=len) if titles else pid
        title_inputs=informative+aliases
        alias_in_abstract=any(len(alias)>=4 and len(alias)<=24 and re.search(r'(?<![\w-])'+re.escape(alias)+r'(?![\w-])',m.get('summary',''),re.I) for alias in aliases if alias not in ('Memory','Agent','Sleep','Graph'))
        is_match=m.get('status')=='ok' and (not informative or any(same_title(t,m.get('title','')) for t in title_inputs) or alias_in_abstract)
        if is_match: title=m['title']; status='primary_metadata_verified'
        elif m.get('status')=='ok': status='identity_conflict'
        elif a: status='metadata_pending'
        else: status='identifier_pending' if any(r['kind'] in PAPER_KINDS for r in obs) else 'non_paper'
        pubdate,precision=date_value(m.get('published','')) if is_match else ('','unknown')
        date_basis='arxiv_v1' if pubdate else ''
        if not pubdate:
            reported=sorted({date_value(r.get('reported_published','')) for r in obs},key=lambda p:(-len(p[0]),p[0]))
            known=[d for d in reported if d[0]]
            if known: pubdate,precision=known[0]; date_basis='historical_report_unverified'
        scopes=sorted(set(r['scope'] for r in obs))
        first_seen=sorted(r.get('source_date','') for r in obs if r.get('source_date') and r['scope']=='task')
        p={'id':pid,'title':title,'aliases':aliases,'historical_titles':titles,'kind':'paper' if any(r['kind'] in PAPER_KINDS for r in obs) else 'other','arxiv_id':a or None,'doi':m.get('doi') if is_match else (obs[0].get('doi') or doi_alias.get(norm(title)) or None),'authors':m.get('authors',[]) if is_match else [],'publication_date':pubdate or None,'date_precision':precision,'date_basis':date_basis or 'unknown','updated_at':m.get('updated') if is_match else None,'latest_version':m.get('latest_version') if is_match else None,'verification_status':status,'review_status':'metadata_only_no_result_reproduction','checked_at':m.get('checked_at') if is_match else None,'url':('https://arxiv.org/abs/'+a) if a else next((r.get('resolved_url') or r.get('reported_url') for r in obs if r.get('resolved_url') or r.get('reported_url')),''),'metadata_candidate':m if status=='identity_conflict' else None,'topics':sorted(set(t.strip() for r in obs for t in r.get('topics','').split(',') if t.strip())),'scopes':scopes,'first_seen_in_recovered_task':first_seen[0] if first_seen else None,'source_ids':sorted(set(r['source_id'] for r in obs)),'observation_ids':[r['observation_id'] for r in obs],'notes':[{'source_id':r['source_id'],'text':r['note']} for r in obs if r.get('note')],'assets':[s['path'] for s in assets if a and s['arxiv_id']==a]}
        manual=resolutions.get('publisher_metadata',{}).get(pid)
        if manual:
            p.update(manual);p['verification_status']='publisher_metadata_verified'
        pubs.append(p)
    pubs.sort(key=sort_key)
    for asset in assets:
        a=asset['arxiv_id']; asset['publication_id']='arxiv:'+a if a else next((r['publication_id'] for r in rows if r.get('source_locator')==asset['path']),None)
    return pubs

SCHEMA='''PRAGMA foreign_keys=ON;
CREATE TABLE publications(id TEXT PRIMARY KEY,title TEXT NOT NULL,kind TEXT NOT NULL,arxiv_id TEXT UNIQUE,doi TEXT,publication_date TEXT,date_precision TEXT,date_basis TEXT,updated_at TEXT,latest_version INTEGER,verification_status TEXT,review_status TEXT,checked_at TEXT,url TEXT,first_seen_in_recovered_task TEXT,authors_json TEXT,aliases_json TEXT,scopes_json TEXT,metadata_candidate_json TEXT);
CREATE TABLE sources(id TEXT PRIMARY KEY,path TEXT,kind TEXT,source_date TEXT,input_count INTEGER,sha256 TEXT);
CREATE TABLE observations(id TEXT PRIMARY KEY,publication_id TEXT NOT NULL REFERENCES publications(id),source_id TEXT NOT NULL REFERENCES sources(id),source_locator TEXT,source_date TEXT,reported_title TEXT,reported_alias TEXT,reported_published TEXT,reported_updated TEXT,reported_arxiv TEXT,reported_url TEXT,note TEXT,scope TEXT);
CREATE TABLE topics(publication_id TEXT REFERENCES publications(id),topic TEXT,PRIMARY KEY(publication_id,topic));
CREATE TABLE corrections(observation_id TEXT REFERENCES observations(id),reported_arxiv TEXT,resolved_arxiv TEXT,basis TEXT,checked_at TEXT);
CREATE TABLE assets(path TEXT PRIMARY KEY,publication_id TEXT REFERENCES publications(id),type TEXT,size_bytes INTEGER,sha256 TEXT,rights TEXT);
CREATE TABLE metadata_snapshots(arxiv_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL);
CREATE TABLE versions(publication_id TEXT REFERENCES publications(id),version INTEGER,date TEXT,source_url TEXT,PRIMARY KEY(publication_id,version));
CREATE TABLE build_info(key TEXT PRIMARY KEY,value TEXT);
CREATE INDEX by_publication_date ON publications(publication_date,title);
CREATE INDEX by_source ON observations(source_id);
CREATE INDEX by_status ON publications(verification_status);
CREATE VIEW chronological AS SELECT * FROM publications WHERE kind='paper' ORDER BY publication_date IS NULL,publication_date,title;
CREATE VIEW task_papers AS SELECT DISTINCT p.* FROM publications p JOIN observations o ON o.publication_id=p.id WHERE p.kind='paper' AND o.scope='task' ORDER BY p.publication_date IS NULL,p.publication_date,p.title;
CREATE VIEW unresolved AS SELECT * FROM publications WHERE verification_status IN ('identity_conflict','metadata_pending','identifier_pending');
'''

def write_outputs(lit,pubs,imp,corrections,cache,network_errors,checked,commit):
    out=lit/'data';out.mkdir(exist_ok=True,parents=True)
    dbpath=out/'papers.sqlite'; tmp=out/'papers.sqlite.tmp'
    tmp.unlink(missing_ok=True); conn=sqlite3.connect(tmp);conn.executescript(SCHEMA)
    j=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True)
    for p in pubs:
        vals=[p[x] for x in ['id','title','kind','arxiv_id','doi','publication_date','date_precision','date_basis','updated_at','latest_version','verification_status','review_status','checked_at','url','first_seen_in_recovered_task']]+[j(p['authors']),j(p['aliases']),j(p['scopes']),j(p['metadata_candidate'])]
        conn.execute('INSERT INTO publications VALUES('+','.join('?'*len(vals))+')',vals)
        for topic in p['topics']:conn.execute('INSERT INTO topics VALUES(?,?)',(p['id'],topic))
    for s in imp.sources.values():conn.execute('INSERT INTO sources VALUES(?,?,?,?,?,?)',tuple(s[k] for k in ['id','path','kind','source_date','input_count','sha256']))
    for r in imp.rows:
        conn.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(r['observation_id'],r['publication_id'],r['source_id'],r.get('source_locator',''),r.get('source_date',''),r['title'],r['alias'],r['reported_published'],r['reported_updated'],r['reported_arxiv'],r['reported_url'],r['note'],r['scope']))
    for c in corrections:conn.execute('INSERT INTO corrections VALUES(?,?,?,?,?)',tuple(c[k] for k in ['observation_id','reported_arxiv','resolved_arxiv','basis','checked_at']))
    for a in imp.assets:conn.execute('INSERT INTO assets VALUES(?,?,?,?,?,?)',tuple(a[k] for k in ['path','publication_id','type','size_bytes','sha256','rights']))
    for a,m in sorted(cache.items()):
        conn.execute('INSERT INTO metadata_snapshots VALUES(?,?)',(a,j(m)))
        if any(p['id']=='arxiv:'+a for p in pubs):
            for v in m.get('versions',[]):conn.execute('INSERT OR IGNORE INTO versions VALUES(?,?,?,?)',('arxiv:'+a,v['version'],v['date'],m.get('url','')))
    for k,v in {'version':VERSION,'built_at':checked,'input_commit':commit,'content_review':'metadata indexing only; no experiment reproduction'}.items():conn.execute('INSERT INTO build_info VALUES(?,?)',(k,v))
    try:
        conn.execute('CREATE VIRTUAL TABLE paper_fts USING fts5(publication_id UNINDEXED,title,aliases,topics,notes)')
        conn.executemany('INSERT INTO paper_fts VALUES(?,?,?,?,?)',[(p['id'],p['title'],' '.join(p['aliases']),' '.join(p['topics']),' '.join(n['text'] for n in p['notes'])) for p in pubs])
        fts=True
    except sqlite3.OperationalError:fts=False
    conn.commit()
    assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    assert conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==len(imp.rows)
    conn.close();os.replace(tmp,dbpath)
    with (out/'papers.jsonl').open('w',encoding='utf-8') as f:
        for p in pubs:f.write(j(p)+'\n')
    disagreements=[]
    pmap={p['id']:p for p in pubs}
    for r in imp.rows:
        p=pmap[r['publication_id']]; old,precision=date_value(r['reported_published'])
        if p['date_basis']=='arxiv_v1' and old and not p['publication_date'].startswith(old):
            disagreements.append({'observation_id':r['observation_id'],'field':'publication_date','reported':r['reported_published'],'primary':p['publication_date'],'basis':'arxiv_v1','note':'May reflect an earlier venue release, a version date, or a historical error; not automatically overwritten in the observation.'})
    dump(out/'field_disagreements.json',disagreements)
    dump(out/'observations.json',imp.rows);dump(out/'sources.json',list(imp.sources.values()));dump(out/'corrections.json',corrections);dump(out/'assets.json',imp.assets)
    fields=['id','publication_date','date_precision','date_basis','title','aliases','authors','arxiv_id','doi','updated_at','latest_version','topics','verification_status','scopes','first_seen_in_recovered_task','url','assets']
    with (out/'papers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in pubs:writer.writerow({k:'; '.join(map(str,p[k])) if isinstance(p[k],list) else p[k] for k in fields})
    paper=[p for p in pubs if p['kind']=='paper']; others=[p for p in pubs if p['kind']!='paper']; task=[p for p in paper if 'task' in p['scopes']]
    status=dict(collections.Counter(p['verification_status'] for p in paper))
    unmapped=[r['observation_id'] for r in imp.rows if not r.get('publication_id')]
    latest=[r for r in imp.rows if r['source_id']=='weekly-2026-10-12']
    coverage={'schema_version':VERSION,'built_at':checked,'input_commit':commit,'scope_statement':'All recovered input observations, existing explicit repository bibliographies, curated TTT corpus and original PDF assets. Not a claim of access to every historical chat or every cited reference inside original papers.','publication_count':len(pubs),'paper_count':len(paper),'task_paper_count':len(task),'other_material_count':len(others),'observation_count':len(imp.rows),'mapped_observation_count':len(imp.rows)-len(unmapped),'unmapped_observations':unmapped,'source_count':len([s for s in imp.sources.values() if s['input_count']]),'status_counts':status,'unknown_date_count':sum(not p['publication_date'] for p in paper),'historical_date_count':sum(p['date_basis']=='historical_report_unverified' for p in paper),'pdf_asset_count':sum(a['type']=='pdf' for a in imp.assets),'correction_count':len(corrections),'date_disagreement_count':len(disagreements),'latest_week_observation_count':len(latest),'latest_week_mapped_count':sum(bool(r.get('publication_id')) for r in latest),'source_counts':{s['id']:s['input_count'] for s in imp.sources.values() if s['input_count']},'fulltext_or_result_reproduction_performed':False,'fts5':fts,'warnings':imp.warnings,'network_error_count':len(network_errors),'network_errors':network_errors}
    dump(lit/'coverage.json',coverage);dump(lit/'inputs/metadata_cache.json',cache)
    def cell(x):return str(x or '').replace('|','\\|').replace('\n',' ')
    def table(rows):
        lines=['| 최초 공개일 / 근거 | 논문 | 식별자 | 수정일 | 검증 상태 |','|---|---|---|---|---|']
        for p in rows:
            d=p['publication_date'] or '미확인'; d+= ' (기록값)' if p['date_basis']=='historical_report_unverified' else ''
            title=cell(p['title']);u=p['url']; title=f'[{title}]({u})' if u.startswith(('https://','http://')) else title
            lines.append(f"| {d} | {title} | {p['arxiv_id'] or p['doi'] or '미확정'} | {p['updated_at'] or '미확인'} | {p['verification_status']} |")
        return '\n'.join(lines)
    header=f'# 논문 발행일순 대장\n\n기준일 {checked} · 논문 {len(paper)}건 · task 고유 논문 후보 {len(task)}건 · 비논문 자료 {len(others)}건 별도.\n\n정렬은 확인된 arXiv v1/발행처 최초 공개일을 우선 사용한다. **(기록값)**은 과거 요약에 적힌 미검증 날짜이며, 월/연도만 아는 날짜에 일을 만들어 넣지 않는다. 수정일은 별도 열이다. 메타데이터 확인은 본문·성과 재현을 뜻하지 않는다.\n\n[검색 화면](index.html) · [SQLite](data/papers.sqlite) · [CSV](data/papers.csv) · [누락 대조](coverage.json) · [미검증 대장](UNRESOLVED.md)\n\n'
    (lit/'CHRONOLOGICAL.md').write_text(header+'## 전체 논문 — 발행일순\n\n'+table(paper)+'\n',encoding='utf-8')
    (lit/'TASK_CHRONOLOGICAL.md').write_text(header+'## 이 task의 회수 문헌만 — 발행일순\n\n'+table(task)+'\n',encoding='utf-8')
    pending=[p for p in paper if p['verification_status'] not in ('primary_metadata_verified','publisher_metadata_verified')]
    (lit/'UNRESOLVED.md').write_text('# 미검증·식별자 충돌 대장\n\n검색 실패는 논문 부재의 증명이 아니다. 항목은 삭제하지 않는다. 과거 제목/ID는 observations에, 조회된 다른 제목은 metadata_candidate에 보존한다.\n\n'+table(pending)+'\n',encoding='utf-8')
    (out/'schema.sql').write_text(SCHEMA,encoding='utf-8')
    write_html(lit,pubs,coverage)
    hashes=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS':hashes.append(sha(p.read_bytes())+'  '+p.name)
    (out/'SHA256SUMS').write_text('\n'.join(hashes)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in coverage.items() if k not in ('network_errors','source_counts')},ensure_ascii=False,indent=2))
    return coverage

def write_html(lit,pubs,coverage):
    # JSON escaping and textContent keep untrusted titles/notes out of HTML markup.
    keys=['id','title','authors','aliases','kind','publication_date','date_basis','updated_at','verification_status','url','topics','scopes','source_ids','assets']
    payload=json.dumps([{k:p[k] for k in keys} for p in pubs],ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    doc=r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neural Memory — Paper DB</title>
<style>
body{font:15px/1.6 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 22px;color:#162b3d;background:#f5f7fa}h1{font-size:32px}header{border-bottom:3px solid #137d88;margin-bottom:25px}input,select{padding:11px;border:1px solid #adb8c3;border-radius:5px;font:inherit;margin:4px}input{width:min(520px,90%)}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;border-bottom:1px solid #dce2e7;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#162b3d;color:white}.small{font-size:12px;color:#536778}a{color:#006875}#counter{font-weight:600;margin:12px 0}details{max-width:720px}footer{margin:35px 0}@media(max-width:750px){th:nth-child(4),td:nth-child(4){display:none}td,th{padding:8px}}
</style>
<header><p>NEURAL-MEMORY-STUDY / RESEARCH CATALOGUE</p><h1>Sleep-time Compute · Agent Memory 논문 DB</h1><p>발행일과 수정일, 문헌의 실재 확인과 실험 재현을 구분합니다. 원문은 링크 또는 기존 저장소 경로로 연결합니다.</p></header>
<input id="q" placeholder="제목 · 저자 · 별칭 · arXiv · 주제 검색">
<select id="scope"><option value="all">전체 수집 범위</option><option value="task">이 task</option><option value="foundation">선행 저장소 문헌</option></select>
<select id="kind"><option value="paper">논문</option><option value="all">논문 + 기타 자료</option></select>
<select id="status"><option value="all">모든 검증 상태</option><option value="verified">메타데이터 확인</option><option value="pending">미검증 / 충돌</option></select>
<select id="order"><option value="asc">발행일 오래된 순</option><option value="desc">발행일 최신 순</option></select>
<div id="counter"></div><p class="small">날짜 뒤 (기록값)은 과거 요약의 미검증 날짜. 날짜 미확인 항목은 맨 끝. 상세 출처와 원문 보관 경로는 펼쳐서 확인하세요. GitHub에서는 HTML을 내려받아 브라우저로 여세요.</p>
<table><thead><tr><th>최초 공개일</th><th>논문 / 출처</th><th>상태</th><th>수정일</th></tr></thead><tbody id="body"></tbody></table>
<footer><a href="data/papers.sqlite">SQLite</a> · <a href="data/papers.csv">CSV</a> · <a href="data/papers.jsonl">JSONL</a> · <a href="coverage.json">전수 대조 통계</a><p class="small">검색은 이 HTML 안에서만 처리되며 외부 서버에 질의를 전송하지 않습니다. PDF 파일 자체를 새로 재배포한 DB가 아닙니다.</p></footer>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
const safeURL=u=>/^https?:\/\//.test(u);
function render(){
 let q=$('q').value.toLowerCase();
 let rows=data.filter(p=>($('kind').value==='all'||p.kind==='paper')&&($('scope').value==='all'||($('scope').value==='task'?p.scopes.includes('task'):!p.scopes.includes('task')))&&JSON.stringify(p).toLowerCase().includes(q));
 if($('status').value!=='all')rows=rows.filter(p=>((p.verification_status==='primary_metadata_verified'||p.verification_status==='publisher_metadata_verified')===($('status').value==='verified')));
 rows.sort((a,b)=>{if(!a.publication_date)return b.publication_date?1:0;if(!b.publication_date)return -1;return (a.publication_date.localeCompare(b.publication_date)||a.title.localeCompare(b.title))*($('order').value==='asc'?1:-1)});
 $('counter').textContent=rows.length+' / '+data.length+' 항목';$('body').replaceChildren();
 for(const p of rows){
  let tr=document.createElement('tr'),d=document.createElement('td');
  d.textContent=(p.publication_date||'미확인')+(p.date_basis==='historical_report_unverified'?' (기록값)':'');tr.append(d);
  let title=document.createElement('td'),a=document.createElement(safeURL(p.url)?'a':'span');a.textContent=p.title;
  if(safeURL(p.url)){a.href=p.url;a.target='_blank';a.rel='noopener'}title.append(a);
  let detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent=p.id+' · '+p.topics.join(' / ');detail.append(summary);
  for(const [label,v] of [['저자',p.authors],['별칭',p.aliases],['출처',p.source_ids],['기존 원문 파일',p.assets]]){
   let item=document.createElement('p');item.className='small';item.textContent=label+': '+(v.join('; ')||'미확인');detail.append(item)
  }
  title.append(detail);tr.append(title);
  let status=document.createElement('td');status.textContent=p.verification_status;status.className='small';tr.append(status);
  let updated=document.createElement('td');updated.textContent=p.updated_at||'미확인';tr.append(updated);$('body').append(tr)
 }
}
for(const id of ['q','scope','kind','status','order'])$(id).addEventListener('input',render);render();
</script></html>'''
    (lit/'index.html').write_text(doc.replace('PAYLOAD',payload),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--online',action='store_true');p.add_argument('--deadline',type=int,default=420);p.add_argument('--as-of',default='2026-10-12');args=p.parse_args()
    root=args.root.resolve();lit=root/'literature';imp=Importer(root,lit).load()
    resolutions=read_json(lit/'inputs/resolutions.json',{});refs=read_json(lit/'inputs/prior_report_references.json',[])
    corrections=resolve_rows(imp.rows,resolutions,refs)
    cache=read_json(lit/'inputs/metadata_cache.json',{});cache.update(read_json(lit/'inputs/verified_metadata.json',{}))
    ids={r['canonical_arxiv'] for r in imp.rows if r['canonical_arxiv']};errors=[]
    if args.online:
        title_log=resolve_title_queries(imp.rows,cache,args.as_of);dump(lit/'data/title_resolution_log.json',title_log)
        ids={r['canonical_arxiv'] for r in imp.rows if r['canonical_arxiv']}
        cache,errors=enrich(ids,cache,args.as_of,args.deadline)
    pubs=build_publications(imp.rows,cache,imp.assets,resolutions)
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit='local-uncommitted-fixture'
    coverage=write_outputs(lit,pubs,imp,corrections,cache,errors,args.as_of,commit)
    nav=root/'README.md'
    if nav.exists() and 'literature/CHRONOLOGICAL.md' not in nav.read_text(encoding='utf-8'):
        with nav.open('a',encoding='utf-8') as f:f.write('\n\n## 전체 논문 DB · 발행일순 대장\n\n[발행일순 목록](literature/CHRONOLOGICAL.md) · [DB 안내](literature/README.md) · [SQLite](literature/data/papers.sqlite) · [전수 대조](literature/coverage.json)\n\n초기 조사·주간 브리핑·통합 보고서와 선행 저장소 문헌을 합친 대장이다. 미검증 자료는 제거하지 않고 상태를 구분한다. 기존 모노그래프와 PDF를 대체하지 않는다.\n')
    if coverage['unmapped_observations']:raise SystemExit('Unmapped input observations')
if __name__=='__main__':main()
