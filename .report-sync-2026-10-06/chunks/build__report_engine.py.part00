from pathlib import Path
from html import escape as E
import json,math
ROOT=Path(__file__).resolve().parents[1]
for _dir in ['assets','sources','deliverables']: (ROOT/_dir).mkdir(parents=True,exist_ok=True)
S=json.loads((ROOT/'sources/references.json').read_text())
REF={s['key']:s for s in S}
PAGES=[]; MEDIA=[]; TABLES=[]
NAV=[]
TEAL='#087f85';NAVY='#123449';INK='#19394d';MUTED='#67808b';PALE='#edf5f5';AMBER='#d5983d';BLUE='#698bc0'
def C(*keys):
 return ' <span class="cite">'+', '.join(f'<a href="#ref-{REF[k]["id"]}">[{REF[k]["id"]:02d}]</a>' for k in keys)+'</span>'
def p(t): return '<p>'+t+'</p>'
def h(t): return '<h3>'+t+'</h3>'
def col(a,b,wide=False):return f'<div class="row"><div class="col{ " wide" if wide else ""}">{a}</div><div class="col">{b}</div></div>'
def panel(title,text,typ=''):return f'<div class="panel {typ}"><h3>{title}</h3>{p(text)}</div>'
def formula(eq,note=''):return f'<div class="formula">{eq}<span class="small">{note}</span></div>'
def stats(items):return '<div class="statrow">'+''.join(f'<div class="stat"><div class="value">{v}</div><div class="label">{l}</div><div class="note">{n}</div></div>' for v,l,n in items)+'</div>'
def table(title,heads,rows,widths=None,compact=False):
 n=len(TABLES)+1;TABLES.append(dict(number=n,title=title,page=len(PAGES)+1))
 return '<div class="tablewrap"><div class="tblcap">표 %02d. %s</div><table class="%s">'%(n,title,'compact' if compact else '')+('<colgroup>'+''.join('<col style="width:%s%%">'%w for w in widths)+'</colgroup>' if widths else '')+'<thead><tr>'+''.join('<th>'+x+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def page(title,section,body,deck='',pid=None):
 pid=pid or 'p'+str(len(PAGES)+1)
 PAGES.append(f'<section class="page" id="{pid}"><div class="eyebrow">{section}</div><h2>{title}</h2>'+ (f'<div class="deck">{deck}</div>' if deck else '')+body+'</section>')
 NAV.append(dict(page=len(PAGES),id=pid,title=title,section=section))
def refbar(*keys):return '<div class="sourcebar">주요 근거'+C(*keys)+' · 논문 결과는 저자 보고값이며 본 보고서에서 재현 실험하지 않았다.</div>'
def card(title,meta,text):return f'<div class="card"><h3>{title}</h3><div class="meta">{meta}</div><p>{text}</p></div>'
class SVG:
 def __init__(self,w=1000,h=420):self.w=w;self.h=h;self.e=[]
 def rect(self,x,y,w,h,fill=PALE,stroke='none',r=10,sw=1):self.e.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>');return self
 def line(self,x1,y1,x2,y2,c=TEAL,sw=2,dash=None,arrow=False):self.e.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arr)"' if arrow else '')+'/>');return self
 def path(self,d,c=TEAL,sw=2,fill='none',dash=None,arrow=False):self.e.append(f'<path d="{d}" stroke="{c}" stroke-width="{sw}" fill="{fill}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arr)"' if arrow else '')+'/>');return self
 def circle(self,x,y,r=5,fill=TEAL):self.e.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>');return self
 def text(self,x,y,txt,size=20,c=INK,bold=False,anchor='start'):
  lines=str(txt).split('\n')
  self.e.append(f'<text x="{x}" y="{y}" fill="{c}" font-size="{size}" font-family="Report,NanumBarunGothic,sans-serif" font-weight="{700 if bold else 400}" text-anchor="{anchor}">'+''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.4}">{E(t)}</tspan>' for i,t in enumerate(lines))+'</text>');return self
 def box(self,x,y,w,h,title,sub='',fill=PALE,color=INK):
  self.rect(x,y,w,h,fill)
  self.text(x+w/2,y+30,title,20,color,True,'middle')
  if sub:self.text(x+w/2,y+57,sub,15,color,False,'middle')
  return self
 def out(self):return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="{TEAL}"/></marker></defs>'+''.join(self.e)+'</svg>'
def fig(name,svg,caption,keys=(),kind='개념 재구성'):
 keys=(keys,) if isinstance(keys,str) else keys
 n=len(MEDIA)+1;file=f'fig-{n:02d}-{name}.svg';(ROOT/'assets'/file).write_text(svg)
 MEDIA.append(dict(number=n,file=file,title=caption,source_keys=list(keys),type=kind,page=len(PAGES)+1))
 return f'<div class="figure">{svg}<div class="caption"><strong>그림 {n:02d}.</strong> {caption} <span class="muted">{kind}.</span>'+C(*keys)+'</div></div>'
def flow(labels,subs=None,h=160,colors=None):
 d=SVG(h=h);n=len(labels);gap=24;w=(960-(n-1)*gap)/n
 for i,l in enumerate(labels):
  x=20+i*(w+gap);d.box(x,25,w,100,l,subs[i] if subs else '',fill=colors[i] if colors else PALE)
  if i<n-1:d.line(x+w+3,75,x+w+gap-6,75,arrow=True)
 return d.out()
def bars(labels,values,xmax=100,unit='%',color=TEAL,title='',baseline=None):
 d=SVG(h=105+len(labels)*69)
 if title:d.text(15,26,title,19,INK,True)
 left=210;right=910;top=65
 for j in range(5):
  x=left+(right-left)*j/4;d.line(x,top-10,x,top+len(labels)*69-10,'#dce7e8',1);d.text(x,top+len(labels)*69+17,f'{xmax*j/4:g}',14,MUTED,anchor='middle')
 for i,(l,v) in enumerate(zip(labels,values)):
  y=top+i*69;d.text(left-18,y+21,l,18,INK,True,'end');d.rect(left,y,(right-left)*v/xmax,32,color,r=3)
  d.text(left+(right-left)*v/xmax+12,y+23,f'{v:g}{unit}',18,color,True)
 return d.out()
