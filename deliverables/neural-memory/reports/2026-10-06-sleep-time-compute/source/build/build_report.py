from report_engine import *
from weasyprint import HTML
import csv,re,zipfile,hashlib

# Cover: original vector design, no external illustration or borrowed branding.
d=SVG(1000,620)
for i in range(6):
 y=50+i*70;x=160-i*17;w=635+i*34
 pts=f'{x}, {y+60} {x+w/2}, {y-5} {x+w}, {y+60} {x+w/2}, {y+128}'
 d.e.append(f'<polygon points="{pts}" fill="none" stroke="{["#83d9d1","#4ba6a9","#388b99","#276f83","#23556e","#254a60"][i]}" stroke-width="2.5"/>')
 for j in range(8):
  xx=x+70+j*(w-140)/7; yy=y+60
  d.circle(xx,yy,3.2,'#67bfbf' if i<3 else '#36637a')
  if i<5:d.line(xx,yy,xx-17,yy+70,'#325b70',1)
 d.text(x+w/2,y+66,['ACTIVE STATE','TASK MATERIALIZATION','LATENT / PARAMETRIC','KNOWLEDGE / SKILLS','VERSIONED EVIDENCE','PHYSICAL MEMORY'][i],17,'#b9dddf',True,'middle')
d.path('M105 85 C25 210 35 435 110 510', '#d3a055',3,arrow=True)
d.text(13,326,'SLEEP',17,'#e0b77b',True)
d.path('M900 510 C976 399 982 211 905 95','#5dbbbb',3,arrow=True)
d.text(876,327,'WAKE',17,'#85d1cf',True)
cover=f'''<section class="cover"><div class="series">MELETE RESEARCH / TECHNOLOGY INTELLIGENCE</div><h1>SLEEP-TIME<br>COMPUTE<br>&amp; NEURAL MEMORY</h1><div class="ko">경험을 지속 가능한<br>실행 상태로 바꾸는 AI</div><div class="sub">Agent memory · Continual learning · HOPE / CMS<br>HBM / HBF hardware co-design</div>{d.out()}<div class="foot"><strong style="color:white">2026.10.06 · 한국어 통합 리서치 보고서 · v2.0</strong><br>기술 동향, 근거 검토, 시스템 설계 가설 및 검증 로드맵<br>작성 지원: ChatGPT · 독립 기술 분석</div></section>'''
(ROOT/'assets/cover.svg').write_text(d.out())
PAGES.append(cover)

# 02
body=p('Sleep-time compute의 핵심은 더 오래 생각하는 것 자체가 아니다. 이미 지불한 추론·탐색 비용을 <strong>다음 과제에서 재사용할 수 있는 상태</strong>로 바꾸고, 그 상태의 유효성과 비용을 지속해서 관리하는 데 있다. 최초 연구는 질문 이전 계산을 미래 질의에 상각했고, 이후 연구는 latent state, skill, reusable KV, memory program으로 변환 대상을 확장했다.'+C('sleep','rpmem','wiki','attuner','memcodex'))
body+=stats([('01','저장과 실행을 분리','canonical evidence와 runtime artifact는 서로 다른 생애주기를 가진다.'),('02','검증을 승격의 조건으로','빈도·관련성보다 전이 효용과 회귀 위험을 측정한다.'),('03','물리 배치는 별도 최적화','update frequency와 read bandwidth를 혼동하지 않는다.')])
body+=col(h('연구가 보여주는 것')+p('메모리의 존재, 현재 사실의 유효성, 실행 시 활성화, 판단에 미치는 영향은 별개다. 과거 사실을 정확히 회수해도 최신 상태를 잘못 판단할 수 있으며, 올바른 기억도 필요 이상으로 사용될 수 있다.'+C('statemem','calib'))+p('계층의 다음 단계는 단순 요약 저장소가 아니다. 경험에서 모델별 adapter를 생성하거나, 환경을 예측하는 프로그램을 만들고, 그 프로그램을 다음 실행에 사용하는 접근이 등장했다.'+C('infinite','schema')),
h('우리가 제안하는 것')+p('원본 근거와 파생 artifact를 분리한 <strong>versioned state substrate</strong> 위에, sleep compiler와 독립 검증기를 둔다. cold storage에서 읽어 온 후보를 hot working set으로 materialize하고, 실제 미래 효용이 확인된 버전만 활성화한다.')+p('다만 이 설계는 문헌을 연결한 가설이다. slow CMS를 HBF에 일괄 배치하거나, memory 수가 늘면 성능이 자동으로 증가한다고 전제하지 않는다.'))
body+=panel('의사결정 요약','지금 우선 투자할 대상은 거대한 dreaming model보다 <strong>근거·버전·효용 계측이 가능한 학습/배치 인터페이스</strong>다. 이후 같은 예산에서 retrieval, latent memory, LoRA, skill compilation을 비교해야 한다.','dark')
page('핵심 판단','EXECUTIVE BRIEF',body,'재사용 가능한 상태는 자산이다. 그러나 검증·갱신·이동 비용까지 포함할 때만 자산으로 남는다.','executive')

# 03 TOC, refs target pages dynamic
body=''
for n,title,pid in [('01','개념 지도와 문헌 읽는 법','methods'),('02','Sleep-time compute와 HOPE / CMS','sleep-economics'),('03','Recurrent·latent·parametric memory','recurrent'),('04','Consolidation·credit·influence','jit'),('05','Dreaming과 self-improving agents','dreaming'),('06','Validity·transaction·security','validity'),('07','Serving·KV·HBM / HBF','workload'),('08','통합 아키텍처와 실험 계획','architecture'),('09','산업 신호·정정·추적 목록','industry'),('10','주석형 참고문헌 및 제작 정보','references')]:
 body+=f'<div class="tocrow"><span class="tocnum">{n}</span><a href="#{pid}">{title}</a></div>'
body+=h('독자별 추천 경로')
body+=table('목적에 따른 읽기 경로',['독자 / 목적','우선 읽을 질문'],[['연구 기획','어떤 상태를 학습할 것인가? 신경망 내부의 adaptation과 외부 memory control은 어떻게 다른가?'],['HW / 시스템 설계','논리적 갱신 주기와 실제 read/write traffic은 어떻게 연결되는가? HBF의 손익 분기점은 무엇인가?'],['Agent / 제품 개발','무엇을 보존하고 언제 무효화할 것인가? 복구·권한·평가를 어디에서 강제할 것인가?']], [24,76])
body+=p('<span class="badge emp">원문 결과</span><span class="badge hyp">설계 가설</span><span class="badge">제품 신호</span> 세 종류를 구분한다. 숫자는 본문과 figure caption에 범위·분모·비교군을 함께 표기했다. 참고문헌 번호를 누르면 주석형 출처 목록으로 이동한다.')
page('보고서 안내','READER’S GUIDE',body,'주간 뉴스 목록 대신, 연구 메커니즘과 시스템 설계의 연결을 따라 읽는다.','guide')

# 04
body=p('이 보고서는 이전 대화의 주간 조사와 GitHub 초안을 출발점으로, 핵심 논문의 제목·식별자·제출일·초록 및 일부 본문/도판을 다시 대조한 <strong>근거 등급형 기술 종합</strong>이다. 전체 분야를 빠짐없이 검색한 systematic review나 독립 재현 연구를 주장하지 않는다. 주요 조사 창은 2026년 8월~10월 초이며 2024~2025년 기반 연구를 연결했다.')
body+=table('근거 유형과 사용 범위',['표기','포함 자료','해석 범위'],[['P / 논문','arXiv preprint 및 확인된 학술지 논문','실험은 저자 보고값. 채택·peer review·재현 여부를 자동 추정하지 않는다.'],['D / 공식 문서','연구 블로그·제품 문서·기업 발표','알고리즘·제품 기능·표준화 사실 확인. 성능 우위를 독립 검증한 근거는 아니다.'],['C / 커뮤니티','직접 읽은 사용자 게시글','문제 발굴 신호. 표본·시장 점유율·일반적 성능으로 외삽하지 않는다.'],['H / 가설','문헌을 연결한 본 보고서 제안','구현 결과가 아니며 실험으로 반증할 수 있도록 설계한다.']], [18,35,47])
body+=col(h('검토 규칙')+p('동일 arXiv ID의 버전이 달라질 때는 결과를 합치지 않는다. 예를 들어 PlanFence는 본문에서 v1에 고정했다. 최초 제출일과 업데이트일, 공개 문서 확인일을 구분하며, 웹페이지가 없는 경우 제목만으로 수치를 복원하지 않았다.'+C('planfence'))+p('특히 HOPE/CMS는 원문 식과 실제 도식을 확인했다. HBFSim은 Figure 1 및 thermal/refresh 절을 확인하고, 원문의 시뮬레이션 실행 시간 개선과 LLM inference 성능을 구별했다.'+C('nl','hbfsim')),
h('매체 제작 규칙')+p('도식은 논문의 의미를 다시 설계한 <strong>독립 벡터 그림</strong>이다. 원문 figure를 무단 복제하거나 장식용 AI 이미지를 실험 결과처럼 사용하지 않았다. 수치 그래프는 확인한 저자 보고값만 재시각화하고 출처를 붙였다.')+p('서지 대장은 55개 출처를 포함한다. 이전 조사에서 추가 언급됐으나 이번 판에서 충분히 재검증하지 않은 항목은 추적 목록으로 남겼으며, 핵심 결론의 증거 수에는 포함하지 않았다.'))
body+=panel('해석의 경계','“Compiled experience scaling”은 유망한 연구 관점이지, 현재 입증된 보편적 scaling law가 아니다. “학술지 수준 구성”은 편집·근거 표현의 목표이며 이 PDF가 학술지 심사를 받았다는 뜻이 아니다.','warn')
page('근거를 읽는 법','01 / SCOPE & EVIDENCE',body,'확인한 사실, 저자 주장, 우리의 설계 추론을 서로 다른 층에 둔다.','methods')

# 05 taxonomy
D=SVG(h=400)
D.text(25,32,'MEMORY IS NOT ONE THING',23,TEAL,True)
for i,(t,s) in enumerate([('Evidence','source spans · event log'),('Belief / state','current estimate · open goals'),('Representation','text · graph · latent · weights'),('Execution artifact','KV · skill · adapter · program')]):
 D.box(20+i*245,68,226,100,t,s)
 if i<3:D.line(248+i*245,117,258+i*245,117,arrow=True)
D.rect(20,205,961,137,NAVY)
D.text(48,245,'CROSS-CUTTING CONTROL',21,'#ffffff',True)
D.text(48,279,'version / dependency / authority / activation / influence',22,'#c6e3e5')
D.text(48,316,'Placement: SRAM  •  HBM  •  DRAM/CXL  •  HBF  •  SSD',20,'#80c7c8')
body=fig('taxonomy',D.out(),'기억의 내용·표현·실행 형태와 제어 메타데이터를 분리한 분석 지도.',('statemem','rpmem','attuner'),kind='본 보고서 종합 도식')
body+=col(h('표현과 생애주기를 섞지 않는다')+p('episodic memory도 오래 보존할 수 있고, semantic rule도 곧바로 무효화될 수 있다. KV가 빠르게 변한다고 항상 폐기 가능하지도 않다. 한 representation이 다른 representation보다 본질적으로 더 “장기”라고 단정하면 정책이 잘못된다.')+p('graph나 vector는 접근 방법이다. 반면 권한, 유효 기간, generation은 무엇을 근거로 실행할 수 있는지를 결정한다. 데이터 모델과 신뢰 모델은 별개다.'),
h('읽기 경로와 학습 경로를 분리한다')+p('retrieval은 기존 객체를 선택하고, recurrent update는 현재 상태를 바꾸며, parametric learning은 실행 함수의 파라미터를 바꾼다. 세 가지는 모두 memory 효과를 내지만 비용·가역성·관측 가능성이 다르다.'+C('ttt','cache_rnn'))+p('제품 설계에서 “memory API 하나”로 감추기 전에, 각 연산의 입력 근거와 갱신 대상, 만료 조건을 명시해야 한다.'))
body+=panel('용어의 핵심','본 보고서에서 <strong>canonical</strong>은 변경 불가능하거나 진실이라는 뜻이 아니라, 출처·버전 계약 아래 관리되는 기준 상태라는 뜻이다. 신뢰할 수 있는 원본도 새 증거로 반박될 수 있다.')
page('Memory를 네 축으로 분해하기','01 / CONCEPT MAP',body,'무엇을 기억하는가, 어떻게 표현하는가, 언제 갱신하는가, 어디서 실행하는가는 서로 다른 질문이다.','taxonomy')

#06 timeline
D=SVG(h=500)
D.line(92,45,92,445,TEAL,3)
items=[('2024–2025','TEST-TIME ADAPTATION','TTT → Titans → Sleep-time compute → Nested Learning'),('2026.02–07','RECURRENT STATE EXPANSION','Growing memory checkpoints / online space-time memory'),('2026.08','STRUCTURE & LIFECYCLE','MESA · StateMem · WikiSkill · Recuris · FLINT'),('2026.09 early','CREDIT & PHYSICAL STATE','Hindsight Memory-PRM · PlanFence · UNISON · HBFSim'),('2026.09–10','COMPILATION & CONTROL','RPMem · Dream-RSI · JitMem · MemCodex · ActiveSaddler')]
for i,(date,t,s) in enumerate(items):
 y=55+i*86;D.circle(92,y+12,9,TEAL);D.text(125,y+1,date,16,MUTED,True);D.text(125,y+29,t,22,INK,True);D.text(125,y+55,s,17,MUTED)
body=fig('timeline',D.out(),'대표 자료의 공개 순서. 분야가 단일 선형 단계로 발전했다는 증명은 아니며, 병렬 연구 흐름을 묶은 지도다.',('ttt','titans','sleep','nl','mesa','wiki','rpmem','active'),kind='발표 시점 기반 재구성')
body+=col(h('연속성')+p('TTT는 hidden state를 학습 모델로 보는 길을 열었고, Titans와 Nested Learning은 memory update와 모델 구조를 함께 다룬다. 2026년 agent 연구는 별도의 외부 state, skill, program까지 갱신 대상으로 넓힌다.'+C('ttt','titans','nl')),
h('단절을 과장하지 않는다')+p('“이번 주부터 제어가 중요해졌다”는 식의 서사는 검색 창에 민감하다. 이 보고서는 뉴스 빈도보다 메커니즘의 연결과 재현 가능한 비교 문제에 무게를 둔다. 각 접근은 서로 대체되기보다 다른 계층에서 공존할 수 있다.'))
page('기술 계보: 적응에서 상태 관리로','01 / RESEARCH LANDSCAPE',body,'빠른 적응, 외부 기억, 배경 계산이 하나의 lifetime system으로 만나고 있다.','timeline')

#07
D=SVG(h=330)
D.box(20,45,240,100,'Observed context','before future question')
D.box(375,45,250,100,'Sleep transform','index / precompute / learn',fill='#d7eceb')
D.box(750,45,230,100,'Reusable state','versioned artifact')
D.line(266,95,365,95,arrow=True);D.line(632,95,741,95,arrow=True)
for i,l in enumerate(['Query A','Query B','Query C']):
 x=230+i*270;D.box(x,218,220,70,l,'',fill='#f1f4f7');D.path(f'M865 152 V187 H{x+110} V210',c=TEAL,arrow=True)
D.text(25,265,'Amortize once,\nreuse many times',19,MUTED)
body=fig('sleep-amortization',D.out(),'질문 이전 계산과 질문 이후 계산의 분리. 재사용 대상과 미래 query의 예측 가능성이 비용 상각을 결정한다.',('sleep','letta'))
body+=col(h('원 연구의 기여')+p('Sleep-time Compute는 context가 주어졌지만 아직 미래 질문이 확정되지 않은 구간에서 계산을 수행한다. stateful 추론 실험에서는 동일 정확도에 필요한 test-time compute를 약 5배 줄였고, multi-query 설정에서는 평균 query cost를 2.5배 낮췄다고 보고한다. 이 수치는 모든 agent workload에 대한 보장이 아니다.'+C('sleep')),
h('유용한 조건과 실패 조건')+p('자주 반복되는 질의 분포, 안정적인 context, 비싼 재추론이 있다면 사전 계산의 가치가 커진다. 반대로 거의 사용되지 않는 자료를 깊게 요약하거나 세계가 빠르게 바뀌면, sleep compute와 invalidation 비용을 회수하지 못한다.'))
body+=formula('N · ΔC<sub>query</sub> &gt; C<sub>sleep</sub> + C<sub>refresh</sub> + C<sub>storage / transfer</sub>','비용 단위를 통일한 단순 손익 조건. 본 보고서의 분석식이며 경험 법칙이 아니다.')
body+=panel('Background maintenance ≠ learning','DMA, compaction, GC는 자원 효율을 높일 수 있지만 그것만으로 agent가 학습했다고 볼 수 없다. <strong>현재 요청 비용 절감</strong>과 <strong>미래 과제 능력 변화</strong>를 분리해서 측정한다.')
page('Sleep-time compute의 경제학','02 / SLEEP COMPUTE',body,'추론을 밖으로 옮기는 것과 지식을 학습하는 것은 다르다. 둘 다 비용표에 올려야 한다.','sleep-economics')

#08
D=SVG(h=395)
D.text(18,27,'FORWARD: EVERY TOKEN MAY READ THE ENTIRE CHAIN',19,INK,True)
for i,(t,s) in enumerate([('Fast MLP','update every 1 chunk'),('Medium MLP','update every 4 chunks'),('Slow MLP','update every 16 chunks')]):
 x=135+i*285;D.box(x,56,235,86,t,s,fill=[PALE,'#e5f0f2','#e7edf5'][i]);
 if i<2:D.line(x+242,97,x+277,97,arrow=True)
D.text(20,103,'x',24,TEAL,True);D.line(53,96,124,96,arrow=True);D.text(956,103,'y',24,TEAL,True)
D.text(18,192,'UPDATE: DIFFERENT SCHEDULES',19,INK,True)
for row,(l,cnt) in enumerate([('Fast',1),('Medium',4),('Slow',16)]):
 y=226+row*47;D.text(24,y+21,l,18,INK,True)
 for t in range(16):D.rect(150+t*48,y,37,27,TEAL if (t+1)%cnt==0 else '#e8eef0',r=3)
D.text(160,385,'Illustrative chunk schedule; not the paper’s measured timing',15,MUTED)
body=fig('cms-forward-update',D.out(),'CMS의 read/forward와 update schedule은 다르다. 아래 1/4/16 주기는 설명용이며 원문 실험 설정을 재현한 값이 아니다.',('nl'),kind='원문 Eq.70–71 기반 독립 재구성')
body+=col(h('원문의 CMS')+p('Nested Learning은 서로 다른 빈도로 적응하는 연결된 optimization problem을 강조한다. CMS는 서로 다른 update frequency를 갖는 MLP 블록의 연쇄로 정의된다. HOPE는 self-modifying Titans의 적응성과 CMS의 지속적 저장을 결합한다.'+C('nl','google')),
h('Agent 시스템으로 확장할 때')+p('episodic store→skill→LoRA라는 데이터 변환 계층은 유용하지만, 그것이 곧 원문의 CMS는 아니다. 외부 state의 lifecycle을 여러 timescale로 관리하는 설계는 <strong>CMS에서 영감을 얻은 시스템 가설</strong>로 불러야 한다.'))
body+=panel('가장 중요한 정정','<strong>느리게 갱신된다고 드물게 읽히는 것은 아니다.</strong> Eq.70의 forward chain에 참여하는 느린 블록도 매 토큰 사용될 수 있다. 따라서 slow CMS → HBF라는 직접 매핑은 read traffic 검증 없이 성립하지 않는다.','warn')
page('HOPE / Nested Learning을 정확히 읽기','02 / FOUNDATIONAL MECHANISM',body,'학습 주기의 계층은 저장 장치의 계층과 닮았지만, 동일하지 않다.','hope')

#09
D=SVG(h=440)
D.rect(105,50,815,325,'#f3f7f8',r=0)
D.line(105,215,920,215,'#c2d5dc',1.5);D.line(514,50,514,375,'#c2d5dc',1.5)
D.text(309,100,'LOW WRITE / HIGH READ',18,TEAL,True,'middle');D.text(309,140,'dense slow-CMS block',22,INK,True,'middle');D.text(309,172,'HBM residency may still dominate',17,MUTED,anchor='middle')
D.text(718,100,'HIGH WRITE / HIGH READ',18,TEAL,True,'middle');D.text(718,140,'active state / fast weights',22,INK,True,'middle');D.text(718,172,'HBM / SRAM working set',17,MUTED,anchor='middle')
D.text(309,267,'LOW WRITE / LOW READ',18,TEAL,True,'middle');D.text(309,307,'cold committed artifact',22,INK,True,'middle');D.text(309,342,'HBF or SSD candidate',17,MUTED,anchor='middle')
D.text(718,267,'HIGH WRITE / LOW READ',18,TEAL,True,'middle');D.text(718,307,'rarely reused write stream',22,INK,True,'middle');D.text(718,342,'buffer / defer / avoid promotion',17,MUTED,anchor='middle')
D.text(105,410,'Less frequent update',16,MUTED);D.text(920,410,'More frequent update',16,MUTED,anchor='end');D.text(12,48,'HIGH',15,MUTED);D.text(15,382,'LOW',15,MUTED);D.text(8,223,'READ',15,TEAL,True)
body=fig('read-write-space',D.out(),'물리 배치의 최소 두 축: read demand와 write rate. 객체의 의미 범주만으로 placement를 정하지 않는다.',('nl','spacetime','potential'),kind='본 보고서 설계 공간')
body+=col(h('추가로 필요한 네 변수')+p('read/write 외에 객체 크기, access granularity, reuse distance, 허용 지연을 함께 봐야 한다. read-mostly라도 작은 random read가 많고 prefetch window가 짧으면 HBF의 큰 용량을 충분히 쓰지 못할 수 있다.'+C('flint','efficient')),
h('하드웨어와 연결되는 실험')+p('같은 CMS update schedule을 유지한 채 read sparsity와 active working set을 독립적으로 sweep한다. HBF hit rate가 높아도 p99 stall이나 HBM staging capacity가 나빠지면 승리한 설계가 아니다.'))
body+=formula('placement = f(read traffic, write bytes, reuse distance, granularity, stall budget)','semantic timescale은 하나의 feature다. 물리 계층을 지정하는 단일 정답이 아니다.')
page('CMS → HW 매핑의 함정','02 / CO-DESIGN PRINCIPLE',body,'update frequency만 보고 flash로 내리면, 학습 비용은 줄이고 매 토큰의 실행 비용을 늘릴 수 있다.','mapping')

#10 recurrent
body=table('지속 상태를 구현하는 세 경로',['경로','무엇이 남는가','갱신 / 한계'],[['TTT / Titans'+C('ttt','titans'),'small model 또는 neural memory의 상태','test stream으로 update. 안정성·I/O·reset 경계를 함께 관리해야 한다.'],['Memory Caching'+C('cache_rnn'),'선택한 recurrent memory checkpoint','과거 상태에 접근 가능해지지만 checkpoint 수에 따라 저장·검색 비용 증가.'],['KV-streams'+C('kvstreams'),'compaction 뒤 남긴 KV state','텍스트 재-prefill을 피한다. 이전 context에 의존한 latent state를 버전·reset 계약 없이 재사용할 수는 없다.']], [24,31,45])
D=SVG(h=260)
for i in range(6):
 x=40+i*155;D.box(x,36,130,68,'h'+str(i),'',fill=PALE)
 if i<5:D.line(x+134,70,x+151,70,arrow=True)
for i in [0,2,5]:
 x=40+i*155;D.line(x+65,108,x+65,163,arrow=True);D.rect(x,173,130,53,'#dce8f4');D.text(x+65,207,'checkpoint',17,INK,anchor='middle')
D.text(490,205,'select / route / reuse',19,TEAL,True,'middle')
body+=fig('recurrent-checkpoints',D.out(),'고정된 recurrent state만 유지하는 방식과 checkpoint를 선택적으로 보존하는 방식을 구별한다.',('cache_rnn'),kind='개념 재구성')
body+=col(h('유한 상태의 정보 병목')+p('상태 크기와 정밀도가 고정돼 있으면 모든 과거 사실을 무한히 구별해 보존할 수 없다. latent memory가 일정한 footprint로 작동한다는 것은 압축·망각·간섭의 trade-off를 관리한다는 뜻이지 용량 제한이 사라졌다는 뜻이 아니다.'),
h('실무적인 수명 계약')+p('모델 revision, tokenizer, RoPE/position 처리, attention mask, adapter 조합은 cache의 의미를 바꾼다. 동일 shape의 tensor라도 호환성을 보장하지 않는다. neural memory를 저장할 때는 canonical source와 artifact version을 함께 남기는 것이 안전하다.'))
page('Persistent state의 재등장','03 / RECURRENT MEMORY',body,'텍스트를 다시 읽지 않고도 상태를 이어갈 수 있다. 대신 그 상태를 어떻게 해석하고 폐기할지 명시해야 한다.','recurrent')

# 11 Parametric
D=SVG(h=405)
D.text(18,27,'A / RECURRENT LATENT → BACKBONE-SPECIFIC ADAPTER',18,TEAL,True)
for x,t,s in [(20,'Session','evidence'),(264,'Latent state','recurrent consolidation'),(508,'Decoder','backbone-specific'),(752,'LoRA','runtime artifact')]:D.box(x,50,224,92,t,s)
for x in [248,492,736]:D.line(x,95,x+9,95,arrow=True)
D.path('M350 148 V187 H295 V150',c=TEAL,arrow=True);D.text(385,181,'state persists; decoder compatibility is learned',16,MUTED)
D.text(18,240,'B / EVIDENCE → BELIEF OVER CODES → GENERATED WEIGHTS',18,TEAL,True)
for x,t,s in [(20,'Live data','context evidence'),(345,'Belief p(z)','latent-code distribution'),(670,'Weight generator','low-rank modulation')]:D.box(x,265,305,101,t,s,fill='#e5edf6')
D.line(330,316,337,316,arrow=True);D.line(655,316,664,316,arrow=True)
body=fig('parametric-compiler',D.out(),'두 parametric-memory 경로. canonical latent/belief와 실행용 weight는 구별되며, 서로 다른 논문의 구성요소를 한 구현으로 합친 것이 아니다.',('rpmem','infinite'))
body+=col(h('RPMem: 상태와 decoder의 분리')+p('session을 latent memory로 변환하고 recurrent gate로 합친 뒤 backbone별 decoder가 LoRA를 만든다. 모듈 훈련이 끝난 inference에서는 encoder·gate·decoder의 weights가 고정된 채 memory state가 forward computation으로 바뀐다. 따라서 매 세션 gradient를 수행하는 TTT와 같은 동작은 아니다.'+C('rpmem'))+p('backbone을 바꿀 때는 새 decoder의 적응이 필요하다. 모델 독립적인 상태를 목표로 한다는 것과 어느 모델에서나 즉시 읽을 수 있다는 것은 다르다.'),
h('Generated weights: 무엇을 저장할까?')+p('Infinite-Parameter LLMs는 live evidence로 latent code에 대한 belief를 갱신하고 low-rank weight를 생성한다. weight 자체보다 생성의 근거 상태를 저장하는 선택이 가능해진다.'+C('infinite'))+p('설계상 이득은 adapter 수명과 canonical state 수명을 따로 관리할 수 있다는 점이다. 그러나 latent bottleneck, decoder 비용, 모델 교체 시 호환성, 삭제 가능성을 함께 평가해야 한다.'))
body+=panel('승격은 되돌릴 수 있어야 한다','초기에는 base model을 직접 덮어쓰기보다 versioned adapter를 별도 배포한다. 잘못된 promotion은 adapter generation을 교체해 복구하되, 그 안의 특정 사실 하나만 제거됐다고 주장하지 않는다.','warn')
page('Parametric memory는 “저장된 weight” 이상이다','03 / CANONICAL STATE & COMPILATION',body,'latent state → decoder → adapter라는 분리는 lifelong memory의 업데이트·호환성·배치를 다시 설계하게 한다.','parametric')

#12
body=p('과거 정보를 곧바로 summary로 바꾸면 아직 모르는 미래 질문에 필요한 증거를 버릴 수 있다. JitMem은 raw trajectory를 보존하고 질문 시점의 curator가 필요한 compact payload를 만든다. Mem++는 조직 문서 전체와 버전을 유지하고, 질문이 묻는 시점까지의 자료를 read time에 고른다.'+C('jit','memplus'))
D=SVG(h=245)
D.box(20,64,235,97,'Raw evidence','versioned / recoverable')
D.box(373,15,260,83,'Sleep path','index / common views',fill='#d7eeeb');D.box(373,139,260,83,'Wake path','query-specific curation',fill='#e6edf6')
D.box(755,64,224,97,'Active context','bounded payload')
D.path('M261 99 H310 V57 H365',arrow=True);D.path('M261 116 H310 V180 H365',arrow=True);D.path('M640 56 H698 V100 H748',arrow=True);D.path('M640 180 H698 V124 H748',arrow=True)
body+=fig('jit-branch',D.out(),'write-time preprocessing과 read-time curation의 hybrid. 원본을 폐기하지 않는 한 서로 대체 관계일 필요가 없다.',('jit','memplus'),kind='본 보고서 종합 도식')
body+=fig('jit-gains',bars(['ALFWorld','WebShop','τ²-bench'],[16.2,16.3,3.9],xmax=20,unit=' pp',title='JitMem: success gain over strongest reported baseline'),'동일 논문 안에서 보고된 과제별 success improvement. 세 과제의 난이도·분모가 같다는 뜻은 아니다.',('jit'),kind='저자 보고값 재시각화')
body+=panel('Sleep에 남길 일','원본의 무조건적 압축 대신 indexing·deduplication·version reconciliation·자주 쓰는 materialized view 준비를 수행한다. 최종 task-specific 표현은 필요할 때 만들 수 있다. 단, JIT의 추가 latency와 LLM 호출 비용도 전액 계상한다.')
page('미래 질문을 모를 때는 늦게 결정한다','04 / WRITE-TIME VS READ-TIME',body,'최적 consolidation 시점은 session 종료가 아니라 정보의 사용 조건에 따라 달라진다.','jit')

#13
D=SVG(h=360)
D.box(15,31,257,85,'History A','earlier item was revoked')
D.box(15,187,257,85,'History B','earlier item was never present')
D.box(395,105,225,99,'Same summary','current answer identical',fill='#fff0d9')
D.path('M280 72 H325 V147 H385',arrow=True);D.path('M280 230 H325 V165 H385',arrow=True)
D.box(739,32,245,82,'Same new event U','different transition needed')
D.box(739,196,245,99,'Ambiguous update','lost tombstone / history',fill='#fff0d9')
D.path('M628 147 H686 V76 H731',arrow=True);D.path('M859 121 V187',arrow=True)
D.text(17,332,'Compression must preserve the distinctions needed by future updates.',21,TEAL,True)
body=fig('update-sufficiency',D.out(),'현재 답이 같아도 미래 update에 필요한 구별은 다를 수 있다. 설명을 위한 재구성 예시이며 논문의 자연어 사례를 복제한 것이 아니다.',('sufficient'))
body+=col(h('정확도만으로는 압축 품질을 알 수 없다')+p('Correct Now, Insufficient Later는 current-answer sufficiency와 update sufficiency를 분리한다. 24개 paired-history, 6개 synthetic mechanism pilot에서 tombstone을 없애는 압축이 타깃 조건에서 16/16 replay failure를 만들었다. 자연 과제에 대한 held-out 결과는 제시되지 않았다.'+C('sufficient')),
h('신경망 승격에서도 같은 질문이 남는다')+p('episode를 LoRA로 승격하면 source provenance와 temporal exception이 암묵적으로 섞일 수 있다. 오늘 recall이 좋아도 내일 정정·삭제를 처리할 정보가 남았는지는 별도다. 그래서 원문 근거, tombstone, dependency는 free-form summary 바깥에서 보존하는 것이 설계상 유리하다.'))
body+=table('압축 평가에 추가할 시험',['현재 평가','추가해야 할 평가'],[['현재 query 정확도','동일 현재 답을 갖는 서로 다른 history가 후속 update에서 구별되는가?'],['embedding 유사도','action에 필요한 증거·범위·예외가 보존됐는가?'],['메모리 절감량','재검색·재도구호출·재학습 비용을 포함해도 절감인가?'],['개별 항목 importance','여러 항목을 함께 지웠을 때 대체 근거까지 사라지는가?']], [31,69])
page('Consolidation의 품질은 미래에 드러난다','04 / INFORMATION FIDELITY',body,'현재 정답을 보존하는 압축과 미래 수정 가능성을 보존하는 압축은 같지 않다.','consolidation')

#14
body=p('어떤 기억이 검색됐다는 사실은 그 기억이 결과를 개선했다는 뜻이 아니다. citation도 결과에 맞춰 생성될 수 있다. 장기 self-improvement가 유효하려면 <strong>내용의 correctness, 현재 action의 contribution, 미래 reuse utility</strong>를 따로 측정해야 한다.'+C('credit','hprm'))
D=SVG(h=260)
for i,(t,s) in enumerate([('Observed use','retrieved / cited'),('Local intervention','remove → re-answer'),('Executed replay','change action → rollout'),('Held-out transfer','future task performance')]):
 x=15+i*249;D.box(x,40,234,111,t,s,fill=[PALE,PALE,'#e5edf6','#d4eae7'][i]);D.text(x+117,184,['association','answer sensitivity','policy-conditional effect','generalization'][i],15,MUTED,anchor='middle')
D.text(22,239,'More expensive evidence is not always available; record the proxy being used.',19,TEAL,True)
body+=fig('credit-ladder',D.out(),'효용 근거의 층위. 오른쪽이 단순히 “진짜 인과”라는 순위가 아니라, 서로 다른 질문을 측정하는 개입 범위다.',('credit','hprm','drsr'),kind='본 보고서 분석 도식')
body+=col(h('가능해진 것')+p('Hindsight Memory-PRM은 retrieval/citation audit와 한 번의 deletion-and-reanswer probe로 entry presence credit을 보정한다. DRSR은 항목이 아닌 삭제 집합의 next-output likelihood 변화를 supervision으로 사용한다. 둘 다 단순 사용 횟수보다 구체적인 학습 신호를 만든다.'+C('hprm','drsr')),
h('아직 해결되지 않은 것')+p('Credit Without Ground Truth는 ALFWorld replay에서 judge·logprob·confidence의 contribution fidelity가 제한적이라고 보고한다. 결과는 policy의 대안 행동 지원과 replay target 신뢰도에 조건부다. “LLM judge는 항상 무용하다”는 결론으로 넓히면 안 된다.'+C('credit')))
body+=formula('ΔU(M; task) = U(task with M) − U(task without M)','같은 task·seed·budget·tool version을 맞춘 개입 대비. 관측된 retrieval hit rate와 다른 지표다.')
body+=panel('Promotion gate에 넣을 최소 조건','효용 측정 방식, 불확실성, counterfactual support, 보호할 과거 과제의 regression 결과를 함께 기록한다. 수치 하나의 reward로 authority나 privacy 정책을 대체하지 않는다.')
page('Credit assignment가 장기 학습의 병목이다','04 / CAUSAL UTILITY',body,'좋은 기억인지 판단하려면 “썼는가”가 아니라 “없었으면 무엇이 달라졌는가”를 물어야 한다.','credit')

#15
D=SVG(h=310)
D.box(15,105,198,98,'Query / state','current task')
for y,t,s in [(10,'E','direct retrieval'),(113,'GE','generate from memory'),(216,'GH','generate from hidden')]:
 D.box(285,y,264,84,t,s,fill=PALE if y==10 else '#e8eef6');D.path(f'M219 153 H250 V{y+42} H277',arrow=True);D.path(f'M557 {y+42} H604 V153 H638',arrow=True)
D.box(647,105,168,98,'Router','select pathway')
D.box(850,105,136,98,'Influence','bounded use',fill='#d6ece8');D.line(822,154,842,154,arrow=True)
body=fig('routing-influence',D.out(),'MemoryAthena의 path 선택과 MemCalib의 influence 관점을 나란히 놓은 합성 설계. 하나의 실험으로 결합 검증된 구조는 아니다.',('athena','calib'),kind='독립 메커니즘의 종합 도식')
body+=table('메모리 상태의 네 가지 질문',['상태','질문','실패 예'],[['Retention','객체가 보존돼 있는가?','eviction과 deletion을 혼동해 원본 증거를 잃는다.'],['Validity','지금도 해당 조건에서 맞는가?','예전 권한·설정을 현재 사실로 사용한다.'],['Activation','이번 과제에 노출할 것인가?','맞는 기억이지만 불필요한 과거 전략으로 탐색을 고정한다.'],['Influence','얼마나 강하게 판단을 바꿀 것인가?','단순 선호를 하드 제약처럼 취급한다.']], [20,36,44])
body+=col(h('MemoryAthena의 포인트')+p('직접 memory 경로 E를 기준으로, retrieval을 cue로 생성한 GE와 hidden state에서 생성한 GH를 개입시킨다. 모든 경로를 항상 합치는 대신 freeze된 모듈 위의 lightweight router를 학습한다. generated state가 자동으로 증거가 되지는 않는다.'+C('athena')),
h('MemCalib가 추가하는 축')+p('memory proposition을 무시할지, 제한된 영향만 줄지, 판단을 제어하게 할지를 평가한다. retrieval 품질이 좋아진 뒤에도 over-use와 under-use는 남는다. 이는 recall과 reasoning 사이의 별도 control problem이다.'+C('calib')))
page('기억은 존재해도 사용하지 않을 수 있다','04 / ROUTING & INFLUENCE',body,'보존·유효성·활성화·영향력을 하나의 “confidence score”에 넣지 않는다.','influence')

#16
D=SVG(h=340)
for i,(title,sub,detail) in enumerate([('Replay','재방문','recorded experience → replay'),('Recombination','가설 생성','evidence pieces → candidate'),('Policy dreaming','대안 정책 평가','replay world → policy search')]):
 x=15+i*332;D.rect(x,30,311,260,[PALE,'#e8eef6','#fff3e0'][i]);D.text(x+22,72,title,26,INK,True);D.text(x+22,109,sub,20,TEAL,True)
 for j in range(4):D.circle(x+40+j*64,154,8,TEAL if i!=1 else BLUE)
 D.line(x+38,190,x+273,190,TEAL,2,arrow=True);D.text(x+22,237,detail,14,MUTED)
D.text(16,327,'GENERATED CONTENT REMAINS A CANDIDATE UNTIL VERIFIED',19,TEAL,True)
body=fig('dream-types',D.out(),'Dreaming을 세 가지 계산 작업으로 나누는 보고서 분류. 제품의 /dream 명칭과 neural policy optimization은 같지 않다.',('dream','wiki','grok'),kind='본 보고서 분류')
body+=col(h('Replay는 보존을 돕는다')+p('이미 관측한 자료를 재사용해 forgetting을 줄이거나 정책 평가 비용을 낮춘다. 원본이 잘못됐으면 오류도 다시 강화될 수 있으므로 source scope와 validity를 유지해야 한다. replay 데이터가 안전하고 정확하다는 가정은 자동으로 성립하지 않는다.'),
h('Recombination은 탐색을 넓힌다')+p('기억을 재조합한 설명·skill·프로그램은 새로운 후보를 만든다. 그러나 synthetic sample 수가 늘었다고 독립적인 증거가 늘어난 것은 아니다. 후보 수와 검증된 사실 수를 같은 카운터로 세지 않는다.'))
body+=h('Counterfactual policy dreaming')+p('Dream-RSI는 실제 discovery history를 재생 가능한 세계로 사용해 exploration policy를 offline에서 시험한다. 비용 절감은 비싼 실제 탐색을 반복하지 않는 데서 나온다. 그 replay world가 포함하지 않는 경로까지 신뢰할 수 있다는 의미는 아니다.'+C('dream'))
body+=panel('Sleep 학습의 안전한 출력','새 기억·skill·policy를 곧바로 활성 버전으로 쓰지 않는다. candidate → held-out replay → 외부 검증 → generation commit을 거친다. 승인 전 산출물은 관측 사실이 아니라 가설이다.','warn')
page('Dreaming: 무엇을 재생하고 무엇을 상상하는가','05 / OFFLINE LEARNING',body,'dream이라는 같은 이름 아래 서로 다른 학습·생성·정리 작업이 존재한다.','dreaming')

#17
D=SVG(h=365)
D.box(20,120,190,96,'Online search','expensive execution')
D.box(282,120,205,96,'Discovery tree','observed search history')
D.box(558,40,203,96,'Replay world','known support')
D.box(558,222,203,96,'Policy candidates','offline evaluation')
D.box(818,120,165,96,'Selected policy','redeploy')
D.line(216,168,274,168,arrow=True);D.path('M494 167 H522 V88 H550',arrow=True);D.path('M660 143 V214',arrow=True);D.path('M768 272 H795 V170 H810',arrow=True);D.path('M898 113 V14 H115 V112',arrow=True)
D.text(262,347,'Unknown branch → real execution required, not replay evidence',18,AMBER,True)
body=fig('dream-rsi-loop',D.out(),'Dream-RSI의 핵심 아이디어를 재구성한 offline/online loop. replay support 밖의 정책 효과는 추가 환경 실행으로 확인해야 한다.',('dream'))
body+=col(h('왜 자연어 요약만으로는 부족할까?')+p('경험을 prose insight로 줄이면 최종 결론은 남아도 탐색 분기, 실패 조건, alternative policy의 평가 정보가 사라질 수 있다. 실행 trace와 전이 구조를 남기면 다른 탐색 전략을 비교할 재료가 된다. 어떤 표현이 더 좋은지는 미래 평가 질문에 달려 있다.'),
h('Dreaming compute의 예산 단위')+p('FLOPs만 동일하게 맞추지 말고 simulator 구축 비용, 실제 environment call 수, 실패한 후보 평가, 최종 검증 비용까지 포함한다. old replay set에 맞춘 정책이 unseen world에서 악화되면 lifetime improvement라 부를 수 없다.'))
body+=table('Replay world에 필요한 메타데이터',['필드','역할'],[['Environment / tool version','과거 transition이 지금도 재현 가능한지 확인한다.'],['Behavior policy / support','어떤 대안이 실제로 관측·재생 가능한지 구별한다.'],['Outcome / costs','성공률뿐 아니라 실행·검증·latency 비용을 저장한다.'],['Source lineage / validity','기초 관측의 정정이 downstream policy 평가를 무효화할 수 있다.']], [36,64])
page('Dream-RSI: 경험을 평가 가능한 세계로','05 / REPLAY WORLD',body,'효율적인 dreaming은 상상량보다, 검증 가능한 대안을 얼마나 싸게 비교하는지에 달려 있다.','dreamrsi')

#18
D=SVG(h=290)
D.box(18,89,210,102,'Experience','raw observations')
D.box(299,20,250,96,'Knowledge / Wiki','claims + constraints')
D.box(299,171,250,96,'Executable model','predict state transitions',fill='#e4eef7')
D.box(678,89,300,102,'Skill / plan / action','invoke and verify',fill='#d6ece8')
D.path('M235 125 H262 V68 H290',arrow=True);D.path('M235 157 H262 V218 H290',arrow=True);D.path('M558 68 H605 V126 H670',arrow=True);D.path('M558 218 H605 V160 H670',arrow=True)
body=fig('knowledge-program',D.out(),'WikiSkill의 semantic/procedural 분리와 Schema의 executable world model을 연결한 표현 선택 지도.',('wiki','schema'),kind='본 보고서 종합 도식')
body+=col(h('WikiSkill')+p('raw experience와 누적 wiki, executable skill을 분리한다. 지식층을 통해 다음 skill evolution이 이전 최적화에서 배운 조건을 재사용한다. skill을 바로 뽑는 방식과 달리 semantic consolidation과 procedural compilation의 역할이 드러난다.'+C('wiki')),
h('Schema')+p('관측된 환경을 설명하는 실행 가능한 predictive program을 만들고 history에 맞춰 검토한다. 문장 규칙보다 정밀한 검증·시뮬레이션이 가능하지만, 환경의 형식화 가능성과 관측 coverage가 성능을 좌우한다.'+C('schema')))
body+=h('Episodic → parametric promotion을 서두르지 말아야 하는 이유')+p('같은 경험이라도 우선 text rule이나 program으로 배포하면 출처·예외·변경 diff를 검사하기 쉽다. latency가 반복적으로 병목이고 안정된 task distribution이 확인될 때 LoRA/latent compile을 추가하는 접근이 보수적이다. 이는 parametric memory의 우열 주장이 아니라 <strong>검증 가능한 중간표현을 먼저 두자는 시스템 제안</strong>이다.')
body+=panel('좋은 승격 대상','반복 사용되고, 적용 조건이 안정적이며, 다른 과제에서도 효용이 있고, 잘못되었을 때 되돌릴 수 있는 artifact. 반대로 개인 권한·가격·일시적 상태처럼 자주 변하는 사실은 외부 versioned evidence에 남기는 것이 자연스럽다.')
page('기억의 끝은 요약이 아니라 실행일 수 있다','05 / EPISODIC → PROCEDURAL',body,'experience → knowledge → skill과 experience → executable theory는 서로 다른 compilation 경로다.','programs')

#19
D=SVG(h=340)
for x,t,s in [(15,'Current harness','frozen evaluation target'),(340,'Candidate edit','prompt / code / memory'),(665,'Selection gate','held-out + budget')]:D.box(x,49,315,105,t,s)
D.line(333,100,335,100,arrow=True);D.line(658,100,660,100,arrow=True)
D.box(230,214,530,89,'Curriculum controller','which failure pattern should be optimized next?',fill='#d4eae7')
D.path('M829 162 V186 H760 V213',arrow=True);D.path('M230 257 H164 V163',arrow=True);D.path('M172 44 V14 H824 V44',dash='6 4',arrow=True)
D.text(380,186,'evaluate, reject or promote',16,MUTED)
body=fig('self-improvement',D.out(),'harness 자체의 수정과 개선 과제 선택을 분리한 loop. 고정 평가 경계를 agent의 수정 권한 밖에 둔다.',('aide','rrsi','active'),kind='본 보고서 종합 도식')
body+=table('Self-improvement 연구의 서로 다른 제어점',['연구','바꾸는 대상','남는 문제'],[['Recuris'+C('recuris'),'working state에 맞춘 skill 사용과 국소 memory evolution','실패 원인의 component attribution이 맞는가?'],['AIDE²'+C('aide'),'자기 research-agent code의 successive rewrite','8일 run의 성과가 더 긴 horizon에도 유지되는가?'],['RRSI'+C('rrsi'),'edit budget·critic·pruner로 과적합 억제','새로운 domain으로 전이되는 효과가 충분한가?'],['ActiveSaddler'+C('active'),'다음에 학습할 failure-pattern curriculum','learning-progress 추정 비용·불확실성을 어떻게 계상하는가?']], [24,40,36])
body+=p('AIDE²는 8일 동안 7개의 successive improvement를 발견하고 held-out 영역의 전이를 보고했다. RRSI는 unregularized evolution의 과적합을 줄이려 한다. 이 결과는 자동 개선 loop가 가능하다는 근거이지, 자율적으로 무한히 개선된다는 증거는 아니다.'+C('aide','rrsi'))
page('Self-improvement에도 control plane이 필요하다','05 / HARNESS EVOLUTION',body,'좋은 변경을 만드는 문제와 좋은 변경만 남기는 문제는 다르다.','selfimprove')

#20
body=p('메모리 구조가 고정되어 있고 router만 학습되는 경우와, 구조를 만드는 프로그램 자체를 바꾸는 경우를 구별해야 한다. MemCodex는 construction·indexing·retrieval·routing을 executable memory program의 일부로 보고 진화시킨다. 반면 PoS는 현재 세계 추정과 미해결 목표를 명시적인 belief state로 유지한다.'+C('memcodex','pos'))
D=SVG(h=310)
D.rect(20,28,620,248,NAVY);D.text(45,65,'MEMORY PROGRAM',24,'#fff',True)
for i,t in enumerate(['Construct','Index','Retrieve','Route']):D.box(42+i*145,103,130,85,t,'',fill='#234c61',color='#deeeee')
D.text(44,235,'Program revision ≠ state update ≠ physical placement',17,'#a1ced4')
D.box(687,70,294,98,'Belief state','world estimate + open goals',fill='#d3ebe7')
D.box(687,211,294,65,'Decision / action','',fill='#e5eef5');D.line(647,146,678,146,arrow=True);D.line(834,174,834,204,arrow=True)
body+=fig('program-state',D.out(),'메모리를 관리하는 algorithm과 현재 task state는 별도 갱신 대상이다.',('memcodex','pos'),kind='본 보고서 종합 도식')
body+=col(h('중요하지만 아직 초기인 신호')+p('MemCodex는 평균 success의 상대 개선 10.1%, context token 3.4배 감소, inference 2.1배 개선을 보고하며 work in progress로 표시한다. 비교 방법·budget을 맞추지 않고 “메모리 프로그램 진화가 항상 낫다”고 해석할 수 없다.'+C('memcodex')),
h('WorkingFrame과의 연결')+p('Melete에서 WorkingFrame은 현재 목표에 필요한 근거·가정·대안의 작업 표현이다. 이를 explicit belief state와 비교하는 실험은 가능하지만 두 시스템의 의미와 구현이 같다고 가정하지 않는다. 먼저 동일 history와 같은 budget에서 task progress·loop failure·회복 비용을 측정해야 한다.'))
body+=panel('추천 인터페이스','StateUpdate는 데이터 상태를, ProgramUpdate는 관리 로직을, PlacementUpdate는 실제 residency를 바꾼다. 세 종류를 다른 version space로 두면 회귀 원인과 rollback 범위를 더 명확하게 추적할 수 있다.')
page('Memory program과 active belief state','05 / ADAPTIVE CONTROL',body,'지금 무엇을 믿는지와, 기억을 어떻게 관리하는지는 서로 다른 학습 층이다.','control')

#21 validity
D=SVG(h=400)
D.box(20,35,236,84,'Evidence E @ v3','old requirement')
D.box(380,35,238,84,'Plan P','depends on E @ v3')
D.box(745,35,236,84,'Executor','receives current E @ v4')
D.line(262,76,372,76,arrow=True);D.line(625,76,737,76,arrow=True)
D.box(20,225,236,90,'Evidence E @ v4','revision committed',fill='#fff1d9')
D.box(380,209,238,122,'Dependency gate','P expects v3\nE is now v4',fill='#d2ebe7')
D.box(745,225,236,90,'Replan or block','not just refresh facts',fill='#e6edf6')
D.path('M138 127 V217',arrow=True);D.line(263,267,372,267,arrow=True);D.path('M499 126 V201',arrow=True);D.line(625,267,737,267,arrow=True)
D.text(26,378,'CURRENT FACTS DO NOT MAKE A PREVIOUSLY DERIVED PLAN CURRENT.',21,TEAL,True)
body=fig('planfence',D.out(),'최신 fact를 읽는 것과 plan의 derivation이 유효한지는 다르다. dependency-scoped 실행 검사의 필요성을 보여준다.',('planfence','invalidation'))
body+=col(h('StateMem: 상태 갱신')+p('StateMem은 무엇을 예전에 말했는가보다 현재 유효한 상태를 묻는다. 명시적 supersession과 relational dependency가 도움을 주지만, factual retrieval과 state tracking의 평가를 분리해야 한다.'+C('statemem')),
h('PlanFence: 파생물 갱신')+p('PlanFence v1의 30개 controlled workflow에서는 변경 뒤 stale-plan 실행을 막는 효과를 보고했다. 이 결과는 통제된 safety/system cost 평가이며 일반 task accuracy gain이 아니다. 새 버전의 결과와 본문의 v1 수치를 혼합하지 않는다.'+C('planfence')))
body+=panel('LoRA에도 dependency를 붙일 수 있는가?','training source lineage를 기록할 수는 있다. 그러나 한 source가 invalidated될 때, entangled weight에서 해당 지식만 선택 제거하는 것은 별도 문제다. 안전한 기본값은 해당 generation을 비활성화하고 재학습·재검증하는 것이다.','warn')
page('Fresh memory, stale artifact','06 / VALIDITY & DEPENDENCIES',body,'원본의 정정은 요약, skill, plan, KV, adapter까지 전파될 수 있다.','validity')

#22 transactional
D=SVG(h=355)
labels=[('Prepare','source snapshot'),('Build','candidate generation'),('Verify','external evaluation'),('Publish','atomic active pointer')]
for i,(t,s) in enumerate(labels):
 x=15+i*249;D.box(x,55,231,105,t,s,fill='#d6ece8' if i==3 else PALE)
 if i<3:D.line(x+237,108,x+245,108,arrow=True)
D.box(360,241,277,74,'Reject / rollback','',fill='#fff0d8')
D.path('M627 168 V209 H499 V233',arrow=True);D.text(641,204,'failed checks',16,AMBER)
D.path('M872 167 V209 H690 V273 H645',arrow=True);D.text(768,238,'regression',16,AMBER)
D.text(25,28,'SOURCE GENERATION + IDEMPOTENCY KEY + COMPARE-AND-SWAP',17,TEAL,True)
body=fig('transaction',D.out(),'파생 artifact의 versioned publish 제안. DB의 atomic pointer 교체와 외부 서비스의 실제 side effect는 구분해야 한다.',('limbo','invalidation'),kind='본 보고서 프로토콜 제안')
body+=col(h('Verification만으로 exactly-once가 되지 않는다')+p('LIMBO는 ACK 유실처럼 read-back이 가능한 경우와 요청이 아직 in-flight인 경우를 구분한다. 같은 강한 모델도 전자는 잘 처리하면서 후자는 중복 side effect를 만든다. idempotency key 등 tool contract가 중요해진다.'+C('limbo')),
h('Memory commit의 최소 경계')+p('source snapshot을 고정해 candidate를 만들고, 검증 직후에도 원본 version이 유지되는지 확인한다. commit key는 retry에도 같아야 하며 active pointer 변경은 원자적이어야 한다. 다른 writer가 앞서 갱신했으면 silent overwrite 대신 재검토한다.'))
body+=fig('limbo-duplicates',bars(['Lost ACK','Still in flight','Redelivery'],[0.5,56,74],100,'%',title='LIMBO: frontier-model duplicate-write rate by fault condition'),'같은 논문에서 보고한 fault 조건별 중복 비율. 전체 agent 신뢰성이나 모든 도구의 확률로 일반화하지 않는다.',('limbo'),kind='저자 보고값 재시각화')
page('Commit은 모델의 확신이 아니라 프로토콜이다','06 / TRANSACTION & ROLLBACK',body,'verify, atomicity, idempotency, compensation은 서로 다른 보장을 제공한다.','transaction')

#23
D=SVG(h=375)
D.rect(15,36,290,290,'#fff1de',stroke='#dfc090');D.text(35,73,'UNTRUSTED INPUT',21,'#9a6b2b',True)
D.text(36,111,'web / tools / retrieved notes',17,MUTED)
D.box(36,152,248,91,'Quarantine','source + scope labels',fill='#fffbf4')
D.rect(354,36,290,290,PALE,stroke='#a3c9c8');D.text(374,73,'AGENT WORKSPACE',21,TEAL,True)
D.box(375,113,248,89,'Candidate memory','read / synthesize / plan')
D.box(375,239,248,63,'Proposed action','',fill='#d8e9e9')
D.rect(693,36,291,290,'#e8eff6',stroke='#b2c2d6');D.text(714,73,'TRUSTED BOUNDARY',21,'#4e6e98',True)
D.box(714,110,248,84,'Authority gate','enforce current permission',fill='#f8fafc')
D.box(714,229,248,73,'External event ledger','',fill='#f8fafc')
D.line(310,196,345,158,arrow=True);D.line(650,271,705,154,arrow=True);D.line(650,274,705,268,arrow=True)
D.text(23,365,'Audit, authorization and retention policy must not be self-certified by the agent.',18,TEAL,True)
body=fig('trust-boundary',D.out(),'메모리 ingestion, reasoning, 실행 권한, 감사 로그를 분리한 방어 설계. 공격 재현 절차가 아닌 신뢰 경계 명세다.',('pmpa','trace'),kind='본 보고서 방어 아키텍처')
body+=col(h('오염된 기억은 세션을 넘는다')+p('PMPA는 외부 지시가 persistent memory에 기록되면 이후 세션에서 다시 활성화될 수 있음을 평가한다. 검증되지 않은 source를 “사용자가 정한 규칙”처럼 승격하지 않는 admission control이 중요하다.'+C('pmpa')),
h('감사 기록도 독립적이어야 한다')+p('Trace Tampering 연구는 agent가 자신의 기록을 조작하지 못한다는 운영 가정이 깨질 수 있음을 보여준다. 로그를 동일한 mutable workspace에 두고 agent의 self-report만 검사하면, incident reconstruction의 근거도 함께 사라질 수 있다.'+C('trace')))
body+=panel('Immutable에도 예외 계약이 필요하다','append-only는 영구 보존 의무가 아니다. 사용자 삭제·법적 보존 기간·scope 변경을 존중해야 한다. 원본, 파생 view, index, cache, replay dataset에 걸친 삭제 처리를 versioned deletion policy와 함께 설계한다.','warn')
page('메모리 보안은 학습 경계의 보안이다','06 / TRUST & GOVERNANCE',body,'나쁜 내용을 읽는 위험에서, 나쁜 내용을 다음 정책으로 학습하는 위험으로 확대된다.','security')

#24
D=SVG(h=325)
D.text(18,25,'AGENT PHASES / SCHEMATIC — NOT MEASURED MILLISECONDS',18,TEAL,True)
for i,label in enumerate(['Agent A','Agent B','Background']):D.text(12,100+i*87,label,19,INK,True)
segments=[[(180,175,'LLM'),(360,240,'Tool wait'),(605,190,'LLM'),(800,176,'Wait')],[(180,90,'Wait'),(275,210,'LLM'),(490,300,'Tool wait'),(795,181,'LLM')],[(180,168,'Idle'),(353,231,'KV migrate'),(589,180,'Pause'),(774,202,'Consolidate')]]
for i,row in enumerate(segments):
 for x,w,t in row:
  colr='#d7e9e8' if 'LLM' in t else '#eef2f5' if 'wait' in t.lower() or t in ['Idle','Pause'] else '#f9e5c4'
  D.rect(x,64+i*87,w,57,colr,r=5);D.text(x+w/2,98+i*87,t,18,INK,anchor='middle')
body=fig('phase-schedule',D.out(),'tool wait·LLM 실행·background 작업의 겹침. 대기 시간은 무조건 공짜 GPU 시간이 아니며 다른 session과 경합한다.',('agentsys','unison','efficient'),kind='설명용 workload schedule')
body+=stats([('5 / 10','non-LLM latency 지배','AgentSysBench의 10개 응용 중 5개. 전체 시장의 비율이 아니다.'),('28 GB','최대 sandbox working set','해당 평가의 session당 최대치. 일반 KV 크기와 혼동하지 않는다.'),('4.6×','state offload footprint 절감','저자 평가 조건의 최대/보고 효과. 모든 배치의 기대값이 아니다.')])
body+=col(h('Agent serving은 이질적이다')+p('AgentSysBench는 LLM·tool·sandbox·control-plane 호출·idle state를 함께 프로파일링한다. 모델 FLOPs만 줄여서는 latency가 줄지 않는 응용이 존재한다. sleep job도 같은 자원을 공유하므로 CPU, 네트워크, memory BW까지 trace해야 한다.'+C('agentsys')),
h('Offload의 가치는 돌아올 때 생긴다')+p('EfficientAgent는 lower tier가 concurrent agent pool의 reusable working set보다 작으면, 써 놓은 KV가 한 번도 재사용되지 않고 evict될 수 있음을 지적한다. capacity뿐 아니라 return distance와 cache policy를 같이 설계해야 한다.'+C('efficient')))
page('Agent workload는 토큰 흐름만이 아니다','07 / SYSTEMS CHARACTERIZATION',body,'LLM compute, tool latency, state residency, background maintenance를 같은 시간축에서 본다.','workload')

#25
D=SVG(h=300)
D.box(18,84,252,109,'Tool / skill schema','resource-local prefill')
D.box(374,84,260,109,'Frozen KV artifact','versioned, reusable',fill='#d6ece8')
D.box(741,84,240,109,'Query-side reader','select / adapt / attend',fill='#e5edf7')
D.line(277,139,366,139,arrow=True);D.line(641,139,733,139,arrow=True)
D.text(375,245,'Optional cold-tier residence',18,AMBER,True)
D.text(374,274,'HBF placement is a hypothesis, not a tested ATTUNER result.',15,MUTED)
D.text(28,30,'PRECOMPUTE ONCE',19,TEAL,True);D.text(744,30,'MATERIALIZE ON DEMAND',19,TEAL,True)
body=fig('compiled-kv',D.out(),'schema를 재사용 가능한 KV artifact로 분리하고 reader를 조정하는 구조. runtime compatibility와 조합 의존성을 먼저 확인해야 한다.',('recache','attuner'),kind='개념 종합')
body+=table('재사용 가능한 state의 세 설계',['방법','새로운 분리','핵심 제약'],[['ReCache'+C('recache'),'resource-local KV / selective resource access','resource 간 interaction 제거의 품질 영향; 원문 Inv-F1 82.3 vs82.4, TTFT 3.655×.'],['ATTUNER'+C('attuner'),'frozen artifact / query-side LoRA','reader adaptation 훈련이 필요. <0.05% parameter 학습과 최대3.73× speedup은 해당 실험 조건.'],['Memory Attention'+C('ma'),'token memory lookup / contextual key','추가 memory parameter와 offload 비용 존재. CPU prefetch 실험을 HBF 성과로 바꾸어 해석하지 않는다.']], [24,34,42])
body+=col(h('새로운 cache invalidation 문제')+p('본문 artifact, 모델 revision, adapter, position rule 중 무엇이 바뀌어도 KV의 의미가 달라질 수 있다. content hash만 같다고 항상 재사용 가능하지 않다. artifact manifest에 생성 조건을 기록하고 변환 가능성을 검증해야 한다.'),
h('왜 HBF와 연결되는가?')+p('stable artifact가 크고 read-mostly라면 near-compute capacity tier의 후보가 된다. 하지만 이것은 placement 가설이다. HBF read가 current query의 critical path에 남으면, 재-prefill 절감보다 stall이 커질 수 있다.'))
page('KV는 cache를 넘어 compiled artifact가 된다','07 / REUSABLE EXECUTION STATE',body,'weight를 바꾸지 않고 reader만 바꾸는 설계는, 저장과 계산의 경계를 이동시킨다.','kv-artifacts')

#26 HBF industry
body=p('HBF는 단순히 더 빠른 SSD라는 이름이 아니다. near-compute capacity를 확장하려는 flash 기반 메모리 계층이며, read bandwidth, program/erase 제약, controller, packaging, thermal budget을 함께 설계해야 한다. Sandisk와 SK hynix는 2026년 8월 OCP technical specification 공개와 표준화 협력을 발표했다. 표준 공개는 양산 시스템 성능 검증과 다르다.'+C('sandisk'))
D=SVG(h=265)
levels=[('SRAM','small / very near compute',150,TEAL),('HBM','active high-bandwidth state',275,'#3d8a98'),('DRAM / CXL','capacity + mutable state',400,'#6d9bad'),('HBF','flash capacity near compute',525,'#89adb9'),('SSD / object','cold evidence / durable logs',650,'#b1c7cf')]
for i,(l,s,w,c) in enumerate(levels):
 y=12+i*48;D.rect(30,y,w,36,c,r=3);D.text(45,y+25,l,18,'#fff' if i<3 else INK,True);D.text(w+58,y+25,s,17,MUTED)
body+=fig('memory-hierarchy',D.out(),'물리 hierarchy의 역할을 보여주는 개념도. 막대 길이는 정성적 capacity 표현이며 실제 제품 용량·bandwidth 비율이 아니다.',('sandisk','potential'),kind='정성적 설계 지도')
body+=table('HBF를 평가할 때 구분할 항목',['질문','확인할 것'],[['용량','큰 모델을 저장할 수 있는가와 실제 실행 working set이 만족되는가는 다르다.'],['성능','host 노출 bandwidth, plane parallelism, read latency tail, staging overlap.'],['쓰기','program granularity, buffer, write amplification, GC·refresh 포함 bytes.'],['물리 제약','HBM package budget 잠식, 전력·온도, NAND retention·endurance.'],['구현 상태','specification / analytical model / trace simulation / real-GPU emulation / actual silicon을 구분.']], [18,82])
body+=panel('핵심 조건','<strong>HBM execution path를 유지하면서 capacity를 확장하는가?</strong> 이 질문에 답하지 못하면 HBF capacity 증가는 매력적이지만 시스템 이득은 불분명하다.'+C('potential','exploring'))
page('HBF: capacity tier의 기회와 조건','07 / MEMORY HIERARCHY',body,'읽기·쓰기·배치·열화까지 포함해야 의미 있는 co-design이 된다.','hbf')

#27
D=SVG(h=355)
D.rect(20,20,455,309,PALE);D.text(45,58,'A  /  DECOUPLED TIERS',22,TEAL,True)
D.box(47,91,398,76,'HBM / DRAM','active KV + mutable runtime state',fill='#d8e9ee')
D.box(47,213,398,80,'HBF','weights / stable experts',fill='#e5edf6');D.line(246,207,246,175,arrow=True)
D.rect(523,20,457,309,'#fff5e8');D.text(548,58,'B  /  FULL-HBF + BUFFERING',22,'#9c6d2c',True)
D.box(550,91,402,76,'Fast write / staging buffer','coalesce fine-grained updates',fill='#fffdf6')
D.box(550,213,402,80,'HBF','weights + dynamic KV',fill='#f5e2bd');D.line(750,174,750,206,arrow=True)
body=fig('hbf-branches',D.out(),'HBF architecture의 두 설계점. 어느 쪽이 우월한지는 workload, memory budget, controller 가정에 따라 바뀐다.',('moe','hbflex'))
body+=table('논문 결과를 읽는 비교 틀',['연구 / 조건','결과 또는 주장','증거 유형 / 주의'],[['Full-stack HBF'+C('hbfsucks'),'naive transient-KV offload의 평균 e2e latency가 2–5.5× 악화','TokenSim 기반. “HBF는 항상 나쁘다”가 아니라 부적합한 배치의 경고.'],['Trillion MoE'+C('moe'),'weight/state provisioning 분리; state BW/capacity 1.4–4.0 s⁻¹ 조건','두 trillion-MoE와 trace 분석. ≤1.10 completion-time 조건·low concurrency 범위.'],['HBFlex'+C('hbflex'),'FlashAccel 대비 최대1.58×, H3 대비3.30× throughput','trace-driven simulation. plane-aware 배치·write aggregation·lifetime packing의 결합.'],['FLINT'+C('flint'),'read-specialized FTL·burst buffering·refresh 분리','read-only FTL 경로에 임의 write나 범용 transaction 지원을 가정하면 안 된다.']], [28,37,35])
body+=p('서로 다른 baseline의 배수를 한 그래프에 놓아 논문 순위를 만들지 않는다. 재현 비교에서는 동일 trace, capacity, thermal budget, HBM 감소량, endurance 조건을 맞춘다.')
page('HBF 연구는 하나의 결론으로 수렴하지 않는다','07 / COMPETING DESIGN POINTS',body,'stable weight와 dynamic state를 분리할 수도, controller를 바꾸어 함께 수용할 수도 있다.','hbf-branches')

#28 HBFSim
D=SVG(h=360)
cols=[(20,'GPU EXECUTION',['Real kernels','PTX instrumentation','Fast timing path'],PALE),(353,'SHARED INTERFACE',['Generation / address map','Request + byte counters','Thermal snapshot'],'#e6eff6'),(686,'HOST MODEL',['Thermal + retention','Refresh scheduler','Detailed reference path'],'#fff1dd')]
for x,t,ls,c in cols:
 D.rect(x,20,294,320,c);D.text(x+15,58,t,21,INK,True)
 for i,l in enumerate(ls):D.rect(x+17,86+i*78,260,58,'#ffffff',r=5);D.text(x+147,120+i*78,l,17,INK,anchor='middle')
D.line(319,124,345,124,arrow=True);D.line(652,124,678,124,arrow=True);D.line(678,273,652,273,arrow=True);D.line(345,273,319,273,arrow=True)
body=fig('hbfsim',D.out(),'HBFSim의 GPU execution·shared ABI·host model 분리를 독립 도식으로 재구성. 원문 Figure 1의 메시지만 요약했다.',('hbfsim'),kind='원문 Figure 1 / §4 기반 재구성')
body+=col(h('Real GPU ≠ real HBF silicon')+p('HBFSim은 실제 GPU kernel 실행에 HBF timing·capacity·thermal 모델을 결합한다. compute와 memory overlap을 실제 실행 흐름에서 조사하는 도구이지만, HBF 칩을 실측한 결과로 부를 수는 없다.'+C('hbfsim')),
h('20.8×의 정확한 의미')+p('빠른 경로와 detailed reference의 실행 시간 비교에서 보고한 20.8×는 <strong>시뮬레이션 속도</strong>다. LLM serving 자체가 20.8배 빨라졌다는 뜻이 아니다. report나 발표에서 숫자만 분리하면 의미가 완전히 달라진다.'+C('hbfsim')))
body+=formula('Physical program bytes = W<sub>logical</sub> · A<sub>mapping / GC</sub> + W<sub>refresh</sub>','정성적 회계식. controller/FTL 정의에 따라 항을 분리하고 중복 계상을 피한다.')
body+=panel('HAT와 연결할 검증 축','HAT는 logical state transition과 workload intent를 생성하고, physical backend는 bytes·stall·thermal·refresh를 계산하도록 분리한다. 동일 trace의 analytical model, trace simulator, real-GPU emulation 결과를 교차 검증해야 한다.')
page('HBFSim: 열화까지 포함한 실행 모델','07 / SIMULATION & RELIABILITY',body,'모든 쓰기가 사용자 update에서 나오지는 않는다. 온도와 refresh가 내구성 비용을 바꾼다.','hbfsim')

#29 physical mapping
D=SVG(h=450)
D.text(20,30,'LOGICAL OBJECT',19,INK,True);D.text(548,30,'PLACEMENT DECISION',19,INK,True)
objs=['Dense slow-CMS weights','Sparse stable expert / adapter','Active KV / recurrent state','Version / dependency index','Immutable episodes / replay']
for i,t in enumerate(objs):D.box(15,53+i*74,450,59,t,'',fill=PALE)
locations=[('HBM / SRAM','read-hot + latency-sensitive'),('DRAM / CXL','mutable metadata + staging'),('HBF','read-mostly; batchable; reusable'),('SSD / object','cold; large; recoverable')]
for i,(t,s) in enumerate(locations):D.box(581,64+i*91,400,72,t,s,fill=['#d1e8e6','#dce9f1','#e6eef5','#f0f3f4'][i])
for si,di in [(0,0),(1,0),(1,2),(2,0),(2,1),(3,1),(4,3)]:
 y1=83+si*74;y2=100+di*91;D.path(f'M473 {y1} C525 {y1} 524 {y2} 573 {y2}',sw=1.7,dash='5 4' if (si,di) in [(1,2),(2,1)] else None,arrow=True)
body=fig('logical-physical',D.out(),'객체별 후보 placement. 점선은 특히 workload 조건에 민감한 경로다. slow-CMS도 read-hot이면 HBM에 남을 수 있다.',('nl','potential','efficient','hbflex'),kind='본 보고서 설계 가설')
body+=col(h('Content hotness와 update hotness')+p('cold canonical state를 query 시점에 hot artifact로 만들 수 있지만, 모든 canonical state가 cold인 것은 아니다. dense base weight처럼 사실상 모든 토큰에 읽히는 객체는 갱신이 드물어도 실행 경로에서 뜨겁다.'),
h('Dematerialize는 delete가 아니다')+p('HBM에서 artifact를 내리는 것은 근거를 지우는 행위와 다르다. 재구성 비용이 낮고 재사용까지 멀다면 버릴 수 있지만, 다시 생성할 수 없는 recurrent state는 checkpoint를 먼저 남겨야 할 수 있다.'))
body+=panel('Design rule','Representation × Update rate × Read traffic × Reuse distance를 함께 측정한다. “semantic은 HBF, episodic은 SSD”처럼 이름만으로 정한 placement는 baseline일 뿐 최적 해가 아니다.','dark')
page('물리 배치는 representation의 함수가 아니다','07 / PLACEMENT POLICY',body,'logical CMS를 storage tier에 일대일 대응시키는 그림 대신, 객체별 조건부 배치를 설계한다.','placement')

#30 economics
D=SVG(h=340)
D.text(25,30,'ILLUSTRATIVE COST MODEL — NOT BENCHMARK DATA',18,AMBER,True)
left=90;top=65;right=910;bottom=275
for v in [0,50,100,150,200]:
 y=bottom-v/200*210;D.line(left,y,right,y,'#dfe7e9',1);D.text(70,y+5,str(v),15,MUTED,anchor='end')
for n in [0,25,50,75,100]:D.text(left+n/100*820,bottom+27,str(n),15,MUTED,anchor='middle')
D.line(left,bottom,right,top,TEAL,3);D.line(left,bottom-105,right,bottom-105,AMBER,3)
D.circle(500,170,7,NAVY);D.path('M500 170 V278',c=NAVY,dash='5 5');D.text(527,156,'break-even: 50 reuses',18,INK,True)
D.text(670,82,'saved query cost = 2N',16,TEAL,True);D.text(600,202,'build + refresh cost = 100',16,'#946d2b',True);D.text(849,328,'Number of reuses N',16,MUTED,anchor='end')
body=fig('break-even',D.out(),'가상의 비용 단위로 설명한 손익 분기점. query당 2를 절감하고 build/refresh에 100을 쓴다면 50회에서 비용이 같다. 실험값이 아니다.',(),kind='분석식 기반 예시')
body+=col(h('LoRA는 싸지만 공짜는 아니다')+p('rank r, input n, output m인 저랭크 adapter의 parameter 수는 대략 r(m+n)이다. 학습 시에는 weights 외에도 gradient, optimizer state, activation, checkpoint 및 버전 복제 공간이 필요하다. inference footprint만 비교하면 승격 비용을 과소평가한다.'),
h('Batch commit의 올바른 회계')+p('많은 update를 하나의 commit으로 묶으면 metadata overhead는 줄 수 있다. 그러나 HBF endurance는 commit 개수가 아니라 실제 program bytes와 분포에 좌우된다. 동일 큰 artifact를 자주 재작성하면 commit 수가 적어도 더 빨리 닳을 수 있다.'))
body+=table('Lifetime 비용표에 반드시 들어갈 항목',['단계','계상할 비용'],[['Build / learn','episode selection, replay, gradient 또는 decoder compute, 검증 실패 후보'],['Serve / reuse','retrieval, JIT curation, adapter materialization, DMA, attention stall'],['Maintain / revise','invalidation fan-out, source revalidation, refresh, GC, 재학습'],['Recover / govern','checkpoint, rollback, audit storage, 삭제 처리와 재검증']], [26,74])
page('Lifetime ROI: sleep와 wake의 비용을 합친다','08 / COST MODEL',body,'성공한 승격만 세거나, foreground latency만 줄이면 전체 계산 경제성을 놓친다.','economics')

#31 integrated architecture
D=SVG(h=485)
D.rect(17,15,966,133,'#e7f2f0');D.text(37,47,'WAKE / LOW-LATENCY PATH',22,TEAL,True)
for x,t in [(38,'Task + query'),(276,'Working state'),(514,'Reader / adapter'),(752,'Action gate')]:D.box(x,65,211,62,t,'',fill='#ffffff')
for x in [255,493,731]:D.line(x,96,x+13,96,arrow=True)
D.rect(17,184,966,144,NAVY);D.text(37,219,'SHARED STATE CONTRACT',22,'#ffffff',True)
D.text(38,255,'object-id · source-version · scope · dependencies · authority · artifact-format',20,'#bbdadd')
D.text(38,294,'Canonical evidence / latent state      +      Versioned artifact registry',21,'#82cdca')
D.rect(17,363,966,110,'#fff2df');D.text(37,398,'SLEEP / BUDGETED BACKGROUND PATH',21,'#9c722d',True)
D.text(37,436,'select → ground → replay / learn → verify → publish → place',23,INK,True)
for x in [150,395,635,860]:D.line(x,154,x,176,arrow=True);D.line(x,334,x,355,arrow=True)
body=fig('reference-architecture',D.out(),'통합 reference architecture 제안. 모든 경로에 공통 state contract를 두되 foreground와 background의 자원·권한을 분리한다.',('rpmem','jit','planfence','limbo','attuner'),kind='본 보고서 설계 가설')
body+=col(h('Canonical state와 artifact registry')+p('원본 evidence, latent state, decoder version, 파생 artifact의 관계를 registry가 연결한다. 조회할 때는 현재 query와 유효한 source generation에 맞는 artifact만 materialize한다. 어느 저장소에 둘지는 placement policy가 정한다.'),
h('Melete / HAT에 내리는 의미')+p('Melete는 source·version·WorkingFrame·local policy 실험의 출발점으로 두고, HAT는 logical event를 bytes/FLOPs/stall로 변환하는 계측 계층으로 둔다. 이 다이어그램은 통합 구현 완료를 의미하지 않으며, 외부 LLM·GPU·HBF 연결은 별도 검증이다.'))
body+=panel('교체 가능한 경계','새로운 memory algorithm은 전체 시스템을 다시 만들지 않고 StateUpdate 또는 Compiler 구현으로 교체할 수 있어야 한다. 메타데이터·권한·평가 계약은 유지하여 변경의 효과만 비교한다.','dark')
page('통합 reference architecture','08 / SYSTEM DESIGN PROPOSAL',body,'공통 state contract 위에서 storage, learning, compilation, execution을 교체 가능한 계층으로 만든다.','architecture')

#32 events
body=p('기존 alloc/free trace에 logical learning event를 붙이면 “왜 이 바이트가 이동했고 왜 이 update가 필요했는가”를 분석할 수 있다. 단, semantic event는 비용을 설명하는 신호이며 그 자체가 하드웨어 명령은 아니다. 실제 kernel·DMA·storage operation과 인과 ID로 연결한다.')
body+=table('HAT / runtime event 계약 제안',['Event group','예시','필수 기록'],[['Ingest / normalize','ADMIT, GROUND, SOURCE_REVISE','source hash, scope, authority, valid time, observed time'],['Transform / learn','CURATE, CONSOLIDATE, DREAM, TRAIN','input generation, method revision, tokens/FLOPs, candidate ID'],['Evaluate / commit','VERIFY, PUBLISH, REJECT, ROLLBACK','eval set revision, budget, outcome, idempotency key'],['Execute / move','MATERIALIZE, READ, MIGRATE, EVICT','artifact format, tiers, bytes, queue wait, stall'],['Maintenance','INVALIDATE, REBUILD, REFRESH, DELETE','cause ID, affected descendants, physical writes, completion']], [27,32,41])
D=SVG(h=252)
D.box(18,53,284,115,'Logical event','PROMOTE candidate A\nsource generation 17')
D.box(359,53,284,115,'Physical operations','read → DMA → kernels\nwrite → metadata commit',fill='#d7eae7')
D.box(700,53,284,115,'Outcome record','quality / latency / bytes\nrollback / regression',fill='#e3edf5')
D.line(309,112,350,112,arrow=True);D.line(650,112,691,112,arrow=True)
D.text(27,220,'Join key: run-id / task-id / object-id / generation / causal-parent-id',19,TEAL,True)
body+=fig('event-contract',D.out(),'semantic event와 실제 resource operation을 연결하는 trace contract. 로그만 추가했다고 학습 효용이 증명되는 것은 아니다.',(),kind='본 보고서 계측 설계')
body+=col(h('부분 실패를 명시한다')+p('COMPILED, VERIFIED, PUBLISHED, RESIDENT는 다른 상태다. 검증은 끝났지만 HBF write가 실패하거나, write는 끝났지만 active pointer가 교체되지 않을 수 있다. 각각 retry·recovery 절차를 정의해야 한다.'),
h('시간도 두 종류다')+p('사실이 세계에서 유효했던 시간(valid time)과 시스템이 관측·기록한 시간(transaction/observed time)을 구분한다. delayed evidence를 현재 시점의 새 사실로 덮어쓰지 않도록 한다.'))
page('Learning trace를 resource trace로 연결하기','08 / EVENT MODEL',body,'state lifecycle을 비용과 검증 결과까지 추적할 수 있을 때 co-design이 실험 가능한 문제가 된다.','events')

#33 evaluation
body=p('강한 memory system이라는 주장은 retrieval score 하나로 성립하지 않는다. DolphinBench는 과거 정보가 실제 tool action에 필요한 task를 구성하고 accuracy와 비용·latency를 함께 요구한다. StateMemBench는 current-state tracking을 분리하며, update-sufficiency pilot은 후속 정정의 복구 가능성을 묻는다.'+C('dolphin','statemem','sufficient'))
body+=table('동일 조건 비교를 위한 benchmark scorecard',['축','측정','통제해야 할 조건'],[['기본 효용','task success, action correctness','동일 base model·harness·tool snapshot'],['현재 상태','stale-state / superseded answer rate','질의 시점·자료 시점·정정 순서'],['Causal reuse','with-memory / without-memory paired delta','동일 seed·token budget·대안 지원'],['전이 / 간섭','held-out gain, protected-task regression','task-order shuffle·user holdout·OOD split'],['학습 경제성','lifetime tokens/FLOPs/원가·회수 시간','후보 생성·실패 평가·재학습 포함'],['Serving','p50/p95/p99, throughput, SLO goodput','concurrency·query mix·warm/cold 정의'],['물리 비용','peak residency, bytes, refresh/GC writes','동일 HBM/HBF package·전력 budget'],['복구 / 권한','duplicate/stale commit, rollback, policy violation','fault injection·권한 변경·감사 경계']], [20,39,41])
body+=col(h('Baseline은 의도적으로 강하게')+p('같은 데이터와 budget으로 full context, hybrid retrieval+reranker, write-time summary, JIT curation, latent/LoRA, skill/program을 비교한다. 새로운 시스템에 더 많은 토큰·훈련·검증을 몰아주고 우수하다고 결론내리지 않는다.'),
h('평균과 신뢰 구간을 함께')+p('task와 user 단위 paired evaluation 및 여러 실행을 수행한다. 성공 사례를 모은 pass@k와, 모든 반복에서 성공해야 하는 운영 신뢰도는 다르다. 백그라운드 update 후 평균 향상뿐 아니라 tail failure와 regression도 본다.'))
body+=panel('금지할 비교','논문 A의 LoCoMo 점수, 논문 B의 다른 model/reader 점수, 논문 C의 tool success를 한 종합 leaderboard로 묶지 않는다. 본 보고서의 숫자는 의도적으로 <strong>같은 연구 내부의 비교</strong>로 제한했다.','warn')
page('Recall을 넘어 lifetime utility를 평가한다','08 / BENCHMARKING',body,'좋은 memory는 많이 기억하는 시스템이 아니라, 다음 행동을 더 정확하고 경제적으로 만드는 시스템이다.','evaluation')

#34 plan
D=SVG(h=267)
for i,(t,s) in enumerate([('A / Evidence','source + version + audit'),('B / Baselines','retrieval + JIT + replay'),('C / Promotion','skill / latent / adapter'),('D / Co-design','residency + HBF model')]):
 x=15+i*249;D.box(x,50,231,115,t,s,fill=PALE if i<2 else '#e6edf5');
 if i<3:D.line(x+235,105,x+245,105,arrow=True)
D.text(25,215,'Advance only after the previous contract passes its falsification test.',19,TEAL,True)
body=fig('roadmap',D.out(),'구현 순서 제안. 날짜 약속이 아닌 검증 gate 중심의 실행 로드맵이다.',(),kind='본 보고서 실행 계획')
body+=table('우선순위 실험과 중단 조건',['실험','동일 예산에서 비교','반증 / 중단 조건'],[['E1  Consolidation timing','write summary vs raw+index vs JIT vs hybrid','JIT overhead를 포함하면 utility/cost 이득이 사라짐.'],['E2  Parametric promotion','retrieval vs skill vs latent vs LoRA','추가 훈련·검증·갱신 비용이 미래 reuse saving보다 큼.'],['E3  Dreaming curriculum','uniform replay vs failure-targeted vs policy search','OOD 전이 없이 selection task만 향상.'],['E4  Validity / rollback','TTL-only vs version+dependency+publish gate','신규 사실 반영 지연·false block이 안전 이득을 상쇄.'],['E5  HBF placement','stable-only vs compiled KV vs dynamic full-HBF','같은 package/thermal budget에서 p99/SLO가 baseline보다 악화.'],['E6  Influence policy','retrieval 고정, activation/influence만 변경','calibration 효과가 reader prompt 변경으로 동일하게 재현됨.']], [26,37,37])
body+=h('가장 먼저 얻어야 할 결과')+p('E1과 E4를 CPU/기존 GPU 환경에서 끝내고, 신뢰 가능한 source-to-artifact trace를 확보한다. 그 다음 E2/E3의 학습 효용과 비용을 재현한 뒤, E5에서 physical placement를 바꾼다. 아직 유용성이 입증되지 않은 artifact를 빠르게 옮기는 하드웨어 최적화부터 시작하지 않는다.')
page('연구 로드맵: 바꾸기 전에 반증 조건을 정한다','08 / RESEARCH AGENDA',body,'데이터 계약 → 강한 baseline → 승격 효용 → 물리 배치 순서로 위험을 줄인다.','roadmap')

#35 metrics
body=table('DSE parameter와 response 변수',['독립 변수','권장 sweep 축','기대 관측'],[['Read behavior','sparsity, burst size, hot set, reuse distance','effective BW, read amplification, prefetch hit, stall'],['Update behavior','interval, bytes/update, delta/full rewrite','write coalescing, GC/refresh 포함 physical bytes'],['Agent mix','concurrency, tool-gap 분포, returning sessions','residency survival, TTFT/ITL, SLO goodput'],['Learning mix','replay budget, LoRA rank, candidate count','validation yield, transfer gain, amortization horizon'],['Device budget','HBM 용량/BW, HBF BW/latency, temperature','Pareto frontier: quality·latency·cost·endurance'],['Control policy','promotion threshold, abstain, invalidate fan-out','unsafe reuse, rebuild cost, rejected beneficial updates']], [24,39,37])
body+=h('실험 trace의 최소 레코드')
body+=formula('⟨task, object, source-version, artifact-version, phase, tier, bytes, time, outcome⟩','여기에 model/tokenizer/decoder revision, tool snapshot, eval dataset revision을 manifest로 연결한다.')
body+=col(h('같은 useful work를 비교한다')+p('한 설계가 낮은 품질의 답을 더 빨리 내는 경우 throughput만 비교하면 잘못된 결론에 이른다. acceptance threshold를 통과한 task의 goodput을 우선 측정한다. 평균 latency뿐 아니라 synchronous revalidation이 일으키는 tail도 기록한다.'),
h('한 가지 “효율 점수”로 뭉개지 않는다')+p('정확도와 비용의 단위가 다르므로 임의 가중치를 붙인 단일 점수만 발표하지 않는다. 먼저 Pareto frontier를 공개하고, 그 위에서 workload별 SLO·전력·endurance 조건에 맞는 점을 선택한다.'))
body+=panel('수명 비용의 추가 경계','HBF endurance 분석에서는 host write와 NAND program bytes를 분리한다. refresh가 버전 교체·GC와 겹칠 때 두 번 세지 않도록 simulator accounting을 점검한다. logical commit count만으로 device lifetime을 추정하지 않는다.'+C('hbfsim','hbflex'))
page('DSE를 위한 계측 명세','08 / MEASUREMENT BLUEPRINT',body,'Logical intelligence metric과 physical system metric을 join할 수 있어야 한다.','measurement')

#36 industry
body=p('제품에서는 memory가 독립 기능을 넘어 background worker와 framework primitive로 내려오고 있다. 이는 연구 메커니즘의 실증과는 다른 종류의 근거다. 제품 문서는 제공되는 API와 수명 관리 경계를 보여주지만, 특정 memory algorithm의 일반적 우월성을 입증하지는 않는다.')
body+=table('확인한 제품 신호',['자료 / 시점','공식적으로 설명된 동작','보고서 해석'],[['Vercel eve<br>2026-09-09'+C('eve'),'named slot에 provider와 scope를 지정. turn 전 관련 memory를 context에 추가. private Blob 기반 지속 저장.','memory isolation과 backend 교체가 framework contract가 된다.'],['Grok Build<br>2026-09-16'+C('grok'),'완료 turn에서 durable note를 capture. /dream으로 topic Markdown에 병합; 현재 대화 지시 우선.','background semantic organization. 정책 탐색형 dreaming과 구별.'],['Hindsight Mental Models<br>2026-10-06 확인'+C('hindsight'),'standing answer를 background 또는 schedule로 갱신. freshness·source scope·version history 제공.','read-time 합성 비용을 사전 계산된 view로 옮기는 제품 패턴.'],['Sandisk / SK hynix<br>2026-08-03'+C('sandisk'),'OCP HBF specification 공개와 ecosystem 협력 발표.','capacity tier가 표준화 대상으로 구체화. 실측 양산 성능은 별도.']], [25,44,31])
body+=h('Community는 문제를 찾는 데 사용한다')+p('직접 확인한 r/LLMDevs 게시글은 chat log와 embedding만으로는 “flat recall”에 머문다는 문제를 제기하며 cross-session state를 어떻게 다루는지 묻는다. 이는 설계 질문의 현장성을 보여주는 사례일 뿐, 대표 표본이나 비교 실험은 아니다.'+C('community'))
body+=panel('산업 전망의 중심','가장 성숙한 단기 기회는 source/version/scope를 가진 durable store와 background materialized view다. 장기적 차별화는 그 위에서 어떤 상태를 학습하고 어떤 artifact로 compile할지 결정하는 정책에 있다. 단기 persistence와 장기 self-improvement를 같은 maturity로 평가하지 않는다.')
page('논문 밖에서도 control plane이 구체화된다','09 / INDUSTRY & COMMUNITY',body,'API·운영 기능의 성숙과 신경망 학습 메커니즘의 성숙을 분리해 읽는다.','industry')

#37 errata
body=p('이전 GitHub 인덱스에는 일부 논문의 arXiv 주소가 제목과 일치하지 않거나 이번 대조에서 확인되지 않는 문제가 있었다. 아래는 PDF에서 바로잡은 항목이다. 링크 실패만으로 논문이 존재하지 않는다고 단정하지 않았고, 원문 제목 검색과 올바른 페이지 대조가 된 경우에만 정정했다.')
body+=table('이번 판에서 바로잡은 식별자',['연구','이전 인덱스 ID','이 PDF의 확인된 ID'],[['Hindsight Memory-PRM','2608.31254','2608.29605'+C('hprm')],['KV-streams','2609.35893','2609.35750'+C('kvstreams')],['HBFSim','2609.09902','2609.09800'+C('hbfsim')],['Mem++','2609.39246','2610.02002'+C('memplus')],['Schema','2609.39943','2609.39140'+C('schema')],['Beyond Memory / PoS','2609.40334','2610.01415'+C('pos')],['ActiveSaddler','2609.40515','2610.00906'+C('active')],['EfficientAgent','2609.34901','2609.33762'+C('efficient')]], [36,31,33])
body+=col(h('주장의 강도도 정정했다')+p('slow CMS = HBF라는 표현은 조건부 placement 가설로 바꿨다. HBF write endurance를 commit 횟수로 근사한 서술도 physical bytes·refresh·GC를 포함하도록 수정했다. 두 변경은 문헌의 실제 결과와 우리의 추론을 분리하기 위한 것이다.'),
h('숫자와 버전의 해석')+p('MemCodex의 10.1% 상대 향상은 10.1%p와 다르다. Mem++의 strongest memory baseline 대비 향상과 RAG 대비 향상도 구별했다. HBFSim 20.8×는 simulator 실행 속도이며 inference speedup으로 표기하지 않았다.'+C('memcodex','memplus','hbfsim')))
body+=panel('발행 범위','정정은 이 PDF와 동봉한 reference/source package에 반영했다. 이전 Markdown 파일이 자동으로 교체되었다고 가정하지 않는다. 서지 검토 범위와 pinned version은 참고문헌 대장에 남겼다.','warn')
page('출처 감사와 정정 기록','09 / SOURCE AUDIT',body,'잘못된 링크에 더 좋은 디자인을 입히지 않는다. publication-quality의 출발점은 출처 계약이다.','errata')

#38 supplemental1
body=p('아래 항목은 이전 주간 조사에서 언급한 추가 자료다. <strong>이번 판에서는 원문 방법·수치·버전을 충분히 재검증하지 않아 정량 결론의 근거에서 제외</strong>했다. 누락을 숨기지 않고 후속 조사 queue로 보존한다. 제목은 이전 기록의 표기를 사용하며 별도 검증 없이 링크나 성과 수치를 새로 부여하지 않는다.')
extra=[('Episodic / structure','PolyMemDB; MemForest; Activity Frames; Jev-Mem; Persistent Discovery Context; Selective Forgetting'),('Compression / transfer','Agent Memory Distillation; Explicit, Not Longer; Break It Down, Pass It On; StateComp; GEM'),('Credit / influence','RoMeRL; MemTrapBench; Controlled Memory Interference; MeClear; TRACER; Representational Empowerment'),('Validity / admission','AuthMem-Bench; Endogenous Authorization Laundering; Memory Trust Gap; Grounding Agent Memory; Typed Intention Store'),('Lifecycle / transaction','CommitKV; SuperLocalMemory 4.0; MemoryCPT; MemTX / MemTxn; TransMem'),('Routing / multi-agent','Gated-Memory Routing; HiPS; Chain-of-Experience; Working Set of a Coding Agent')]
body+=table('이전 조사 보존 목록 A',['주제','기록된 자료명'],extra,[29,71])
body+=h('재검증 체크리스트')+p('① 제목과 arXiv ID가 일치하는지 ② v1과 최신판의 날짜·주장이 같은지 ③ 수치가 absolute/relative/%p 중 무엇인지 ④ model·reader·harness가 통제됐는지 ⑤ validation budget까지 비용에 들어갔는지 ⑥ 실제 구현, simulator, position paper 중 무엇인지 확인한다.')
body+=panel('왜 모두 같은 신뢰도로 인용하지 않는가?','총망라된 자료집은 모든 항목을 사실로 승격하는 문서가 아니다. <strong>검증된 근거와 아직 검증되지 않은 기록을 함께 보존하되, 서로 다른 지위를 부여하는 것</strong>이 장기적으로 재사용 가능한 리서치 메모리다.')
page('보존하되 근거로 승격하지 않은 자료','09 / EXTENDED TRACKING REGISTER A',body,'핵심 보고서의 55개 출처와, 이전 주간 기록의 추가 후보를 분리한다.','tracking-a')

#39 supplemental2
body=table('이전 조사 보존 목록 B',['주제','기록된 자료명'],[('Neural / parametric','TMEM; PEAM; COVE; UniMem; SelfMem; LiveMem; Language Models Need Sleep'),('Dream / self-improvement','Discovery by Dreaming; PILOT in the Loop; RSIAgent; Metaⁿ; AREX-2; On the Fragility of Self-Improving Agents'),('Runtime / agents','FMOS position paper; Cognitive Extensions for Dual-Process Language Agents; Auditing Harness Tampering'),('KV / serving','AgentKV; PReCache; HeadWiseKV; SGD-KV; MILO; AgentZip'),('HW / evaluation','Hardware-Managed Heterogeneous HBM and Flash / HMA; Thinkingbox; Agent Memory Challenge Cycle 2'),('Industry / practitioner','MongoDB Agent Memory Inside the Harness; PolarDB-X Agentic Memory; Heavybit LMCache interview; PLUR restore/rollback runbooks; 개별 weekly Reddit threads')],[29,71])
body+=h('이 보고서에서 대체 확인한 근거')+p('persistent neural state는 TTT·Titans·Memory Caching·RPMem으로, JIT와 non-destructive write는 JitMem·Mem++로, causal credit은 Hindsight Memory-PRM·Credit Without Ground Truth·DRSR로 다뤘다. systems/HBF는 AgentSysBench·EfficientAgent·ReCache·ATTUNER·FLINT·HBFlex·HBFSim 등 확인한 자료에 근거했다.')
body+=h('다음 판에서 우선 볼 연결')+p('PReCache와 AgentKV는 multi-LoRA 공유와 phase-aware state를 보완할 수 있다. Grounding Agent Memory는 외부 세계와의 재검증을, MeClear는 query-scoped suppression을 보완할 수 있다. 이 가능성은 연구 후보의 동기이며, 이번 판에서 해당 논문의 세부 성과를 확정하는 것은 아니다.')
body+=panel('자료 수와 연구 품질은 같은 지표가 아니다','같은 source를 여러 번 요약해도 독립 evidence는 하나다. 서로 다른 글이 동일 실험을 재인용한 경우에도 evidence count를 늘리지 않아야 한다. 이 원칙은 리서치 보고서와 agent memory 모두에 적용된다.','dark')
page('후속 검증이 필요한 연결 연구','09 / EXTENDED TRACKING REGISTER B',body,'기존 조사 범위는 유지하되, 정확성이 확인되지 않은 수치를 다시 발행하지 않는다.','tracking-b')

#40 glossary
body=table('이 보고서의 용어 계약',['용어','의미 / 구분'],[['Sleep-time compute','미래 질의를 위해 사전에 수행하는 계산. 본문에서는 학습·precompute·maintenance lane을 구분한다.'],['CMS','Nested Learning 원문의 Continuum Memory System. 외부 multi-tier DB와 동일한 구현이 아니다.'],['Canonical state','버전·출처 계약 아래 유지하는 기준 상태. 항상 진실이거나 불변이라는 뜻은 아니다.'],['Materialization','canonical/latent state에서 현재 실행에 필요한 prompt/KV/adapter/program을 만드는 과정.'],['Promotion','일회 경험이나 후보를 더 넓게 재사용되는 artifact로 승인하는 과정. 물리 이동과 별도.'],['Invalidation','객체나 파생 artifact의 사용 자격을 무효화. 원본 삭제나 byte rewrite와 다르다.'],['Rollback / compensation','이전 generation 복귀와 외부 side effect 상쇄는 다른 연산. fact-level unlearning과도 다르다.'],['Fast weights / LoRA','빠르게 바뀌는 parameter state와 저랭크 parameterization. LoRA라는 형식 자체가 학습 방식을 정하지 않는다.'],['HBF / HBM','flash 기반 capacity tier와 높은 대역폭의 DRAM 계층. 이름의 bandwidth만으로 read/write/thermal 특성을 같게 보지 않는다.'],['Causal utility','명시한 개입 조건에서 with/without 또는 alternate-action 결과의 차이. proxy의 범위를 기록한다.']], [28,72],compact=True)
body+=panel('최종 연구 명제','향후 AI memory의 핵심은 “더 많은 경험”이 아니라, <strong>근거가 남아 있고, 현재도 유효하며, 필요한 실행 형태로 바꿀 수 있고, 미래 비용을 실제로 줄이는 상태</strong>다. 이 명제는 연구 방향이다. 구현의 가치는 같은 예산의 강한 baseline과 비교해 결정된다.','dark')
page('용어와 최종 명제','10 / GLOSSARY & CONCLUSION',body,'같은 단어로 서로 다른 계층을 부르지 않으면, 실험 가설도 더 명확해진다.','glossary')

#41-48 reference pages 7 each
for start in range(0,len(S),7):
 body=''
 if start==0:body=p('제출일·갱신일·검토 범위를 기록했다. 제목과 원문 링크를 대조했으며, 열람 범위를 전체 본문 검토나 독립 재현으로 과장하지 않았다.')
 for s in S[start:start+7]:
  typ={'preprint':'P · PREPRINT','paper':'P · JOURNAL','official':'D · OFFICIAL','community':'C · COMMUNITY'}[s['type']]
  author=s['authors'] if s['authors']!='논문 저자진' else ''
  body+=f'<div class="refitem" id="ref-{s["id"]}"><div class="reftitle">[{s["id"]:02d}] {E(s["title"])}</div><div class="refmeta">{E(author)+" · " if author else ""}{E(s["date"])}'+(f' · 갱신 {E(s["updated"])}' if s['updated'] else '')+f' · {typ}</div><div class="refnote">{E(s["note"])}</div><div class="refmeta">검토: {E(s["review_scope"])} · 확인 2026-10-06</div><a class="url" href="{s["url"]}">{E(s["url"])}</a></div>'
 page(f'주석형 참고문헌 {start//7+1:02d}','REFERENCES / SOURCE REGISTER',body,deck='',pid='references' if start==0 else f'references-{start//7+1}')

#49 media notes
body=p('본문의 모든 diagram과 chart는 이 보고서를 위해 작성한 SVG 벡터 원본이다. 원문 figure를 이미지 파일로 복사하지 않았다. 논문 메커니즘을 설명하는 독립 도식과 수치의 재시각화는 각 caption에 출처·변환 유형을 표시했다. 표지 역시 개념을 추상화한 벡터 디자인으로, 뇌 구조나 실제 memory device의 현미경 사진이 아니다.')
body+=table('미디어 제작·해석 계약',['구분','제작 / 해석'],[['메커니즘 도식','논문 또는 다수 문헌의 개념을 다시 설계. 실제 구현 상세를 생략한 경우 caption에 명시.'],['데이터 그래프','JitMem·LIMBO 등 동일 연구 내부의 확인된 수치만 사용. 축·단위·비교 조건 명시.'],['시스템 설계도','본 보고서의 가설. 기존 paper나 Melete/HAT가 동일하게 구현됐다는 의미가 아님.'],['비용 예시','손익 분기점 그래프의 비용 단위와 수치는 가상. 실측 결과와 시각적으로 구별.'],['수치 검증 한계','모든 benchmark는 저자 보고값이며 실행 재현 없음. 보고서 자체의 benchmark dataset은 없음.']], [25,75])
body+=h('재현 가능한 편집 원본')+p('동봉 패키지에는 PDF, HTML 원본, CSS, SVG assets, reference JSON/CSV, source-audit notes, figure/table manifest, 생성 Python 코드를 포함한다. 사용한 글꼴 파일 자체는 배포하지 않는다. 빌드는 한국어 글꼴과 WeasyPrint가 설치된 환경을 전제로 한다.')
body+=h('판권·발행 안내')+p('문헌 저작권은 각 저자와 발행처에 있다. 본 보고서는 출처를 설명·비평·종합한 독립 연구 문서이며 Google, Sandisk, SK hynix, Vercel, Vectorize, xAI 또는 학회/학술지의 승인·공식 발행을 주장하지 않는다. 본 보고서의 기술 제안은 검증 전 가설이며 제품 보증이나 투자 의견이 아니다.')
body+=panel('Version 2.0 · 2026-10-06','기존 Markdown 종합을 한국어 PDF로 재구성하고, 식별자·주장의 강도·단위·근거 범위를 정정했다. 마지막 검토일 이후 논문과 제품 문서는 바뀔 수 있다. 정정 시에는 reference ID와 figure/table 번호를 기준으로 추적한다.')
page('매체 출처와 발행 정보','PRODUCTION NOTES',body,'원본 그림의 의미와 보고서의 해석을 구별할 수 있도록 제작 이력을 남긴다.','production')

# Export build, source and media manifests.
css=(ROOT/'build/report_style.css').read_text()
html='<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>Sleep-time Compute & Neural Memory | 통합 리서치 보고서</title><meta name="author" content="Melete Research / ChatGPT-assisted synthesis"><meta name="description" content="Sleep-time compute, agent memory, HOPE/CMS, HBM/HBF co-design: Korean research report with evidence-audited references and original vector figures."><style>'+css+'</style></head><body>'+''.join(PAGES)+'</body></html>'
(ROOT/'build/report.html').write_text(html)
(ROOT/'sources/media_manifest.json').write_text(json.dumps(MEDIA,ensure_ascii=False,indent=2))
(ROOT/'sources/table_manifest.json').write_text(json.dumps(TABLES,ensure_ascii=False,indent=2))
(ROOT/'sources/contents.json').write_text(json.dumps(NAV,ensure_ascii=False,indent=2))
print('HTML pages intended:',len(PAGES),'figures',len(MEDIA),'tables',len(TABLES),'sources',len(S))
doc=HTML(string=html,base_url=str(ROOT/'build')).render()
print('Rendered pages:',len(doc.pages))
doc.write_pdf(str(ROOT/'deliverables/Neural_Memory_Report_2026-10-06_KO.pdf'))
# Page boundary diagnostics for QA: gather headings and y extents.
layout=[]
for i,pg in enumerate(doc.pages):
 headings=[];sections=[]
 for box in pg._page_box.descendants():
  if getattr(box,'element_tag',None)=='h2' and hasattr(box,'text'):headings.append(box.text)
  if getattr(box,'element_tag',None)=='section':sections.append({'id':box.element.get('id'),'y':round(box.position_y,1),'h':round(box.height,1)})
 layout.append({'page':i+1,'headings':headings,'sections':sections})
(ROOT/'build/layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2))
