#!/usr/bin/env python3
"""Initial metadata snapshots; identifier resolution is not scientific verification.

Search-record metadata does not establish a complete current version history.
The online builder keeps failed lookups pending rather than deleting records.
"""
import json
from pathlib import Path
I=Path(__file__).resolve().parents[1]/'inputs'
A={
'KV-streams':'2609.35750','Mem++':'2610.02002','Hindsight Memory-PRM':'2608.29605','ActiveSaddler':'2610.00906','Schema':'2609.39140','PoS':'2610.01415','EfficientAgent':'2609.33762','HBFSim':'2609.09800',
'Invalidation Contracts':'2609.00243','PlanFence':'2609.03340','Endogenous Authorization Laundering':'2609.01836','Memory Trust Gap':'2609.01852','Representational Empowerment':'2609.02322','Gated-Memory Routing':'2609.00237','Typed Intention Store':'2609.01272','Persistent Discovery Context':'2609.02129','Selective Forgetting':'2608.28978','Harness Tampering':'2609.00069','HeadWiseKV':'2609.02029','SGD-KV':'2609.03235','Grounding Agent Memory':'2609.11060','MeClear':'2609.09115','MemForest':'2609.08273','UNISON':'2609.09643','AgentZip':'2609.11294','MILO':'2609.29913',
'Lachesis':'2610.08378','DPPM':'2610.11776','SGUID':'2610.12367','REMORY':'2610.11287','Attic-KV':'2610.12133','MINDSET':'2610.08586','Beyond Corrected Memory':'2610.08101','Persistent Memory in Multi-Agent LLM Inference':'2610.07782'}
R={'checked_at':'2026-10-12','rule':'Aliases apply only to recovered task observations, never globally to a same-acronym paper in another corpus. A resolved candidate identifier is not metadata/result verification.','aliases':A,'dois':{},'urls':{'HMA':'https://ora.ox.ac.uk/objects/uuid%3Aa9f80041-ff2e-4714-9bc4-51c513604d36'},'publisher_metadata':{}}
M={}
def rec(a,title,date,authors='',updated='',version=None,method='arxiv_search_metadata',notes=''):
    M[a]={'arxiv_id':a,'title':title,'published':date,'updated':updated,'latest_version':version,'authors':authors.split(';') if authors else [],'url':'https://arxiv.org/abs/'+a,'checked_at':'2026-10-12','method':method,'status':'ok','notes':notes}
rec('2504.13171','Sleep-time Compute: Beyond Inference Scaling at Test-time','2025-04-17','Kevin Lin;Charlie Snell;Yu Wang;Charles Packer;Sarah Wooders;Ion Stoica;Joseph E. Gonzalez','2025-04-17',1,'arxiv_abstract_metadata')
rec('2605.26099','Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference','2026-05-25','Sangyun Lee;Sean McLeish;Tom Goldstein;Giulia Fanti','2026-06-05',3,'arxiv_abstract_metadata')
rec('2606.03979','Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories','2026-06-02','Ali Behrouz;Farnoosh Hashemi;Adel Javanmard;Vahab Mirrokni','2026-07-10',2,'arxiv_abstract_metadata','arXiv comments state that a version was available on OpenReview in September 2025. Exact earlier release day was not verified; arXiv-v1 ordering is explicitly labelled.')
rec('2605.20616','Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents','2026-05-20','Chongrui Ye;Yuxiang Liu;Yu Wang;Haofei Yu;Yining Zhao;Ge Liu;Julian McAuley;Jiaxuan You','2026-05-20',1,'arxiv_abstract_metadata')
rec('2603.14517','Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models','2026-03-15','Ying Xie','2026-03-15',1,'arxiv_abstract_metadata')
rec('2609.39765','MemCodex: Self-Programming Hierarchical Memory for Language Agents','2026-09-30','Xiaoqiang Wang;Bang Liu','2026-09-30',1,'arxiv_abstract_metadata','Work in progress.')
rec('2610.10533','EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory','2026-10-07','Hongru Cai;Ran Wei;Wenjie Wang;Chengfa Wu;Ning Song;Yongqi Li;Wenjie Li','2026-10-07',1,'arxiv_abstract_metadata')
rec('2610.08378','Lachesis: Lifetime-Aware KV Cache Placement for Agent Serving across HBM and High-Bandwidth Flash','2026-10-06','Jaehoon Yang;Jeongmin Lee;Haneul Park;Seung Yul Lee;Nam Sung Kim;Jae W. Lee')
rec('2609.01836','Agent Memory Is a Surface for Endogenous Authorization Laundering','2026-09-01','Tommaso Cerruti;Mika Okamoto;Ansel Kaplan Erol')
rec('2609.01852','The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents','2026-09-01','Jundong Hu;Shekar Ramachandran')
rec('2609.02322','What Is Worth Representing? Representational Empowerment for Continual Model Construction','2026-09-02','Fei Dai;Hanqi Zhou;Alison Gopnik;Charley Wu')
rec('2609.00237','Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems','2026-08-31','Rakibul Hasan Rajib;Mengxin Zheng;Qian Lou','2026-09-09',2,'arxiv_abstract_metadata','Author spelling follows current arXiv abstract page; older search metadata used Mengxing.')
rec('2609.01272','Making Prospective Memory SLM-Shaped: Typed Intention Stores for Small-Model Agents','2026-09-01','Jinqing Zhao;Chengcan Wu')
rec('2609.02129','Beyond Context Windows: Persistent Discovery Context for Data-Centric Agents','2026-09-02','Jalal Mahmud')
rec('2609.00069','Auditing Harness Tampering in Self-Improving Agents','2026-08-30','Xing Wang;Xiaoyi Zhang;Jie Shao')
rec('2609.02029','HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid Long-Context Language Models','2026-09-02','Renjie Xie;Juncheng Yang;Aoting Hu;Mingxi Zhang;Liyao Wu;Zheheng Hong;Wei Xu')
rec('2609.03235','SGD-KV: Summarization Guided KV Cache Compression','2026-09-03','Zeyu Liu;Woomin Song;Xuandi Fu;Sai Muralidhar Jayanthi;Vivek Govindan;Aram Galstyan;Sravan Babu Bodapati;Srikanth Ronanki',notes='USC ISI page lists a 2025 workshop year; stored arXiv-v1 date is 2026-09-03. Venue/year conflict requires review; no silent normalization.')
rec('2609.11294','Memory Compression for High-Fanout Agent Sandboxes','2026-09-10','Mengming Li;Ceyu Xu;Qijun Zhang;Jiangnan Yu;Xiangfeng Sun;Haohui Mai;Zhiyao Xie')
rec('2609.08273','MemForest: Efficient Agent Memory Management via EventTree Partitioning and Progressive Merging','2026-09-08','Junxi Wang;Te Sun;Jiayi Zhu;Chen Zhang;Siyuan Li;Xuyang Liu;Zichen Wen;Xiaobing Tu;Jinkui Ren;Xiantao Zhang;Ziqi Yuan;Linfeng Zhang',notes='Not the different MemForest paper arXiv:2605.23986.')
rec('2609.29913','MILO: Efficient Many-shot In-Context Learning with Block-wise Low-rank Compression','2026-09-24','Youpeng Zhao;Tian Tan;Liqian Peng;Jun Wang;Alec Go')
rec('2610.08586','MINDSET: Energy-based Schema Evolution for Long Conversational Agent Memory','2026-10-06','Sujato Dutta;Sreekruthy Tummala;Shashank Vanga;Ayushmi Pavani')
rec('2610.08101','Beyond Corrected Memory: Execution Consistency in Multi-Agent Systems','2026-10-06','Zhe Yu;Zixuan Wang;Peidong Wang;Hehai Lin;Ruochen Zhao;Chengwei Qin')
rec('2610.07782','Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell','2026-10-06','Hochan Son;Kyungdoe Han;Jaehan Koh;Xiaowu Dai;Wenlu Xu;Guang Cheng')

if __name__=='__main__':
    I.mkdir(parents=True,exist_ok=True)
    for filename,data in [('resolutions.json',R),('verified_metadata.json',M)]:
        path=I/filename
        # Existing human-reviewed amendments are authoritative on later builds.
        if not path.exists():
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(len(A),'initial alias resolutions;',len(M),'primary metadata snapshots')
