#!/usr/bin/env python3
"""Append a study-QA entry to the associative-memory DB and regenerate the index.

Each question Jimmy asks while studying is classified along taskops' "안다 4축"
(uncertaintyState: known / known_unknown / unknown_known / unknown_unknown) and stored
as an associative-memory record, keyed by concepts so it can be recalled/clustered later
for the seminar storyline.

Usage:
  python3 append_qa.py '<entry json>'      # append one entry (id auto-assigned if omitted)
  python3 append_qa.py --reindex           # rebuild LOG.md + INDEX.md from qa.jsonl
"""
import json, sys, os
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "qa.jsonl")
LOG = os.path.join(HERE, "LOG.md")
INDEX = os.path.join(HERE, "INDEX.md")
AXES = ["known_unknown", "unknown_unknown", "unknown_known", "known"]
AXIS_KR = {
    "known": "안다는 걸 안다 (known)",
    "known_unknown": "모른다는 걸 안다 (known_unknown → decomposition)",
    "unknown_known": "아는데 안 드러남 (unknown_known → prototype/react)",
    "unknown_unknown": "모른다는 것도 모른다 (unknown_unknown → exploration)",
}


def load():
    if not os.path.exists(DB):
        return []
    return [json.loads(l) for l in open(DB, encoding="utf-8") if l.strip()]


def next_id(entries):
    n = 0
    for e in entries:
        try:
            n = max(n, int(str(e.get("id", "Q0"))[1:]))
        except Exception:
            pass
    return f"Q{n+1:03d}"


def reindex(entries):
    # LOG.md — human-readable, chronological
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("# QA LOG — 공부 질문 기록 (chronological)\n\n")
        f.write(f"총 {len(entries)}건.\n\n")
        for e in entries:
            ax = e.get("axis", {})
            f.write(f"## {e.get('id')} · {e.get('date','')} · {e.get('topic','')}\n\n")
            f.write(f"**Q.** {e.get('question','')}\n\n")
            f.write(f"**A.** {e.get('answer_summary','')}\n\n")
            f.write(f"- 축: `{ax.get('before','?')}` → `{ax.get('after','?')}`"
                    + (f" · comprehension: {ax.get('comprehension','')}" if ax.get('comprehension') else "") + "\n")
            if ax.get("surfaced"):
                f.write("- 새로 드러난 것: " + "; ".join(ax["surfaced"]) + "\n")
            if e.get("concepts"):
                f.write("- 개념 key: " + ", ".join(e["concepts"]) + "\n")
            if e.get("think_about"):
                f.write("- 생각할 것: " + "; ".join(e["think_about"]) + "\n")
            if e.get("storyline_seed"):
                f.write("- storyline seed: " + e["storyline_seed"] + "\n")
            if e.get("links"):
                f.write("- 연상: " + ", ".join(e["links"]) + "\n")
            f.write("\n")

    # INDEX.md — associative recall: by axis, by concept, open threads, storyline seeds
    by_axis = defaultdict(list)
    by_concept = defaultdict(list)
    threads = []
    seeds = []
    for e in entries:
        ax = e.get("axis", {})
        by_axis[ax.get("before", "?")].append(e)
        for s in ax.get("surfaced", []):
            q = s.split(":", 1)[0].strip()
            if q in AXES:
                by_axis[q].append(e)
        for c in e.get("concepts", []):
            by_concept[c.lower()].append(e["id"])
        for t in e.get("think_about", []):
            threads.append((e["id"], t))
        if e.get("storyline_seed"):
            seeds.append((e["id"], e.get("topic", ""), e["storyline_seed"]))

    with open(INDEX, "w", encoding="utf-8") as f:
        f.write("# QA INDEX — 연상 회상용 색인 (auto-generated)\n\n")
        f.write(f"{len(entries)}건. story line 짤 때 여기서 꺼낸다.\n\n")
        f.write("## 1. 안다-4축별 클러스터\n\n")
        for a in AXES:
            es = by_axis.get(a, [])
            f.write(f"### {AXIS_KR[a]} — {len(es)}건\n")
            for e in es:
                f.write(f"- {e['id']}: {e.get('question','')[:80]}\n")
            f.write("\n")
        f.write("## 2. 개념 key별 클러스터 (연상)\n\n")
        for c in sorted(by_concept, key=lambda k: -len(by_concept[k])):
            ids = sorted(set(by_concept[c]))
            f.write(f"- **{c}** ({len(ids)}): {', '.join(ids)}\n")
        f.write("\n## 3. 열린 실 (think_about)\n\n")
        for qid, t in threads:
            f.write(f"- [{qid}] {t}\n")
        f.write("\n## 4. Storyline seeds\n\n")
        for qid, topic, s in seeds:
            f.write(f"- [{qid}] ({topic}) {s}\n")


def main():
    entries = load()
    if len(sys.argv) >= 2 and sys.argv[1] == "--reindex":
        reindex(entries)
        print(f"reindexed {len(entries)} entries")
        return
    if len(sys.argv) < 2:
        print("usage: append_qa.py '<entry json>' | --reindex", file=sys.stderr)
        sys.exit(1)
    entry = json.loads(sys.argv[1])
    entry.setdefault("id", next_id(entries))
    entries.append(entry)
    with open(DB, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    reindex(entries)
    print(f"appended {entry['id']} ({entry.get('topic','')}); total {len(entries)}")


if __name__ == "__main__":
    main()
