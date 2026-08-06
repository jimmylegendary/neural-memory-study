#!/usr/bin/env python3
"""Verification for claims/stc-v2/bundle.json — structure AND grade consistency.

Why this exists next to verify_bundle.py
----------------------------------------
verify_bundle.py (merge round, 2026-08-06) checks that the bundle is well formed:
ids unique, evidence admissible and git-pinned, JSON pointers resolve, transcribed
metrics byte-match.  All of that is sound and is re-implemented here independently.

What it does NOT check — and what let a falsified claim through the veridraft gate —
is whether a STATEMENT is consistent with the result.json it cites.  The veridraft
gate is structural too: it counts admissible evidence, it never reads the claim.
So a claim can assert an ordering that its own transcribed numbers refute and still
score [PASS].  Exactly that happened to
p1_stc_x1_regime_ordering_is_the_load_bearing_result (deleted 2026-08-06).

Sections B and C below are the grade-consistency layer: the honesty contract of
experiments/REPRODUCE.md §0 turned into executable assertions, plus regression
tests pinning the two defects this round found so they cannot come back.

Known limits — state them rather than overclaim
-----------------------------------------------
* Section C is a NECESSARY, not a sufficient, condition. It asks whether a number
  a claim asserts exists anywhere in the result.json it cites. A number that
  occurs somewhere unrelated therefore passes: an invented census count of "13"
  slips through because X3 really does contain 13 (a per-file n_hits, and "13
  stickers" inside a quoted GSM8K example). Scoping the search to each claim's own
  pointer subtrees was tried and rejected — the JSON-pointer PATH carries values
  the subtree does not (kappa_cost=20.0, 14000_tokens, dense-8B), so it produced
  six false alarms on correct claims and would have needed an allowlist long
  enough to hollow out the check.
* Sections B1-B4 pin the specific defects found in the 2026-08-06 verification
  round plus the standing rules of the honesty contract. They are regression
  tests, not a general grader: a NEW mis-grading in a NEW shape is not caught by
  them, and only a human comparing statement against findings will find it.
* Nothing here judges whether a P2 statement fairly characterises its paper. It
  checks that the numbers exist in the vendored text and that a locator is
  present; faithfulness is a human read.

    python3 claims/stc-v2/verify_claims.py        # from the repo root; exit 0 = pass
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "claims/stc-v2/bundle.json"
EXPS = {
    "X1": "experiments/stc/X1-unified-cost-accounting/result.json",
    "X2": "experiments/stc/X2-state-capacity-accounting/result.json",
    "X3": "experiments/stc/X3-budget-axis/result.json",
    "X4": "experiments/stc/X4-consolidation-decay/result.json",
}

fails: list[str] = []
checks = 0


def check(cond, msg: str) -> bool:
    global checks
    checks += 1
    if not cond:
        fails.append(msg)
    return bool(cond)


_json: dict[str, object] = {}


def load(rel: str):
    if rel not in _json:
        _json[rel] = json.loads((REPO / rel).read_text(encoding="utf-8"))
    return _json[rel]


def walk(doc, ptr: str):
    cur = doc
    for tok in [t for t in ptr.split("/") if t != ""]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


_text: dict[str, str] = {}


def text(rel: str) -> str:
    if rel not in _text:
        p = REPO / rel
        _text[rel] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    return _text[rel]


B = json.loads(BUNDLE.read_text(encoding="utf-8"))
claims, results = B["claims"], B["results"]
manifest = B["provenance_manifest"]
P1 = [c for c in claims if c["type"] == "P1"]
P2 = [c for c in claims if c["type"] == "P2"]

# =============================================================================
# A.  STRUCTURE
# =============================================================================
for k in ("bundle_id", "boundary", "provenance_manifest", "claims", "results"):
    check(k in B, f"A: top-level key missing: {k}")

ids = [c["claim_id"] for c in claims]
dupes = sorted({i for i, n in Counter(ids).items() if n > 1})
check(not dupes, f"A: duplicate claim_id: {dupes}")

for c in claims:
    cid = c.get("claim_id", "<no id>")
    for k in ("claim_id", "type", "statement"):
        check(str(c.get(k, "")).strip() != "", f"A: {cid}: empty/missing field {k!r}")
    check(c.get("type") in ("P1", "P2"), f"A: {cid}: type is {c.get('type')!r}")
    check(bool(c.get("evidence")), f"A: {cid}: evidence is empty")
    for e in c.get("evidence", []):
        check(e["kind"] not in {"note", "prose_note", "generated_text"},
              f"A: {cid}: inadmissible evidence kind {e['kind']!r}")
    if c["type"] == "P1":
        check(bool(c.get("result_refs")), f"A: {cid}: P1 with empty result_refs")
        check(bool(c.get("result_pointers")), f"A: {cid}: P1 without result_pointers")
        check(bool(c.get("caveat_tags")), f"A: {cid}: P1 without caveat_tags")
        check(bool(str(c.get("transport_rule", "")).strip()), f"A: {cid}: P1 without transport_rule")
    else:
        for e in c["evidence"]:
            check(str(e.get("locator", "")).strip() != "",
                  f"A: {cid}: P2 evidence {e['id']} without a source locator")

rids = {r["result_id"] for r in results}
check(len(rids) == len(results), "A: duplicate result_id")
referenced = {r for c in claims for r in c.get("result_refs", [])}
check(rids == referenced,
      f"A: orphan results={sorted(rids - referenced)} undeclared={sorted(referenced - rids)}")

# every ref path exists on disk AND is git-tracked at the pin
def tracked(path: str, commit: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{path}"],
                          capture_output=True).returncode == 0


for c in claims:
    for e in c.get("evidence", []) + (c.get("reading_provenance") or []):
        path, _, commit = e["ref"].partition("@")
        check((REPO / path).exists(), f"A: {c['claim_id']}: evidence path missing on disk: {path}")
        check(bool(commit) and tracked(path, commit),
              f"A: {c['claim_id']}: {path} not git-tracked at {commit[:8]}")

# pointers resolve and transcriptions match, against the WORKING TREE
n_ptr = 0
for r in results:
    src = r.get("source_ref", "").split("@")[0]
    check(bool(src), f"A: {r['result_id']}: no source_ref")
    check(bool(r.get("pointers")), f"A: {r['result_id']}: no pointers")
    for p in r.get("pointers", []):
        n_ptr += 1
        path, _, ptr = p.partition("#")
        path = path.split("@")[0]
        check(path == src, f"A: {r['result_id']}: pointer outside source_ref: {p}")
        try:
            walk(load(path), ptr)
        except Exception as ex:
            check(False, f"A: {r['result_id']}: pointer does not resolve: {p} ({type(ex).__name__})")
    names = [m["name"] for m in r.get("metrics", [])] + list(r.get("verbatim_text_fields") or {})
    check(set(names) == {p.split("#", 1)[1] for p in r.get("pointers", [])},
          f"A: {r['result_id']}: transcribed fields do not cover the pointers 1:1")
    for m in r.get("metrics", []):
        try:
            actual = walk(load(src), m["name"])
        except Exception:
            check(False, f"A: {r['result_id']}: metric {m['name']!r} does not resolve")
            continue
        check(actual == m["value"] and type(actual) is type(m["value"]),
              f"A: {r['result_id']}: metric {m['name']!r} bundle={m['value']!r} file={actual!r}")
    for k, t in (r.get("verbatim_text_fields") or {}).items():
        try:
            actual = walk(load(src), k)
        except Exception:
            check(False, f"A: {r['result_id']}: verbatim field {k!r} does not resolve")
            continue
        exp = actual if isinstance(actual, str) else json.dumps(actual, ensure_ascii=False)
        check(exp == t, f"A: {r['result_id']}: verbatim field {k!r} does not match the result.json")

for c in P1:
    for p in c.get("result_pointers", []):
        n_ptr += 1
        path, _, ptr = p.partition("#")
        try:
            walk(load(path.split("@")[0]), ptr)
        except Exception as ex:
            check(False, f"A: {c['claim_id']}: claim pointer does not resolve: {p} ({type(ex).__name__})")

manifest_claims = {cid for e in manifest.get("experiments", []) for cid in e["claims"]}
check(manifest_claims == {c["claim_id"] for c in P1},
      f"A: manifest experiment->claim map disagrees with the P1 set: "
      f"{sorted(manifest_claims ^ {c['claim_id'] for c in P1})}")

# =============================================================================
# B.  GRADE CONSISTENCY — the honesty contract, executable
#     (experiments/REPRODUCE.md §0: assertive prose may carry ratios, orders of
#      magnitude, crossover positions, orderings and bound classifications;
#      absolutes travel only as lower bounds or as direction.)
# =============================================================================

# B1. X1 — the regime ordering asserted in prose must be the ordering the sweep
#     actually produces, in EVERY swept cell.  This is the regression test for
#     p1_stc_x1_regime_ordering_is_the_load_bearing_result (deleted 2026-08-06:
#     it asserted GSM >> AIME > SWE while all 8 non-null cells give AIME > GSM > SWE).
x1 = load(EXPS["X1"])["Q1_Q2_critical_length_by_regime"]["values"]
conds = list(x1["stateful_gsm_symbolic"]["critical_len_chat"])
orders = set()
for cond in conds:
    row = {r: x1[r]["critical_len_chat"][cond]["L_star_over_len_c"] for r in x1}
    if any(v is None for v in row.values()):
        continue
    orders.add(tuple(k for k, _ in sorted(row.items(), key=lambda kv: -kv[1])))
check(len(orders) == 1, f"B1: the sweep does not induce a single regime ordering: {orders}")
true_order = next(iter(orders))
check(true_order == ("stateful_aime", "stateful_gsm_symbolic", "swe_features"),
      f"B1: unexpected regime ordering in X1: {true_order}")
FALSIFIED_ORDERINGS = [
    r"GSM-?Symbolic\s*>>\s*(Stateful\s*)?AIME",
    r"GSM\s*>>\s*AIME",
]
for c in claims:
    for pat in FALSIFIED_ORDERINGS:
        check(not re.search(pat, c["statement"], re.I),
              f"B1: {c['claim_id']}: asserts the regime ordering GSM >> AIME, which every cell of "
              f"X1's own sweep contradicts (true order: AIME > GSM > SWE)")

# B2. X2 — 3.64 bits/param is a LOWER bound on capacity, so 16/3.64 = 4.4 bounds the
#     quantisation headroom from ABOVE.  No claim may assert a MINIMUM headroom, and
#     no claim may present bits_per_stored_byte without marking it a floor.
bpp = load(EXPS["X2"])["meta"]["sourced_constants"]["bits_per_parameter"]
check(bpp["bound_direction"] == "LOWER BOUND",
      f"B2: X2 no longer declares bits_per_parameter a LOWER BOUND ({bpp.get('bound_direction')})")
MIN_HEADROOM = [r"최소\s*4\.4\s*배", r"at least\s*4\.4", r"적어도\s*4\.4\s*배"]
QUOTED = re.compile(r"'[^']*'")


def asserted(statement: str) -> str:
    """The part of a statement the claim ASSERTS. A withdrawn sentence may be
    reproduced inside single quotes so the reader knows what was withdrawn; that
    is a mention, not an assertion, and is exempt — but only if the claim also
    says it is withdrawing it."""
    if "철회" in statement or "WITHDRAWN" in statement:
        return QUOTED.sub(" ", statement)
    return statement


for c in claims:
    body = asserted(c["statement"])
    for pat in MIN_HEADROOM:
        check(not re.search(pat, body, re.I),
              f"B2: {c['claim_id']}: asserts a MINIMUM 4.4x storage headroom. 3.64 bits/param is a "
              f"lower bound, so 16/3.64 is a CEILING on the headroom, not a floor.")
    check(not re.search(r"양자화는.{0,20}전제(다|이다)", body),
          f"B2: {c['claim_id']}: revives the withdrawn '양자화는 전제다' conclusion, which rested on "
          f"the inverted bound direction")
    if re.search(r"bits/stored-byte|bits_per_stored_byte", c["statement"]):
        check(re.search(r"적어도|하한", c["statement"]),
              f"B2: {c['claim_id']}: quotes bits/stored-byte without marking it a lower bound")
# a claim that quotes a withdrawn sentence must actually withdraw it
for c in claims:
    if re.search(r"최소\s*4\.4\s*배", c["statement"]):
        check("철회" in c["statement"],
              f"B2: {c['claim_id']}: reproduces the '최소 4.4배' sentence without withdrawing it")

# B3. X4 is not an LLM experiment: absolute rho / retention % / K* round counts /
#     required omega / half-life must not appear as transported numbers.
X4_FORBIDDEN = [
    (r"K\s*\*?\s*(?:는|은|=|≈)\s*\d+\s*(?:라운드|rounds?)", "an absolute K* round count"),
    (r"반감기\s*(?:는|은|=|≈)?\s*\d", "an absolute half-life"),
    (r"유지율\s*(?:이|은|는|=|≈)?\s*\d+\s*%", "an absolute retention percentage"),
    (r"라운드당\s*\d+(\.\d+)?\s*(pp|%)", "an absolute per-round rho"),
]
for c in P1:
    if "NOT-AN-LLM-EXPERIMENT" not in c.get("caveat_tags", []):
        continue
    for pat, what in X4_FORBIDDEN:
        check(not re.search(pat, c["statement"]),
              f"B3: {c['claim_id']}: transports {what}; X4's caveat NOT-AN-LLM-EXPERIMENT allows only "
              f"sign, functional form, ordering and crossover")

# B4. CORPUS-SCOPED claims must not generalise past the 29 deep-read papers (X3 D9).
for c in P1:
    if "CORPUS-SCOPED" not in c.get("caveat_tags", []):
        continue
    check(not re.search(r"문헌의\s*\d+\s*%|전체\s*문헌의\s*\d", c["statement"]),
          f"B4: {c['claim_id']}: generalises a rate to the literature; X3 D9 scopes every count to "
          f"'이 corpus의 29편 중'")

# B5. Every P1 claim's caveat_tags must be declared in the manifest grade legend,
#     and the tags that name a grade must match the experiment it cites.
legend = set(manifest.get("grade_legend", {}))
for c in P1:
    unknown = [t for t in c["caveat_tags"] if t not in legend]
    check(not unknown, f"B5: {c['claim_id']}: caveat tags absent from grade_legend: {unknown}")
    exp = {p.partition("#")[0].split("@")[0] for p in c["result_pointers"]}
    check(len(exp) == 1, f"B5: {c['claim_id']}: pointers span several experiments: {sorted(exp)}")
    src = next(iter(exp))
    grade_tag = {EXPS["X1"]: "ACCOUNTING-NOT-MEASURED", EXPS["X2"]: "ACCOUNTING-NOT-MEASURED",
                 EXPS["X3"]: "AUDIT-NOT-MEASUREMENT", EXPS["X4"]: "NOT-AN-LLM-EXPERIMENT"}[src]
    check(grade_tag in c["caveat_tags"] or "WITHDRAWN" in c["caveat_tags"] or
          "CORPUS-SCOPED" in c["caveat_tags"],
          f"B5: {c['claim_id']}: cites {os.path.basename(os.path.dirname(src))} but carries neither "
          f"{grade_tag} nor an exempting tag")

# =============================================================================
# C.  NUMBER PROVENANCE — every number asserted must exist in the warrant
# =============================================================================
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def canon(tok: str) -> str | None:
    """Canonical form of a numeric token, so 12 / 12.0 / 12.00 compare equal.
    Returns None for anything not numeric."""
    try:
        return f"{float(tok.replace(',', '')):.6f}"
    except ValueError:
        return None


def haystack(rel: str) -> set[str]:
    """The SET of numeric tokens the result.json actually contains — as tokens, not
    as a concatenated blob.  Substring matching over a blob is far too permissive:
    it lets an invented '13' match inside the arXiv id '2504.13171'."""
    out: set[str] = set()
    tok = re.compile(r"\d+(?:\.\d+)?")

    def add(s: str):
        for m in tok.finditer(s.replace(",", "")):
            c = canon(m.group(0))
            if c:
                out.add(c)

    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                add(str(k))
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)
        elif isinstance(o, bool) or o is None:
            pass
        elif isinstance(o, (int, float)):
            out.add(f"{float(o):.6f}")
        else:
            add(str(o))

    rec(load(rel))
    return out


# Documented exceptions: a rounding the claim itself flags as approximate.
# Keep this list SHORT and justified — a long allowlist means the check has been
# tuned to pass rather than the bundle fixed.
ROUNDING_OK = {
    ("p1_stc_x1_breakeven_Nq_spread_across_regimes", "37"):
        "rounds Q3 swe_features flops |c_hat|/|c|=4.0 = 36.75, written 'N_q ≈ 37'",
    ("p1_stc_x4_one_over_i_weighting_is_a_property_of_the_schedule", "122"):
        "rounds A1_variance_pinned.ratio_replace_over_accumulate[5] = 121.955195, written '약 122배'",
}
n_num = 0
for c in P1:
    src = next(iter({p.partition("#")[0].split("@")[0] for p in c["result_pointers"]}))
    hay = haystack(src)
    for m in NUM.finditer(c["statement"]):
        tok = m.group(0)
        n_num += 1
        if canon(tok) in hay:
            continue
        if (c["claim_id"], tok) in ROUNDING_OK:
            continue
        ctx = c["statement"][max(0, m.start() - 40):m.end() + 25].replace("\n", " ")
        check(False, f"C: {c['claim_id']}: the number {tok!r} is not in {os.path.basename(os.path.dirname(src))}"
                     f"/result.json  …{ctx}…")

# P2: every number asserted must appear in the vendored source text
P2NUM = re.compile(r"\d+\.\d+|\d{2,}")
PATH = re.compile(r"([\w./\-]+\.txt)")
for c in P2:
    want = sorted(set(P2NUM.findall(c["statement"])))
    if not want:
        continue
    files: set[str] = set()
    for e in c["evidence"]:
        files |= set(PATH.findall(e.get("locator", "")))
        pdf = e["ref"].split("@")[0]
        if pdf.endswith(".pdf"):
            files.add(pdf[:-4] + ".txt")
    body = "".join(text(f) for f in files).replace(",", "")
    for w in want:
        n_num += 1
        check(w in body,
              f"C: {c['claim_id']}: asserts {w!r} but it does not occur in the vendored source "
              f"{sorted(files)}")

# =============================================================================
print(f"bundle           : {BUNDLE}")
print(f"bundle_id        : {B['bundle_id']}   boundary={B['boundary']}")
print(f"claims           : {len(claims)}  (P1={len(P1)}, P2={len(P2)})   duplicate ids: {len(dupes)}")
print(f"results          : {len(results)}   orphans: {len(rids - referenced)}")
print(f"A structure      : {n_ptr} JSON pointers resolved; every transcribed metric/text byte-compared;")
print(f"                   every evidence path present on disk and git-tracked at its pin")
print(f"B grade rules    : X1 regime ordering recomputed from all {len(conds)} swept cells "
      f"(true order: AIME > GSM-Symbolic > SWE-Features);")
print(f"                   X2 bound direction (3.64 bits/param = LOWER bound => 4.4x is a CEILING);")
print(f"                   X4 absolute-transport ban; X3 corpus-scoping; tag/grade coherence")
print(f"C number warrant : {n_num} numbers checked against the cited result.json / vendored source text")
print(f"assertions       : {checks}")
if fails:
    print(f"\nFAILED ({len(fails)}):", file=sys.stderr)
    for f in fails:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("\nVERIFICATION: PASS (0 failures)")
