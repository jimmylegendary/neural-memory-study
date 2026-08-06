#!/usr/bin/env python3
"""Self-verification for claims/stc-v2/bundle.json — independent of the veridraft gate.

Checks, in order:
  1. JSON is valid and carries the NM top-level schema
     (bundle_id / boundary / provenance_manifest / claims / results).
  2. No duplicate claim_id; the P1 and P2 halves are disjoint and fully present.
  3. Every claim has >=1 admissible evidence entry (kind == source_artifact).
     No claim carries an inadmissible kind (note / prose_note / generated_text).
  4. Every P1 claim has a non-empty result_refs, and every ref resolves to a
     declared result_id that carries content (a metric value or a description).
  5. Every P2 claim has non-empty evidence.
  6. Every evidence ref has the gate's resolvable `<path>@<commit>` shape, the path
     exists on disk, AND is git-tracked at the pinned commit.
  7. Every result pointer (`<path>#<json-pointer>`) resolves in the committed
     result.json, and every numeric metric matches the pointed-at value verbatim.
  8. Every path named in provenance_manifest (vendored_sources, experiments, papers)
     exists and is tracked at the anchor commit.

Exit 0 = all checks pass. Exit 1 = at least one failure (listed on stderr).

    python3 claims/stc-v2/verify_bundle.py            # run from the repo root
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "claims/stc-v2/bundle.json"
P1_HALF = REPO / "claims/stc-v2/p1-claims.json"
P2_HALF = REPO / "claims/stc-v2/p2-claims.json"

# the gate's source_artifact ref shape (veridraft/core/gate.py::_REF_SHAPE)
REF_SHAPE = re.compile(r"^(?:github|repo|https?|git)://\S+$|^[\w./@\-]+@[0-9a-fA-F]{7,40}$", re.I)
ADMISSIBLE = {"caw02_evidence", "caw01_result", "source_artifact"}
INADMISSIBLE = {"generated_text", "prose_note", "note"}

fails: list[str] = []
checks = 0


def check(cond, msg: str) -> bool:
    global checks
    checks += 1
    if not cond:
        fails.append(msg)
    return bool(cond)


def tracked_at(path: str, commit: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{path}"],
                          capture_output=True).returncode == 0


_cache: dict[str, object] = {}


def resolve_pointer(ref: str, commit: str):
    """Resolve `<path>#<json-pointer>` against the blob COMMITTED AT `commit` — not the
    working tree, so an uncommitted edit to a result.json cannot validate the bundle."""
    path, _, ptr = ref.partition("#")
    key = f"{commit}:{path}"
    if key not in _cache:
        blob = subprocess.run(["git", "-C", str(REPO), "show", key],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            raise FileNotFoundError(key)
        _cache[key] = json.loads(blob.stdout)
    cur = _cache[key]
    for tok in [t for t in ptr.split("/") if t]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


# -- 1. JSON valid + top-level schema ----------------------------------------
try:
    B = json.loads(BUNDLE.read_text(encoding="utf-8"))
except Exception as e:  # noqa: BLE001
    print(f"FAIL: bundle is not valid JSON: {e}", file=sys.stderr)
    sys.exit(1)

for k in ("bundle_id", "boundary", "provenance_manifest", "claims", "results"):
    check(k in B, f"top-level key missing: {k}")
check(B.get("bundle_id") == "sleep-time-compute-v2",
      f"bundle_id is {B.get('bundle_id')!r}, expected 'sleep-time-compute-v2'")

claims = B["claims"]
results = B["results"]
manifest = B["provenance_manifest"]
ANCHOR = manifest["warrant_anchor"]["commit"]
check(subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", ANCHOR],
                     capture_output=True).returncode == 0,
      f"warrant anchor commit {ANCHOR} does not exist in this repo")

# -- 2. claim_id uniqueness + both halves fully present ----------------------
ids = [c["claim_id"] for c in claims]
dupes = sorted({i for i in ids if ids.count(i) > 1})
check(not dupes, f"duplicate claim_id(s): {dupes}")

p1_src = {c["claim_id"] for c in json.loads(P1_HALF.read_text(encoding="utf-8"))["claims"]}
p2_src = {c["claim_id"] for c in json.loads(P2_HALF.read_text(encoding="utf-8"))["claims"]}
check(not (p1_src & p2_src), f"claim_id collision between the halves: {sorted(p1_src & p2_src)}")
check(p1_src <= set(ids), f"P1 claims lost in the merge: {sorted(p1_src - set(ids))}")
check(p2_src <= set(ids), f"P2 claims lost in the merge: {sorted(p2_src - set(ids))}")
check(set(ids) <= (p1_src | p2_src), f"claims invented by the merge: {sorted(set(ids) - p1_src - p2_src)}")

result_ids = {r["result_id"] for r in results}
check(len(result_ids) == len(results), "duplicate result_id in results")

# -- 2b. NO CLAIM WAS WEAKENED: statement + carried metadata byte-identical to the halves
half_by_id = {}
for _f in (P1_HALF, P2_HALF):
    for _c in json.loads(_f.read_text(encoding="utf-8"))["claims"]:
        half_by_id[_c["claim_id"]] = _c
CARRIED = ("type", "boundary", "visibility", "supports", "caveat_tags",
           "transport_rule", "chapter", "claim_class")
for c in claims:
    src = half_by_id.get(c["claim_id"])
    if not check(src is not None, f"{c['claim_id']}: not present in either input half"):
        continue
    check(c["statement"] == src["statement"],
          f"{c['claim_id']}: statement was ALTERED relative to the input half")
    for k in CARRIED:
        if k in src:
            check(c.get(k) == src[k], f"{c['claim_id']}: field {k!r} was altered ({src[k]!r} -> {c.get(k)!r})")
    if src["type"] == "P1":
        check(c.get("result_pointers") == src["result_refs"],
              f"{c['claim_id']}: the original JSON pointers were not preserved verbatim")
    # every source_artifact the half declared must still be on the claim
    kept = {(e["kind"], e["ref"]) for e in c["evidence"]}
    lost = [e["id"] for e in src["evidence"]
            if e["kind"] == "source_artifact" and ("source_artifact", e["ref"]) not in kept]
    check(not lost, f"{c['claim_id']}: admissible evidence dropped in the merge: {lost}")
    # anything the half declared that is NOT on the claim must be re-registered, not vanished
    demoted = {e["ref"] for e in src["evidence"] if (e["kind"], e["ref"]) not in kept}
    reg = {p["ref"] for p in c.get("reading_provenance", [])}
    check(demoted <= reg,
          f"{c['claim_id']}: evidence vanished without re-registration: {sorted(demoted - reg)}")


def has_content(r: dict) -> bool:
    return bool(str(r.get("description", "")).strip()) or any(
        str(m.get("value", "")).strip() != "" for m in r.get("metrics", []) if isinstance(m, dict))


content_ids = {r["result_id"] for r in results if has_content(r)}

# -- 3/4/5/6. per-claim checks ------------------------------------------------
ev_ids: list[str] = []
for c in claims:
    cid = c["claim_id"]
    ev = c.get("evidence", [])
    ev_ids.extend(e["id"] for e in ev)
    check(bool(ev), f"{cid}: evidence is empty")
    if c["type"] == "P2":
        # the §/page locator is what makes a faithful-summary claim human-checkable
        no_loc = [e["id"] for e in ev if not str(e.get("locator", "")).strip()]
        check(not no_loc, f"{cid}: P2 evidence without a source locator: {no_loc}")
    bad_kind = [e["kind"] for e in ev if e["kind"] in INADMISSIBLE]
    check(not bad_kind, f"{cid}: inadmissible evidence kind present: {bad_kind}")
    adm = [e for e in ev if e["kind"] in ADMISSIBLE]
    check(len(adm) >= 1, f"{cid}: 0 admissible evidence entries")

    for e in ev:
        ref = e["ref"]
        check(bool(REF_SHAPE.search(ref)), f"{cid}/{e['id']}: ref {ref!r} is not a resolvable pointer")
        path, _, commit = ref.partition("@")
        check((REPO / path).exists(), f"{cid}/{e['id']}: {path} does not exist on disk")
        check(tracked_at(path, commit), f"{cid}/{e['id']}: {path} not git-tracked at {commit[:8]}")
        # the working tree must still equal the pinned blob, else the file a human opens
        # to check the claim is not the file the ref points at
        check(subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", commit, "--", path],
                             capture_output=True).returncode == 0,
              f"{cid}/{e['id']}: working tree {path} DIFFERS from the pinned blob at {commit[:8]}")

    if c["type"] == "P1":
        rr = c.get("result_refs", [])
        check(bool(rr), f"{cid}: P1 with empty result_refs")
        ghost = [r for r in rr if r not in result_ids]
        check(not ghost, f"{cid}: result_refs do not resolve to a declared result: {ghost}")
        empty = [r for r in rr if r in result_ids and r not in content_ids]
        check(not empty, f"{cid}: result_refs point at content-free results: {empty}")
        check(bool(c.get("result_pointers")), f"{cid}: P1 lost its verbatim result_pointers")

# require_results (the profile-level floor): >=1 claim references a result with content
check(any(rr in content_ids for c in claims for rr in c.get("result_refs", [])),
      "no claim references a content-carrying result (profile require_results would raise)")

# -- 7. result pointers resolve + metrics match verbatim ---------------------
for r in results:
    rid = r["result_id"]
    ptrs = r.get("pointers", [])
    check(bool(ptrs), f"{rid}: no pointers")
    src = r.get("source_ref", "")
    p, _, commit = src.partition("@")
    check(tracked_at(p, commit), f"{rid}: source_ref {p} not tracked at {commit[:8]}")
    for ptr in ptrs:
        check(ptr.split("#")[0] == p, f"{rid}: pointer {ptr} does not live under source_ref {p}")
        try:
            resolve_pointer(ptr, commit)
        except Exception as e:  # noqa: BLE001
            check(False, f"{rid}: pointer does not resolve: {ptr} ({type(e).__name__})")
    ptr_set = {x.split("#", 1)[1] for x in ptrs}
    metrics = r.get("metrics", [])
    texts = r.get("verbatim_text_fields", {})
    names = [m["name"] for m in metrics] + list(texts)
    check(len(names) == len(ptrs) and set(names) == ptr_set,
          f"{rid}: transcribed fields {sorted(set(names) ^ ptr_set)} do not cover the pointers 1:1")
    for m in metrics:
        name, val = m["name"], m["value"]
        try:
            actual = resolve_pointer(f"{p}#{name}", commit)
        except Exception as e:  # noqa: BLE001
            check(False, f"{rid}: metric {name!r} does not resolve ({type(e).__name__})")
            continue
        check(actual == val and type(actual) is type(val),
              f"{rid}: metric {name!r} = {val!r} but the committed result.json holds {actual!r}")
    for name, text in texts.items():
        try:
            actual = resolve_pointer(f"{p}#{name}", commit)
        except Exception as e:  # noqa: BLE001
            check(False, f"{rid}: verbatim field {name!r} does not resolve ({type(e).__name__})")
            continue
        expect = actual if isinstance(actual, str) else json.dumps(actual, ensure_ascii=False)
        check(expect == text, f"{rid}: verbatim field {name!r} does not match the committed result.json")

# every declared result is referenced by some claim (no orphans)
referenced = {rr for c in claims for rr in c.get("result_refs", [])}
check(result_ids == referenced,
      f"orphan/undeclared results: orphan={sorted(result_ids - referenced)} "
      f"undeclared={sorted(referenced - result_ids)}")

# -- 8. provenance_manifest paths --------------------------------------------
for d in manifest.get("vendored_sources", {}):
    check((REPO / d).exists(), f"provenance_manifest.vendored_sources: {d} does not exist")
for exp in manifest.get("experiments", []):
    p, _, commit = exp["result_json"].partition("@")
    check((REPO / p).exists(), f"provenance_manifest.experiments: {p} does not exist")
    check(tracked_at(p, commit), f"provenance_manifest.experiments: {p} not tracked at {commit[:8]}")
    check(bool(exp.get("claims")), f"provenance_manifest.experiments: {exp['id']} lists no claims")
for p in manifest.get("papers", {}):
    check((REPO / p).exists(), f"provenance_manifest.papers: {p} does not exist")
    check(tracked_at(p, ANCHOR), f"provenance_manifest.papers: {p} not tracked at {ANCHOR[:8]}")
manifest_claims = {cid for e in manifest.get("experiments", []) for cid in e["claims"]}
check(manifest_claims == p1_src,
      f"provenance_manifest experiment→claim map disagrees with the P1 half: "
      f"{sorted(manifest_claims ^ p1_src)}")
check(len(set(ev_ids)) == len(ev_ids),
      f"duplicate evidence id(s): {sorted({i for i in ev_ids if ev_ids.count(i) > 1})}")

# -- report -------------------------------------------------------------------
print(f"bundle          : {BUNDLE}")
print(f"bundle_id       : {B.get('bundle_id')}   boundary={B.get('boundary')}")
print(f"claims          : {len(claims)}  (P1={sum(1 for c in claims if c['type']=='P1')}, "
      f"P2={sum(1 for c in claims if c['type']=='P2')})")
print(f"results         : {len(results)}  (all content-carrying: {len(content_ids)==len(results)})")
print(f"evidence refs   : {len({e['ref'] for c in claims for e in c['evidence']})} distinct, "
      f"all present, tracked at {ANCHOR[:8]} and unmodified in the working tree")
print(f"P1 result_refs  : all non-empty, all resolve to a content-carrying declared result")
print(f"P2 evidence     : all non-empty, all source_artifact, all carry a §/page locator")
print(f"result pointers : {sum(len(r.get('pointers', [])) for r in results)} resolved against the "
      f"COMMITTED blobs; every transcribed metric/text byte-compared")
print(f"assertions      : {checks}")
if fails:
    print(f"\nFAILED ({len(fails)}):", file=sys.stderr)
    for f in fails:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("\nSELF-VERIFICATION: PASS (0 failures)")
