#!/usr/bin/env python3
"""Build the laya-judge pilot manifest (MISSION D, track D).

Stratified sample: 30 SGD (incl. silent-failure candidates), 10 MultiWOZ,
10 tau-bench. Windowing: per turn, text = last 2 turns + current, joined;
at most 8 windows per conversation, EVENLY SPACED across the full trajectory
(keeps turn 0 = common init, and the final line = close categories).
Deterministic: seed 20261027. Output: JSON LIST of items
{corpus, id, path, outcome, n_turns, windows:[{turn_idx, text}]}.

No laya calls. Local only.
"""
import csv
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # builds/trace2grammar
TB = r"C:/Users/U77225D/Desktop/proyectos/hf/builds/where-conversations-die/tb_transcripts"
OUT = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/systemone_spec_2026-10-27/pilot_manifest.json"
SEED = 20261027
MAX_WINDOWS = 8
rng = random.Random(SEED)


def parse_turns(path):
    """Transcript dialect -> turns. A turn starts at each 'user:' line and
    carries the following assistant/tool lines until the next user line.
    Returns list of turn texts (prefixed with the original line tags)."""
    turns, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("user:"):
                if cur is not None:
                    turns.append("\n".join(cur))
                cur = [line]
            elif cur is not None:
                cur.append(line)
    if cur is not None:
        turns.append("\n".join(cur))
    return turns


def windows_for(turns, cap=MAX_WINDOWS):
    """Evenly spaced turn indices across the FULL trajectory (first+last kept);
    window text = last 2 turns + current joined with a blank line."""
    n = len(turns)
    if n == 0:
        return []
    if n <= cap:
        idxs = list(range(n))
    else:
        step = (n - 1) / (cap - 1)
        idxs = sorted({round(i * step) for i in range(cap)})
    out = []
    for i in idxs:
        ctx = turns[max(0, i - 2): i + 1]
        out.append({"turn_idx": i, "text": "\n\n".join(ctx)})
    return out


def exists(p):
    return os.path.isfile(p)


def load_csv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


items = []

# ---------- SGD (30) ----------
sgd_rows = load_csv(os.path.join(BASE, "dataset/sgd/dialogs.csv"))
by_id = {r["dialog_id"]: r for r in sgd_rows}
outcomes = {}
for r in sgd_rows:
    outcomes[r["outcome_calibrated"]] = outcomes.get(r["outcome_calibrated"], 0) + 1
print("SGD outcome_calibrated distribution:", outcomes)

sf_ids = [r["dialog_id"] for r in sgd_rows if r["silent_failure_candidate"] == "True"]
print("SGD silent_failure_candidate rows:", len(sf_ids))
sf_examples = set(json.load(open(os.path.join(BASE, "sgd_silent_failure.json"),
                                  encoding="utf-8"))["examples"])
sf_pick = sorted(set(sf_examples) | set(sf_ids), key=lambda x: (x not in sf_examples, x))
rng.shuffle(sf_pick)
sf_pick = sf_pick[:6]

remaining_targets = {"theta_cont": 12, "theta_aban": 8}
# theta_hand does not exist in SGD (paper: handoff=0) -> redistribute
remaining_targets["theta_cont"] += 30 - 6 - sum(remaining_targets.values())

pools = {}
for r in sgd_rows:
    if r["dialog_id"] in sf_pick:
        continue
    if not exists(os.path.join(BASE, "sgd_test_transcript", r["dialog_id"] + ".txt")):
        continue
    pools.setdefault(r["outcome_calibrated"], []).append(r)

sgd_sel = list(sf_pick)
for oc, k in remaining_targets.items():
    pool = sorted(pools.get(oc, []), key=lambda r: r["dialog_id"])
    # span n_turns diversity: sort by n_turns and take evenly spaced
    pool.sort(key=lambda r: int(r["n_turns"]))
    if len(pool) <= k:
        take = [r["dialog_id"] for r in pool]
    else:
        step = (len(pool) - 1) / (k - 1)
        take = [pool[round(i * step)]["dialog_id"] for i in range(k)]
    sgd_sel.extend(take)

for did in sgd_sel:
    r = by_id[did]
    path = os.path.join(BASE, "sgd_test_transcript", did + ".txt")
    turns = parse_turns(path)
    items.append({
        "corpus": "sgd",
        "id": did,
        "path": path,
        "outcome": r["outcome_calibrated"],
        "n_turns": len(turns),
        "silent_failure_candidate": r["silent_failure_candidate"] == "True" or did in sf_examples,
        "windows": windows_for(turns),
    })

# ---------- MultiWOZ (10) ----------
mw_rows = load_csv(os.path.join(BASE, "dataset/multiwoz/dialogs.csv"))
mw_outcomes = {}
for r in mw_rows:
    mw_outcomes[r["outcome"]] = mw_outcomes.get(r["outcome"], 0) + 1
print("MultiWOZ outcome distribution:", mw_outcomes)

# multiwoz_test files are MUL*.txt; match by id if present in csv, else sample files
mw_dir = os.path.join(BASE, "multiwoz_test")
mw_files = sorted(f[:-4] for f in os.listdir(mw_dir) if f.endswith(".txt"))
mw_by_id = {r["dialog_id"]: r for r in mw_rows if r["dialog_id"] in set(mw_files)}
print("MultiWOZ transcripts on disk:", len(mw_files), "| matched in csv:", len(mw_by_id))

per_class = {}
for did in sorted(mw_by_id):
    per_class.setdefault(mw_by_id[did]["outcome"], []).append(did)
mw_sel = []
kcls = 5
for oc, ids in sorted(per_class.items()):
    rng.shuffle(ids)
    mw_sel.extend(ids[:kcls])
mw_sel = mw_sel[:10]
while len(mw_sel) < 10:  # backfill if fewer classes than expected
    rest = [d for d in mw_files if d not in mw_sel]
    rng.shuffle(rest)
    mw_sel.extend(rest[:10 - len(mw_sel)])

for did in sorted(mw_sel):
    r = mw_by_id.get(did)
    path = os.path.join(mw_dir, did + ".txt")
    turns = parse_turns(path)
    items.append({
        "corpus": "multiwoz",
        "id": did,
        "path": path,
        "outcome": (r["outcome"] if r else "unknown"),
        "n_turns": len(turns),
        "silent_failure_candidate": False,
        "windows": windows_for(turns),
    })

# ---------- tau-bench (10) ----------
tb_rows = load_csv(os.path.join(BASE, "dataset/taubench/episodes.csv"))
tb_outcomes = {}
for r in tb_rows:
    tb_outcomes[r["outcome_convo"]] = tb_outcomes.get(r["outcome_convo"], 0) + 1
print("tau outcome_convo distribution:", tb_outcomes)

sf_tb = [r for r in tb_rows if r["silent_failure_candidate"] == "True"]
print("tau silent_failure_candidate rows:", len(sf_tb))

targets = {"theta_cont": 4, "theta_aban": 6}  # no theta_hand in tau (0 rows); SF overlay 0 rows
picked = []
by_oc = {}
for r in tb_rows:
    by_oc.setdefault(r["outcome_convo"], []).append(r)
for oc, k in targets.items():
    pool = sorted(by_oc.get(oc, []), key=lambda x: (x["domain"], x["episode_id"]))
    # domain diversity: round-robin over domains (id-set dedup)
    domains = sorted({r["domain"] for r in pool})
    by_dom = {d: [r for r in pool if r["domain"] == d] for d in domains}
    rr, seen = [], set()
    i = 0
    while len(rr) < min(k, len(pool)) and i < 50 * len(pool):
        d = domains[i % len(domains)]
        cand = [r for r in by_dom[d] if r["episode_id"] not in seen]
        if cand:
            r = cand[(i // len(domains)) % len(cand)]
            seen.add(r["episode_id"])
            rr.append(r)
        i += 1
    for r in rr:
        fid = r["episode_id"].replace(":", ".") + ".txt"
        if exists(os.path.join(TB, fid)):
            picked.append((r, fid, False))

for r, fid, is_sf in picked:
    path = os.path.join(TB, fid)
    turns = parse_turns(path)
    items.append({
        "corpus": "taubench",
        "id": r["episode_id"],
        "path": path,
        "outcome": r["outcome_convo"],
        "n_turns": len(turns),
        "silent_failure_candidate": is_sf or r["silent_failure_candidate"] == "True",
        "windows": windows_for(turns),
    })

# ---------- write ----------
total_windows = sum(len(it["windows"]) for it in items)
per_corpus = {}
for it in items:
    c = per_corpus.setdefault(it["corpus"], {"items": 0, "windows": 0,
                                             "outcomes": {}, "sf": 0})
    c["items"] += 1
    c["windows"] += len(it["windows"])
    c["outcomes"][it["outcome"]] = c["outcomes"].get(it["outcome"], 0) + 1
    c["sf"] += int(it["silent_failure_candidate"])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=1)

print()
print("WROTE", OUT)
print("total items:", len(items), "| total windows:", total_windows)
for c, v in per_corpus.items():
    print(f"  {c}: items={v['items']} windows={v['windows']} outcomes={v['outcomes']} sf={v['sf']}")
assert all(exists(it["path"]) for it in items), "missing transcript file!"
assert all(isinstance(it["windows"], list) for it in items)
print("all transcript paths exist: OK")
