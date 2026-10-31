"""E3 — separation point between twin conversations (t2g-latent-v2, cap 4+3).

Predeclared (G2):
  Pairs: from the 102 mixed-outcome groups with byte-identical first_user_text (e2's
  grouping), one FAIL episode + one PASS episode each. Band: pooled pass-pass same-task
  prefix distances (semantic, per turn k), threshold = p90 at each k.
  tau* = first turn k where d_sem(prefixF_k, prefixP_k) > p90_k  (semantic divergence).
  Structural confirmation: first turn in F where retries_so_far>0 or D_so_far>0 (from the
  shipped h3 prefix rows); confirmed if it falls within tau* +- 2.
  P-A PASS: tau* defined in >= 60% of pairs (before the failing episode's last turn).
  P-B PASS: structural confirmation within +-2 in >= 50% of pairs with tau*.

Embeddings: prefix texts via laya (batched 50, cached in runtime/latent2/prefix_embed.json).
"""
import csv
import json
import math
import os
import ssl
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
CACHE = os.path.join(HF, "runtime", "latent2")
os.makedirs(CACHE, exist_ok=True)
CSV = os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv")
ANCHOR = os.path.join(HF, "runtime", "taubench_anchor")
H3 = os.path.join(HF, "runtime/h3_early_warning/h3_rows.json")
SEED = 20261030

BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("E2_EMBED_MODEL", "gemini-embedding-001")
if not (BASE_URL and API_KEY):
    sys.exit("SIN ENDPOINT: exporta OPENAI_BASE_URL/OPENAI_API_KEY (laya)")


def embed_batch(texts):
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180,
                                        context=ssl.create_default_context()) as r:
                d = json.load(r)
            return [x["embedding"] for x in d["data"]]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


# ------------------------------------------------------------------ groups (same logic as e2)
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
    r["id"] = f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}"
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["domain"], r["key"])].append(r)

pairsF, bands = [], []
for k, g in groups.items():
    fails = [r for r in g if r["reward"] < 1]
    oks = [r for r in g if r["reward"] >= 1]
    if fails and oks:
        pairsF.append((fails[0], oks[0]))
    if len(oks) >= 2:
        bands.append((oks[0], oks[1]))
print(f"mixed pairs: {len(pairsF)} | pass-pass band pairs: {len(bands)}")

# ------------------------------------------------------------------ raw prefixes
FILES = {"gpt4o": ("gpt-4o-airline.json", "gpt-4o-retail.json"),
         "sonnet": ("sonnet-35-new-airline.json", "sonnet-35-new-retail.json")}
traj = {}
for model, fns in FILES.items():
    for fn in fns:
        domain = "airline" if "airline" in fn else "retail"
        for ep in json.load(open(os.path.join(ANCHOR, fn), encoding="utf-8")):
            traj[f"{model}:{domain}:{ep['task_id']}:{ep['trial']}"] = ep["traj"]

def prefix_text(t, k):
    """Messages up to and including the k-th user turn (k is 1-based)."""
    out, seen_user = [], 0
    for m in t:
        role = m.get("role")
        c = str(m.get("content") or "")[:400]
        if role == "assistant" and m.get("tool_calls"):
            out.append("assistant: [tool calls]")
            continue
        out.append(f"{role}: {c}")
        if role == "user":
            seen_user += 1
            if seen_user >= k:
                break
    return "\n".join(out)

def n_user_turns(t):
    return sum(1 for m in t if m.get("role") == "user")

# ------------------------------------------------------------------ embeddings (cached)
cache_file = os.path.join(CACHE, "prefix_embed.json")
emb = json.load(open(cache_file, encoding="utf-8")) if os.path.exists(cache_file) else {}

need = set()
for a, b in pairsF + bands:
    ka = min(n_user_turns(traj[a["id"]]), n_user_turns(traj[b["id"]]))
    for k in range(1, ka + 1):
        need.add((a["id"], k)); need.add((b["id"], k))
todo = sorted(f"{i}::{k}" for i, k in need if f"{i}::{k}" not in emb)
print(f"prefix embeds: {len(emb)} cached, {len(todo)} to fetch")
B = 50
texts = {f"{i}::{k}": prefix_text(traj[i], k) for i, k in need}
for s in range(0, len(todo), B):
    chunk = todo[s:s + B]
    t0 = time.time()
    vecs = embed_batch([texts[c] for c in chunk])
    for c, v in zip(chunk, vecs):
        emb[c] = v
    json.dump(emb, open(cache_file, "w", encoding="utf-8"))
    print(f"  batch {s//B + 1}/{(len(todo)+B-1)//B} ({time.time()-t0:.1f}s)")
json.dump(emb, open(cache_file, "w", encoding="utf-8"))
E = {c: np.array(v) for c, v in emb.items()}
def nrm(v):
    return v / (np.linalg.norm(v) + 1e-12)
def dsem(a, b, k):
    va, vb = E.get(f"{a}::{k}"), E.get(f"{b}::{k}")
    if va is None or vb is None:
        return None
    va, vb = nrm(va), nrm(vb)
    return float(1.0 - np.dot(va, vb))

# ------------------------------------------------------------------ band p90 per k
band_by_k = defaultdict(list)
for a, b in bands:
    km = min(n_user_turns(traj[a["id"]]), n_user_turns(traj[b["id"]]))
    for k in range(1, km + 1):
        d = dsem(a["id"], b["id"], k)
        if d is not None:
            band_by_k[k].append(d)
p90 = {k: float(np.percentile(v, 90)) for k, v in band_by_k.items() if len(v) >= 5}
print(f"band: {len(band_by_k)} turns, p90 computed for {len(p90)}")

# ------------------------------------------------------------------ structural (h3 rows)
h3 = json.load(open(H3, encoding="utf-8"))
err_turn = {}
for r in h3:
    if r["retries_so_far"] > 0 or r["D_so_far"] > 0:
        err_turn.setdefault(r["id"], r["k"])  # first k with structural alarm

# ------------------------------------------------------------------ divergence points
res = []
for f, p in pairsF:
    kf = n_user_turns(traj[f["id"]])
    kp = n_user_turns(traj[p["id"]])
    km = min(kf, kp)
    tau = None
    for k in range(1, km + 1):
        if k not in p90:
            continue
        d = dsem(f["id"], p["id"], k)
        if d is not None and d > p90[k]:
            tau = k
            break
    et = err_turn.get(f["id"])
    res.append({"pair": (f["id"], p["id"]), "task": f["task_id"], "domain": f["domain"],
                "model": f["model"], "tau_sem": tau, "err_turn": et,
                "confirmed": bool(tau and et is not None and abs(et - tau) <= 2),
                "shared_turns": km})

with_tau = [r for r in res if r["tau_sem"] is not None]
p_a = 100.0 * len(with_tau) / len(res)
p_b = (100.0 * sum(1 for r in with_tau if r["confirmed"]) / len(with_tau)) if with_tau else 0.0
taus = [r["tau_sem"] for r in with_tau]
print(f"\nP-A tau* defined: {len(with_tau)}/{len(res)} = {p_a:.1f}%")
print(f"P-B structural confirm (+-2): {sum(1 for r in with_tau if r['confirmed'])}/{len(with_tau)} = {p_b:.1f}%")
print("tau* distribution:", {k: taus.count(k) for k in sorted(set(taus))})

va = "PASS" if p_a >= 60 else "FAIL"
vb = "PASS" if p_b >= 50 else "FAIL"
print("\nVERDICT: P-A", va, "| P-B", vb)

json.dump({"protocol": {"n_pairs": len(res), "n_band_pairs": len(bands), "band": "p90 per k",
                        "predicate": "P-A>=60% defined, P-B>=50% confirmed(+/-2)"},
           "P_A_pct": round(p_a, 1), "P_A": va,
           "P_B_pct": round(p_b, 1), "P_B": vb,
           "tau_distribution": {str(k): taus.count(k) for k in sorted(set(taus))},
           "pairs": res},
          open(os.path.join(HERE, "e3_results.json"), "w", encoding="utf-8"), indent=1)
