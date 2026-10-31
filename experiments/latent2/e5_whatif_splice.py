"""E5 — what-if splice: does the failing twin bend back to the pass manifold? (cap 5)

Predeclared (G2): for each mixed twin pair with a semantic divergence point tau* (from
e3_results.json), build the counterfactual C = F[:tau*-1] + P[tau*:] (the failing twin's
prefix, then the passing twin's continuation from the split point). Measure:
  flip:   walker verdict on the spliced conversation == theta_cont (the pass outcome)
  rejoin: fraction of post-splice turns k with d_sem(C_k, P_k) <= p90_k (inside the pass band)
P-E5 PASS: flip >= 60% AND rejoin >= 60% (mean over pairs).
Control (descriptive): reverse splice P[:tau*-1] + F[tau*:].
"""
import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, "C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar")
from trace2grammar.extract import extract_transcript  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
CACHE = os.path.join(HF, "runtime", "latent2")
E3 = json.load(open(os.path.join(HERE, "e3_results.json"), encoding="utf-8"))
ANCHOR = os.path.join(HF, "runtime", "taubench_anchor")

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
                return [x["embedding"] for x in json.load(r)["data"]]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


# band p90 per k (recompute from the cached prefix embeddings, same as e3)
emb = json.load(open(os.path.join(CACHE, "prefix_embed.json"), encoding="utf-8"))
E = {c: np.array(v) for c, v in emb.items()}

def nrm(v):
    return v / (np.linalg.norm(v) + 1e-12)

def dsem(a, b, k):
    va, vb = E.get(f"{a}::{k}"), E.get(f"{b}::{k}")
    if va is None or vb is None:
        return None
    return float(1.0 - np.dot(nrm(va), nrm(vb)))

# band pairs: recompute grouping (same as e2/e3)
rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
    r["id"] = f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}"
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["domain"], r["key"])].append(r)
band_pairs = []
for g in groups.values():
    oks = [r for r in g if r["reward"] >= 1]
    if len(oks) >= 2:
        band_pairs.append((oks[0], oks[1]))

FILES = {"gpt4o": ("gpt-4o-airline.json", "gpt-4o-retail.json"),
         "sonnet": ("sonnet-35-new-airline.json", "sonnet-35-new-retail.json")}
traj = {}
for model, fns in FILES.items():
    for fn in fns:
        domain = "airline" if "airline" in fn else "retail"
        for ep in json.load(open(os.path.join(ANCHOR, fn), encoding="utf-8")):
            traj[f"{model}:{domain}:{ep['task_id']}:{ep['trial']}"] = ep["traj"]

def user_turn_start(t, k):
    """Message index where the k-th user turn starts."""
    seen = 0
    for i, m in enumerate(t):
        if m.get("role") == "user":
            seen += 1
            if seen == k:
                return i
    return None

band_by_k = defaultdict(list)
for a, b in band_pairs:
    ta, tb = traj[a["id"]], traj[b["id"]]
    ka = sum(1 for m in ta if m.get("role") == "user")
    kb = sum(1 for m in tb if m.get("role") == "user")
    for k in range(1, min(ka, kb) + 1):
        d = dsem(a["id"], b["id"], k)
        if d is not None:
            band_by_k[k].append(d)
p90 = {k: float(np.percentile(v, 90)) for k, v in band_by_k.items() if len(v) >= 5}

# ------------------------------------------------------------------ splice pairs with tau*
pairs = [(r["pair"][0], r["pair"][1], r["tau_sem"]) for r in E3["pairs"] if r["tau_sem"]]
print(f"pairs with tau*: {len(pairs)}")

# embed C's prefixes (cache)
cemb = json.load(open(os.path.join(CACHE, "splice_embed.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(CACHE, "splice_embed.json")) else {}

def build_C(tid_f, tid_p, tau):
    tf, tp = traj[tid_f], traj[tid_p]
    s_f = user_turn_start(tf, tau - 1)          # F keeps turns 1..tau-1 inclusive
    s_p = user_turn_start(tp, tau)              # P continues from turn tau
    if s_f is None or s_p is None:
        return None
    C = tf[:s_f + 1] + tp[s_p:]
    return C

def prefix_texts_C(cid, C):
    out, txt, seen_user = {}, [], 0
    for m in C:
        role = m.get("role")
        c = str(m.get("content") or "")[:400]
        if role == "assistant" and m.get("tool_calls"):
            txt.append("assistant: [tool calls]")
        else:
            txt.append(f"{role}: {c}")
        if role == "user":
            seen_user += 1
            out[seen_user] = "\n".join(txt)
    return out

need = {}
for fid, pid, tau in pairs:
    C = build_C(fid, pid, tau)
    if C is None:
        continue
    kc = sum(1 for m in C if m.get("role") == "user")
    pt = prefix_texts_C(f"{fid}|{pid}", C)
    for k in range(tau, kc + 1):
        need[f"{fid}|{pid}::{k}"] = pt.get(k, "")
todo = [c for c, t in need.items() if c not in cemb and t.strip()]
print(f"splice embeds to fetch: {len(todo)}")
B = 50
items = sorted(todo, key=lambda x: int(x.rsplit("::", 1)[1]))
for s in range(0, len(items), B):
    chunk = items[s:s + B]
    t0 = time.time()
    vecs = embed_batch([need[c] for c in chunk])
    for c, v in zip(chunk, vecs):
        cemb[c] = v
    json.dump(cemb, open(os.path.join(CACHE, "splice_embed.json"), "w", encoding="utf-8"))
    print(f"  batch {s//B + 1} ({time.time()-t0:.1f}s)")
json.dump(cemb, open(os.path.join(CACHE, "splice_embed.json"), "w", encoding="utf-8"))
EC = {c: np.array(v) for c, v in cemb.items()}

# ------------------------------------------------------------------ metrics
n_flip = n_rejoin = n_done = 0
detail = []
for fid, pid, tau in pairs:
    C = build_C(fid, pid, tau)
    if C is None:
        continue
    n_done += 1
    # walker on the spliced conversation (transcript rendering)
    lines = []
    for m in C:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                lines.append(f"tool({tc.get('name','tool')}): call")
        elif role == "tool":
            lines.append(f"tool({m.get('name','tool')}): ok {str(m.get('content'))[:60]}")
        else:
            lines.append(f"{role}: {str(m.get('content'))[:300]}")
    try:
        res = extract_transcript("\n".join(lines))
        flipped = res["outcome"] == "theta_cont"
    except Exception:
        flipped = False
    # rejoin: post-splice turns inside the pass band
    inside = tot = 0
    kc = sum(1 for m in C if m.get("role") == "user")
    for k in range(tau, kc + 1):
        if k not in p90:
            continue
        va, vb = EC.get(f"{fid}|{pid}::{k}"), E.get(f"{pid}::{k}")
        if va is None or vb is None:
            continue
        d = float(1.0 - np.dot(nrm(va), nrm(vb)))
        tot += 1
        inside += d <= p90[k]
    rejoin = inside / tot if tot else 0.0
    n_flip += flipped
    n_rejoin += rejoin >= 0.6
    detail.append({"pair": (fid, pid), "tau": tau, "flip": flipped, "rejoin": round(rejoin, 2)})

p_flip = 100.0 * n_flip / n_done if n_done else 0.0
p_rejoin = 100.0 * n_rejoin / n_done if n_done else 0.0
print(f"\nflip (walker contained tras empalme): {n_flip}/{n_done} = {p_flip:.1f}%")
print(f"rejoin (post-splice dentro de banda pass, >=60% turnos): {n_rejoin}/{n_done} = {p_rejoin:.1f}%")
verdict = "PASS" if p_flip >= 60 and p_rejoin >= 60 else "FAIL"
print("VERDICT: P-E5", verdict)

json.dump({"protocol": {"n_pairs": n_done, "predicate": "flip>=60% AND rejoin>=60%",
                        "band": "p90 pass-pass per k", "split": "C = F[:tau-1] + P[tau:]"},
           "P_E5": verdict, "flip_pct": round(p_flip, 1), "rejoin_pct": round(p_rejoin, 1),
           "detail": detail},
          open(os.path.join(HERE, "e5_results.json"), "w", encoding="utf-8"), indent=1)
