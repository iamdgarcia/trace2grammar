"""E10 — the objective's literal what-if: pending -> resolved at the schema level (cap 5).

Predeclared (G2): for the 354 cached ELDA schema windows: build the swapped schema
(pending_request joins resolved_requests, pending = none), re-serialize, embed both.
Metric (ELDA T3 discipline):
  swap-delta     = d_sem(original_schema, swapped_schema) per window
  content-change = mean cross-window same-corpus distance (the scale of a real content change)
P-E10 PASS: mean(swap-delta) / content-change >= 0.8 (the space responds to the
counterfactual like it responds to content).
708 embeddings via laya (15 batches).
"""
import json
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

BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("E2_EMBED_MODEL", "gemini-embedding-001")
if not (BASE_URL and API_KEY):
    sys.exit("SIN ENDPOINT")

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


wins = [json.loads(l) for l in open(os.path.join(HF, "builds/trace2grammar/experiments/elda/annotations/descriptions.jsonl"), encoding="utf-8")]

def serialize(d):
    return ("Phase: {phase}. Resolved before: {res}. Pending now: {pen}. Open loops: {loops}. "
            "User stance: {st}.").format(
        phase=d.get("phase", "?"),
        res=", ".join(d.get("resolved_requests", [])) or "none",
        pen=d.get("pending_request", "none"),
        loops="; ".join(d.get("open_loops", [])) or "none",
        st=d.get("user_stance", "?"))

def swapped(d):
    s = json.loads(json.dumps(d))
    pen = s.get("pending_request", "none")
    if pen and pen != "none":
        s["resolved_requests"] = list(s.get("resolved_requests", [])) + [pen]
        s["pending_request"] = "none"
    return s

cache = json.load(open(os.path.join(CACHE, "schema_embed.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(CACHE, "schema_embed.json")) else {}
orig = {w["pid"]: serialize(w["desc"]) for w in wins}
swap = {w["pid"]: serialize(swapped(w["desc"])) for w in wins}
todo = []
for pid in orig:
    if f"{pid}|o" not in cache: todo.append((f"{pid}|o", orig[pid]))
    if f"{pid}|s" not in cache: todo.append((f"{pid}|s", swap[pid]))
print(f"schema embeds to fetch: {len(todo)} / {2*len(wins)}")
todo.sort(key=lambda x: x[0])
for s in range(0, len(todo), 50):
    chunk = todo[s:s+50]
    t0 = time.time()
    vecs = embed_batch([t for _, t in chunk])
    for (k, _), v in zip(chunk, vecs):
        cache[k] = v
    json.dump(cache, open(os.path.join(CACHE, "schema_embed.json"), "w", encoding="utf-8"))
    print(f"  batch {s//50 + 1} ({time.time()-t0:.1f}s)")
json.dump(cache, open(os.path.join(CACHE, "schema_embed.json"), "w", encoding="utf-8"))
E = {k: np.array(v) for k, v in cache.items()}
def nrm(v):
    return v / (np.linalg.norm(v) + 1e-12)

swap_deltas, corpus_of = [], {}
for w in wins:
    cid = w["pid"].split(":")[0]
    corpus_of[w["pid"]] = cid
    va, vb = E[f"{w['pid']}|o"], E[f"{w['pid']}|s"]
    swap_deltas.append(float(1 - np.dot(nrm(va), nrm(vb))))

by_corpus = defaultdict(list)
for pid in orig:
    by_corpus[corpus_of[pid]].append(pid)
cross = []
for cid, pids in by_corpus.items():
    for i, a in enumerate(pids):
        for b in pids[i+1:]:
            cross.append(float(1 - np.dot(nrm(E[f"{a}|o"]), nrm(E[f"{b}|o"]))))

ms, mc = float(np.mean(swap_deltas)), float(np.mean(cross))
ratio = ms / mc if mc else 0.0
print(f"\nmean swap-delta {ms:.4f} | mean content-change {mc:.4f} | ratio {ratio:.3f}")
verdict = "PASS" if ratio >= 0.8 else "FAIL"
print("VERDICT: P-E10", verdict)
json.dump({"protocol": {"n_windows": len(wins), "predicate": "ratio >= 0.8"},
           "mean_swap_delta": round(ms, 4), "mean_content_change": round(mc, 4),
           "ratio": round(ratio, 3), "verdict": verdict},
          open(os.path.join(HERE, "e10_results.json"), "w", encoding="utf-8"), indent=1)
