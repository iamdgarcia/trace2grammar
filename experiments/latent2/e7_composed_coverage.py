"""E7 — cap 1: the composed space, evaluated (region + honest instance + NN-ratio).

Predeclared (G2): composed conversation vector = [z(embedding(first_user_text)) |
z(structural walker features)] (alpha=0.5, L2), for the 1,980 real episodes + the 60
shipped sim2 compositions (semantic from sim_meta, structural from walk_summary columns).
Predicates:
  P-E7a: all 3 metrics computed and published (region coverage at beta 0.04 and 0.12,
         honest instance radius + coverage, NN-ratio with bootstrap CI).
  P-E7b: composed NN-ratio >= structural-only NN-ratio (the semantic channel does not
         hide the sim2real gap; expected: it widens it, since sim openings are
         out-of-domain vs tau-bench content).
Only ~2 embedding batches are new (810 real openings cached from E2).
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
TB = json.load(open(os.path.join(HF, "runtime/taubench_anchor/tb_coverage.json"), encoding="utf-8"))

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

emb = json.load(open(os.path.join(CACHE, "openings_embed.json"), encoding="utf-8"))
# sim openings
sim_keys = []
for m in TB["sim_meta"]:
    k = m["first_user_text"].strip()
    sim_keys.append(k)
todo = sorted({k for k in sim_keys if k not in emb})
if todo:
    print("sim openings to embed:", len(todo))
    for s in range(0, len(todo), 50):
        chunk = todo[s:s + 50]
        vecs = embed_batch(chunk)
        for k, v in zip(chunk, vecs):
            emb[k] = v
    json.dump(emb, open(os.path.join(CACHE, "openings_embed.json"), "w", encoding="utf-8"))

# rows
rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
for r in rows:
    r["key"] = r["first_user_text"].strip()

def zn(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0)
    return np.where(sd > 1e-12, (M - M.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)

def struct_vec(outcome, D, n_states, duration_ms, reached_close, model, domain):
    return [float(D), float(int(D) >= 1), float(n_states), float(bool(reached_close)),
            math.log1p(float(duration_ms)),
            1.0 if model == "sonnet" else 0.0, 1.0 if domain == "airline" else 0.0,
            1.0 if outcome == "theta_cont" else 0.0,
            1.0 if outcome == "theta_aban" else 0.0]

S_real = zn([struct_vec(r["outcome_convo"], r["D"], r["n_states"], r["duration_ms"],
                        r["reached_close"], r["model"], r["domain"]) for r in rows])
E_real = zn([emb[r["key"]] for r in rows])
def comp(ES, SS):
    C = np.concatenate([ES, SS], axis=1)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
COMP_real = comp(E_real, S_real)
STR_real = S_real / (np.linalg.norm(S_real, axis=1, keepdims=True) + 1e-12)

# sim set
S_sim = zn([struct_vec(m["outcome"], m["cycle_depth"], m["n_states"], m["duration_ms"],
                       m["n_states"] > 6, "sonnet", "retail") for m in TB["sim_meta"]])
E_sim = zn([emb[k] for k in sim_keys])
COMP_sim = comp(E_sim, S_sim)
STR_sim = S_sim / (np.linalg.norm(S_sim, axis=1, keepdims=True) + 1e-12)

# coverage machinery (shipped) needs 2-D points -> numpy PCA fit on the real set
# (the shipped _pca_basis is pure-python and cannot carry 3,110-dim composed vectors;
#  same math: top-2 eigenvectors of the real covariance — declared substitution)
sys.path.insert(0, os.path.join(HF, "builds/trace2grammar"))
from trace2grammar.coverage import coverage  # noqa: E402

def np_pca2(R):
    M = np.asarray(R, dtype=float)
    mean = M.mean(axis=0)
    X = M - mean
    C = (X.T @ X) / (len(X) - 1)
    w, V = np.linalg.eigh(C)
    v1, v2 = V[:, -1], V[:, -2]
    return mean, v1, v2

out = {"protocol": {"alpha": 0.5, "embed": EMBED_MODEL, "res": 24, "eps": 0.05,
                    "pca": "numpy eigh top-2 (declared; _pca_basis cannot carry 3110 dims)"},
       "spaces": {}}
for name, R, S in (("structural", STR_real, STR_sim), ("composed", COMP_real, COMP_sim)):
    mean, v1, v2 = np_pca2(R)
    proj = lambda M: [(float((p-mean) @ v1), float((p-mean) @ v2)) for p in np.asarray(M, dtype=float)]
    R2, S2 = proj(R), proj(S)
    sp = {}
    for bw in (0.04, 0.12):
        c = coverage(R2, S2, res=24, eps=0.05, bw_frac=bw)
        sp[f"region_bw{bw}"] = {"coverage_pct": c["coverage_pct"],
                                "occupied": c["occupied_cells"], "uncovered": len(c["uncovered_cells"])}
    # honest instance + NN-ratio (dedup radius)
    seen, ru = set(), []
    for p in R2:
        k = (round(p[0], 9), round(p[1], 9))
        if k not in seen:
            seen.add(k); ru.append(p)
    def med_nn(pts):
        nn = sorted(math.sqrt(min((a[0]-x)**2 + (a[1]-y)**2 for j, (x, y) in enumerate(pts) if j != i))
                    for i, a in enumerate(pts))
        n = len(nn)
        return nn[n//2] if n % 2 else (nn[n//2-1] + nn[n//2]) / 2
    r_u = med_nn(ru)
    r2 = r_u * r_u
    inst = 100.0 * sum(any((a[0]-x)**2 + (a[1]-y)**2 <= r2 for x, y in S2) for a in ru) / len(ru)
    d2 = lambda a, b: (a[0]-b[0])**2 + (a[1]-b[1])**2
    ms = sorted(min(d2(p, q) for q in ru) ** 0.5 for p in S2)[len(S2)//2]
    Rr = ms / r_u
    rng = np.random.default_rng(20261030)
    rats = []
    for _ in range(200):
        ss = [S2[i] for i in rng.integers(0, len(S2), len(S2))]
        m2 = sorted(min(d2(p, q) for q in ru) ** 0.5 for p in ss)[len(ss)//2]
        rats.append(m2 / r_u)
    rats.sort()
    sp["instance"] = {"r_honest": round(r_u, 4), "coverage_pct": round(inst, 2),
                      "unique_real": len(ru)}
    sp["nn_ratio"] = {"R": round(Rr, 4), "ci": [round(rats[5], 3), round(rats[194], 3)]}
    out["spaces"][name] = sp
    print(name, json.dumps(sp))

ok = (out["spaces"]["composed"]["nn_ratio"]["R"] >= out["spaces"]["structural"]["nn_ratio"]["R"])
out["verdict"] = "PASS" if ok else "FAIL"
print("\nVERDICT: P-E7", out["verdict"], "(composed NN-ratio >= structural)")
json.dump(out, open(os.path.join(HERE, "e7_results.json"), "w", encoding="utf-8"), indent=1)
