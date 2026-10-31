"""E2 — composed-space property: silent-failure twins separate only through structure.

The paper's honest-cost paragraph, turned into a measured property (t2g-latent-v2, cap 2+4):

  In a SEMANTIC-ONLY space, two conversations with the same opening are the same point —
  including a PASS twin and a FAIL twin (the silent-failure family). In the composed space
  (semantic + structural blocks), the twins must separate.

Predeclared (G2, before running):
  - Groups: tau-bench episodes with BYTE-IDENTICAL first_user_text and >=2 trials mixing
    outcomes (pass/fail). Positive pairs = (fail twin, pass twin) same group;
    negative pairs = (pass, pass) same group.
  - Metric: AUROC of pairwise distance classifying mixed-vs-same-outcome twin pairs,
    per space: semantic-only (first-user-turn embedding), structural-only (walker CSV
    features), composed (alpha=0.5 concat of z-normed blocks).
  - PASS: composed AUROC >= 0.60 AND semantic AUROC <= 0.55 AND structural <= composed.
  - Bootstrap 95% CI (2000 resamples over groups) on composed AUROC.

Model calls: ONLY via the laya gateway (env). Embeddings cached to runtime/latent2/.
Stdlib + numpy only.
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
            with urllib.request.urlopen(req, timeout=120,
                                        context=ssl.create_default_context()) as r:
                d = json.load(r)
            return [x["embedding"] for x in d["data"]]
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


# ------------------------------------------------------------------ load + groups
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[(r["model"], r["domain"], r["key"])].append((i, r))

mixed, pass_only = [], []
for k, g in groups.items():
    fails = [(i, r) for i, r in g if r["reward"] < 1]
    oks = [(i, r) for i, r in g if r["reward"] >= 1]
    if fails and oks:
        mixed.append((fails, oks))
    elif len(oks) >= 2:
        pass_only.append(oks)
print(f"groups: {len(groups)} | mixed-outcome: {len(mixed)} | pass-only(>=2): {len(pass_only)}")

# ------------------------------------------------------------------ embeddings (cached)
cache_file = os.path.join(CACHE, "openings_embed.json")
emb = {}
if os.path.exists(cache_file):
    emb = json.load(open(cache_file, encoding="utf-8"))
todo = sorted({r["key"] for r in rows if r["key"] not in emb})
print(f"embeddings: {len(emb)} cached, {len(todo)} to fetch")
B = 50
for i in range(0, len(todo), B):
    chunk = todo[i:i + B]
    t0 = time.time()
    vecs = embed_batch(chunk)
    for k, v in zip(chunk, vecs):
        emb[k] = v
    json.dump(emb, open(cache_file, "w", encoding="utf-8"))
    print(f"  batch {i//B + 1}/{(len(todo)+B-1)//B}: {len(chunk)} keys ({time.time()-t0:.1f}s)")
json.dump(emb, open(cache_file, "w", encoding="utf-8"))

# ------------------------------------------------------------------ spaces
def zn(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0)
    out = np.where(sd > 1e-12, (M - M.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)
    return out

sem = zn([emb[r["key"]] for r in rows])
struct_cols = lambda r: [float(r["D"]), float(int(r["D"]) >= 1), float(r["n_states"]),
                         float(r["error_count"]), float(r["reached_close"] == "True"),
                         math.log1p(float(r["duration_ms"])),
                         1.0 if r["model"] == "sonnet" else 0.0,
                         1.0 if r["domain"] == "airline" else 0.0]
struct = zn([struct_cols(r) for r in rows])
SEM = sem / (np.linalg.norm(sem, axis=1, keepdims=True) + 1e-12)
STR = struct / (np.linalg.norm(struct, axis=1, keepdims=True) + 1e-12)
COMP = np.concatenate([SEM, STR], axis=1)
COMP = COMP / (np.linalg.norm(COMP, axis=1, keepdims=True) + 1e-12)

pos = [(i1, i2) for fails, oks in mixed for i1, r1 in fails for i2, r2 in oks]
neg = [(i1, i2) for g in pass_only for (i1, r1), (i2, r2) in zip(g, g[1:])]
print(f"pairs: mixed {len(pos)} | pass-pass {len(neg)}")

rng = np.random.default_rng(SEED)

def dist(M, a, b):
    return float(1.0 - np.dot(M[a], M[b]))

def auroc(M, posp, negp, boot=0):
    dp = np.array([dist(M, a, b) for a, b in posp])
    dn = np.array([dist(M, a, b) for a, b in negp])
    allv = np.concatenate([dp, dn])
    r = allv.argsort().argsort() + 1
    u = r[:len(dp)].sum() - len(dp) * (len(dp) + 1) / 2
    a = 1 - u / (len(dp) * len(dn))
    if boot:
        bs = []
        for _ in range(boot):
            ps = [posp[i] for i in rng.integers(0, len(posp), len(posp))]
            ns = [negp[i] for i in rng.integers(0, len(negp), len(negp))]
            dpb = np.array([dist(M, x, y) for x, y in ps])
            dnb = np.array([dist(M, x, y) for x, y in ns])
            av = np.concatenate([dpb, dnb])
            rr = av.argsort().argsort() + 1
            ub = rr[:len(dpb)].sum() - len(dpb) * (len(dpb) + 1) / 2
            bs.append(1 - ub / (len(dpb) * len(dnb)))
        return a, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return a

res = {}
for name, M in (("semantic", SEM), ("structural", STR), ("composed", COMP)):
    a, ci = auroc(M, pos, neg, boot=2000)
    res[name] = {"auroc": round(float(a), 4), "ci": [round(ci[0], 4), round(ci[1], 4)]}
    print(f"{name}: AUROC {a:.4f} CI [{ci[0]:.4f}, {ci[1]:.4f}]")

verdict = ("PASS" if res["composed"]["auroc"] >= 0.60 and res["semantic"]["auroc"] <= 0.55
           and res["structural"]["auroc"] <= res["composed"]["auroc"] else "FAIL")
print("\nVERDICT:", verdict)

json.dump({"protocol": {"n_groups_mixed": len(mixed), "n_groups_passonly": len(pass_only),
                        "n_pairs": {"mixed": len(pos), "passpass": len(neg)},
                        "embed_model": EMBED_MODEL, "alpha": 0.5, "seed": SEED,
                        "predicate": "composed>=0.60 AND semantic<=0.55 AND structural<=composed"},
           "results": res, "verdict": verdict},
          open(os.path.join(HERE, "e2_results.json"), "w", encoding="utf-8"), indent=1)
