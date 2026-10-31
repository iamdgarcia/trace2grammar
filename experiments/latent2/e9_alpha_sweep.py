"""E9 — close the composed-alpha via: alpha sweep on the twin task (kill_criterion).

The thread's kill_criterion names the alternative (raw window embeddings) — but E2's
semantic channel WAS raw text (first_user_text embeddings, no schema). The remaining
composition freedom is the mixing weight. Sweep alpha in 0..1 (11 values): composed =
concat(alpha*z(sem), (1-alpha)*z(struct)). Predeclared kill: if max_auroc(alpha) <
structural_auroc + 0.05, the composed-alpha via CLOSES with DROP honest (the dialogue
channel does not carry the verdict at any mixing weight). All from cache; no model calls.
"""
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"

emb = json.load(open(os.path.join(HF, "runtime/latent2/openings_embed.json"), encoding="utf-8"))
rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
    r["id"] = f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}"
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["domain"], r["key"])].append(r)
pairsF, pass_only = [], []
for g in groups.values():
    fails = [r for r in g if r["reward"] < 1]
    oks = [r for r in g if r["reward"] >= 1]
    if fails and oks:
        pairsF.append((fails[0], oks[0]))
    if len(oks) >= 2:
        pass_only.append(oks)

def zn(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0)
    return np.where(sd > 1e-12, (M - M.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)

sem = zn([emb[r["key"]] for r in rows])
sc = lambda r: [float(r["D"]), float(int(r["D"]) >= 1), float(r["n_states"]),
                float(r["error_count"]), float(r["reached_close"] == "True"),
                math.log1p(float(r["duration_ms"])),
                1.0 if r["model"] == "sonnet" else 0.0,
                1.0 if r["domain"] == "airline" else 0.0]
struct = zn([sc(r) for r in rows])
# build pairs by index
idx_of = {r["id"]: i for i, r in enumerate(rows)}
pos, neg = [], []
for g in groups.values():
    fails = [r for r in g if r["reward"] < 1]
    oks = [r for r in g if r["reward"] >= 1]
    if fails and oks:
        pos.append((idx_of[fails[0]["id"]], idx_of[oks[0]["id"]]))
    if len(oks) >= 2:
        neg.append((idx_of[oks[0]["id"]], idx_of[oks[1]["id"]]))
print(f"pairs: mixed {len(pos)} | pass-pass {len(neg)}")

def twinauroc(M):
    d = lambda a, b: float(1.0 - np.dot(M[a], M[b]))
    dp = np.array([d(a, b) for a, b in pos]); dn = np.array([d(a, b) for a, b in neg])
    allv = np.concatenate([dp, dn]); r = allv.argsort().argsort() + 1
    u = r[:len(dp)].sum() - len(dp) * (len(dp) + 1) / 2
    return 1 - u / (len(dp) * len(dn))

semn = sem / (np.linalg.norm(sem, axis=1, keepdims=True) + 1e-12)
strn = struct / (np.linalg.norm(struct, axis=1, keepdims=True) + 1e-12)
sweep = []
for i in range(11):
    a = i / 10
    C = np.concatenate([a * semn, (1 - a) * strn], axis=1)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sweep.append({"alpha": a, "auroc": round(twinauroc(C), 4)})
    print(f"alpha={a:.1f}: twin AUROC {sweep[-1]['auroc']}")
best = max(sweep, key=lambda s: s["auroc"])
struct_only = twinauroc(strn)
print(f"best alpha {best['alpha']}: {best['auroc']} | structural-only {struct_only:.4f}")
kill = best["auroc"] < struct_only + 0.05
verdict = "DROP" if kill else "KEEP"
print("KILL_CRITERION:", "TRIGGERED -> via closes" if kill else "not triggered")
print("VERDICT:", verdict)
json.dump({"sweep": sweep, "structural_only": round(float(struct_only), 4),
           "best": best, "kill_triggered": bool(kill), "verdict": verdict},
          open(os.path.join(HERE, "e9_results.json"), "w", encoding="utf-8"), indent=1)
