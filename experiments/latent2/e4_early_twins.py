"""E4 — early failure signal inside twin pairs (t2g-latent-v2, cap 3).

Predeclared (G2):
  Refit the shipped h3 risk model (GBM on the 5 prefix features, dev = task_id even) on
  h3_rows.json, score every prefix row. For each mixed twin pair (F fail, P pass):
    - terminal signal: risk(F, last k) > risk(P, same k)
    - early signal: exists k < last_shared_k with risk(F,k) - risk(P,k) >= 0.15
  P-E4 PASS: early signal in >= 60% of pairs.
No model calls (features cached in h3_rows.json; model fit is local sklearn).
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
CSV = os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv")
H3 = os.path.join(HF, "runtime/h3_early_warning/h3_rows.json")
SEED = 20261030
FEATS = ["D_so_far", "retries_so_far", "n_states_so_far", "reached_close_so_far", "terminal_so_far"]

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
    r["id"] = f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}"
    groups[(r["model"], r["domain"], r["key"])].append(r)

pairs = []
for g in groups.values():
    fails = [r for r in g if r["reward"] < 1]
    oks = [r for r in g if r["reward"] >= 1]
    if fails and oks:
        pairs.append((fails[0], oks[0]))
print("mixed pairs:", len(pairs))

h3 = json.load(open(H3, encoding="utf-8"))
by_ep = defaultdict(dict)
for r in h3:
    by_ep[r["id"]][r["k"]] = r

dev = [r for r in h3 if r["task_id"] % 2 == 0]
test = [r for r in h3 if r["task_id"] % 2 == 1]
Xtr = np.array([[r[f] for f in FEATS] for r in dev])
ytr = np.array([r["y_fail"] for r in dev])
Xte = np.array([[r[f] for f in FEATS] for r in test])
yte = np.array([r["y_fail"] for r in test])
gbm = HistGradientBoostingClassifier(max_iter=300, random_state=SEED).fit(Xtr, ytr)
auc_refit = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
print(f"h3 GBM refit (pooled k): test AUROC {auc_refit:.4f} "
      f"(shipped per-k protocol 0.6477; pooled variant declared in results)")
assert auc_refit > 0.55, "refit not above chance"

def risk(ep_id, k):
    r = by_ep[ep_id].get(k)
    if r is None:
        return None
    return float(gbm.predict_proba([[r[f] for f in FEATS]])[0, 1])

n_term = n_early = n_pair = 0
gaps = []
for f, p in pairs:
    ks = sorted(set(by_ep[f["id"]]) & set(by_ep[p["id"]]))
    if not ks:
        continue
    n_pair += 1
    rf_last, rp_last = risk(f["id"], ks[-1]), risk(p["id"], ks[-1])
    if rf_last is not None and rp_last is not None and rf_last > rp_last:
        n_term += 1
    early = any((risk(f["id"], k) is not None and risk(p["id"], k) is not None
                 and risk(f["id"], k) - risk(p["id"], k) >= 0.15) for k in ks[:-1])
    n_early += early
    gaps.append((f["id"], p["id"], ks[-1],
                 round(rf_last - rp_last, 3) if rf_last is not None else None, early))

p_early = 100.0 * n_early / n_pair if n_pair else 0.0
p_term = 100.0 * n_term / n_pair if n_pair else 0.0
print(f"\nearly signal (gap>=0.15 before last turn): {n_early}/{n_pair} = {p_early:.1f}%")
print(f"terminal signal (risk F > risk P at shared last k): {n_term}/{n_pair} = {p_term:.1f}%")
verdict = "PASS" if p_early >= 60 else "FAIL"
print("VERDICT: P-E4", verdict)

json.dump({"protocol": {"n_pairs": n_pair, "margin": 0.15, "predicate": "early>=60%",
                        "model": "GBM(300) refit on h3_rows pooled-k, dev=task even, seed=20261030",
                        "refit_test_auroc": round(float(auc_refit), 4),
                        "note": "pooled-k refit vs shipped per-k protocol (0.6477); predicate is on twin gaps"},
           "P_E4_pct": round(p_early, 1), "P_E4": verdict,
           "terminal_signal_pct": round(p_term, 1),
           "pairs": gaps},
          open(os.path.join(HERE, "e4_results.json"), "w", encoding="utf-8"), indent=1)
