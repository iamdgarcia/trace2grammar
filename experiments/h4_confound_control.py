"""h4_confound_control.py — kill-or-confirm the H1 headline (walk adds signal beyond text).

Predeclared (threads/t2g-honesty/PLAN.md, Claim 1; probe 8 personas 2026-10-30):

Confounds found in h1_walk_ablation.py:
  (a) B1 TF-IDF filters role=='user': denies the tool-result channel, which is exactly
      what walk features encode (D>0 iff error_count>0 in 1,979/1,980 tau episodes).
  (b) M1 walk feature set contains model one-hot (sonnet) and domain one-hots: model/domain
      identity, not grammar.
  (c) dev CV folds by row -> task leakage (self-documented in h1 as cv_caveat).

Arms (ALL get model/domain one-hots where listed; C tuned on dev ONLY by GROUPED 5-fold
CV, groups=task_id; test predicted exactly once per arm):
  A  B1'  TF-IDF over ALL roles except system (policy boilerplate) + one-hots [model,domain] + LR
  B  A + walk-pure: [D, D>=1, D>=2, n_states, reached_close, outcome one-hot x3, n_states*D]
       (NO error_count, NO model/domain one-hots)
  C  B2'  [D>=1, D>=2, reached_close, n_states] + one-hots [model,domain] + LR
  D  M1-pure  walk-pure only (grammar-only signal, no metadata)
  E  content-blind: [n_states, D, D>=1, D>=2, reached_close] + LR  (naagarajan2026structural control)

Split: task_id parity (dev=even, test=odd), identical to h1. Label fail = reward<1.
Verdict: CONFIRM if dAUROC(B-A) >= +0.01 AND cluster-bootstrap 95% CI excludes 0
(resampling task_ids, 2000 resamples, seed 20261025); else REFRAME.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor"
SEED = 20261025

rows = json.load(open(SRC + "/rows_tb.json", encoding="utf-8"))
rows = [r for r in rows if r.get("sanity_ok", True)]

RAW = {}
for fn in ("gpt-4o-airline.json", "gpt-4o-retail.json",
           "sonnet-35-new-airline.json", "sonnet-35-new-retail.json"):
    eps = json.load(open(os.path.join(SRC, fn), encoding="utf-8"))
    model = "gpt4o" if fn.startswith("gpt") else "sonnet"
    domain = "airline" if "airline" in fn else "retail"
    for ep in eps:
        # all roles except system (policy boilerplate identical across episodes)
        parts = [str(m.get("content", "")) for m in ep["traj"] if m.get("role") != "system"]
        RAW[f"{model}:{domain}:{ep['task_id']}:{ep['trial']}"] = "\n".join(parts)

missing = []
for r in rows:
    r["text"] = RAW.get(r["id"], "")
    if not r["text"]:
        missing.append(r["id"])
    r["fail"] = 1.0 if r["reward"] < 1.0 else 0.0
print(f"episodes: {len(rows)} | text matched: {len(rows)-len(missing)} | missing: {len(missing)}")
rows = [r for r in rows if r["text"]]

dev = [r for r in rows if r["task_id"] % 2 == 0]
test = [r for r in rows if r["task_id"] % 2 == 1]
y_dev = np.array([r["fail"] for r in dev])
y_test = np.array([r["fail"] for r in test])
print(f"dev {len(dev)} (fail {y_dev.mean():.3f}) | test {len(test)} (fail {y_test.mean():.3f})")


def onehots(episodes):
    return [[1.0 if r["model"] == "sonnet" else 0.0,
             1.0 if r["domain"] == "airline" else 0.0] for r in episodes]


def walk_pure(episodes):
    cols = []
    for r in episodes:
        oh = [1.0 if r["outcome_convo"] == k else 0.0
              for k in ("theta_cont", "theta_aban", "theta_hand")]
        cols.append([float(r["D"]), float(r["D"] >= 1), float(r["D"] >= 2),
                     float(r["n_states"]), float(bool(r["reached_close"])),
                     *oh, float(r["n_states"]) * float(r["D"])])
    return np.array(cols)


def rule_plus_meta(episodes):
    cols = []
    for r in episodes:
        cols.append([float(r["D"] >= 1), float(r["D"] >= 2),
                     float(bool(r["reached_close"])), float(r["n_states"]),
                     1.0 if r["model"] == "sonnet" else 0.0,
                     1.0 if r["domain"] == "airline" else 0.0])
    return np.array(cols)


def content_blind(episodes):
    cols = []
    for r in episodes:
        cols.append([float(r["n_states"]), float(r["D"]), float(r["D"] >= 1),
                     float(r["D"] >= 2), float(bool(r["reached_close"]))])
    return np.array(cols)


Xm_dev, Xm_test = onehots(dev), onehots(test)
Xw_dev, Xw_test = walk_pure(dev), walk_pure(test)
Xr_dev, Xr_test = rule_plus_meta(dev), rule_plus_meta(test)
Xc_dev, Xc_test = content_blind(dev), content_blind(test)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
Xt_dev = tfidf.fit_transform([r["text"] for r in dev])
Xt_test = tfidf.transform([r["text"] for r in test])

XA_dev = hstack([Xt_dev, csr_matrix(np.array(Xm_dev))]).tocsr()
XA_test = hstack([Xt_test, csr_matrix(np.array(Xm_test))]).tocsr()
XB_dev = hstack([XA_dev, csr_matrix(Xw_dev)]).tocsr()
XB_test = hstack([XA_test, csr_matrix(Xw_dev_test := Xw_test)]).tocsr()

groups_dev = np.array([r["task_id"] for r in dev])
out = {"protocol": {"split": "task_id parity (dev=even, test=odd)",
                    "tuning": "C in {0.1,1,10} by GROUPED 5-fold CV (groups=task_id) on dev only",
                    "seed": SEED, "label": "fail=positive (reward<1)",
                    "text": "all roles except system (policy boilerplate)",
                    "verdict_rule": "CONFIRM if dAUROC(B-A) >= +0.01 AND cluster-bootstrap 95% CI excludes 0"},
       "missing_text_episodes": missing, "results": []}


def fit_arm(name, Xd, Xt):
    gkf = GroupKFold(n_splits=5)
    best = None
    for C in (0.1, 1.0, 10.0):
        aucs = []
        for tr, va in gkf.split(Xd, y_dev, groups=groups_dev):
            m = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            m.fit(Xd[tr], y_dev[tr])
            aucs.append(roc_auc_score(y_dev[va], m.predict_proba(Xd[va])[:, 1]))
        mean = float(np.mean(aucs))
        print(f"  {name} C={C}: grouped dev CV AUROC {mean:.4f}")
        if best is None or mean > best[0]:
            best = (mean, C)
    model = LogisticRegression(C=best[1], max_iter=2000, random_state=SEED).fit(Xd, y_dev)
    p = model.predict_proba(Xt)[:, 1]
    auc = float(roc_auc_score(y_test, p))
    out["results"].append({"arm": name, "C": best[1],
                           "dev_cv_grouped": round(best[0], 4), "test_auroc": round(auc, 4)})
    print(f"  -> {name}: test AUROC {auc:.4f} (C={best[1]})")
    return p, auc


pA, aucA = fit_arm("A B1p tfidf-all-roles+meta", XA_dev, XA_test)
pB, aucB = fit_arm("B A+walk-pure", XB_dev, XB_test)
pC, aucC = fit_arm("C B2p rules+meta", np.array(Xr_dev), np.array(Xr_test))
pD, aucD = fit_arm("D M1-pure walk-only", Xw_dev, Xw_test)
pE, aucE = fit_arm("E content-blind", Xc_dev, Xc_test)

delta = aucB - aucA
print(f"\nHEADLINE dAUROC(B-A) = {delta:+.4f}")

# cluster bootstrap: resample task_ids with replacement
test_tasks = sorted(set(r["task_id"] for r in test))
by_task = {}
for i, r in enumerate(test):
    by_task.setdefault(r["task_id"], []).append(i)
rng = np.random.default_rng(SEED)
deltas = []
for _ in range(2000):
    pick = rng.choice(test_tasks, size=len(test_tasks), replace=True)
    idx = np.concatenate([by_task[t] for t in pick])
    deltas.append(roc_auc_score(y_test[idx], pB[idx]) - roc_auc_score(y_test[idx], pA[idx]))
ci = [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)]
print("cluster-bootstrap 95% CI (task-level):", ci)

confirmed = bool(delta >= 0.01 and ci[0] > 0)
out["headline"] = {"dAUROC_B_minus_A": round(delta, 4),
                   "cluster_bootstrap_ci": ci, "verdict": "CONFIRM" if confirmed else "REFRAME",
                   "aux": {"M1pure_minus_contentblind": round(aucD - aucE, 4),
                           "B2p_test": round(aucC, 4),
                           "M1pure_test": round(aucD, 4),
                           "contentblind_test": round(aucE, 4)}}
json.dump(out, open(os.path.join(HERE, "h4_results.json"), "w", encoding="utf-8"), indent=1)
print("\nVERDICT:", out["headline"]["verdict"])
print(json.dumps(out["headline"], indent=1))
