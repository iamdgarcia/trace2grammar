"""h1_ablation.py — does the grammar walk carry delivered-outcome signal beyond
surface text and beyond the shipped rule verdict?

Protocol (predeclared in the task brief):
- Corpus: 1,980 official tau-bench episodes (walker verdicts + env-verified reward).
- Split: dev = task_id even, test = task_id odd (tasks never straddle the split).
- Tuning: ONLY on dev, 5-fold CV (AUROC), fixed random_state. Test is predicted
  exactly once per configuration, then reported (overall + per model slice).
- Label: fail = positive (reward < 1).
- All configurations tried are reported, including the ones that lose.

Feature sets:
  B0  majority class (constant predictor)
  B1  TF-IDF(1-2 grams, min_df=3, max 20k) of concatenated USER messages + LR
  B2  shipped-rule risk vector [D>=1, D>=2, reached_close, n_states] + LR
  M1  richer walk features + LR
  M2  M1 features + HistGradientBoosting
  M3  TF-IDF + M1 features (does the walk ADD to text?)
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor"
os.makedirs(HERE, exist_ok=True)

SEED = 20261025
results = []          # every configuration tried
missing = []

# ---------------------------------------------------------------- load data
rows = json.load(open(SRC + "/rows_tb.json", encoding="utf-8"))
rows = [r for r in rows if r.get("sanity_ok", True)]

raw = []
for fn in ("gpt-4o-airline.json", "gpt-4o-retail.json",
           "sonnet-35-new-airline.json", "sonnet-35-new-retail.json"):
    eps = json.load(open(os.path.join(SRC, fn), encoding="utf-8"))
    model = "gpt4o" if fn.startswith("gpt") else "sonnet"
    domain = "airline" if "airline" in fn else "retail"
    for ep in eps:
        raw.append((model, domain, ep))

RAW = {}
for model, domain, ep in raw:
    key = f"{model}:{domain}:{ep['task_id']}:{ep['trial']}"
    user_text = " ".join(
        str(m.get("content", "")) for m in ep["traj"] if m.get("role") == "user")
    RAW[key] = user_text

for r in rows:
    t = RAW.get(r["id"])
    if t is None:
        missing.append(r["id"])
    r["text"] = t or ""
    r["fail"] = 1.0 if r["reward"] < 1.0 else 0.0

print(f"episodes: {len(rows)} | text matched: {len(rows) - len(missing)} | missing: {len(missing)}")
if missing:
    print("missing ids (excluded):", missing[:5], "...")
    rows = [r for r in rows if r["text"] is not None and r["text"] != ""]

dev = [r for r in rows if r["task_id"] % 2 == 0]
test = [r for r in rows if r["task_id"] % 2 == 1]
print(f"dev {len(dev)} (fail rate {np.mean([r['fail'] for r in dev]):.3f}) | "
      f"test {len(test)} (fail rate {np.mean([r['fail'] for r in test]):.3f})")

y_dev = np.array([r["fail"] for r in dev])
y_test = np.array([r["fail"] for r in test])


def walk_features(episodes):
    """M1 feature matrix (dense)."""
    cols = []
    for r in episodes:
        onehot = [1.0 if r["outcome_convo"] == k else 0.0
                  for k in ("theta_cont", "theta_aban", "theta_hand")]
        cols.append([
            float(r["D"]), float(r["D"] >= 1), float(r["D"] >= 2),
            float(r["error_count"]), float(r["n_states"]),
            float(bool(r["reached_close"])),
            *onehot,
            1.0 if r["domain"] == "airline" else 0.0,
            1.0 if r["domain"] == "retail" else 0.0,
            1.0 if r["model"] == "sonnet" else 0.0,
            float(r["n_states"]) * float(r["D"]),           # interaction
        ])
    return np.array(cols)


B2_COLS = 4


def rule_features(episodes):
    cols = []
    for r in episodes:
        cols.append([float(r["D"] >= 1), float(r["D"] >= 2),
                     float(bool(r["reached_close"])), float(r["n_states"])])
    return np.array(cols)


Xw_dev, Xw_test = walk_features(dev), walk_features(test)
Xr_dev, Xr_test = rule_features(dev), rule_features(test)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
Xt_dev = tfidf.fit_transform([r["text"] for r in dev])
Xt_test = tfidf.transform([r["text"] for r in test])
print(f"tf-idf: dev {Xt_dev.shape}, test {Xt_test.shape}")

Xm3_dev = hstack([Xt_dev, csr_matrix(Xw_dev)]).tocsr()
Xm3_test = hstack([Xt_test, csr_matrix(Xw_test)]).tocsr()


def cv_auroc(make_model, X, y, k=5):
    aucs = []
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    for tr, va in kf.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def record(name, params, dev_cv, model, X_test, slice_note=""):
    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    flags = p_test >= 0.5
    flagged = int(flags.sum())
    fr = float(y_test[flags].mean()) if flagged else float("nan")
    entry = {"config": name, "params": params,
             "dev_cv_auroc_mean": round(dev_cv[0], 4),
             "dev_cv_auroc_std": round(dev_cv[1], 4),
             "test_auroc": round(float(auc), 4),
             "test_flagged_n": flagged, "test_flag_failrate": round(fr, 4),
             "test_failrate_base": round(float(y_test.mean()), 4),
             "slice": slice_note}
    results.append(entry)
    print(json.dumps(entry))
    return entry


# ---------------------------------------------------------------- B0 majority
p_const = np.full(len(test), y_dev.mean())
results.append({"config": "B0 majority-class constant",
                "params": {"p": round(float(y_dev.mean()), 4)},
                "dev_cv_auroc_mean": 0.5, "dev_cv_auroc_std": 0.0,
                "test_auroc": 0.5,
                "test_flagged_n": len(test) if y_dev.mean() >= 0.5 else 0,
                "test_flag_failrate": round(float(y_test.mean()), 4),
                "test_failrate_base": round(float(y_test.mean()), 4),
                "slice": "constant predictor; AUROC 0.5 by definition"})
print("B0 recorded (AUROC 0.5 by definition)")

# ---------------------------------------------------------------- B1 tf-idf + LR
best = None
for C in (0.1, 1.0, 10.0):
    auc, sd = cv_auroc(lambda C=C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
                       Xt_dev, y_dev)
    print(f"  B1 C={C}: dev CV AUROC {auc:.4f} +/- {sd:.4f}")
    if best is None or auc > best[0]:
        best = (auc, C)
b1_model = LogisticRegression(C=best[1], max_iter=2000, random_state=SEED).fit(Xt_dev, y_dev)
record("B1 tfidf+LR", {"C": best[1]}, (best[0], 0.0), b1_model, Xt_test)
p_test_b1 = b1_model.predict_proba(Xt_test)[:, 1]

# ---------------------------------------------------------------- B2 rules + LR
best = None
for C in (0.1, 1.0, 10.0):
    auc, sd = cv_auroc(lambda C=C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
                       Xr_dev, y_dev)
    print(f"  B2 C={C}: dev CV AUROC {auc:.4f} +/- {sd:.4f}")
    if best is None or auc > best[0]:
        best = (auc, C)
record("B2 rule-vector+LR", {"C": best[1]}, (best[0], 0.0),
       LogisticRegression(C=best[1], max_iter=2000, random_state=SEED).fit(Xr_dev, y_dev),
       Xr_test)

# ---------------------------------------------------------------- M1 walk + LR
best = None
for C in (0.1, 1.0, 10.0):
    auc, sd = cv_auroc(lambda C=C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
                       Xw_dev, y_dev)
    print(f"  M1 C={C}: dev CV AUROC {auc:.4f} +/- {sd:.4f}")
    if best is None or auc > best[0]:
        best = (auc, C)
m1_model = LogisticRegression(C=best[1], max_iter=2000, random_state=SEED).fit(Xw_dev, y_dev)
m1 = record("M1 walk+LR", {"C": best[1]}, (best[0], 0.0), m1_model, Xw_test)

# ---------------------------------------------------------------- M2 walk + GBM
best = None
for it in (100, 300):
    auc, sd = cv_auroc(
        lambda it=it: HistGradientBoostingClassifier(max_iter=it, random_state=SEED),
        Xw_dev, y_dev)
    print(f"  M2 max_iter={it}: dev CV AUROC {auc:.4f} +/- {sd:.4f}")
    if best is None or auc > best[0]:
        best = (auc, it)
record("M2 walk+GBM", {"max_iter": best[1]}, (best[0], 0.0),
       HistGradientBoostingClassifier(max_iter=best[1], random_state=SEED).fit(Xw_dev, y_dev),
       Xw_test)

# ---------------------------------------------------------------- M3 text + walk
best = None
for C in (0.1, 1.0, 10.0):
    auc, sd = cv_auroc(lambda C=C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
                       Xm3_dev, y_dev)
    print(f"  M3 C={C}: dev CV AUROC {auc:.4f} +/- {sd:.4f}")
    if best is None or auc > best[0]:
        best = (auc, C)
m3_model = LogisticRegression(C=best[1], max_iter=2000, random_state=SEED).fit(Xm3_dev, y_dev)
m3 = record("M3 tfidf+walk+LR", {"C": best[1]}, (best[0], 0.0), m3_model, Xm3_test)

# ---------------------------------------------------------------- per-model slices
p_test_m1 = m1_model.predict_proba(Xw_test)[:, 1]
p_test_m3 = m3_model.predict_proba(Xm3_test)[:, 1]
for model in ("gpt4o", "sonnet"):
    idx = np.array([i for i, r in enumerate(test) if r["model"] == model])
    yy = y_test[idx]
    results.append({"config": f"slice {model}: M1 walk+LR", "params": {"C": m1["params"]["C"]},
                    "dev_cv_auroc_mean": m1["dev_cv_auroc_mean"], "dev_cv_auroc_std": 0.0,
                    "test_auroc": round(float(roc_auc_score(yy, p_test_m1[idx])), 4),
                    "test_flagged_n": int((p_test_m1[idx] >= 0.5).sum()),
                    "test_flag_failrate": round(float(yy[p_test_m1[idx] >= 0.5].mean()), 4),
                    "test_failrate_base": round(float(yy.mean()), 4),
                    "slice": f"{model} n={len(idx)}"})
    results.append({"config": f"slice {model}: M3 tfidf+walk+LR", "params": {"C": m3["params"]["C"]},
                    "dev_cv_auroc_mean": m3["dev_cv_auroc_mean"], "dev_cv_auroc_std": 0.0,
                    "test_auroc": round(float(roc_auc_score(yy, p_test_m3[idx])), 4),
                    "test_flagged_n": int((p_test_m3[idx] >= 0.5).sum()),
                    "test_flag_failrate": round(float(yy[p_test_m3[idx] >= 0.5].mean()), 4),
                    "test_failrate_base": round(float(yy.mean()), 4),
                    "slice": f"{model} n={len(idx)}"})

json.dump({"protocol": {"split": "task_id parity (dev=even, test=odd)",
                        "tuning": "5-fold CV on dev only; test predicted once per config",
                        "seed": SEED, "label": "fail=positive (reward<1)",
                        "cv_caveat": "CV folds inside dev split by row, so tasks repeat across folds -> dev CV is task-leaked and optimistic; the held-out test (disjoint task_ids) is the honest number"},
           "missing_text_episodes": missing,
           "results": results},
          open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
          indent=1)

# paired bootstrap CI for the headline delta M3 - B1 on test
rng = np.random.default_rng(SEED)
deltas = []
for _ in range(2000):
    idx = rng.integers(0, len(y_test), len(y_test))
    a1 = roc_auc_score(y_test[idx], p_test_m3[idx])
    a0 = roc_auc_score(y_test[idx], p_test_b1[idx])
    deltas.append(a1 - a0)
ci = [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)]
print("M3-B1 paired bootstrap 95% CI:", ci)

# ---------------------------------------------------------------- THE QUESTION
by = {e["config"]: e["test_auroc"] for e in results if "slice" not in e.get("slice", "")}
b0, b1 = by["B0 majority-class constant"], by["B1 tfidf+LR"]
b2, m1v = by["B2 rule-vector+LR"], by["M1 walk+LR"]
m2v, m3v = by["M2 walk+GBM"], by["M3 tfidf+walk+LR"]
answer = {
    "test_auroc": {"B0": b0, "B1": b1, "B2": b2, "M1": m1v, "M2": m2v, "M3": m3v},
    "M3_minus_B1": round(m3v - b1, 4),
    "M1_minus_B2": round(m1v - b2, 4),
    "M3_best": m3v >= max(b0, b1, b2, m1v, m2v) - 1e-9,
    "walk_adds_to_text": bool(m3v - b1 >= 0.01),
    "walk_beats_rules": bool(m1v > b2),
}
json.dump(answer, open(os.path.join(HERE, "answer.json"), "w", encoding="utf-8"), indent=1)
print("\nANSWER:", json.dumps(answer, indent=1))
