"""h11_decompose.py — iteration 7: task-prior vs per-episode evidence in the text SOTA.

Question (predeclared, knowledge_base/threads/t2g-sota/PLAN.md iter-7): the text
channel's turn-level curve (0.72@k1 -> 0.80@k8, mean k4-9 0.787) — is the transcript
adding discrimination WITHIN a task over knowing the task? In deployment the task
spec IS in telemetry, so the stopping rule always knows the task; the per-episode
component is what makes an in-flight rule actionable.

Protocol:
- T arm verbatim (h10d): TF-IDF (1,2) min_df=3 max_features=20000 fit on DEV prefixes
  (task parity), one-hots [sonnet, airline], LR, C by GroupKFold(5) grouped CV on dev
  -> score ALL 14,011 rows (dev + test).
- Within-task AUC per turn k: mean over tasks with >=2 trials of BOTH outcomes at k
  (per-task AUC = the task's own ranking). Demeaned-pooled variant: AUC of
  (score - task mean at k) over all eligible rows. Shuffle control: labels permuted
  WITHIN task at k (rng 20261031) = empirical null of the metric.
- Families (E3 tau_sem): episode in a pair with tau_sem != None -> divergent;
  tau_sem == None -> silent (write-check family). Per-family within-task AUC.

Predicates (verbatim from the goal):
- P-WITHIN: within-task AUC >= 0.60 at some k<=6 with >=40 eligible tasks; else the
  task-prior framing is DECLARED.
- P-FAMILY: divergent-family within-task AUC >= 0.65 (n declared) AND silent-family
  reported honestly (expected ~chance per E3).

Output: runtime/h10_sota_curve/h11_results.json.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUT = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/h10_sota_curve"
SEED = 20261031


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    rows = json.load(open(os.path.join(OUT, "h10_rows.json"), encoding="utf-8"))
    dev = [r for r in rows if r["task_id"] % 2 == 0]
    ydev = np.array([r["y_fail"] for r in dev])

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
    Xt_dev = tfidf.fit_transform([r["text"] for r in dev])
    oh_dev = np.array([[1.0 if r["model"] == "sonnet" else 0.0,
                        1.0 if r["domain"] == "airline" else 0.0] for r in dev])
    XT_dev = hstack([Xt_dev, csr_matrix(oh_dev)]).tocsr()
    groups = np.array([r["task_id"] for r in dev])

    # C by grouped CV (h10d protocol); expect 0.1
    cv = {}
    for C in (0.1, 1.0, 10.0):
        aucs = []
        for tr, va in GroupKFold(n_splits=5).split(XT_dev, ydev, groups):
            m = LogisticRegression(max_iter=2000, C=C)
            m.fit(XT_dev[tr], ydev[tr])
            a = safe_auc(ydev[va], m.predict_proba(XT_dev[va])[:, 1])
            if not np.isnan(a):
                aucs.append(a)
        cv[C] = float(np.mean(aucs))
    C = max(cv, key=cv.get)

    tfidf_all = tfidf  # fit on dev only (same vectorizer)
    oh_all = np.array([[1.0 if r["model"] == "sonnet" else 0.0,
                        1.0 if r["domain"] == "airline" else 0.0] for r in rows])
    XT_all = hstack([tfidf_all.transform([r["text"] for r in rows]), csr_matrix(oh_all)]).tocsr()
    m = LogisticRegression(max_iter=2000, C=C)
    m.fit(XT_dev, ydev)
    for i, r in enumerate(rows):
        r["t_score"] = float(m.predict_proba(XT_all[i:i + 1])[0, 1])
    print(f"C={C} (cv {cv}) | scored {len(rows)} rows")

    # E3 families
    e3 = json.load(open(r"C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar/experiments/latent2/e3_results.json",
                        encoding="utf-8"))
    fam = {}
    for p in e3["pairs"]:
        f = "divergent" if p.get("tau_sem") is not None else "silent"
        for ep_id in p["pair"]:
            fam[ep_id] = f
    print(f"families: divergent={sum(1 for v in fam.values() if v=='divergent')} "
          f"silent={sum(1 for v in fam.values() if v=='silent')} (of 204 pair slots)")

    # within-task decomposition per k
    by_k = {}
    for i, r in enumerate(rows):
        by_k.setdefault(r["k"], []).append(i)
    rng = np.random.default_rng(SEED)
    within, demeaned, shuffled, n_tasks = {}, {}, {}, {}
    for k, idx in by_k.items():
        per_task = {}
        for i in idx:
            per_task.setdefault(rows[i]["task_id"], []).append(i)
        elig = {t: ix for t, ix in per_task.items()
                if len(ix) >= 4 and len({rows[i]["y_fail"] for i in ix}) == 2}
        aucs = [safe_auc([rows[i]["y_fail"] for i in ix], [rows[i]["t_score"] for i in ix])
                for ix in elig.values()]
        aucs = [a for a in aucs if not np.isnan(a)]
        if not elig:
            continue
        within[k] = float(np.mean(aucs))
        n_tasks[k] = len(elig)
        # demeaned pooled
        dx, dy = [], []
        for ix in elig.values():
            mu = float(np.mean([rows[i]["t_score"] for i in ix]))
            dx += [rows[i]["t_score"] - mu for i in ix]
            dy += [rows[i]["y_fail"] for i in ix]
        demeaned[k] = safe_auc(dy, dx)
        # shuffle control (within task, same eligibility): permute labels inside each
        # task block -> destroys episode evidence, keeps the task prior
        sy2, start = [], 0
        for ix in elig.values():
            n = len(ix)
            block = dy[start:start + n]
            perm = rng.permutation(n)
            sy2 += [block[j] for j in perm]
            start += n
        shuffled[k] = safe_auc(sy2, dx)

    ks = sorted(within)
    # family split: within-task AUC at final turn per family (episode's own last k),
    # restricted to family-labeled episodes in eligible tasks
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])
    fam_final = {}
    for f in ("divergent", "silent"):
        per_task = {}
        for r in rows:
            if fam.get(r["id"]) != f or r["k"] != FINAL_K[r["id"]]:
                continue
            per_task.setdefault(r["task_id"], []).append(r)
        elig = {t: ix for t, ix in per_task.items()
                if len(ix) >= 2 and len({x["y_fail"] for x in ix}) == 2}
        aucs = [safe_auc([x["y_fail"] for x in ix], [x["t_score"] for x in ix]) for ix in elig.values()]
        aucs = [a for a in aucs if not np.isnan(a)]
        fam_final[f] = {"n_tasks": len(elig), "mean_within_auc": float(np.mean(aucs)) if aucs else None,
                        "n_episodes": sum(len(ix) for ix in elig.values())}

    # predicates (verbatim)
    p_within = {"bar": "within-task AUC >=0.60 at some k<=6 with >=40 eligible tasks",
                "curve": {str(k): {"auc": within[k], "n_tasks": n_tasks[k],
                                   "demeaned": demeaned.get(k), "shuffled_null": shuffled.get(k)}
                          for k in ks},
                "pass": any(k <= 6 and within[k] >= 0.60 and n_tasks[k] >= 40 for k in ks)}
    p_family = {"bar": "divergent within-task >=0.65 AND silent reported honestly",
                **fam_final,
                "pass": fam_final["divergent"]["mean_within_auc"] is not None
                        and fam_final["divergent"]["mean_within_auc"] >= 0.65}

    # cross-task reference on the same scored rows (test side only, as h10d)
    test = [r for r in rows if r["task_id"] % 2 == 1]
    test_by_k = {}
    for r in test:
        test_by_k.setdefault(r["k"], []).append(r)
    cross = {k: safe_auc([r["y_fail"] for r in ix], [r["t_score"] for r in ix])
             for k, ix in sorted(test_by_k.items()) if len(ix) >= 30}

    out = {"protocol": {"seed": SEED, "arm": "T (h10d verbatim)", "C": C, "cv": cv,
                        "eligible_task_rule": ">=4 trials at k with both outcomes (within-task); "
                                              ">=2 with both (final-turn family split)",
                        "predicates": {"P-WITHIN": p_within, "P-FAMILY": p_family}},
           "cross_task_curve_test": {str(k): round(v, 4) for k, v in cross.items() if not np.isnan(v)},
           "within_task": {str(k): round(within[k], 4) for k in ks},
           "demeaned": {str(k): (round(demeaned[k], 4) if not np.isnan(demeaned[k]) else None) for k in ks},
           "shuffled_null": {str(k): (round(shuffled[k], 4) if not np.isnan(shuffled[k]) else None) for k in ks},
           "n_tasks": {str(k): n_tasks[k] for k in ks},
           "family_final": {f: {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in d.items()} for f, d in fam_final.items()}}
    json.dump(out, open(os.path.join(OUT, "h11_results.json"), "w", encoding="utf-8"), indent=1)

    print(f"{'k':>3} {'n_tasks':>7} {'within':>7} {'demeaned':>8} {'shuffle':>7} {'cross(test)':>11}")
    for k in ks:
        print(f"{k:>3} {n_tasks[k]:>7} {within[k]:>7.4f} {demeaned[k]:>8.4f} {shuffled[k]:>7.4f} "
              f"{cross.get(k, float('nan')):>11.4f}")
    print("family final-turn:", json.dumps(out["family_final"]))
    for name, p in (("P-WITHIN", p_within), ("P-FAMILY", p_family)):
        print(f"**{name}: {'PASS' if p['pass'] else 'FAIL'}** {p['bar']}")


if __name__ == "__main__":
    main()
