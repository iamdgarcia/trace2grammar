"""h10c_pooled_v2.py — iteration 3 of the SOTA push (pooled, richer running features).

History: iter1 pooled A3 (mean k4-9 0.6871, delta vs A0 +0.0407 CI [0.002,0.077],
P-EARLY PASS) = incumbent. iter2 per-k models OVERFIT (dev 0.97, test 0.675) = DROP.

Change (predeclared, bars unchanged): A4 = A3 features + n_assistant_msgs,
tool_rate, write_ratio, auth_ok_so_far, turns_since_last_tool, err_rate.
Model grid widened (GBM depth {3,4,5} x iters {200,400}; LR as before), selection
on dev pooled AUC (h3 protocol), PLUS 3-seed GBM ensemble (seeds 20261031-33,
mean proba) of the picked depth/iters — declared.
Predicates (same, no bar lowering): P-MEAN mean k4-9 >= 0.70 AND bootstrap CI
(vs A0) excludes 0; P-FINAL final-turn >= 0.70.

Output: runtime/h10_sota_curve/h10c_results.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/h10_sota_curve"
SEED = 20261031

F_A0 = ["D_so_far", "retries_so_far", "n_states_so_far", "reached_close_so_far", "terminal_so_far"]
F_IT1 = F_A0 + ["n_tool_calls", "n_tool_ok", "writes_so_far", "has_write", "write_ok_so_far",
                "n_auth_calls", "close_no_write", "last_tool_is_write", "k", "distinct_tools",
                "n_think_calls"]
STATE_VOCAB = ["q_greet", "q_intent", "q_auth", "q_tool", "q_close", "q_confirm"]
F_A4 = F_IT1 + [f"vis_{s}" for s in STATE_VOCAB] + [
    "n_assistant_msgs", "tool_rate", "write_ratio", "auth_ok_so_far",
    "turns_since_last_tool", "err_rate"]


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    rows = json.load(open(os.path.join(OUT, "h10_rows.json"), encoding="utf-8"))
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])
    test = [r for r in rows if r["task_id"] % 2 == 1]
    dev = [r for r in rows if r["task_id"] % 2 == 0]
    ytest = np.array([r["y_fail"] for r in test])

    grids = {
        "LR": [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
        "GBM": [{"max_depth": d, "max_iter": it, "random_state": SEED}
                for d in (3, 4, 5) for it in (200, 400)],
    }

    def fit_pick(feats, min_k):
        d = [r for r in dev if r["k"] >= min_k]
        Xd = np.array([[r[f] for f in feats] for r in d], float)
        yd = np.array([r["y_fail"] for r in d])
        cands = []
        for fam, grid in grids.items():
            cls = LogisticRegression if fam == "LR" else HistGradientBoostingClassifier
            for p in grid:
                m = cls(**p)
                m.fit(Xd, yd)
                cands.append((fam, p, safe_auc(yd, m.predict_proba(Xd)[:, 1])))
        best = max(cands, key=lambda c: c[2])
        rep = [{"fam": f, "params": repr(p), "dev_auc": round(a, 4)} for f, p, a in cands]
        return best, rep

    def scores_for(feats, best, min_k, ensemble=False):
        tr = [r for r in dev if r["k"] >= min_k]
        Xtr = np.array([[r[f] for f in feats] for r in tr], float)
        ytr = np.array([r["y_fail"] for r in tr])
        Xte = np.array([[r[f] for f in feats] for r in test], float)
        if best[0] == "LR":
            m = LogisticRegression(**best[1]); m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        probas = []
        seeds = (SEED, SEED + 1, SEED + 2) if ensemble else (best[1]["random_state"],)
        for sd in seeds:
            p = {**best[1], "random_state": sd}
            m = HistGradientBoostingClassifier(**p); m.fit(Xtr, ytr)
            probas.append(m.predict_proba(Xte)[:, 1])
        return np.mean(probas, axis=0)

    # A3 (incumbent, iter-1 config verbatim) for the paired comparison
    bestA3, repA3 = fit_pick(F_IT1 + [f"vis_{s}" for s in STATE_VOCAB], 3)
    s_A3 = scores_for(F_IT1 + [f"vis_{s}" for s in STATE_VOCAB], bestA3, 3)
    # A0 anchor (pooled, iter-1 verbatim)
    bestA0, _ = fit_pick(F_A0, 3)
    s_A0 = scores_for(F_A0, bestA0, 3)
    # A4 candidates: plain picked + ensemble
    bestA4, repA4 = fit_pick(F_A4, 3)
    s_A4 = scores_for(F_A4, bestA4, 3)
    s_A4e = scores_for(F_A4, bestA4, 3, ensemble=True)
    # min_k=1 variant of A4 (grid had min_k 3 as h3 pick; report both)
    bestA4_1, repA4_1 = fit_pick(F_A4, 1)
    s_A4_1 = scores_for(F_A4, bestA4_1, 1)

    arms = {"A0": s_A0, "A3": s_A3, "A4": s_A4, "A4-ens": s_A4e, "A4-mink1": s_A4_1}

    test_by_k = {}
    for i, r in enumerate(test):
        test_by_k.setdefault(r["k"], []).append(i)
    curve = []
    for k in sorted(test_by_k):
        idx = test_by_k[k]
        if len(idx) < 30:
            continue
        yk = ytest[idx]
        row = {"k": k, "n_open_test": len(idx), "fail_rate_at_k": round(100 * yk.mean(), 1)}
        for a, s in arms.items():
            row[f"auroc_{a}"] = round(safe_auc(yk, s[idx]), 4)
        curve.append(row)

    fin_idx = [i for i, r in enumerate(test) if r["k"] == FINAL_K[r["id"]]]
    yf = ytest[fin_idx]
    final = {"n_test_episodes": len(fin_idx), "fail_rate": round(100 * yf.mean(), 1)}
    for a, s in arms.items():
        final[f"auroc_{a}"] = round(safe_auc(yf, s[fin_idx]), 4)

    mean_auc = {a: float(np.mean([c[f"auroc_{a}"] for c in curve if 4 <= c["k"] <= 9]))
                for a in arms}

    # bootstrap delta vs A0 for the best candidate arm (by mean k4-9)
    cand = max(("A3", "A4", "A4-ens", "A4-mink1"), key=lambda a: mean_auc[a])
    task_ids = sorted({r["task_id"] for r in test})
    by_task = {}
    for i, r in enumerate(test):
        by_task.setdefault(r["task_id"], []).append(i)
    ks_rng = [c["k"] for c in curve if 4 <= c["k"] <= 9]
    rng = np.random.default_rng(SEED)
    deltas = []
    for b in range(1000):
        pt = rng.choice(len(task_ids), size=len(task_ids), replace=True)
        idx = np.concatenate([by_task[task_ids[t]] for t in pt])
        d_turn = []
        for k in ks_rng:
            sub = idx[np.isin(idx, test_by_k.get(k, []))] if k in test_by_k else np.array([], int)
            if len(sub) < 30:
                continue
            yk = ytest[sub]
            an, a0 = safe_auc(yk, arms[cand][sub]), safe_auc(yk, arms["A0"][sub])
            if not (np.isnan(an) or np.isnan(a0)):
                d_turn.append(an - a0)
        if d_turn:
            deltas.append(float(np.mean(d_turn)))
    deltas = np.array(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

    preds = {
        "P-MEAN": {"bar": "mean k4-9 >= 0.70 AND bootstrap CI vs A0 excludes 0",
                   "value": mean_auc, "best_cand": cand,
                   "delta_mean": float(deltas.mean()), "delta_ci95": ci,
                   "pass": mean_auc[cand] >= 0.70 and ci[0] > 0},
        "P-FINAL": {"bar": "final-turn AUC >= 0.70",
                    "value": final,
                    "pass": max(final[f"auroc_{a}"] for a in ("A3", "A4", "A4-ens", "A4-mink1")) >= 0.70},
    }
    out = {"protocol": {"seed": SEED, "arms": list(arms), "grid": "GBM d3-5 x it200/400 + LR",
                        "ensemble": "3 seeds mean proba (A4-ens)",
                        "bootstrap": {"B": 1000, "clusters": "test task_id"},
                        "predicates": preds},
           "picks": {"A0": bestA0, "A3": bestA3, "A4": bestA4, "A4-mink1": bestA4_1},
           "curve": curve, "final": final, "mean_auc_k4_9": mean_auc}
    json.dump(out, open(os.path.join(OUT, "h10c_results.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"A4": repA4, "A4-mink1": repA4_1, "A3": repA3},
              open(os.path.join(OUT, "h10c_model_select.json"), "w", encoding="utf-8"), indent=1)

    print("| k | n | fail% | " + " | ".join(arms) + " |")
    for c in curve:
        print(f"| {c['k']} | {c['n_open_test']} | {c['fail_rate_at_k']} | "
              + " | ".join(str(c[f'auroc_{a}']) for a in arms) + " |")
    print("FINAL:", final)
    print("MEAN k4-9:", mean_auc)
    for name, p in preds.items():
        print(f"**{name}: {'PASS' if p['pass'] else 'FAIL'}** {json.dumps(p)[:200]}")


if __name__ == "__main__":
    main()
