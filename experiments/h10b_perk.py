"""h10b_perk.py — iteration 2 of the SOTA push: PER-K models + 3 shape features.

Iteration-1 result (h10_sota_curve, results.json): P-EARLY PASS (first stable
0.65 at k<=6), delta vs A0 +0.0407 CI [0.0023,0.0773], but P-MEAN 0.6871 (<0.70)
and P-FINAL 0.6621 (<0.70) FAIL. Bars unchanged (no lowering).

Change (one lever, predeclared): replace the pooled model with one model PER TURN
k (trained on dev rows at that k, task-parity split kept; configs LR/GBM x grid,
picked by dev AUROC at that k; all reported), plus 3 running shape features:
n_assistant_msgs, tool_rate (n_tool_calls/k), write_ratio (writes/n_tool_calls).

Predicates re-evaluated on the same definitions: P-MEAN mean k4-9 >= 0.70 with
cluster-bootstrap CI excluding A0; P-FINAL final-turn >= 0.70. P-EARLY stays PASS
from iter 1 (kept, not re-claimed).

Output: runtime/h10_sota_curve/h10b_results.json (+ REPORT appended section).
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
F_IT2 = F_A0 + ["n_tool_calls", "n_tool_ok", "writes_so_far", "has_write", "write_ok_so_far",
                "n_auth_calls", "close_no_write", "last_tool_is_write", "distinct_tools",
                "n_think_calls", "n_assistant_msgs", "tool_rate", "write_ratio"]
STATE_VOCAB = ["q_greet", "q_intent", "q_auth", "q_tool", "q_close", "q_confirm"]


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

    grids = {
        "LR": [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
        "GBM": [{"max_depth": d, "max_iter": 200, "random_state": SEED} for d in (3, 5)],
    }

    def per_k_train(k, feats):
        dev = [r for r in rows if r["task_id"] % 2 == 0 and r["k"] == k]
        if len(dev) < 60:
            return None, None, []
        Xd = np.array([[r[f] for f in feats] for r in dev], float)
        yd = np.array([r["y_fail"] for r in dev])
        cands = []
        for fam, grid in grids.items():
            cls = LogisticRegression if fam == "LR" else HistGradientBoostingClassifier
            for p in grid:
                m = cls(**p)
                m.fit(Xd, yd)
                cands.append((fam, p, safe_auc(yd, m.predict_proba(Xd)[:, 1])))
        best = max(cands, key=lambda c: c[2])
        m = (LogisticRegression if best[0] == "LR" else HistGradientBoostingClassifier)(**best[1])
        m.fit(Xd, yd)
        rep = [{"fam": f, "params": repr(p), "dev_auc": round(a, 4)} for f, p, a in cands]
        return m, best, rep

    # score every test row with its per-k model (A3b feature set); A0 anchor with
    # per-k models too, so the pairing stays protocol-identical
    scores = {"A0-perk": np.full(len(test), np.nan), "A3b-perk": np.full(len(test), np.nan)}
    models_by_k = {"A0-perk": {}, "A3b-perk": {}}
    picks, all_reports = {}, {}
    ytest = np.array([r["y_fail"] for r in test])
    test_by_k = {}
    for i, r in enumerate(test):
        test_by_k.setdefault(r["k"], []).append(i)

    for k in sorted(test_by_k):
        idx = test_by_k[k]
        yk = ytest[idx]
        if len(idx) < 30 or len(set(yk.tolist())) < 2:
            continue
        for tag, feats in (("A0-perk", F_A0), ("A3b-perk", F_IT2 + [f"vis_{s}" for s in STATE_VOCAB])):
            m, best, rep = per_k_train(k, feats)
            if m is None:
                continue
            models_by_k[tag][k] = (m, feats)
            Xs = np.array([[test[i][f] for f in feats] for i in idx], float)
            scores[tag][idx] = m.predict_proba(Xs)[:, 1]
            if tag == "A3b-perk":
                picks[k] = {"fam": best[0], "params": repr(best[1]), "dev_auc": round(best[2], 4)}
                all_reports[k] = rep

    # fallback for k without a trained model (rare long tail): nearest LOWER k model
    # (declared; n_rows_fallback reported)
    trained_ks = sorted(models_by_k["A3b-perk"])
    n_fallback = 0
    for i, r in enumerate(test):
        if not np.isnan(scores["A3b-perk"][i]):
            continue
        lower = [kk for kk in trained_ks if kk <= r["k"]]
        kk = lower[-1] if lower else trained_ks[0]
        m, feats = models_by_k["A3b-perk"][kk]
        scores["A3b-perk"][i] = m.predict_proba(np.array([[r[f] for f in feats]], float))[0, 1]
        m0, f0 = models_by_k["A0-perk"].get(kk, (None, None))
        if m0 is not None:
            scores["A0-perk"][i] = m0.predict_proba(np.array([[r[f] for f in f0]], float))[0, 1]
        n_fallback += 1
    print(f"fallback rows (k without trained model): {n_fallback}")

    curve = []
    for k in sorted(test_by_k):
        idx = test_by_k[k]
        if len(idx) < 30 or np.isnan(scores["A3b-perk"][idx]).all():
            continue
        yk = ytest[idx]
        row = {"k": k, "n_open_test": len(idx), "fail_rate_at_k": round(100 * yk.mean(), 1),
               "pick": picks.get(k)}
        for tag in scores:
            s = scores[tag][idx]
            row[f"auroc_{tag}"] = round(safe_auc(yk, s), 4) if not np.isnan(s).all() else None
        curve.append(row)

    # final-turn with per-k models
    fin_idx = [i for i, r in enumerate(test) if r["k"] == FINAL_K[r["id"]]]
    yf = ytest[fin_idx]
    final = {"n_test_episodes": len(fin_idx), "fail_rate": round(100 * yf.mean(), 1)}
    for tag in scores:
        s = scores[tag][fin_idx]
        final[f"auroc_{tag}"] = round(safe_auc(yf, s), 4) if not np.isnan(s).all() else None

    mean_auc = {}
    for tag in scores:
        vals = [c[f"auroc_{tag}"] for c in curve if 4 <= c["k"] <= 9 and c.get(f"auroc_{tag}") is not None]
        mean_auc[tag] = float(np.mean(vals)) if vals else None

    # paired cluster bootstrap (same as iter 1): delta mean-k4-9 (A3b-perk - A0-perk)
    task_ids = sorted({r["task_id"] for r in test})
    by_task = {}
    for i, r in enumerate(test):
        by_task.setdefault(r["task_id"], []).append(i)
    ks_rng = [c["k"] for c in curve if 4 <= c["k"] <= 9]
    rng = np.random.default_rng(SEED)
    deltas = []
    for b in range(1000):
        pick_tasks = rng.choice(len(task_ids), size=len(task_ids), replace=True)
        idx = np.concatenate([by_task[task_ids[t]] for t in pick_tasks])
        d_turn = []
        for k in ks_rng:
            sub = idx[np.isin(idx, test_by_k.get(k, []))] if k in test_by_k else np.array([], int)
            if len(sub) < 30:
                continue
            yk = ytest[sub]
            a_n = safe_auc(yk, scores["A3b-perk"][sub])
            a_0 = safe_auc(yk, scores["A0-perk"][sub])
            if not (np.isnan(a_n) or np.isnan(a_0)):
                d_turn.append(a_n - a_0)
        if d_turn:
            deltas.append(float(np.mean(d_turn)))
    deltas = np.array(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

    preds = {
        "P-MEAN": {"bar": "mean k4-9 >= 0.70 (h3: 0.6459)",
                   "value": mean_auc, "delta_mean": float(deltas.mean()), "delta_ci95": ci,
                   "pass": mean_auc["A3b-perk"] >= 0.70 and ci[0] > 0},
        "P-FINAL": {"bar": "final-turn AUC >= 0.70 (h3: 0.6477)",
                    "value": final, "pass": (final["auroc_A3b-perk"] or 0) >= 0.70},
    }
    out = {"protocol": {"seed": SEED, "models": "per-k (one model per turn, dev-picked)",
                        "fallback": "nearest lower k model for k without training rows",
                        "n_rows_fallback": n_fallback,
                        "features_A3b": F_IT2 + [f"vis_{s}" for s in STATE_VOCAB],
                        "bootstrap": {"B": 1000, "clusters": "test task_id"},
                        "predicates": preds},
           "picks": picks, "curve": curve, "final": final, "mean_auc_k4_9": mean_auc}
    json.dump(out, open(os.path.join(OUT, "h10b_results.json"), "w", encoding="utf-8"), indent=1)
    json.dump(all_reports, open(os.path.join(OUT, "h10b_model_select.json"), "w", encoding="utf-8"), indent=1)

    print("| k | n | fail% | A0-perk | A3b-perk | pick |")
    for c in curve:
        print(f"| {c['k']} | {c['n_open_test']} | {c['fail_rate_at_k']} | {c['auroc_A0-perk']} | "
              f"{c['auroc_A3b-perk']} | {c['pick']} |")
    print("FINAL:", final)
    print("MEAN k4-9:", mean_auc)
    for name, p in preds.items():
        print(f"**{name}: {'PASS' if p['pass'] else 'FAIL'}** {json.dumps(p)[:180]}")


if __name__ == "__main__":
    main()
