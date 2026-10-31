"""h10d_fusion.py — iteration 4 of the SOTA push: text+walk fusion AT TURN LEVEL.

h4 tested fusion only at FINAL turn (B arm 0.7914 vs text 0.7814, dAUROC +0.0099
n.s.). The in-flight question — does the walk add signal beyond text at turn k,
where a stopping rule would act — was never tested. This is it.

Protocol (predeclared in knowledge_base/threads/t2g-sota/PLAN.md, iter-4 section):
- text per prefix = all roles except system, "\n".join (h4 verbatim).
- T arm: TF-IDF (1,2)-grams min_df=3 max_features=20000 fit on DEV prefixes +
  one-hots [sonnet, airline] + LR, C picked by GroupKFold(5) grouped CV on dev
  (h4 protocol) -> per-turn scores.
- W arm: A4 walk-only pooled (iter-3 verbatim, GBM/LR dev-pick, min_k=3).
- F arm: hstack(TFIDF, one-hots, walk A4 features) + LR, C by grouped CV.
- Same seed 20261031, same task-parity split.

Predicates (new; iter-1..3 bars untouched):
- P-FUSE-TURN: mean k4-9 AUC(F) - AUC(T) >= +0.01 AND cluster bootstrap (task_id,
  1000x) CI95 of the delta excludes 0.
- P-FUSE-FINAL: final-turn AUC(F) >= 0.80 OR delta vs T >= +0.01 with CI excluding 0.
- Walk-only bars (P-MEAN/P-FINAL 0.70) reported unchanged; if <0.70 -> declared
  PLATEAU of the structural frontier (consistent with the E2-E5 epistemic wall).

Output: runtime/h10_sota_curve/h10d_results.json.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/h10_sota_curve"
SEED = 20261031

F_A0 = ["D_so_far", "retries_so_far", "n_states_so_far", "reached_close_so_far", "terminal_so_far"]
F_A4 = F_A0 + ["n_tool_calls", "n_tool_ok", "writes_so_far", "has_write", "write_ok_so_far",
               "n_auth_calls", "close_no_write", "last_tool_is_write", "k", "distinct_tools",
               "n_think_calls",
               "vis_q_greet", "vis_q_intent", "vis_q_auth", "vis_q_tool", "vis_q_close",
               "vis_q_confirm",
               "n_assistant_msgs", "tool_rate", "write_ratio", "auth_ok_so_far",
               "turns_since_last_tool", "err_rate"]
STATE_VOCAB = ["q_greet", "q_intent", "q_auth", "q_tool", "q_close", "q_confirm"]


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def onehots(rows):
    return [[1.0 if r["model"] == "sonnet" else 0.0,
             1.0 if r["domain"] == "airline" else 0.0] for r in rows]


def pick_C(Xd, yd, groups, Cs=(0.1, 1.0, 10.0)):
    gkf = GroupKFold(n_splits=5)
    best, best_auc = None, -1
    allc = []
    for C in Cs:
        aucs = []
        for tr, va in gkf.split(Xd, yd, groups):
            m = LogisticRegression(max_iter=2000, C=C)
            m.fit(Xd[tr], yd[tr])
            a = safe_auc(yd[va], m.predict_proba(Xd[va])[:, 1])
            if not np.isnan(a):
                aucs.append(a)
        mu = float(np.mean(aucs)) if aucs else float("nan")
        allc.append({"C": C, "dev_cv_auc": round(mu, 4)})
        if mu > best_auc:
            best, best_auc = C, mu
    return best, allc


def main():
    rows = json.load(open(os.path.join(OUT, "h10_rows.json"), encoding="utf-8"))
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])
    dev = [r for r in rows if r["task_id"] % 2 == 0]
    test = [r for r in rows if r["task_id"] % 2 == 1]
    ytest = np.array([r["y_fail"] for r in test])
    test_by_k = {}
    for i, r in enumerate(test):
        test_by_k.setdefault(r["k"], []).append(i)

    # ---- W arm (A4 pooled, iter-3 protocol verbatim) -------------------------------
    grids = {
        "LR": [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
        "GBM": [{"max_depth": d, "max_iter": it, "random_state": SEED}
                for d in (3, 4, 5) for it in (200, 400)],
    }
    Xd_w = np.array([[r[f] for f in F_A4] for r in dev if r["k"] >= 3], float)
    yd_w = np.array([r["y_fail"] for r in dev if r["k"] >= 3])
    cands = []
    for fam, grid in grids.items():
        cls = LogisticRegression if fam == "LR" else HistGradientBoostingClassifier
        for p in grid:
            m = cls(**p); m.fit(Xd_w, yd_w)
            cands.append((fam, p, safe_auc(yd_w, m.predict_proba(Xd_w)[:, 1])))
    bestW = max(cands, key=lambda c: c[2])
    tr_idx = [i for i, r in enumerate(dev) if r["k"] >= 3]
    if bestW[0] == "LR":
        mw = LogisticRegression(**bestW[1]); mw.fit(Xd_w, yd_w)
        s_W = mw.predict_proba(np.array([[r[f] for f in F_A4] for r in test], float))[:, 1]
    else:
        mw = HistGradientBoostingClassifier(**bestW[1]); mw.fit(Xd_w, yd_w)
        s_W = mw.predict_proba(np.array([[r[f] for f in F_A4] for r in test], float))[:, 1]
    Xd_W_for_F = np.array([[r[f] for f in F_A4] for r in dev], float)  # full dev for fusion

    # ---- T and F arms (TF-IDF on DEV prefixes) --------------------------------------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
    Xt_dev = tfidf.fit_transform([r["text"] for r in dev])
    Xt_test = tfidf.transform([r["text"] for r in test])
    Xd_oh = np.array(onehots(dev)); Xte_oh = np.array(onehots(test))
    Xd_Wfull = np.array([[r[f] for f in F_A4] for r in dev], float)
    Xte_Wfull = np.array([[r[f] for f in F_A4] for r in test], float)

    XT_dev = hstack([Xt_dev, csr_matrix(Xd_oh)]).tocsr()
    XT_test = hstack([Xt_test, csr_matrix(Xte_oh)]).tocsr()
    XF_dev = hstack([Xt_dev, csr_matrix(Xd_oh), csr_matrix(Xd_Wfull)]).tocsr()
    XF_test = hstack([Xt_test, csr_matrix(Xte_oh), csr_matrix(Xte_Wfull)]).tocsr()

    groups_dev = np.array([r["task_id"] for r in dev])
    ydev = np.array([r["y_fail"] for r in dev])
    C_T, cvT = pick_C(XT_dev, ydev, groups_dev)
    C_F, cvF = pick_C(XF_dev, ydev, groups_dev)
    mT = LogisticRegression(max_iter=2000, C=C_T); mT.fit(XT_dev, ydev)
    mF = LogisticRegression(max_iter=2000, C=C_F); mF.fit(XF_dev, ydev)
    s_T = mT.predict_proba(XT_test)[:, 1]
    s_F = mF.predict_proba(XF_test)[:, 1]
    print(f"C picks: T={C_T} (cv {max(c['dev_cv_auc'] for c in cvT)}), "
          f"F={C_F} (cv {max(c['dev_cv_auc'] for c in cvF)})")

    arms = {"T": s_T, "W": s_W, "F": s_F}
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

    # ---- paired cluster bootstrap F - T (mean k4-9 and final turn) -------------------
    task_ids = sorted({r["task_id"] for r in test})
    by_task = {}
    for i, r in enumerate(test):
        by_task.setdefault(r["task_id"], []).append(i)
    ks_rng = [c["k"] for c in curve if 4 <= c["k"] <= 9]
    rng = np.random.default_rng(SEED)
    d_mean, d_fin = [], []
    for b in range(1000):
        pt = rng.choice(len(task_ids), size=len(task_ids), replace=True)
        idx = np.concatenate([by_task[task_ids[t]] for t in pt])
        d_turn = []
        for k in ks_rng:
            sub = idx[np.isin(idx, test_by_k.get(k, []))] if k in test_by_k else np.array([], int)
            if len(sub) < 30:
                continue
            yk = ytest[sub]
            af, at = safe_auc(yk, s_F[sub]), safe_auc(yk, s_T[sub])
            if not (np.isnan(af) or np.isnan(at)):
                d_turn.append(af - at)
        if d_turn:
            d_mean.append(float(np.mean(d_turn)))
        subf = idx[np.isin(idx, fin_idx)]
        ysub = ytest[subf]
        af, at = safe_auc(ysub, s_F[subf]), safe_auc(ysub, s_T[subf])
        if not (np.isnan(af) or np.isnan(at)):
            d_fin.append(af - at)
    d_mean, d_fin = np.array(d_mean), np.array(d_fin)
    ci_mean = [float(np.percentile(d_mean, 2.5)), float(np.percentile(d_mean, 97.5))]
    ci_fin = [float(np.percentile(d_fin, 2.5)), float(np.percentile(d_fin, 97.5))]

    preds = {
        "P-FUSE-TURN": {"bar": "mean k4-9 AUC(F)-AUC(T) >= +0.01 AND CI excludes 0",
                        "mean_T": mean_auc["T"], "mean_F": mean_auc["F"],
                        "delta_mean": float(d_mean.mean()), "delta_ci95": ci_mean,
                        "pass": (mean_auc["F"] - mean_auc["T"]) >= 0.01 and ci_mean[0] > 0},
        "P-FUSE-FINAL": {"bar": "final AUC(F) >= 0.80 OR delta>=+0.01 with CI excludes 0",
                         "final": final, "delta_mean": float(d_fin.mean()), "delta_ci95": ci_fin,
                         "pass": final["auroc_F"] >= 0.80 or
                                 ((final["auroc_F"] - final["auroc_T"]) >= 0.01 and ci_fin[0] > 0)},
        "P-MEAN-walk-only (unchanged bar)": {"bar": "mean k4-9 >= 0.70 (walk channel)",
                                             "value": mean_auc["W"], "pass": mean_auc["W"] >= 0.70},
        "P-FINAL-walk-only (unchanged bar)": {"bar": "final-turn >= 0.70 (walk channel)",
                                              "value": final["auroc_W"], "pass": final["auroc_W"] >= 0.70},
    }
    out = {"protocol": {"seed": SEED, "arms": "T=tfidf+oh+LR | W=A4 pooled | F=fusion, LR C by GroupKFold5",
                        "tfidf": "ngram(1,2) min_df=3 max_features=20000 fit on DEV",
                        "bootstrap": {"B": 1000, "clusters": "test task_id"},
                        "predicates": preds, "cv_T": cvT, "cv_F": cvF},
           "curve": curve, "final": final, "mean_auc_k4_9": mean_auc}
    json.dump(out, open(os.path.join(OUT, "h10d_results.json"), "w", encoding="utf-8"), indent=1)

    print("| k | n | fail% | " + " | ".join(arms) + " |")
    for c in curve:
        print(f"| {c['k']} | {c['n_open_test']} | {c['fail_rate_at_k']} | "
              + " | ".join(str(c[f'auroc_{a}']) for a in arms) + " |")
    print("FINAL:", final)
    print("MEAN k4-9:", mean_auc)
    for name, p in preds.items():
        print(f"**{name}: {'PASS' if p['pass'] else 'FAIL'}** {json.dumps(p)[:220]}")


if __name__ == "__main__":
    main()
