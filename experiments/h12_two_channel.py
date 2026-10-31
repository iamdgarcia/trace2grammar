"""h12_two_channel.py — iteration 9: complete the two-channel decomposition.

Predeclared (knowledge_base/threads/t2g-sota/PLAN.md iter-9): does the WALK channel
(A4, 0.69 cross-task) carry per-episode evidence like the text channel does
(0.6409@k1 out-of-sample), or is it pure task prior? And does any channel carry
within-task signal on the silent (write-check) family, where text is exactly at
chance (0.5000) but the structural omission flag was causally actionable (2/2
env-verified flips)?

Protocol:
- A4 arm verbatim (h10c iter-3: grid GBM d3-5 x it200/400 + LR, min_k=3, seed
  20261031, dev fit task-parity) -> walk score for ALL 14,011 rows.
- Text channel refit (h11b protocol, deterministic) for the per-family comparison.
- Within-task, TEST tasks only, >=2-trials-of-both-outcomes rule (predeclared):
  final turn + per-turn curve for BOTH channels; omission flag as a score.
- Both bootstrap estimands (mean-per-task and pooled-AUC cluster, 1000x), declared
  context. Reading (predeclared): prior-dominated if the mean-per-task CI includes
  0.5; episode-bearing if its lower bound > 0.5.

Output: runtime/h10_sota_curve/h12_results.json.
"""
import json
import os

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def within_task(rows, score_key, min_trials=2):
    """Per-task eligible AUCs at final turn + per-turn curve. Test tasks only is the
    caller's job."""
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])
    # final turn per task
    pt = {}
    for r in rows:
        if r["k"] == FINAL_K[r["id"]]:
            pt.setdefault(r["task_id"], []).append(r)
    elig = {t: g for t, g in pt.items()
            if len(g) >= min_trials and len({x["y_fail"] for x in g}) == 2}
    aucs = [safe_auc([x["y_fail"] for x in g], [x[score_key] for x in g]) for g in elig.values()]
    aucs = [a for a in aucs if not np.isnan(a)]
    final = {"n_tasks": len(elig), "n_episodes": sum(len(g) for g in elig.values()),
             "mean_per_task_auc": round(float(np.mean(aucs)), 4) if aucs else None}
    # per-turn curve
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)
    curve = {}
    for k, ix in sorted(by_k.items()):
        pt_k = {}
        for r in ix:
            pt_k.setdefault(r["task_id"], []).append(r)
        el = [g for g in pt_k.values()
              if len(g) >= min_trials and len({x["y_fail"] for x in g}) == 2]
        aa = [safe_auc([x["y_fail"] for x in g], [x[score_key] for x in g]) for g in el]
        aa = [a for a in aa if not np.isnan(a)]
        if aa:
            curve[k] = {"auc": round(float(np.mean(aa)), 4), "n_tasks": len(el)}
    return final, curve, elig


def boot_cis(elig, score_key, B=1000, seed=SEED):
    """Both estimands over resampled tasks."""
    groups = {t: [(x["y_fail"], x[score_key]) for x in g] for t, g in elig.items()}
    per_task = [safe_auc([y for y, _ in g], [s for _, s in g]) for g in groups.values()]
    per_task = [a for a in per_task if not np.isnan(a)]
    rng = np.random.default_rng(seed)
    tlist = list(groups.values())
    m_means, p_aucs = [], []
    for _ in range(B):
        pick = rng.choice(len(tlist), size=len(tlist), replace=True)
        m_means.append(float(np.mean([per_task[t] for t in pick])))
        ys = [y for t in pick for y, _ in tlist[t]]
        ss = [s for t in pick for _, s in tlist[t]]
        a = safe_auc(ys, ss)
        if not np.isnan(a):
            p_aucs.append(a)
    return {"mean_per_task_ci95": [round(float(np.percentile(m_means, 2.5)), 4),
                                   round(float(np.percentile(m_means, 97.5)), 4)],
            "pooled_auc_ci95": [round(float(np.percentile(p_aucs, 2.5)), 4),
                                round(float(np.percentile(p_aucs, 97.5)), 4)]}


def main():
    rows = json.load(open(os.path.join(OUT, "h10_rows.json"), encoding="utf-8"))
    dev = [r for r in rows if r["task_id"] % 2 == 0]
    ydev = np.array([r["y_fail"] for r in dev])

    # --- walk channel: A4 verbatim (h10c) ---
    dev_k3 = [r for r in dev if r["k"] >= 3]
    Xd = np.array([[r[f] for f in F_A4] for r in dev_k3], float)
    yd = np.array([r["y_fail"] for r in dev_k3])
    grids = ([{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")] +
             [{"max_depth": d, "max_iter": it, "random_state": SEED}
              for d in (3, 4, 5) for it in (200, 400)])
    cands = []
    for p in grids:
        cls = LogisticRegression if "class_weight" in p else HistGradientBoostingClassifier
        mm = cls(**p); mm.fit(Xd, yd)
        cands.append(("LR" if "class_weight" in p else "GBM", p,
                      safe_auc(yd, mm.predict_proba(Xd)[:, 1])))
    fam, prm, _ = max(cands, key=lambda c: c[2])
    cls = LogisticRegression if fam == "LR" else HistGradientBoostingClassifier
    mw = cls(**prm); mw.fit(Xd, yd)
    Xa = np.array([[r[f] for f in F_A4] for r in rows], float)
    for r, s in zip(rows, mw.predict_proba(Xa)[:, 1]):
        r["w_score"] = float(s)
    print(f"walk arm pick: {fam} {prm}")

    # --- text channel: h11b protocol verbatim ---
    Xt = None
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
    Xt = tfidf.fit_transform([r["text"] for r in dev])
    oh = np.array([[1.0 if r["model"] == "sonnet" else 0.0,
                    1.0 if r["domain"] == "airline" else 0.0] for r in dev])
    mt = LogisticRegression(max_iter=2000, C=0.1).fit(hstack([Xt, csr_matrix(oh)]).tocsr(), ydev)
    oh_all = np.array([[1.0 if r["model"] == "sonnet" else 0.0,
                        1.0 if r["domain"] == "airline" else 0.0] for r in rows])
    Xtt = hstack([tfidf.transform([r["text"] for r in rows]), csr_matrix(oh_all)]).tocsr()
    for r, s in zip(rows, mt.predict_proba(Xtt)[:, 1]):
        r["t_score"] = float(s)

    e3 = json.load(open(r"C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar"
                        "/experiments/latent2/e3_results.json", encoding="utf-8"))
    fam_map = {}
    for p in e3["pairs"]:
        f = "divergent" if p.get("tau_sem") is not None else "silent"
        for ep_id in p["pair"]:
            fam_map[ep_id] = f

    test_rows = [r for r in rows if r["task_id"] % 2 == 1]

    # --- within-task: both channels, test only ---
    res = {"protocol": {"seed": SEED, "walk_pick": f"{fam} {prm}",
                        "rule": ">=2 trials both outcomes, test tasks only, final turn",
                        "bootstrap": "both estimands, 1000x, clusters=test tasks",
                        "reading": "prior-dominated if mean-per-task CI includes 0.5; "
                                   "episode-bearing if lower bound > 0.5"},
           "channels": {}, "families": {}, "flag_on_silent": {}}
    for ch, key in (("walk", "w_score"), ("text", "t_score")):
        final, curve, elig = within_task(test_rows, key)
        cis = boot_cis(elig, key)
        lo = cis["mean_per_task_ci95"][0]
        res["channels"][ch] = {**final, **cis,
                               "per_turn": curve,
                               "verdict": "episode-bearing" if lo > 0.5 else "prior-dominated"}
        print(f"{ch}: final within {final['mean_per_task_auc']} n_tasks={final['n_tasks']} "
              f"CI {cis['mean_per_task_ci95']} | pooled CI {cis['pooled_auc_ci95']} -> "
              f"{res['channels'][ch]['verdict']}")

    # --- per-family within-task (test only), both channels + omission flag ---
    for f in ("divergent", "silent"):
        sub = [r for r in test_rows if fam_map.get(r["id"]) == f]
        row_f = {}
        for ch, key in (("walk", "w_score"), ("text", "t_score")):
            fin, _, elig = within_task(sub, key)
            cis = boot_cis(elig, key)
            row_f[ch] = {**fin, **cis}
        # omission flag as score on this family
        fin, _, elig = within_task(sub, "close_no_write")
        cis = boot_cis(elig, "close_no_write")
        row_f["close_no_write_flag"] = {**fin, **cis}
        res["families"][f] = row_f
        print(f"[{f}] walk {row_f['walk']['mean_per_task_auc']} "
              f"(CI {row_f['walk']['mean_per_task_ci95']}) | text "
              f"{row_f['text']['mean_per_task_auc']} (CI {row_f['text']['mean_per_task_ci95']}) "
              f"| flag {row_f['close_no_write_flag']['mean_per_task_auc']} "
              f"(CI {row_f['close_no_write_flag']['mean_per_task_ci95']}) n_tasks="
              f"{row_f['walk']['n_tasks']}")

    json.dump(res, open(os.path.join(OUT, "h12_results.json"), "w", encoding="utf-8"), indent=1)
    print("wrote h12_results.json")


if __name__ == "__main__":
    main()
