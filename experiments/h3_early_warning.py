"""h3_early_warning.py — H3: turn-level early-warning curve on real agent episodes.

Question: how early can a transcript-only system flag a failing episode?

METHOD (protocol predeclared):
- For each tau-bench episode, at each user-turn boundary k (prefix = everything up to
  and INCLUDING the k-th user message; nothing after it), run the VERBATIM walker
  (extract fmt='taubench' on the partial episode -> strip the injected terminal ->
  infer_states with TAUBENCH_CONFIG) and record running features:
    D_so_far, retries_so_far (tool errors), n_states_so_far, reached_close_so_far,
    terminal_so_far, turns_so_far = k.
- Label y = 1 if the episode FAILED the task (reward != 1.0) -> AUROC = risk detection.
- Models: LogisticRegression (class_weight in {None,'balanced'}, picked on DEV);
  optional HistGradientBoosting. Trained on ALL dev prefix rows (task_id even),
  evaluated per turn k on test rows (task_id odd).
- Baselines: (b1) turns_so_far alone (degenerate within fixed k by construction ->
  reported as global AUROC + documented), (b1b) n_states_so_far per k (length proxy),
  (b2) the shipped fixed rule retries_so_far >= 2 (no training) per k.
- Split by task_id parity keeps every trial of a task on one side (no leakage).
- ALL tried configurations are reported, including worse ones.

Outputs (this dir): h3_rows.json (long table), curve.json, REPORT.md.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ART = r"C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar"
ANCH = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor"
sys.path.insert(0, ART)
sys.path.insert(0, ANCH)

from trace2grammar.extract import extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from tb_to_events import _antml_markup, strip_injected_terminal  # noqa: E402

FILES = {
    ("gpt4o", "airline"): "gpt-4o-airline.json",
    ("gpt4o", "retail"): "gpt-4o-retail.json",
    ("sonnet", "airline"): "sonnet-35-new-airline.json",
    ("sonnet", "retail"): "sonnet-35-new-retail.json",
}

FEATS = ["D_so_far", "retries_so_far", "n_states_so_far", "reached_close_so_far", "terminal_so_far"]


def convert_msg(msg):
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        return {"role": "assistant", "content": _antml_markup(msg["tool_calls"]), "name": None}
    return {"role": msg.get("role"), "content": msg.get("content"), "name": msg.get("name")}


def build_rows():
    t0 = time.time()
    rows = []
    prefix_errors = 0
    n_eps = 0
    for (model, domain), fname in FILES.items():
        eps = json.load(open(os.path.join(ANCH, fname), encoding="utf-8"))
        for ep in eps:
            n_eps += 1
            label = 0 if ep.get("reward") == 1.0 else 1  # 1 = FAIL (risk)
            conv = [convert_msg(m) for m in (ep.get("traj") or [])]
            prefix = []
            k = 0
            for m in conv:
                prefix.append(m)
                if m.get("role") != "user":
                    continue
                k += 1
                obj = {"task_id": ep.get("task_id"), "reward": ep.get("reward"),
                       "info": ep.get("info"), "traj": prefix}
                try:
                    res = extract(obj, "taubench")
                    ev = strip_injected_terminal(res["events"])
                    c = infer_states(ev, TAUBENCH_CONFIG)
                except Exception:
                    prefix_errors += 1
                    continue
                walk_states = {s["state"] for s in c["walk"]}
                rows.append({
                    "id": f"{model}:{domain}:{ep.get('task_id')}:{ep.get('trial')}",
                    "model": model, "domain": domain,
                    "task_id": ep.get("task_id"), "trial": ep.get("trial"),
                    "k": k, "y_fail": label,
                    "D_so_far": c["cycle_depth"],
                    "retries_so_far": c["error_count"],
                    "n_states_so_far": len(c["walk"]),
                    "reached_close_so_far": int("q_close" in walk_states),
                    "terminal_so_far": int(c["outcome"] in
                                           ("theta_cont", "theta_aban", "theta_hand", "theta_esc")),
                })
    print(f"episodes: {n_eps} | prefix rows: {len(rows)} | prefix extraction errors: {prefix_errors} "
          f"| elapsed {time.time()-t0:.1f}s")
    return rows


def fit_pick(rows, model_cls, params_list, tag, min_k=1):
    """Fit each config on DEV, pick by mean dev AUROC (all results reported)."""
    dev = [r for r in rows if r["task_id"] % 2 == 0 and r["k"] >= min_k]
    test = [r for r in rows if r["task_id"] % 2 == 1]
    Xd = np.array([[r[f] for f in FEATS] for r in dev], dtype=float)
    yd = np.array([r["y_fail"] for r in dev])
    best, best_cfg, all_cfg = None, None, []
    for prm in params_list:
        mdl = model_cls(**prm)
        mdl.fit(Xd, yd)
        # dev AUROC on all dev rows (pooled) + final-turn dev AUROC
        a_all = safe_auc(yd, mdl.predict_proba(Xd)[:, 1])
        fin = [r for r in dev if is_final(r, dev)]
        Xf = np.array([[r[f] for f in FEATS] for r in fin], dtype=float)
        yf = np.array([r["y_fail"] for r in fin])
        a_fin = safe_auc(yf, mdl.predict_proba(Xf)[:, 1])
        all_cfg.append({"tag": tag, "params": repr(prm), "dev_auc_pooled": round(a_all, 4),
                        "dev_auc_final_turn": round(a_fin, 4)})
        if best is None or a_all > best:
            best, best_cfg = a_all, prm
    return best_cfg, all_cfg


def is_final(r, pool):
    # memo per episode: max k
    return r["k"] == FINAL_K.get(r["id"])


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    rows = build_rows()
    json.dump(rows, open(os.path.join(HERE, "h3_rows.json"), "w", encoding="utf-8"))

    global FINAL_K
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])

    dev = [r for r in rows if r["task_id"] % 2 == 0]
    test = [r for r in rows if r["task_id"] % 2 == 1]

    # ---- model selection on DEV only (all configs reported) -----------------------
    # variants (predeclared): full-prefix training vs mid-flight training (k>=3:
    # turns 1-2 carry no variance and dilute a pooled fit)
    lr_cfg, lr_report = fit_pick(rows, LogisticRegression,
                                 [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
                                 "LR")
    gbm_cfg, gbm_report = fit_pick(rows, HistGradientBoostingClassifier,
                                   [{"max_depth": d, "max_iter": 200} for d in (3, 5)],
                                   "GBM")
    lr3_cfg, lr3_report = fit_pick(rows, LogisticRegression,
                                   [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
                                   "LR(k>=3)", min_k=3)
    gbm3_cfg, gbm3_report = fit_pick(rows, HistGradientBoostingClassifier,
                                     [{"max_depth": d, "max_iter": 200} for d in (3, 5)],
                                     "GBM(k>=3)", min_k=3)
    cands = [("LR", lr_report), ("GBM", gbm_report), ("LR(k>=3)", lr3_report), ("GBM(k>=3)", gbm3_report)]
    best_name, _ = max(cands, key=lambda c: c[1][0]["dev_auc_pooled"])
    print("model families tried (dev pooled AUC):", [(c[0], c[1][0]["dev_auc_pooled"]) for c in cands],
          "-> picked", best_name)

    def train_fit(cls, cfg, min_k):
        tr = [r for r in dev if r["k"] >= min_k]
        m = cls(**cfg)
        m.fit(np.array([[r[f] for f in FEATS] for r in tr], float),
              np.array([r["y_fail"] for r in tr]))
        return m
    lr = train_fit(LogisticRegression, lr_cfg, 1)
    gbm = train_fit(HistGradientBoostingClassifier, gbm_cfg, 1)
    models = {
        "LR": lr,
        "GBM": gbm,
        "LR(k>=3)": train_fit(LogisticRegression, lr3_cfg, 3),
        "GBM(k>=3)": train_fit(HistGradientBoostingClassifier, gbm3_cfg, 3),
    }
    best_model = models[best_name]

    # ---- per-k curves on TEST -----------------------------------------------------
    ks = sorted({r["k"] for r in rows})
    curve = []
    for k in ks:
        sub = [r for r in test if r["k"] == k]
        if len(sub) < 30:
            continue  # too few open episodes to score a turn; declared, not hidden
        y = np.array([r["y_fail"] for r in sub])
        Xm = np.array([[r[f] for f in FEATS] for r in sub], float)
        row = {
            "k": k,
            "n_open_test": len(sub),
            "fail_rate_at_k": round(100 * y.mean(), 1),
            "auroc_M_best": round(safe_auc(y, best_model.predict_proba(Xm)[:, 1]), 4),
            "auroc_M_LR": round(safe_auc(y, lr.predict_proba(Xm)[:, 1]), 4),
            "auroc_M_GBM": round(safe_auc(y, gbm.predict_proba(Xm)[:, 1]), 4),
            "auroc_b1b_n_states": round(safe_auc(y, np.array([r["n_states_so_far"] for r in sub], float)), 4),
            "auroc_b2_flag": round(safe_auc(y, np.array([r["retries_so_far"] >= 2 for r in sub], float)), 4),
            "b2_n_flagged": int(sum(1 for r in sub if r["retries_so_far"] >= 2)),
            "b2_failrate_flag": round(100 * np.mean([r["y_fail"] for r in sub if r["retries_so_far"] >= 2]), 1)
                                if any(r["retries_so_far"] >= 2 for r in sub) else None,
            "b2_failrate_unflag": round(100 * np.mean([r["y_fail"] for r in sub if r["retries_so_far"] < 2]), 1)
                                  if any(r["retries_so_far"] < 2 for r in sub) else None,
        }
        curve.append(row)

    # final-turn reference (each episode at its own last turn, test)
    fin = [r for r in test if r["k"] == FINAL_K[r["id"]]]
    y = np.array([r["y_fail"] for r in fin])
    Xm = np.array([[r[f] for f in FEATS] for r in fin], float)
    final = {
        "n_test_episodes": len(fin),
        "fail_rate": round(100 * float(y.mean()), 1),
        "picked_model": best_name,
        "auroc_M_best": round(safe_auc(y, best_model.predict_proba(Xm)[:, 1]), 4),
        "auroc_M_LR": round(safe_auc(y, lr.predict_proba(Xm)[:, 1]), 4),
        "auroc_M_GBM": round(safe_auc(y, gbm.predict_proba(Xm)[:, 1]), 4),
        "auroc_b1_turns_global": None,
        "auroc_b1b_n_states_final": round(safe_auc(y, np.array([r["n_states_so_far"] for r in fin], float)), 4),
        "auroc_b2_flag_final": round(safe_auc(y, np.array([r["retries_so_far"] >= 2 for r in fin], float)), 4),
    }
    # b1 global (documented degeneracy: turns_so_far is constant within a fixed k)
    y_all = np.array([r["y_fail"] for r in test])
    t_all = np.array([r["k"] for r in test], float)
    final["auroc_b1_turns_global"] = round(safe_auc(y_all, t_all), 4)

    json.dump({"curve": curve, "final": final,
               "model_select": {"LR": lr_report, "GBM": gbm_report,
                                "LR(k>=3)": lr3_report, "GBM(k>=3)": gbm3_report,
                                "picked": best_name},
               "features": FEATS, "n_rows": len(rows), "n_dev": len(dev), "n_test": len(test)},
              open(os.path.join(HERE, "curve.json"), "w", encoding="utf-8"), indent=1)

    # ---- REPORT.md -----------------------------------------------------------------
    lines = ["# H3 report: turn-level early warning on 1,980 real tau-bench episodes", ""]
    lines.append(f"- prefix rows: {len(rows)} (dev {len(dev)} / test {len(test)}; split by task_id parity, "
                 "all trials of a task stay on one side)")
    lines.append(f"- label y=1 = episode FAILED (reward!=1); test fail base rate at final turn: {final['fail_rate']}%")
    lines.append("- model: GBM and LR on running walk features; family+variant picked on DEV (four families "
                 "tested: LR/GBM x full-prefix/k>=3 training; all results in curve.json.model_select)")
    lines.append("- baseline b1 (turns_so_far) is degenerate within a fixed k (constant for all open "
                 "episodes): reported as a single global AUROC, not a curve")
    lines.append("")
    lines.append("## per-turn curve (TEST)")
    lines.append(f"- picked model (by dev pooled AUC): **{best_name}**")
    lines.append("| k | n_open | fail% | AUC M(best) | AUC M(LR) | AUC M(GBM) | AUC b1b(len) | AUC b2(flag) | b2 flag fail% vs unflag% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in curve:
        b2 = "-" if c["b2_failrate_flag"] is None else f"{c['b2_failrate_flag']} vs {c['b2_failrate_unflag']} (n_flag={c['b2_n_flagged']})"
        lines.append(f"| {c['k']} | {c['n_open_test']} | {c['fail_rate_at_k']} | {c['auroc_M_best']} | {c['auroc_M_LR']} | "
                     f"{c['auroc_M_GBM']} | {c['auroc_b1b_n_states']} | {c['auroc_b2_flag']} | {b2} |")
    lines.append("")
    lines.append(f"## final-turn reference (each episode at its last turn, test): n={final['n_test_episodes']}, "
                 f"fail_rate={final['fail_rate']}%, AUC M(best)={final['auroc_M_best']}, "
                 f"M(LR)={final['auroc_M_LR']}, M(GBM)={final['auroc_M_GBM']}, "
                 f"b1 turns global={final['auroc_b1_turns_global']}, b1b len={final['auroc_b1b_n_states_final']}, "
                 f"b2 flag={final['auroc_b2_flag_final']}")
    lines.append("")
    # earliest stable detection
    stab = None
    for i in range(len(curve) - 1):
        if curve[i]["auroc_M_best"] >= 0.65 and curve[i + 1]["auroc_M_best"] >= 0.65 \
                and curve[i]["n_open_test"] >= 100:
            stab = curve[i]["k"]
            break
    lines.append(f"- earliest turn with M(best) test AUROC >= 0.65 stably (k and k+1, n_open>=100): **k={stab}**")
    if curve:
        best_k = max(curve, key=lambda c: c["auroc_M_best"])
        lines.append(f"- best mid-flight turn: k={best_k['k']} AUC {best_k['auroc_M_best']}")
    lines.append("- the paper's shipped rule (retries>=2) = b2 column; compare its per-k AUC and lift "
                 "against the trained model")
    open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote curve.json + h3_rows.json + REPORT.md")


if __name__ == "__main__":
    main()
