"""h10_sota_curve.py — H10: improve the turn-level early-warning curve (SOTA push).

Mandate (user, 31-oct): keep working until we improve the SOTA. The SOTA = the h3
turn-level curve (the paper's headline): first stable AUROC>=0.65 at k=7, mean k4-9
0.6459, final-turn 0.6477 (5 walk-running features).

New edge from t2g-replay (DONE 6/6): tau-bench failures are OMISSIONS (missing
writes, env-verified by replay). The h3 features count ERRORS, not MISSING WRITES,
so they cannot see the silent-failure family. H10 adds the write channel +
close_no_write omission flag as RUNNING structural features (no text).

PREDICATES PREDECLARED (knowledge_base/threads/t2g-sota/PLAN.md — before running):
- P-EARLY: exists k<=6 with test AUROC >=0.65 at k AND k+1, n_open>=100 both
  (h3: first stable at k=7).
- P-MEAN: mean test AUROC over k in {4..9} >= 0.70 (h3: 0.6459) AND cluster
  bootstrap (1000x, clusters=test task_id) 95% CI of the paired delta
  (best arm - A0 anchor) excludes 0.
- P-FINAL: final-turn test AUROC >= 0.70 (h3 anchor 0.6477; content-blind 0.6327).

PAIRED ARMS (same seed 20261031, same split task-parity dev=even/test=odd):
- A0 = h3's original 5 features (anchor reproduction)
- A1 = A0 + write channel (n_tool_calls, n_tool_ok, writes_so_far, has_write,
  write_ok_so_far, n_auth_calls)
- A2 = A1 + omission flags (close_no_write, last_tool_is_write)
- A3 = A2 + k + per-state visit counts + distinct_tools + n_think_calls
Models GBM+LR, variant pick on dev pooled AUC exactly like h3 (min_k in {1,3});
ALL configs reported.

Outputs (runtime/h10_sota_curve/): h10_rows.json, results.json, REPORT.md.
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
OUT = r"C:/Users/U77225D/Desktop/proyectos/hf/runtime/h10_sota_curve"
sys.path.insert(0, ART)
sys.path.insert(0, ANCH)

from trace2grammar.extract import extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from tb_to_events import _antml_markup, strip_injected_terminal  # noqa: E402

SEED = 20261031
FILES = {
    ("gpt4o", "airline"): "gpt-4o-airline.json",
    ("gpt4o", "retail"): "gpt-4o-retail.json",
    ("sonnet", "airline"): "sonnet-35-new-airline.json",
    ("sonnet", "retail"): "sonnet-35-new-retail.json",
}

# predeclared write/auth patterns per domain (from the shipped tool vocabulary)
WRITES = {
    "retail": ("modify_", "cancel_pending_order", "exchange_delivered_order_items",
               "return_delivered_order_items", "modify_user_address"),
    "airline": ("update_reservation", "cancel_reservation", "book_reservation",
                "send_certificate"),
}
AUTH = ("get_user_details", "find_user_id_by_email", "find_user_id_by_name_zip")
STATE_VOCAB = ["q_greet", "q_intent", "q_auth", "q_tool", "q_close", "q_confirm"]

F_A0 = ["D_so_far", "retries_so_far", "n_states_so_far", "reached_close_so_far", "terminal_so_far"]
F_A1 = F_A0 + ["n_tool_calls", "n_tool_ok", "writes_so_far", "has_write", "write_ok_so_far",
               "n_auth_calls"]
F_A2 = F_A1 + ["close_no_write", "last_tool_is_write"]
F_A3 = F_A2 + ["k", "distinct_tools", "n_think_calls"] + [f"vis_{s}" for s in STATE_VOCAB]
ARMS = {"A0": F_A0, "A1": F_A1, "A2": F_A2, "A3": F_A3}


def convert_msg(msg):
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        return {"role": "assistant", "content": _antml_markup(msg["tool_calls"]), "name": None}
    return {"role": msg.get("role"), "content": msg.get("content"), "name": msg.get("name")}


def is_write(name, domain):
    return any(name.startswith(p) or name == p for p in WRITES[domain])


def build_rows():
    t0 = time.time()
    rows = []
    n_eps = prefix_errors = 0
    for (model, domain), fname in FILES.items():
        eps = json.load(open(os.path.join(ANCH, fname), encoding="utf-8"))
        for ep in eps:
            n_eps += 1
            label = 0 if ep.get("reward") == 1.0 else 1
            conv = [convert_msg(m) for m in (ep.get("traj") or [])]
            prefix, k = [], 0
            prefix_text = []
            for m in conv:
                prefix.append(m)
                if m.get("role") != "system":
                    prefix_text.append(str(m.get("content", "")))
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
                walk_states = [s["state"] for s in c["walk"]]
                wset = set(walk_states)
                tool_calls = [e for e in ev if e["kind"] == "tool_call"]
                tool_names = [e["name"] or "" for e in tool_calls]
                writes = [n for n in tool_names if is_write(n, domain)]
                write_ok = sum(1 for e in ev if e["kind"] == "tool_result" and is_write(e.get("name") or "", domain))
                vis = {s: walk_states.count(s) for s in STATE_VOCAB}
                rows.append({
                    "id": f"{model}:{domain}:{ep.get('task_id')}:{ep.get('trial')}",
                    "text": "\n".join(prefix_text),
                    "model": model, "domain": domain,
                    "task_id": ep.get("task_id"), "trial": ep.get("trial"),
                    "k": k, "y_fail": label,
                    "D_so_far": c["cycle_depth"],
                    "retries_so_far": c["error_count"],
                    "n_states_so_far": len(c["walk"]),
                    "reached_close_so_far": int("q_close" in wset),
                    "terminal_so_far": int(c["outcome"] in
                                           ("theta_cont", "theta_aban", "theta_hand", "theta_esc")),
                    "n_tool_calls": len(tool_calls),
                    "n_tool_ok": sum(1 for e in ev if e["kind"] == "tool_result"),
                    "writes_so_far": len(writes),
                    "has_write": int(len(writes) > 0),
                    "write_ok_so_far": write_ok,
                    "n_auth_calls": sum(1 for n in tool_names if n in AUTH),
                    "close_no_write": int(("q_close" in wset) and len(writes) == 0),
                    "last_tool_is_write": int(bool(tool_names) and is_write(tool_names[-1], domain)),
                    "distinct_tools": len(set(tool_names)),
                    "n_think_calls": sum(1 for n in tool_names if n == "think"),
                    "n_assistant_msgs": sum(1 for e in ev if e["kind"] == "assistant_msg"),
                    "tool_rate": len(tool_calls) / max(1, k),
                    "write_ratio": len(writes) / max(1, len(tool_calls)),
                    "auth_ok_so_far": sum(1 for e in ev if e["kind"] == "tool_result" and (e.get("name") or "") in AUTH),
                    "err_rate": c["error_count"] / max(1, k),
                    **({"turns_since_last_tool": sum(1 for e in ev[last_tool_i + 1:] if e["kind"] == "user_msg")
                        if (last_tool_i := max((i for i, e in enumerate(ev) if e["kind"] == "tool_call"), default=-1)) >= 0
                        else k}),
                    **{f"vis_{s}": vis[s] for s in STATE_VOCAB},
                })
    print(f"episodes: {n_eps} | prefix rows: {len(rows)} | prefix errors: {prefix_errors} "
          f"| elapsed {time.time()-t0:.1f}s")
    return rows


def safe_auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fit_arm(rows, feats, tag, min_k):
    """h3's selection protocol verbatim: fit both class_weights (LR) / depths (GBM),
    pick by dev pooled AUROC. Returns (pick, report)."""
    dev = [r for r in rows if r["task_id"] % 2 == 0 and r["k"] >= min_k]
    Xd = np.array([[r[f] for f in feats] for r in dev], float)
    yd = np.array([r["y_fail"] for r in dev])
    grids = {
        "LR": [{"max_iter": 2000, "class_weight": w} for w in (None, "balanced")],
        "GBM": [{"max_depth": d, "max_iter": 200, "random_state": SEED} for d in (3, 5)],
    }
    cands = []
    for fam, grid in grids.items():
        for p in grid:
            m = (LogisticRegression if fam == "LR" else HistGradientBoostingClassifier)(**p)
            m.fit(Xd, yd)
            cands.append((fam, p, safe_auc(yd, m.predict_proba(Xd)[:, 1])))
    best = max(cands, key=lambda c: c[2])
    report = [{"fam": f, "params": repr(p), "dev_auc_pooled": round(a, 4)} for f, p, a in cands]
    return {"fam": best[0], "params": best[1], "min_k": min_k}, report


def train(rows, feats, fam, prm, min_k):
    dev = [r for r in rows if r["task_id"] % 2 == 0 and r["k"] >= min_k]
    m = (LogisticRegression if fam == "LR" else HistGradientBoostingClassifier)(**prm)
    m.fit(np.array([[r[f] for f in feats] for r in dev], float),
          np.array([r["y_fail"] for r in dev]))
    return m


def score_all(rows, feats, model):
    return model.predict_proba(np.array([[r[f] for f in feats] for r in rows], float))[:, 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "h10_rows.json")
    if os.path.exists(rows_path):
        rows = json.load(open(rows_path, encoding="utf-8"))
        print(f"loaded cached rows: {len(rows)}")
    else:
        rows = build_rows()
        json.dump(rows, open(rows_path, "w", encoding="utf-8"))

    test = [r for r in rows if r["task_id"] % 2 == 1]
    ks = sorted({r["k"] for r in rows if r["k"] <= 20})
    FINAL_K = {}
    for r in rows:
        FINAL_K[r["id"]] = max(FINAL_K.get(r["id"], 0), r["k"])

    # ---- arms: pick config on dev, train, score test rows -------------------------
    arm_pick, arm_report, arm_scores = {}, {}, {}
    for arm, feats in ARMS.items():
        # evaluate both min_k variants, keep the dev-better one
        cands = []
        for min_k in (1, 3):
            p, rep = fit_arm(rows, feats, arm, min_k)
            dev_auc = max(c["dev_auc_pooled"] for c in rep)
            cands.append((dev_auc, min_k, p, rep))
        cands.sort(key=lambda c: -c[0])
        dev_auc, min_k, pick, rep = cands[0]
        arm_pick[arm] = {"dev_auc_pooled": dev_auc, **pick}
        arm_report[arm] = [{**c, "min_k": mk} for c in cands[0][3] for mk in (min_k,)][:0] + \
                          [dict(min_k=min_k, **c) for c in cands[0][3]] + \
                          [dict(min_k=mk, **c) for _, mk, _, rep2 in cands[1:] for c in rep2]
        m = train(rows, feats, pick["fam"], pick["params"], min_k)
        arm_scores[arm] = score_all(test, feats, m)
        print(f"{arm}: pick={pick['fam']}{pick['params']} min_k={min_k} dev={dev_auc:.4f}")

    # ---- per-k test curve ----------------------------------------------------------
    test_by_k = {}
    for i, r in enumerate(test):
        test_by_k.setdefault(r["k"], []).append(i)
    ytest = np.array([r["y_fail"] for r in test])
    curve = []
    for k in ks:
        idx = test_by_k.get(k, [])
        if len(idx) < 30:
            continue
        yk = ytest[idx]
        row = {"k": k, "n_open_test": len(idx), "fail_rate_at_k": round(100 * yk.mean(), 1)}
        for arm in ARMS:
            row[f"auroc_{arm}"] = round(safe_auc(yk, arm_scores[arm][idx]), 4)
        curve.append(row)

    # final-turn reference
    fin_idx = [i for i, r in enumerate(test) if r["k"] == FINAL_K[r["id"]]]
    yf = ytest[fin_idx]
    final = {"n_test_episodes": len(fin_idx), "fail_rate": round(100 * yf.mean(), 1)}
    for arm in ARMS:
        final[f"auroc_{arm}"] = round(safe_auc(yf, arm_scores[arm][fin_idx]), 4)

    def mean_k(curve_or_scores, arm, kfrom=4, kto=9, per_turn=True):
        vals = []
        for c in curve_or_scores:
            if kfrom <= c["k"] <= kto and c["n_open_test"] >= 30:
                vals.append(c[f"auroc_{arm}"] if per_turn else None)
        return vals

    mean_auc = {arm: float(np.mean(mean_k(curve, arm))) for arm in ARMS}
    early_stable = None
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        if a["k"] <= 6 and a[f"auroc_A3"] >= 0.65 and b[f"auroc_A3"] >= 0.65 \
                and a["n_open_test"] >= 100 and b["n_open_test"] >= 100:
            early_stable = a["k"]
            break
    # also check all arms for P-EARLY
    early_by_arm = {}
    for arm in ARMS:
        es = None
        for i in range(len(curve) - 1):
            a, b = curve[i], curve[i + 1]
            if a["k"] <= 6 and a[f"auroc_{arm}"] >= 0.65 and b[f"auroc_{arm}"] >= 0.65 \
                    and a["n_open_test"] >= 100 and b["n_open_test"] >= 100:
                es = a["k"]
                break
        early_by_arm[arm] = es

    # ---- paired cluster bootstrap: mean-k4-9 delta vs A0 ----------------------------
    # clusters = test task_ids; resample clusters with replacement, recompute per-turn
    # AUCs for the best non-A0 arm and A0, mean over k4..9, delta.
    task_ids = sorted({r["task_id"] for r in test})
    by_task = {}
    for i, r in enumerate(test):
        by_task.setdefault(r["task_id"], []).append(i)
    best_arm = max([a for a in ARMS if a != "A0"], key=lambda a: mean_auc[a])
    rng = np.random.default_rng(SEED)
    B = 1000
    deltas = []
    ks_rng = [c["k"] for c in curve if 4 <= c["k"] <= 9]
    for b in range(B):
        pick_tasks = rng.choice(len(task_ids), size=len(task_ids), replace=True)
        idx = np.concatenate([by_task[task_ids[t]] for t in pick_tasks])
        d_turn = []
        for k in ks_rng:
            sub = idx[np.isin(idx, test_by_k.get(k, []))] if k in test_by_k else np.array([], int)
            if len(sub) < 30:
                continue
            yk = ytest[sub]
            a_best = safe_auc(yk, arm_scores[best_arm][sub])
            a_a0 = safe_auc(yk, arm_scores["A0"][sub])
            if not (np.isnan(a_best) or np.isnan(a_a0)):
                d_turn.append(a_best - a_a0)
        if d_turn:
            deltas.append(float(np.mean(d_turn)))
    deltas = np.array(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

    # ---- predicates -----------------------------------------------------------------
    preds = {
        "P-EARLY": {"bar": "stable AUC>=0.65 at k<=6 (k,k+1, n>=100) | h3: k=7",
                    "value": early_by_arm, "pass": any(v is not None for v in early_by_arm.values())},
        "P-MEAN": {"bar": "mean k4-9 >= 0.70 (h3: 0.6459)",
                   "value": mean_auc, "best_arm": best_arm,
                   "delta_vs_A0_mean": float(deltas.mean()), "delta_ci95": ci,
                   "pass": mean_auc[best_arm] >= 0.70 and ci[0] > 0},
        "P-FINAL": {"bar": "final-turn AUC >= 0.70 (h3: 0.6477, content-blind: 0.6327)",
                    "value": final, "pass": max(final[f"auroc_{a}"] for a in ARMS if a != "A0") >= 0.70},
    }

    out = {"protocol": {"seed": SEED, "split": "task_id parity (dev=even, test=odd)",
                        "selection": "dev pooled AUC, h3 protocol verbatim (LR/GBM x min_k 1/3)",
                        "arms": ARMS, "bootstrap": {"B": B, "clusters": "test task_id"},
                        "predicates": preds},
           "model_pick": arm_pick, "curve": curve, "final": final,
           "mean_auc_k4_9": mean_auc, "early_stable_best_arm": early_by_arm}
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)
    json.dump(arm_report, open(os.path.join(OUT, "model_select.json"), "w", encoding="utf-8"), indent=1)

    lines = ["# H10 report: write-channel + omission flag on the early-warning curve", ""]
    lines.append(f"- rows: {len(rows)} (test episodes {final['n_test_episodes']}, fail {final['fail_rate']}%)")
    lines.append(f"- picks: {json.dumps(arm_pick)}")
    lines.append("")
    lines.append("| k | n_open | fail% | " + " | ".join(f"AUC {a}" for a in ARMS) + " |")
    lines.append("|---|---|---|" + "---|" * len(ARMS))
    for c in curve:
        lines.append(f"| {c['k']} | {c['n_open_test']} | {c['fail_rate_at_k']} | "
                     + " | ".join(str(c[f'auroc_{a}']) for a in ARMS) + " |")
    lines.append("")
    lines.append(f"- final-turn: {json.dumps(final)}")
    lines.append(f"- mean k4-9: {json.dumps(mean_auc)}")
    lines.append(f"- P-EARLY per arm: {json.dumps(early_by_arm)}")
    lines.append(f"- P-MEAN delta ({best_arm} - A0): mean {deltas.mean():.4f} CI95 {ci}")
    for name, p in preds.items():
        lines.append(f"- **{name}: {'PASS' if p['pass'] else 'FAIL'}** ({p['bar']})")
    open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[-6:]))
    print("wrote results.json + model_select.json + REPORT.md")


if __name__ == "__main__":
    main()
