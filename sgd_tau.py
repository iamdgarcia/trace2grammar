#!/usr/bin/env python3
"""Run tau (extraction) verbatim over the 4,201 SGD dstc8 test dialogues and
validate the atlas claims against the dataset's structural ground truth.

Same contract as multiwoz_tau.py (23-oct): infer_states()/extract() from
builds/trace2grammar/trace2grammar/extract.py runs VERBATIM (zero changes, zero
table tuning). New here: SGD carries structural ground truth (service_call with
non-empty service_results = the simulated API actually returned data), so the
walker's verdict can be confronted with a third-party truth for the first time.

Validations:
  A. mapping rate: dialogues extracted without exception / total.
  B. verdict vs structural truth: contained-vs-gt, abandoned-vs-gt agreement,
     first disagreement cases quoted verbatim for manual audit.
  C. close rates per service (the MultiWOZ signature: rates separate domains).
  D. corpus signature: handoff==0 and D==0 expected (wizard-completed corpus).

Output: sgd_validation/results.json + sgd_validation/REPORT.md
"""
import json
import os
import sys
from collections import Counter

# artifact package root: scripts live at the repo root; out-of-repo layouts
# set the T2G_ROOT env var to the artifact checkout
ROOT = os.environ.get("T2G_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "sgd_test_transcript")
meta = json.load(open(os.path.join(SRC, "_meta.json"), encoding="utf-8"))

rows = []
errors = []
for did, m in sorted(meta.items()):
    txt = open(os.path.join(SRC, f"{did}.txt"), encoding="utf-8").read()
    try:
        r = extract(txt, "transcript")
    except Exception as e:  # noqa: BLE001 - mapping-rate accounting, not swallowing
        errors.append({"id": did, "error": repr(e)[:200]})
        continue
    walk = r["walk"]
    reached_close = any(w["state"] == "q_close" for w in walk)
    rows.append({
        "id": did,
        "primary": m["primary"],
        "services": m["services"],
        "outcome": r["outcome"],
        "D": r["cycle_depth"],
        "n_states": len(walk),
        "reached_close": reached_close,
        "gt_effective_call": m["gt_effective_call"],
        "n_service_calls": m["n_service_calls"],
        "n_effective": m["n_effective"],
    })

n_total = len(meta)
n_ok = len(rows)
mapping_rate = 100.0 * n_ok / n_total

# --- B: verdict vs structural truth ----------------------------------------
# Contingency of walker verdict (contained/abandoned) against gt_effective_call.
conf = Counter((r["outcome"], r["gt_effective_call"]) for r in rows)
agree = sum(v for (o, g), v in conf.items()
            if (o == "theta_cont") == g)
agreement_pct = 100.0 * agree / n_ok
dis = [r for r in rows if (r["outcome"] == "theta_cont") != r["gt_effective_call"]]
dis_first = []
for r in dis[:5]:
    txt = open(os.path.join(SRC, f"{r['id']}.txt"), encoding="utf-8").read()
    dis_first.append({
        "id": r["id"], "outcome": r["outcome"], "gt": r["gt_effective_call"],
        "n_calls": r["n_service_calls"], "n_effective": r["n_effective"],
        "reached_close": r["reached_close"],
        "last_lines": "\n".join(txt.strip().splitlines()[-4:])[:400],
    })

# --- C: per-service table ---------------------------------------------------
by_dom = {}
for r in rows:
    by_dom.setdefault(r["primary"], []).append(r)
table = []
for dom, rs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
    oc = Counter(x["outcome"] for x in rs)
    table.append({
        "service": dom, "n": len(rs),
        "contained": round(100.0 * oc.get("theta_cont", 0) / len(rs), 1),
        "abandoned": round(100.0 * oc.get("theta_aban", 0) / len(rs), 1),
        "handoff": round(100.0 * oc.get("theta_hand", 0) / len(rs), 1),
        "D_mean": round(sum(x["D"] for x in rs) / len(rs), 2),
        "D_max": max(x["D"] for x in rs),
        "close_rate": round(100.0 * sum(1 for x in rs if x["reached_close"]) / len(rs), 1),
        "gt_effective": round(100.0 * sum(1 for x in rs if x["gt_effective_call"]) / len(rs), 1),
    })
rates = [t["close_rate"] for t in table]
separates = (max(rates) - min(rates)) > 5.0  # same lens as MultiWOZ (56.7 vs 48.8)

# --- D: corpus signature ----------------------------------------------------
n_hand = sum(1 for r in rows if r["outcome"] == "theta_hand")
n_d0 = sum(1 for r in rows if r["D"] == 0)

out = {
    "n_total": n_total, "n_extracted": n_ok, "mapping_rate": round(mapping_rate, 2),
    "errors": errors[:20],
    "confusion": {f"{o}|gt={g}": v for (o, g), v in conf.items()},
    "agreement_exact_pct": round(agreement_pct, 2),
    "disagreement_first5": dis_first,
    "by_service": table,
    "close_rate_spread": {"min": min(rates), "max": max(rates), "separates": separates},
    "handoff_total": n_hand,
    "D_zero_pct": round(100.0 * n_d0 / n_ok, 2),
    "outcome_totals": dict(Counter(r["outcome"] for r in rows)),
}
with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
with open(os.path.join(BASE, "rows.json"), "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1, ensure_ascii=False)

print(f"A. mapping: {n_ok}/{n_total} = {mapping_rate:.2f}% (errors: {len(errors)})")
print(f"B. agreement exact: {agreement_pct:.2f}% | confusion: {out['confusion']}")
print(f"C. services: {len(table)} | close-rate spread {min(rates)}..{max(rates)} -> separates={separates}")
for t in table:
    print(f"   | {t['service'][:24]:24} | n={t['n']:4} | cont={t['contained']:5} | aban={t['abandoned']:5} | D={t['D_mean']:4} | close={t['close_rate']:5} | gt={t['gt_effective']:5}")
print(f"D. handoff={n_hand} | D==0: {out['D_zero_pct']}% | totals: {out['outcome_totals']}")
