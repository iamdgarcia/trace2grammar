#!/usr/bin/env python3
"""TEST measurement with the DEV-calibrated close vocabulary (the clean claim).

Pipeline: dev_calibration.json (patterns derived from dev ONLY, see
dev_calibrate.py) -> cfg built by extending ExtractionConfig.close_assistant_patterns
through the declared mechanism -> re-extract the 4,201 test dialogues -> agreement
vs structural ground truth. Comparison set:
  - verbatim default cfg:      55.34% (results.json)
  - contaminated post-hoc:     77.22% (patterns hand-picked from test, PH-2)
  - dev-calibrated (THIS):     the number the paper may quote.

Same tau verbatim contract: zero changes to builds/, patterns travel via cfg.
"""
import json
import os
import sys
from collections import Counter

# artifact package root: scripts live at the repo root; out-of-repo layouts
# set the T2G_ROOT env var to the artifact checkout
ROOT = os.environ.get("T2G_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract, ExtractionConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "sgd_test_transcript")
meta = json.load(open(os.path.join(SRC, "_meta.json"), encoding="utf-8"))
calib = json.load(open(os.path.join(BASE, "dev_calibration.json"), encoding="utf-8"))

cfg = ExtractionConfig()
cfg.close_assistant_patterns = cfg.close_assistant_patterns + tuple(
    p["regex"] for p in calib["patterns"])

rows = []
errors = []
for did, m in sorted(meta.items()):
    txt = open(os.path.join(SRC, f"{did}.txt"), encoding="utf-8").read()
    try:
        r = extract(txt, "transcript", cfg)
    except Exception as e:  # noqa: BLE001
        errors.append({"id": did, "error": repr(e)[:200]})
        continue
    walk = r["walk"]
    rows.append({
        "id": did,
        "primary": m["primary"],
        "outcome": r["outcome"],
        "D": r["cycle_depth"],
        "reached_close": any(w["state"] == "q_close" for w in walk),
        "gt_effective_call": m["gt_effective_call"],
    })

n_ok = len(rows)
conf = Counter((r["outcome"], r["gt_effective_call"]) for r in rows)
agree = sum(v for (o, g), v in conf.items() if (o == "theta_cont") == g)
agreement = 100.0 * agree / n_ok

# sanity: the 5 known courteous-close cases must flip to their agreement cell
rows_by = {r["id"]: r for r in rows}
gold = [r for r in json.load(open(os.path.join(BASE, "rows.json"), encoding="utf-8"))
        if r["outcome"] == "theta_cont" and r["gt_effective_call"] is False]
flipped = [g["id"] for g in gold if rows_by[g["id"]]["outcome"] == "theta_aban"]

# per-service close rates (firma multiwoz bajo cfg calibrado)
by_dom = {}
for r in rows:
    by_dom.setdefault(r["primary"], []).append(r)
table = []
for dom, rs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
    oc = Counter(x["outcome"] for x in rs)
    table.append({
        "service": dom, "n": len(rs),
        "contained": round(100.0 * oc.get("theta_cont", 0) / len(rs), 1),
        "close_rate": round(100.0 * sum(1 for x in rs if x["reached_close"]) / len(rs), 1),
        "gt_effective": round(100.0 * sum(1 for x in rs if x["gt_effective_call"]) / len(rs), 1),
    })
rates = [t["close_rate"] for t in table]

out = {
    "n_total": len(meta), "n_extracted": n_ok, "errors": errors[:10],
    "cfg_patterns_added": [p["regex"] for p in calib["patterns"]],
    "confusion": {f"{o}|gt={g}": v for (o, g), v in conf.items()},
    "agreement_devcalibrated_pct": round(agreement, 2),
    "comparison": {
        "verbatim_default_pct": 55.34,
        "dev_calibrated_pct": round(agreement, 2),
        "contaminated_posthoc_pct": 77.22,
    },
    "courteous5_flipped_to_aban": flipped,
    "close_rate_spread": {"min": min(rates), "max": max(rates),
                          "separates": (max(rates) - min(rates)) > 5.0},
    "by_service": table,
    "outcome_totals": dict(Counter(r["outcome"] for r in rows)),
}
json.dump(out, open(os.path.join(BASE, "results_calib.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

print(f"TEST agreement with DEV-calibrated cfg: {agreement:.2f}% ({agree}/{n_ok})")
print(f"  verbatim 55.34% | dev-calibrated {agreement:.2f}% | contaminated 77.22%")
print(f"  courteous-5 flipped to theta_aban: {len(flipped)}/5 {flipped}")
print(f"  close rates {min(rates)}..{max(rates)} separates={out['close_rate_spread']['separates']}")
print(f"  confusion: {out['confusion']}")
