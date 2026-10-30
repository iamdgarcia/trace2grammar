"""h5_inversion_stratified.py — kill-or-confirm the polite-close inversion (Claim 2).

Predeclared (threads/t2g-honesty/PLAN.md): the shipped headline "polite close in 66.8% of
PASS vs 76.4% of FAILED" may be a length/composition artifact (probe: pooled d=+9.6pp but
by n_states tercile short +7.3 / mid -1.0 / long -0.6; by model gpt4o +2.4 vs sonnet +13.4).

Test: per-cell (model x domain) table + CMH (Cochran-Mantel-Haenszel) over 6 strata
(model x n_states tercile), tercile edges computed on the pooled 1,980 episodes.
CONFIRM if CMH common OR keeps inversion sign (OR>1: close more likely on FAIL) with p<0.05.
Else REFRAME to a length-confounded descriptive.

Stdlib only. Data: dataset/taubench/episodes.csv.
"""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar/dataset/taubench/episodes.csv"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["n_states"] = int(r["n_states"])
    r["close"] = r["reached_close"] == "True"

n = len(rows)
fail = [r for r in rows if r["reward"] < 1]
ok = [r for r in rows if r["reward"] >= 1]
print(f"n={n} pass={len(ok)} fail={len(fail)}")
print(f"pooled: polite-close PASS {sum(r['close'] for r in ok)/len(ok):.1%} "
      f"FAIL {sum(r['close'] for r in fail)/len(fail):.1%}")

# tercile edges on pooled n_states
xs = sorted(r["n_states"] for r in rows)
q1, q2 = xs[n // 3], xs[2 * n // 3]
print(f"n_states tercile edges: {q1}, {q2}")

# per-cell table (model x domain)
cells = {}
for r in rows:
    cells.setdefault((r["model"], r["domain"]), []).append(r)
print("\nper-cell polite-close rate (PASS vs FAIL):")
cell_rows = []
for k in sorted(cells):
    rr = cells[k]
    p = [r for r in rr if r["reward"] >= 1]
    f = [r for r in rr if r["reward"] < 1]
    if not p or not f:
        continue
    rp, rf = sum(r["close"] for r in p) / len(p), sum(r["close"] for r in f) / len(f)
    cell_rows.append({"cell": f"{k[0]}/{k[1]}", "n": len(rr),
                      "pass_close": round(rp, 4), "fail_close": round(rf, 4),
                      "delta_pp": round((rf - rp) * 100, 1)})
    print(f"  {k[0]}/{k[1]}: n={len(rr)} PASS {rp:.1%} FAIL {rf:.1%} delta {100*(rf-rp):+.1f}pp")

# strata = model x tercile
def tercile(v):
    return 0 if v < q1 else (1 if v < q2 else 2)

strata = {}
for r in rows:
    strata.setdefault((r["model"], tercile(r["n_states"])), []).append(r)

# CMH: for each stratum, 2x2 table close x fail; a = close & fail
num = den = 0.0
strat_rows = []
for k in sorted(strata):
    rr = strata[k]
    f = sum(1 for r in rr if r["reward"] < 1)
    c = sum(1 for r in rr if r["close"])
    a = sum(1 for r in rr if r["close"] and r["reward"] < 1)
    n_k = len(rr)
    if n_k < 10 or f in (0, n_k) or c in (0, n_k):
        continue
    e_a = f * c / n_k
    v = f * (n_k - f) * c * (n_k - c) / (n_k**2 * (n_k - 1)) if n_k > 1 else 0.0
    num += a - e_a
    den += v
    strat_rows.append({"stratum": f"{k[0]}/T{k[1]}", "n": n_k,
                       "fail": f, "close": c, "close_fail": a,
                       "delta_pp": round((sum(1 for r in rr if r['close'] and r['reward']<1)/f
                                          - sum(1 for r in rr if r['close'] and r['reward']>=1)/(n_k-f))*100, 1)})
chi2 = (abs(num) - 0.5) ** 2 / den if den > 0 else float("nan")
p_val = math.erfc(math.sqrt(chi2 / 2))  # chi2, 1 df
# common OR (Mantel-Haenszel)
or_num = or_den = 0.0
for k in sorted(strata):
    rr = strata[k]
    f = sum(1 for r in rr if r["reward"] < 1)
    c = sum(1 for r in rr if r["close"])
    a = sum(1 for r in rr if r["close"] and r["reward"] < 1)
    n_k = len(rr)
    if f in (0, n_k) or c in (0, n_k) or n_k == 0:
        continue
    or_num += a * (n_k - c - (n_k - f - a)) / n_k
    or_den += (c - a) * (f - a) / n_k
common_or = or_num / or_den if or_den else float("nan")

print("\nCMH over strata (model x tercile):")
for s in strat_rows:
    print(f"  {s['stratum']}: n={s['n']} delta {s['delta_pp']:+.1f}pp")
print(f"CMH common OR (fail-vs-pass close odds) = {common_or:.3f} | chi2={chi2:.2f} p={p_val:.4f}")

verdict = "CONFIRM" if (common_or > 1 and p_val < 0.05) else "REFRAME"
print("\nVERDICT:", verdict)

json.dump({"n": n, "tercile_edges": [q1, q2],
           "pooled": {"pass_close": round(sum(r['close'] for r in ok)/len(ok), 4),
                      "fail_close": round(sum(r['close'] for r in fail)/len(fail), 4)},
           "cells": cell_rows, "strata": strat_rows,
           "cmh_common_or": round(common_or, 4), "cmh_p": round(p_val, 5),
           "verdict": verdict},
          open(os.path.join(HERE, "h5_results.json"), "w", encoding="utf-8"), indent=1)
