"""h7_d_crosstab.py — kill-or-confirm D-as-drift on tau-bench (Claim 4).

Predeclared (threads/t2g-honesty/PLAN.md; probe p8): D>0 iff error_count>0 in tau-bench
-> D there measures tool-retry depth, not conversational re-entry. Package the crosstab
and the 3-corpus D table (the paper's sim2real loop-gap claim) as one committed artifact.

Verdict rule: REFRAME (kill D-as-drift on outcome-bearing tau-bench) iff D>0 implies
error_count>0 holds in >=99% of rows (probe measured 1,979/1,980).

Stdlib only. Data: the three dataset CSVs.
"""
import csv
import json
import os
from collections import Counter

BASE = "C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar/dataset"
HERE = os.path.dirname(os.path.abspath(__file__))

tb = list(csv.DictReader(open(BASE + "/taubench/episodes.csv", encoding="utf-8")))
sgd = list(csv.DictReader(open(BASE + "/sgd/dialogs.csv", encoding="utf-8")))
mwz = list(csv.DictReader(open(BASE + "/multiwoz/dialogs.csv", encoding="utf-8")))

# crosstab D>0 vs error_count>0 (tau-bench)
ct = Counter((int(r["D"]) > 0, int(r["error_count"]) > 0) for r in tb)
print("tau-bench crosstab (D>0, error_count>0):", dict(ct))
d_pos = sum(v for (d, _), v in ct.items() if d)
agree = ct.get((True, True), 0) + ct.get((False, False), 0)
print(f"D>0 rows: {d_pos} | D==error_count agreement: {agree}/{len(tb)}")

# 3-corpus D table
def d_pos_rate(rows, col):
    pos = sum(1 for r in rows if float(r[col]) > 0)
    return pos, len(rows), pos / len(rows)

t_pos, t_n, t_rate = d_pos_rate(tb, "D")
s_pos, s_n, s_rate = d_pos_rate(sgd, "D_verbatim")
s2_pos, s2_n, s2_rate = d_pos_rate(sgd, "D_calibrated")
m_pos, m_n, m_rate = d_pos_rate(mwz, "D")
print(f"\nD>0 rates: tau-bench {t_pos}/{t_n} ({t_rate:.1%}) | SGD verbatim {s_pos}/{s_n} "
      f"({s_rate:.1%}) | SGD calibrated {s2_pos}/{s2_n} ({s2_rate:.1%}) | MultiWOZ {m_pos}/{m_n} ({m_rate:.1%})")
print("tau D distribution:", sorted(Counter(int(r['D']) for r in tb).items()))

frac = agree / len(tb)
verdict = "REFRAME" if frac >= 0.99 else "CONFIRM"
print("\nVERDICT:", verdict, f"(D==error_count in {frac:.1%} of tau rows)")

# SGD silent-failure recovery split (README verdict-vs-truth paragraph)
cand = [r for r in sgd if r["silent_failure_candidate"] == "True"]
rec = [r for r in cand if r["gt_effective_call"] == "True"]
term = [r for r in cand if r["gt_effective_call"] != "True"]
missed = [r for r in term if r["outcome_calibrated"] == "theta_cont"]
print(f"\nSGD silent-failure candidates: {len(cand)} | recovered: {len(rec)} "
      f"({len(rec)/len(cand):.2%}) | terminal: {len(term)} | walker misses (theta_cont): {len(missed)}")
print("terminal walker misses:", [r["dialog_id"] for r in missed])

json.dump({"tau_crosstab": {str(k): v for k, v in ct.items()},
           "tau_D_equals_error_count": f"{agree}/{len(tb)}",
           "d_gt0_table": {"taubench": [t_pos, t_n, round(t_rate, 4)],
                           "sgd_verbatim": [s_pos, s_n, round(s_rate, 4)],
                           "sgd_calibrated": [s2_pos, s2_n, round(s2_rate, 4)],
                           "multiwoz": [m_pos, m_n, round(m_rate, 4)]},
           "tau_D_distribution": dict(sorted(Counter(int(r['D']) for r in tb).items())),
           "sgd_silent_failure": {"candidates": len(cand),
                                  "recovered": len(rec),
                                  "recovered_rate": round(len(rec) / len(cand), 4),
                                  "terminal": len(term),
                                  "terminal_missed_as_contained": len(missed),
                                  "terminal_missed_ids": [r["dialog_id"] for r in missed]},
           "verdict": verdict},
          open(os.path.join(HERE, "h7_results.json"), "w", encoding="utf-8"), indent=1)
