"""E8 — the typed answer must be TRUE: no-semantic-divergence pairs concentrate walker misses.

Predeclared (G2): the instrument's typed verdict "no semantic divergence = silent-failure
family" is only honest if it is factually concentrated. Among the 102 mixed twin pairs
(e3): population N (no tau*, n=50) vs population D (divergence, n=52).
  Metric: rate of walker-miss (fail twin's walker verdict == theta_cont, i.e. the walker
  reads the failed episode as contained) in N vs D.
  P-E8 PASS: miss rate(N) - miss rate(D) >= +15pp.
Walker verdicts come from episodes.csv outcome_convo (terminal stripped, shipped). No
model calls.
"""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
E3 = json.load(open(os.path.join(HERE, "e3_results.json"), encoding="utf-8"))

rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
v = {f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}": r["outcome_convo"] for r in rows}

pop = {"N": [r for r in E3["pairs"] if r["tau_sem"] is None],
       "D": [r for r in E3["pairs"] if r["tau_sem"] is not None]}
res = {}
for k, pp in pop.items():
    miss = sum(1 for r in pp if v.get(r["pair"][0]) == "theta_cont")
    res[k] = {"n": len(pp), "walker_miss": miss,
              "miss_rate": round(100.0 * miss / len(pp), 1) if pp else 0.0}
    print(f"population {k}: n={len(pp)} | walker-miss (fail twin reads contained) "
          f"{miss}/{len(pp)} = {res[k]['miss_rate']}%")

diff = round(res["N"]["miss_rate"] - res["D"]["miss_rate"], 1)
verdict = "PASS" if diff >= 15 else "FAIL"
print(f"difference (N - D): {diff}pp")
print("VERDICT: P-E8", verdict)

json.dump({"protocol": {"predicate": "miss_rate(N) - miss_rate(D) >= +15pp",
                        "definition": "walker-miss = fail twin's shipped walker verdict == theta_cont"},
           "populations": res, "diff_pp": diff, "verdict": verdict},
          open(os.path.join(HERE, "e8_results.json"), "w", encoding="utf-8"), indent=1)
