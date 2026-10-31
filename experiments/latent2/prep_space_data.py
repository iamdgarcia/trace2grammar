"""prep_space_data.py — build data.js for the Conversation Latent Map Space (one pass).

Everything from committed artifacts: e7's composed 2-D projections (recomputed here, same
code), real_meta/sim_meta (openings + verdicts), e3 twin pairs (tau*/typed answers),
e6 lie examples (2 flipped). No model calls.
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
sys.path.insert(0, os.path.join(HF, "builds/trace2grammar"))
TB = json.load(open(os.path.join(HF, "runtime/taubench_anchor/tb_coverage.json"), encoding="utf-8"))
E3 = json.load(open(os.path.join(HERE, "e3_results.json"), encoding="utf-8"))
E7 = json.load(open(os.path.join(HERE, "e7_results.json"), encoding="utf-8"))
OUT = os.path.join(HF, "builds", "conversation-latent-map")
os.makedirs(OUT, exist_ok=True)

emb = json.load(open(os.path.join(HF, "runtime/latent2/openings_embed.json"), encoding="utf-8"))
rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
for r in rows:
    r["key"] = r["first_user_text"].strip()

def zn(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0)
    return np.where(sd > 1e-12, (M - M.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)

def struct_vec(r):
    return [float(r["D"]), float(int(r["D"]) >= 1), float(r["n_states"]),
            float(r["error_count"]), float(r["reached_close"] == "True"),
            math.log1p(float(r["duration_ms"])),
            1.0 if r["model"] == "sonnet" else 0.0, 1.0 if r["domain"] == "airline" else 0.0,
            1.0 if r["outcome_convo"] == "theta_cont" else 0.0,
            1.0 if r["outcome_convo"] == "theta_aban" else 0.0]

S = zn([struct_vec(r) for r in rows])
Em = zn([emb[r["key"]] for r in rows])
C = np.concatenate([Em, S], axis=1)
C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
mean = C.mean(axis=0)
X = C - mean
Cv = (X.T @ X) / (len(X) - 1)
w, V = np.linalg.eigh(Cv)
v1, v2 = V[:, -1], V[:, -2]
real2d = [(float((p - mean) @ v1), float((p - mean) @ v2)) for p in C]

real = [{"x": round(x, 4), "y": round(y, 4),
         "o": r["outcome_convo"], "D": int(r["D"]), "m": r["model"], "d": r["domain"],
         "t": (r["first_user_text"].strip()[:90])}
        for (x, y), r in zip(real2d, rows)]

sim = [{"x": 0, "y": 0, "o": m["outcome"], "t": m["first_user_text"].strip()[:90]}
       for m in TB["sim_meta"]]
# project sims onto the same axes (semantic from their openings cached in emb; structural approx: skip)
Es = zn([emb[k.strip()] for k in [m["first_user_text"] for m in TB["sim_meta"]]])
Ss = zn([[float(m["cycle_depth"]), float(int(m["cycle_depth"]) >= 1), float(m["n_states"]), 0.0,
          1.0 if m["n_states"] > 6 else 0.0,
          math.log1p(float(m["duration_ms"])), 1.0, 0.0,
          1.0 if m["outcome"] == "theta_cont" else 0.0,
          1.0 if m["outcome"] == "theta_aban" else 0.0] for m in TB["sim_meta"]])
Cs = np.concatenate([Es, Ss], axis=1)
Cs = Cs / (np.linalg.norm(Cs, axis=1, keepdims=True) + 1e-12)
sim = [{"x": round(float((p - mean) @ v1), 4), "y": round(float((p - mean) @ v2), 4),
        "o": m["outcome"], "t": m["first_user_text"].strip()[:90]}
       for p, m in zip(Cs, TB["sim_meta"])]

# twins (from e3)
twins = [{"f": r["pair"][0], "p": r["pair"][1], "tau": r["tau_sem"], "err": r.get("err_turn"),
          "t": r["pair"][0]} for r in E3["pairs"]]

# lie examples: pick 3 flipped
E6 = json.load(open(os.path.join(HERE, "e6_results.json"), encoding="utf-8"))
lies = [d for d in E6["detail"] if d.get("flip")][:3]
lie_out = []
for d in lies:
    p = os.path.join(HF, "runtime/latent2/e6_lies", f"lie_{d['id'].replace(':', '_')}.txt")
    lie_out.append({"id": d["id"], "v0": d.get("verdict0"), "v2": d.get("verdict2"),
                    "text": open(p, encoding="utf-8").read()[:500]})

data = {"real": real, "sim": sim, "twins": twins, "lies": lie_out,
        "headline": {"region_structural": E7["spaces"]["structural"]["region_bw0.04"]["coverage_pct"],
                     "region_composed": E7["spaces"]["composed"]["region_bw0.04"]["coverage_pct"],
                     "nn_structural": E7["spaces"]["structural"]["nn_ratio"]["R"],
                     "nn_composed": E7["spaces"]["composed"]["nn_ratio"]["R"],
                     "lie_pct": E6["flip_pct"], "sep_pct": E3["P_A_pct"]}}
js = "const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
open(os.path.join(OUT, "data.js"), "w", encoding="utf-8").write(js)
print("data.js written:", len(js) // 1024, "KB |", len(real), "real +", len(sim), "sim +",
      len(twins), "twins +", len(lie_out), "lies")
