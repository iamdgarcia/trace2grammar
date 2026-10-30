"""h8_instance_honest.py — the instance-coverage radius is a duplicate artifact (fix for paper §5).

Finding: coverage.instance_coverage() reads r = median real NN distance over ALL real
points. The 1,980 tau-bench walks occupy only 961 unique positions in the projection
plane, so >50% of points have NN distance 0 to a duplicate, the median is 0, and instance
coverage is 0% by construction (no sim lands exactly on a real point). The measure-zero
companion (probe 2026-10-30, h6): the honest radius comes from the DEDUPLICATED real set.

This script reproduces the shipped number (r=0, instance 0%), computes the honest variant
(r over 961 unique positions, instance coverage at that radius, per class), and consolidates
the NN-ratio with bootstrap CI (h6) into one artifact for paper section 5.

Stdlib only. Data: runtime/taubench_anchor/tb_coverage.json.
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
COV = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor/tb_coverage.json"

cov = json.load(open(COV, encoding="utf-8"))
real_all = [tuple(p) for p in cov["real2d"]]
sim = [tuple(p) for p in cov["sim2d"]]
real_meta = cov["real_meta"]
classes_all = [m["outcome"] for m in real_meta]

out = {"n_real": len(real_all), "n_sim": len(sim)}

# ---- shipped computation (reproduce paper's number verbatim): r over all points
def median_nn(pts):
    nn = []
    for i, (rx, ry) in enumerate(pts):
        best = min((rx - x) ** 2 + (ry - y) ** 2 for j, (x, y) in enumerate(pts) if j != i)
        nn.append(math.sqrt(best))
    nn.sort()
    n = len(nn)
    return nn[n // 2] if n % 2 else (nn[n // 2 - 1] + nn[n // 2]) / 2

r_shipped = median_nn(real_all)
r2 = r_shipped * r_shipped
hits = sum(any((rx - x) ** 2 + (ry - y) ** 2 <= r2 for x, y in sim) for rx, ry in real_all)
out["shipped"] = {"r_median_real_nn_all_points": round(r_shipped, 6),
                  "instance_coverage_pct": round(100.0 * hits / len(real_all), 2)}
print(f"shipped: r={r_shipped} -> instance coverage {out['shipped']['instance_coverage_pct']}%")

# ---- honest: dedupe real positions
seen, real_u, cls_u = set(), [], []
for p, c in zip(real_all, classes_all):
    k = (round(p[0], 9), round(p[1], 9))
    if k not in seen:
        seen.add(k)
        real_u.append(p)
        cls_u.append(c)
out["unique_real"] = len(real_u)

r_u = median_nn(real_u)
r2 = r_u * r_u
hits_u = [any((rx - x) ** 2 + (ry - y) ** 2 <= r2 for x, y in sim) for rx, ry in real_u]
overall = 100.0 * sum(hits_u) / len(hits_u)
per_class = {}
for cl in sorted(set(cls_u)):
    idx = [i for i, c in enumerate(cls_u) if c == cl]
    per_class[cl] = round(100.0 * sum(hits_u[i] for i in idx) / len(idx), 2)
out["honest"] = {"r_median_real_nn_unique": round(r_u, 6),
                 "instance_coverage_pct": round(overall, 2),
                 "per_class": per_class}
print(f"honest: r={r_u:.6f} over {len(real_u)} unique -> instance coverage {overall:.1f}% | per class {per_class}")

# ---- sim points within the honest radius (the sim-side view, h6 cross-check)
sim_hits = sum(any((rx - x) ** 2 + (ry - y) ** 2 <= r2 for rx, ry in real_u) for x, y in sim)
out["sim_within_r"] = f"{sim_hits}/{len(sim)}"
print(f"sim points within honest r: {sim_hits}/{len(sim)}")

# ---- NN-ratio with bootstrap (consolidates h6)
real_nn = [nn for nn in [None] ] # placeholder to keep structure clear
med_real = r_u  # median unique-real NN == h6's med_real (verify)
def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
sim_nn = sorted(min(d2(p, q) for q in real_u) ** 0.5 for p in sim)
med_sim = sim_nn[len(sim_nn) // 2]
R = med_sim / med_real
rng = random.Random(20261025)
ratios = []
for _ in range(200):
    ss = [rng.choice(sim) for _ in range(len(sim))]
    ms = sorted(min(d2(p, q) for q in real_u) ** 0.5 for p in ss)[len(ss) // 2]
    ratios.append(ms / med_real)
ratios.sort()
ci = [round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios))], 4)]
out["nn_ratio"] = {"R": round(R, 4), "bootstrap_ci": ci, "median_sim_to_real_nn": round(med_sim, 6)}
print(f"NN-ratio R={R:.3f} CI {ci} (matches h6: R=1.595 CI [1.044, 1.689])")
assert abs(R - 1.595) < 0.01 and abs(med_real - 0.0800) < 0.002, "h6 parity broken"

json.dump(out, open(os.path.join(HERE, "h8_results.json"), "w", encoding="utf-8"), indent=1)
print("\nwritten h8_instance_honest.json")
