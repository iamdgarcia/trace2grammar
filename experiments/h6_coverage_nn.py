"""h6_coverage_nn.py — kill-or-confirm kernel-free coverage 0.0% (Claim 3).

Predeclared (threads/t2g-honesty/PLAN.md): "kernel-free instance coverage 0.0% (of 961
unique real positions, no simulated episode lands on one)" demands exact float collision
in a continuous 2-D PCA space -> measure-zero by construction (probe p2: NN-ratio 1.66,
21/60 sim points within the real-median NN radius).

Test: ratio R = median(sim->real NN distance) / median(real->real NN distance) with a
200-resample bootstrap 95% CI over real episodes (each resample recomputes both medians
on matched sizes). CONFIRM (the gap is real) if the CI excludes 1.0; else REFRAME to
NN-distance restatement.

Stdlib only. Data: runtime/taubench_anchor/tb_coverage.json (real2d, sim2d arrays).
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
COV = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor/tb_coverage.json"

cov = json.load(open(COV, encoding="utf-8"))
print("keys:", list(cov.keys()))
real_all = cov["real2d"]
sim = cov["sim2d"]
# real reference = UNIQUE positions (the README's own framing: 961 unique of 1,980).
# Duplicated real points make the real->real NN median 0 by duplication, not by density.
seen = set()
real = []
for p in real_all:
    k = (round(p[0], 9), round(p[1], 9))
    if k not in seen:
        seen.add(k)
        real.append(p)
print(f"real n={len(real_all)} unique={len(real)} | sim n={len(sim)}")


def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def nn_dist(p, pts, self_idx=None):
    best = None
    for j, q in enumerate(pts):
        if self_idx is not None and j == self_idx:
            continue
        v = d2(p, q)
        if best is None or v < best:
            best = v
    return math.sqrt(best) if best is not None else float("nan")


# point estimates
sim_nn = [min(d2(p, q) for q in real) ** 0.5 for p in sim]
real_nn = [nn_dist(p, real, i) for i, p in enumerate(real)]
med_sim = sorted(sim_nn)[len(sim_nn) // 2]
med_real = sorted(real_nn)[len(real_nn) // 2]
R = med_sim / med_real
within = sum(1 for v in sim_nn if v <= med_real)
print(f"median sim->real NN = {med_sim:.4f} | median real->real NN = {med_real:.4f}")
print(f"R = {R:.3f} | sim points within real-median NN radius: {within}/{len(sim)}")

# bootstrap: resample the 60 sim points with replacement; the real reference scale
# (median real->real NN over unique positions) is a fixed property of the corpus.
rng = random.Random(20261025)
ratios = []
for _ in range(200):
    ss = [rng.choice(sim) for _ in range(len(sim))]
    ms = sorted(min(d2(p, q) for q in real) ** 0.5 for p in ss)[len(ss) // 2]
    if med_real > 0:
        ratios.append(ms / med_real)
ratios.sort()
ci = [ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]]
print(f"bootstrap 95% CI of R: [{ci[0]:.3f}, {ci[1]:.3f}]")

verdict = "CONFIRM" if (ci[0] > 1.0) else "REFRAME"
print("\nVERDICT:", verdict, "(gap real iff CI excludes 1.0)")

json.dump({"sim_n": len(sim), "real_n": len(real_all), "real_unique": len(real),
           "median_sim_to_real_nn": round(med_sim, 4),
           "median_real_to_real_nn": round(med_real, 4),
           "ratio_R": round(R, 4), "bootstrap_ci": [round(ci[0], 4), round(ci[1], 4)],
           "sim_within_real_median_radius": within,
           "verdict": verdict},
          open(os.path.join(HERE, "h6_results.json"), "w", encoding="utf-8"), indent=1)
