"""Region-coverage checks. Run: python tests/test_coverage.py"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace2grammar.coverage import (
    coverage,
    feature_matrix,
    pca2d,
    top_k_uncovered,
    znorm,
)

checks = []


def check(name, ok):
    checks.append((name, ok))
    print(("PASS" if ok else "FAIL"), "-", name)


# (a) single real point, sim at the same point -> 100% coverage (grid transform is
# extent-relative, so "tight cluster" = single point; multi-point cases in (c)).
cov = coverage([(0.0, 0.0)], [(0.0, 0.0)])
check("point+point: coverage 100%", cov["coverage_pct"] == 100.0)
check("point+point: no uncovered cells", len(cov["uncovered_cells"]) == 0)

# (a2) sim set mirroring the real set cell-for-cell -> 100%
real = [(0.0, 0.0), (10.0, 10.0)]
cov = coverage(real, [(0.0, 0.0), (10.0, 10.0)])
check("mirrored sets: coverage 100%", cov["coverage_pct"] == 100.0)

# (b) sim far away -> 0% coverage, uncovered cells with correct nearest exemplar
cov2 = coverage(real, [(100.0, 100.0)])
check("far sim: coverage 0%", cov2["coverage_pct"] == 0.0)
check("far sim: uncovered cells reported", len(cov2["uncovered_cells"]) > 0)
u = cov2["uncovered_cells"][0]
dists = [((rx - u["cx"]) ** 2 + (ry - u["cy"]) ** 2) for rx, ry in real]
check(
    "far sim: nearest exemplar index is argmin of distances",
    u["exemplar_idx"] == dists.index(min(dists)),
)

# (c) per-class coverage: sim reaches only the contained region
cls = ["theta_cont", "theta_hand"]
cov3 = coverage(real, [(0.0, 0.0)], classes=cls)
check(
    "per-class: contained 100%, handoff 0%",
    cov3["per_class_coverage"]["theta_cont"] == 100.0
    and cov3["per_class_coverage"]["theta_hand"] == 0.0,
)

# (d) znorm: constant column -> zeros, no NaN
rows = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
zn = znorm(rows)
ok = all(math.isfinite(x) for r in zn for x in r) and all(r[1] == 0.0 for r in zn)
check("znorm: constant column -> zeros (no NaN)", ok)

# (e) PCA: 2-point case separates on PC1
two = [[-5.0, 0.0, 1.0], [5.0, 0.0, -1.0]]
proj = pca2d(two)
check("pca2d: 2 points separate on PC1", proj[0][0] < 0 < proj[1][0])
check(
    "pca2d: PC1 spread >= PC2 spread", abs(proj[0][0] - proj[1][0]) >= abs(proj[0][1])
)

# (f) top_k_uncovered: dedups neighbors, sorts by density
cov4 = coverage(real, [(100.0, 100.0)])
tops = top_k_uncovered(cov4, 3)
dens = [t["local_real_density"] for t in tops]
check("top_k: sorted desc by density", dens == sorted(dens, reverse=True))
ok = all(
    max(abs(a["gx"] - b["gx"]), abs(a["gy"] - b["gy"])) > 2
    for a in tops
    for b in tops
    if a is not b
)
check("top_k: neighbor cells deduped", ok)

# (g) feature_matrix end-to-end: rows keep dimensions, no NaN
fake_results = [
    {
        "walk": [
            {"t_ms": 0, "state": "q_greet", "transition_type": "start"},
            {"t_ms": 100, "state": "q_intent", "transition_type": "user-answer"},
            {"t_ms": 900, "state": "theta_cont", "transition_type": "service-complete"},
        ],
        "outcome": "theta_cont",
        "cycle_depth": 0,
        "error_count": 0,
    },
    {
        "walk": [
            {"t_ms": 0, "state": "q_greet", "transition_type": "start"},
            {"t_ms": 100, "state": "q_tool", "transition_type": "tool-call"},
            {"t_ms": 200, "state": "q_tool", "transition_type": "tool-fail"},
            {"t_ms": 1200, "state": "theta_aban", "transition_type": "hangup"},
        ],
        "outcome": "theta_aban",
        "cycle_depth": 1,
        "error_count": 1,
    },
]
for mode in ("grammar", "bigram"):
    m = feature_matrix(fake_results, mode=mode)
    finite = all(math.isfinite(x) for r in m for x in r)
    check(
        f"feature_matrix[{mode}]: finite, consistent width ({len(m[0])} dims)",
        finite and len(m[0]) == len(m[1]),
    )

# --- summary ---------------------------------------------------------------------
fails = [n for n, ok in checks if not ok]
print()
if fails:
    print(f"{len(fails)} CHECKS FAILED")
    sys.exit(1)
print("ALL CHECKS PASSED")
