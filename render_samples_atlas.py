# Render the sample-set atlas (2-D behavior plane, real vs sim) from samples/.
# Data-true: every point is a walk from samples/*.txt computed by the shipped
# extract()+coverage_from_walks(); the sim batch is the over-cooperative set.
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract
from trace2grammar.coverage import coverage_from_walks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(prefix, exclude_prefix=None):
    out = []
    for f in sorted(os.listdir(os.path.join(ROOT, "samples"))):
        if f.startswith(prefix) and f.endswith(".txt") and not (exclude_prefix and f.startswith(exclude_prefix)):
            with open(os.path.join(ROOT, "samples", f), encoding="utf-8") as fh:
                out.append(extract(fh.read(), "transcript"))
    return out

real = [r for r in load("cont") + load("aban") + load("hand")]
sim = load("sim", exclude_prefix="sim2")
sim2 = load("sim2")
print("real walks:", len(real), "| sim control:", len(sim), "| sim2 diverse:", len(sim2))

cov = coverage_from_walks(real, sim + sim2, res=24, eps=0.05, mode="grammar")
print("coverage:", cov["coverage_pct"], "% |", cov["covered_cells"], "/", cov["occupied_cells"],
      "cells | per class:", cov["per_class_coverage"])

BLUE, ORANGE, RED = "#1e5aa8", "#ef6c00", "#c62828"
LBL = {"theta_cont": "contained", "theta_aban": "abandoned", "theta_hand": "handoff"}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
from matplotlib.patches import Rectangle
for u in cov["uncovered_cells"]:
    pass  # grid cell overlay needs the transform; keep points readable: draw cells
from trace2grammar.coverage import _grid_transform
x0, y0, cw, ch = _grid_transform(cov["real2d"], cov["grid"]["res"], 0.15)
res = cov["grid"]["res"]
uncovered_set = {(u["gx"], u["gy"]) for u in cov["uncovered_cells"]}
for i, j in [(i, j) for i in range(res) for j in range(res)]:
    if (i, j) in uncovered_set:
        ax.add_patch(Rectangle((x0+i*cw, y0+j*ch), cw, ch, fc=RED, alpha=0.16, ec="none"))

markers = {"theta_cont": "o", "theta_aban": "s", "theta_hand": "^"}
colors = {"theta_cont": "#2e8b57", "theta_aban": RED, "theta_hand": "#7b3fa0"}
for (x, y), m in zip(cov["real2d"], cov["real_meta"]):
    ax.plot([x], [y], markers[m["outcome"]], color=colors[m["outcome"]], ms=9,
            mec="white", mew=1.0, zorder=4)
for (x, y), m in zip(cov["sim2d"][:len(sim)], cov["sim_meta"][:len(sim)]):
    ax.plot([x], [y], "D", color=ORANGE, ms=8, mec="white", mew=1.0, zorder=4)
for (x, y), m in zip(cov["sim2d"][len(sim):], cov["sim_meta"][len(sim):]):
    ax.plot([x], [y], "D", color="#b8860b", ms=7, mec="white", mew=1.0, zorder=4)

from matplotlib.lines import Line2D
leg = [Line2D([], [], marker=markers[k], ls="none", color=colors[k],
              label="real: " + LBL[k]) for k in ("theta_cont", "theta_aban", "theta_hand")]
leg.append(Line2D([], [], marker="D", ls="none", color=ORANGE, label="sim control (over-cooperative)"))
leg.append(Line2D([], [], marker="D", ls="none", color="#b8860b", label="sim2 (diverse 20/20/20)"))
leg.append(Rectangle((0, 0), 1, 1, fc=RED, alpha=0.16, label="uncovered cells"))
ax.legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.9)

ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#cccccc")
ax.set_title("sample-set atlas: %d real + %d+sim control + %d diverse simulated\n(samples/ in the trace2grammar repo)" % (len(real), len(sim), len(sim2)), fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "samples_atlas.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(ROOT, "samples_atlas.png"), bbox_inches="tight", dpi=150)

json.dump({k: v for k, v in cov.items() if k != "classes"},
          open(os.path.join(ROOT, "samples_coverage.json"), "w", encoding="utf-8"),
          indent=1, default=list)
print("samples_atlas.pdf/.png + samples_coverage.json written")
