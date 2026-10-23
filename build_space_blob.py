# Build the data blob for the where-conversations-die Space.
# Everything data-true: walks computed by the shipped extract(), PCA real-anchored,
# grid+coverage by the shipped coverage(). The blob embeds the FULL transcripts so
# the demo can show the conversation behind every point without a server.
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract
from trace2grammar.coverage import coverage_from_walks, _grid_transform
from trace2grammar.operativa import label_operativa

def load(prefix, exclude_prefix=None):
    out = []
    for f in sorted(os.listdir(os.path.join(ROOT, "samples"))):
        if f.startswith(prefix) and f.endswith(".txt") and not (exclude_prefix and f.startswith(exclude_prefix)):
            with open(os.path.join(ROOT, "samples", f), encoding="utf-8") as fh:
                txt = fh.read()
            r = extract(txt, "transcript")
            out.append({"name": f[:-4], "transcript": txt, "res": r})
    return out

real = load("cont") + load("aban") + load("hand")
sim = load("sim", exclude_prefix="sim2") + load("sim2")
cov = coverage_from_walks([r["res"] for r in real], [r["res"] for r in sim], res=24, eps=0.10, mode="grammar")
print("coverage:", cov["coverage_pct"], "% |", cov["covered_cells"], "/", cov["occupied_cells"])

LBL = {"theta_cont": "contained", "theta_aban": "abandoned", "theta_hand": "handoff"}

def pt(c, meta, i):
    m = meta[i]
    r = c["res"]
    walk_states = [w["state"] for w in r["walk"]]
    # per-line state annotation: map events to transcript lines via attrs.line (transcript dialect carries line)
    return {
        "x": round(cov["real2d" if meta is cov["real_meta"] else "sim2d"][i][0], 4) if False else None,
    }

def buildPts(entries, coords, meta, kind):
    pts = []
    for i, e in enumerate(entries):
        r = e["res"]
        m = meta[i]
        pts.append({
            "id": e["name"],
            "x": round(coords[i][0], 4),
            "y": round(coords[i][1], 4),
            "kind": kind,
            "outcome": m["outcome"],
            "op": label_operativa(e["transcript"], __import__("re").findall(r"tool\(([\w.\-]+)\)", e["transcript"])),
            "D": m["cycle_depth"],
            "dur_s": round(m["duration_ms"] / 1000, 1),
            "n_states": m["n_states"],
            "first": m["first_user_text"][:90],
            "walk": [w["state"] for w in r["walk"]],
            "transcript": e["transcript"],
        })
    return pts

real_pts = buildPts(real, cov["real2d"], cov["real_meta"], "real")
sim_pts = buildPts(sim, cov["sim2d"], cov["sim_meta"], "sim")

x0, y0, cw, ch = _grid_transform(cov["real2d"], 24, 0.15)
blob = {
    "generated": "2026-10-23",
    "grid": {"res": 24, "x0": x0, "y0": y0, "cw": cw, "ch": ch,
             "eps": cov["grid"]["eps"], "bw": cov["grid"]["bw_frac"]},
    "coverage": {"pct": cov["coverage_pct"], "occupied": cov["occupied_cells"],
                 "covered": cov["covered_cells"], "per_class": cov["per_class_coverage"]},
    "uncovered": [{"gx": u["gx"], "gy": u["gy"],
                   "ex": u["exemplar_idx"], "d": round(u["local_real_density"], 3)}
                  for u in sorted(cov["uncovered_cells"], key=lambda u: -u["local_real_density"])],
    "points": real_pts + sim_pts,
    "labels": LBL,
}
# ponytail: fixed absolute output path to the sibling Space build; move to config when a second consumer appears
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "where-conversations-die", "atlas_data.js")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("var ATLAS = ")
    json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"))
    fh.write(";")
print("blob written:", os.path.getsize(out), "bytes |", len(blob["uncovered"]), "uncovered cells |", len(blob["points"]), "points")
