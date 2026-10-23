"""Parity check: cover_js.js (browser port) vs trace2grammar.coverage (python).

Runs the exact same projected coordinates through both implementations across the
configs the demo exposes (bandwidth slider x simulator selector) and asserts the
aggregate numbers agree. A mismatch means the Space would show numbers the paper
does not back. Run before every deploy of where-conversations-die.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trace2grammar"))
from trace2grammar.coverage import coverage as py_coverage  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

blob = json.loads(open(f"{ROOT}/atlas_data.js", encoding="utf-8").read()
                  .removeprefix("var ATLAS = ").removesuffix(";"))

real = [[p["x"], p["y"]] for p in blob["points"] if p["kind"] == "real"]
classes = [p["outcome"] for p in blob["points"] if p["kind"] == "real"]
control = [[p["x"], p["y"]] for p in blob["points"]
           if p["kind"] == "sim" and not p["id"].startswith("sim2")]
sim2 = [[p["x"], p["y"]] for p in blob["points"] if p["id"].startswith("sim2")]
assert len(real) == 500 and len(control) == 80 and len(sim2) == 60, "blob shape changed"

CONFIGS = []
for bw in (0.12, 0.08, 0.04, 0.02):
    for name, sim in (("control", control), ("diverse", sim2), ("both", control + sim2)):
        CONFIGS.append((name, bw, sim))

node_script = """
const path = require('path');
const HERE = '__HERE__';
const COVERJS = require(path.join(HERE, 'cover_js.js'));
const blob = JSON.parse(require('fs').readFileSync(
  path.join(HERE, 'atlas_data.js'), 'utf8')
  .replace(/^var ATLAS = /, '').replace(/;$/, ''));
const real = blob.points.filter(p => p.kind === 'real').map(p => [p.x, p.y]);
const classes = blob.points.filter(p => p.kind === 'real').map(p => p.outcome);
const control = blob.points.filter(p => p.kind === 'sim' && !p.id.startsWith('sim2')).map(p => [p.x, p.y]);
const sim2 = blob.points.filter(p => p.id.startsWith('sim2')).map(p => [p.x, p.y]);
const CONFIGS = [];
for (const bw of [0.12, 0.08, 0.04, 0.02]) {
  for (const [name, sim] of [['control', control], ['diverse', sim2], ['both', control.concat(sim2)]]) {
    CONFIGS.push([name, bw, sim]);
  }
}
const out = CONFIGS.map(([name, bw, sim]) => {
  const c = COVERJS.coverage(real, sim, classes, 24, 0.05, bw, 2);
  return { name, bw, pct: c.coveragePct, occupied: c.occupiedCells,
           covered: c.coveredCells, perClass: c.perClass };
});
console.log(JSON.stringify(out));
"""
r = subprocess.run(["node", "-e", node_script.replace("__HERE__", ROOT.replace(chr(92), "/"))], capture_output=True, encoding="utf-8", timeout=120)
if r.returncode != 0:
    print("NODE FAIL:\n", r.stderr)
    sys.exit(1)
js = json.loads(r.stdout)

fails = 0
for (name, bw, sim), j in zip(CONFIGS, js):
    py = py_coverage(real, sim, classes, 24, 0.05, bw_frac=bw)
    same = (j["pct"] == py["coverage_pct"] and j["occupied"] == py["occupied_cells"]
            and j["covered"] == py["covered_cells"])
    pcls = {k: v for k, v in py["per_class_coverage"].items()}
    same_cls = all(abs(j["perClass"].get(k, -1) - v) < 0.15 for k, v in pcls.items())
    status = "OK " if (same and same_cls) else "FAIL"
    if not (same and same_cls):
        fails += 1
    print(f"[{status}] {name:8s} bw={bw:.2f}: js {j['pct']:5.1f}% ({j['covered']}/{j['occupied']})"
          f"  py {py['coverage_pct']:5.1f}% ({py['covered_cells']}/{py['occupied_cells']})"
          f"  perClass js={j['perClass']} py={pcls}")

print(f"\n{len(CONFIGS) - fails}/{len(CONFIGS)} parity")
sys.exit(1 if fails else 0)
