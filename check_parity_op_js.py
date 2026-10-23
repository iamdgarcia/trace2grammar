#!/usr/bin/env python3
"""Parity check #2: per-operativa reach numbers (JS demo vs python).

The dashboard shows, per operativa, region coverage (kernel) and instance coverage
(kernel-free) of that operativa's real mass against the active sim set. This script
recomputes the same numbers with the python implementation and asserts agreement.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trace2grammar"))
from trace2grammar.coverage import coverage as py_coverage, instance_coverage  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
blob = json.loads(open(f"{ROOT}/atlas_data.js", encoding="utf-8").read()
                  .removeprefix("var ATLAS = ").removesuffix(";"))

real_pts = [p for p in blob["points"] if p["kind"] == "real"]
real = [[p["x"], p["y"]] for p in real_pts]
ops = sorted({p["op"] for p in real_pts})
control = [[p["x"], p["y"]] for p in blob["points"]
           if p["kind"] == "sim" and not p["id"].startswith("sim2")]
sim2 = [[p["x"], p["y"]] for p in blob["points"] if p["id"].startswith("sim2")]

CONFIGS = [("control", 0.12, control), ("diverse", 0.02, sim2)]

node_script = """
const path = require('path');
const HERE = '__HERE__';
const COVERJS = require(path.join(HERE, 'cover_js.js'));
const blob = JSON.parse(require('fs').readFileSync(
  path.join(HERE, 'atlas_data.js'), 'utf8')
  .replace(/^var ATLAS = /, '').replace(/;$/, ''));
const real = blob.points.filter(p => p.kind === 'real');
const control = blob.points.filter(p => p.kind === 'sim' && !p.id.startsWith('sim2')).map(p => [p.x, p.y]);
const sim2 = blob.points.filter(p => p.id.startsWith('sim2')).map(p => [p.x, p.y]);
const byOp = {};
real.forEach(p => (byOp[p.op] = byOp[p.op] || []).push([p.x, p.y]));
const out = [];
for (const [name, bw, sim] of [['control', 0.12, control], ['diverse', 0.02, sim2]]) {
  for (const op of Object.keys(byOp).sort()) {
    const sub = byOp[op];
    const c = COVERJS.coverage(sub, sim, null, 24, 0.05, bw, 2);
    const inst = COVERJS.instanceCoverage(sub, sim);
    out.push({ name, op, region: c.coveragePct, inst: inst.instanceCoverage });
  }
}
console.log(JSON.stringify(out));
"""
r = subprocess.run(["node", "-e", node_script.replace("__HERE__", ROOT.replace(chr(92), "/"))], capture_output=True, encoding="utf-8", timeout=120)
if r.returncode != 0:
    print("NODE FAIL:\n", r.stderr)
    sys.exit(1)
js = json.loads(r.stdout)

fails = 0
jsmap = {(j["name"], j["op"]): j for j in js}
for name, bw, sim in CONFIGS:
    for op in ops:
        sub = [[p["x"], p["y"]] for p in real_pts if p["op"] == op]
        j = jsmap[(name, op)]
        py = py_coverage(sub, sim, None, 24, 0.05, bw_frac=bw)
        pyi = instance_coverage(sub, sim)
        ok = (j["region"] == py["coverage_pct"] and j["inst"] == pyi["instance_coverage"])
        if not ok:
            fails += 1
            print(f"[FAIL] {name}@{bw} {op}: js {j['region']}/{j['inst']} "
                  f"py {py['coverage_pct']}/{pyi['instance_coverage']}")
total = len(CONFIGS) * len(ops)
print(f"{total - fails}/{total} per-op parity")
sys.exit(1 if fails else 0)
