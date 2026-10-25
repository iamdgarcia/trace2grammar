import os
# Parity check: tau_port.js (JS) vs trace2grammar/extract.py (python) on the golden
# outputs of the 7 shipped fixtures. Any mismatch = port bug; the Space claims parity.
import json, subprocess, os

ROOT = os.path.dirname(os.path.abspath(__file__))  # repo root: portable, no hf/ layout assumed
GOLDEN = json.load(open(ROOT + "/runtime/tau_golden.json", encoding="utf-8"))

runner = """
const fs = require('fs');
const TAU = require(process.argv[2]);
const events = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const cfg = process.argv[4]
  ? Object.assign({}, TAU.CFG, {auth_tool_patterns: JSON.parse(process.argv[4])})
  : TAU.CFG;
const r = TAU.inferStates(events, cfg);
process.stdout.write(JSON.stringify({walk: r.walk, outcome: r.outcome, cycle_depth: r.cycle_depth, error_count: r.error_count, silent_failure_candidate: r.silent_failure_candidate}));
"""
runner_js = ROOT + "/runtime/_parity_runner_%d.js" % os.getpid()
open(runner_js, "w").write(runner)

fails = 0
for fname, gold in GOLDEN.items():
    ev_path = ROOT + "/runtime/_parity_events_%d.json" % os.getpid()
    json.dump(gold["events"], open(ev_path, "w", encoding="utf-8"))
    tb = '["verify","identity","auth","otp","login","find_user"]' if 'taubench' in fname else ''
    r = subprocess.run(["node", runner_js,
                        ROOT + "/autopsy_space/tau_port.js", ev_path, tb],
                       capture_output=True, text=True, timeout=60,
                       cwd=ROOT)
    if r.returncode != 0:
        print(fname, "NODE ERROR:", r.stderr[:200]); fails += 1; continue
    got = json.loads(r.stdout)
    ok = (got["walk"] == gold["walk"] and got["outcome"] == gold["outcome"]
          and got["cycle_depth"] == gold["cycle_depth"] and got["error_count"] == gold["error_count"]
          and got["silent_failure_candidate"] == gold["silent_failure_candidate"])
    print(("PASS " if ok else "FAIL "), fname, got["outcome"], "D=", got["cycle_depth"])
    if not ok:
        fails += 1
        if got["outcome"] != gold["outcome"]:
            print("   outcome:", got["outcome"], "vs", gold["outcome"])
        if got["walk"] != gold["walk"]:
            for i, (a, b) in enumerate(zip(got["walk"], gold["walk"])):
                if a != b:
                    print("   walk[", i, "]:", a, "vs", b); break
            print("   len:", len(got["walk"]), "vs", len(gold["walk"]))
print("PARITY:", "OK" if fails == 0 else str(fails) + " FAILURES")
os.remove(ev_path)
os.remove(runner_js)
