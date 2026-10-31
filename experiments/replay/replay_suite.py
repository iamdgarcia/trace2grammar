"""replay_suite — property suite for the tau-bench replay substrate (assert-based, exit 0 = PASS).

P1 deterministic reset: same task_index twice -> identical instruction.
P2 repair flips: GT repair on failed seed 1 -> reward 1.0.
P3 wrong-action control: rejected-variant exchange stays 0.0.
P4 reward reproducible: same repair twice -> same reward.
P5 omission classification stable: all-k flips for seeds 1/2/4 (from g4_results.json).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g3_separation import build_env, correct_acts, wrong_acts, repair_at, finish  # noqa: E402
from g4_failure_localization import repair_gt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

def p1():
    i1 = build_env(1).task.instruction
    i2 = build_env(1).task.instruction
    assert i1 == i2, "reset is not deterministic"

def p2():
    r, _ = repair_gt(1, 0)
    assert r == 1.0, f"GT repair should flip seed 1, got {r}"

def p3():
    r = repair_at(1, 3, lambda e, t: wrong_acts(e, t, correct_acts(e, t)))
    assert r == 0.0, f"wrong-action control should NOT flip, got {r}"

def p4():
    a, _ = repair_gt(2, 0)
    b, _ = repair_gt(2, 0)
    assert a == b == 1.0, f"reward not reproducible: {a} vs {b}"

def p5():
    g = json.load(open(os.path.join(HERE, "g4_results.json"), encoding="utf-8"))
    for t in (1, 2, 4):
        assert g["results"][str(t)]["type"] == "omission", f"seed {t} classification changed"

if __name__ == "__main__":
    checks = [p1, p2, p3, p4, p5]
    for c in checks:
        c()
        print(f"PASS {c.__name__}")
    print(f"\nreplay_suite: {len(checks)}/{len(checks)} PASS")
    sys.exit(0)
