#!/usr/bin/env python3
"""Build station_data.js for iamdgarcia/silent-failure-radio.

The radio broadcasts the final assistant line of REAL conversations that scored
failed, next to identical-sounding closings of delivered ones. Selection
criteria (predeclared, published walker verbatim):

  FAILED (tau):    reward == 0.0 AND outcome == theta_cont (official eval runs;
                   polite close on a failed task). D>=2 first (hard cases).
  FAILED (sgd):    the agent textually CONFESSED the failure and still closed
                   politely (the 5 'cortesia sin solucion' dialogs, gt False).
  DELIVERED (tau): reward > 0.0 AND outcome == theta_cont.
  DELIVERED (sgd): outcome theta_cont AND reached_close AND gt_effective_call
                   AND no silent_failure flag.

Fixed seed sampling; mojibake apostrophe cleanup for speech. HERE-relative.
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # .../hf
TAU_DIR = os.path.join(ROOT, "builds", "where-conversations-die")
SGD_ROWS = os.path.join(ROOT, "runtime", "sgd_validation", "rows.json")
SGD_TX = os.path.join(ROOT, "builds", "trace2grammar", "sgd_test_transcript")
T2G = os.path.join(ROOT, "builds", "trace2grammar")

SEED = 20261026
N_PER_POOL = 45

sys.path.insert(0, T2G)
from trace2grammar.extract import extract_transcript  # noqa: E402


def clean(t):
    """Mojibake apostrophes and transcript artifacts that would be read aloud."""
    t = re.sub(r"([A-Za-z])\?([A-Za-z])", r"\1'\2", t)   # don?t -> don't
    t = t.replace("###STOP###", "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def speech_ok(t):
    return 8 <= len(t) <= 280 and re.search(r"[A-Za-z]", t)


def final_lines(txt):
    alines, ulines, tools = [], [], []
    for l in txt.splitlines():
        if l.startswith("assistant:"):
            alines.append(l[len("assistant:"):].strip())
        elif l.startswith("user:"):
            ulines.append(l[len("user:"):].strip())
        elif l.startswith("tool("):
            m = re.match(r"tool\(([^)]+)\):\s*(ok|error|call)", l)
            if m and m.group(2) in ("ok", "error"):
                tools.append(m.group(1) + " " + m.group(2))
    return alines, ulines, tools


def main():
    out = []

    # ---- tau ----
    blob = open(os.path.join(TAU_DIR, "atlas_data.js"), encoding="utf-8").read()
    atlas = json.loads(re.search(r"var ATLAS = ({.*});?\s*$", blob, re.S).group(1))
    eps = [e for e in (atlas.get("eps") or atlas.get("points")) if e.get("kind") == "real"]
    rng = random.Random(SEED)

    failed = [e for e in eps if e.get("reward", 0) == 0 and e.get("outcome") == "theta_cont"]
    delivered = [e for e in eps if e.get("reward", 0) > 0 and e.get("outcome") == "theta_cont"]
    # D>=2 first: the early-warning hard cases
    failed.sort(key=lambda e: -e.get("D", 0))
    picked_fail = failed[:30] + rng.sample(failed[30:], min(15, len(failed) - 30))
    picked_good = rng.sample(delivered, min(N_PER_POOL, len(delivered)))

    for e, cat in ([(e, "failed") for e in picked_fail] + [(e, "delivered") for e in picked_good]):
        p = os.path.join(TAU_DIR, e["tfile"])
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        alines, ulines, tlines = final_lines(txt)
        if not alines:
            continue
        line = clean(alines[-1])
        prev = clean(ulines[-1]) if ulines else ""
        if not speech_ok(line):
            continue
        out.append({
            "id": e["id"], "corpus": "tau", "cat": cat,
            "line": line, "prev": prev[:140],
            "ask": clean(ulines[0])[:160] if ulines else "",
            "ledger": tlines[:8],
            "model": e.get("model"), "dom": e.get("domain"),
            "D": e.get("D", 0), "n": e.get("n_states"),
            "walk": e.get("walk", [])[-6:],
            "task": e.get("task_id"),
            "truth": "the official scoreboard (tau-bench): the user's task was NOT completed (reward 0)" if cat == "failed"
                     else "the official scoreboard (tau-bench): the user's task was completed (reward 1)",
        })

    # ---- SGD ----
    rows = json.load(open(SGD_ROWS, encoding="utf-8"))
    sgd_fail, sgd_good = [], []
    for r in rows:
        if r["outcome"] != "theta_cont":
            continue
        p = os.path.join(SGD_TX, r["id"] + ".txt")
        if not os.path.exists(p):
            continue
        res = extract_transcript(open(p, encoding="utf-8").read())
        if res.get("silent_failure_candidate") and not r["gt_effective_call"]:
            sgd_fail.append(r["id"])
        elif not res.get("silent_failure_candidate") and r["gt_effective_call"] and r["reached_close"]:
            sgd_good.append(r["id"])

    for did, cat in ([(d, "failed") for d in sgd_fail]
                     + [(d, "delivered") for d in rng.sample(sgd_good, min(15, len(sgd_good)))]):
        txt = open(os.path.join(SGD_TX, did + ".txt"), encoding="utf-8").read()
        alines, ulines, tlines = final_lines(txt)
        if not alines:
            continue
        line = clean(alines[-1])
        prev = clean(ulines[-1]) if ulines else ""
        if not speech_ok(line):
            continue
        out.append({
            "id": did, "corpus": "sgd", "cat": cat,
            "line": line, "prev": prev[:140],
            "ask": clean(ulines[0])[:160] if ulines else "",
            "ledger": tlines[:8],
            "model": None, "dom": "SGD",
            "D": None, "n": None, "walk": [], "task": None,
            "truth": "the agent itself said the task failed mid-conversation, then said goodbye nicely"
                     if cat == "failed" else "the dataset's ground truth: the user's request was fulfilled",
        })

    rng.shuffle(out)
    # blind test: balance the two categories to the same count
    n_fail = sum(1 for t in out if t["cat"] == "failed")
    n_del = sum(1 for t in out if t["cat"] == "delivered")
    if n_del > n_fail:
        i = 0
        trimmed = []
        for t in out:
            if t["cat"] == "delivered" and i < n_del - n_fail:
                i += 1
                continue
            trimmed.append(t)
        out = trimmed
    n_fail = sum(1 for t in out if t["cat"] == "failed")
    doc = {
        "generated": "2026-10-26",
        "seed": SEED,
        "counts": {"tracks": len(out), "failed": n_fail, "delivered": len(out) - n_fail},
        "criteria": {
            "failed_tau": "reward==0.0 and walker outcome theta_cont (polite close on failed task); D>=2 first",
            "failed_sgd": "agent-confessed failure (gt_effective_call=false), polite close",
            "delivered_tau": "reward>0.0 and walker outcome theta_cont",
            "delivered_sgd": "theta_cont, reached_close, gt_effective_call, no zero-result tool",
        },
        "tracks": out,
    }
    path = os.path.join(HERE, "station_data.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("var STATION = " + json.dumps(doc, ensure_ascii=False) + ";\n")
    print("tracks:", len(out), "| failed:", n_fail, "| delivered:", len(out) - n_fail)
    print("->", path)


if __name__ == "__main__":
    main()
