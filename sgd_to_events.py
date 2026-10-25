#!/usr/bin/env python3
"""Convert the SGD (Schema-Guided Dialogue, dstc8) TEST split into the tau
transcript dialect, mirroring multiwoz_fetch.py's data-true contract.

In-repo use: download the dataset's split folder (test/ or dev/) from
github.com/google-research-datasets/dstc8-schema-guided-dialogue, then run
with SGD_SRC pointing at the raw JSON folder; defaults assume the layout next
to this script (raw in sgd_test/, transcripts out to sgd_test_transcript/,
which is the committed fixture for the test split).

Source: google-research-datasets/dstc8-schema-guided-dialogue (master), test/
split, already downloaded to sgd_validation/sgd_test/ (34 files, 404 past 034).
SGD is third-party human-human wizard-of-oz task dialogue with NATIVE API-call
annotations: every SYSTEM turn frame may carry service_call {method, parameters}
and service_results. We reformat only; no tau tables were tuned for SGD.

Mapping decisions (all documented here, provenance-honest like make_samples):
1. USER turn -> 'user: <text>'; SYSTEM turn -> 'assistant: <text>'.
2. service_call in a SYSTEM turn -> 'tool(<method>): ok n_results=<k>' line
   BEFORE the assistant text of the same turn (the call precedes the spoken
   reply, same event order as the otel dialect).
3. SGD does NOT annotate API errors: a failed call is structurally visible
   (empty service_results + text like "your reservation could not be made")
   but carries no error mark. All calls are therefore emitted as ': ok' with
   the result count; the walker cannot see the failure. Documented limitation.
4. state.active_intent (USER frames) is NOT emitted as text: the walker derives
   q_intent from the first user turn; injecting the label would contaminate the
   natural transcript. The label lives in _meta.json for reporting only.
5. Domain label = services[0] as returned by the dataset (no merging, no
   renaming); multi-service dialogues keep their full services list in meta.
6. handoff: SGD has no handoff concept; no method name should match the tau
   handoff_tool_patterns (checked in the report, expected handoff=0).

Output: sgd_validation/sgd_test_transcript/<did>.txt + _meta.json with
structural ground truth per dialogue:
  gt_effective_call = any frame with service_call AND non-empty service_results
  (an effective call = the simulated API actually returned results; the 4-file
  sample confirmed empty results == real failures, e.g. "reservation could
  not be made").
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Split override (dev-calibration wave 25-oct): SGD_SRC/SGD_OUT env vars select the
# split to convert; defaults preserve the original test behavior verbatim.
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("SGD_SRC", os.path.join(BASE, "sgd_test"))
OUT = os.environ.get("SGD_OUT", os.path.join(BASE, "sgd_test_transcript"))
os.makedirs(OUT, exist_ok=True)

meta = {}
n_files = 0
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".json"):
        continue
    n_files += 1
    batch = json.load(open(os.path.join(SRC, fn), encoding="utf-8"))
    for dia in batch:
        did = dia["dialogue_id"]
        lines = []
        n_calls = 0
        n_eff = 0
        gt = False
        for turn in dia["turns"]:
            spk = turn["speaker"]
            # SGD dstc8 names the spoken text 'utterance' (verified against the
            # raw files; 'text' does not exist on turns -> only tool lines).
            text = (turn.get("utterance") or turn.get("text") or "").strip()
            if spk == "SYSTEM":
                for f in turn["frames"]:
                    sc = f.get("service_call")
                    if sc:
                        n_calls += 1
                        res = f.get("service_results") or []
                        if res:
                            n_eff += 1
                            gt = True
                        lines.append(f"tool({sc['method']}): ok n_results={len(res)}")
                if text:
                    lines.append(f"assistant: {text}")
            else:
                if text:
                    lines.append(f"user: {text}")
        txt = "\n".join(lines) + "\n"
        with open(os.path.join(OUT, f"{did}.txt"), "w", encoding="utf-8") as fh:
            fh.write(txt)
        meta[did] = {
            "services": dia.get("services", []),
            "primary": (dia.get("services") or ["none"])[0],
            "n_turns": len(dia.get("turns", [])),
            "n_service_calls": n_calls,
            "n_effective": n_eff,
            "gt_effective_call": gt,
        }

with open(os.path.join(OUT, "_meta.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=1)
print(f"converted {len(meta)} dialogues from {n_files} files -> {OUT}")
from collections import Counter
gt_c = Counter(v["gt_effective_call"] for v in meta.values())
print("gt_effective_call:", dict(gt_c))
print("services seen:", len(set(v["primary"] for v in meta.values())))
