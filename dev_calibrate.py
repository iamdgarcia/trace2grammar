#!/usr/bin/env python3
"""DEV-CALIBRATION of tau close vocabulary (clean version of the PH-2 idea).

In-repo use: the dev transcripts are NOT committed (test-only fixture policy,
size); regenerate them first with sgd_to_events.py (SGD_SRC=<raw dev JSON>,
SGD_OUT=sgd_dev_transcript), then run this script. Patterns land in
sgd_dev_calibration.json (committed) and sgd_tau_calib.py measures the test
split with them.

Method contract (the fix for the contaminated PH-2):
  1. Run tau VERBATIM (cfg default) over the SGD dev split (2,482 dialogues).
  2. Collect the FINAL assistant turn of every dialogue the walker calls
     theta_aban (i.e. it never recognized a close). Extract the last sentence
     of that turn, normalize, and count final n-grams -> data-driven clusters
     of unrecognized farewell shapes. NOTHING from test is used here.
  3. Derive candidate regex patterns from the dev clusters ONLY; each pattern
     is documented with how many dev farewells it captures (single-covered
     removed first, then union).
  4. dev_calibration.json holds the exact pattern list + per-pattern dev
     captures. The TEST measurement is done by the runner with the calibrated
     cfg (see REPORT-dev-calib.md): the calibration set (dev) and the test set
     never mix.

tau runs VERBATIM from builds/trace2grammar; patterns travel through the
declared ExtractionConfig mechanism (no code edits to shipped tau).
"""
import json
import os
import re
import sys
from collections import Counter

# artifact package root: scripts live at the repo root; out-of-repo layouts
# set the T2G_ROOT env var to the artifact checkout
ROOT = os.environ.get("T2G_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "sgd_dev_transcript")
meta = json.load(open(os.path.join(SRC, "_meta.json"), encoding="utf-8"))

# --- 1. tau default over dev -------------------------------------------------
dev_rows = {}
aban_final = []
for did, m in sorted(meta.items()):
    txt = open(os.path.join(SRC, f"{did}.txt"), encoding="utf-8").read()
    r = extract(txt, "transcript")
    walk = r["walk"]
    reached = any(w["state"] == "q_close" for w in walk)
    dev_rows[did] = {"outcome": r["outcome"], "reached_close": reached,
                     "gt": m["gt_effective_call"]}
    if r["outcome"] == "theta_aban":
        # final assistant turn = last 'assistant:' line
        lines = [l for l in txt.strip().splitlines() if l.startswith("assistant:")]
        if lines:
            last = lines[-1][len("assistant:"):].strip()
            aban_final.append({"id": did, "text": last, "gt": m["gt_effective_call"]})

oc = Counter(v["outcome"] for v in dev_rows.values())
print(f"dev outcomes: {dict(oc)} | theta_aban with final assistant turn: {len(aban_final)}")

# --- 2. final-sentence n-gram clustering (dev only) --------------------------
def last_sentence(text):
    parts = re.split(r"[.!?]+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else text.strip()

def toks(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()

finals = [last_sentence(a["text"]) for a in aban_final]
ngrams = Counter()
for s in finals:
    t = toks(s)
    for n in (2, 3, 4):
        for i in range(len(t) - n + 1):
            ngrams[" ".join(t[i:i + n])] += 1

print("\nTop trailing n-grams among unrecognized closes (dev):")
for g, c in ngrams.most_common(40):
    if c >= 3:
        print(f"  {c:4} | {g}")

# --- 3. derive patterns from dev clusters (documented, per-pattern captures) --
# Derivation rule: farewell-shaped clusters only (leavetaking vocabulary),
# minimal generalization of each dev cluster. "thank you" clusters (56) are
# EXCLUDED on purpose: close_assistant_patterns match ANY assistant line, and
# "thank you for waiting/patience" mid-conversation would create false closes.
# Greedy attribution: a dev dialogue unrecognized by the default cfg is
# attributed to the FIRST pattern (in this order) that matches its final
# assistant sentence; "unattributed" = no candidate matches it.
DEV_PATTERNS = [
    (r"\bhave (yourself )?an? (wonderful|pleasant|fantastic|lovely|beautiful|awesome|cheerful)\b",
     "have (yourself) a(n) <adj-despedida> ..."),
    (r"\benjoy (your|the rest of your)\b",
     "enjoy your <...> / enjoy the rest of your <...>"),
    (r"\bsee you (later|soon)\b",
     "see you later/soon"),
    (r"\btake care\b",
     "take care"),
    (r"\bbye\b",
     "bye / bye now (goodbye ya cubierto por el default)"),
    (r"\bcheers\b",
     "cheers (leavetaking britanico)"),
]

def final_assistant_text(txt):
    lines = [l for l in txt.strip().splitlines() if l.startswith("assistant:")]
    return lines[-1][len("assistant:"):].strip() if lines else ""

attributed = Counter()
unattributed = []
aban_ids = {did for did, v in dev_rows.items() if v["outcome"] == "theta_aban"}
for did in sorted(aban_ids):
    txt = open(os.path.join(SRC, f"{did}.txt"), encoding="utf-8").read()
    last = final_assistant_text(txt)
    hit = None
    for pat, label in DEV_PATTERNS:
        if re.search(pat, last, re.I):
            hit = pat
            break
    if hit:
        attributed[hit] += 1
    else:
        unattributed.append({"id": did, "final": last[:120]})

total_recognized = sum(attributed.values())
print("\nPer-pattern dev captures (theta_aban dialogues now recognized):")
for pat, label in DEV_PATTERNS:
    print(f"  {attributed.get(pat,0):4} | {label}")
print(f"  dev theta_aban: {len(aban_ids)} | recognized by calibration: {total_recognized} "
      f"({100.0*total_recognized/max(1,len(aban_ids)):.1f}%) | unattributed: {len(unattributed)}")
print("sample unattributed finals:")
for u in unattributed[:12]:
    print(f"   {u['id']}: {u['final']}")

json.dump({
    "method": "patterns derived from SGD dev split ONLY (2,482 dialogues); "
              "test never touched during calibration; 'thank you' excluded (mid-"
              "conversation false closes)",
    "n_dev": len(dev_rows),
    "dev_outcomes": dict(oc),
    "n_theta_aban": len(aban_ids),
    "patterns": [
        {"regex": p, "shape": label, "dev_captures": attributed.get(p, 0)}
        for p, label in DEV_PATTERNS
    ],
    "dev_recognized_by_calibration": total_recognized,
    "dev_unattributed": unattributed[:60],
}, open(os.path.join(BASE, "dev_calibration.json"), "w", encoding="utf-8"),
    indent=1, ensure_ascii=False)
print("-> dev_calibration.json")
