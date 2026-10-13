"""Trace-to-grammar extraction checks. Run: python tests/test_extract.py"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace2grammar.extract import (
    ExtractionConfig,
    KINDS,
    extract,
    normalize_trace,
)
from trace2grammar.report import (
    ascii_timeline,
    atlas_html,
    atlas_svg,
    conversation_report,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")

checks = []


def check(name, ok):
    checks.append((name, ok))
    print(("PASS" if ok else "FAIL"), "-", name)


def load(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


# --- 1. every fixture parses into a canonical event stream -------------------
for fname, fmt in [
    ("otel_genai_trace.json", "otel"),
    ("otel_genai_abandon.json", "otel"),
    ("openinference_trace.json", "openinference"),
    ("langsmith_run.json", "langsmith"),
    ("langsmith_run_handoff.json", "langsmith"),
]:
    obj = load(fname)
    events = normalize_trace(obj, fmt)
    ok = len(events) > 0 and all(e["kind"] in KINDS for e in events)
    ts = [e["t_ms"] for e in events]
    ok = ok and ts == sorted(ts)
    check(f"normalize {fname} ({fmt}): {len(events)} events, sorted, valid kinds", ok)

# --- 2. contained story: otel fixture with one timeout retry -----------------
otel = load("otel_genai_trace.json")
r = extract(otel, "otel")
check("otel contained: outcome theta_cont", r["outcome"] == "theta_cont")
check("otel contained: cycle_depth >= 1 (timeout retry counted)", r["cycle_depth"] >= 1)
states = [w["state"] for w in r["walk"]]
check(
    "otel contained: passes q_auth and q_tool",
    "q_auth" in states and "q_tool" in states,
)
check(
    "otel contained: retry visible as tool-fail transition",
    any(w["transition_type"] == "tool-fail" for w in r["walk"]),
)
check("otel contained: starts at q_greet", states[0] == "q_greet")

# --- 3. abandonment: hangup mid-auth ------------------------------------------
aban = load("otel_genai_abandon.json")
r2 = extract(aban, "otel")
check("abandon: outcome theta_aban", r2["outcome"] == "theta_aban")
check("abandon: walk ends in terminal", r2["walk"][-1]["state"] == "theta_aban")
check(
    "abandon: transition type is hangup", r2["walk"][-1]["transition_type"] == "hangup"
)

# --- 4. langsmith abandonment (silence hangup after user msg) -----------------
ls = load("langsmith_run.json")
r3 = extract(ls, "langsmith")
check("langsmith abandon: outcome theta_aban", r3["outcome"] == "theta_aban")

# --- 5. handoff fixtures (openinference + langsmith) ---------------------------
oi = load("openinference_trace.json")
r4 = extract(oi, "openinference")
check("openinference handoff: outcome theta_hand", r4["outcome"] == "theta_hand")
check("openinference handoff: auth-fail retry counted (D>=1)", r4["cycle_depth"] >= 1)
check(
    "openinference handoff: escalation transition present",
    any(w["transition_type"] == "escalation" for w in r4["walk"]),
)

lsh = load("langsmith_run_handoff.json")
r5 = extract(lsh, "langsmith")
check("langsmith handoff: outcome theta_hand", r5["outcome"] == "theta_hand")
check("langsmith handoff: timeout retry counted (D>=1)", r5["cycle_depth"] >= 1)

# --- 6. config adaptability ----------------------------------------------------
custom = ExtractionConfig(auth_tool_patterns=("kyc",))
r6 = extract(otel, "otel", custom)
check(
    "config: swapping auth patterns changes pairing (verify no longer auth)",
    "q_auth" not in [w["state"] for w in r6["walk"]],
)

# --- 7. reports + atlas ---------------------------------------------------------
rep = conversation_report(otel, "otel")
check("report: dwell times present", len(rep["dwell_ms"]) > 0)
check("report: duration positive", rep["duration_ms"] > 0)
tl = ascii_timeline(rep)
check("ascii timeline renders", isinstance(tl, str) and len(tl) > 10)
reports = [
    conversation_report(load(f), fm)
    for f, fm in [
        ("otel_genai_trace.json", "otel"),
        ("otel_genai_abandon.json", "otel"),
        ("openinference_trace.json", "openinference"),
        ("langsmith_run.json", "langsmith"),
        ("langsmith_run_handoff.json", "langsmith"),
    ]
]
svg = atlas_svg(reports)
check(
    "atlas SVG: starts with <svg> and has one polyline per conversation",
    svg.startswith("<svg") and svg.count("<polyline") == len(reports),
)
html_out = atlas_html(reports)
check(
    "atlas HTML: standalone page with table",
    "<table>" in html_out and "svg" in html_out,
)
with open(os.path.join(ROOT, "atlas_demo.html"), "w", encoding="utf-8") as fh:
    fh.write(html_out)
check(
    "atlas_demo.html written (open in browser)",
    os.path.exists(os.path.join(ROOT, "atlas_demo.html")),
)

# --- summary ---------------------------------------------------------------------
fails = [n for n, ok in checks if not ok]
print()
if fails:
    print(f"{len(fails)} CHECKS FAILED")
    sys.exit(1)
print("ALL CHECKS PASSED")
