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
    ("taubench_episodes.json", "taubench"),
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

# --- 7b. otel split call/result spans: correlate by gen_ai.tool.call.id --------
# otel-genai #489/#525: execute_tool spans carry a provider tool call ID; newer
# exporters emit call and result as separate spans sharing it. Without id
# correlation each span became a self-contained phantom (2 tool_calls, 2 results).
split = load("otel_genai_split_spans.json")
ev_split = normalize_trace(split, "otel")
check(
    "otel split spans: one tool_call (id correlation, no phantom duplicate)",
    sum(1 for e in ev_split if e["kind"] == "tool_call") == 1,
)
check(
    "otel split spans: one tool_result paired via call id",
    sum(1 for e in ev_split if e["kind"] == "tool_result") == 1,
)
check(
    "otel split spans: result timestamped after call",
    [e["t_ms"] for e in ev_split if e["kind"] == "tool_call"][0]
    < [e["t_ms"] for e in ev_split if e["kind"] == "tool_result"][0],
)
r_split = extract(split, "otel")
check(
    "otel split spans: walker sees one clean auth step (no false retry cycle)",
    r_split["cycle_depth"] == 0
    and "q_auth" in [w["state"] for w in r_split["walk"]]
    and not any(w["transition_type"] == "tool-fail" for w in r_split["walk"]),
)

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

# --- tau-bench: environment-verified outcomes vs heuristic walker ------------
tb = load("taubench_episodes.json")
tb_pass = extract(tb[0], "taubench")
tb_fail = extract(tb[1], "taubench")
check("taubench pass episode: outcome theta_cont", tb_pass["outcome"] == "theta_cont")
check(
    "taubench fail episode: outcome theta_hand (public analogue)",
    tb_fail["outcome"] == "theta_hand",
)
check(
    "taubench: tool calls extracted from antml markup",
    any(e["kind"] == "tool_call" for e in tb_pass["events"]),
)
check(
    "taubench: auth channel recognized (find_user_* tools)",
    "q_auth" in [w["state"] for w in tb_pass["walk"]],
)
check(
    "taubench: multi-episode list ordered (fail episode after pass)",
    extract(list(tb), "taubench")["events"][-1]["attrs"].get("task_id")
    == tb[1]["task_id"],
)

# --- transcript dialect + sample set ----------------------------------------------
SAMPLES = os.path.join(ROOT, "samples")

_tx = """assistant: Hello, how can I help?
user: I need a replacement card.
tool(verify_identity): ok
tool(order_card): ok
assistant: Done. Anything else?
user: No, thanks.
assistant: Thanks for calling, have a great day!"""
_r = extract(_tx, "transcript")
check("transcript: parses roles and tools", _r["outcome"] == "theta_cont")
check(
    "transcript: tool events paired",
    sum(1 for e in _r["events"] if e["kind"] == "tool_call") == 2
    and sum(1 for e in _r["events"] if e["kind"] == "tool_result") == 2,
)

_tx_hand = """assistant: Hello, how can I help?
user: I want a person.
tool(escalate_to_human): ok
assistant: Connecting you, please hold."""
check("transcript: handoff tool drives theta_hand", extract(_tx_hand, "transcript")["outcome"] == "theta_hand")

# every sample transcript classifies into a terminal, and the sim batch is all-contained
_sample_outcomes = {}
for _f in sorted(os.listdir(SAMPLES)):
    if _f.endswith(".txt"):
        with open(os.path.join(SAMPLES, _f), encoding="utf-8") as fh:
            _sample_outcomes[_f[:-4]] = extract(fh.read(), "transcript")["outcome"]
check("samples: 580 transcripts, all classified", len(_sample_outcomes) == 580)
check("samples: sim batch all theta_cont (collapse config)",
      all(_sample_outcomes[k] == "theta_cont" for k in _sample_outcomes if k.startswith("sim")))
check("samples: real set spans 3 outcome classes",
      {v for k, v in _sample_outcomes.items() if not k.startswith("sim")} == {"theta_cont", "theta_aban", "theta_hand"})

# --- summary ---------------------------------------------------------------------
fails = [n for n, ok in checks if not ok]
print()
if fails:
    print(f"{len(fails)} CHECKS FAILED")
    sys.exit(1)
print("ALL CHECKS PASSED")
