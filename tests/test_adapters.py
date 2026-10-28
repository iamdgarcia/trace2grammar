"""Adapter-registry checks: conversation -> grammar embedding. Run: python tests/test_adapters.py"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace2grammar import (
    ADAPTERS,
    PROGRESS_STATES,
    fit_basis,
    project_features,
    register_adapter,
    to_grammar,
)
from trace2grammar.coverage import pca2d, znorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")

checks = []


def check(name, ok):
    checks.append((name, ok))
    print(("PASS" if ok else "FAIL"), "-", name)


PIPECAT_CONV = [
    {"role": "assistant", "content": "Hi, how can I help?"},
    {"role": "user", "content": "I need to change my flight to Friday."},
    {"role": "tool_call", "name": "verify_identity", "arguments": {"dob": "1990-01-01"}, "cancelled": False},
    {"role": "assistant", "content": "Thanks, you are verified."},
    {"role": "tool_call", "name": "change_flight", "arguments": {"day": "Friday"}, "cancelled": False},
    {"role": "assistant", "content": "Done. Anything else?"},
    {"role": "user", "content": "No, thanks."},
    {"role": "assistant", "content": "Have a great day, goodbye!"},
]

# ---- the registry: one interface, every transport -------------------------
check("registry has the 5 paper dialects + pipecat-evals",
      {"otel", "openinference", "langsmith", "taubench", "transcript", "pipecat-evals"} <= set(ADAPTERS))

emb = to_grammar(PIPECAT_CONV, "pipecat-evals", meta={"domain": "airline", "model": "demo"})
check("walk starts at q_greet and ends at q_close", emb["walk"][0] == "q_greet" and emb["walk"][-1] == "q_close")
check("walk holds only progress states (outcome-free)",
      all(s in PROGRESS_STATES for s in emb["walk"]))
check("embedding carries no outcome key", "outcome" not in emb)
check("auth tool routed the walk through q_auth", "q_auth" in emb["walk"])
check("tool call routed the walk through q_tool", "q_tool" in emb["walk"])
check("meta round-trips (domain + model)", emb["meta"] == {"domain": "airline", "model": "demo"})
check("feature vector is 20-dim", len(emb["features"]) == 20)
check("cycle_depth is a non-negative int", isinstance(emb["cycle_depth"], int) and emb["cycle_depth"] >= 0)

# ---- the pipecat results.jsonl container ----------------------------------
record = {"bot": "voice-bot-1", "name": "sim-01", "scenario": "change_flight", "kind": "simulation",
          "passed": True, "turns": 4, "messages": PIPECAT_CONV, "duration_ms": 41234}
emb2 = to_grammar(record, "pipecat-evals")
check("results.jsonl record ('messages' key) gives the SAME walk as the bare list", emb2["walk"] == emb["walk"])
rec_custom = {"msgs": PIPECAT_CONV, "note": "custom container"}
emb3 = to_grammar(rec_custom, "pipecat-evals", key="msgs")
check("custom container key via key=...", emb3["walk"] == emb["walk"])
record_noreply = {"bot": "b", "messages": [PIPECAT_CONV[0], PIPECAT_CONV[2]]}  # auth tool called, never answered
emb4 = to_grammar(record_noreply, "pipecat-evals")
check("tool_call-only record keeps the tool step in the walk", "q_auth" in emb4["walk"])
check("cancelled flag rides in attrs (no event invented)", emb4["walk"][1] == "q_auth")

# ---- the abstraction: a third transport is one function -------------------
def _toy(raw, cfg):
    return [
        {"t_ms": 0, "kind": "user_msg", "name": None, "attrs": {"content": "hello"}},
        {"t_ms": 1000, "kind": "tool_call", "name": "lookup", "attrs": {}},
    ]

register_adapter("toy", _toy)
check("register_adapter plugs a new transport into the SAME machine",
      to_grammar({"x": 1}, "toy")["walk"][:3] == ["q_greet", "q_intent", "q_tool"])

# ---- existing dialects still extract (regression) --------------------------
otel = json.load(open(os.path.join(FIX, "otel_genai_trace.json"), encoding="utf-8"))
ls = json.load(open(os.path.join(FIX, "langsmith_run.json"), encoding="utf-8"))
check("otel fixture -> walk via registry", to_grammar(otel, "otel")["walk"][0] == "q_greet")
check("langsmith fixture -> walk via registry", to_grammar(ls, "langsmith")["walk"][0] == "q_greet")
check("transcript -> walk via registry",
      to_grammar("assistant: hello, welcome\nuser: I need help\nassistant: thanks for calling, have a great day, goodbye\n",
                 "transcript")["walk"][-1] == "q_close")

# ---- the 2D plane: fit on features, project, parity vs coverage.pca2d ------
rows = [emb["features"],
        to_grammar(PIPECAT_CONV[:-5], "pipecat-evals")["features"],
        to_grammar(PIPECAT_CONV[:-7], "pipecat-evals")["features"]]
basis = fit_basis(rows)
ok_par = True
for i, row in enumerate(rows):
    px, py = project_features(row, basis)
    rx, ry = pca2d(znorm(rows))[i]
    if abs(px - rx) > 1e-9 or abs(py - ry) > 1e-9:
        ok_par = False
check("project_features == coverage.pca2d on the fitting corpus (parity)", ok_par)

# ---- the CLI end-to-end: conversation + domain -> embedding ----------------
import tempfile

tmpdir = tempfile.mkdtemp()
conv_path = os.path.join(tmpdir, "conv.json")
json.dump(PIPECAT_CONV, open(conv_path, "w", encoding="utf-8"))
jsonl_path = os.path.join(tmpdir, "results.jsonl")
with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(record) + "\n")
    fh.write(json.dumps(record_noreply) + "\n")
basis_path = os.path.join(tmpdir, "basis.json")

r = subprocess.run(
    [sys.executable, os.path.join(ROOT, "embed_conversation.py"), conv_path,
     "--fmt", "pipecat-evals", "--domain", "airline", "--meta", "model=gpt-4o"],
    capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
try:
    out1 = json.loads(r.stdout)
except Exception:
    out1 = None
check("CLI single conversation -> one embedding JSON",
      r.returncode == 0 and out1 and out1["meta"]["domain"] == "airline" and out1["walk"][0] == "q_greet")

r = subprocess.run(
    [sys.executable, os.path.join(ROOT, "embed_conversation.py"), jsonl_path,
     "--fmt", "pipecat-evals", "--domain", "airline", "--save-basis", basis_path],
    capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
try:
    out2 = json.loads(r.stdout)
except Exception:
    out2 = None
check("CLI results.jsonl -> one embedding per line (batch array)",
      r.returncode == 0 and isinstance(out2, list) and len(out2) == 2)
check("CLI --save-basis writes a usable basis",
      os.path.exists(basis_path) and "v1" in json.load(open(basis_path, encoding="utf-8")))

r = subprocess.run(
    [sys.executable, os.path.join(ROOT, "embed_conversation.py"), conv_path,
     "--fmt", "pipecat-evals", "--basis", basis_path],
    capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
try:
    out3 = json.loads(r.stdout)
except Exception:
    out3 = None
check("CLI --basis projects onto the fitted plane (2d present)",
      r.returncode == 0 and out3 and isinstance(out3.get("2d"), list) and len(out3["2d"]) == 2)

r = subprocess.run([sys.executable, os.path.join(ROOT, "embed_conversation.py"), "--self-check"],
                   capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
check("CLI --self-check exits 0", r.returncode == 0 and "SELF-CHECK OK" in r.stdout)

n_fail = sum(1 for _, ok in checks if not ok)
print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
sys.exit(1 if n_fail else 0)
