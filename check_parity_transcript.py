# Parity check for the TRANSCRIPT dialect: python _normalize_transcript+infer_states
# vs tau_port.js normalizeTranscript+inferStates, on the trace2grammar sample files
# + crafted edge cases (handoff tool, continuation lines, tool error, alias roles).
# The 7/7 golden check (check_parity.py) covers otel/langsmith/taubench with REAL
# timestamps; this one covers the pasted-text path the autopsy Space actually runs.
import json, subprocess, os, sys

ROOT = "C:/Users/U77225D/Desktop/proyectos/hf"
sys.path.insert(0, ROOT + "/builds/trace2grammar")
from trace2grammar.extract import extract_transcript  # noqa: E402

runner = """
const fs = require('fs');
const TAU = require(process.argv[2]);
const text = fs.readFileSync(process.argv[3], 'utf8');
const norm = TAU.normalizeTranscript(text);
const r = TAU.inferStates(norm.events);
process.stdout.write(JSON.stringify({walk: r.walk, outcome: r.outcome,
  cycle_depth: r.cycle_depth, error_count: r.error_count, events: norm.events}));
"""
runner_path = ROOT + "/runtime/_parity_transcript_runner.js"
open(runner_path, "w", encoding="utf-8").write(runner)

SAMPLES_DIR = ROOT + "/builds/trace2grammar/samples"
cases = []
for fname in sorted(os.listdir(SAMPLES_DIR)):
    if fname.endswith(".txt"):
        cases.append((fname, open(os.path.join(SAMPLES_DIR, fname), encoding="utf-8").read()))

# crafted edge cases: each exercises a branch the samples may not
cases += [
    ("edge_handoff_tool.txt", "user: hola\nassistant: le paso con un agente\ntool(escalate_to_human): ok transferred"),
    ("edge_tool_error.txt", "user: mi saldo\nassistant: un momento\ntool(get_balance): error timeout after 30s\nuser: vaya"),
    ("edge_continuation.txt", "user: quiero saber\n  mi saldo y también\nassistant: claro, dime qué cuenta\nuser: la principal"),
    ("edge_no_terminal.txt", "assistant: hola\nuser: una consulta sobre la factura\ntool(lookup_invoice)"),
    ("edge_close.txt", "user: hola\nassistant: le ayudo con la factura\nuser: sí, gracias\nassistant: gracias por su llamada, que tenga un buen día"),
]

fails = 0
for name, text in cases:
    py = extract_transcript(text)
    ev_path = ROOT + "/runtime/_parity_transcript_case.txt"
    open(ev_path, "w", encoding="utf-8", newline="\n").write(text)
    r = subprocess.run(["node", runner_path,
                        ROOT + "/builds/conversation-autopsy/tau_port.js", ev_path],
                       capture_output=True, text=True, encoding="utf-8", timeout=60,
                       cwd=ROOT + "/builds/trace2grammar")
    if r.returncode != 0:
        print(name, "NODE ERROR:", r.stderr[:200]); fails += 1; continue
    js = json.loads(r.stdout)
    py_events = [{k: ev[k] for k in ("t_ms", "kind", "name", "attrs")} for ev in py["events"]]
    ok = (js["events"] == py_events and js["walk"] == py["walk"]
          and js["outcome"] == py["outcome"]
          and js["cycle_depth"] == py["cycle_depth"]
          and js["error_count"] == py["error_count"])
    print(("PASS " if ok else "FAIL "), name, "->", py["outcome"], "D=", py["cycle_depth"])
    if not ok:
        fails += 1
        if js["outcome"] != py["outcome"]:
            print("   outcome:", js["outcome"], "vs", py["outcome"])
        if js["events"] != py_events:
            for i, (a, b) in enumerate(zip(js["events"], py_events)):
                if a != b:
                    print("   events[%d]:" % i, json.dumps(a, ensure_ascii=False)[:160])
                    print("   py      :", json.dumps(b, ensure_ascii=False)[:160])
                    break
            print("   n_events:", len(js["events"]), "vs", len(py_events))
        if js["walk"] != py["walk"]:
            for i, (a, b) in enumerate(zip(js["walk"], py["walk"])):
                if a != b:
                    print("   walk[%d]:" % i, a, "vs", b); break
            print("   len:", len(js["walk"]), "vs", len(py["walk"]))

os.remove(ev_path)
os.remove(runner_path)
print("TRANSCRIPT PARITY:", "OK" if fails == 0 else "%d FAILURES" % fails)
sys.exit(1 if fails else 0)
