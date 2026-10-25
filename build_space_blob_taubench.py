# Re-anchor the where-conversations-die Space on REAL agent runs.
# Real = 1,980 official tau-bench episodes (gpt-4o + sonnet-3.5, airline+retail) via the
# shipped extract(fmt='taubench'), untouched. Sim = the 60 production-shaped sim2 templates
# (same set the sim2 wave shipped; labeled template-based in the copy). Same blob contract
# as build_space_blob.py (uncovered cells keep ex/d keys; renderer untouched except where
# noted). Real transcripts are NOT embedded (1,980 x ~15KB would bloat the JS): each is
# written as tb_transcripts/{id}.txt (rendered FROM the normalized events the walker
# actually saw — data-true) and fetched lazily by the Space on click.
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runtime", "taubench_anchor")  # hf/runtime/taubench_anchor (relative, clean-room safe)
sys.path.insert(0, HERE)
from trace2grammar.extract import extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from trace2grammar.coverage import coverage_from_walks, _grid_transform  # noqa: E402
from trace2grammar.report import conversation_report  # noqa: E402  (may be unused; keep import near usage)
from tb_to_events import to_extract_obj, strip_injected_terminal  # noqa: E402

LBL = {"theta_cont": "contained", "theta_aban": "abandoned", "theta_hand": "handoff"}
OUT_SPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "where-conversations-die")

# ---- real: extract verbatim, walks + conversational verdict (no injected terminal) ----
FILES = {
    ("gpt4o", "airline"): "gpt-4o-airline.json",
    ("gpt4o", "retail"): "gpt-4o-retail.json",
    ("sonnet", "airline"): "sonnet-35-new-airline.json",
    ("sonnet", "retail"): "sonnet-35-new-retail.json",
}

def events_to_transcript(events):
    out = []
    for e in events:
        k = e["kind"]
        c = str((e.get("attrs") or {}).get("content") or "")
        c = c.replace("\n", " ")[:300]
        if k == "user_msg":
            out.append("user: " + c)
        elif k == "assistant_msg":
            out.append("assistant: " + c)
        elif k == "tool_call":
            out.append(f"tool({e['name']}): call")
        elif k == "tool_result":
            out.append(f"tool({e['name']}): ok" + (f" {c[:60]}" if c and not c.startswith("<") else ""))
        elif k == "tool_error":
            out.append(f"tool({e['name']}): error {c[:60]}")
        elif k == "handoff":
            out.append(f"handoff({e.get('name') or 'human'})")
        elif k == "session_end":
            out.append(f"session_end({e.get('name') or ''})")
    return "\n".join(out)

real_results, real_pts, real_events = [], [], []
tb_dir = os.path.join(OUT_SPACE, "tb_transcripts")
os.makedirs(tb_dir, exist_ok=True)
n_err = 0
for (model, domain), fname in FILES.items():
    eps = json.load(open(os.path.join(HERE, fname), encoding="utf-8"))
    for ep in eps:
        eid = f"{model}.{domain}.{ep.get('task_id')}.{ep.get('trial')}"
        try:
            res = extract(to_extract_obj(ep), "taubench")
        except Exception:
            n_err += 1
            continue
        convo = infer_states(strip_injected_terminal(res["events"]), TAUBENCH_CONFIG)
        real_results.append(convo)
        real_events.append(res["events"])
        real_pts.append({
            "id": eid,
            "model": model,
            "domain": domain,
            "task_id": ep.get("task_id"),
            "reward": ep.get("reward"),
        })
assert len(real_results) == 1980, f"expected 1,980 episodes, got {len(real_results)} (errors={n_err})"

# ---- sim: the 60 sim2 production-shaped templates (embedded transcripts, small) ----
def load_sim2():
    out = []
    sdir = os.path.join(ROOT, "samples")
    for f in sorted(os.listdir(sdir)):
        if f.startswith("sim2") and f.endswith(".txt"):
            txt = open(os.path.join(sdir, f), encoding="utf-8", newline="").read().replace("\r\n", "\n")
            out.append({"name": f[:-4], "transcript": txt, "res": extract(txt, "transcript")})
    return out

sim = load_sim2()
assert len(sim) == 60, f"expected 60 sim2 templates, got {len(sim)}"

# ---- coverage: shipped coverage(), real-anchored plane; assert vs the paper snapshot ----
cov = coverage_from_walks(real_results, [s["res"] for s in sim], res=24, eps=0.05, mode="grammar")
snap = json.load(open(os.path.join(HERE, "tb_coverage.json"), encoding="utf-8"))
assert cov["occupied_cells"] == snap["occupied_cells"], (cov["occupied_cells"], snap["occupied_cells"])
assert cov["covered_cells"] == snap["covered_cells"], (cov["covered_cells"], snap["covered_cells"])
print("coverage:", cov["coverage_pct"], "% |", cov["covered_cells"], "/", cov["occupied_cells"],
      "| matches paper snapshot:", cov["occupied_cells"] == snap["occupied_cells"])

report = json.load(open(os.path.join(HERE, "tb_report.json"), encoding="utf-8"))

# ---- points ----
def build_real():
    pts = []
    for i, convo in enumerate(real_results):
        m = real_pts[i]
        meta = cov["real_meta"][i]
        pts.append({
            "id": m["id"],
            "x": round(cov["real2d"][i][0], 4),
            "y": round(cov["real2d"][i][1], 4),
            "kind": "real",
            "outcome": convo["outcome"],
            "op": m["domain"],
            "D": convo["cycle_depth"],
            "dur_s": None,
            "n_states": meta["n_states"],
            "first": meta["first_user_text"][:90],
            "walk": [w["state"] for w in convo["walk"]],
            "model": m["model"],
            "domain": m["domain"],
            "reward": m["reward"],
            "task_id": m["task_id"],
            "sf": convo["silent_failure_candidate"],
            "tfile": f"tb_transcripts/{m['id']}.txt",
        })
    return pts

def build_sim():
    pts = []
    for i, e in enumerate(sim):
        m = cov["sim_meta"][i]
        pts.append({
            "id": e["name"],
            "x": round(cov["sim2d"][i][0], 4),
            "y": round(cov["sim2d"][i][1], 4),
            "kind": "sim",
            "outcome": m["outcome"],
            "op": "sim",
            "D": m["cycle_depth"],
            "dur_s": round(m["duration_ms"] / 1000, 1),
            "n_states": m["n_states"],
            "first": m["first_user_text"][:90],
            "walk": [w["state"] for w in e["res"]["walk"]],
            "transcript": e["transcript"],
        })
    return pts

# ---- write transcripts (lazy files) ----
for i, convo in enumerate(real_results):
    eid = real_pts[i]["id"]
    with open(os.path.join(tb_dir, eid + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(events_to_transcript(real_events[i]))

x0, y0, cw, ch = _grid_transform(cov["real2d"], 24, 0.15)
blob = {
    "generated": "2026-10-25",
    "anchor": "tau-bench historical_trajectories (sierra-research/tau-bench, official eval runs)",
    "grid": {"res": 24, "x0": x0, "y0": y0, "cw": cw, "ch": ch,
             "eps": cov["grid"]["eps"], "bw": cov["grid"]["bw_frac"]},
    "coverage": {"pct": cov["coverage_pct"], "occupied": cov["occupied_cells"],
                 "covered": cov["covered_cells"], "per_class": cov["per_class_coverage"]},
    "probe": {
        "kernel": report.get("bw_ladder"),
        "instance": report.get("instance_coverage"),
        "confrontation": report.get("overall"),
        "by_model": report.get("by_model"),
        "D_gt0": report.get("n_D_gt0"),
        "D_max": report.get("D_max"),
    },
    "uncovered": [{"gx": u["gx"], "gy": u["gy"],
                   "ex": u["exemplar_idx"], "d": round(u["local_real_density"], 3)}
                  for u in sorted(cov["uncovered_cells"], key=lambda u: -u["local_real_density"])],
    "points": build_real() + build_sim(),
    "labels": LBL,
}
out = os.path.join(OUT_SPACE, "atlas_data.js")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("var ATLAS = ")
    json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"))
    fh.write(";")
n_tx = len(os.listdir(tb_dir))
print("blob written:", os.path.getsize(out), "bytes |", len(blob["uncovered"]), "uncovered |",
      len(blob["points"]), "points |", n_tx, "transcript files")
