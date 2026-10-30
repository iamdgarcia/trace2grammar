"""h9_advgen_loop.py — the recommendation loop, executed: naive sim set -> uncovered
regions -> generated adversarial tests -> re-measured coverage.

Predeclared (threads/t2g-latent-paper/PLAN.md, G2):
  P1 landing:    >=60% of generated tests land in their target cell or 8-neighborhood
  P2 delta:      region coverage of naive∪adv >= naive + 15pp (same config)
  P3 substance:  >=30% of generated tests have cycle depth D >= 1
  Baseline:      the naive set lands on 0 targets by construction (targets ARE its holes)

Pipeline reuses the shipped machinery only (no re-implementation):
  extract_transcript / extract(fmt='taubench') / coverage_from_walks / top_k_uncovered /
  _grid_transform. The generated conversations are produced by an OpenAI-compatible
  endpoint (factory gateway via env; no endpoint details in this repo) — if the env is
  absent the script exits before touching data (fail-fast guard).

Stdlib only. Outputs:
  h9_results.json (metrics) + raw transcripts in hf/runtime/adv_gen_2026-10-30/.
"""
import json
import math
import os
import random
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor")

from trace2grammar.extract import extract_transcript, extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from trace2grammar.coverage import (  # noqa: E402
    coverage_from_walks, top_k_uncovered, _grid_transform, instance_coverage,
    feature_matrix, _pca_basis, coverage,
)
from tb_to_events import to_extract_obj, strip_injected_terminal  # noqa: E402

# ---- env guard: the generation endpoint comes from the environment, never from here
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("H9_MODEL", "gemini-2.5-flash-lite")
if not (BASE_URL and API_KEY):
    sys.exit("SIN ENDPOINT: exporta OPENAI_BASE_URL y OPENAI_API_KEY (gateway de la fabrica)")

RUN_DIR = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/adv_gen_2026-10-30"
os.makedirs(RUN_DIR, exist_ok=True)
SEED = 20261030

# ------------------------------------------------------------------ real anchor
FILES = {
    ("gpt4o", "airline"): "gpt-4o-airline.json",
    ("gpt4o", "retail"): "gpt-4o-retail.json",
    ("sonnet", "airline"): "sonnet-35-new-airline.json",
    ("sonnet", "retail"): "sonnet-35-new-retail.json",
}
ANCHOR = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor"

results_real = []
for (model, domain), fn in FILES.items():
    eps = json.load(open(os.path.join(ANCHOR, fn), encoding="utf-8"))
    for ep in eps:
        res = extract(to_extract_obj(ep), "taubench")
        events = strip_injected_terminal(res["events"])
        res = infer_states(events, TAUBENCH_CONFIG)
        res["first_user_text"] = next(
            (str(e.get("attrs", {}).get("content", ""))[:90] for e in events
             if e["kind"] == "user_msg"), "")
        results_real.append(res)
print(f"real extracted: {len(results_real)}")

# ------------------------------------------------------------------ naive sim set
PRODUCTS = [
    ("checking my account balance", "get_account_balance"),
    ("booking a flight to Seattle on May 20th", "search_flights"),
    ("rescheduling my dentist appointment", "get_appointments"),
    ("a refund for a damaged package", "get_order_details"),
    ("resetting my password", "verify_identity"),
    ("reserving a hotel room for two nights", "search_hotels"),
    ("transferring money to a friend", "get_accounts"),
    ("checking the status of my order", "get_order_status"),
    ("updating my shipping address", "get_profile"),
    ("cancelling a subscription", "get_subscription"),
]
rng = random.Random(SEED)

def naive_conversation(i):
    opener, tool = PRODUCTS[i % len(PRODUCTS)]
    n_clar = i % 3                       # 0-2 clarifying rounds
    n_tools = 2 + (i % 3)                # 2-4 successful tool calls
    lines = ["assistant: Hello! How can I help you today?",
             f"user: Hi, I'm interested in {opener}."]
    for c in range(n_clar):
        lines.append(f"assistant: Sure. Could you confirm detail number {c+1} for me?")
        lines.append(f"user: Yes, here it is: {1000 + c + i}.")
    for t in range(n_tools):
        lines.append(f"tool({tool}_{t+1}): ok {{\"status\": \"success\"}}")
    lines.append("assistant: All done! Is there anything else I can help you with?")
    lines.append("user: No, that's everything. Thanks!")
    lines.append("assistant: You're welcome. Have a great day!")
    return "\n".join(lines)

naive_results = [extract_transcript(naive_conversation(i)) for i in range(20)]
print(f"naive set: {len(naive_results)} conversations, "
      f"D dist {sorted(set(r['cycle_depth'] for r in naive_results))}, "
      f"outcomes {sorted(set(r['outcome'] for r in naive_results))}")

# ------------------------------------------------------------------ coverage of naive
# coverage_from_walks pins bw_frac=0.12 (kernel-masking regime: any small set covers 100%).
# The paper's own discriminating regime is beta <= 0.04: same blocks, honest bandwidth.
BW = 0.04
real_m = feature_matrix(results_real)
naive_m = feature_matrix(naive_results)
MEAN, V1, V2 = _pca_basis(real_m)

def _project(m):
    d = len(real_m[0])
    return [(sum((r[j] - MEAN[j]) * V1[j] for j in range(d)),
             sum((r[j] - MEAN[j]) * V2[j] for j in range(d))) for r in m]

real2d = _project(real_m)
naive2d = _project(naive_m)
cov_naive = coverage(real2d, naive2d,
                     classes=[r["outcome"] for r in results_real],
                     res=24, eps=0.05, bw_frac=BW)
print(f"naive coverage (bw={BW}): {cov_naive['coverage_pct']}% | "
      f"occupied {cov_naive['occupied_cells']} | uncovered {len(cov_naive['uncovered_cells'])}")

# parity: my real2d must match the shipped tb_coverage.json (same pipeline, deterministic)
tb = json.load(open(os.path.join(ANCHOR, "tb_coverage.json"), encoding="utf-8"))
max_d = max(abs(a[0] - b[0]) + abs(a[1] - b[1])
            for a, b in zip(real2d[:50], tb["real2d"][:50]))
assert max_d < 1e-6, f"real2d parity broken: {max_d}"
print(f"real2d parity vs shipped tb_coverage.json OK (max delta {max_d:.2e})")

if not cov_naive["uncovered_cells"]:
    print("no holes even at beta=0.04 — aborting (kill_criterion territory)")
    sys.exit(1)

targets = sorted(cov_naive["uncovered_cells"], key=lambda u: -u["local_real_density"])
print(f"targets (all uncovered cells, density order): {len(targets)}")

# nearest real exemplar meta per target cell
x0, y0, cw, ch = _grid_transform(real2d, cov_naive["grid"]["res"], 0.15)
def cell_center(gx, gy):
    return (x0 + (gx + 0.5) * cw, y0 + (gy + 0.5) * ch)

tgt_meta = []
for t in targets:
    cx, cy = cell_center(t["gx"], t["gy"])
    best = min(range(len(results_real)),
               key=lambda i: (real2d[i][0] - cx) ** 2 + (real2d[i][1] - cy) ** 2)
    m = results_real[best]
    tgt_meta.append({"gx": t["gx"], "gy": t["gy"], "density": t["local_real_density"],
                     "exemplar": {"first_user_text": m["first_user_text"],
                                  "outcome": m["outcome"], "D": m["cycle_depth"],
                                  "n_states": len(m["walk"])}})

# ------------------------------------------------------------------ generation
def gen_conversation(m, feedback=None):
    ex = m["exemplar"]
    opening = ex["first_user_text"]
    outcome = ex["outcome"]
    if outcome == "theta_aban":
        ending = ("The assistant keeps offering help but the user goes silent: write the last two "
                  "lines as the assistant asking twice whether the user is still there, with no user reply.")
    elif outcome == "theta_hand":
        ending = ("The request turns out to be beyond what the assistant may do: the assistant must "
                  "transfer the user to a human specialist and the transcript ends with that transfer.")
    else:
        ending = ("The task resolves fully and both sides say goodbye politely.")
    fb = ""
    if feedback:
        fb = (f"\n- IMPORTANT: a previous draft of this conversation measured at grid cell "
              f"({feedback['measured'][0]},{feedback['measured'][1]}) but the target cell is "
              f"({m['gx']},{m['gy']}). The plane axes track conversation SIZE (total lines/exchanges) "
              f"and loop depth. {feedback['hint']} Keep the same intent and ending rules.")
    prompt = f"""Write ONE customer-support conversation transcript, in exactly this line format:

user: <text>
assistant: <text>
tool(<tool_name>): ok {{"json": "result"}}
tool(<tool_name>): error <message>

Rules:
- Line 1 is a user opening equivalent (same intent, different wording) to: "{opening}"
- The user never becomes hostile and never uses offensive language.
- {ending}
- Make the assistant's tools fail with an error TWICE during the middle of the conversation
  (the user re-explains or retries after each error), then succeed on the third attempt.
- Length: about {ex['n_states']} total lines (this controls where the conversation lands in a
  behavioral map; take it seriously and count your lines before answering).
- Output ONLY the transcript lines, nothing else.{fb}
"""
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 900, "temperature": 0.8}).encode(),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.load(resp)
    txt = d["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith(("transcript", "\n")):
            txt = txt.split("\n", 1)[1] if "\n" in txt else txt
    return txt

adv_raw, round0 = [], []
for i, m in enumerate(tgt_meta):
    try:
        txt = gen_conversation(m)
        res = extract_transcript(txt)
        round0.append({"i": i, "lines": len(txt.splitlines()), "D": res["cycle_depth"],
                       "outcome": res["outcome"]})
        adv_raw.append(txt)
        open(os.path.join(RUN_DIR, f"adv_r0_{i:02d}.txt"), "w", encoding="utf-8", newline="").write(txt)
        print(f"  gen r0 {i:02d}: {len(txt.splitlines())} lines, D={res['cycle_depth']}, {res['outcome']}")
    except Exception as e:
        print(f"  gen r0 {i:02d}: FAILED {e}")
print(f"generated round0: {len(adv_raw)}/{len(tgt_meta)}")

# ---- measure round0, regenerate misses once with measured-position feedback ---------
r0_results = [extract_transcript(t) for t in adv_raw]
r0_m = feature_matrix(naive_results + r0_results)
r0_2d = _project(r0_m)[len(naive_results):]

def cheb(pt, m):
    gx = math.floor((pt[0] - x0) / cw)
    gy = math.floor((pt[1] - y0) / ch)
    return max(abs(gx - m["gx"]), abs(gy - m["gy"])), [gx, gy]

r0_hits = 0
adv_final, round1 = [], []
for i, (pt, m) in enumerate(zip(r0_2d, tgt_meta[:len(r0_2d)])):
    d, cell = cheb(pt, m)
    r0_hits += d <= 1
    if d <= 1:
        adv_final.append(adv_raw[i])
        continue
    # feedback round: tell the generator where the draft landed
    hint = ("Write MANY more lines" if cell[1] < m["gy"] or len(r0_results[i]["walk"]) < m["exemplar"]["n_states"]
            else "Write fewer lines")
    try:
        txt = gen_conversation(m, feedback={"measured": cell, "hint": hint})
        res = extract_transcript(txt)
        round1.append({"i": i, "lines": len(txt.splitlines()), "D": res["cycle_depth"],
                       "outcome": res["outcome"]})
        adv_final.append(txt)
        open(os.path.join(RUN_DIR, f"adv_r1_{i:02d}.txt"), "w", encoding="utf-8", newline="").write(txt)
        print(f"  regen r1 {i:02d}: landed draft at {cell}, target ({m['gx']},{m['gy']}), {hint.lower()}")
    except Exception as e:
        adv_final.append(adv_raw[i])
        print(f"  regen r1 {i:02d}: FAILED ({e}), keeping round0 draft")
print(f"round0 landing: {r0_hits}/{len(r0_2d)} | final set: {len(adv_final)}")
adv_raw = adv_final

# ------------------------------------------------------------------ validation
adv_results = [extract_transcript(t) for t in adv_raw]
combined = naive_results + adv_results
comb_m = feature_matrix(combined)
combined2d = _project(comb_m)
cov_comb = coverage(real2d, combined2d,
                    classes=[r["outcome"] for r in results_real],
                    res=24, eps=0.05, bw_frac=BW)
adv2d = combined2d[len(naive_results):]

hits = 0
landing = []
for i, (pt, m) in enumerate(zip(adv2d, tgt_meta[:len(adv2d)])):
    gx = math.floor((pt[0] - x0) / cw)
    gy = math.floor((pt[1] - y0) / ch)
    d = max(abs(gx - m["gx"]), abs(gy - m["gy"]))
    landing.append({"i": i, "cell": [gx, gy], "target": [m["gx"], m["gy"]],
                    "chebyshev": d, "hit": d <= 1})
    hits += d <= 1
p1 = 100.0 * hits / len(adv2d) if adv2d else 0.0

d_dist = sorted(r["cycle_depth"] for r in adv_results)
p3 = 100.0 * sum(1 for r in adv_results if r["cycle_depth"] >= 1) / len(adv_results)

# honest instance coverage (dedup radius) for naive vs combined
real_u, seen = [], set()
for p in real2d:
    k = (round(p[0], 9), round(p[1], 9))
    if k not in seen:
        seen.add(k); real_u.append(p)
def med_nn(pts):
    nn = []
    for i, (rx, ry) in enumerate(pts):
        nn.append(math.sqrt(min((rx - x) ** 2 + (ry - y) ** 2
                                for j, (x, y) in enumerate(pts) if j != i)))
    nn.sort(); n = len(nn)
    return nn[n // 2] if n % 2 else (nn[n // 2 - 1] + nn[n // 2]) / 2
r_u = med_nn(real_u)
def inst(pts):
    r2 = r_u * r_u
    return 100.0 * sum(any((rx - x) ** 2 + (ry - y) ** 2 <= r2 for x, y in pts)
                       for rx, ry in real_u) / len(real_u)

out = {
    "protocol": {"res": 24, "eps": 0.05, "bw_frac": BW, "seed": SEED, "model": MODEL,
                 "n_real": len(results_real), "n_naive": len(naive_results),
                 "n_adv": len(adv_results),
                 "predicates": {"P1_landing_pct_min": 60, "P2_delta_pp_min": 15,
                                "P3_D_ge1_pct_min": 30}},
    "naive": {"coverage_pct": cov_naive["coverage_pct"],
              "uncovered": len(cov_naive["uncovered_cells"]),
              "instance_honest_pct": round(inst(cov_comb["sim2d"][:len(naive_results)]), 2)},
    "combined": {"coverage_pct": cov_comb["coverage_pct"],
                 "uncovered": len(cov_comb["uncovered_cells"]),
                 "instance_honest_pct": round(inst(cov_comb["sim2d"]), 2)},
    "P1_landing_pct": round(p1, 1),
    "P2_delta_pp": round(cov_comb["coverage_pct"] - cov_naive["coverage_pct"], 1),
    "P3_D_ge1_pct": round(p3, 1),
    "round0": {"n": len(r0_2d), "landing_pct": round(100.0 * r0_hits / len(r0_2d) if r0_2d else 0.0, 1),
               "stats": round0},
    "round1": {"n_regenerated": len(round1), "stats": round1},
    "adv_D_distribution": d_dist,
    "adv_outcomes": sorted(set(r["outcome"] for r in adv_results)),
    "landing": landing,
    "verdicts": {
        "P1": "PASS" if p1 >= 60 else "FAIL",
        "P2": "PASS" if cov_comb["coverage_pct"] - cov_naive["coverage_pct"] >= 15 else "FAIL",
        "P3": "PASS" if p3 >= 30 else "FAIL",
    },
}
json.dump(out, open(os.path.join(HERE, "h9_results.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: out[k] for k in ("P1_landing_pct", "P2_delta_pp", "P3_D_ge1_pct",
                                      "verdicts", "naive", "combined")}, indent=1))
