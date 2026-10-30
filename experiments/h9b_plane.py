"""h9b_plane.py — reconstruct the h9 plane from saved artifacts (no LLM calls).

Bit-consistency check of the paper figure data: rebuild the naive set (template identical
to h9_advgen_loop.naive_conversation), re-extract the final adversarial set (the 18 r1
transcripts — round 0 missed all 18 targets, so every slot was regenerated), re-extract the
1,980 real walks, recompute the beta=0.04 regime, and assert the numbers match
h9_results.json. Writes h9_plane.json for the figure script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor")

from trace2grammar.extract import extract, extract_transcript, infer_states, TAUBENCH_CONFIG  # noqa: E402
from trace2grammar.coverage import feature_matrix, _pca_basis, coverage  # noqa: E402
from tb_to_events import to_extract_obj, strip_injected_terminal  # noqa: E402

ANCHOR = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor"
RUN_DIR = "C:/Users/U77225D/Desktop/proyectos/hf/runtime/adv_gen_2026-10-30"
BW = 0.04

# --- naive template: MUST stay identical to h9_advgen_loop.naive_conversation ---
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

def naive_conversation(i):
    opener, tool = PRODUCTS[i % len(PRODUCTS)]
    n_clar = i % 3
    n_tools = 2 + (i % 3)
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

# --- real walks (same flow as tb_tau.py / h9) ---
FILES = {
    ("gpt4o", "airline"): "gpt-4o-airline.json",
    ("gpt4o", "retail"): "gpt-4o-retail.json",
    ("sonnet", "airline"): "sonnet-35-new-airline.json",
    ("sonnet", "retail"): "sonnet-35-new-retail.json",
}
results_real = []
for (model, domain), fn in FILES.items():
    eps = json.load(open(os.path.join(ANCHOR, fn), encoding="utf-8"))
    for ep in eps:
        res = extract(to_extract_obj(ep), "taubench")
        results_real.append(infer_states(strip_injected_terminal(res["events"]), TAUBENCH_CONFIG))
print("real extracted:", len(results_real))

real_m = feature_matrix(results_real)
MEAN, V1, V2 = _pca_basis(real_m)
d = len(real_m[0])

def _project(m):
    return [(sum((r[j] - MEAN[j]) * V1[j] for j in range(d)),
             sum((r[j] - MEAN[j]) * V2[j] for j in range(d))) for r in m]

real2d = _project(real_m)

# parity with the shipped snapshot (deterministic pipeline)
tb = json.load(open(os.path.join(ANCHOR, "tb_coverage.json"), encoding="utf-8"))
max_d = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(real2d[:50], tb["real2d"][:50]))
assert max_d < 1e-6, f"parity broken {max_d}"

# --- naive + adversarial (final = the 18 r1 transcripts) ---
naive_results = [extract_transcript(naive_conversation(i)) for i in range(20)]
adv_files = sorted(f for f in os.listdir(RUN_DIR) if f.startswith("adv_r1_"))
adv_results = [extract_transcript(open(os.path.join(RUN_DIR, f), encoding="utf-8").read())
               for f in adv_files]
print(f"naive {len(naive_results)} | adversarial (r1): {len(adv_results)}")

naive2d = _project(feature_matrix(naive_results))
comb2d = _project(feature_matrix(naive_results + adv_results))
adv2d = comb2d[len(naive_results):]

classes = [r["outcome"] for r in results_real]
cov_naive = coverage(real2d, naive2d, classes=classes, res=24, eps=0.05, bw_frac=BW)
cov_comb = coverage(real2d, comb2d, classes=classes, res=24, eps=0.05, bw_frac=BW)

h9 = json.load(open(os.path.join(HERE, "h9_results.json"), encoding="utf-8"))
assert cov_naive["coverage_pct"] == h9["naive"]["coverage_pct"], "naive mismatch"
assert cov_comb["coverage_pct"] == h9["combined"]["coverage_pct"], "combined mismatch"
print(f"bit-consistency with h9_results.json OK: naive {cov_naive['coverage_pct']}% -> "
      f"combined {cov_comb['coverage_pct']}% | uncovered {len(cov_naive['uncovered_cells'])}"
      f" -> {len(cov_comb['uncovered_cells'])}")

json.dump({"real2d": real2d, "classes": classes,
           "naive2d": naive2d, "adv2d": adv2d,
           "grid": {"res": 24, "eps": 0.05, "bw_frac": BW},
           "naive_cov": cov_naive["coverage_pct"], "comb_cov": cov_comb["coverage_pct"],
           "naive_uncovered": len(cov_naive["uncovered_cells"]),
           "comb_uncovered": len(cov_comb["uncovered_cells"])},
          open(os.path.join(HERE, "h9_plane.json"), "w", encoding="utf-8"))
print("written h9_plane.json")
