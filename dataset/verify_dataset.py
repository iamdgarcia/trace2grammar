"""verify_dataset.py — independent checker for the TraceGrammar-Anchor release.

Re-derives counts from the ORIGINAL artifacts and re-derives 3 fields for 5 random
episodes per corpus from the RAW sources. Exit 0 = release verified.
Run: python verify_dataset.py  (paths resolved relative to this file)
"""
import csv, hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # hf/
REPO = os.path.abspath(os.path.join(HERE, ".."))
RUN = ROOT + "/runtime"
sys.path.insert(0, REPO)
from trace2grammar.extract import extract_transcript, infer_states  # noqa: E402

fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond: fails += 1

def readcsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

tb = readcsv(HERE + "/taubench/episodes.csv")
sg = readcsv(HERE + "/sgd/dialogs.csv")
mw = readcsv(HERE + "/multiwoz/dialogs.csv")

# counts vs published reports
rep = json.load(open(RUN + "/taubench_anchor/tb_report.json", encoding="utf-8"))
check("taubench count == tb_report.n_episodes", len(tb) == rep["n_episodes"], f"{len(tb)} vs {rep['n_episodes']}")
sres = json.load(open(RUN + "/sgd_validation/results.json", encoding="utf-8"))
check("sgd count == results.n_total", len(sg) == sres["n_total"], f"{len(sg)} vs {sres['n_total']}")
check("sgd verbatim agreement matches published",
      abs(sum(1 for r in sg if r["outcome_verbatim"] == ("theta_cont" if r["gt_effective_call"] == "True" else "theta_aban")) / len(sg) * 100 - sres["agreement_exact_pct"]) < 0.5,
      str(sres["agreement_exact_pct"]))
mres = json.load(open(RUN + "/multiwoz_tau_2026-10-23.json", encoding="utf-8"))["rows"]
check("multiwoz count == runtime rows", len(mw) == len(mres) == 1000, f"{len(mw)}")

# spot-check tau-bench: reward/task/trial from RAW + first_user_text from real_meta
raws = {}
for fn in ("gpt-4o-airline", "gpt-4o-retail", "sonnet-35-new-airline", "sonnet-35-new-retail"):
    raws[fn] = json.load(open(f"{RUN}/taubench_anchor/{fn}.json", encoding="utf-8"))
cov = json.load(open(RUN + "/taubench_anchor/tb_coverage.json", encoding="utf-8"))["real_meta"]
random.seed(20261025)
tb_idx = random.sample(range(len(tb)), 5)
for i in tb_idx:
    r = tb[i]
    model, dom = r["model"], r["domain"]
    fn = ("gpt-4o" if model == "gpt4o" else "sonnet-35-new") + "-" + dom
    ep = next(e for e in raws[fn] if e["task_id"] == int(r["task_id"]) and e["trial"] == int(r["trial"]))
    check(f"tb[{i}] reward matches raw", abs(float(r["reward"]) - float(ep["reward"])) < 1e-9)
    check(f"tb[{i}] task/trial match", ep["task_id"] == int(r["task_id"]) and ep["trial"] == int(r["trial"]))
    check(f"tb[i] first_user_text matches real_meta", r["first_user_text"][:40] == cov[i]["first_user_text"].replace("\n", " ").strip()[:40])

# spot-check SGD: services + walker verdict re-run from transcript
sgd_ids = random.sample(range(len(sg)), 5)
for i in sgd_ids:
    r = sg[i]
    m = json.load(open(REPO + "/sgd_test_transcript/_meta.json", encoding="utf-8"))[r["dialog_id"]]
    check(f"sgd[{i}] services match meta", "|".join(m["services"]) == r["services"])
    txt = open(REPO + f"/sgd_test_transcript/{r['dialog_id']}.txt", encoding="utf-8").read()
    res = infer_states(extract_transcript(txt)["events"])
    check(f"sgd[{i}] walker verdict reproduces", res["outcome"] == r["outcome_verbatim"],
          f'{res["outcome"]} vs {r["outcome_verbatim"]}')
    check(f"sgd[{i}] gt == (n_effective>0)", (r["gt_effective_call"] == "True") == (int(r["n_effective"]) > 0))

# spot-check MultiWOZ: services from meta + walker verdict re-run
mmeta = json.load(open(REPO + "/multiwoz_test/_meta.json", encoding="utf-8"))
mw_idx = random.sample(range(len(mw)), 5)
for i in mw_idx:
    r = mw[i]
    check(f"mw[{i}] services match meta", "|".join(mmeta[r["dialog_id"]]["services"]) == r["services"])
    txt = open(REPO + f"/multiwoz_test/{r['dialog_id']}.txt", encoding="utf-8").read()
    res = infer_states(extract_transcript(txt)["events"])
    check(f"mw[{i}] walker verdict reproduces", res["outcome"] == r["outcome"],
          f'{res["outcome"]} vs {r["outcome"]}')

# manifest
man = HERE + "/MANIFEST.sha256"
if os.path.exists(man):
    ok = True
    for line in open(man, encoding="utf-8"):
        h, rel = line.strip().split("  ", 1)
        p = os.path.join(HERE, rel)
        if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            ok = False; check("manifest: " + rel, False, "missing or hash mismatch")
    check("manifest: all hashes match", ok)
else:
    check("manifest present", False, "MANIFEST.sha256 missing (run the builder)")

print("RELEASE VERIFIED" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
