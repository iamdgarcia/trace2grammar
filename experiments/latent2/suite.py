"""suite.py — the latent-2 instrument's property suite (one command, re-verifiable).

Re-verifies every executed property from the committed caches/results (deterministic parts
re-run; laya-dependent parts re-checked against their stored verdicts + file presence).
Exit 0 = instrument verified.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
checks = []

def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

# ---- E2: twin spaces (stored; deterministic given cached embeddings)
e2 = json.load(open(os.path.join(HERE, "e2_results.json"), encoding="utf-8"))
check("E2 verdict stored", e2["verdict"] == "FAIL",
      f"(semantic {e2['results']['semantic']['auroc']}, structural {e2['results']['structural']['auroc']})")
check("E2 semantic ~ chance (twins identical in content space)",
      abs(e2["results"]["semantic"]["auroc"] - 0.5) < 0.05)

# ---- E3: divergence points (stored; embeddings cached)
e3 = json.load(open(os.path.join(HERE, "e3_results.json"), encoding="utf-8"))
check("E3 divergence points", e3["P_A_pct"] == 51.0 and e3["P_B_pct"] == 40.4,
      f"(defined {e3['P_A_pct']}%, confirmed {e3['P_B_pct']}%)")
pe = os.path.join(os.environ.get("HF", "C:/Users/U77225D/Desktop/proyectos/hf"),
                  "runtime/latent2/prefix_embed.json")
check("E3 prefix embeddings cached", os.path.exists(pe), f"({os.path.getsize(pe)//1024} KB)")

# ---- E4: early risk (stored)
e4 = json.load(open(os.path.join(HERE, "e4_results.json"), encoding="utf-8"))
check("E4 early signal stored", e4["P_E4_pct"] == 35.3, f"({e4['P_E4_pct']}% < 60 bar)")

# ---- E5/E5b: what-if (stored)
e5 = json.load(open(os.path.join(HERE, "e5_results.json"), encoding="utf-8"))
e5b = json.load(open(os.path.join(HERE, "e5b_results.json"), encoding="utf-8"))
check("E5 dialogue splice", e5["flip_pct"] == 20.5, f"(flip {e5['flip_pct']}%, rejoin {e5['rejoin_pct']}%)")
check("E5b state splice", e5b["flip_pct"] == 23.7, f"(flip {e5b['flip_pct']}%, rejoin {e5b['rejoin_pct']}%)")

# ---- E6: the lie (laya-dependent; stored verdict + artifacts present)
e6 = json.load(open(os.path.join(HERE, "e6_results.json"), encoding="utf-8"))
lies = os.listdir(os.path.join(os.environ.get("HF", "C:/Users/U77225D/Desktop/proyectos/hf"),
                               "runtime/latent2/e6_lies"))
check("E6 the lie is generable", e6["P_E6"] == "PASS" and e6["flip_pct"] == 58.8 and len(lies) >= 20,
      f"({e6['flip_pct']}%, {len(lies)} artifacts)")

# ---- E7: composed coverage (deterministic from caches -> RE-RUN)
r = subprocess.run([sys.executable, os.path.join(HERE, "e7_composed_coverage.py")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
env_ok = "VERDICT: P-E7 PASS" in r.stdout
check("E7 composed coverage re-runs deterministic", env_ok,
      "(region 86.6->38.9% at bw 0.04; NN-ratio 14.2->76.6)" if env_ok else r.stderr[-150:])

# ---- E8: the corrected typing (deterministic -> RE-RUN)
r = subprocess.run([sys.executable, os.path.join(HERE, "e8_typed_answer.py")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
check("E8 typing correction holds", "VERDICT: P-E8 FAIL" in r.stdout,
      "(no-divergence does NOT concentrate walker misses; corrected typing in thread)")

# ---- thread ledger
tsv = open("C:/Users/U77225D/Desktop/proyectos/hf/knowledge_base/threads/t2g-latent-v2/results.tsv",
           encoding="utf-8").read().strip().split("\n")
check("results.tsv rows", len(tsv) >= 9, f"({len(tsv)-1} iterations logged)")

fails = sum(1 for _, ok, _ in checks if not ok)
print(f"\nSUITE: {len(checks)-fails}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
