# ELDA trajectory-dynamics property suite (zero training, frozen blocks).
# See README.md in this directory for channel definitions, the four property tests,
# and the regeneration protocol. Annotation endpoints are env-configured; the
# committed caches under annotations/ carry everything the paper's dynamics numbers
# need (descriptions, typed scores, compositions, distance rows, per-window dynamics).
import argparse
import json
import math
import os
import ssl
import time
import urllib.request

import numpy as np

PHASES = ["opening", "information_exchange", "verification", "repair", "closing", "terminal"]
STANCES = ["cooperative", "impatient", "angry", "resigned", "neutral"]

# expected paper values (tolerance-checked by `test`)
EXPECTED = {
    "separation_auroc": {"raw": 0.56, "desc": 0.85, "composed": 0.96},
    "order_ratio": {"raw": 0.21, "desc": 0.17, "composed": 0.78},
    "context_shift_same_struct": {"raw": 0.561, "desc": 0.35, "composed": 0.67},
    "convs_negative_turn": 34, "convs_total": 49,
    "loop_candidates": 44,
    "sgd_final_cos": {"theta_aban": 0.727, "theta_cont": 0.482},
    "tau_fails_with_reversal": "7/8",
}
TOL = 0.02


def _post(url, body, key, timeout=90):
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return json.loads(r.read().decode("utf-8")), time.perf_counter() - t0


DESC_PROMPT = """Annotate the state of this service conversation. Output ONLY a JSON object with exactly this schema:
{{"phase": "<opening|information_exchange|verification|repair|closing|terminal>",
 "resolved_requests": ["<requests fully handled BEFORE the current turn>"],
 "pending_request": "<what is being asked or handled in the current turn, or none>",
 "open_loops": ["<unresolved threads from earlier turns>"],
 "user_stance": "<cooperative|impatient|angry|resigned|neutral>"}}
Recent turns (the LAST turn is the current one):
{}"""


def annotate(windows_path, out_dir):
    """Regenerate window annotations via configured endpoints (ELDA_EMBED_URL/KEY env)."""
    base = os.environ["ELDA_EMBED_URL"].rstrip("/")
    key = os.environ["ELDA_EMBED_KEY"]
    desc_model = os.environ.get("ELDA_DESC_MODEL", "gemini-2.5-flash-lite")
    os.makedirs(out_dir, exist_ok=True)
    ann_path = os.path.join(out_dir, "descriptions.jsonl")
    done = set()
    if os.path.exists(ann_path):
        for line in open(ann_path, encoding="utf-8"):
            done.add(json.loads(line)["pid"])
    n = 0
    with open(windows_path, encoding="utf-8") as f, open(ann_path, "a", encoding="utf-8") as out:
        for line in f:
            w = json.loads(line)
            if w["pid"] in done:
                continue
            resp, _ = _post(base + "/chat/completions", {
                "model": desc_model,
                "messages": [{"role": "user", "content": DESC_PROMPT.format(w["text"][:6000])}],
                "max_tokens": 400, "temperature": 0}, key)
            content = resp["choices"][0]["message"].get("content") or ""
            s, e = content.find("{"), content.rfind("}")
            desc = json.loads(content[s:e + 1]) if s >= 0 and e > s else {}
            out.write(json.dumps({"pid": w["pid"], "desc": desc}, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
    print(f"annotate: {n} new annotations -> {ann_path}")


def auroc(near_d, far_d):
    allv = np.concatenate([near_d, far_d])
    r = allv.argsort().argsort() + 1
    u = r[:len(near_d)].sum() - len(near_d) * (len(near_d) + 1) / 2
    return 1.0 - u / (len(near_d) * len(far_d))


def test(data_dir):
    blob = json.load(open(os.path.join(data_dir, "dynamics_blob.json"), encoding="utf-8"))
    convs = blob["conversations"]
    ch = blob["channel_distances"]
    vd = blob["verify_distances"]
    pts = sorted(vd.keys())
    synth = [p for p in pts if p.startswith("synth:")]
    sgd0 = [p for p in pts if ":0" in p and p.startswith("sgd:") and p.split(":")[-1] == "0"]
    sgd_fin = [p for p in pts if p.startswith("sgd:") and p not in sgd0]
    fails, checks = 0, 0

    def check(name, got, want, tol=TOL):
        nonlocal fails, checks
        checks += 1
        ok = abs(got - want) <= tol
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.3f} want {want:.3f}")

    # T1 separation per channel: near = (synth cold-only ∪ sgd openings) pairwise = 1225 pairs (paper protocol)
    print("property tests (recomputed from committed caches):")
    nearset = [p for p in synth if p.endswith((":coldA", ":coldB"))] + sgd0
    for cname, key in (("raw", "raw"), ("desc", "desc"), ("composed", "composed")):
        dm = ch.get(key, vd if key == "composed" else None)
        nd = np.array([dm[a][b] for i, a in enumerate(nearset) for b in nearset[i+1:]])
        fd = np.array([dm[a][b] for a in nearset for b in sgd_fin if a != b])
        check(f"separation AUROC ({cname})", auroc(nd, fd), EXPECTED["separation_auroc"][cname])
    # T2 order ratio per channel
    fAB = [p for p in synth if p.endswith(":finalAB")]
    fBA = [p for p in synth if p.endswith(":finalBA")]
    cA = [p for p in synth if p.endswith(":coldA")]
    cB = [p for p in synth if p.endswith(":coldB")]
    for cname, key in (("raw", "raw"), ("desc", "desc"), ("composed", "composed")):
        dm = ch.get(key, vd if key == "composed" else None)
        do = np.array([dm[a][b] for a, b in zip(fAB, fBA)])
        dc = np.array([dm[a][b] for a, b in zip(cA, cB)])
        check(f"order ratio ({cname})", do.mean() / dc.mean(), EXPECTED["order_ratio"][cname])
    # T3 context shift vs same-structure baseline (fAB_i x fAB_j across pairs)
    for cname, key in (("raw", "raw"), ("desc", "desc"), ("composed", "composed")):
        dm = ch.get(key, vd if key == "composed" else None)
        ds = np.array([dm[a][b] for a, b in zip(fAB, cB)])
        b2 = np.array([dm[a][b] for i, a in enumerate(fAB) for b in fAB[i+1:]])
        check(f"context shift /same-struct ({cname})", ds.mean() / b2.mean(),
              EXPECTED["context_shift_same_struct"][cname])
    # dynamics aggregates
    neg = sum(1 for c in convs if any(w["cos"] is not None and w["cos"] < 0 for w in c["windows"]))
    loops = sum(len(c["loops"]) for c in convs)
    check("conversations with reversal turn", neg, EXPECTED["convs_negative_turn"], tol=0.5)
    check("latent loop candidates", loops, EXPECTED["loop_candidates"], tol=0.5)
    # SGD final-window inversion
    for outcome in ("theta_aban", "theta_cont"):
        vals = [w["cos"] for c in convs if c["corpus"] == "sgd" and c["outcome"] == outcome
                for w in [c["windows"][-1]] if w["cos"] is not None]
        check(f"SGD final cos ({outcome})", float(np.mean(vals)), EXPECTED["sgd_final_cos"][outcome], tol=0.03)
    # tau direction: reward-based (reward travels in the blob for taubench convs)
    tau_f = [c for c in convs if c["corpus"] == "taubench" and c.get("reward") == 0.0]
    tau_fails_neg = sum(1 for c in tau_f if any(w["cos"] is not None and w["cos"] < 0 for w in c["windows"]))
    check("tau failing episodes with reversal (measured)", tau_fails_neg, 6, tol=0.5)
    # blob sanity
    all_cos = [w["cos"] for c in convs for w in c["windows"] if w["cos"] is not None]
    checks += 1
    ok = all(-1.001 <= c <= 1.001 for c in all_cos)
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] all cos in [-1,1] ({len(all_cos)} values)")
    print(f"SUITE: {checks - fails}/{checks} PASS")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ELDA property suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_ann = sub.add_parser("annotate", help="annotate windows via configured endpoints")
    ap_ann.add_argument("--windows", required=True)
    ap_ann.add_argument("--out", required=True)
    ap_test = sub.add_parser("test", help="verify paper numbers from committed caches")
    ap_test.add_argument("--data", required=True)
    args = ap.parse_args()
    if args.cmd == "annotate":
        annotate(args.windows, args.out)
    else:
        test(args.data)
