"""E5b — what-if at the STATE level: splice before the first differing tool call (cap 5).

Predeclared (G2): for each mixed twin pair (F fail, P pass) with aligned tool-call
sequences, find the first position j where the tool calls differ (name + normalized JSON
args). Build C_state = F's messages before the assistant message emitting call j, plus P's
messages from that point on. The state damage is now REMOVED (the harmful call never
happens; P's calls run instead).
  flip:   dialogue-only walker verdict (extract + strip_injected_terminal + infer_states)
          on C_state == theta_cont
  rejoin: fraction of post-splice turns k with d_sem(C_k, P_k) <= p90_k (e3's band)
P-E5b PASS: flip >= 60% AND rejoin >= 60%.
"""
import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar")
sys.path.insert(0, "C:/Users/U77225D/Desktop/proyectos/hf/runtime/taubench_anchor")
from trace2grammar.extract import extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from tb_to_events import to_extract_obj, strip_injected_terminal  # noqa: E402

HF = "C:/Users/U77225D/Desktop/proyectos/hf"
CACHE = os.path.join(HF, "runtime", "latent2")

BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("E2_EMBED_MODEL", "gemini-embedding-001")
if not (BASE_URL and API_KEY):
    sys.exit("SIN ENDPOINT: exporta OPENAI_BASE_URL/OPENAI_API_KEY (laya)")


def embed_batch(texts):
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180,
                                        context=ssl.create_default_context()) as r:
                return [x["embedding"] for x in json.load(r)["data"]]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
for r in rows:
    r["reward"] = float(r["reward"])
    r["key"] = r["first_user_text"].strip()
    r["id"] = f"{r['model']}:{r['domain']}:{r['task_id']}:{r['trial']}"
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["domain"], r["key"])].append(r)
pairs, band_pairs = [], []
for g in groups.values():
    fails = [r for r in g if r["reward"] < 1]
    oks = [r for r in g if r["reward"] >= 1]
    if fails and oks:
        pairs.append((fails[0], oks[0]))
    if len(oks) >= 2:
        band_pairs.append((oks[0], oks[1]))

FILES = {"gpt4o": ("gpt-4o-airline.json", "gpt-4o-retail.json"),
         "sonnet": ("sonnet-35-new-airline.json", "sonnet-35-new-retail.json")}
traj = {}
for model, fns in FILES.items():
    for fn in fns:
        domain = "airline" if "airline" in fn else "retail"
        for ep in json.load(open(os.path.join(ANCHOR := os.path.join(HF, "runtime/taubench_anchor"), fn), encoding="utf-8")):
            traj[f"{model}:{domain}:{ep['task_id']}:{ep['trial']}"] = ep["traj"]


def norm_args(s):
    try:
        return json.dumps(json.loads(s), sort_keys=True)
    except Exception:
        return str(s)


def tool_calls_of(t):
    """[(msg_index, name, norm_args)] for every assistant tool call in order."""
    out = []
    for i, m in enumerate(t):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                out.append((i, fn.get("name", "?"), norm_args(fn.get("arguments", ""))))
    return out


def first_split(tf, tp):
    cf, cp = tool_calls_of(tf), tool_calls_of(tp)
    for j in range(min(len(cf), len(cp))):
        if cf[j][1] != cp[j][1] or cf[j][2] != cp[j][2]:
            return cf[j][0], cp[j][0], j
    if len(cf) != len(cp):  # one side makes MORE calls: split at the extra call
        if len(cf) > len(cp):
            return cf[len(cp)][0], None, len(cp)
        return None, cp[len(cf)][0], len(cf)
    return None, None, None


# e3's cached prefix embeddings + band
emb = json.load(open(os.path.join(CACHE, "prefix_embed.json"), encoding="utf-8"))
E = {c: np.array(v) for c, v in emb.items()}
def nrm(v):
    return v / (np.linalg.norm(v) + 1e-12)
def dsem(a, b, k):
    va, vb = E.get(f"{a}::{k}"), E.get(f"{b}::{k}")
    if va is None or vb is None:
        return None
    return float(1.0 - np.dot(nrm(va), nrm(vb)))

band_by_k = defaultdict(list)
for a, b in band_pairs:
    ta, tb = traj[a["id"]], traj[b["id"]]
    ka = sum(1 for m in ta if m.get("role") == "user")
    kb = sum(1 for m in tb if m.get("role") == "user")
    for k in range(1, min(ka, kb) + 1):
        d = dsem(a["id"], b["id"], k)
        if d is not None:
            band_by_k[k].append(d)
p90 = {k: float(np.percentile(v, 90)) for k, v in band_by_k.items() if len(v) >= 5}

# ------------------------------------------------------------------ splice + metrics
cemb = json.load(open(os.path.join(CACHE, "state_splice_embed.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(CACHE, "state_splice_embed.json")) else {}
need = {}
meta = []
for f, p in pairs:
    tf, tp = traj[f["id"]], traj[p["id"]]
    i_f, i_p, j = first_split(tf, tp)
    if i_f is None or i_p is None:
        continue
    C = tf[:i_f] + tp[i_p:]
    cid = f"{f['id']}|{p['id']}"
    kc = sum(1 for m in C if m.get("role") == "user")
    # prefix texts for post-splice turns
    txt, seen = [], 0
    need_local = {}
    for m in C:
        role = m.get("role")
        c = str(m.get("content") or "")[:400]
        if role == "assistant" and m.get("tool_calls"):
            txt.append("assistant: [tool calls]")
        else:
            txt.append(f"{role}: {c}")
        if role == "user":
            seen += 1
            need_local[seen] = "\n".join(txt)
    for k, t in need_local.items():
        if k >= (j + 1):
            need[f"{cid}::{k}"] = t
    meta.append({"pair": (f["id"], p["id"]), "split_call_j": j, "C": C, "kc": kc, "cid": cid})

todo = [c for c in sorted(need) if c not in cemb and need[c].strip()]
print(f"state-splice pairs: {len(meta)} | prefix embeds to fetch: {len(todo)}")
B = 50
texts = need
for s in range(0, len(todo), B):
    chunk = todo[s:s + B]
    t0 = time.time()
    vecs = embed_batch([texts[c] for c in chunk])
    for c, v in zip(chunk, vecs):
        cemb[c] = v
    json.dump(cemb, open(os.path.join(CACHE, "state_splice_embed.json"), "w", encoding="utf-8"))
    print(f"  batch {s//B + 1} ({time.time()-t0:.1f}s)")
json.dump(cemb, open(os.path.join(CACHE, "state_splice_embed.json"), "w", encoding="utf-8"))
EC = {c: np.array(v) for c, v in cemb.items()}

n_flip = n_rejoin = n_done = 0
detail = []
for m in meta:
    n_done += 1
    C_ep = {"task_id": m["pair"][0].split(":")[2], "reward": 1.0, "info": {}, "traj": m["C"]}
    try:
        res = extract(to_extract_obj(C_ep), "taubench")
        convo = infer_states(strip_injected_terminal(res["events"]), TAUBENCH_CONFIG)
        flipped = convo["outcome"] == "theta_cont"
    except Exception:
        flipped = False
    inside = tot = 0
    for k in range(m["split_call_j"], m["kc"] + 1):
        if k not in p90:
            continue
        va, vb = EC.get(f"{m['cid']}::{k}"), E.get(f"{m['pair'][1]}::{k}")
        if va is None or vb is None:
            continue
        tot += 1
        inside += float(1.0 - np.dot(nrm(va), nrm(vb))) <= p90[k]
    rejoin = inside / tot if tot else 0.0
    n_flip += flipped
    n_rejoin += rejoin >= 0.6
    detail.append({"pair": m["pair"], "split_j": m["split_call_j"],
                   "flip": flipped, "rejoin": round(rejoin, 2)})

p_flip = 100.0 * n_flip / n_done if n_done else 0.0
p_rejoin = 100.0 * n_rejoin / n_done if n_done else 0.0
print(f"\nflip (dialogue-only verdict contained tras empalme de estado): {n_flip}/{n_done} = {p_flip:.1f}%")
print(f"rejoin (>=60% post-splice turns en banda): {n_rejoin}/{n_done} = {p_rejoin:.1f}%")
verdict = "PASS" if p_flip >= 60 and p_rejoin >= 60 else "FAIL"
print("VERDICT: P-E5b", verdict)

json.dump({"protocol": {"n_pairs": n_done, "predicate": "flip>=60% AND rejoin>=60%",
                        "split": "before the first differing tool call (state level)"},
           "P_E5b": verdict, "flip_pct": round(p_flip, 1), "rejoin_pct": round(p_rejoin, 1),
           "detail": detail},
          open(os.path.join(HERE, "e5b_results.json"), "w", encoding="utf-8"), indent=1)
