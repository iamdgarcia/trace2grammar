#!/usr/bin/env python3
"""Build station_detector.js: the locked H2 Tier-B farewell centroid, shipped.

Centroid = mean of the 354 S2 dev seed embeddings (MiniLM-L6-v2, mean pooling,
L2-normalized), re-normalized; threshold 0.3 locked on dev F1 (REPORT.md).
Embs come from the frozen emb_cache.npz of the H2 run (no model run here).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
H2 = os.path.join(ROOT, "runtime", "h2_agnostic_close")

z = np.load(os.path.join(H2, "emb_cache.npz"), allow_pickle=True)
embs = {k: z[k] for k in z.files}
seeds = json.load(open(os.path.join(H2, "seeds_dev.json"), encoding="utf-8"))
S2 = seeds["S2"]
assert len(S2) == 354

# the H2 cache stored per-corpus matrices; row order = sorted(set(assistant_lines))
# recomputed EXACTLY as tierB did (dev corpus from the frozen fixtures)
DEV_DIR = os.path.join(ROOT, "runtime", "sgd_validation", "sgd_dev_transcript")
meta = json.load(open(os.path.join(DEV_DIR, "_meta.json"), encoding="utf-8"))
lines = set()
for did in meta:
    txt = open(os.path.join(DEV_DIR, did + ".txt"), encoding="utf-8").read()
    lines.update(l[len("assistant:"):].strip() for l in txt.splitlines() if l.startswith("assistant:"))
uniq = sorted(lines)
assert len(uniq) == embs["dev"].shape[0], f"order mismatch: {len(uniq)} vs {embs['dev'].shape[0]}"
idx = {l: i for i, l in enumerate(uniq)}

missing = [l for l in S2 if l not in idx]
assert not missing, f"seeds missing from cache: {missing[:3]}"

sv = np.vstack([embs["dev"][idx[l]] for l in S2])
cent = sv.mean(axis=0)
cent /= (np.linalg.norm(cent) + 1e-12)

doc = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "mechanism": "cosine(final assistant line, S2 seed centroid); locked t=0.3 on dev F1 (h2 REPORT.md); final-line-only protocol",
    "n_seeds": 354,
    "dim": int(cent.shape[0]),
    "threshold": 0.3,
    "seed": "20261026",
    "centroid": [round(float(x), 6) for x in cent],
}
path = os.path.join(HERE, "station_detector.js")
with open(path, "w", encoding="utf-8") as f:
    f.write("var DETECTOR = " + json.dumps(doc) + ";\n")
print("centroid dim", doc["dim"], "| norm", round(float(np.linalg.norm(cent)), 4))
print("->", path, os.path.getsize(path), "bytes")
