"""_prep_views.py — add the semantic-only plane (alpha=1) to data.js (E9's two-views design).

PCA on the unique opening embeddings (810) -> project every real row by key; sims by their
opening keys. Points that share an opening share a position: that collapse IS the E2 finding,
visible. Rewrites data.js with a sem:{x,y} field per point. No model calls.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "C:/Users/U77225D/Desktop/proyectos/hf"
OUT = os.path.join(HF, "builds", "conversation-latent-map")

emb = json.load(open(os.path.join(HF, "runtime/latent2/openings_embed.json"), encoding="utf-8"))
data_js = open(os.path.join(OUT, "data.js"), encoding="utf-8").read()
data = json.loads(data_js[data_js.find("{"):data_js.rfind("}") + 1])

keys = sorted(emb)
M = np.array([emb[k] for k in keys])
km = {k: i for i, k in enumerate(keys)}
mean = M.mean(axis=0)
X = M - mean
C = (X.T @ X) / (len(X) - 1)
w, V = np.linalg.eigh(C)
v1, v2 = V[:, -1], V[:, -2]
proj = lambda k: (round(float((emb[k] - mean) @ v1), 4), round(float((emb[k] - mean) @ v2), 4))

rows = list(csv.DictReader(open(os.path.join(HF, "builds/trace2grammar/dataset/taubench/episodes.csv"), encoding="utf-8")))
assert len(rows) == len(data["real"])
for r, p in zip(rows, data["real"]):
    p["sem"] = proj(r["first_user_text"].strip())
for s in data["sim"]:
    k = s["t"].strip()
    if k in km:
        s["sem"] = proj(k)
    else:  # sim opening never embedded (defensive): reuse composed position
        s["sem"] = [s["x"], s["y"]]

js = "const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
open(os.path.join(OUT, "data.js"), "w", encoding="utf-8").write(js)
print("data.js updated:", len(js) // 1024, "KB | sem views on", len(data["real"]), "+", len(data["sim"]))
