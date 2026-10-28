#!/usr/bin/env python3
"""Parity check: python re-derives the induced grammar from atlas_data.js with an
independent implementation and compares it, number by number, against the digest
that node smoke_grammar.js wrote (grammar_derived.js -> grammar_derived.json).

Run (from the repo root, or anywhere): python check_parity_grammar.py <space_dir>
Exits non-zero on any mismatch.
"""
import json
import sys
from collections import Counter
from pathlib import Path

SPINE = {"q_greet": 0, "q_auth": 1, "q_intent": 2, "q_tool": 3, "q_clarify": 4, "q_confirm": 5, "q_close": 6}


def edge_type(a: str, b: str) -> str:
    if b.startswith("theta_"):
        return "terminal"
    if a == b:
        return "self"
    if SPINE.get(b, 0) < SPINE.get(a, 0):
        return "backward"
    return "progress"


def main() -> int:
    space = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "trajectory-grammar-atlas")
    src = (space / "atlas_data.js").read_text(encoding="utf-8")
    blob = json.loads(src[src.index("=") + 1:].rstrip().rstrip(";"))
    real = [p for p in blob["points"] if p["kind"] == "real"]
    derived = json.loads((space / "grammar_derived.json").read_text(encoding="utf-8"))

    edges: Counter = Counter()
    templates: Counter = Counter()
    visits: Counter = Counter()
    reach = {"theta_cont": [0, 0], "theta_aban": [0, 0], "theta_hand": [0, 0], "theta_esc": [0, 0]}
    n_trans = 0
    for p in real:
        w = p["walk"]
        for i, st in enumerate(w):
            visits[st] += 1
            b = w[i + 1] if i < len(w) - 1 else p["outcome"]
            edges[(st, b)] += 1
            n_trans += 1
        comp = []
        for st in w:
            if not comp or comp[-1] != st:
                comp.append(st)
        templates[(">".join(comp), p["outcome"])] += 1
        reach[p["outcome"]][0] += 1 if "q_close" in w else 0
        reach[p["outcome"]][1] += 1

    fails = []

    def check(name, mine, theirs):
        if mine != theirs:
            fails.append(f"{name}: python {mine!r} != js {theirs!r}")

    check("nRuns", len(real), derived["nRuns"])
    check("nTransitions", n_trans, derived["nTransitions"])
    check("nDistinctEdges", len(edges), derived["nDistinctEdges"])
    py_edges = [{"a": a, "b": b, "count": c, "type": edge_type(a, b)} for (a, b), c in
                sorted(edges.items(), key=lambda kv: (-kv[1], kv[0]))]
    check("edges (all, ordered)", py_edges, derived["edges"])
    py_templates = [{"seq": seq, "outcome": oc, "count": c} for (seq, oc), c in
                    sorted(templates.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))]
    check("top templates", py_templates[: len(derived["topTemplates"])], derived["topTemplates"])
    check("nTemplates", len(templates), derived["nTemplates"])
    check("reachClose", reach, derived["reachClose"])
    check("visits", dict(visits), {k: int(v) for k, v in derived["visits"].items()})
    top6 = sum(c for _, c in edges.most_common(6))
    check("top6Share", round(top6 / n_trans, 10), round(derived["top6Share"], 10))

    if fails:
        print("PARITY FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print(f"PARITY OK: python and JS agree on {len(edges)} edges / {n_trans} transitions / "
          f"{len(templates)} templates over {len(real)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
