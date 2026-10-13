"""Region coverage: which regions of conversational-behavior space do we NOT test?

Given real-conversation grammar walks and a simulated (or candidate-agent) set, project both
into a 2-D behavior plane, estimate where real conversations live (density grid), and report
which occupied regions the sim set reaches — per outcome class — plus the top-K uncovered
regions with their nearest real exemplars: the tests you are not running, made concrete.

Deterministic, stdlib-only baseline (features -> PCA -> grid density -> coverage). The
embedding-based (MiniLM) variant and voice-physics channels are future work; the module
interface is designed so they replace the feature step only.
"""

from __future__ import annotations

import argparse
import html
import json
import math

from .extract import ALPHABET, TERMINALS, extract

OUTCOME_LABELS = {
    "theta_cont": "contained",
    "theta_aban": "abandoned",
    "theta_hand": "handoff",
    "theta_esc": "compliance-escape",
}


# --------------------------------------------------------------- feature encodings


def conversation_features(res, alphabet=ALPHABET, k=6, mode="grammar") -> list[float]:
    """One walk -> feature vector.

    mode='grammar': [n_states, duration_s, cycle_depth, *per-state visit counts,
                     n_tool_calls, n_errors, *terminal-onehot]
    mode='bigram':  histogram of the first k state-to-state transitions, flattened in the
                    fixed alphabet order (len(alphabet)^2 dims).
    """
    walk = res["walk"]
    states = [s["state"] for s in walk]
    duration_s = (
        ((walk[-1]["t_ms"] - walk[0]["t_ms"]) / 1000.0) if len(walk) > 1 else 0.0
    )
    if mode == "bigram":
        idx = {st: i for i, st in enumerate(alphabet)}
        counts = [0.0] * (len(alphabet) * len(alphabet))
        for a, b in zip(states[: k + 1], states[1 : k + 1]):
            counts[idx.get(a, 0) * len(alphabet) + idx.get(b, 0)] += 1.0
        return counts
    visits = [float(states.count(st)) for st in alphabet]
    onehot = [1.0 if res["outcome"] == t else 0.0 for t in TERMINALS]
    n_tool_calls = sum(1 for s in walk if s["transition_type"] == "tool-call")
    return [
        float(len(walk)),
        duration_s,
        float(res["cycle_depth"]),
        *visits,
        float(n_tool_calls),
        float(res.get("error_count", 0)),
        *onehot,
    ]


def znorm(rows: list[list[float]]) -> list[list[float]]:
    """Column z-normalization; constant columns map to 0 (no NaN on zero variance)."""
    cols = list(zip(*rows))
    out = []
    for col in cols:
        m = sum(col) / len(col)
        sd = math.sqrt(sum((x - m) ** 2 for x in col) / len(col))
        out.append([(x - m) / sd if sd > 1e-12 else 0.0 for x in col])
    return [list(r) for r in zip(*out)]


def feature_matrix(
    results, mode="grammar", alphabet=ALPHABET, k=6
) -> list[list[float]]:
    return znorm([conversation_features(r, alphabet, k, mode) for r in results])


# --------------------------------------------------------------- 2-D projection


def _pca_basis(rows):
    """Fit PCA: returns (mean, v1, v2) so sim sets can re-project onto the SAME axes."""
    n, d = len(rows), len(rows[0])
    mean = [sum(col) / n for col in zip(*rows)]
    pts = [[r[j] - mean[j] for j in range(d)] for r in rows]
    cov = [
        [sum(pts[i][a] * pts[i][b] for i in range(n)) / (n - 1) for b in range(d)]
        for a in range(d)
    ]

    def power_iter(M):
        v = [1.0 / math.sqrt(d)] * d
        for _ in range(200):
            w = [sum(M[a][b] * v[b] for b in range(d)) for a in range(d)]
            norm = math.sqrt(sum(x * x for x in w))
            if norm < 1e-15:
                return v
            v = [x / norm for x in w]
        return v

    v1 = power_iter(cov)
    lam1 = sum(v1[a] * cov[a][b] * v1[b] for a in range(d) for b in range(d))
    cov2 = [[cov[a][b] - lam1 * v1[a] * v1[b] for b in range(d)] for a in range(d)]
    v2 = power_iter(cov2)
    p = sum(v2[a] * v1[a] for a in range(d))
    v2 = [v2[a] - p * v1[a] for a in range(d)]
    n2 = math.sqrt(sum(x * x for x in v2))
    if n2 > 1e-15:
        v2 = [x / n2 for x in v2]
    return mean, v1, v2


def pca2d(rows: list[list[float]]) -> list[tuple[float, float]]:
    """Project onto the top-2 principal components (power iteration + deflation)."""
    n, d = len(rows), len(rows[0])
    if n < 2 or d < 2:
        return [(r[0] if d else 0.0, 0.0) for r in rows]
    mean, v1, v2 = _pca_basis(rows)
    pts = [[r[j] - mean[j] for j in range(d)] for r in rows]
    return [
        (
            sum(pts[i][a] * v1[a] for a in range(d)),
            sum(pts[i][a] * v2[a] for a in range(d)),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------- density + coverage


def _grid_transform(real2d, res, pad_frac):
    xs = [p[0] for p in real2d]
    ys = [p[1] for p in real2d]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ext = max(x1 - x0, y1 - y0)
    pad = ext * pad_frac if ext > 1e-12 else 1.0
    x0, y0 = x0 - pad, y0 - pad
    return x0, y0, (x1 + pad - x0) / res, (y1 + pad - y0) / res


def _cell_of(x, y, x0, y0, cw, ch, res):
    i = max(0, min(res - 1, int((x - x0) / cw)))
    j = max(0, min(res - 1, int((y - y0) / ch)))
    return i, j


def _bump_grid(pts, res, sigma, x0, y0, cw, ch, normalize_by=None, skip_outside=False):
    """Gaussian-bump accumulation on the grid.

    normalize_by: when given, scale by that constant instead of the field's own peak — used
    so the sim field shares the REAL field's scale (self-normalization would destroy the
    relative magnitudes that coverage compares).
    skip_outside: drop points outside the grid instead of clamping them to the border cells
    (a sim point 10 extents away must not look like a corner point).
    """
    g = [[0.0] * res for _ in range(res)]
    for x, y in pts:
        if skip_outside and not (x0 <= x <= x0 + res * cw and y0 <= y <= y0 + res * ch):
            continue
        ci, cj = _cell_of(x, y, x0, y0, cw, ch, res)
        for i in range(res):
            dx = (i - ci) ** 2
            for j in range(res):
                g[i][j] += math.exp(-(dx + (j - cj) ** 2) / (2 * sigma * sigma))
    scale = (
        normalize_by
        if normalize_by
        else (max((max(row) for row in g), default=0.0) or 1.0)
    )
    return [[v / scale for v in row] for row in g]


def _box_blur(g, iters):
    res = len(g)
    for _ in range(iters):
        g = [
            [
                sum(
                    g[a][b]
                    for a in (i - 1, i, i + 1)
                    if 0 <= a < res
                    for b in (j - 1, j, j + 1)
                    if 0 <= b < res
                )
                / 9
                for j in range(res)
            ]
            for i in range(res)
        ]
    return g


def coverage(
    real2d, sim2d, classes=None, res=24, eps=0.05, bw_frac=0.12, blur_iters=2
) -> dict:
    """Occupied-real-cell coverage by the sim set.

    A cell is covered when a sim point falls inside it OR the sim kernel puts >= eps relative
    mass there (point membership alone would undercount contiguous regions). Uncovered cells
    carry their nearest real exemplar index and local real density.

    eps is relative to the blurred real field's own peak, and occupancy additionally requires
    the cell to sit within the dense core: eps is compared against the cell value rescaled by
    the field's 99th-percentile-to-peak ratio. In practice: eps=0.05 keeps compact regions
    around the real mass instead of the blur's long tails.
    """
    if not real2d:
        raise ValueError("real set is empty")
    x0, y0, cw, ch = _grid_transform(real2d, res, 0.15)
    sigma = bw_frac * res
    real_g = _box_blur(_bump_grid(real2d, res, sigma, x0, y0, cw, ch), blur_iters)
    g_x1, g_y1 = x0 + res * cw, y0 + res * ch
    sim_cells = {
        _cell_of(x, y, x0, y0, cw, ch, res)
        for x, y in sim2d
        if x0 <= x <= g_x1 and y0 <= y <= g_y1
    }

    # one shared threshold: occupancy AND coverage both compare against eps * core(real)
    real_peak = max((max(row) for row in real_g), default=0.0) or 1.0
    flat = sorted((v for row in real_g for v in row), reverse=True)
    p99 = flat[max(0, int(0.01 * len(flat)))] if flat else 1.0
    core = max(p99, 0.5 * real_peak)
    thresh = eps * core
    sim_g = _box_blur(
        _bump_grid(
            sim2d, res, sigma, x0, y0, cw, ch, normalize_by=real_peak, skip_outside=True
        ),
        blur_iters,
    )

    occupied = [(i, j) for i in range(res) for j in range(res) if real_g[i][j] > thresh]
    uncovered = []
    n_covered = 0
    # nearest real exemplar per occupied cell (for classes + uncovered metadata)
    cell_meta: dict[tuple, int] = {}
    for i, j in occupied:
        cx, cy = x0 + (i + 0.5) * cw, y0 + (j + 0.5) * ch
        idx = min(
            range(len(real2d)),
            key=lambda k: (real2d[k][0] - cx) ** 2 + (real2d[k][1] - cy) ** 2,
        )
        cell_meta[(i, j)] = idx
        if (i, j) in sim_cells or sim_g[i][j] >= thresh:
            n_covered += 1
        else:
            uncovered.append(
                {
                    "gx": i,
                    "gy": j,
                    "cx": cx,
                    "cy": cy,
                    "exemplar_idx": idx,
                    "local_real_density": real_g[i][j],
                }
            )
    per_class: dict[str, float] = {}
    if classes is not None:
        uncovered_set = {(u["gx"], u["gy"]) for u in uncovered}
        for cl in sorted(set(classes)):
            cells = [c for c, ix in cell_meta.items() if classes[ix] == cl]
            if not cells:
                continue
            cov_n = sum(1 for c in cells if c not in uncovered_set)
            per_class[cl] = round(100.0 * cov_n / len(cells), 1)
    return {
        "coverage_pct": round(100.0 * n_covered / len(occupied), 1)
        if occupied
        else 100.0,
        "occupied_cells": len(occupied),
        "covered_cells": n_covered,
        "uncovered_cells": uncovered,
        "per_class_coverage": per_class,
        "grid": {
            "res": res,
            "eps": eps,
            "bw_frac": bw_frac,
            "sigma_cells": round(sigma, 2),
        },
        "real2d": real2d,
        "sim2d": sim2d,
        "classes": classes,
    }


def top_k_uncovered(cov: dict, k: int) -> list[dict]:
    """Highest-density uncovered regions, neighbors deduped: the tests you are not running."""
    cells = sorted(cov["uncovered_cells"], key=lambda u: -u["local_real_density"])
    picked: list[dict] = []
    for u in cells:
        if any(
            max(abs(u["gx"] - p["gx"]), abs(u["gy"] - p["gy"])) <= 2 for p in picked
        ):
            continue
        picked.append(u)
        if len(picked) == k:
            break
    return picked


# --------------------------------------------------------------- walk-level plumbing


def walk_summary(res) -> dict:
    """Compact exemplar metadata for reports/tooltips."""
    walk = res["walk"]
    duration_ms = (walk[-1]["t_ms"] - walk[0]["t_ms"]) if len(walk) > 1 else 0
    first_user = next(
        (
            str((e.get("attrs") or {}).get("content", ""))
            for e in res.get("events", [])
            if e["kind"] == "user_msg"
        ),
        "",
    )
    return {
        "outcome": res["outcome"],
        "cycle_depth": res["cycle_depth"],
        "duration_ms": duration_ms,
        "n_states": len(walk),
        "first_user_text": first_user,
    }


def coverage_from_walks(
    real_results, sim_results, res=24, eps=0.05, mode="grammar"
) -> dict:
    """extract()-results -> feature matrix -> PCA -> coverage dict + exemplar summaries.

    The PCA basis is fit on the REAL set only — the behavior space is defined by production;
    sim points are projected into it (joint fitting would let the sim set move the axes).
    """
    real_m = feature_matrix(real_results, mode=mode)
    sim_m = feature_matrix(sim_results, mode=mode)
    mean, v1, v2 = _pca_basis(real_m)
    d = len(real_m[0])

    def project(m):
        return [
            (
                sum((r[j] - mean[j]) * v1[j] for j in range(d)),
                sum((r[j] - mean[j]) * v2[j] for j in range(d)),
            )
            for r in m
        ]

    real2d = project(real_m)
    sim2d = project(sim_m) if sim_m else []
    real_meta = [walk_summary(r) for r in real_results]
    cov = coverage(
        real2d, sim2d, res=res, eps=eps, classes=[m["outcome"] for m in real_meta]
    )
    cov["real_meta"] = real_meta
    cov["sim_meta"] = [walk_summary(r) for r in sim_results]
    return cov


# --------------------------------------------------------------- reports


def _exemplar_label(meta: dict) -> str:
    text = meta["first_user_text"][:80]
    dur = f"{meta['duration_ms'] / 1000:.1f}s"
    return (
        f'"{text}" [{OUTCOME_LABELS.get(meta["outcome"], meta["outcome"])}, '
        f"D={meta['cycle_depth']}, {dur}]"
    )


def coverage_html(
    cov: dict, title="Region coverage — real vs sim behavior space"
) -> str:
    W, H, PAD = 680, 440, 52
    pts = cov["real2d"] + cov["sim2d"]
    xs = [p[0] for p in pts] or [0]
    ys = [p[1] for p in pts] or [0]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ex = max(x1 - x0, 1e-9)
    ey = max(y1 - y0, 1e-9)

    def px(x):
        return PAD + (x - x0) / ex * (W - 2 * PAD)

    def py(y):
        return H - PAD - (y - y0) / ey * (H - 2 * PAD)

    x0g, y0g, cw, ch = _grid_transform(cov["real2d"], cov["grid"]["res"], 0.15)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='monospace' font-size='11'>",
        f"<rect width='{W}' height='{H}' fill='#fafafa'/>",
        f"<text x='{PAD}' y='24' fill='#333'>{html.escape(title)}</text>",
        f"<text x='{W // 2 - 40}' y='{H - 14}' fill='#888'>PC1</text>",
        f"<text x='14' y='{H // 2}' fill='#888' transform='rotate(-90 14 {H // 2})'>PC2</text>",
    ]
    for u in cov["uncovered_cells"]:
        gx = px(x0g + u["gx"] * cw)
        gy = py(y0g + (u["gy"] + 1) * ch)
        summary = _exemplar_label(cov["real_meta"][u["exemplar_idx"]])
        parts.append(
            f"<g><title>uncovered: nearest real {html.escape(summary)}</title>"
            f"<rect x='{gx:.1f}' y='{gy:.1f}' width='{px(x0g + (u['gx'] + 1) * cw) - gx:.1f}' "
            f"height='{py(y0g + u['gy'] * ch) - gy:.1f}' fill='#c62828' fill-opacity='0.25'/></g>"
        )
    for meta, (x, y) in zip(cov["real_meta"], cov["real2d"]):
        parts.append(
            f"<g><title>real: {html.escape(_exemplar_label(meta))}</title>"
            f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='5' fill='#1e5aa8' fill-opacity='0.85'/></g>"
        )
    for meta, (x, y) in zip(cov["sim_meta"], cov["sim2d"]):
        cx, cy = px(x), py(y)
        parts.append(
            f"<g><title>sim: {html.escape(_exemplar_label(meta))}</title>"
            f"<polygon points='{cx:.1f},{cy - 6:.1f} {cx + 6:.1f},{cy:.1f} {cx:.1f},{cy + 6:.1f} "
            f"{cx - 6:.1f},{cy:.1f}' fill='#ef6c00' fill-opacity='0.9'/></g>"
        )
    parts.append(
        f"<circle cx='{PAD}' cy='26' r='5' fill='#1e5aa8'/>"
        f"<text x='{PAD + 9}' y='30' fill='#444'>real</text>"
        f"<polygon points='{PAD + 62},20 {PAD + 68},26 {PAD + 62},32 {PAD + 56},26' fill='#ef6c00'/>"
        f"<text x='{PAD + 74}' y='30' fill='#444'>sim</text>"
        f"<rect x='{PAD + 118}' y='20' width='12' height='12' fill='#c62828' fill-opacity='0.25'/>"
        f"<text x='{PAD + 136}' y='30' fill='#444'>uncovered</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def coverage_html_page(cov: dict) -> str:
    pc = ", ".join(
        f"{OUTCOME_LABELS.get(k, k)} {v}%" for k, v in cov["per_class_coverage"].items()
    )
    rows = "".join(
        f"<tr><td>{u['gx']},{u['gy']}</td><td>{u['local_real_density']:.2f}</td>"
        f"<td>{html.escape(_exemplar_label(cov['real_meta'][u['exemplar_idx']]))}</td></tr>"
        for u in top_k_uncovered(cov, 10)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Region coverage report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;margin-top:1rem}}td,th{{border:1px solid #ddd;padding:4px 10px}}
svg{{max-width:100%;height:auto}}</style></head>
<body><h1>Region coverage — what are we NOT testing?</h1>
<p><b>{cov["coverage_pct"]}%</b> of occupied real behavior space is reached by the sim set
({cov["covered_cells"]}/{cov["occupied_cells"]} cells). Per outcome class: {pc or "n/a"}.</p>
{coverage_html(cov)}
<h2>Top uncovered regions (nearest real exemplars = recommended test cases)</h2>
<table><tr><th>cell</th><th>local real density</th><th>nearest real exemplar</th></tr>{rows}</table>
</body></html>"""


# --------------------------------------------------------------- CLI


def detect_fmt(obj) -> str:
    if "resourceSpans" in obj:
        for rs in obj["resourceSpans"]:
            for scope in rs.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    for item in span.get("attributes", []):
                        if item.get("key") == "openinference.span.kind":
                            return "openinference"
        return "otel"
    if "runs" in obj:
        return "langsmith"
    raise ValueError(
        "cannot detect trace format (expected otel/openinference/langsmith)"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="trace2grammar.coverage")
    ap.add_argument(
        "--real", action="append", required=True, help="real trace JSON path"
    )
    ap.add_argument(
        "--sim", action="append", required=True, help="simulated trace JSON path"
    )
    ap.add_argument(
        "--top", type=int, default=3, help="top-K uncovered regions to print"
    )
    ap.add_argument("--res", type=int, default=24, help="grid resolution")
    ap.add_argument(
        "--eps", type=float, default=0.05, help="occupancy threshold (relative)"
    )
    ap.add_argument("--mode", choices=("grammar", "bigram"), default="grammar")
    ap.add_argument("--out", default="coverage_report", help="output file prefix")
    args = ap.parse_args(argv)

    def load_walks(paths):
        out = []
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                obj = json.load(fh)
            out.append(extract(obj, detect_fmt(obj)))
        return out

    real = load_walks(args.real)
    sim = load_walks(args.sim)
    cov = coverage_from_walks(real, sim, res=args.res, eps=args.eps, mode=args.mode)
    print(f"real n={len(real)}  sim n={len(sim)}  features={args.mode}")
    print(
        f"grid {cov['grid']['res']}x{cov['grid']['res']}  occupied {cov['occupied_cells']} cells "
        f"(eps={cov['grid']['eps']}, bw={cov['grid']['bw_frac']})"
    )
    pc = ", ".join(
        f"{OUTCOME_LABELS.get(k, k)} {v}%" for k, v in cov["per_class_coverage"].items()
    )
    print(f"coverage: {cov['coverage_pct']}% overall | per class: {pc or 'n/a'}")
    print(f"top-{args.top} uncovered regions (the tests you are not running):")
    for rank, u in enumerate(top_k_uncovered(cov, args.top), 1):
        meta = cov["real_meta"][u["exemplar_idx"]]
        print(
            f" {rank}. cell ({u['gx']},{u['gy']}) density {u['local_real_density']:.2f} | ",
            end="",
        )
        print(f"nearest real: {_exemplar_label(meta)}")
    with open(f"{args.out}.html", "w", encoding="utf-8") as fh:
        fh.write(coverage_html_page(cov))
    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in cov.items() if k != "classes"}, fh, indent=1)
    print(f"written: {args.out}.html, {args.out}.json")


if __name__ == "__main__":
    main()
