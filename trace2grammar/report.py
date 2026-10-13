"""Per-conversation reports and the (cycle_depth, duration) atlas embryo.

The atlas here is deliberately NOT an embedding projection: it plots the paper's own
two first-class statistics (cycle depth D, duration) so runs of equal length but
different drift (Proposition: longer is not off-path) separate visually. The
embedding-based latent atlas comes later; this is its testable embryo.
"""

from __future__ import annotations

import html
import statistics

from .extract import extract


def conversation_report(obj, fmt: str, config=None) -> dict:
    """Full report for one conversation trace."""
    res = extract(obj, fmt, config)
    events = res["events"]
    walk = res["walk"]
    dwell: dict[str, list[float]] = {}
    for a, b in zip(walk, walk[1:]):
        dwell.setdefault(a["state"], []).append(b["t_ms"] - a["t_ms"])
    duration = (events[-1]["t_ms"] - events[0]["t_ms"]) if events else 0
    return {
        "outcome": res["outcome"],
        "cycle_depth": res["cycle_depth"],
        "error_count": res["error_count"],
        "n_states": len(walk),
        "duration_ms": duration,
        "dwell_ms": {k: round(statistics.median(v)) for k, v in dwell.items()},
        "walk": walk,
        "events": events,
    }


def ascii_timeline(report: dict, width: int = 60) -> str:
    """One-line state timeline: q_greet----q_auth--q_tool~~q_clarify==>theta_hand."""
    walk = report["walk"]
    if not walk:
        return "(empty)"
    t0, t1 = walk[0]["t_ms"], max(walk[-1]["t_ms"], walk[0]["t_ms"] + 1)
    span = t1 - t0
    marks = []
    for i, step in enumerate(walk):
        col = int((step["t_ms"] - t0) / span * (width - 1))
        marks.append((col, step["state"]))
    line = ["-"] * width
    placed = []
    for col, st in marks:
        if placed and col - placed[-1][0] < 2:
            col = placed[-1][0] + 2
        if col >= width:
            col = width - 1
        placed.append((col, st))
    for col, st in placed:
        tag = st.replace("theta_", "θ").replace("q_", "")
        for j, ch in enumerate(tag):
            if col + j < width:
                line[col + j] = ch
        col += len(tag)
        if col < width - 1:
            line[col] = (
                "→" if placed.index((placed[-1][0], placed[-1][1])) >= 0 else "→"
            )
    return "".join(line).rstrip("-→")


OUTCOME_COLORS = {
    "theta_cont": "#2e7d32",
    "theta_aban": "#c62828",
    "theta_hand": "#ef6c00",
    "theta_esc": "#6a1b9a",
}
OUTCOME_LABELS = {
    "theta_cont": "contained",
    "theta_aban": "abandoned",
    "theta_hand": "handoff",
    "theta_esc": "compliance-escape",
}


def atlas_svg(
    reports: list[dict], title: str = "Trajectory atlas — (cycle depth, duration)"
) -> str:
    """Scatter of conversations on the (cycle_depth, duration_norm) plane; color = outcome.

    The embryo of the latent-space atlas: with the paper's discrete statistics first.
    """
    if not reports:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    dmax = max(r["duration_ms"] for r in reports) or 1
    dmaxc = max(r["cycle_depth"] for r in reports) or 1
    W, H, PAD = 640, 400, 56
    plot_w, plot_h = W - PAD * 2, H - PAD * 2

    def x_of(depth):
        return PAD + (depth / dmaxc) * plot_w

    def y_of(dur):
        return H - PAD - (dur / dmax) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='monospace' font-size='11'>",
        f"<rect width='{W}' height='{H}' fill='#fafafa'/>",
        f"<text x='{PAD}' y='24' fill='#333'>{html.escape(title)}</text>",
    ]
    # gridlines
    for g in range(0, dmaxc + 1):
        parts.append(
            f"<line x1='{x_of(g)}' y1='{H - PAD}' x2='{x_of(g)}' y2='{PAD}' stroke='#e0e0e0'/>"
        )
        parts.append(
            f"<text x='{x_of(g) - 4}' y='{H - PAD + 16}' fill='#888'>D={g}</text>"
        )
    parts.append(
        f"<text x='{W // 2 - 60}' y='{H - 16}' fill='#888'>cycle depth (regressions)</text>"
    )
    parts.append(
        f"<text x='12' y='{H // 2}' fill='#888' transform='rotate(-90 12 {H // 2})'>duration (ms)</text>"
    )
    # paths (walk shape inside the plane): polyline of (D-so-far, elapsed-so-far)
    for r in reports:
        color = OUTCOME_COLORS.get(r["outcome"], "#555")
        walk = r["walk"]
        if len(walk) < 2:
            continue
        t0 = walk[0]["t_ms"]
        pts = []
        d_so_far = 0
        for step in walk:
            pts.append(f"{x_of(d_so_far):.1f},{y_of(step['t_ms'] - t0):.1f}")
        pts.append(f"{x_of(r['cycle_depth']):.1f},{y_of(r['duration_ms']):.1f}")
        parts.append(
            f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-opacity='0.35' stroke-width='1.5'/>"
        )
    # endpoint dots
    for r in reports:
        color = OUTCOME_COLORS.get(r["outcome"], "#555")
        cx, cy = x_of(r["cycle_depth"]), y_of(r["duration_ms"])
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{color}' fill-opacity='0.85'/>"
        )
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='9' fill='none' stroke='{color}' stroke-opacity='0.4'/>"
        )
    # legend
    lx = PAD
    for oc in ("theta_cont", "theta_aban", "theta_hand", "theta_esc"):
        parts.append(f"<circle cx='{lx}' cy='24' r='5' fill='{OUTCOME_COLORS[oc]}'/>")
        parts.append(
            f"<text x='{lx + 9}' y='28' fill='#444'>{OUTCOME_LABELS[oc]}</text>"
        )
        lx += 110
    parts.append("</svg>")
    return "\n".join(parts)


def atlas_html(reports: list[dict]) -> str:
    """Standalone HTML page embedding the SVG atlas."""
    body = atlas_svg(reports)
    rows = "".join(
        f"<tr><td>{OUTCOME_LABELS.get(r['outcome'], r['outcome'])}</td><td>{r['cycle_depth']}</td>"
        f"<td>{r['duration_ms']}</td><td>{r['n_states']}</td><td>{r['error_count']}</td></tr>"
        for r in reports
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Trajectory Atlas</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;margin-top:1rem}}td,th{{border:1px solid #ddd;padding:4px 10px}}
svg{{max-width:100%;height:auto}}</style></head>
<body><h1>Trajectory Atlas</h1><p>Each path = one conversation reduced to the grammar's two
first-class statistics. Color = terminal outcome class.</p>
{body}
<table><tr><th>outcome</th><th>cycle depth</th><th>duration (ms)</th><th>states</th><th>tool errors</th></tr>{rows}</table>
</body></html>"""


def bundle(objs_fmts: list[tuple], config=None) -> list[dict]:
    """Report many (obj, fmt) conversations at once."""
    return [conversation_report(o, f, config) for o, f in objs_fmts]
