"""Conversation -> grammar adapters.

The abstract seam of the extraction layer: a named registry that turns a raw
conversation export (whatever its transport) into the canonical event stream,
plus one entry point, `to_grammar()`, that returns the latent representation of
a trajectory.

Built-in adapters cover the formats grammaticality is checked from:
'otel' (OTel GenAI), 'openinference', 'langsmith', 'taubench', 'transcript'
(the universal paste format), and 'pipecat-evals' (pipecat's EvalConversation
files — JSON lists of user/assistant/tool_call messages — and its simulation
`results.jsonl` records, whose `messages` field holds the same list).

Adding a transport is one function + `register_adapter(name, fn)`; the grammar
walk machine is shared and unchanged.

HONEST STATUS: the state classifier is the rule-based baseline of the paper
(seed, not the final learned emission model).
"""

from __future__ import annotations

import math
from typing import Any, Callable

from .coverage import _pca_basis, conversation_features, detect_fmt, znorm
from .extract import (
    Event,
    ExtractionConfig,
    PROGRESS_STATES,
    TAUBENCH_CONFIG,
    infer_states,
    normalize_trace,
)

AdapterFn = Callable[[Any, ExtractionConfig], list[Event]]

ADAPTERS: dict[str, AdapterFn] = {}


def register_adapter(name: str, fn: AdapterFn) -> None:
    """Register a transport under a name usable from `to_grammar(..., fmt=name)`."""
    ADAPTERS[name] = fn


def _dialect(dialect: str) -> AdapterFn:
    def fn(raw, cfg):
        return normalize_trace(raw, dialect, cfg)

    return fn


for _d in ("otel", "openinference", "langsmith", "taubench", "transcript"):
    register_adapter(_d, _dialect(_d))


# ---------------------------------------------------------------- pipecat-evals


_PIPECAT_KEYS = ("messages", "conversation", "transcript", "turns")


def _pipecat_messages(obj, key: str | None = None) -> list[dict]:
    """Pull the message list out of any pipecat-evals shape.

    Accepts the bare EvalConversation file (a JSON list of messages) or a
    results.jsonl record (dict) whose conversation lives under `key` (default
    order: messages, conversation, transcript, turns).
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ((key,) if key else _PIPECAT_KEYS):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    raise ValueError(
        "no conversation found: pass an EvalConversation file (JSON list) or a "
        "results.jsonl record with a 'messages'/'conversation' list "
        "(or pass --key)"
    )


def normalize_pipecat_evals(obj, cfg: ExtractionConfig, key: str | None = None) -> list[Event]:
    """pipecat-evals conversation -> canonical events.

    Message roles map 1:1 to canonical kinds. A `tool_call` message emits only
    the tool_call event (pipecat conversation files carry no tool results and
    no timing); `cancelled` rides in attrs. Timestamps are monotonic synthetic
    stamps (message index * 1000) preserving order only.
    """
    events: list[Event] = []
    for i, m in enumerate(_pipecat_messages(obj, key)):
        t = i * 1000
        role = m.get("role")
        if role == "user":
            events.append(
                {"t_ms": t, "kind": "user_msg", "name": None, "attrs": {"content": str(m.get("content") or "")}}
            )
        elif role == "assistant":
            events.append(
                {"t_ms": t, "kind": "assistant_msg", "name": None, "attrs": {"content": str(m.get("content") or "")}}
            )
        elif role == "tool_call":
            events.append(
                {
                    "t_ms": t,
                    "kind": "tool_call",
                    "name": str(m.get("name") or "tool"),
                    "attrs": {"cancelled": bool(m.get("cancelled")), "arguments": m.get("arguments") or {}},
                }
            )
    return events


register_adapter("pipecat-evals", normalize_pipecat_evals)
register_adapter("pipecat", normalize_pipecat_evals)


# ---------------------------------------------------------------- the embedding


def to_grammar(
    raw,
    fmt: str | None = None,
    config: ExtractionConfig | None = None,
    meta: dict | None = None,
    **adapter_kwargs,
) -> dict:
    """Raw conversation export -> latent representation (the embedding).

    Returns {walk, cycle_depth, error_count, silent_failure_candidate, features,
    meta} where walk is the PROGRESS-state trajectory (outcome-free by mandate:
    terminal classes belong to the paper's decision layer, not to the
    representation) and features is the walk's grammar feature vector — the
    input of the 2D plane (see `fit_basis` / `project_features`).

    adapter_kwargs pass through to the transport (e.g. pipecat-evals' `key`).
    """
    if fmt is None:
        fmt = detect_fmt(raw)
    if fmt not in ADAPTERS:
        raise KeyError(f"unknown fmt {fmt!r}; built-ins: {', '.join(sorted(ADAPTERS))}")
    if config is None and fmt == "taubench":
        config = TAUBENCH_CONFIG
    cfg = config or ExtractionConfig()
    events = ADAPTERS[fmt](raw, cfg, **adapter_kwargs)
    result = infer_states(events, cfg)
    return {
        "walk": [s["state"] for s in result["walk"] if s["state"] in PROGRESS_STATES],
        "cycle_depth": result["cycle_depth"],
        "error_count": result["error_count"],
        "silent_failure_candidate": result["silent_failure_candidate"],
        "features": conversation_features(
            {**result, "outcome": None}, mode="grammar"
        ),
        "meta": dict(meta or {}),
    }


# ---------------------------------------------------------------- 2D plane


def fit_basis(feature_rows: list[list[float]], mode: str = "grammar") -> dict:
    """Fit the 2D behavior plane on a corpus of grammar feature vectors.

    The returned basis carries the per-column znorm stats (col_mean/col_sd) so
    single new conversations can be projected onto the SAME axes (comparability
    with the corpus the plane was fit on).
    """
    zrows = znorm(feature_rows)
    mean, v1, v2 = _pca_basis(zrows)
    stats = _znorm_stats(feature_rows)
    return {"mode": mode, "col_mean": stats["col_mean"], "col_sd": stats["col_sd"],
            "mean": list(mean), "v1": list(v1), "v2": list(v2)}


def _znorm_stats(feats_raw) -> dict:
    """col_mean/col_sd of the raw feature matrix (z = (x-m)/sd, mirroring coverage.znorm)."""
    cols = list(zip(*feats_raw))
    m = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((x - mi) ** 2 for x in c) / len(c)) for c, mi in zip(cols, m)]
    return {"col_mean": m, "col_sd": sd}


def project_features(feats: list[float], basis: dict) -> tuple[float, float]:
    """Grammar feature vector -> (x, y) on the basis's plane."""
    z = [
        (x - mi) / si if si > 1e-12 else 0.0
        for x, mi, si in zip(feats, basis.get("col_mean", basis["mean"]), basis.get("col_sd", [1.0] * len(basis["mean"])))
    ]
    mu = basis["mean"]
    dz = [z[j] - mu[j] for j in range(len(mu))]
    x = sum(dz[j] * basis["v1"][j] for j in range(len(mu)))
    y = sum(dz[j] * basis["v2"][j] for j in range(len(mu)))
    return (x, y)
