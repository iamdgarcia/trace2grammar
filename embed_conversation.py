#!/usr/bin/env python3
"""embed_conversation: conversation + metadata -> latent representation (embedding).

The simplest entry point for the pipecat-evaluations use case: feed it a
conversation (a pipecat EvalConversation file, a simulation results.jsonl
record, an otel/openinference/langsmith export, a tau-bench episode, or a
plain pasted transcript), add the domain and any metadata, and get back the
grammar embedding: the walk plus its grammar feature vector, optionally
projected onto a fitted 2D behavior plane.

Usage:
    python embed_conversation.py conv.json --fmt pipecat-evals --domain banking --meta model=gpt-4o
    python embed_conversation.py run/results.jsonl                # one embedding per line
    python embed_conversation.py call.txt                         # pasted transcript (default fmt)
    python embed_conversation.py corpus/*.json --fmt transcript --save-basis basis.json
    python embed_conversation.py conv.json --fmt pipecat-evals --basis basis.json   # project onto it
    python embed_conversation.py --self-check

Output: one JSON object (single input conversation) or a JSON array (batch),
printed to stdout or written with --out.

HONEST STATUS: rule-based baseline classifier (paper seed); the walk is
outcome-free by mandate (terminal classes live in the paper's decision layer,
not in the representation).
"""

import argparse
import json
import sys

from trace2grammar.adapters import ADAPTERS, fit_basis, project_features, to_grammar


def sniff(path: str) -> str:
    """File extension -> payload kind: {'json','jsonl','transcript'}."""
    p = path.lower()
    if p.endswith(".jsonl"):
        return "jsonl"
    if p.endswith(".json"):
        return "json"
    return "transcript"


def load(path: str) -> tuple[str, list]:
    """-> (kind, payloads). kind in {'json','jsonl','transcript'}."""
    if path == "-":
        return "transcript", [sys.stdin.read()]
    text = open(path, encoding="utf-8").read()
    kind = sniff(path)
    if kind == "json":
        return "json", [json.loads(text)]
    if kind == "jsonl":
        return "jsonl", [json.loads(line) for line in text.splitlines() if line.strip()]
    return "transcript", [text]


def resolve_fmt(kind: str, raw, fmt_arg: str | None) -> str | None:
    """--fmt wins; else sniff the payload shape."""
    if fmt_arg:
        return fmt_arg
    if kind == "transcript":
        return "transcript"
    if isinstance(raw, list):  # bare EvalConversation file: JSON list of messages
        return "pipecat-evals"
    if isinstance(raw, dict) and any(
        k in raw for k in ("messages", "conversation", "transcript", "turns")
    ):
        return "pipecat-evals"  # a pipecat results.jsonl record
    try:
        from trace2grammar.coverage import detect_fmt

        return detect_fmt(raw)  # otel / openinference / langsmith by content
    except ValueError:
        return None


def _self_check() -> int:
    """The one runnable check: synthetic pipecat conversation -> embedding -> plane."""
    conv = [
        {"role": "assistant", "content": "Hi, how can I help?"},
        {"role": "user", "content": "I need to change my flight to Friday."},
        {"role": "tool_call", "name": "verify_identity", "arguments": {"dob": "1990-01-01"}, "cancelled": False},
        {"role": "assistant", "content": "Thanks, you are verified."},
        {"role": "tool_call", "name": "change_flight", "arguments": {"day": "Friday"}, "cancelled": False},
        {"role": "assistant", "content": "Done. Anything else?"},
        {"role": "user", "content": "No, thanks."},
        {"role": "assistant", "content": "Have a great day, goodbye!"},
    ]
    emb = to_grammar(conv, "pipecat-evals", meta={"domain": "banking", "model": "demo"})
    assert emb["walk"][0] == "q_greet", emb["walk"]
    assert emb["walk"][-1] == "q_close", emb["walk"]
    assert "q_auth" in emb["walk"] and "q_tool" in emb["walk"], emb["walk"]
    assert all(not s.startswith("theta_") for s in emb["walk"]), "outcome leaked into the walk"
    assert "outcome" not in emb, "outcome leaked into the embedding"
    assert emb["meta"] == {"domain": "banking", "model": "demo"}
    assert emb["cycle_depth"] == 0 and emb["error_count"] == 0
    assert len(emb["features"]) == 20, len(emb["features"])  # 3 summary + 11 visits + 2 counts + 4 zeroed onehot
    # plane: fit on 3 conversations, project back, parity vs coverage.pca2d
    from trace2grammar.adapters import ADAPTERS  # noqa: F401  (registry already warm)
    from trace2grammar.coverage import pca2d, znorm

    rows = [emb["features"]]
    for drop in (5, 7):  # two truncated variants: same machine, shorter trajectories
        rows.append(to_grammar(conv[:-drop], "pipecat-evals")["features"])
    basis = fit_basis(rows)
    assert any(s > 1e-12 for s in basis["col_sd"]), "degenerate basis: all feature columns constant"
    for i, row in enumerate(rows):
        px, py = project_features(row, basis)
        rx, ry = pca2d(znorm(rows))[i]
        assert abs(px - rx) < 1e-9 and abs(py - ry) < 1e-9, (i, (px, py), (rx, ry))
    print("SELF-CHECK OK: pipecat-evals -> embedding -> 2D plane parity")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="embed_conversation",
        description="conversation + metadata -> grammar embedding (walk + features [+ 2d])",
    )
    ap.add_argument("inputs", nargs="*", help="conversation file(s): .json / .jsonl / .txt, or '-' for stdin")
    ap.add_argument("--fmt", default=None,
                    help="transport: " + " | ".join(sorted(ADAPTERS)) + " (default: sniffed from shape)")
    ap.add_argument("--key", default=None,
                    help="results.jsonl key holding the messages (default: messages|conversation|transcript|turns)")
    ap.add_argument("--domain", default=None, help="domain tag stored in meta")
    ap.add_argument("--meta", action="append", default=[], metavar="K=V", help="extra metadata (repeatable)")
    ap.add_argument("--basis", default=None, help="basis JSON (from --save-basis): adds a 2d point")
    ap.add_argument("--save-basis", default=None, help="fit the 2D plane on the inputs and save the basis here")
    ap.add_argument("--out", default=None, help="write the result here instead of stdout")
    ap.add_argument("--self-check", action="store_true", help="run the built-in check and exit")
    args = ap.parse_args(argv)

    if args.self_check:
        return _self_check()
    if not args.inputs:
        ap.error("give at least one input file (or '-' for stdin)")

    meta: dict = {}
    for kv in args.meta:
        if "=" not in kv:
            ap.error(f"--meta expects K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        meta[k] = v
    if args.domain:
        meta["domain"] = args.domain

    embeddings = []
    for path in args.inputs:
        kind, payloads = load(path)
        for i, raw in enumerate(payloads):
            where = f"{path}[{i}]"
            fmt = resolve_fmt(kind, raw, args.fmt)
            if fmt is None:
                raise SystemExit(f"{where}: cannot infer format; pass --fmt ({' | '.join(sorted(ADAPTERS))})")
            kwargs = {"key": args.key} if (fmt.startswith("pipecat") and args.key) else {}
            try:
                embeddings.append(to_grammar(raw, fmt, meta=meta, **kwargs))
            except (KeyError, ValueError) as e:
                raise SystemExit(f"{where}: {e}")

    if args.save_basis:
        basis = fit_basis([e["features"] for e in embeddings])
        with open(args.save_basis, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(basis, fh, indent=1)
        print(f"basis saved: {args.save_basis} (fit on {len(embeddings)} conversations)", file=sys.stderr)

    if args.basis:
        with open(args.basis, encoding="utf-8") as fh:
            basis = json.load(fh)
        for e in embeddings:
            e["2d"] = [round(v, 4) for v in project_features(e["features"], basis)]

    result = embeddings[0] if (len(embeddings) == 1 and len(args.inputs) == 1) else embeddings
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
