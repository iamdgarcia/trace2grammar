"""Trace-to-trajectory extraction: convert observability exports into grammar walks.

The bridge between production observability (OTel GenAI, OpenInference, LangSmith) and the
trajectory-grammar formalism G = (Q, Sigma, delta) of "Turn-Level Trajectory Grammars for
Voice Agents" (D. Garcia Pena). Every state and transition corresponds to events a
production cascade already emits, so grammaticality is checkable from telemetry alone.

HONEST STATUS: the state classifier is a rule-based baseline heuristic (keyword patterns +
sequencing rules). It is the seed of the paper's "the extraction layer is measurable"
contribution, NOT the final learned emission model. The config exposes every pattern so a
domain owner can adapt the mapping without touching code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical event kinds (the paper's emitter-side event classes, transport-neutral).
KINDS = (
    "user_msg",
    "assistant_msg",
    "tool_call",
    "tool_result",
    "tool_error",
    "hangup",
    "handoff",
    "session_end",
)

PROGRESS_STATES = (
    "q_greet",
    "q_auth",
    "q_intent",
    "q_tool",
    "q_clarify",
    "q_confirm",
    "q_close",
)
TERMINALS = ("theta_cont", "theta_aban", "theta_hand", "theta_esc")
ALPHABET = PROGRESS_STATES + TERMINALS


@dataclass
class ExtractionConfig:
    """Keyword/state mappings — adapt these per domain instead of editing code."""

    auth_tool_patterns: tuple[str, ...] = ("verify", "identity", "auth", "otp", "login")
    handoff_tool_patterns: tuple[str, ...] = (
        "escalate",
        "transfer",
        "handoff",
        "human",
        "agent_queue",
    )
    clarify_user_patterns: tuple[str, ...] = (
        r"\bperd[oó]n\b",
        r"\bd[ií]galo otra vez\b",
        r"\bno entiendo\b",
        r"\bwhat do you mean\b",
        r"\bcome again\b",
        r"\bcan you repeat\b",
        r"\ba ver\b",
    )
    confirm_user_patterns: tuple[str, ...] = (
        r"\bs[ií]\b",
        r"\bcorrecto\b",
        r"\bconfirmo\b",
        r"\byes\b",
        r"\bthat'?s right\b",
    )
    close_assistant_patterns: tuple[str, ...] = (
        r"\bgracias por su llamada\b",
        r"\bque tenga\b",
        r"\bhave a (good|great)\b",
        r"\bchao\b",
        r"\badi[oó]s\b",
    )
    complete_event_names: tuple[str, ...] = (
        "service_complete",
        "service.complete",
        "call_completed",
        "hangup_complete",
    )
    hangup_event_names: tuple[str, ...] = (
        "caller_hangup",
        "user_hangup",
        "hangup",
        "caller_disconnected",
    )
    handoff_event_names: tuple[str, ...] = (
        "transfer_complete",
        "escalated",
        "human_transfer",
    )
    max_identity_span_ms: int = (
        10_000  # tool_result older than this won't pair with a call
    )


Event = dict  # {t_ms, kind, name, attrs}


def _attr(attrs, key):
    """Fetch one attribute from any of the three attr spellings we support."""
    if isinstance(attrs, dict):
        return attrs.get(key)
    return None


# ---------------------------------------------------------------- normalize


def normalize_trace(
    obj, fmt: str, config: ExtractionConfig | None = None
) -> list[Event]:
    """Convert a raw export into a canonical event stream sorted by time.

    fmt: 'otel' (OTel ResourceSpans with GenAI semconv), 'openinference'
    (OpenInference span attributes), 'langsmith' ({runs: [...]} API export),
    'taubench' (tau-bench episode: {task_id, reward, info, traj[...]} — accepts a
    single episode dict or a list; environment-verified outcome drives the terminal).
    """
    cfg = config or ExtractionConfig()
    if fmt == "otel":
        return _normalize_otel(obj, cfg)
    if fmt == "openinference":
        return _normalize_otel(obj, cfg, openinference=True)
    if fmt == "langsmith":
        return _normalize_langsmith(obj, cfg)
    if fmt == "taubench":
        return _normalize_taubench(obj, cfg)
    raise ValueError(
        f"unknown format: {fmt!r} (use 'otel' | 'openinference' | 'langsmith' | 'taubench')"
    )


# Tuned config for tau-bench episodes: its identification tools (find_user_id_*) are
# the q_auth channel; everything else maps through the default patterns.
TAUBENCH_CONFIG = ExtractionConfig(
    auth_tool_patterns=("verify", "identity", "auth", "otp", "login", "find_user"),
)


def _antml_tool_names(content: str) -> list[str]:
    """Tool names from tau-bench assistant markup (<antml:invoke name="...">)."""
    return re.findall(r'<antml:invoke name="([^"]+)"', content or "")


def _normalize_taubench(obj, cfg: ExtractionConfig) -> list[Event]:
    """tau-bench episode export -> canonical events.

    The export carries NO wall-clock timestamps, so monotonic synthetic stamps
    (message index * 1000) preserve ordering only. Terminal outcome comes from the
    environment-verified reward: pass -> session_end (the walker closes into
    theta_cont); fail -> handoff, the documented public analogue for 'the episode
    needed capabilities the agent could not provide' (NO write-check refinement is
    derivable from the export: only gt_data_hash survives, not the env DB diff).
    Heuristic-vs-environment outcome agreement is therefore a measurable quantity,
    not an assumption.
    """
    episodes = obj if isinstance(obj, list) else [obj]
    events: list[Event] = []
    for ep in episodes:
        base = 0 if not events else events[-1]["t_ms"] + 2000
        traj = ep.get("traj") or []
        for i, msg in enumerate(traj):
            role = msg.get("role")
            content = str(msg.get("content") or "")
            name = msg.get("name")
            t = base + i * 1000
            if role == "user":
                events.append(
                    {
                        "t_ms": t,
                        "kind": "user_msg",
                        "name": None,
                        "attrs": {"content": content},
                    }
                )
            elif role == "assistant":
                calls = _antml_tool_names(content)
                if calls:
                    events.append(
                        {
                            "t_ms": t,
                            "kind": "assistant_msg",
                            "name": None,
                            "attrs": {"content": content},
                        }
                    )
                    for cname in calls:
                        events.append(
                            {"t_ms": t, "kind": "tool_call", "name": cname, "attrs": {}}
                        )
                else:
                    events.append(
                        {
                            "t_ms": t,
                            "kind": "assistant_msg",
                            "name": None,
                            "attrs": {"content": content},
                        }
                    )
            elif role == "tool":
                kind = (
                    "tool_error"
                    if content.lower().startswith("error")
                    else "tool_result"
                )
                events.append(
                    {
                        "t_ms": t,
                        "kind": kind,
                        "name": name,
                        "attrs": {"content": content},
                    }
                )
        reward = ep.get("reward")
        t_end = base + len(traj) * 1000
        if reward == 1.0:
            events.append(
                {
                    "t_ms": t_end,
                    "kind": "session_end",
                    "name": "taubench_pass",
                    "attrs": {"task_id": ep.get("task_id")},
                }
            )
        else:
            # Public analogue: fail -> handoff (see docstring for why not theta_esc).
            events.append(
                {
                    "t_ms": t_end,
                    "kind": "handoff",
                    "name": "taubench_fail",
                    "attrs": {"task_id": ep.get("task_id")},
                }
            )
    return events


def _flatten_attrs(raw) -> dict:
    """OTel-style [{key, value: {stringValue|doubleValue|intValue}}] -> flat dict."""
    out = {}
    if not raw:
        return out
    for item in raw:
        key = item.get("key")
        val = item.get("value") or {}
        for suffix in ("stringValue", "doubleValue", "intValue", "boolValue"):
            if suffix in val:
                out[key] = val[suffix]
                break
    return out


def _ns_to_ms(ns: str | int) -> int:
    return int(ns) // 1_000_000


def _iso_to_ms(iso: str) -> int:
    import datetime

    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _normalize_otel(
    obj, cfg: ExtractionConfig, openinference: bool = False
) -> list[Event]:
    # First pass: collect tool spans keyed by gen_ai.tool.call.id (required by
    # otel-genai #489/#525 on execute_tool spans). Newer exports emit the call and
    # its result as SEPARATE spans sharing the id; pairing them here prevents
    # phantom duplicate tool_calls (inflated n_tool_calls, false retry cycles).
    # Older exports without ids fall back to per-span self-contained handling.
    by_call_id: dict[str, dict] = {}
    for rs in obj.get("resourceSpans", []):
        for scope in rs.get("scopeSpans", []):
            for span in scope.get("spans", []):
                a = _flatten_attrs(span.get("attributes"))
                cid = a.get("gen_ai.tool.call.id")
                if cid and (
                    a.get("gen_ai.tool.name") or "execute_tool" in span.get("name", "")
                ):
                    by_call_id.setdefault(cid, []).append((span, a))

    events: list[Event] = []
    for rs in obj.get("resourceSpans", []):
        for scope in rs.get("scopeSpans", []):
            for span in scope.get("spans", []):
                a = _flatten_attrs(span.get("attributes"))
                kind_flag = a.get("openinference.span.kind", "")
                name = span.get("name", "").strip()
                t0 = _ns_to_ms(span.get("startTimeUnixNano", 0))
                t1 = _ns_to_ms(span.get("endTimeUnixNano", 0))
                if openinference:
                    is_tool = kind_flag == "TOOL" or bool(
                        re.search("|".join(cfg.handoff_tool_patterns), name, re.I)
                    )
                    is_llm = kind_flag == "LLM"
                    is_agent = kind_flag in ("AGENT", "CHAIN")
                else:
                    is_tool = name in ("execute_tool",) or "gen_ai.tool.name" in a
                    is_llm = name == "llm"
                    is_agent = name in ("agent", "send_message", "chain")
                tool_name = a.get("gen_ai.tool.name")
                if tool_name is None and is_tool:
                    # OpenInference keeps the tool name inside llm.tool_calls JSON
                    tc = a.get("llm.tool_calls")
                    if tc:
                        m = re.search(r'"name"\s*:\s*"([^"]+)"', tc)
                        tool_name = m.group(1) if m else name
                if is_tool and tool_name:
                    cid = a.get("gen_ai.tool.call.id")
                    paired = by_call_id.get(cid, []) if cid else []
                    if len(paired) > 1:
                        # call/result live on separate spans sharing the id: the call
                        # span (no result attr) emits only tool_call; its result span
                        # carries the tool_result. Skip the result half here.
                        has_own_result = "gen_ai.tool.call.result" in a or bool(
                            a.get("output.value")
                        )
                        if not has_own_result:
                            events.append(
                                {
                                    "t_ms": t0,
                                    "kind": "tool_call",
                                    "name": tool_name,
                                    "attrs": a,
                                }
                            )
                            continue
                        # result half: skip its phantom tool_call duplicate
                        errored = "error.type" in a
                        if errored:
                            events.append(
                                {
                                    "t_ms": t1,
                                    "kind": "tool_error",
                                    "name": tool_name,
                                    "attrs": a,
                                }
                            )
                        else:
                            events.append(
                                {
                                    "t_ms": t1,
                                    "kind": "tool_result",
                                    "name": tool_name,
                                    "attrs": a,
                                }
                            )
                        continue
                    errored = "error.type" in a
                    if errored:
                        events.append(
                            {
                                "t_ms": t1,
                                "kind": "tool_error",
                                "name": tool_name,
                                "attrs": a,
                            }
                        )
                    events.append(
                        {"t_ms": t0, "kind": "tool_call", "name": tool_name, "attrs": a}
                    )
                    if not errored and (
                        "gen_ai.tool.call.result" in a or a.get("output.value")
                    ):
                        events.append(
                            {
                                "t_ms": t1,
                                "kind": "tool_result",
                                "name": tool_name,
                                "attrs": a,
                            }
                        )
                elif is_llm:
                    content = (
                        a.get("gen_ai.completion.0.content")
                        or a.get("output.value")
                        or ""
                    )
                    prompt = a.get("input.value") or ""
                    if prompt and not content:
                        events.append(
                            {
                                "t_ms": t0,
                                "kind": "user_msg",
                                "name": name,
                                "attrs": {"content": prompt},
                            }
                        )
                    if content:
                        events.append(
                            {
                                "t_ms": t1,
                                "kind": "assistant_msg",
                                "name": name,
                                "attrs": {"content": content},
                            }
                        )
                elif is_agent or name in ("send_message",):
                    # event-style spans: prompt.0 = opening assistant line; event.role/content = turns
                    if a.get("gen_ai.prompt.0.role") == "assistant" and a.get(
                        "gen_ai.prompt.0.content"
                    ):
                        events.append(
                            {
                                "t_ms": t0,
                                "kind": "assistant_msg",
                                "name": "greeting",
                                "attrs": {"content": a["gen_ai.prompt.0.content"]},
                            }
                        )
                    role = a.get("gen_ai.event.role")
                    content = a.get("gen_ai.event.content")
                    if role and content:
                        kind = "user_msg" if role == "user" else "assistant_msg"
                        events.append(
                            {
                                "t_ms": t0 + 1,
                                "kind": kind,
                                "name": name,
                                "attrs": {"content": content},
                            }
                        )
                    ev_name = a.get("gen_ai.event.name", "")
                    if ev_name in cfg.complete_event_names:
                        events.append(
                            {
                                "t_ms": t1,
                                "kind": "session_end",
                                "name": ev_name,
                                "attrs": {},
                            }
                        )
                    if ev_name in cfg.hangup_event_names:
                        events.append(
                            {"t_ms": t1, "kind": "hangup", "name": ev_name, "attrs": {}}
                        )
                    if ev_name in cfg.handoff_event_names:
                        events.append(
                            {
                                "t_ms": t1,
                                "kind": "handoff",
                                "name": ev_name,
                                "attrs": {},
                            }
                        )
                    # OpenInference AGENT root span carries the opening user input
                    if openinference and kind_flag == "AGENT" and a.get("input.value"):
                        events.append(
                            {
                                "t_ms": t0 - 1,
                                "kind": "user_msg",
                                "name": "root",
                                "attrs": {"content": a["input.value"]},
                            }
                        )
                # explicit terminal events by span name (both dialects)
                if re.search("|".join(cfg.hangup_event_names), name, re.I):
                    events.append(
                        {"t_ms": t1, "kind": "hangup", "name": name, "attrs": a}
                    )
                if re.search(
                    "|".join(cfg.handoff_tool_patterns), (tool_name or name), re.I
                ):
                    events.append(
                        {
                            "t_ms": t1,
                            "kind": "handoff",
                            "name": tool_name or name,
                            "attrs": a,
                        }
                    )
    events.sort(key=lambda e: e["t_ms"])
    return _dedupe(events)


def _normalize_langsmith(obj, cfg: ExtractionConfig) -> list[Event]:
    events: list[Event] = []
    for run in obj.get("runs", []):
        t0 = _iso_to_ms(run["start_time"])
        t1 = _iso_to_ms(run["end_time"])
        name = run.get("name", "")
        rtype = run.get("run_type", "")
        inputs = run.get("inputs") or {}
        outputs = run.get("outputs") or {}
        error = run.get("error")
        if rtype == "tool":
            if re.search("|".join(cfg.handoff_tool_patterns), name, re.I):
                events.append(
                    {"t_ms": t1, "kind": "handoff", "name": name, "attrs": dict(inputs)}
                )
                continue
            if error:
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "tool_error",
                        "name": name,
                        "attrs": {"error": error},
                    }
                )
            elif outputs:
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "tool_result",
                        "name": name,
                        "attrs": {"result": outputs},
                    }
                )
            else:
                events.append(
                    {
                        "t_ms": t0,
                        "kind": "tool_call",
                        "name": name,
                        "attrs": {"args": inputs},
                    }
                )
        elif rtype == "llm":
            text = (outputs or {}).get("text") or (outputs or {}).get("final") or ""
            if text:
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "assistant_msg",
                        "name": name,
                        "attrs": {"content": text},
                    }
                )
        elif rtype == "chain":
            event_key = str(inputs.get("event", ""))
            if (
                re.search("|".join(cfg.hangup_event_names), event_key, re.I)
                or "hangup" in name.lower()
            ):
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "hangup",
                        "name": event_key or name,
                        "attrs": dict(outputs),
                    }
                )
            elif re.search("|".join(cfg.handoff_event_names), event_key, re.I):
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "handoff",
                        "name": event_key or name,
                        "attrs": dict(outputs),
                    }
                )
            elif name.lower() in ("user_hangup",) or event_key == "caller_disconnected":
                events.append(
                    {
                        "t_ms": t1,
                        "kind": "hangup",
                        "name": event_key or name,
                        "attrs": dict(outputs),
                    }
                )
            # root chain may carry the opening user message
            msg = inputs.get("message")
            if msg and run.get("parent_run_id") is None:
                events.append(
                    {
                        "t_ms": t0,
                        "kind": "user_msg",
                        "name": "root",
                        "attrs": {"content": msg},
                    }
                )
    events.sort(key=lambda e: e["t_ms"])
    return _dedupe(events)


def _dedupe(events: list[Event]) -> list[Event]:
    """Collapse identical (t_ms, kind, name) events (span dual-emit protection)."""
    seen: set[tuple] = set()
    out = []
    for e in events:
        key = (e["t_ms"], e["kind"], e["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------- state inference


def infer_states(events: list[Event], config: ExtractionConfig | None = None) -> dict:
    """Rule-based classifier: canonical events -> grammar walk.

    Returns {walk, outcome, cycle_depth, error_count}. The walk is a list of
    {t_ms, state, transition_type} steps where state in PROGRESS_STATES + TERMINALS and
    transition_type is a typed transition of the grammar (or 'event' for non-mutating ones).
    """
    cfg = config or ExtractionConfig()
    walk: list[dict] = [
        {
            "t_ms": events[0]["t_ms"] if events else 0,
            "state": "q_greet",
            "transition_type": "start",
        }
    ]
    state = "q_greet"
    visited: set[str] = {"q_greet"}
    settled: set[str] = set()  # states exited via progress (for clarify re-open test)
    cycle_depth = 0
    error_count = 0
    pending_auth_call = False
    last_user_t = None
    outcome = None

    def emit(t_ms, st, ttype):
        nonlocal state
        walk.append({"t_ms": t_ms, "state": st, "transition_type": ttype})
        state = st
        visited.add(st)

    for ev in events:
        kind = ev["kind"]
        name = ev.get("name") or ""
        content = str((ev.get("attrs") or {}).get("content", ""))
        if kind == "user_msg":
            last_user_t = ev["t_ms"]
            if state == "q_greet":
                emit(ev["t_ms"], "q_intent", "user-answer")
            elif re.search("|".join(cfg.clarify_user_patterns), content, re.I):
                # user asking to repeat = re-open the current state if it was settled
                if state in settled:
                    emit(ev["t_ms"], "q_clarify", "agent-clarify")
                    cycle_depth += 1
                else:
                    emit(ev["t_ms"], "q_clarify", "agent-clarify")
            elif state == "q_clarify":
                emit(ev["t_ms"], "q_intent", "user-answer")
            elif state == "q_tool":
                # user answered a mid-tool question -> clarify resolution, back to intent
                emit(ev["t_ms"], "q_intent", "user-answer")
            elif state in ("q_intent", "q_confirm"):
                if (
                    re.search("|".join(cfg.confirm_user_patterns), content, re.I)
                    and state == "q_confirm"
                ):
                    emit(ev["t_ms"], "q_close", "user-answer")
                # otherwise stay: more intent detail
        elif kind == "assistant_msg":
            if re.search("|".join(cfg.close_assistant_patterns), content, re.I):
                emit(ev["t_ms"], "q_close", "user-answer")
            elif state == "q_clarify":
                emit(ev["t_ms"], "q_intent", "user-answer")
            elif state == "q_tool":
                # assistant answered after tool results: if tool errors happened (retry), the
                # recovery stays in q_tool (resolved iteration); else move to confirm.
                if cycle_depth == 0 or not any(
                    w["transition_type"] in ("tool-fail", "auth-fail") for w in walk
                ):
                    emit(ev["t_ms"], "q_confirm", "user-answer")
            # greeting content already counted at q_greet; other assistant msgs don't mutate state
        elif kind == "tool_call":
            if re.search("|".join(cfg.auth_tool_patterns), name, re.I):
                # same-name auth tool re-entered after a result = auth-fail retry (cycle).
                if state == "q_auth" and not pending_auth_call:
                    emit(ev["t_ms"], "q_auth", "auth-fail")
                    cycle_depth += 1
                elif state != "q_auth":
                    emit(ev["t_ms"], "q_auth", "user-answer")
                pending_auth_call = True
            else:
                if state in ("q_greet", "q_auth", "q_intent", "q_clarify"):
                    emit(ev["t_ms"], "q_tool", "tool-call")
                elif state == "q_tool":
                    pass  # iteration: progress self-loop, no new step
                else:
                    emit(ev["t_ms"], "q_tool", "tool-call")
        elif kind == "tool_result":
            # A result on an auth tool means the verification stage completed; whether it was a
            # failure (retry → cycle) is decided by the next auth tool_call with the same name.
            pending_auth_call = False
            # tool result inside q_tool stays (the next assistant/user event resolves it)
            # tool result inside q_tool stays (the next assistant/user event resolves it)
        elif kind == "tool_error":
            error_count += 1
            if state == "q_auth":
                emit(ev["t_ms"], "q_auth", "auth-fail")
                cycle_depth += 1
            else:
                emit(ev["t_ms"], "q_tool", "tool-fail")
                cycle_depth += 1
        elif kind == "handoff":
            emit(ev["t_ms"], "theta_hand", "escalation")
            outcome = "theta_hand"
        elif kind == "hangup":
            emit(ev["t_ms"], "theta_aban", "hangup")
            outcome = "theta_aban"
        elif kind == "session_end":
            emit(
                ev["t_ms"],
                "q_close" if state not in TERMINALS else state,
                "disclosure" if state == "q_confirm" else "service-complete",
            )
            if state != "theta_aban" and state != "theta_hand":
                outcome = "theta_cont"

    if outcome is None:
        # no explicit terminal event: decide from where the walk ended
        if state in ("q_close",):
            outcome = "theta_cont"
        elif (
            last_user_t is not None
            and walk
            and walk[-1]["t_ms"] - last_user_t > 2_000
            and state in ("q_auth", "q_tool", "q_intent", "q_clarify")
        ):
            outcome = "theta_aban"  # session ended right after user msg with no reply
        else:
            outcome = "theta_cont" if state == "q_close" else "theta_aban"
    return {
        "walk": walk,
        "outcome": outcome,
        "cycle_depth": cycle_depth,
        "error_count": error_count,
    }


def extract(obj, fmt: str, config: ExtractionConfig | None = None) -> dict:
    """One-call convenience: normalize + infer."""
    if config is None and fmt == "taubench":
        config = TAUBENCH_CONFIG
    cfg = config or ExtractionConfig()
    events = normalize_trace(obj, fmt, cfg)
    result = infer_states(events, cfg)
    result["events"] = events
    return result
