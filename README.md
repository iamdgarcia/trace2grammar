# trace2grammar

**Trace → trajectory extractor**: convert production observability exports (OTel GenAI,
OpenInference, LangSmith) into **trajectory-grammar walks** — the bridge between production
observability and the trajectory-grammar formalism of *Turn-Level Trajectory Grammars for Voice
Agents* (D. García Peña).

The paper's claim is deliberately machine-level: every grammar state (`q_greet … q_close`, terminals
`θ_cont / θ_aban / θ_hand / θ_esc`) corresponds to events a production cascade already emits, so
grammaticality can be checked from telemetry alone. This library is the first, rule-based instance
of that claim: **real trace formats in, grammar walk + outcome class + cycle depth out.**

## What it does

```
OTel GenAI ─┐
OpenInference ─┼─► normalize_trace() ─► canonical events ─► infer_states() ─► walk σ
LangSmith ─┘        (attribute mapping)   {t_ms, kind, name}    (rule baseline)   {state, transition_type}
                                                                    │
                                                    report.py ──────┤
                                                        ├─ outcome class (θ_cont/θ_aban/θ_hand/θ_esc)
                                                        ├─ cycle depth D (regression count = drift)
                                                        ├─ per-state dwell times
                                                        ├─ ascii timeline
                                                        └─ atlas SVG: (cycle_depth, duration) plane,
                                                           color = outcome
```

## Supported formats (and how to add one)

| Format | Import flag | Shape |
|---|---|---|
| **OTel GenAI semconv** | `fmt='otel'` | `resourceSpans[].scopeSpans[].spans[]` with `gen_ai.*` attributes |
| **OpenInference** | `fmt='openinference'` | same span envelope, `openinference.span.kind` (LLM/AGENT/TOOL/CHAIN), `input.value`/`output.value`, `llm.tool_calls` |
| **LangSmith API export** | `fmt='langsmith'` | `{runs: [{name, run_type, inputs, outputs, start_time, end_time, parent_run_id, error}]}` |

To add a format: write one `_normalize_<fmt>(obj, cfg) -> list[Event]` function producing the
canonical stream (`{t_ms, kind, name, attrs}` with `kind ∈ user_msg | assistant_msg | tool_call |
tool_result | tool_error | hangup | handoff | session_end`) and register it in `normalize_trace()`.
Everything downstream is format-agnostic.

## The state-inference config

`ExtractionConfig` exposes every pattern the classifier uses — **adapt per domain without touching
code**:

- `auth_tool_patterns` — tool-name keywords that mean "identity/authentication" (`verify`, `auth`, `otp`…)
- `handoff_tool_patterns` — tool names that mean "escalate to human" (`transfer`, `escalate`…)
- `clarify_user_patterns` / `confirm_user_patterns` — regexes over the user's words
- `close_assistant_patterns` — farewell phrases
- `complete_event_names` / `hangup_event_names` / `handoff_event_names` — explicit terminal events

## Honest status: this is a heuristic baseline

The classifier is **rule-based**: keyword patterns + sequencing rules. It will misclassify edge
cases (mixed-order events, multi-lingual politeness, out-of-band hangups). That is deliberate — it
is the seed of the paper's *"the extraction layer is measurable"* contribution and the ground truth
generator for training the learned emission mapper later, not the final model. When the learned
mapper lands, `infer_states()` is the function it replaces; the interface does not change.

## Quick start

```bash
python tests/test_extract.py        # ALL CHECKS PASSED
# then open atlas_demo.html in a browser
```

```python
import json
from trace2grammar.extract import extract

obj = json.load(open("fixtures/otel_genai_trace.json"))
r = extract(obj, "otel")
print(r["outcome"], "cycle_depth =", r["cycle_depth"])
# theta_cont cycle_depth = 1
```

## Relation to the research program

- **Paper**: *Turn-Level Trajectory Grammars for Voice Agents* — the grammar G = (Q, Σ, δ) this
  extractor targets; cycle depth as the drift statistic; the stopping rule that acts on partial runs.
- **Companion tools**: the [event-driven voice orchestrator](https://github.com/iamdgarcia/event-driven-voice-orchestrator)
  is the runtime whose event stream this library consumes; the conversation simulator (in progress)
  fills the same atlas with synthetic runs to expose sim-to-real gaps.
- Author: [iamdgarcia on Hugging Face](https://huggingface.co/iamdgarcia) ·
  writeup: [Watching your voice agent think](https://iamdgarcia.medium.com/watching-your-voice-agent-think-three-instruments-you-can-open-in-a-browser-tab-f7ed16ec2a2a)

Clean-room: no corporate references, zero dependencies beyond the Python stdlib.
