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
transcript (paste) ┘
                                                                    │
                                                    report.py ──────┤
                                                        ├─ outcome class (θ_cont/θ_aban/θ_hand/θ_esc)
                                                        ├─ cycle depth D (regression count = drift)
                                                        ├─ per-state dwell times
                                                        ├─ ascii timeline
                                                        └─ atlas SVG: (cycle_depth, duration) plane,
                                                           color = outcome
```

**Live in a browser:** the walker ships as a verified JavaScript port
(`autopsy_space/tau_port.js`, transcript parity 585/585 via `check_parity_transcript.py`)
powering three Hugging Face Spaces: [Conversation Autopsy](https://iamdgarcia-conversation-autopsy.static.hf.space)
(paste a conversation, watch where and why it died),
[Where Conversations Die](https://iamdgarcia-where-conversations-die.static.hf.space)
(the latent-space atlas of untested behavior) and
[The Four Endings](https://iamdgarcia-the-four-endings.static.hf.space)
(chat with a scripted bot and speedrun the four endings — contained / abandoned / handoff /
escape — with the same walker ruling the verdict live; `four_endings_space/`, engine in
`game.js`, walker byte-identical to `autopsy_space/tau_port.js`).

## Supported formats (and how to add one)

| Format | Import flag | Shape |
|---|---|---|
| **OTel GenAI semconv** | `fmt='otel'` | `resourceSpans[].scopeSpans[].spans[]` with `gen_ai.*` attributes |
| **OpenInference** | `fmt='openinference'` | same span envelope, `openinference.span.kind` (LLM/AGENT/TOOL/CHAIN), `input.value`/`output.value`, `llm.tool_calls` |
| **LangSmith API export** | `fmt='langsmith'` | `{runs: [{name, run_type, inputs, outputs, start_time, end_time, parent_run_id, error}]}` |
| **tau-bench episode** | `fmt='taubench'` | `{task_id, reward, info, traj[]}` (single or list) from `historical_trajectories/*.json` — environment-verified outcome drives the terminal (pass→contained, fail→handoff as documented public analogue) |
| **Plain transcript** | `fmt='transcript'` | pasted text: `assistant:`/`user:` lines, `tool(name): ok|error <msg>` — the one export every human owns |

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

## Sample conversations (samples/)

**25 hand-written synthetic transcripts, English**, two verticals — customer support (Acme Bank)
and appointment booking (Northside Clinic) — so you can run the whole pipeline with zero
infrastructure. Outcomes are **not labeled in the files**: the walker decides them. The set:

- `cont01…cont08` — 8 contained conversations: straight through, clarify loops, an OTP retry
  (D=2), a dispute with two clarifies, a declined human-transfer offer
- `aban01…aban06` — 6 abandoned: silent mid-auth, silent after an offer, giving up after two
  clarify rounds, hanging up right after a tool error
- `hand01…hand05` — 5 handoffs: angry caller demands a human, out-of-scope vertical, three
  failed identity checks forcing escalation, refund above bot authority
- `sim01…sim06` — 6 **deliberately over-cooperative simulated** conversations (never err, never
  leave, never escalate): the collapse configuration of the atlas, shipped as the built-in sim set

```python
from trace2grammar.extract import extract
r = extract(open("samples/hand02_mortgage_out_of_scope.txt").read(), "transcript")
print(r["outcome"])   # theta_hand
```

`python render_samples_atlas.py` paints the full generated set (640 transcripts) into the 2-D
behavior plane (`samples_atlas.png` + `samples_coverage.json`). The sim composition is the
atlas's experiment: the 80-walk over-cooperative control alone covers 100/100 with 70-of-447
cells uncovered (the collapse-with-holes pattern §5 describes); the 60-walk **diverse sim2
batch** (composed through the same production-shaped generators as the real set, 20/20/20
families) lifts coverage to 100% of all occupied cells. Red cells on a coverage atlas measure
the sim set's imagination, not the world's: reachability is cheap when you own the generator.
`make_samples.py` regenerates the set (fixed seed) and prints the walker's summary table.

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

Same one-call shape for the other dialects — e.g. a tau-bench episode (single dict or list of
episodes from `historical_trajectories/*.json`); the environment-verified reward drives the
terminal class (pass → contained, fail → the handoff public analogue):

```python
obj = json.load(open("fixtures/taubench_episodes.json"))
r = extract(obj, "taubench")   # accepts one episode or a list
print(r["outcome"], "cycle_depth =", r["cycle_depth"])
# theta_hand cycle_depth = 1
```

## Region coverage — what are we NOT testing?

The headline use: given **real** production walks and a **simulated** (or candidate-agent) set,
which regions of conversational-behavior space does the sim set cover, and which does it miss?
Made measurable, per outcome class.

**Method** (deterministic, stdlib-only):
1. **Features** — each walk becomes a vector, two encodings: `grammar` (n_states, duration,
   cycle_depth, per-state visit counts, tool calls, errors, terminal one-hot) or `bigram`
   (first-K transition histogram); columns z-normalized (constant columns → 0, no NaN).
2. **Projection** — PCA (power iteration, hand-rolled) fit on the REAL set; sim points projected
   onto the same axes (production defines the space; sim must not move it).
3. **Density** — Gaussian-bump accumulation on a res×res grid over the real extent + smoothing
   (box blur), normalized by the real peak so both clouds share a scale.
4. **Coverage** — a cell is *occupied* when real density > eps·core; *covered* when a sim point
   falls inside it or the sim kernel puts ≥ eps·core mass there. Uncovered cells carry their
   nearest real exemplar + local density; `top_k_uncovered()` dedups neighbors and returns the
   highest-density exemplars: **the tests you are not running**.

**CLI** (real output from the shipped fixtures — 2 real conversations, 1 simulated):

```
$ python -m trace2grammar.coverage \
    --real fixtures/otel_genai_trace.json \
    --real fixtures/openinference_trace.json \
    --sim fixtures/langsmith_run.json --top 3
real n=2  sim n=1  features=grammar
grid 24x24  occupied 256 cells (eps=0.05, bw=0.12)
coverage: 40.2% overall | per class: contained 45.3%, handoff 35.2%
top-3 uncovered regions (the tests you are not running):
 1. cell (2,11) density 0.86 | nearest real: "No puedo entrar en mi banca online" [handoff, D=1, 19.0s]
 2. cell (21,12) density 0.86 | nearest real: "Hola, quiero consultar el saldo de mi cuenta" [contained, D=1, 11.5s]
 3. cell (2,14) density 0.54 | nearest real: "No puedo entrar en mi banca online" [handoff, D=1, 19.0s]
written: coverage_report.html, coverage_report.json
```

`coverage_report.html` renders the atlas: real = blue dots, sim = orange diamonds, uncovered
cells shaded red with nearest-exemplar tooltips.

**Honest scope:** this 2-D projection + grid density is the deterministic baseline (any learned
feature step can replace `conversation_features` without touching the coverage math). The
MiniLM-embedding variant, KDE-calibrated bandwidth, and voice-physics channels are future work.

## Relation to the research program

- **Paper**: *Turn-Level Trajectory Grammars for Voice Agents* — the grammar G = (Q, Σ, δ) this
  extractor targets; cycle depth as the drift statistic; the stopping rule that acts on partial runs.
- **Companion tools**: the [event-driven voice orchestrator](https://github.com/iamdgarcia/event-driven-voice-orchestrator)
  is the runtime whose event stream this library consumes; the conversation simulator (in progress)
  fills the same atlas with synthetic runs to expose sim-to-real gaps.
- Author: [iamdgarcia on Hugging Face](https://huggingface.co/iamdgarcia) ·
  writeup: [Watching your voice agent think](https://iamdgarcia.medium.com/watching-your-voice-agent-think-three-instruments-you-can-open-in-a-browser-tab-f7ed16ec2a2a)

Clean-room: no corporate references, zero dependencies beyond the Python stdlib.
