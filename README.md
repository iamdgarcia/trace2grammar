# trace2grammar

**Trace → trajectory extractor**: convert production observability exports (OTel GenAI,
OpenInference, LangSmith) into **trajectory-grammar walks** — the bridge between production
observability and the trajectory-grammar formalism of *Turn-Level Trajectory Grammars for Voice
Agents* (D. García Peña).

The paper's claim is deliberately machine-level: every grammar state (`q_greet … q_close`, terminals
`θ_cont / θ_aban / θ_hand / θ_esc`) corresponds to events a production cascade already emits, so
grammaticality can be checked from telemetry alone. This library is the first, rule-based instance
of that claim: **real trace formats in, grammar walk + outcome class + cycle depth out.**

## What this repo probes

**The question:** when you evaluate a conversational agent with simulated users, which behaviors
does your evaluation actually exercise — and can you trust the outcome your transcript shows?

**The three instruments in this repo answer it:**

1. **Extraction (τ)** — deterministic mapping from any agent-observability format to a grammar
   walk + terminal outcome + cycle-depth drift `D`. Validated verbatim on three third-party
   corpora: MultiWOZ 2.2 (1,000 dialogues), Schema-Guided Dialogue (4,201 test dialogues, 20
   services), and 1,980 official tau-bench episodes of real agents (gpt-4o, claude-3.5-sonnet).
2. **Coverage** — which regions of the behavior plane your simulated users never reach: the
   tests you are not running. With the real tau-bench anchor: kernel-masked coverage stays 100%
   at every bandwidth of the ladder (101/130/243 cells), but kernel-free instance coverage is
   0.0% (of 961 unique real positions, no simulated episode lands on one). And 628 of 1,980 real
   episodes (31.7%) re-enter settled states (D up to 9) where every human wizard corpus shows D=0:
   real agents exercise the loop benchmarks never do.
3. **Verdict-vs-truth confrontation** — on the tau-bench anchor every episode carries the
   environment-verified reward: the closing phrase does not predict the delivered outcome
   (polite close in 66.8% of PASS episodes vs 76.4% of FAILED ones pooled; the inversion is
   compositional and does not survive model × length stratification, `h5_inversion_stratified.py`)
   — the deliverable claim is the *absence* of a reliable surface signal, not an inverted one.
   In SGD, the "confident closing, silent failure" mode appears as 9 terminal courteous-closes
   over failed service calls (of 198 silent-failure-candidate dialogs, 189 — 95.45% — recover
   via a later effective call; the walker misses 6 of the 9 terminal ones, `h7_d_crosstab.py`
   companion check on `dataset/sgd/dialogs.csv`), plus 4.6% of contained verdicts carrying the
   tool-ok/empty-result signal as candidates; in the real-agent anchor it is 21.7% of failures
   (173/797) — 8.7% of all runs. If you count outcomes from transcripts alone, you absorb this
   silently.
4. **Executed results, kill-or-confirm pass (25-oct + 30-oct honesty audit)** — (a) walk-vs-text
   ablation, DE-CONFOUNDED (`experiments/h4_confound_control.py`): once the text baseline sees ALL
   roles (the tool-result channel) and both arms get model/domain one-hots, the walk's add-on
   delta is +0.0099 (cluster-bootstrap 95% CI [-0.0142, +0.033], includes 0) — the walk CONDENSES
   structural regularity + the tool-error channel (D equals error_count in 1,980/1,980 tau
   episodes, `h7_d_crosstab.py`), it is not signal beyond text on this corpus. The earlier
   +0.020/+0.072 deltas carried a model one-hot in the walk arm. (b) turn-level risk curve:
   14,011 walker-inferred prefixes, deterministic; turns 1–3 chance (stated), turn 4 first
   signal (+13.4pp fixed-flag lift), first stable AUROC ≥ 0.65 at turn 7, best in-flight 0.82 at
   turn 11 (small cell, declared). **h10 SOTA push (31-oct, `h10_sota_curve.py` + `h10d_fusion.py`,
   outputs `runtime/h10_sota_curve/`):** adding the write channel + omission flag (failures are
   env-verified omissions, replay thread) moves the walk-only stable-0.65 detection from turn 7
   to turn 5 (paired delta vs the h3 anchor +0.046 mean AUC over turns 4–9, cluster-bootstrap
   95% CI [0.014, 0.080]); the text channel measured AT TURN LEVEL for the first time is the new
   practical SOTA — AUROC 0.72 from turn 1, 0.76 at turn 4, 0.80 at turn 8 (mean 4–9: 0.787) —
   and the walk adds no AUC beyond text in-flight either (fusion dAUROC −0.008, CI [−0.030,
   +0.011]): the walk's value is interpretability + format-agnosticism, not add-on accuracy.
   Causal receipt on fresh episodes (10 new flash-lite runs, `experiments/replay/h10e*` +
   `h10f*`): the omission flag fires in-flight on 2/7 failures and BOTH flip the environment
   reward 0→1 when repaired AT the flag turn (k*=8, k*=5, oracle = the env's own expected
   actions); one of them repairs at k* but NOT after the agent's wrong writes — the in-flight
   advantage made causal. The other 5 failures are not write-repairable (wrong answers on
   read-only tasks / strict checks) — typed, not hidden. **h11 decomposition
   (`h11_decompose.py`): the text curve is not just a task prior — within-task AUC
   (transcript evidence over knowing the task) is 0.6734@k1 pooled (shuffle null 0.49) and
   0.6409@k1 out-of-sample (test tasks only, 49 tasks), 0.67@k3-5; both failure families
   rank within task (divergent 0.68, silent 0.66 — the silent family is invisible to the
   divergence-POINT method of E3 but rankable by a trained model): the cross-task curve =
   task prior + real per-episode evidence from turn 1. (c) the polite-close inversion (66.8% of PASS vs 76.4% of
   FAILED episodes end politely) does NOT survive stratification: CMH p=0.37, per-cell deltas
   from +31.9pp (sonnet/airline) to -1.3pp (gpt4o/retail) — compositional, treat as descriptive
   (`h5_inversion_stratified.py`). (d) the sim-to-real gap itself is real: median sim→real NN
   distance is 1.6× the real→real NN scale (bootstrap CI [1.04, 1.69]; 21/60 sim points inside
   the real-median radius) — the honest restatement of kernel-free coverage 0.0%, which demanded
   exact float collision (`h6_coverage_nn.py`).
   **TraceGrammar-Anchor release:** `dataset/` (3 corpora, one schema, DATASET_CARD +
   MANIFEST.sha256, verify_dataset.py 45/45 PASS) + `experiments/` (the scripts that produce the
   numbers).

**Who consumes this:** the paper (*Turn-Level Trajectory Grammars for Voice Agents*), the three
live demos it powers (below), and anyone with agent traces — paste a transcript or load an OTel
export and get a verdict you can reproduce bit-for-bit (JS↔Python parity 645/645).

**What it is NOT:** it is not an LLM, not an NLU classifier, not a topic or intent model. It is
a deterministic state machine over conversation shape; it sees structure, not meaning. Every
number above has a source file in the repo (`runtime/` reports and `results*.json` fixtures).

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
(`autopsy_space/tau_port.js`, transcript parity 645/645 via `check_parity_transcript.py`)
powering three Hugging Face Spaces: [Conversation Autopsy](https://iamdgarcia-conversation-autopsy.static.hf.space)
(paste a conversation, watch where and why it died),
[Trajectory Grammar Atlas](https://iamdgarcia-trajectory-grammar-atlas.static.hf.space)
(a table of real conversations next to their 2D latent-space map, with the grammar of any
selection induced live from its walks — `grammar.js`, verified by `smoke_grammar.js` +
`check_parity_grammar.py`) and
[The Four Endings](https://iamdgarcia-the-four-endings.static.hf.space)
(chat with a scripted bot and speedrun the four endings — contained / abandoned / handoff /
escape — with the same walker ruling the verdict live; `four_endings_space/`, engine in
`game.js`, walker byte-identical to `autopsy_space/tau_port.js`) and
[Silent Failure Radio](https://iamdgarcia-silent-failure-radio.static.hf.space)
(a blind listening test: every track is the real final line of a real conversation; 33 failed closings,
33 delivered; the voice sounds identical on both sides, which is exactly the
*confident closing, silent failure* mode of section 5 made audible; `silent_failure_radio_space/`,
voice = KittenTTS in-browser, verdicts = published walker).

## Supported formats (and how to add one)

| Format | Import flag | Shape |
|---|---|---|
| **OTel GenAI semconv** | `fmt='otel'` | `resourceSpans[].scopeSpans[].spans[]` with `gen_ai.*` attributes |
| **OpenInference** | `fmt='openinference'` | same span envelope, `openinference.span.kind` (LLM/AGENT/TOOL/CHAIN), `input.value`/`output.value`, `llm.tool_calls` |
| **LangSmith API export** | `fmt='langsmith'` | `{runs: [{name, run_type, inputs, outputs, start_time, end_time, parent_run_id, error}]}` |
| **tau-bench episode** | `fmt='taubench'` | `{task_id, reward, info, traj[]}` (single or list) from `historical_trajectories/*.json` — environment-verified outcome drives the terminal (pass→contained, fail→handoff as documented public analogue) |
| **Plain transcript** | `fmt='transcript'` | pasted text: `assistant:`/`user:` lines, `tool(name): ok|error <msg>` — the one export every human owns |
| **pipecat-evals** | `fmt='pipecat-evals'` (alias `pipecat`) | pipecat's `EvalConversation` files (JSON list of `user`/`assistant`/`tool_call` messages, no results, no timing) and its simulation `results.jsonl` records (`messages` field; container keys tried in order: `messages`, `conversation`, `transcript`, `turns`, or `--key`). Checks: `tests/test_adapters.py` |

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
# one conversation + domain/metadata -> the embedding (walk + feature vector):
python embed_conversation.py conv.json --fmt pipecat-evals --domain banking --meta model=gpt-4o

# a whole pipecat evaluations run (results.jsonl): one embedding per line:
python embed_conversation.py run/results.jsonl --fmt pipecat-evals --domain banking

# fit the 2D behavior plane on a corpus, then project single conversations onto it:
python embed_conversation.py corpus/*.json --fmt transcript --save-basis basis.json
python embed_conversation.py conv.json --fmt pipecat-evals --basis basis.json

# programmatically:
from trace2grammar import to_grammar, fit_basis, project_features
emb = to_grammar(conversation, "pipecat-evals", meta={"domain": "banking"})
emb["2d"] = project_features(emb["features"], fit_basis(corpus_features))
```

The embedding is outcome-free by design: it is the trajectory representation
(walk, cycle depth, feature vector). The built-in check runs with
`python embed_conversation.py --self-check`; the adapter registry (23 checks
including the otel/langsmith fixtures and the pipecat containers) with
`python tests/test_adapters.py`.

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
