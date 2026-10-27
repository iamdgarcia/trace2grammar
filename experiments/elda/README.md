# ELDA — trajectory-dynamics property suite

Acceptance suite for a latent trajectory space over conversation windows, plus the
trajectory-dynamics readouts (goal alignment, critical turn, latent loops) used in the
paper's "From atlas to orbit" section (Sections `sec:orbit` / `sec:property_tests`).

Zero training: every representation block is frozen.

## Channels

- `V_sem` (semantic block): frozen embedding of a fixed per-window state schema annotated
  by an instruction-tuned LLM: `{phase, resolved_requests, pending_request, open_loops,
  user_stance}`. Declared models used in the paper: `gemini-embedding-001` over
  `gemini-2.5-flash-lite` annotations.
- `V_seq` (sequential block): the same schema's slots bound to their roles as role-bound
  symbolic coordinates (resolved-at-k / pending-at-k / open-loop), joined with one-hot
  phase/stance and typed decision scores from a decision-model judge.
- `composed(alpha)`: `[alpha * z(V_sem) || (1-alpha) * z(V_seq)]`, z = per-dimension
  standardization over the point cloud (goals included).

The goal is counterfactual and per-conversation: the resolved-terminal rendering of that
conversation's own requests (schema with `pending` swapped to `resolved`), so alignment is
task-respecting and needs no extra model.

## Property tests (the arbiter; no loss curve exists without training)

1. **Separation** — same-phase windows (openings) closer than cross-phase pairs
   (openings vs SGD finals). Report: distance AUROC (paper pilot: raw window embed 0.56 =
   phase-blind; schema embed 0.85; role-bound block 0.97; composed 0.96).
2. **Order** — two conversations composed from the same two requests in opposite orders
   (A->B vs B->A, built synthetically from Schema-Guided Dialogue first exchanges) must be
   far. Report: order-change distance / content-change distance between the same requests
   asked cold (paper pilot: 0.17-0.26 for embedding channels; 0.84-0.86 role-bound — below
   parity, the honest open edge).
3. **Cross-position** — asking B after A was resolved differs from asking B cold by as much
   as a same-structure content change (fair baseline: two A->B conversations from different
   request pairs). Paper pilot: 0.35 pure semantics (invisible), 0.95 role-bound.
4. **Divergence geometry** — within-conversation distance from the first window grows
   monotonically.

Measured design rule pinned by the suite: similarity lives in embeddings; order and
context-position live in role-bound symbolic structure. A single-vector encoding cannot
carry both.

## Usage

    # 1) annotate windows (calls the configured endpoints; needs env vars)
    export ELDA_EMBED_URL=...      # OpenAI-compatible /v1/embeddings base
    export ELDA_EMBED_KEY=...      # bearer token
    export ELDA_EMBED_MODEL=gemini-embedding-001
    export ELDA_DESC_MODEL=gemini-2.5-flash-lite
    python experiments/elda/elda_suite.py annotate --windows windows.jsonl --out annotations/

    # 2) run the property suite from the annotation cache
    python experiments/elda/elda_suite.py test --data experiments/elda/annotations/

`windows.jsonl` items: `{pid, corpus, cid, turn_idx, text}` where `text` is the sampled
window (current turn + up to 2 previous, role-prefixed lines). The committed annotation
caches under `annotations/` were produced from the paper's 50-conversation pilot
(30 Schema-Guided Dialogue test, 10 MultiWOZ 2.2 test, 10 tau-bench episodes; <= 8 sampled
windows per conversation). The ~22 MB vector caches are not committed; regenerate with the
`annotate` step or point `--data` at your own cache.

## Outputs

- property-test table (per channel: separation AUROC, order ratio, context-shift ratio,
  divergence monotonicity)
- trajectory-dynamics readouts per conversation: per-window cos(theta) to the
  counterfactual goal, critical turn t_crit (first sustained reversal), latent-loop
  candidates (same-conversation near-repeats, gap >= 2, distance below the 5th percentile
  of a cross-conversation null)

Paper numbers regenerate from the committed annotation caches plus the vector cache
regeneration step.
