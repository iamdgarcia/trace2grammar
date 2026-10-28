---
sdk: static
app_file: index.html
title: Trajectory Grammar Atlas
short_description: The latent grammar space of 1,980 real agent conversations
emoji: 🗺️
colorFrom: gray
colorTo: blue
---

# Trajectory Grammar Atlas 🗺️

**▶ Try it live:** https://iamdgarcia-where-conversations-die.static.hf.space

**Code on GitHub:** https://github.com/iamdgarcia/trace2grammar

## What this Space shows

The [trajectory-grammars paper](https://github.com/iamdgarcia/trace2grammar) proposes one
thing above all: a **latent representation of conversations**. A trajectory grammar
G = (Q, Σ, δ) fixes a state space of 7 progress states (greet, auth, intent, tool,
clarify, confirm, close) and 4 terminal outcome classes (contained, abandoned, handoff,
compliance-escape). Every conversation — real or synthetic — becomes a **walk** through
those states, and every walk is a point in a latent space you can look at.

This Space is that space, on a public dataset: the **1,980 official tau-bench evaluation
runs** (sierra-research/tau-bench, historical trajectories: gpt-4o 660, claude-3.5-sonnet
1,320; airline + retail), extracted to grammar walks with the verbatim mapping shipped in
the repo. A table of conversations sits next to the 2D map (PCA of walk features). And
because the representation is the point, **the grammar is not drawn by hand**: pick any
subset of conversations — filters, or drag a box on the map — and the state machine below
is re-induced from those walks alone, edge by edge, with counts.

What falls out of the representation on this dataset:

- 30,556 state transitions collapse into just **24 distinct grammar edges**; the six
  heaviest edges carry **76% of all transition mass**. Conversational flow is heavily
  structured in this space.
- The tool–confirm loop dominates: `q_tool → q_confirm` (9,310 transitions) and
  `q_confirm → q_tool` (6,324) are the workhorse edges of real agent service.
- **q_close is not outcome-exclusive**: 1,021 of 1,602 abandoned runs (63.7%) still walk
  through the close state; every contained run does. Vocabulary does not decide outcome;
  structure does.
- 628 runs (31.7%) contain at least one backward edge (cycle depth D up to 9), while
  human-wizard corpora in the study show none — a structural property this representation
  makes countable per run.

## Provenance

Data: tau-bench `historical_trajectories` (public, sierra-research/tau-bench). Walks,
verdicts and the PCA plane are computed by
[iamdgarcia/trace2grammar](https://github.com/iamdgarcia/trace2grammar) (`build_space_blob_taubench.py`)
and embedded as static data. The induced-grammar derivation (`grammar.js`) is verified
two ways: `smoke_grammar.js` (node, golden numbers) and `check_parity_grammar.py`
(python re-derivation agrees edge-for-edge). Transcripts are lazy-loaded per conversation
(`tb_transcripts/`). The page runs offline: no backend, nothing is measured live, no
number is typed by hand.

Part of the trajectory-grammar line:
[the-four-endings](https://huggingface.co/spaces/iamdgarcia/the-four-endings) (live) ·
[conversation-autopsy](https://huggingface.co/spaces/iamdgarcia/conversation-autopsy)
(paste your own).
