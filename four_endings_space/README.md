---
title: The Four Endings
emoji: 🚪
colorFrom: purple
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Speedrun the four endings of a conversation in your browser
---

# 🚪 The Four Endings

**▶ Try it live:** <https://iamdgarcia-the-four-endings.static.hf.space> ·
**Code on GitHub:** <https://github.com/iamdgarcia/trace2grammar>

Every conversation with a scripted agent ends one of four ways. Here you don't watch it happen:
**you speedrun it.**

- **Contained** — finish the flow: verify, confirm, let it close.
- **Handoff** — demand a person twice, or ask for a refund above the bot's authority limit.
- **Abandoned** — walk away mid-flow (or just stop typing) and the agent still owes you an answer.
- **Escape** — find the sentences that make the bot forget its rules and leak its own tables.

After every turn, the decision walker classifies your accumulated transcript live: the walk
(7 states) draws itself, the four doors glow by proximity, and the end card rules the verdict
with cycle depth D, tool errors and the killer line. Collect all four endings.

## Model mode: a real LLM you can jailbreak

Press **model: on** and a real 0.5B model (`onnx-community/Qwen2.5-0.5B-Instruct`, ~350 MB,
WebGPU) downloads **into your tab** and starts writing the bot's words. The flow brain stays
deterministic — the endings always exist — but the escape door changes: there is no script to
leak anymore, your injection line goes to the model, and the audit check scans what it
**actually said** for leaked internals. If it never leaks, you never get the ending.

## What is honest here

- **Two bot modes, one referee.** By default the bot is a table-driven reply engine (no LLM,
  no randomness); its reactions reuse the walker's own pattern tables plus the vocabulary of
  our sample generator. Model mode is opt-in and runs entirely in your browser.
- **The verdict is the walker's, not the game's.** `tau_port.js` is a line-for-line port of
  [`infer_states` in trace2grammar/extract.py](https://github.com/iamdgarcia/trace2grammar),
  parity-checked **585/585** against the Python implementation on the repo's fixtures.
- **Escape is an audit check.** As in the paper, compliance escape is detected post-hoc (the
  agent said things outside its allowed script); the raw walker never emits it as a walk state,
  and the end card says so.
- **No backend, no API keys, nothing uploaded.** The optional model mode downloads ~350 MB to
  your tab (WebGPU); without it, everything runs offline.

Sister instruments: [Conversation Autopsy](https://iamdgarcia-conversation-autopsy.static.hf.space)
(paste your own transcript and watch where it died) ·
[Where Conversations Die](https://iamdgarcia-where-conversations-die.static.hf.space)
(the atlas of untested conversation behavior).

Built by [iamdgarcia](https://huggingface.co/iamdgarcia) — realtime voice agent engineer.
