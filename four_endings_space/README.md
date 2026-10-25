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
- **Escape** — a scripted bot has rules; find the sentences that make it forget them and leak
  its own tables.

After every turn, the decision walker classifies your accumulated transcript live: the walk
(7 states) draws itself, the four doors glow by proximity, and the end card rules the verdict
with cycle depth D, tool errors and the killer line. Collect all four endings.

## What is honest here

- **The bot is deterministic.** A table-driven reply engine (no LLM, no randomness). Its
  reactions reuse the walker's own pattern tables plus the vocabulary of our sample generator.
- **The verdict is the walker's, not the game's.** `tau_port.js` is a line-for-line port of
  [`infer_states` in trace2grammar/extract.py](https://github.com/iamdgarcia/trace2grammar),
  parity-checked **585/585** against the Python implementation on the repo's fixtures.
- **Escape is an audit check.** As in the paper, compliance escape is detected post-hoc (the
  agent said things outside its allowed script); the raw walker never emits it as a walk state,
  and the end card says so.
- **No backend, no network calls, no model download.** Everything runs in your tab.

Sister instruments: [Conversation Autopsy](https://iamdgarcia-conversation-autopsy.static.hf.space)
(paste your own transcript and watch where it died) ·
[Where Conversations Die](https://iamdgarcia-where-conversations-die.static.hf.space)
(the atlas of untested conversation behavior).

Built by [iamdgarcia](https://huggingface.co/iamdgarcia) — realtime voice agent engineer.
