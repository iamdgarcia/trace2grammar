---
title: Conversation Autopsy
emoji: 🔪
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Paste a conversation, watch where and why it died
---

# 🔪 Conversation Autopsy

**▶ Try it live:** <https://iamdgarcia-conversation-autopsy.static.hf.space> ·
**Code on GitHub:** <https://github.com/iamdgarcia/trace2grammar>

Paste any conversation (support chat, voice-agent log, pasted WhatsApp thread) and a
decision walker dissects it live: every state decision lights up in order, and the
walker rules on the **cause of death**:

- **Contained** — the conversation reached a close with the goal attended.
- **Abandoned** — the user stopped answering while the agent still owed them something;
  the walker marks the exact line where they gave up.
- **Handoff** — the conversation escaped the machine to a human.

Everything runs in your browser: no backend, nothing uploaded. The walker is a
line-for-line JavaScript port of [`infer_states` in
trace2grammar/extract.py](https://github.com/iamdgarcia/trace2grammar), **parity-checked
against the Python implementation on 585 transcripts** (the repo's sample set plus
crafted edge cases: handoff tools, tool errors, continuation lines, no-terminal closes).

Related research: the walker is the same instrument that powers
[where-conversations-die](https://iamdgarcia-where-conversations-die.static.hf.space)
(the latent-space atlas of untested conversation behavior) and the coverage grammar in
the trace2trajectory paper.

Built by [iamdgarcia](https://huggingface.co/iamdgarcia) — realtime voice agent engineer.
