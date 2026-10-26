---
sdk: static
app_file: index.html
title: Silent Failure Radio
short_description: 📻 Blind-test the polite goodbyes of failed agent talks
emoji: 📻
colorFrom: gray
colorTo: red
---

# Silent Failure Radio 📻

**What you will learn here:** support agents that fail politely are invisible. QA reads the
last line, sees gratitude, logs a success. In **1,980 real evaluated conversations**, tasks
the environment marked FAILED ended in a polite close **more often** than tasks that passed
(76.4% vs 66.8%): the closing phrase is a blind proxy on real agent runs.

Every track is the **last thing an agent said** in one of those conversations. Some delivered;
some didn't, and the agent closed politely anyway. Listen, vote DELIVERED or FAILED, and flip
the card. **The voice sounds identical on both sides. That is the point.**

## What is on air

- **66 tracks** (33 failed / 33 delivered, seed-fixed sample), each one the final
  assistant line of a real conversation.
- **Failed (tau-bench)**: official eval runs with **reward 0.0** that still ended in a
  polite close — the walker's *contained* verdict on a task the environment marked
  failed. Retry depth **D >= 2** goes first: the early-warning band from the paper's
  sweep (61.2% failure at D >= 2 vs 40.6% held-out).
- **Failed (SGD)**: the agent **textually confessed** the failure
  ("Sorry, the reservation was not successful.") and still said goodbye nicely.
- **Delivered**: reward > 0 (tau) or effective service call (SGD), with the same
  polite closing shape.

## Why it matters

The closing phrase is what dashboards, logs and QA reviews actually see. On real agent
runs the proxy's sign **inverts**: PASSED episodes close politely 66.8% of the time,
FAILED episodes 76.4% (*closure != delivery*). This station makes the failure mode of
the trace2grammar paper — **confident closing, silent failure** — audible.

The verdicts come from the published walker (byte-identical to
[Conversation Autopsy](https://iamdgarcia-conversation-autopsy.static.hf.space/));
the behavior plane of the same episodes is
[Where Conversations Die](https://iamdgarcia-where-conversations-die.static.hf.space/).

## Voice

[KittenTTS](https://github.com/KittenML/KittenTTS) (nano-int8, Apache-2.0) runs
**locally in your browser** via ONNX Runtime Web (~26&nbsp;MB of assets, cached after
the first load; no text leaves the page). If the engine cannot start, the station
falls back to the browser system voice and says so — the data is the closing lines,
not the voice.

Attribution: KittenML KittenTTS (Apache-2.0); SGD (Dstc8) and tau-bench (MIT) corpora,
used per their licenses as in the trace2grammar artifact.
