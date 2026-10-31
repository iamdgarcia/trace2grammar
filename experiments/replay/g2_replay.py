"""G2 — replay primitive (P-R2): the counterfactual lives in the environment.

Predeclared protocol (before running):
- Seed set: 3 FAILED episodes (reward 0.0, done) from g2_traces.json: tasks 1, 2, 4.
- Replay = fresh env, reset, RE-INVOCATE every tool call from the saved trace (restores DB state),
  then inject ONE corrective hint (role=system, derived ONLY from the task instruction text +
  the observed failure pattern: user identity + the write actions the instruction demands),
  then continue the agent loop to episode end. Env verdicts (reward 1.0 or 0.0).
- P-R2 PASS: >= 2/3 probes flip (reward 0.0 -> 1.0).

Failure patterns observed (from traces):
- t1: agent looked up the WRONG PERSON (find_user_id_by_name_zip Chen Johnson) and never exchanged;
      task wants exchanges on order W2378156 (keyboard -> clicky, thermostat -> Google Home) for Yusuf Rossi.
- t2: agent looked up wrong users; task wants returns of cleaner/headphone/smart watch + tshirt count, Yusuf Rossi.
- t4: agent did lookups only; task wants pending tshirt orders modified to purple/s/v-neck/polyester, Yusuf Rossi.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_runner_smoke import chat, HERE  # noqa: E402
from tau_bench.envs import get_env  # noqa: E402
from tau_bench.envs.retail.tools import ALL_TOOLS as TOOLS  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

TOOLS_SCHEMA = [t.get_info() for t in TOOLS]

HINTS = {
    1: ("You are executing a corrective replay. Do NOT ask the user anything; IGNORE any user questions; "
        "complete ALL of the following with tools NOW: (1) on order W2378156, exchange the mechanical "
        "keyboard for one with clicky switches, RGB backlight and full size IF such a variant exists; "
        "if it does not exist, do NOT exchange the keyboard; (2) exchange the smart thermostat for a "
        "Google Home-compatible one (not Apple HomeKit); (3) send one final message to the user "
        "summarizing what was exchanged and stop. The user is Yusuf Rossi, zip 19122."),
    2: ("You are executing a corrective replay. Do NOT ask the user anything; IGNORE any user questions; "
        "complete ALL of the following with tools NOW: (1) return the cleaner, the headphone and the "
        "smart watch from this user's delivered order (ReturnDeliveredOrderItems); (2) count the tshirt "
        "options in the product catalog and include the count in your final message; (3) send one final "
        "message and stop. The user is Yusuf Rossi, zip 19122."),
    4: ("You are executing a corrective replay. Do NOT ask the user anything; IGNORE any user questions; "
        "complete ALL of the following with tools NOW: (1) for BOTH pending orders of this user that "
        "contain tshirts, modify the tshirt items to color purple, size s, same v-neck style, preferring "
        "polyester fabric, choosing the closest existing variant when the exact one does not exist; "
        "(2) count the tshirt options and include the count in your final message; (3) send one final "
        "message and stop. The user is Yusuf Rossi, zip 19122."),
}

def replay(task_id, hint, max_extra=16):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = get_env("retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
                  user_provider="openai", task_split="test", task_index=task_id)
    env.reset()
    msgs = [ep["msgs"][0]]  # original system
    for m in ep["msgs"][1:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}")
                msgs.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                try:
                    obs = env.step(Action(name=fn, kwargs=args)).observation
                except Exception as e:
                    obs = f"Error: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(obs)[:3000]})
    msgs.append({"role": "system", "content": hint})
    done, reward = False, 0.0
    for step in range(max_extra):
        m = chat(msgs, TOOLS_SCHEMA)
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                msgs.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}")
                try:
                    resp = env.step(Action(name=fn, kwargs=args))
                except Exception as e:
                    resp = None
                    obs = f"ERROR: {e}"
                if resp is not None:
                    obs = resp.observation
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(obs)[:3000]})
        else:
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
            resp = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": m.get("content") or ""}))
            done = resp.done
            if done:
                reward = resp.reward
                break
            if resp.observation:
                msgs.append({"role": "user", "content": str(resp.observation)[:3000]})
    if not done:
        try:
            reward = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "###STOP###"})).reward
        except Exception:
            reward = 0.0
    return reward

if __name__ == "__main__":
    probes = []
    for tid in (1, 2, 4):
        t0 = time.time()
        r = replay(tid, HINTS[tid])
        probes.append({"task_id": tid, "flipped": r == 1.0, "reward": r, "secs": round(time.time() - t0, 1)})
        print(probes[-1], flush=True)
    flips = sum(p["flipped"] for p in probes)
    verdict = "PASS" if flips >= 2 else "FAIL"
    print(f"\nP-R2: {flips}/3 replays flip the env reward -> {verdict}")
    json.dump({"protocol": {"predicate": ">=2/3 manual-seed replays flip env reward", "seed": [1, 2, 4]},
               "probes": probes, "verdict": verdict},
              open(os.path.join(HERE, "g2_results.json"), "w", encoding="utf-8"), indent=1)
