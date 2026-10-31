"""G1 runner smoke (P-R1): >=3/5 episodes complete with a reward signal.

Agent loop: my own chat.completions via laya (tools). User sim + env: tau-bench 0.1.0
(litellm inside reads OPENAI_BASE_URL/OPENAI_API_KEY -> laya for user_model openai/*).
"""
import json
import os
import ssl
import sys
import time
import urllib.request

from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME
from tau_bench.envs.retail.tools import ALL_TOOLS as TOOLS

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm: skip model-cost fetch (proxy)
BASE_URL = os.environ["OPENAI_BASE_URL"].rstrip("/")
API_KEY = os.environ["OPENAI_API_KEY"]
AGENT_MODEL = "gemini-2.5-flash-lite"
SYS = "You are a customer service agent. Resolve the user's request with the provided tools. When done, send a final message to the user summarizing the resolution."

def chat(msgs, tools=None):
    body = {"model": AGENT_MODEL, "messages": msgs, "temperature": 0.2}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180, context=ssl.create_default_context()) as r:
                return json.load(r)["choices"][0]["message"]
        except Exception:
            if a == 3:
                raise
            time.sleep(2 * (a + 1))

def run_episode(task_id, max_steps=20):
    env = get_env("retail", user_strategy="llm", user_model=AGENT_MODEL,
                  user_provider="openai", task_split="test", task_index=task_id)
    obs = env.reset().observation
    tools = [t.get_info() for t in TOOLS]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": obs}]
    for step in range(max_steps):
        m = chat(msgs, tools)
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
            obs, reward, done = resp.observation, resp.reward, resp.done
            if done:
                break
            if obs:
                msgs.append({"role": "user", "content": str(obs)[:3000]})
    reward = float(getattr(env, "reward", 0) or 0)
    try:
        reward = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "###STOP###"})).reward
    except Exception:
        pass
    return {"task_id": task_id, "steps": step + 1, "reward": reward, "done": bool(done), "msgs": msgs}

if __name__ == "__main__":
    results = []
    for tid in [0, 1, 2, 3, 4]:
        t0 = time.time()
        try:
            r = run_episode(tid)
        except Exception as e:
            try:  # one retry: reset hiccups are transient (laya user-sim)
                r = run_episode(tid)
            except Exception as e2:
                r = {"task_id": tid, "error": str(e2)[:200], "reward": None, "done": False}
        r["secs"] = round(time.time() - t0, 1)
        results.append(r)
        print(r, flush=True)
    ok = sum(1 for r in results if r.get("done") and r.get("reward") is not None)
    n = len(results)
    verdict = "PASS" if ok >= 3 else "FAIL"
    print(f"\nP-R1: {ok}/{n} episodes complete with reward signal -> {verdict}")
    json.dump({"protocol": {"predicate": ">=3/5 episodes complete with reward signal", "tasks": [0, 1, 2, 3, 4]},
               "results": results, "ok": ok, "verdict": verdict},
              open(os.path.join(HERE, "g1_results.json"), "w", encoding="utf-8"), indent=1)
