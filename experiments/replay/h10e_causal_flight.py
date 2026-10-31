"""h10e_causal_flight.py — causal receipt for the omission flag, IN FLIGHT.

Predeclared (knowledge_base/threads/t2g-sota/PLAN.md): fresh retail tasks {5,6,7},
flash-lite agent via laya, deterministic reset(task_index). For each FAILED
episode: (a) does close_no_write fire in-flight (earliest k*)? (b) derive the
missing write with a STRONG model via laya (oracle side only); (c) replay the
prefix at k* + the derived write -> env reward. P-CAUSAL-FLIGHT: >=2/3 flip.

Reusable machinery: g1_runner_smoke.py (runner) + g2_replay_v6.py (replay shape).
Laya credential via laya_common (env > secrets.json); tau-bench litellm user-sim
reads OPENAI_BASE_URL/OPENAI_API_KEY -> set from laya_common BEFORE env calls.
"""
import json
import os
import ssl
import sys
import time
import urllib.request

REPO = r"C:/Users/U77225D/Desktop/proyectos/hf"
sys.path.insert(0, os.path.join(REPO, ".agents/skills/platform-litellm/scripts"))
sys.path.insert(0, os.path.join(REPO, "builds/trace2grammar"))
sys.path.insert(0, os.path.join(REPO, "runtime/taubench_anchor"))
import laya_common as L  # noqa: E402

os.environ.setdefault("OPENAI_BASE_URL", L.BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", L.API_KEY)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
# Netskope TLS: httpx/litellm + urllib honor SSL_CERT_FILE; the repo bundle is the
# valid one (hf/.env carries a stale path that no longer exists)
os.environ["SSL_CERT_FILE"] = os.path.join(REPO, "cacert_1.clean.pem")

from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402
from tau_bench.envs.retail.tools import ALL_TOOLS as TOOLS  # noqa: E402

from trace2grammar.extract import extract, infer_states, TAUBENCH_CONFIG  # noqa: E402
from tb_to_events import _antml_markup, strip_injected_terminal  # noqa: E402

HERE = os.path.join(REPO, "builds/trace2grammar/experiments/replay")
AGENT_MODEL = "gemini-2.5-flash-lite"
ORACLE_MODEL = "claude-sonnet-4-6-sa"
BASE_URL = L.BASE_URL.rstrip("/")
API_KEY = L.API_KEY
SYS = ("You are a customer service agent. Resolve the user's request with the provided tools. "
       "When done, send a final message to the user summarizing the resolution.")

WRITES = ("modify_", "cancel_pending_order", "exchange_delivered_order_items",
          "return_delivered_order_items", "modify_user_address")


def chat(model, msgs, tools=None, temperature=0.2):
    body = {"model": model, "messages": msgs, "temperature": temperature}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + API_KEY,
                                          "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=240, context=ssl.create_default_context()) as r:
                return json.load(r)["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"laya HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
        except Exception:
            if a == 3:
                raise
            time.sleep(3 * (a + 1))


def run_episode(task_id, max_steps=20):
    env = get_env("retail", user_strategy="llm", user_model=AGENT_MODEL,
                  user_provider="openai", task_split="test", task_index=task_id)
    tools = [t.get_info() for t in TOOLS]
    obs0 = env.reset(task_index=task_id).observation  # determinism lesson (g2 v5)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": str(obs0)[:4000]}]
    done = False
    for step in range(max_steps):
        m = chat(AGENT_MODEL, msgs, tools)
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                msgs.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}")
                try:
                    resp = env.step(Action(name=fn, kwargs=args))
                    obs = resp.observation
                except Exception as e:
                    obs = f"ERROR: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(obs)[:3000]})
        else:
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
            resp = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": m.get("content") or ""}))
            obs, reward, done = resp.observation, resp.reward, resp.done
            if done:
                break
            if obs:
                msgs.append({"role": "user", "content": str(obs)[:3000]})
    try:
        reward = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "###STOP###"})).reward
    except Exception:
        reward = float(getattr(env, "reward", 0) or 0)
    return {"task_id": task_id, "steps": step + 1, "reward": float(reward or 0),
            "done": bool(done), "msgs": msgs}


def omission_ks(task_id, trial, msgs):
    """h10 features over prefixes; earliest k with close_no_write=1 (None if never)."""
    conv = []
    for m in msgs[1:]:  # skip system
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            conv.append({"role": "assistant", "content": _antml_markup(m["tool_calls"]), "name": None})
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": m.get("content"), "name": None})
    prefix, k, first_k = [], 0, None
    for m in conv:
        prefix.append(m)
        if m.get("role") != "user":
            continue
        k += 1
        try:
            ev = strip_injected_terminal(extract({"task_id": task_id, "reward": 0.0,
                                                  "info": {}, "traj": prefix}, "taubench")["events"])
            c = infer_states(ev, TAUBENCH_CONFIG)
        except Exception:
            continue
        names = [e.get("name") or "" for e in ev if e["kind"] == "tool_call"]
        writes = [n for n in names if any(n.startswith(p) or n == p for p in WRITES)]
        if "q_close" in {s["state"] for s in c["walk"]} and not writes and first_k is None:
            first_k = k
    return first_k


def derive_repair(env, msgs):
    """Strong-model derivation of the missing write(s) from task + env.data + traj."""
    task = env.task
    data = {"user": {k: v for k, v in env.data.get("users", {}).items()
                     if isinstance(v, dict)} if isinstance(env.data.get("users"), dict) else env.data.get("users"),
            "orders": env.data.get("orders")}
    traj = [f"{m.get('role')}: {str(m.get('content') or m.get('tool_calls') or '')[:220]}"
            for m in msgs[1:] if m.get("role") in ("user", "assistant", "tool")]
    prompt = (
        "You are auditing a FAILED customer-service episode. The environment reward is 0 because "
        "the agent OMITTED a required database write (or wrote it wrong). Derive the EXACT tool "
        "call(s) that would make the DB satisfy the task's write expectations.\n\n"
        f"TASK INSTRUCTION:\n{task.instruction}\n\n"
        f"ENV DB (users + orders): {json.dumps(data, ensure_ascii=False, default=str)[:6000]}\n\n"
        "EPISODE TRAJECTORY (role: content):\n" + "\n".join(traj)[:6000] + "\n\n"
        "Respond ONLY with JSON: {\"calls\": [{\"name\": tool_name, \"kwargs\": {...}}]}. "
        "Use only retail tools that MUTATE state (modify_*, cancel_pending_order, "
        "exchange_delivered_order_items, return_delivered_order_items). If the task expects no "
        "write, return {\"calls\": []}."
    )
    m = chat(ORACLE_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
    txt = (m.get("content") or "").strip()
    i = txt.find("{")
    if i < 0:
        return []
    try:  # raw_decode: first valid JSON object, tolerant of trailing prose
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
        return obj.get("calls", [])
    except Exception:
        return []


def replay_at(env, task_id, msgs, k_star, extra_calls):
    """Replay tool calls before the k_star-th user message, then extra writes."""
    env.reset(task_index=task_id)
    n_user = 0
    for m in msgs[1:]:
        role = m.get("role")
        if role == "user":
            n_user += 1
            if n_user > k_star:
                break
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    env.step(Action(name=tc["function"]["name"],
                                    kwargs=json.loads(tc["function"]["arguments"] or "{}")))
                except Exception:
                    pass
    for c in extra_calls:
        try:
            env.step(Action(name=c["name"], kwargs=c["kwargs"]))
        except Exception:
            pass
    try:
        env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Done."}))
    except Exception:
        pass
    return float(env.calculate_reward().reward or 0)


def main():
    out_path = os.path.join(HERE, "h10e_results.json")
    traces_path = os.path.join(HERE, "h10e_traces.json")
    TASKS = list(range(5, 15))  # fresh batch: predicate needs >=2 flagged failures
    results = {"protocol": {"predicate": "P-CAUSAL-FLIGHT: >=2/3 flagged omission failures flip "
                                        "reward at the flag turn k* (prefix replay + derived write)",
                            "not_evaluable_if": "fewer than 2 flagged failures in the batch",
                            "tasks": TASKS, "agent": AGENT_MODEL, "oracle": ORACLE_MODEL,
                            "predeclared": "knowledge_base/threads/t2g-sota/PLAN.md"},
               "episodes": []}
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)  # incremental save

    traces = {}
    if os.path.exists(traces_path):
        traces = json.load(open(traces_path, encoding="utf-8"))
    for tid in TASKS:
        t0 = time.time()
        ep = None
        for attempt in (1, 2):
            try:
                ep = run_episode(tid)
                break
            except Exception as e:
                print(f"task {tid} attempt {attempt} failed: {str(e)[:150]}", flush=True)
        if ep is None:
            results["episodes"].append({"task_id": tid, "error": "runner failed twice"})
            json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)
            continue
        ep["secs"] = round(time.time() - t0, 1)
        traces[str(tid)] = ep["msgs"]
        json.dump(traces, open(traces_path, "w", encoding="utf-8"))
        rec = {"task_id": tid, "reward": ep["reward"], "done": ep["done"], "secs": ep["secs"]}
        print(f"task {tid}: reward={ep['reward']} done={ep['done']} ({ep['secs']}s)", flush=True)
        if ep["reward"] == 0.0:
            k_star = omission_ks(tid, 0, ep["msgs"])
            rec["k_star_omission_flag"] = k_star
            env2 = get_env("retail", user_strategy="llm", user_model=AGENT_MODEL,
                           user_provider="openai", task_split="test", task_index=tid)
            env2.reset(task_index=tid)
            calls = derive_repair(env2, ep["msgs"])
            rec["derived_calls"] = calls
            results.setdefault("families", {})
            if k_star is not None and calls:
                rec["reward_at_k_star_after_repair"] = replay_at(env2, tid, ep["msgs"], k_star, calls)
                rec["flipped"] = rec["reward_at_k_star_after_repair"] == 1.0
                results["families"]["omission-flagged"] = results["families"].get("omission-flagged", 0) + 1
            elif k_star is None:
                rec["family"] = "no-omission-flag (wrong-write or read-only; typed answer)"
                results["families"]["no-flag"] = results["families"].get("no-flag", 0) + 1
                # exploratory receipt (NOT the predeclared predicate): repair the same
                # failure from the FULL trajectory (g2 v6 shape) and report the flip
                if calls:
                    rec["reward_full_replay_after_repair"] = replay_at(env2, tid, ep["msgs"], 10 ** 6, calls)
                    rec["flipped_full"] = rec["reward_full_replay_after_repair"] == 1.0
            else:
                rec["family"] = "omission flagged but derivation empty"
                results["families"]["flag-no-derive"] = results["families"].get("flag-no-derive", 0) + 1
        results["episodes"].append(rec)
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)

    flagged = [r for r in results["episodes"] if r.get("k_star_omission_flag") is not None]
    flips = sum(1 for r in flagged if r.get("flipped"))
    if len(flagged) < 2:
        verdict = "NOT-EVALUABLE (n<2 flagged failures in batch; typed answer below)"
    else:
        verdict = "PASS" if flips >= max(1, (2 * len(flagged) + 2) // 3) else "FAIL"
    results["P-CAUSAL-FLIGHT"] = {"flagged": len(flagged), "flips": flips, "verdict": verdict}
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"\nP-CAUSAL-FLIGHT: {flips}/{len(flagged)} flagged-omission repairs flip env reward -> {verdict}")
    print("families:", json.dumps(results.get("families", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
