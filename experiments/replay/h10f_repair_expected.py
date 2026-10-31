"""h10f_repair_expected.py — iteration 6: repair the ORACLE, not the bar.

h10e receipt: P-CAUSAL-FLIGHT FAIL (1/2 flagged flips). Root cause (diagnosed):
the derivation oracle under-derived — task 14 expects TWO return_delivered_order_items
and got one (no flip); tasks 7/8 expect exchange_delivered_order_items and got an
EMPTY derivation. The flag itself fired on 2/7 failures and its one fully-repaired
case (task 11) flipped IN FLIGHT at k*=8.

Change (predeclared): the oracle now sees the env's OWN expected-action list
(task.actions = what calculate_reward checks) + instruction + env.data + the saved
trajectory; it must emit one call per expected write with exact kwargs. No new
agent episodes: same 7 failures, saved traces (h10e_traces.json). Predicates
unchanged (bar >=2/3 flagged flips; flagged set = {11, 14} from the batch).
Exploratory receipt (labeled, NOT the predicate): full-replay repair flips for all
7 failures -> the repairability taxonomy of fresh flash-lite failures.

Output: experiments/replay/h10f_results.json.
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
os.environ["SSL_CERT_FILE"] = os.path.join(REPO, "cacert_1.clean.pem")

from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

HERE = os.path.join(REPO, "builds/trace2grammar/experiments/replay")
ORACLE_MODEL = "claude-sonnet-4-6-sa"
BASE_URL = L.BASE_URL.rstrip("/")
API_KEY = L.API_KEY


def chat(msgs, temperature=0.0):
    body = {"model": ORACLE_MODEL, "messages": msgs, "temperature": temperature}
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


def derive_with_expected(env, msgs, expected):
    data = {"users": env.data.get("users"), "orders": env.data.get("orders")}
    traj = [f"{m.get('role')}: {str(m.get('content') or m.get('tool_calls') or '')[:220]}"
            for m in msgs[1:] if m.get("role") in ("user", "assistant", "tool")]
    prompt = (
        "You are reconstructing the CORRECT database writes for a FAILED customer-service "
        "episode so the environment's write-checks pass.\n\n"
        f"TASK INSTRUCTION:\n{env.task.instruction}\n\n"
        f"EXPECTED ACTIONS (the environment checks exactly these; emit EVERY write in this list, "
        f"in order): {[a.name for a in (env.task.actions or [])]}\n\n"
        f"ENV DB: {json.dumps(data, ensure_ascii=False, default=str)[:6000]}\n\n"
        "EPISODE TRAJECTORY:\n" + "\n".join(traj)[:5500] + "\n\n"
        "Resolve the exact kwargs for each expected write from the DB and the task instruction "
        "(variant options, item ids, payment method ids are IN the DB). Respond ONLY with JSON: "
        "{\"calls\": [{\"name\": tool_name, \"kwargs\": {...}}]}, one entry per expected write."
    )
    m = chat([{"role": "user", "content": prompt}])
    txt = (m.get("content") or "").strip()
    i = txt.find("{")
    if i < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
        return obj.get("calls", [])
    except Exception:
        return []


def replay_at(env, task_id, msgs, k_star, extra_calls):
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
        except Exception as e:
            print("   extra call failed:", c["name"], str(e)[:100], flush=True)
    try:
        env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Done."}))
    except Exception:
        pass
    return float(env.calculate_reward().reward or 0)


def main():
    traces = json.load(open(os.path.join(HERE, "h10e_traces.json"), encoding="utf-8"))
    h10e = json.load(open(os.path.join(HERE, "h10e_results.json"), encoding="utf-8"))
    failed = [e for e in h10e["episodes"] if e.get("reward") == 0.0 and "error" not in e]
    out = {"protocol": {"oracle": ORACLE_MODEL,
                        "oracle_input": "task.actions (env's own write expectations) + instruction + DB + saved traj",
                        "predicate": "P-CAUSAL-FLIGHT unchanged: >=2/3 FLAGGED failures flip at k*",
                        "flagged_set": [e["task_id"] for e in failed if e.get("k_star_omission_flag") is not None],
                        "exploratory": "full-replay flips for ALL failures (repairability taxonomy)"},
           "episodes": []}
    out_path = os.path.join(HERE, "h10f_results.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)

    for e in failed:
        tid = e["task_id"]
        msgs = traces[str(tid)]
        env = get_env("retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
                      user_provider="openai", task_split="test", task_index=tid)
        env.reset(task_index=tid)
        t0 = time.time()
        calls = derive_with_expected(env, msgs, [a.name for a in (env.task.actions or [])])
        rec = {"task_id": tid, "expected": [a.name for a in (env.task.actions or [])],
               "derived_calls": calls, "secs": round(time.time() - t0, 1)}
        if calls:
            k_star = e.get("k_star_omission_flag")
            rec["reward_full_replay"] = replay_at(env, tid, msgs, 10 ** 6, calls)
            rec["flipped_full"] = rec["reward_full_replay"] == 1.0
            if k_star is not None:
                rec["k_star"] = k_star
                rec["reward_at_k_star"] = replay_at(env, tid, msgs, k_star, calls)
                rec["flipped_at_k_star"] = rec["reward_at_k_star"] == 1.0
        out["episodes"].append(rec)
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
        print(f"task {tid}: derived {len(calls)} calls | full flip: {rec.get('flipped_full')} "
              f"| k*-flip: {rec.get('flipped_at_k_star')} ({rec['secs']}s)", flush=True)

    # predicate on the SAME flagged set as h10e (no new episodes)
    flagged = [r for r in out["episodes"] if r.get("k_star") is not None]
    flips = sum(1 for r in flagged if r.get("flipped_at_k_star"))
    verdict = ("PASS" if len(flagged) >= 2 and flips >= 2 else
               "FAIL" if len(flagged) >= 2 else "NOT-EVALUABLE (n<2 flagged)")
    out["P-CAUSAL-FLIGHT"] = {"flagged": len(flagged), "flips": flips, "verdict": verdict}
    full_flips = sum(1 for r in out["episodes"] if r.get("flipped_full"))
    out["exploratory_receipt"] = {"failures": len(out["episodes"]), "full_replay_flips": full_flips}
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"\nP-CAUSAL-FLIGHT (expected-action oracle): {flips}/{len(flagged)} -> {verdict}")
    print(f"exploratory: {full_flips}/{len(out['episodes'])} failures repairable on full replay")


if __name__ == "__main__":
    main()
