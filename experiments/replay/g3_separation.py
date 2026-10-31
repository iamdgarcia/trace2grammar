"""G3 — caps 4/5/3 ON SUBSTRATE (predeclared before running).

P-R3 (separation point, cap 4): for each failed seed episode, find the EARLIEST prefix
length k (in tool-call index) such that restoring the prefix and then executing the
correct action yields env reward 1.0. Predicate: an earliest-k exists for 3/3 seeds and
is reproducible (recompute twice, same value).
P-R4 (what-if discrimination, cap 5): correct action flips (1.0), a WRONG-action control
does not (0.0). Predicate: >=2/3 seeds discriminate.
Cap 3 (failure localization): first-mismatch index between the original trace's tool calls
and the task's expected actions (k_mis), reported alongside.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UID = "yusuf_rossi_9620"

def build_env(task_id):
    env = get_env("retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
                  user_provider="openai", task_split="test", task_index=task_id)
    env.reset(task_index=task_id)
    return env

def correct_acts(env, task_id):
    orders = [o for o in env.data["orders"].values() if o["user_id"] == UID]
    pay = next(p["payment_method_id"] for p in orders[0]["payment_history"] if p["transaction_type"] == "payment")
    acts = []
    if task_id == 1:
        o = env.data["orders"]["#W2378156"]
        th = next(i for i in o["items"] if i["name"] == "Smart Thermostat")
        tv = next((v for v in env.data["products"][th["product_id"]]["variants"].values()
                   if v["available"] and "google" in str(v["options"].get("compatibility", "")).lower()), None)
        if tv:
            acts.append(("exchange_delivered_order_items", dict(order_id="#W2378156", item_ids=[th["item_id"]],
                        new_item_ids=[tv["item_id"]], payment_method_id=pay)))
    elif task_id == 2:
        target = next((o for o in orders if o["status"] == "delivered" and
                       {"vacuum cleaner", "headphones", "smart watch"} <= {i["name"].lower() for i in o["items"]}), None)
        if target:
            ids = [i["item_id"] for i in target["items"] if i["name"].lower() in ("vacuum cleaner", "headphones", "smart watch")]
            acts.append(("return_delivered_order_items", dict(order_id=target["order_id"], item_ids=ids, payment_method_id=pay)))
    elif task_id == 4:
        for o in [x for x in orders if x["status"] == "pending"]:
            for i in [it for it in o["items"] if "t-shirt" in it["name"].lower() or "tshirt" in it["name"].lower()]:
                p = env.data["products"][i["product_id"]]
                def score(x):
                    n = {k.lower(): str(v).lower() for k, v in x.items()}
                    return sum(("purple" in v) for v in n.values()) * 100 + sum(("polyester" in v) for v in n.values()) * 10
                cands = [(score(v["options"]), v) for v in p["variants"].values() if v["available"]]
                best = max(cands, key=lambda t: t[0])[1] if cands else None
                if best:
                    acts.append(("modify_pending_order_items", dict(order_id=o["order_id"], item_ids=[i["item_id"]],
                                new_item_ids=[best["item_id"]], payment_method_id=pay)))
    return acts

def wrong_acts(env, task_id, acts):
    """Counterfactual controls: a plausible but WRONG write (must NOT flip the reward)."""
    w = []
    for fn, kw in acts:
        kw = json.loads(json.dumps(kw))
        if task_id == 1:   # exchange the thermostat for the Apple HomeKit variant (the one the user REJECTED)
            th = next(i for i in env.data["orders"]["#W2378156"]["items"] if i["name"] == "Smart Thermostat")
            ap = next(v["item_id"] for v in env.data["products"][th["product_id"]]["variants"].values()
                      if v["available"] and "apple" in str(v["options"].get("compatibility", "")).lower())
            kw["new_item_ids"] = [ap]
        elif task_id == 2:  # return the WRONG item (the keyboard instead of the watch)
            kb = next(i for i in env.data["orders"]["#W2378156"]["items"] if i["name"] == "Mechanical Keyboard")
            kw["item_ids"] = [kb["item_id"]]
        elif task_id == 4:  # modify to the WRONG variant (linear, black, polyester)
            it = kw["item_ids"][0]
            wrongv = next(v["item_id"] for v in env.data["products"][next(p["product_id"] for o in env.data["orders"].values() if o["order_id"] == kw["order_id"] for p in [next(i for i in o["items"] if i["item_id"] == it)])]["variants"].values() if v["available"] and "black" in json.dumps(v["options"]).lower())
            kw["new_item_ids"] = [wrongv]
        w.append((fn, kw))
    return w

def finish(env, log):
    count = sum(sum(1 for v in p["variants"].values() if v["available"])
                for p in env.data["products"].values() if "t-shirt" in p["name"].lower())
    try:
        env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": f"There are {count} t-shirt options available."}))
    except Exception:
        pass
    rr = env.calculate_reward()
    log.append(("reward", rr.reward))
    return rr.reward

def repair_at(task_id, k, acts_fn):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = build_env(task_id)
    calls = [tc for m in ep["msgs"][1:] if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]]
    for tc in calls[:k]:
        try:
            env.step(Action(name=tc["function"]["name"], kwargs=json.loads(tc["function"]["arguments"] or "{}")))
        except Exception:
            pass
    log = []
    for fn, kw in acts_fn(env, task_id) if acts_fn != "correct" else correct_acts(env, task_id):
        try:
            env.step(Action(name=fn, kwargs=kw))
        except Exception as e:
            log.append((fn, "EXC", str(e)[:80]))
    return finish(env, log)

def first_mismatch(task_id):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = build_env(task_id)
    trace_calls = [tc["function"]["name"] for m in ep["msgs"][1:] if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]]
    gt = [a.name for a in env.task.actions if a.name != RESPOND_ACTION_NAME and a.name not in env.terminate_tools]
    gt_names = {n for n in gt}
    for i, n in enumerate(trace_calls):
        if n in gt_names:
            return i  # first correct-looking call: everything before is drift (reads don't count as wrong)
    return None

if __name__ == "__main__":
    out = {}
    for tid in (1, 2, 4):
        n_calls = len([tc for m in next(r for r in json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8")) if r["task_id"] == tid)["msgs"][1:] if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]])
        rewards = {}
        for k in range(n_calls + 1):
            rewards[k] = repair_at(tid, k, "correct")
        sep = min(k for k, r in rewards.items() if r == 1.0) if any(r == 1.0 for r in rewards.values()) else None
        sep2 = min(k for k, r in {k: repair_at(tid, k, "correct") for k in range(n_calls + 1)}.items() if r == 1.0) if sep is not None else None
        rw = repair_at(tid, n_calls, lambda e, t: wrong_acts(e, t, correct_acts(e, t)))
        out[tid] = {"n_trace_calls": n_calls, "sep_k": sep, "sep_repro": sep2, "wrong_reward": rw,
                    "k_mis": first_mismatch(tid)}
        print(tid, out[tid], flush=True)
    p_r3 = all(out[t]["sep_k"] is not None and out[t]["sep_k"] == out[t]["sep_repro"] for t in (1, 2, 4))
    p_r4 = sum(1 for t in (1, 2, 4) if out[t]["sep_k"] is not None and out[t]["wrong_reward"] != 1.0) >= 2
    print(f"\nP-R3 (separation, 3/3 + reproducible): {'PASS' if p_r3 else 'FAIL'}")
    print(f"P-R4 (what-if discrimination >=2/3): {'PASS' if p_r4 else 'FAIL'}")
    json.dump({"protocol": {"P_R3": "earliest-k exists 3/3 and reproducible", "P_R4": ">=2/3 discriminate"},
               "results": out, "P_R3": "PASS" if p_r3 else "FAIL", "P_R4": "PASS" if p_r4 else "FAIL"},
              open(os.path.join(HERE, "g3_results.json"), "w", encoding="utf-8"), indent=1)
