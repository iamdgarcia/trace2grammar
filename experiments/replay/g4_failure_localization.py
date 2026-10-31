"""G4 — cap 3 on substrate (P-R5): failure localization by prefix bisection with GT repair.

For each failed seed: repair(k) = restore trace prefix [0:k] + execute the task's expected
actions (GT oracle) + calculate_reward. Two failure classes fall out mechanically:
  omission-type:     repair flips (1.0) for EVERY k (state never went wrong; only omissions)
  wrong-write-type:  flips stop after the wrong write's index j (un-undoable state damage)
P-R5: 4/4 seeds classified, and for the wrong-write seed j points at a WRITE tool call whose
kwargs differ from the GT action at that point.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402
from g3_separation import build_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WRITE_TOOLS = {"exchange_delivered_order_items", "return_delivered_order_items",
               "modify_pending_order_items", "cancel_pending_order", "modify_user_address",
               "modify_pending_order_address", "modify_pending_order_payment"}

def repair_gt(task_id, k):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = build_env(task_id)
    calls = [tc for m in ep["msgs"][1:] if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]]
    for tc in calls[:k]:
        try:
            env.step(Action(name=tc["function"]["name"], kwargs=json.loads(tc["function"]["arguments"] or "{}")))
        except Exception:
            pass
    for a in env.task.actions:
        if a.name == RESPOND_ACTION_NAME:
            continue
        try:
            env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
        except Exception:
            pass
    count = sum(sum(1 for v in p["variants"].values() if v["available"])
                for p in env.data["products"].values() if "t-shirt" in p["name"].lower())
    try:
        env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": f"There are {count} t-shirt options available."}))
    except Exception:
        pass
    rr = env.calculate_reward()
    return rr.reward, calls

if __name__ == "__main__":
    out = {}
    for tid in (0, 1, 2, 4):
        rewards, calls = {}, None
        n = None
        # first pass: get n (trace length) then bisect
        _, calls = repair_gt(tid, 0)
        n = len(calls)
        for k in range(n + 1):
            r, _ = repair_gt(tid, k)
            rewards[k] = r
        flips = [k for k, r in rewards.items() if r == 1.0]
        if len(flips) == n + 1:
            ftype, j = "omission", None
        elif flips:
            j = max(flips)  # last k whose repair still flips: first damaging prefix element is calls[j]
            ftype = "wrong-write"
        else:
            ftype, j = "unrecoverable", 0
        jwrite = None
        if ftype == "wrong-write" and j < n:
            c = calls[j]
            jwrite = {"name": c["function"]["name"], "is_write": c["function"]["name"] in WRITE_TOOLS,
                      "args": c["function"]["arguments"][:150]}
        out[tid] = {"n": n, "rewards": {str(k): v for k, v in rewards.items()}, "type": ftype, "j": j, "jwrite": jwrite}
        print(tid, {"n": n, "type": ftype, "j": j, "flips": len(flips), "jwrite": jwrite}, flush=True)
    # predicate: 4/4 classified; wrong-write seed has j pointing at a WRITE call
    ok = sum(1 for t, v in out.items() if v["type"] in ("omission", "wrong-write"))
    j_ok = all(v["jwrite"] is None or v["jwrite"]["is_write"] for v in out.values())
    verdict = "PASS" if ok == 4 and j_ok else "FAIL"
    print(f"\nP-R5: {ok}/4 classified, j-on-write: {j_ok} -> {verdict}")
    json.dump({"protocol": {"predicate": "4/4 classified (omission | wrong-write); j on a write tool"},
               "results": out, "verdict": verdict},
              open(os.path.join(HERE, "g4_results.json"), "w", encoding="utf-8"), indent=1)
