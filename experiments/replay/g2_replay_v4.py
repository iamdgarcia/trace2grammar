"""G2 v4 — scripted counterfactual replay. Derives correct actions from env.data (the DB
the write-checks inspect) + task instruction rules; executes via the env's own tools.

P-R2: >=2/3 flips (env reward 0.0 -> 1.0).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UID = "yusuf_rossi_9620"

def exch_variant(prod, pred):
    return next((v for v in prod["variants"].values() if v["available"] and pred(v["options"])), None)

def replay_scripted(task_id):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = get_env("retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
                  user_provider="openai", task_split="test", task_index=task_id)
    env.reset()
    for m in ep["msgs"][1:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    env.step(Action(name=tc["function"]["name"], kwargs=json.loads(tc["function"]["arguments"] or "{}")))
                except Exception:
                    pass
    log, acts = [], []
    orders = [o for o in env.data["orders"].values() if o["user_id"] == UID]
    pay = next(p["payment_method_id"] for p in orders[0]["payment_history"] if p["transaction_type"] == "payment")

    if task_id == 1:  # exchange W2378156: keyboard IF clicky+RGB+full size exists, thermostat -> Google Home
        o = env.data["orders"]["#W2378156"]
        kb = next(i for i in o["items"] if i["name"] == "Mechanical Keyboard")
        th = next(i for i in o["items"] if i["name"] == "Smart Thermostat")
        kbp = env.data["products"][kb["product_id"]]
        kv = exch_variant(kbp, lambda x: x.get("switch type") == "clicky" and x.get("backlight") == "RGB" and x.get("size") == "full size")
        thp = env.data["products"][th["product_id"]]
        tv = exch_variant(thp, lambda x: "google" in str(x.get("compatibility", "")).lower())
        log.append(("kb clicky+RGB+full:", bool(kv), "| th google:", tv and tv["item_id"]))
        if tv:
            item_ids, new_ids = [th["item_id"]], [tv["item_id"]]
            if kv:
                item_ids.append(kb["item_id"]); new_ids.append(kv["item_id"])
            acts.append(("exchange_delivered_order_items", dict(order_id="#W2378156", item_ids=item_ids,
                        new_item_ids=new_ids, payment_method_id=pay)))
    elif task_id == 2:  # return cleaner+headphone+smart watch from a delivered order
        target = next((o for o in orders if o["status"] == "delivered" and
                       {"vacuum cleaner", "headphones", "smart watch"} <= {i["name"].lower() for i in o["items"]}), None)
        log.append(("target:", target and target["order_id"]))
        if target:
            ids = [i["item_id"] for i in target["items"] if i["name"].lower() in ("vacuum cleaner", "headphones", "smart watch")]
            acts.append(("return_delivered_order_items", dict(order_id=target["order_id"], item_ids=ids,
                        payment_method_id=pay, reason="no longer needed")))
    elif task_id == 4:  # modify BOTH pending orders' tshirts -> purple / s / v-neck / polyester
        for o in [x for x in orders if x["status"] == "pending"]:
            ts = [i for i in o["items"] if "t-shirt" in i["name"].lower() or "tshirt" in i["name"].lower()]
            if not ts:
                continue
            for i in ts:
                p = env.data["products"][i["product_id"]]
                def score(x):
                    n = {k.lower(): str(v).lower() for k, v in x.items()}
                    return sum(("purple" in v) for v in n.values()) * 100 + sum(("polyester" in v) for v in n.values()) * 10 + \
                        sum(("v-neck" in v or "v neck" in v) for v in n.values()) + (any(v == "s" for v in n.values()))
                cands = [(score(v["options"]), v) for v in p["variants"].values() if v["available"]]
                best = max(cands, key=lambda t: t[0])[1] if cands else None
                log.append((o["order_id"], i["item_id"], "->", best and best["item_id"], "opts", best and json.dumps(best["options"], ensure_ascii=False)[:80]))
                if best:
                    acts.append(("modify_pending_order_items", dict(order_id=o["order_id"], item_ids=[i["item_id"]],
                                options={i["item_id"]: best["item_id"]}, payment_method_id=pay)))
    for fn, kw in acts:
        try:
            r = env.step(Action(name=fn, kwargs=kw))
            log.append((fn, str(r.observation)[:150]))
        except Exception as e:
            log.append((fn, "EXC", str(e)[:120]))
    resp = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "###STOP###"}))
    return resp.reward, log

if __name__ == "__main__":
    probes = []
    for tid in (1, 2, 4):
        r, log = replay_scripted(tid)
        probes.append({"task_id": tid, "flipped": r == 1.0, "reward": r})
        print({"task_id": tid, "flipped": r == 1.0, "reward": r}, flush=True)
        for x in log:
            print("   ", str(x)[:230], flush=True)
    flips = sum(p["flipped"] for p in probes)
    verdict = "PASS" if flips >= 2 else "FAIL"
    print(f"\nP-R2 v4: {flips}/3 scripted-correct replays flip the env reward -> {verdict}")
    json.dump({"protocol": {"predicate": ">=2/3 scripted-correct replays flip env reward", "seed": [1, 2, 4], "version": 4},
               "probes": probes, "verdict": verdict},
              open(os.path.join(HERE, "g2_results.json"), "w", encoding="utf-8"), indent=1)
