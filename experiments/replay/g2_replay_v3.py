"""G2 v3 — scripted counterfactual replay (P-R2): does the ENV flip when the correct
action is executed? Derives the correct actions from the DB via the env's own tools
(oracle = task instruction + tool results), executes them scripted, reads the env verdict.

v1 (hints, agent-executed): 0/3 — user-sim hijack + strict DB checks beat flash-lite.
v2 (execution-dominant hints): 0/2 (+1 crash) — same.
v3 (this): decouple substrate proof from agent quality.
Guards: None-obs user-sim responses no longer crash.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import Action  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

def call(env, fn, **kw):
    try:
        r = env.step(Action(name=fn, kwargs=kw))
        return r.observation
    except Exception as e:
        return f"Error: {e}"

def opt_with(names, pred):
    for o in names.get("options", []):
        if pred(o.get("name", "").lower(), o.get("availability", "")):
            return o
    return None

def replay_scripted(task_id):
    tr = json.load(open(os.path.join(HERE, "g2_traces.json"), encoding="utf-8"))
    ep = next(r for r in tr if r["task_id"] == task_id)
    env = get_env("retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
                  user_provider="openai", task_split="test", task_index=task_id)
    env.reset()
    for m in ep["msgs"][1:]:  # restore DB state
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}")
                call(env, fn, **args)
    log = []
    if task_id == 1:
        order = call(env, "get_order_details", order_id="#W2378156".lstrip("#"))
        log.append(("order", str(order)[:120]))
        items = {i["item_id"]: i for i in order["order_items"]} if isinstance(order, dict) else {}
        prod = call(env, "get_product_details", product_id="1656367028")
        kb_opt = opt_with(prod, lambda n, a: "clicky" in n and "rgb" in n and "full size" in n) if isinstance(prod, dict) else None
        th_item = next((i for i in items.values() if "thermostat" in str(i.get("product_id", "")) or "thermostat" in str(i.get("name", "")).lower()), None)
        if th_item is None:
            for i in items.values():
                p = call(env, "get_product_details", product_id=str(i.get("product_id")))
                if isinstance(p, dict) and "thermostat" in str(p.get("name", "")).lower():
                    th_item = i
                    break
        log.append(("kb_opt", kb_opt and kb_opt["id"], "th_item", th_item and th_item["item_id"]))
        if th_item:
            th_prod = call(env, "get_product_details", product_id=str(th_item["product_id"]))
            g_opt = opt_with(th_prod, lambda n, a: "google" in n) if isinstance(th_prod, dict) else None
            log.append(("google_opt", g_opt and g_opt["id"]))
            if g_opt:
                new_items = {str(th_item["item_id"]): str(g_opt["id"])}
                if kb_opt and str(kb_opt.get("item_id") or kb_opt.get("id")):
                    kb_item = next((i for i in items.values() if str(i.get("product_id")) == "1656367028"), None)
                    if kb_item:
                        new_items[str(kb_item["item_id"])] = str(kb_opt["id"])
                pm = order.get("payment_methods") or list(order.get("order_payment_methods", {}).values())
                pay = pm[0]["payment_method_id"] if pm and isinstance(pm[0], dict) else (order.get("payment_method_id") or "credit_card")
                kw = dict(order_id="W2378156", new_item_options=new_items, payment_method=pay)
                obs = call(env, "exchange_delivered_order_items", **kw)
                log.append(("EXCHANGE", str(obs)[:220]))
    elif task_id == 2:
        uid = call(env, "find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122")
        log.append(("uid", str(uid)[:60]))
        ud = call(env, "get_user_details", user_id=str(uid))
        delivered = [o for o in ud.get("orders", []) if o.get("status") == "delivered"] if isinstance(ud, dict) else []
        target = None
        for oid in delivered:
            od = call(env, "get_order_details", order_id=oid)
            names = " ".join(str(i.get("name", "")).lower() for i in od.get("order_items", [])) if isinstance(od, dict) else ""
            if "cleaner" in names and "headphone" in names and "watch" in names:
                target = (oid, od)
                break
        if target:
            oid, od = target
            item_ids = [str(i["item_id"]) for i in od["order_items"] if any(k in str(i.get("name", "")).lower() for k in ("cleaner", "headphone", "watch"))]
            obs = call(env, "return_delivered_order_items", order_id=oid, item_ids=item_ids,
                       payment_method=od.get("payment_method_id", "credit_card"), reason="no longer needed")
            log.append(("RETURN", oid, item_ids, str(obs)[:220]))
    elif task_id == 4:
        uid = call(env, "find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122")
        ud = call(env, "get_user_details", user_id=str(uid))
        pending = [o for o in ud.get("orders", []) if o.get("status") == "pending"] if isinstance(ud, dict) else []
        for oid in pending:
            od = call(env, "get_order_details", order_id=oid)
            tshirts = [i for i in od.get("order_items", []) if "tshirt" in str(i.get("name", "")).lower()] if isinstance(od, dict) else []
            if not tshirts:
                continue
            new_items = {}
            for i in tshirts:
                p = call(env, "get_product_details", product_id=str(i["product_id"]))
                best = None
                if isinstance(p, dict):
                    for o in p.get("options", []):
                        n = o.get("name", "").lower()
                        score = ("purple" in n) + (" s" == n.split(",")[1].strip() if "," in n else "size s" in n) + ("polyester" in n) + ("v-neck" in n or "v neck" in n)
                        if o.get("available", True) and (best is None or score > best[0]):
                            best = (score, o["id"])
                if best:
                    new_items[str(i["item_id"])] = str(best[1])
            if new_items:
                obs = call(env, "modify_pending_order_items", order_id=oid, item_options=new_items, payment_method=od.get("payment_method_id", "credit_card"))
                log.append(("MODIFY", oid, new_items, str(obs)[:180]))
    rew = None
    for probe_fn, kw in (("exchange_delivered_order_items", {}),):
        break
    # verdict: compute env reward by ending the episode
    from tau_bench.types import Action as A, RESPOND_ACTION_NAME
    resp = env.step(A(name=RESPOND_ACTION_NAME, kwargs={"content": "###STOP###"}))
    return resp.reward, log

if __name__ == "__main__":
    probes = []
    for tid in (1, 2, 4):
        r, log = replay_scripted(tid)
        probes.append({"task_id": tid, "flipped": r == 1.0, "reward": r, "log": [str(x)[:160] for x in log]})
        print({"task_id": tid, "flipped": r == 1.0, "reward": r}, flush=True)
        for x in log:
            print("   ", str(x)[:200], flush=True)
    flips = sum(p["flipped"] for p in probes)
    verdict = "PASS" if flips >= 2 else "FAIL"
    print(f"\nP-R2 v3: {flips}/3 scripted-correct replays flip the env reward -> {verdict}")
    json.dump({"protocol": {"predicate": ">=2/3 scripted-correct replays flip env reward", "seed": [1, 2, 4], "version": 3},
               "probes": probes, "verdict": verdict},
              open(os.path.join(HERE, "g2_results.json"), "w", encoding="utf-8"), indent=1)
