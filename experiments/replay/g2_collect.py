"""G2 collector: run 8 episodes (tasks 0-7), save full traces for failed ones.
Trace = the message list + final env reward. Output: g2_traces.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_runner_smoke import run_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    out = []
    for tid in range(8):
        try:
            r = run_episode(tid)
        except Exception as e:
            r = {"task_id": tid, "error": str(e)[:200], "reward": None, "done": False}
        r["trace_len"] = len(r.get("msgs", [])) if "msgs" in r else None
        out.append(r)
        print({k: v for k, v in r.items() if k != "msgs"}, flush=True)
    json.dump(out, open(os.path.join(HERE, "g2_traces.json"), "w", encoding="utf-8"), indent=1)
    print("saved", len(out))
