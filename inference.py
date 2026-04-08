import json, os, urllib.request, urllib.error
from typing import Any

BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")

def _req(url, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def pick_action(obs):
    for r in obs.get("resources", []):
        if not r.get("is_prod") and r.get("utilization", 1.0) < 0.15:
            return {"cmd": "terminate", "target_id": r["id"]}
    return {"cmd": "nop", "target_id": None}

def run():
    try:
        res = _req(f"{BASE_URL}/reset", "POST")
        eid, obs = res["episode_id"], res["observation"]
        print(f"[START] task=finops episode_id={eid}")
        
        done, step, total_r = False, 0, 0.0
        while not done and step < 50:
            act = pick_action(obs)
            s_res = _req(f"{BASE_URL}/episodes/{eid}/step", "POST", {"action": act})
            obs, done, r = s_res["observation"], s_res["done"], s_res["reward"]
            total_r += r
            print(f"[STEP] step={step} action={act['cmd']} target={act.get('target_id')} reward={r:.2f} done={done}")
            step += 1
            
        g = _req(f"{BASE_URL}/episodes/{eid}/grade")
        print(f"[END] episode_id={eid} steps={step} total_reward={total_r:.2f} score={g['cost_optimisation']['score']:.3f}")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    run()