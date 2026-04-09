from __future__ import annotations
import json, os, sys, time, urllib.error, urllib.request
from typing import Any

API_BASE_URL     = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY          = os.environ.get("API_KEY", "no-key")
MODEL_NAME       = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "")

BASE_URL  = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS = 20

TASKS = [
    "finops_cost_optimisation",
    "finops_production_protection",
    "finops_idle_resource_cleanup",
]

from openai import OpenAI
_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM = (
    "You are a FinOps agent managing cloud infrastructure costs. "
    "Given a JSON observation with resources (id, type, cost, utilization, is_prod), "
    "return ONE action as raw JSON only, no markdown.\n"
    "Valid actions:\n"
    '  {"cmd":"terminate","target_id":"<id>"}  -- for idle non-prod resource\n'
    '  {"cmd":"resize","target_id":"<id>"}     -- for underused non-prod resource\n'
    '  {"cmd":"nop","target_id":null}           -- nothing to do\n'
    "NEVER touch resources where is_prod=true."
)

def _post(path, payload=None):
    url = BASE_URL + path
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(urllib.request.Request(BASE_URL+path, method="GET"), timeout=20) as r: return json.loads(r.read())

def wait():
    d = time.time()+60
    while time.time()<d:
        try: _get("/health"); return True
        except: time.sleep(3)
    return False

def choose(obs: dict) -> dict:
    try:
        resp = _llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": json.dumps(obs)},
            ],
            max_tokens=64,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        action = json.loads(raw)
        if "cmd" not in action:
            raise ValueError("no cmd")
        return action
    except Exception:
        for r in obs.get("resources", []):
            if not r.get("is_prod") and float(r.get("utilization", 100)) < 10:
                return {"cmd": "terminate", "target_id": r["id"]}
        for r in obs.get("resources", []):
            if not r.get("is_prod") and float(r.get("utilization", 100)) < 20:
                return {"cmd": "resize", "target_id": r["id"]}
        return {"cmd": "nop", "target_id": None}

def run_task(task_name: str) -> tuple[float, int]:
    step_idx = 0
    final_score = 0.0
    try:
        try: ep = _post("/episodes")
        except Exception as e:
            print(f"[ERROR] {task_name} create failed: {e}", flush=True)
            return 0.01, 0

        eid = ep.get("episode_id", "")
        if not eid:
            return 0.01, 0

        obs = ep.get("initial_observation", {}); done = False

        for step_idx in range(1, MAX_STEPS+1):
            if done: break
            action = choose(obs)
            try: sr = _post(f"/episodes/{eid}/step", {"action": action})
            except urllib.error.HTTPError as e:
                print(f"[ERROR] HTTP {e.code}: {e.read().decode()}", flush=True)
                print(f"[STEP] step={step_idx} reward=0.0000", flush=True)
                break
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
                print(f"[STEP] step={step_idx} reward=0.0000", flush=True)
                break
            reward = float(sr.get("reward", 0))
            done   = bool(sr.get("done", False))
            obs    = sr.get("observation", obs)
            print(f"[STEP] step={step_idx} reward={reward:.4f}", flush=True)

        try:
            g = _get(f"/episodes/{eid}/grade")
            easy = float(g.get("cost_optimisation",    {}).get("score", 0.0))
            hard = float(g.get("production_protection",{}).get("score", 0.0))
            raw_score = round(0.5 * easy + 0.5 * hard, 4)
            # Clamp strictly between 0 and 1 (exclusive)
            final_score = max(0.01, min(0.99, raw_score))
        except Exception as e:
            print(f"[ERROR] grade: {e}", flush=True)
            final_score = 0.01

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        final_score = 0.01

    return final_score, step_idx


def main():
    if not wait():
        for task in TASKS:
            print(f"[START] task={task}", flush=True)
            print(f"[STEP] step=1 reward=0.0000", flush=True)
            print(f"[END] task={task} score=0.01 steps=1", flush=True)
        sys.exit(1)

    for task in TASKS:
        print(f"[START] task={task}", flush=True)
        score, steps = run_task(task)
        print(f"[END] task={task} score={score:.4f} steps={steps}", flush=True)


if __name__ == "__main__": main()
