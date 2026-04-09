from __future__ import annotations
import json, os, sys, time, urllib.error, urllib.request
from typing import Any
BASE_URL  = os.environ.get('ENV_URL', 'http://localhost:7860').rstrip('/')
TASK_NAME = 'finops_cost_optimisation'
MAX_STEPS = 20

def _post(path, payload=None):
    url = BASE_URL + path
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(urllib.request.Request(BASE_URL+path, method='GET'), timeout=20) as r: return json.loads(r.read())

def wait():
    d = time.time()+60
    while time.time()<d:
        try: _get('/health'); return True
        except: time.sleep(3)
    return False

def choose(obs):
    for r in obs.get('resources',[]):
        if not r.get('is_prod') and float(r.get('utilization',100))<10: return {'cmd':'terminate','target_id':r['id']}
    for r in obs.get('resources',[]):
        if not r.get('is_prod') and float(r.get('utilization',100))<20: return {'cmd':'resize','target_id':r['id']}
    return {'cmd':'nop','target_id':None}

def main():
    print(f'[START] task={TASK_NAME}', flush=True)
    step_idx = 0; final_score = 0.0
    try:
        if not wait():
            print('[ERROR] Server not ready', flush=True)
            print(f'[END] task={TASK_NAME} score=0.0000 steps=0', flush=True); sys.exit(1)
        try: ep = _post('/episodes')
        except Exception as e:
            print(f'[ERROR] {e}', flush=True)
            print(f'[END] task={TASK_NAME} score=0.0000 steps=0', flush=True); sys.exit(1)
        eid = ep.get('episode_id','')
        if not eid:
            print('[ERROR] no episode_id', flush=True)
            print(f'[END] task={TASK_NAME} score=0.0000 steps=0', flush=True); sys.exit(1)
        obs = ep.get('initial_observation',{}); done = False
        for step_idx in range(1, MAX_STEPS+1):
            if done: break
            action = choose(obs)
            try: sr = _post(f'/episodes/{eid}/step', {'action': action})
            except urllib.error.HTTPError as e:
                print(f'[ERROR] HTTP {e.code}: {e.read().decode()}', flush=True)
                print(f'[STEP] step={step_idx} reward=0.0000', flush=True); break
            except Exception as e:
                print(f'[ERROR] {e}', flush=True)
                print(f'[STEP] step={step_idx} reward=0.0000', flush=True); break
            reward = float(sr.get('reward',0)); done = bool(sr.get('done',False)); obs = sr.get('observation',obs)
            print(f'[STEP] step={step_idx} reward={reward:.4f}', flush=True)
        try:
            g = _get(f'/episodes/{eid}/grade')
            final_score = round(0.5*float(g.get('cost_optimisation',{}).get('score',0))+0.5*float(g.get('production_protection',{}).get('score',0)),4)
        except Exception as e: print(f'[ERROR] grade: {e}', flush=True)
    except Exception as e: print(f'[ERROR] {e}', flush=True)
    print(f'[END] task={TASK_NAME} score={final_score:.4f} steps={step_idx}', flush=True)

if __name__ == '__main__': main()
