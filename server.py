from __future__ import annotations
import time, uuid, os
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.env import FinOpsEnv
    from src.graders import grade_cost_optimisation, grade_production_protection
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.env import FinOpsEnv
    from src.graders import grade_cost_optimisation, grade_production_protection

class EpisodeStore:
    def __init__(self):
        self._episodes = {}
    def create(self):
        episode_id = str(uuid.uuid4())
        env = FinOpsEnv()
        raw = env.reset()
        obs = raw.model_dump() if hasattr(raw, 'model_dump') else (raw.dict() if hasattr(raw, 'dict') else dict(raw))
        record = {'id': episode_id, 'env': env, 'initial_obs': obs, 'current_obs': obs, 'done': False, 'steps': [], 'created_at': time.time()}
        self._episodes[episode_id] = record
        return episode_id, record
    def get(self, eid):
        r = self._episodes.get(eid)
        if r is None: raise HTTPException(status_code=404, detail='Episode not found')
        return r
    def delete(self, eid):
        self.get(eid); del self._episodes[eid]
    def list_ids(self): return list(self._episodes.keys())

store = EpisodeStore()

class CreateEpisodeResponse(BaseModel):
    episode_id: str
    initial_observation: dict
    message: str = 'Episode created.'
class StepRequest(BaseModel):
    action: dict
class StepResponse(BaseModel):
    step: int
    observation: dict
    reward: float
    done: bool
    info: dict
class EpisodeDetail(BaseModel):
    id: str
    done: bool
    num_steps: int
    created_at: float
    steps: list
class GradeResponse(BaseModel):
    episode_id: str
    cost_optimisation: dict
    production_protection: dict
class HealthResponse(BaseModel):
    status: str = 'ok'
    active_episodes: int

def _normalise(raw):
    from src.models import Action
    if 'cmd' in raw: return Action(cmd=raw.get('cmd','nop'), target_id=raw.get('target_id'))
    t = raw.get('type','noop'); tgt = raw.get('service') or raw.get('target_id')
    if t == 'terminate_service': return Action(cmd='terminate', target_id=tgt)
    if t in ('scale_down','resize'): return Action(cmd='resize', target_id=tgt)
    return Action(cmd='nop', target_id=None)

@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title='openenv-finops API', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ── All API routes FIRST ─────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
async def root():
    return {
        "name": "openenv-finops API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "health":   "GET  /health",
            "episodes": "POST /episodes  |  GET /episodes",
            "step":     "POST /episodes/{eid}/step",
            "grade":    "GET  /episodes/{eid}/grade",
            "reset":    "POST /reset",
        }
    }

@app.get('/health', response_model=HealthResponse)
async def health(): return HealthResponse(active_episodes=len(store.list_ids()))

@app.post('/episodes', response_model=CreateEpisodeResponse, status_code=201)
async def create_episode():
    eid, rec = store.create()
    return CreateEpisodeResponse(episode_id=eid, initial_observation=rec['initial_obs'])

@app.get('/episodes', response_model=list)
async def list_episodes(): return store.list_ids()

@app.get('/episodes/{eid}', response_model=EpisodeDetail)
async def get_episode(eid: str):
    r = store.get(eid)
    return EpisodeDetail(id=eid, done=r['done'], num_steps=len(r['steps']), created_at=r['created_at'], steps=r['steps'])

@app.post('/episodes/{eid}/step', response_model=StepResponse)
async def step(eid: str, body: StepRequest):
    r = store.get(eid)
    if r['done']: raise HTTPException(status_code=409, detail='Episode done.')
    action = _normalise(body.action)
    obs, reward, done, info = r['env'].step(action)
    obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else (obs.dict() if hasattr(obs, 'dict') else (dict(obs) if not isinstance(obs, dict) else obs))
    idx = len(r['steps'])
    r['steps'].append({'step': idx, 'action': body.action, 'observation': obs_dict, 'reward': reward, 'done': done, 'info': info if isinstance(info, dict) else {}})
    r['current_obs'] = obs_dict; r['done'] = done
    return StepResponse(step=idx, observation=obs_dict, reward=reward, done=done, info=info if isinstance(info, dict) else {})

@app.get('/episodes/{eid}/grade', response_model=GradeResponse)
async def grade_episode(eid: str):
    r = store.get(eid)
    actions = [s['action'] for s in r['steps']]
    easy = grade_cost_optimisation(r['initial_obs'], r['current_obs'], actions)
    hard = grade_production_protection(r['initial_obs'], r['current_obs'], actions)
    return GradeResponse(episode_id=eid,
        cost_optimisation={'task':easy.task,'passed':easy.passed,'score':easy.score,'breakdown':easy.breakdown,'violations':easy.violations,'notes':easy.notes},
        production_protection={'task':hard.task,'passed':hard.passed,'score':hard.score,'breakdown':hard.breakdown,'violations':hard.violations,'notes':hard.notes})

@app.delete('/episodes/{eid}', status_code=204)
async def delete_episode(eid: str): store.delete(eid)

@app.post('/reset')
async def reset():
    eid, rec = store.create()
    return {"episode_id": eid, "initial_observation": rec["initial_obs"]}

# ── Static files LAST — mount the dashboard at /ui, after API routes ─────
from fastapi.staticfiles import StaticFiles
if os.path.exists('web'):
    app.mount('/ui', StaticFiles(directory='web', html=True), name='web')