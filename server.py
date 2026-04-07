"""
server.py — FastAPI server for openenv-finops.

Exposes the FinOpsEnv as a stateful HTTP API so any HTTP client
(curl, Python requests, external agents) can interact with it.

Routes
------
POST /episodes                 Create & reset a new episode, returns episode_id
GET  /episodes/{id}            Fetch episode metadata + full step history
POST /episodes/{id}/step       Send an action, receive obs/reward/done/info
GET  /episodes/{id}/grade      Run both graders, return scores
DELETE /episodes/{id}          Tear down episode

GET  /health                   Liveness probe
GET  /docs                     Auto-generated Swagger UI (FastAPI built-in)

Run
---
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── local imports ─────────────────────────────────────────────────────────────
try:
    from src.env import FinOpsEnv
    from src.graders import grade_cost_optimisation, grade_production_protection
except ImportError:  # allow running from repo root without install
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from src.env import FinOpsEnv  # type: ignore[no-redef]
    from src.graders import grade_cost_optimisation, grade_production_protection  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────────
# In-memory episode store
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeStore:
    """Thread-safe (single-process) episode registry."""

    def __init__(self) -> None:
        self._episodes: dict[str, dict[str, Any]] = {}

    def create(self) -> tuple[str, dict[str, Any]]:
        episode_id = str(uuid.uuid4())
        env = FinOpsEnv()
        initial_obs = env.reset()
        record = {
            "id": episode_id,
            "env": env,
            "initial_obs": initial_obs,
            "current_obs": initial_obs,
            "done": False,
            "steps": [],
            "created_at": time.time(),
        }
        self._episodes[episode_id] = record
        return episode_id, record

    def get(self, episode_id: str) -> dict[str, Any]:
        record = self._episodes.get(episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found")
        return record

    def delete(self, episode_id: str) -> None:
        self.get(episode_id)  # raises 404 if missing
        del self._episodes[episode_id]

    def list_ids(self) -> list[str]:
        return list(self._episodes.keys())


store = EpisodeStore()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class CreateEpisodeResponse(BaseModel):
    episode_id: str
    initial_observation: dict[str, Any]
    message: str = "Episode created. POST /episodes/{episode_id}/step to interact."


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(
        ...,
        examples=[{"type": "scale_down", "service": "analytics-cluster"}],
    )


class StepResponse(BaseModel):
    step: int
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class EpisodeDetail(BaseModel):
    id: str
    done: bool
    num_steps: int
    created_at: float
    steps: list[dict[str, Any]]


class GradeResponse(BaseModel):
    episode_id: str
    cost_optimisation: dict[str, Any]
    production_protection: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    active_episodes: int


# ─────────────────────────────────────────────────────────────────────────────
# App + middleware
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown — nothing to clean up for in-memory store


app = FastAPI(
    title="openenv-finops API",
    description=(
        "HTTP interface for the FinOps reinforcement-learning environment. "
        "Create episodes, send agent actions, and retrieve graded results."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    return HealthResponse(active_episodes=len(store.list_ids()))


@app.post("/episodes", response_model=CreateEpisodeResponse, status_code=201, tags=["Episodes"])
async def create_episode() -> CreateEpisodeResponse:
    """Reset the environment and start a new episode."""
    episode_id, record = store.create()
    return CreateEpisodeResponse(
        episode_id=episode_id,
        initial_observation=record["initial_obs"],
    )


@app.get("/episodes", response_model=list[str], tags=["Episodes"])
async def list_episodes() -> list[str]:
    """Return all active episode IDs."""
    return store.list_ids()


@app.get("/episodes/{episode_id}", response_model=EpisodeDetail, tags=["Episodes"])
async def get_episode(episode_id: str) -> EpisodeDetail:
    """Fetch metadata and full step history for an episode."""
    record = store.get(episode_id)
    return EpisodeDetail(
        id=episode_id,
        done=record["done"],
        num_steps=len(record["steps"]),
        created_at=record["created_at"],
        steps=record["steps"],
    )


@app.post("/episodes/{episode_id}/step", response_model=StepResponse, tags=["Episodes"])
async def step(episode_id: str, body: StepRequest) -> StepResponse:
    """
    Advance the episode by one step.

    Send an action dict; receive the next observation, reward, and done flag.
    Returns HTTP 409 if the episode has already terminated.
    """
    record = store.get(episode_id)

    if record["done"]:
        raise HTTPException(
            status_code=409,
            detail="Episode is already done. Create a new episode to continue.",
        )

    env: FinOpsEnv = record["env"]
    obs, reward, done, info = env.step(body.action)

    step_idx = len(record["steps"])
    step_record = {
        "step": step_idx,
        "action": body.action,
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }
    record["steps"].append(step_record)
    record["current_obs"] = obs
    record["done"] = done

    return StepResponse(
        step=step_idx,
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/episodes/{episode_id}/grade", response_model=GradeResponse, tags=["Grading"])
async def grade_episode(episode_id: str) -> GradeResponse:
    """
    Run both graders against the completed episode.

    Can be called mid-episode, but results are most meaningful after done=True.
    """
    record = store.get(episode_id)
    actions = [s["action"] for s in record["steps"]]
    initial_obs = record["initial_obs"]
    final_obs = record["current_obs"]

    easy = grade_cost_optimisation(initial_obs, final_obs, actions)
    hard = grade_production_protection(initial_obs, final_obs, actions)

    return GradeResponse(
        episode_id=episode_id,
        cost_optimisation={
            "task": easy.task,
            "passed": easy.passed,
            "score": easy.score,
            "breakdown": easy.breakdown,
            "violations": easy.violations,
            "notes": easy.notes,
        },
        production_protection={
            "task": hard.task,
            "passed": hard.passed,
            "score": hard.score,
            "breakdown": hard.breakdown,
            "violations": hard.violations,
            "notes": hard.notes,
        },
    )


@app.delete("/episodes/{episode_id}", status_code=200, tags=["Episodes"])
async def delete_episode(episode_id: str) -> dict:
    """Remove an episode from memory."""
    store.delete(episode_id)
    return {"deleted": episode_id}

# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv Standard API  (POST /reset, POST /step, GET /state)
# Required by the OpenEnv submission checker
# ─────────────────────────────────────────────────────────────────────────────

_global_env = FinOpsEnv()


class StepRequestStandard(BaseModel):
    action: dict[str, Any]


@app.post("/reset", tags=["OpenEnv"])
async def openenv_reset() -> dict[str, Any]:
    """Reset the environment and return the initial observation (OpenEnv standard)."""
    obs = _global_env.reset()
    return obs.model_dump()


@app.post("/step", tags=["OpenEnv"])
async def openenv_step(body: StepRequestStandard) -> dict[str, Any]:
    """Take one step in the environment (OpenEnv standard)."""
    from src.models import Action
    action_data = body.action
    action = Action(
        cmd=action_data.get("cmd", "nop"),
        target_id=action_data.get("target_id"),
    )
    obs, reward, done, info = _global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["OpenEnv"])
async def openenv_state() -> dict[str, Any]:
    """Return the current environment state (OpenEnv standard)."""
    return _global_env.state().model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Static UI Mount (MUST BE AT THE END)
# ─────────────────────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
import os

if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")


def main():
    """Entry point for openenv multi-mode deployment (referenced in pyproject.toml)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

