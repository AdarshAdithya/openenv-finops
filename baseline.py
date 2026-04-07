"""
baseline.py — Baseline agent for openenv-finops.

Supports two modes:
  - rule_based : deterministic heuristics, no LLM required
  - llm        : Anthropic-powered reasoning via claude-sonnet-4-20250514
  - hybrid     : rule-based first; falls back to LLM when rules yield no action

Usage:
    python baseline.py --mode rule_based
    python baseline.py --mode llm    --api-key sk-ant-...
    python baseline.py --mode hybrid --api-key sk-ant-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from src.env import FinOpsEnv
from src.models import Observation, Action


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — bridge between Pydantic models and plain dicts
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_dict(obs: Observation) -> dict[str, Any]:
    """Observation pydantic model → plain dict for agent logic."""
    return {
        "total_cost": obs.total_cost,
        "resources": [r.model_dump() for r in obs.resources],
        "alerts": [
            f"{r.id}: low utilisation ({r.utilization:.0%})"
            for r in obs.resources
            if r.utilization < 0.20
        ],
    }


def dict_to_action(action_dict: dict[str, Any]) -> Action:
    """Agent action dict → Action pydantic model for env.step()."""
    atype = action_dict.get("type", "noop")
    if atype == "terminate_service":
        return Action(cmd="terminate", target_id=action_dict.get("service"))
    elif atype in ("scale_down", "scale_up", "resize"):
        return Action(cmd="resize", target_id=action_dict.get("service"))
    return Action(cmd="nop", target_id=None)


# ─────────────────────────────────────────────────────────────────────────────
# Episode data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class Episode:
    mode: str
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_reward": self.total_reward,
            "num_steps": len(self.steps),
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "reward": s.reward,
                    "done": s.done,
                }
                for s in self.steps
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent base
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Return an action dict given obs as a plain dict."""

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based agent
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedAgent(BaseAgent):
    """
    Priority order:
      1. Terminate non-prod resources with utilisation < 10%
      2. Resize (scale down) non-prod resources with utilisation < 20%
      3. Nop — never touches prod resources
    """

    name = "rule_based"

    def act(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        resources: list[dict] = obs_dict.get("resources", [])

        # 1. Terminate idle non-prod
        for r in resources:
            if not r.get("is_prod") and r.get("utilization", 1.0) < 0.10:
                return {
                    "type": "terminate_service",
                    "service": r["id"],
                    "reason": f"non-prod idle ({r['utilization']:.0%} util)",
                }

        # 2. Resize under-utilised non-prod
        for r in resources:
            if not r.get("is_prod") and r.get("utilization", 1.0) < 0.20:
                return {
                    "type": "scale_down",
                    "service": r["id"],
                    "reason": f"non-prod under-utilised ({r['utilization']:.0%} util)",
                }

        return {"type": "noop", "reason": "nothing actionable"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a FinOps agent managing cloud infrastructure costs.
You receive a JSON observation with these fields:
  - total_cost   : current total monthly cost (float)
  - resources    : list of {id, type, cost, utilization, is_prod}
  - alerts       : list of warning strings

Decide on exactly ONE action. Respond ONLY with a JSON object — no markdown, no commentary.
Valid actions:
  {"type": "noop"}
  {"type": "terminate_service", "service": "<id>", "reason": "..."}
  {"type": "scale_down",        "service": "<id>", "reason": "..."}

RULE: Never terminate or scale_down any resource where is_prod=true.
"""


class LLMAgent(BaseAgent):
    name = "llm"

    def __init__(self, api_key: str | None = None) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("Run: pip install anthropic")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required.")
        self._client = anthropic.Anthropic( api_key=api_key,
    base_url="https://openrouter.ai/api/v1",)
        self._history: list[dict[str, str]] = []

    def reset(self) -> None:
        self._history.clear()

    def act(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        user_msg = json.dumps(obs_dict, indent=2)
        self._history.append({"role": "user", "content": user_msg})

        response = self._client.messages.create(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=self._history,
        )

        raw = response.content[0].text.strip()
        self._history.append({"role": "assistant", "content": raw})

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"[LLMAgent] parse error: {raw!r}", file=sys.stderr)
            return {"type": "noop", "reason": "LLM parse error"}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid agent
# ─────────────────────────────────────────────────────────────────────────────

class HybridAgent(BaseAgent):
    name = "hybrid"

    def __init__(self, api_key: str | None = None) -> None:
        self._rule = RuleBasedAgent()
        self._llm = LLMAgent(api_key=api_key)

    def reset(self) -> None:
        self._rule.reset()
        self._llm.reset()

    def act(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        rule_action = self._rule.act(obs_dict)
        # Escalate to LLM only when rules return noop but alerts exist
        if rule_action.get("type") == "noop" and obs_dict.get("alerts"):
            print("[HybridAgent] Escalating to LLM …")
            return self._llm.act(obs_dict)
        return rule_action


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(mode: str, api_key: str | None) -> BaseAgent:
    if mode == "rule_based":
        return RuleBasedAgent()
    elif mode == "llm":
        return LLMAgent(api_key=api_key)
    elif mode == "hybrid":
        return HybridAgent(api_key=api_key)
    raise ValueError(f"Unknown mode: {mode!r}")


def run_episode(agent: BaseAgent, env: FinOpsEnv, max_steps: int = 50) -> Episode:
    episode = Episode(mode=agent.name)
    raw_obs = env.reset()
    agent.reset()

    for step_idx in range(max_steps):
        # Always convert to dict before passing to agent
        obs_dict = obs_to_dict(raw_obs) if isinstance(raw_obs, Observation) else raw_obs

        action_dict = agent.act(obs_dict)

        # Convert to Action pydantic model for env.step()
        action = dict_to_action(action_dict)
        result = env.step(action)

        # Handle both (obs, reward, done, info) and bare obs returns
        if isinstance(result, tuple):
            raw_obs, reward, done, info = result
        else:
            raw_obs, reward, done, info = result, 0.0, False, {}

        record = StepRecord(
            step=step_idx,
            observation=obs_dict,
            action=action_dict,
            reward=float(reward),
            done=bool(done),
            info=info if isinstance(info, dict) else {},
        )
        episode.steps.append(record)

        print(
            f"  step={step_idx:03d}  action={action_dict.get('type'):<22}"
            f"  reward={reward:+.3f}  done={done}"
        )

        if done:
            break

    return episode


def save_episode(episode: Episode, output_dir: str = "data") -> str:
    import time
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/episode_{episode.mode}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(episode.to_dict(), f, indent=2)
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rule_based", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    env = FinOpsEnv()
    agent = build_agent(args.mode, args.api_key)
    episode = run_episode(agent, env, max_steps=args.max_steps)

    print(f"\n✅ Episode complete — reward={episode.total_reward:.3f}  steps={len(episode.steps)}")

    if args.save:
        path = save_episode(episode)
        print(f"💾 Saved → {path}")


if __name__ == "__main__":
    main()