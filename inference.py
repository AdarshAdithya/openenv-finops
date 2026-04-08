"""
inference.py — OpenEnv-compatible inference script for openenv-finops.

Connects to the running environment server and runs a rule-based agent
using the standard OpenEnv API (POST /reset, POST /step, GET /state).

Uses only Python standard library modules (urllib) so it runs in any
bare Python environment without requiring pip installs.

Usage:
    python inference.py --base-url http://localhost:7860
    python inference.py --base-url https://adarshmukunda-openenv-finops.hf.space
"""

import argparse
import json
import urllib.request
import urllib.error


def _post(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    """POST JSON to url, return parsed JSON response."""
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e


def pick_action(observation: dict) -> dict:
    """
    Rule-based policy:
      1. Terminate non-prod resource with utilization < 10%
      2. Resize non-prod resource with utilization < 20%
      3. No-op
    """
    resources = observation.get("resources", [])
    for r in resources:
        if not r["is_prod"] and r["utilization"] < 10.0:
            return {"cmd": "terminate", "target_id": r["id"]}
    for r in resources:
        if not r["is_prod"] and r["utilization"] < 20.0:
            return {"cmd": "resize", "target_id": r["id"]}
    return {"cmd": "nop", "target_id": None}


def run(base_url: str, max_steps: int = 20) -> None:
    base_url = base_url.rstrip("/")

    # Reset
    obs = _post(f"{base_url}/reset")
    print(f"[reset] total_cost={obs['total_cost']:.2f}  resources={len(obs['resources'])}")

    total_reward = 0.0
    for step in range(max_steps):
        action = pick_action(obs)
        result = _post(f"{base_url}/step", {"action": action})

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        total_reward += reward

        print(
            f"[step {step+1:02d}] action={json.dumps(action)}  "
            f"reward={reward:+.1f}  total_reward={total_reward:+.1f}  done={done}  "
            f"resources={len(obs['resources'])}  cost={obs['total_cost']:.2f}"
        )

        if done:
            break

    print(f"\nEpisode finished. Total reward: {total_reward:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv FinOps inference script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the openenv-finops server",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    run(args.base_url, args.max_steps)
