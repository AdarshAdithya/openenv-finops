"""
inference.py — OpenEnv FinOps agent inference script.

Uses ONLY the Python standard library (no third-party packages).
Compatible with any bare Python 3.8+ environment.

Flow:
  1. POST /reset            → receive episode_id + initial observation
  2. Loop: POST /episodes/{episode_id}/step → send action, receive obs/reward/done
  3. Exit 0 on success

Policy (rule-based):
  - Skip prod resources (is_prod=True)
  - terminate if utilization < 0.10
  - resize    if utilization < 0.20
  - nop       otherwise

Usage:
    python inference.py
    python inference.py --base-url http://localhost:7860
    python inference.py --base-url https://adarshmukunda-openenv-finops.hf.space
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────

def _request(method: str, url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    """Make an HTTP request and return parsed JSON. Raises RuntimeError on failure."""
    data = json.dumps(payload or {}).encode("utf-8") if payload is not None else b"{}"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URLError {method} {url}: {exc.reason}") from exc


def post(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    return _request("POST", url, payload, timeout)


# ── Policy ────────────────────────────────────────────────────────────────────

def pick_action(observation: dict) -> dict:
    """Rule-based FinOps policy."""
    resources = observation.get("resources", [])

    # Terminate: non-prod, very low utilization
    for r in resources:
        if r.get("is_prod", True):
            continue
        if r.get("utilization", 1.0) < 0.10:
            return {"cmd": "terminate", "target_id": r["id"]}

    # Resize: non-prod, low utilization
    for r in resources:
        if r.get("is_prod", True):
            continue
        if r.get("utilization", 1.0) < 0.20:
            return {"cmd": "resize", "target_id": r["id"]}

    # No-op
    return {"cmd": "nop", "target_id": None}


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(base_url: str, max_steps: int = 20) -> None:
    base_url = base_url.rstrip("/")

    episode_id = None
    obs = None

    # Strategy 1: POST /episodes → always returns {episode_id, initial_observation}
    try:
        resp = post(f"{base_url}/episodes")
        episode_id = resp.get("episode_id")
        obs = resp.get("initial_observation", resp.get("observation", {}))
        if episode_id:
            print(f"[reset] via /episodes  episode_id={episode_id}  "
                  f"total_cost={obs.get('total_cost', 0.0):.2f}  "
                  f"resources={len(obs.get('resources', []))}")
    except RuntimeError:
        pass

    # Strategy 2: POST /reset → new format returns {episode_id, observation}
    if not episode_id:
        try:
            resp = post(f"{base_url}/reset")
            episode_id = resp.get("episode_id")
            obs = resp.get("observation", resp)  # flat fallback
            if episode_id:
                print(f"[reset] via /reset  episode_id={episode_id}  "
                      f"total_cost={obs.get('total_cost', 0.0):.2f}  "
                      f"resources={len(obs.get('resources', []))}")
        except RuntimeError as exc:
            print(f"[ERROR] /reset failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Strategy 3: Classic OpenEnv — /reset returned flat obs, use POST /step
    use_classic_step = (episode_id is None)
    if use_classic_step:
        # obs already set from /reset response (flat observation dict)
        print(f"[reset] via /reset (classic)  "
              f"total_cost={obs.get('total_cost', 0.0):.2f}  "
              f"resources={len(obs.get('resources', []))}")

    # ── Step loop ──────────────────────────────────────────────────────────────
    total_reward = 0.0

    for step_num in range(1, max_steps + 1):
        action = pick_action(obs)

        try:
            if use_classic_step:
                result = post(f"{base_url}/step", {"action": action})
                obs    = result.get("observation", {})
            else:
                result = post(
                    f"{base_url}/episodes/{episode_id}/step",
                    {"action": action},
                )
                obs = result.get("observation", {})
        except RuntimeError as exc:
            print(f"[ERROR] Step {step_num} failed: {exc}", file=sys.stderr)
            sys.exit(1)

        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
        total_reward += reward

        print(
            f"[step {step_num:02d}] action={json.dumps(action)}  "
            f"reward={reward:+.2f}  total_reward={total_reward:+.2f}  "
            f"done={done}  resources={len(obs.get('resources', []))}  "
            f"cost={obs.get('total_cost', 0.0):.2f}"
        )

        if done:
            break

    print(f"\nEpisode finished. Total reward: {total_reward:+.2f}")
    sys.exit(0)



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv FinOps inference script (stdlib only)")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the openenv-finops server (default: http://localhost:7860)",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per episode")
    args = parser.parse_args()
    run(args.base_url, args.max_steps)

