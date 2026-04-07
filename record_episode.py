"""
record_episode.py — Smoke-test recorder for openenv-finops.

Runs one episode per mode (rule_based, llm, hybrid) and saves
each to data/episode_{mode}_{timestamp}.json.

Usage:
    python record_episode.py                        # rule_based only
    python record_episode.py --api-key sk-ant-...   # all 3 modes
    python record_episode.py --api-key sk-ant-... --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from src.env import FinOpsEnv
from src.models import Observation
from baseline import build_agent, run_episode, save_episode, Episode


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────

def record_all(
    api_key: str | None,
    max_steps: int,
    output_dir: str,
) -> list[dict[str, Any]]:
    modes = ["rule_based", "llm", "hybrid"]
    results = []

    for mode in modes:
        if mode in ("llm", "hybrid") and not api_key:
            print(f"\n⏭️  Skipping '{mode}' — no --api-key provided")
            continue

        print(f"\n{'─'*52}")
        print(f"  Mode: {mode}")
        print(f"{'─'*52}")

        try:
            env = FinOpsEnv()
            agent = build_agent(mode, api_key)
            episode: Episode = run_episode(agent, env, max_steps=max_steps)
            path = save_episode(episode, output_dir=output_dir)

            summary = {
                "mode":         mode,
                "num_steps":    len(episode.steps),
                "total_reward": round(episode.total_reward, 4),
                "saved_to":     path,
                "status":       "ok",
            }

        except Exception as exc:
            print(f"  ❌ {mode} failed: {exc}", file=sys.stderr)
            summary = {"mode": mode, "status": "error", "error": str(exc)}

        results.append(summary)
        if summary["status"] == "ok":
            print(f"\n  ✅ saved → {summary['saved_to']}")

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'═'*52}")
    print("  Recording Summary")
    print(f"{'═'*52}")
    print(f"  {'Mode':<14} {'Steps':>6} {'Reward':>10}  File")
    print(f"  {'─'*48}")
    for r in results:
        if r["status"] == "ok":
            fname = os.path.basename(r["saved_to"])
            print(f"  {r['mode']:<14} {r['num_steps']:>6} {r['total_reward']:>10.3f}  {fname}")
        else:
            print(f"  {r['mode']:<14}  ❌ {r.get('error', 'unknown error')}")
    print()


def save_manifest(results: list[dict[str, Any]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump({"recorded_at": time.time(), "episodes": results}, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Record one smoke-test episode per agent mode.")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per episode (default: 15)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    print("\n🚀  openenv-finops — episode recorder (smoke test: 1 episode per mode)")
    if not api_key:
        print("ℹ️   No API key — only rule_based mode will run.")

    results = record_all(api_key, args.max_steps, args.output_dir)
    print_summary(results)

    manifest_path = save_manifest(results, args.output_dir)
    print(f"📋  Manifest → {manifest_path}\n")


if __name__ == "__main__":
    main()