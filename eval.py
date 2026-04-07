"""
eval.py — Evaluation harness for openenv-finops.

Loads all episode JSON files from data/, runs both graders on each,
and prints a comparison table across modes.

Usage:
    python eval.py
    python eval.py --data-dir data/
    python eval.py --data-dir data/ --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from src.graders import grade_cost_optimisation, grade_production_protection


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_episodes(data_dir: str) -> list[dict[str, Any]]:
    """Load all episode_*.json files from data_dir."""
    episodes = []
    if not os.path.isdir(data_dir):
        print(f"⚠️  Directory '{data_dir}' not found.")
        return episodes

    for fname in sorted(os.listdir(data_dir)):
        if not (fname.startswith("episode_") and fname.endswith(".json")):
            continue
        path = os.path.join(data_dir, fname)
        try:
            with open(path) as f:
                ep = json.load(f)
            ep["_filename"] = fname
            episodes.append(ep)
        except Exception as e:
            print(f"  ⚠️  Could not load {fname}: {e}")

    return episodes


def extract_obs_and_actions(
    episode: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    Pull initial_obs, final_obs, and actions list out of an episode dict.

    Episode JSON structure (from baseline.py save_episode):
    {
      "mode": "rule_based",
      "total_reward": 49.5,
      "num_steps": 10,
      "steps": [
        {"step": 0, "action": {...}, "reward": 10.0, "done": false},
        ...
      ]
    }
    """
    steps = episode.get("steps", [])
    actions = [s.get("action", {}) for s in steps]

    # Observations may or may not be stored per-step.
    # If present, use first/last; otherwise build minimal stubs from actions.
    initial_obs = steps[0].get("observation", {}) if steps else {}
    final_obs = steps[-1].get("observation", {}) if steps else {}

    return initial_obs, final_obs, actions


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_episode(episode: dict[str, Any]) -> dict[str, Any]:
    initial_obs, final_obs, actions = extract_obs_and_actions(episode)

    easy = grade_cost_optimisation(initial_obs, final_obs, actions)
    hard = grade_production_protection(initial_obs, final_obs, actions)

    return {
        "mode":          episode.get("mode", "unknown"),
        "filename":      episode.get("_filename", ""),
        "num_steps":     episode.get("num_steps", len(episode.get("steps", []))),
        "total_reward":  episode.get("total_reward", 0.0),
        "easy": {
            "task":       easy.task,
            "passed":     easy.passed,
            "score":      easy.score,
            "breakdown":  easy.breakdown,
            "violations": easy.violations,
        },
        "hard": {
            "task":       hard.task,
            "passed":     hard.passed,
            "score":      hard.score,
            "breakdown":  hard.breakdown,
            "violations": hard.violations,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _pass(flag: bool) -> str:
    return "✅ PASS" if flag else "❌ FAIL"


def print_results(results: list[dict[str, Any]], verbose: bool = False) -> None:
    if not results:
        print("No episodes found to evaluate.")
        return

    # ── per-episode detail ────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  Episode Results")
    print(f"{'═'*70}")

    for r in results:
        print(f"\n  📄 {r['filename']}  [mode={r['mode']}]")
        print(f"     steps={r['num_steps']}  reward={r['total_reward']:+.3f}")
        print(f"     cost-optimisation  : {_pass(r['easy']['passed'])}  "
              f"score={r['easy']['score']:.3f}")
        print(f"     production-protect : {_pass(r['hard']['passed'])}  "
              f"score={r['hard']['score']:.3f}")

        if verbose:
            print("     ── cost-optimisation breakdown ──")
            for k, v in r["easy"]["breakdown"].items():
                print(f"        {k}: {v:.4f}")
            if r["easy"]["violations"]:
                for v in r["easy"]["violations"]:
                    print(f"        ⚠️  {v}")

            print("     ── production-protection breakdown ──")
            for k, v in r["hard"]["breakdown"].items():
                print(f"        {k}: {v:.4f}")
            if r["hard"]["violations"]:
                for v in r["hard"]["violations"]:
                    print(f"        ⚠️  {v}")

    # ── aggregate by mode ─────────────────────────────────────────────────────
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print(f"\n{'═'*70}")
    print("  Aggregate Comparison by Mode")
    print(f"{'═'*70}")
    print(f"  {'Mode':<14} {'Episodes':>9} {'Avg Reward':>11} "
          f"{'Easy Score':>11} {'Easy Pass%':>11} "
          f"{'Hard Score':>11} {'Hard Pass%':>11}")
    print(f"  {'─'*66}")

    for mode in sorted(by_mode.keys()):
        eps = by_mode[mode]
        n = len(eps)
        avg_reward    = sum(e["total_reward"]     for e in eps) / n
        avg_easy      = sum(e["easy"]["score"]    for e in eps) / n
        easy_pass_pct = sum(e["easy"]["passed"]   for e in eps) / n * 100
        avg_hard      = sum(e["hard"]["score"]    for e in eps) / n
        hard_pass_pct = sum(e["hard"]["passed"]   for e in eps) / n * 100

        print(f"  {mode:<14} {n:>9} {avg_reward:>+11.3f} "
              f"{avg_easy:>11.3f} {easy_pass_pct:>10.0f}% "
              f"{avg_hard:>11.3f} {hard_pass_pct:>10.0f}%")

    # ── winner ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    best_easy = max(by_mode.items(),
                    key=lambda kv: sum(e["easy"]["score"] for e in kv[1]) / len(kv[1]))
    best_hard = max(by_mode.items(),
                    key=lambda kv: sum(e["hard"]["score"] for e in kv[1]) / len(kv[1]))
    best_reward = max(by_mode.items(),
                      key=lambda kv: sum(e["total_reward"] for e in kv[1]) / len(kv[1]))

    print(f"  🏆 Best cost-optimisation : {best_easy[0]}")
    print(f"  🏆 Best prod-protection   : {best_hard[0]}")
    print(f"  🏆 Best avg reward        : {best_reward[0]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all recorded episodes and compare agent modes."
    )
    parser.add_argument("--data-dir", default="data", help="Episode directory (default: data/)")
    parser.add_argument("--verbose", action="store_true", help="Show per-score breakdowns")
    parser.add_argument("--save", default=None, metavar="FILE",
                        help="Save evaluation results to a JSON file")
    args = parser.parse_args()

    print(f"\n🔍  openenv-finops — evaluation harness")
    print(f"    Loading episodes from '{args.data_dir}/' …")

    episodes = load_episodes(args.data_dir)
    if not episodes:
        print("    No episodes found. Run record_episode.py first.")
        return

    print(f"    Found {len(episodes)} episode(s).\n")

    results = [evaluate_episode(ep) for ep in episodes]
    print_results(results, verbose=args.verbose)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾  Results saved → {args.save}\n")


if __name__ == "__main__":
    main()