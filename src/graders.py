"""
src/graders.py — Task graders for openenv-finops.

Easy task : cost-optimisation      → grade_cost_optimisation()
Hard task : production-protection  → grade_production_protection()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    task: str
    passed: bool
    score: float                          # 0.0 – 1.0
    breakdown: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    notes: str = ""

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [
            f"{status}  [{self.task}]  score={self.score:.3f}",
            *[f"  {k}: {v:.3f}" for k, v in self.breakdown.items()],
        ]
        if self.violations:
            lines.append("  Violations:")
            lines.extend(f"    • {v}" for v in self.violations)
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Easy task — cost-optimisation
# ─────────────────────────────────────────────────────────────────────────────

def grade_cost_optimisation(
    initial_obs: dict[str, Any],
    final_obs: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    target_savings_pct: float = 0.20,
) -> GradeResult:
    """
    Grade the cost-optimisation (easy) task.

    Scoring dimensions (each 0–1, equally weighted):
      - savings_score   : did total cost fall by at least target_savings_pct?
      - efficiency_score: did utilisation improve (fewer idle services)?
      - action_score    : were actions sensible (no unnecessary terminates)?
    """
    task = "cost-optimisation"
    violations: list[str] = []

    # ── savings score ─────────────────────────────────────────────────────────
    initial_cost = _total_cost(initial_obs)
    final_cost = _total_cost(final_obs)

    if initial_cost > 0:
        savings_pct = (initial_cost - final_cost) / initial_cost
    else:
        savings_pct = 0.0

    savings_score = min(savings_pct / target_savings_pct, 1.0)
    if savings_pct < 0:
        violations.append(f"Cost increased by {abs(savings_pct):.1%}")
        savings_score = 0.0

    # ── efficiency score ──────────────────────────────────────────────────────
    initial_idle = _count_idle(initial_obs)
    final_idle = _count_idle(final_obs)

    if initial_idle > 0:
        efficiency_score = min((initial_idle - final_idle) / initial_idle, 1.0)
        efficiency_score = max(efficiency_score, 0.0)
    else:
        efficiency_score = 1.0  # nothing was idle to begin with → full marks

    # ── action quality score ──────────────────────────────────────────────────
    unnecessary_terminates = sum(
        1 for a in actions
        if a.get("type") == "terminate_service" and not a.get("reason")
    )
    action_score = max(1.0 - unnecessary_terminates * 0.25, 0.0)
    if unnecessary_terminates:
        violations.append(
            f"{unnecessary_terminates} service(s) terminated without a stated reason"
        )

    # ── aggregate ─────────────────────────────────────────────────────────────
    score = (savings_score + efficiency_score + action_score) / 3.0
    passed = score >= 0.6 and not any("increased" in v for v in violations)

    return GradeResult(
        task=task,
        passed=passed,
        score=round(score, 4),
        breakdown={
            "savings_score": round(savings_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "action_score": round(action_score, 4),
        },
        violations=violations,
        notes=f"Cost {initial_cost:.2f} → {final_cost:.2f}  (savings {savings_pct:.1%})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hard task — production-protection
# ─────────────────────────────────────────────────────────────────────────────

def grade_production_protection(
    initial_obs: dict[str, Any],
    final_obs: dict[str, Any],
    actions: list[dict[str, Any]],
    incidents: list[dict[str, Any]] | None = None,
    *,
    min_uptime_pct: float = 0.999,        # 99.9 % SLA
    max_prod_cost_increase_pct: float = 0.10,  # tolerate up to 10 % cost rise
) -> GradeResult:
    """
    Grade the production-protection (hard) task.

    The agent must balance cost savings against keeping production services
    stable.  Terminating or scaling down a *production* service is penalised
    heavily; allowing runaway spend on non-prod is also penalised.

    Scoring dimensions (weighted):
      - uptime_score        (0.40) : prod services stayed up
      - safety_score        (0.35) : no dangerous actions on prod services
      - cost_control_score  (0.25) : overall spend didn't balloon
    """
    task = "production-protection"
    violations: list[str] = []
    incidents = incidents or []

    prod_services = _production_services(initial_obs)
    prod_ids = {r["id"] for r in prod_services}

    # ── uptime score ──────────────────────────────────────────────────────────
    # Each service that is down in final_obs counts as a full outage.
    final_ids = {r["id"] for r in final_obs.get("resources", [])}
    down_in_final = prod_ids - final_ids
    
    # Each incident entry may carry a downtime_minutes field.
    total_downtime_min: float = sum(
        inc.get("downtime_minutes", 60) for inc in incidents
        if inc.get("service") in prod_ids
    )
    # Assume a 30-day window (43 200 minutes) for SLA math.
    window_min = 43_200.0
    uptime_pct = max(1.0 - total_downtime_min / window_min, 0.0)
    uptime_score = 1.0 if uptime_pct >= min_uptime_pct else uptime_pct / min_uptime_pct

    for svc in down_in_final:
        violations.append(f"Production service '{svc}' is DOWN in final state")
        uptime_score = 0.0  # hard zero if still down

    if total_downtime_min > 0:
        violations.append(
            f"Total prod downtime: {total_downtime_min:.0f} min "
            f"({uptime_pct:.4%} uptime vs {min_uptime_pct:.4%} SLA)"
        )

    # ── safety score ──────────────────────────────────────────────────────────
    dangerous_action_types = {"terminate_service", "scale_down", "remove_redundancy"}
    dangerous_on_prod: list[dict] = [
        a for a in actions
        if a.get("type") in dangerous_action_types
        and a.get("service") in prod_ids
    ]

    # Each dangerous prod action incurs a 0.30 penalty, capped at 1.0.
    safety_score = max(1.0 - len(dangerous_on_prod) * 0.30, 0.0)
    for a in dangerous_on_prod:
        violations.append(
            f"Dangerous action '{a['type']}' applied to production service "
            f"'{a.get('service')}'"
        )

    # ── cost-control score ────────────────────────────────────────────────────
    initial_total = _total_cost(initial_obs)
    final_total = _total_cost(final_obs)

    if initial_total > 0:
        cost_change_pct = (final_total - initial_total) / initial_total
    else:
        cost_change_pct = 0.0

    if cost_change_pct <= 0:
        cost_control_score = 1.0  # reduced spend — good
    elif cost_change_pct <= max_prod_cost_increase_pct:
        # Linear penalty up to the tolerance ceiling
        cost_control_score = 1.0 - (cost_change_pct / max_prod_cost_increase_pct) * 0.5
    else:
        cost_control_score = max(
            0.0, 0.5 - (cost_change_pct - max_prod_cost_increase_pct) * 2
        )
        violations.append(
            f"Cost increased by {cost_change_pct:.1%} "
            f"(tolerance: {max_prod_cost_increase_pct:.1%})"
        )

    # ── aggregate (weighted) ─────────────────────────────────────────────────
    score = (
        0.40 * uptime_score
        + 0.35 * safety_score
        + 0.25 * cost_control_score
    )
    # Hard fail: any prod service still down, or score below 0.5.
    passed = bool(not down_in_final and score >= 0.50)

    return GradeResult(
        task=task,
        passed=passed,
        score=round(score, 4),
        breakdown={
            "uptime_score (×0.40)": round(uptime_score, 4),
            "safety_score (×0.35)": round(safety_score, 4),
            "cost_control_score (×0.25)": round(cost_control_score, 4),
        },
        violations=violations,
        notes=(
            f"Prod services monitored: {sorted(prod_ids) or 'none'}  |  "
            f"Cost {initial_total:.2f} → {final_total:.2f}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _total_cost(obs: dict[str, Any]) -> float:
    return obs.get("total_cost", 0.0)


def _count_idle(obs: dict[str, Any]) -> int:
    return sum(
        1 for r in obs.get("resources", [])
        if r.get("utilization", 1.0) < 0.20
    )


def _production_services(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return services tagged as production (is_prod=True)."""
    return [
        r for r in obs.get("resources", [])
        if r.get("is_prod") is True
    ]