"""Column-and-constraint generation for budgeted robust production.

The robust constraint is

    max_z (a + d*z)'x <= capacity
    0 <= z_i <= 1, sum_i z_i <= Gamma.

CCG starts from the nominal scenario, solves a master problem, calls an
adversarial uncertainty subproblem, and adds the worst-case scenario whenever
it violates capacity. A compact Bertsimas-Sim robust counterpart is solved
independently for verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class RobustProductionInstance:
    profit: np.ndarray
    nominal_use: np.ndarray
    deviation: np.ndarray
    capacity: float
    upper_bounds: np.ndarray
    gamma: float


@dataclass
class CCGResult:
    production: np.ndarray
    profit: float
    compact_profit: float
    iterations: int
    scenarios: list[np.ndarray]
    worst_case_consumption: float
    violation: float
    history: list[dict[str, float]]


def example_instance() -> RobustProductionInstance:
    return RobustProductionInstance(
        profit=np.array([8.0, 7.0, 6.0]),
        nominal_use=np.array([3.0, 2.0, 2.0]),
        deviation=np.array([1.0, 1.5, 0.5]),
        capacity=18.0,
        upper_bounds=np.array([5.0, 5.0, 5.0]),
        gamma=1.5,
    )


def _solve_master(
    instance: RobustProductionInstance, scenarios: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    a_ub = np.vstack(
        [instance.nominal_use + instance.deviation * z for z in scenarios]
    )
    b_ub = np.full(len(scenarios), instance.capacity)
    result = linprog(
        -instance.profit,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, float(u)) for u in instance.upper_bounds],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CCG master failed: {result.message}")
    return np.asarray(result.x), float(-result.fun)


def adversarial_scenario(
    instance: RobustProductionInstance, production: np.ndarray
) -> tuple[np.ndarray, float]:
    n = instance.profit.size
    result = linprog(
        -(instance.deviation * production),
        A_ub=np.ones((1, n)),
        b_ub=np.array([instance.gamma]),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CCG adversarial problem failed: {result.message}")
    z = np.asarray(result.x)
    consumption = float(
        (instance.nominal_use + instance.deviation * z) @ production
    )
    return z, consumption


def solve_compact_robust_counterpart(
    instance: RobustProductionInstance | None = None,
) -> tuple[float, np.ndarray]:
    instance = instance or example_instance()
    n = instance.profit.size
    # Variables: x, rho, q.
    c = np.zeros(2 * n + 1)
    c[:n] = -instance.profit

    rows = []
    rhs = []

    row = np.zeros(2 * n + 1)
    row[:n] = instance.nominal_use
    row[n] = instance.gamma
    row[n + 1 :] = 1.0
    rows.append(row)
    rhs.append(instance.capacity)

    for i in range(n):
        row = np.zeros(2 * n + 1)
        row[i] = instance.deviation[i]
        row[n] = -1.0
        row[n + 1 + i] = -1.0
        rows.append(row)
        rhs.append(0.0)

    bounds = [(0.0, float(u)) for u in instance.upper_bounds]
    bounds += [(0.0, None)] * (n + 1)

    result = linprog(
        c,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Compact robust counterpart failed: {result.message}")
    return float(-result.fun), np.asarray(result.x[:n])


def solve_ccg(
    instance: RobustProductionInstance | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> CCGResult:
    instance = instance or example_instance()
    scenarios = [np.zeros(instance.profit.size)]
    history: list[dict[str, float]] = []

    for iteration in range(1, max_iterations + 1):
        x, profit = _solve_master(instance, scenarios)
        z, worst_consumption = adversarial_scenario(instance, x)
        violation = worst_consumption - instance.capacity
        history.append(
            {
                "iteration": float(iteration),
                "profit": profit,
                "worst_case_consumption": worst_consumption,
                "violation": violation,
            }
        )

        if violation <= tolerance:
            compact_profit, _ = solve_compact_robust_counterpart(instance)
            return CCGResult(
                production=x,
                profit=profit,
                compact_profit=compact_profit,
                iterations=iteration,
                scenarios=[scenario.copy() for scenario in scenarios],
                worst_case_consumption=worst_consumption,
                violation=violation,
                history=history,
            )

        if any(np.allclose(z, existing, atol=1e-10) for existing in scenarios):
            raise RuntimeError("CCG adversary repeated a violated scenario.")
        scenarios.append(z)

    raise RuntimeError("CCG reached max_iterations.")


def main() -> None:
    result = solve_ccg()
    print(
        {
            "CCG_profit": result.profit,
            "compact_profit": result.compact_profit,
            "production": result.production.round(5).tolist(),
            "iterations": result.iterations,
            "scenarios_generated": len(result.scenarios),
            "final_violation": result.violation,
        }
    )


if __name__ == "__main__":
    main()
