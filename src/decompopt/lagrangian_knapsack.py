"""Lagrangian relaxation of a binary knapsack capacity constraint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class LagrangianResult:
    exact_objective: float
    best_dual_bound: float
    best_feasible_value: float
    duality_gap: float
    iterations: int
    multiplier: float


def example_instance() -> tuple[np.ndarray, np.ndarray, float]:
    profits = np.array([18.0, 14.0, 12.0, 10.0, 7.0, 6.0])
    weights = np.array([8.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    capacity = 15.0
    return profits, weights, capacity


def exact_knapsack(
    profits: np.ndarray, weights: np.ndarray, capacity: float
) -> tuple[float, np.ndarray]:
    n = profits.size
    result = milp(
        -profits,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(weights.reshape(1, -1), -np.inf, capacity),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Exact knapsack failed: {result.message}")
    return float(-result.fun), np.rint(result.x).astype(int)


def _greedy_repair(
    selected: np.ndarray, profits: np.ndarray, weights: np.ndarray, capacity: float
) -> np.ndarray:
    x = selected.astype(int).copy()
    while float(weights @ x) > capacity:
        active = np.flatnonzero(x)
        remove = active[np.argmin(profits[active] / weights[active])]
        x[remove] = 0

    for j in np.argsort(-(profits / weights)):
        if x[j] == 0 and float(weights @ x + weights[j]) <= capacity:
            x[j] = 1
    return x


def solve_lagrangian(
    profits: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    capacity: float | None = None,
    max_iterations: int = 400,
    initial_multiplier: float = 0.0,
) -> LagrangianResult:
    if profits is None:
        profits, weights, capacity = example_instance()
    assert weights is not None and capacity is not None
    profits = np.asarray(profits, dtype=float)
    weights = np.asarray(weights, dtype=float)

    exact, _ = exact_knapsack(profits, weights, capacity)
    multiplier = float(initial_multiplier)
    best_dual = np.inf
    best_feasible = 0.0

    for k in range(1, max_iterations + 1):
        reduced = profits - multiplier * weights
        selected = (reduced > 0.0).astype(int)
        dual_value = float(multiplier * capacity + np.maximum(reduced, 0.0).sum())
        best_dual = min(best_dual, dual_value)

        repaired = _greedy_repair(selected, profits, weights, capacity)
        best_feasible = max(best_feasible, float(profits @ repaired))

        subgradient = float(weights @ selected - capacity)
        step = 2.0 / np.sqrt(k)
        multiplier = max(0.0, multiplier + step * subgradient)

    return LagrangianResult(
        exact_objective=exact,
        best_dual_bound=float(best_dual),
        best_feasible_value=float(best_feasible),
        duality_gap=float(best_dual - exact),
        iterations=max_iterations,
        multiplier=float(multiplier),
    )


def main() -> None:
    result = solve_lagrangian()
    print(result)


if __name__ == "__main__":
    main()
