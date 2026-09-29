"""Column generation for the LP relaxation of a one-dimensional cutting-stock model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass
class ColumnGenerationResult:
    lp_objective: float
    patterns: np.ndarray
    pattern_usage: np.ndarray
    iterations: int
    last_reduced_cost: float


def example_instance() -> tuple[int, np.ndarray, np.ndarray]:
    return 10, np.array([3, 4, 5]), np.array([4, 3, 2])


def enumerate_patterns(stock_length: int, item_sizes: np.ndarray) -> np.ndarray:
    maxima = [stock_length // int(s) for s in item_sizes]
    patterns = []
    for values in product(*(range(v + 1) for v in maxima)):
        a = np.asarray(values, dtype=int)
        if a.sum() == 0:
            continue
        if int(item_sizes @ a) <= stock_length:
            patterns.append(a)
    return np.asarray(patterns, dtype=int)


def _solve_master(patterns: np.ndarray, demand: np.ndarray):
    # min 1'y  s.t. A y >= d  -> -A y <= -d
    result = linprog(
        np.ones(patterns.shape[0]),
        A_ub=-patterns.T,
        b_ub=-demand,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Restricted master failed: {result.message}")
    dual_prices = -np.asarray(result.ineqlin.marginals, dtype=float)
    return result, dual_prices


def _pricing(
    stock_length: int, item_sizes: np.ndarray, dual_prices: np.ndarray
) -> tuple[np.ndarray, float]:
    n = item_sizes.size
    result = milp(
        -dual_prices,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.floor(stock_length / item_sizes)),
        constraints=LinearConstraint(item_sizes.reshape(1, -1), -np.inf, stock_length),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Pricing problem failed: {result.message}")
    pattern = np.rint(result.x).astype(int)
    reduced_cost = 1.0 - float(dual_prices @ pattern)
    return pattern, reduced_cost


def solve_column_generation(
    stock_length: int | None = None,
    item_sizes: np.ndarray | None = None,
    demand: np.ndarray | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> ColumnGenerationResult:
    if stock_length is None:
        stock_length, item_sizes, demand = example_instance()
    assert item_sizes is not None and demand is not None

    item_sizes = np.asarray(item_sizes, dtype=int)
    demand = np.asarray(demand, dtype=float)

    # Feasible pure-item starting basis.
    initial = []
    for i, size in enumerate(item_sizes):
        pattern = np.zeros(item_sizes.size, dtype=int)
        pattern[i] = stock_length // int(size)
        initial.append(pattern)
    patterns = np.asarray(initial, dtype=int)

    last_rc = np.nan
    for iteration in range(1, max_iterations + 1):
        master, prices = _solve_master(patterns, demand)
        pattern, last_rc = _pricing(stock_length, item_sizes, prices)

        if last_rc >= -tolerance:
            return ColumnGenerationResult(
                lp_objective=float(master.fun),
                patterns=patterns,
                pattern_usage=np.asarray(master.x),
                iterations=iteration,
                last_reduced_cost=float(last_rc),
            )

        if any(np.array_equal(pattern, p) for p in patterns):
            raise RuntimeError("Pricing returned an existing negative-reduced-cost column.")
        patterns = np.vstack([patterns, pattern])

    raise RuntimeError("Column generation reached max_iterations.")


def full_lp_reference(
    stock_length: int | None = None,
    item_sizes: np.ndarray | None = None,
    demand: np.ndarray | None = None,
) -> float:
    if stock_length is None:
        stock_length, item_sizes, demand = example_instance()
    assert item_sizes is not None and demand is not None
    patterns = enumerate_patterns(stock_length, np.asarray(item_sizes))
    result, _ = _solve_master(patterns, np.asarray(demand, dtype=float))
    return float(result.fun)


def main() -> None:
    result = solve_column_generation()
    print(
        {
            "lp_objective": result.lp_objective,
            "full_lp_reference": full_lp_reference(),
            "columns": int(result.patterns.shape[0]),
            "iterations": result.iterations,
            "last_reduced_cost": result.last_reduced_cost,
        }
    )


if __name__ == "__main__":
    main()
