"""Progressive Hedging for a small two-stage stochastic production problem.

The first-stage quantity must be nonanticipative across demand scenarios.
Scenario recourse penalizes shortage and excess inventory. Scenario subproblems
are one-dimensional convex problems, solved with bounded scalar minimization.

A compact extensive-form LP provides an independent reference optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize_scalar


@dataclass(frozen=True)
class StochasticProductionInstance:
    demand: np.ndarray
    probability: np.ndarray
    first_stage_cost: float
    shortage_cost: float
    holding_cost: float
    upper_bound: float


@dataclass
class ProgressiveHedgingResult:
    consensus_quantity: float
    objective: float
    reference_objective: float
    reference_quantity: float
    iterations: int
    residual: float
    history: list[dict[str, float]]


def example_instance() -> StochasticProductionInstance:
    return StochasticProductionInstance(
        demand=np.array([6.0, 10.0, 14.0]),
        probability=np.array([1.0 / 3.0] * 3),
        first_stage_cost=2.0,
        shortage_cost=7.0,
        holding_cost=1.0,
        upper_bound=18.0,
    )


def scenario_cost(x: float, demand: float, instance: StochasticProductionInstance) -> float:
    shortage = max(demand - x, 0.0)
    excess = max(x - demand, 0.0)
    return (
        instance.first_stage_cost * x
        + instance.shortage_cost * shortage
        + instance.holding_cost * excess
    )


def expected_cost(x: float, instance: StochasticProductionInstance) -> float:
    return float(
        sum(
            p * scenario_cost(x, d, instance)
            for p, d in zip(instance.probability, instance.demand)
        )
    )


def extensive_form_reference(
    instance: StochasticProductionInstance | None = None,
) -> tuple[float, float]:
    instance = instance or example_instance()
    s = instance.demand.size

    # Variables: x, shortage_s, excess_s
    c = np.concatenate(
        [
            [instance.first_stage_cost],
            instance.probability * instance.shortage_cost,
            instance.probability * instance.holding_cost,
        ]
    )

    a_eq = np.zeros((s, 1 + 2 * s))
    b_eq = instance.demand.copy()
    for k in range(s):
        # x + shortage_k - excess_k = demand_k
        a_eq[k, 0] = 1.0
        a_eq[k, 1 + k] = 1.0
        a_eq[k, 1 + s + k] = -1.0

    bounds = [(0.0, instance.upper_bound)] + [(0.0, None)] * (2 * s)
    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Extensive-form LP failed: {result.message}")
    return float(result.fun), float(result.x[0])


def solve_progressive_hedging(
    instance: StochasticProductionInstance | None = None,
    rho: float = 4.0,
    tolerance: float = 1e-6,
    max_iterations: int = 500,
) -> ProgressiveHedgingResult:
    instance = instance or example_instance()
    if rho <= 0:
        raise ValueError("rho must be positive")

    s = instance.demand.size
    if not np.isclose(instance.probability.sum(), 1.0):
        raise ValueError("scenario probabilities must sum to one")
    if not np.allclose(instance.probability, np.full(s, 1.0 / s)):
        raise ValueError("reference implementation currently assumes equal probabilities")

    # Scenario-optimal starts.
    x = np.empty(s)
    for k, demand in enumerate(instance.demand):
        result = minimize_scalar(
            lambda value: scenario_cost(value, demand, instance),
            bounds=(0.0, instance.upper_bound),
            method="bounded",
            options={"xatol": 1e-12},
        )
        x[k] = result.x

    multipliers = np.zeros(s)
    consensus = float(instance.probability @ x)
    history: list[dict[str, float]] = []

    for iteration in range(1, max_iterations + 1):
        previous_consensus = consensus

        for k, demand in enumerate(instance.demand):
            w = multipliers[k]
            result = minimize_scalar(
                lambda value: (
                    scenario_cost(value, demand, instance)
                    + w * value
                    + 0.5 * rho * (value - previous_consensus) ** 2
                ),
                bounds=(0.0, instance.upper_bound),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if not result.success:
                raise RuntimeError("Progressive Hedging scenario solve failed.")
            x[k] = result.x

        consensus = float(instance.probability @ x)
        multipliers += rho * (x - consensus)
        residual = float(np.max(np.abs(x - consensus)))
        history.append(
            {
                "iteration": float(iteration),
                "consensus": consensus,
                "residual": residual,
                "objective": expected_cost(consensus, instance),
            }
        )

        if residual <= tolerance:
            break
    else:
        raise RuntimeError("Progressive Hedging reached max_iterations.")

    reference_obj, reference_x = extensive_form_reference(instance)
    return ProgressiveHedgingResult(
        consensus_quantity=consensus,
        objective=expected_cost(consensus, instance),
        reference_objective=reference_obj,
        reference_quantity=reference_x,
        iterations=iteration,
        residual=residual,
        history=history,
    )


def main() -> None:
    result = solve_progressive_hedging()
    print(
        {
            "PH_quantity": result.consensus_quantity,
            "PH_objective": result.objective,
            "reference_quantity": result.reference_quantity,
            "reference_objective": result.reference_objective,
            "iterations": result.iterations,
            "residual": result.residual,
        }
    )


if __name__ == "__main__":
    main()
