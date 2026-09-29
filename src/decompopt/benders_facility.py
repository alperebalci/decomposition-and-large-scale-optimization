"""Classical Benders decomposition for a small capacitated facility-location model.

The master chooses binary facility openings. The subproblem assigns customer
mass continuously to open facilities. The LP dual produces an optimality cut.

The example is deliberately small so Benders can be checked against one
extensive-form MILP solved by SciPy/HiGHS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class FacilityInstance:
    fixed_cost: np.ndarray
    capacity: np.ndarray
    assignment_cost: np.ndarray

    @property
    def n_facilities(self) -> int:
        return int(self.fixed_cost.size)

    @property
    def n_customers(self) -> int:
        return int(self.assignment_cost.shape[0])


@dataclass
class BendersResult:
    objective: float
    open_facilities: np.ndarray
    lower_bound: float
    upper_bound: float
    iterations: int
    history: list[dict[str, float]]


def example_instance() -> FacilityInstance:
    return FacilityInstance(
        fixed_cost=np.array([7.0, 8.0, 10.0]),
        capacity=np.array([2.0, 3.0, 4.0]),
        assignment_cost=np.array(
            [
                [2.0, 5.0, 4.0],
                [3.0, 2.0, 6.0],
                [4.0, 3.0, 2.0],
                [6.0, 4.0, 3.0],
            ]
        ),
    )


def solve_assignment_subproblem(
    instance: FacilityInstance, open_facilities: np.ndarray
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Solve assignment recourse and return objective, allocation and dual cut.

    Returns
    -------
    objective
        Optimal assignment cost.
    allocation
        Customer-facility assignment matrix.
    intercept
        Constant term of the Benders optimality cut.
    coefficients
        Coefficients multiplying the binary opening variables.
    """

    n, m = instance.n_customers, instance.n_facilities
    c = instance.assignment_cost.reshape(-1)

    a_eq = np.zeros((n, n * m))
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    b_eq = np.ones(n)

    a_ub = np.zeros((m, n * m))
    for j in range(m):
        for i in range(n):
            a_ub[j, i * m + j] = 1.0
    b_ub = instance.capacity * open_facilities

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Assignment subproblem failed: {result.message}")

    # For min c'x, Ax <= b, Ex = d, HiGHS marginals are valid dual
    # multipliers. Hence q(y) >= d'lambda + (capacity*y)'mu.
    equality_duals = np.asarray(result.eqlin.marginals, dtype=float)
    capacity_duals = np.asarray(result.ineqlin.marginals, dtype=float)
    intercept = float(equality_duals.sum())
    coefficients = instance.capacity * capacity_duals

    return (
        float(result.fun),
        np.asarray(result.x).reshape(n, m),
        intercept,
        coefficients,
    )


def _solve_master(
    instance: FacilityInstance, cuts: list[tuple[float, np.ndarray]]
) -> tuple[np.ndarray, float, float]:
    m = instance.n_facilities
    c = np.concatenate([instance.fixed_cost, [1.0]])
    integrality = np.concatenate([np.ones(m), [0]])
    bounds = Bounds(
        np.concatenate([np.zeros(m), [0.0]]),
        np.concatenate([np.ones(m), [np.inf]]),
    )

    rows = []
    lower = []
    upper = []

    # Keep every master solution recourse-feasible.
    rows.append(np.concatenate([instance.capacity, [0.0]]))
    lower.append(float(instance.n_customers))
    upper.append(np.inf)

    # theta >= intercept + beta'y  <=> beta'y - theta <= -intercept
    for intercept, beta in cuts:
        rows.append(np.concatenate([beta, [-1.0]]))
        lower.append(-np.inf)
        upper.append(-intercept)

    constraint = LinearConstraint(
        np.vstack(rows), np.asarray(lower), np.asarray(upper)
    )
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Master problem failed: {result.message}")

    y = np.rint(result.x[:m]).astype(int)
    theta = float(result.x[m])
    return y, theta, float(result.fun)


def solve_benders(
    instance: FacilityInstance | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> BendersResult:
    instance = instance or example_instance()
    cuts: list[tuple[float, np.ndarray]] = []
    incumbent = np.inf
    incumbent_y: np.ndarray | None = None
    history: list[dict[str, float]] = []
    lower_bound = -np.inf

    for iteration in range(1, max_iterations + 1):
        y, theta, master_obj = _solve_master(instance, cuts)
        lower_bound = master_obj

        recourse, _, intercept, beta = solve_assignment_subproblem(instance, y)
        candidate = float(instance.fixed_cost @ y + recourse)
        if candidate < incumbent:
            incumbent = candidate
            incumbent_y = y.copy()

        gap = max(0.0, incumbent - lower_bound)
        history.append(
            {
                "iteration": float(iteration),
                "lower_bound": float(lower_bound),
                "upper_bound": float(incumbent),
                "gap": float(gap),
                "theta": float(theta),
                "recourse": float(recourse),
            }
        )

        # Add the cut generated at the current design before testing convergence.
        cuts.append((intercept, beta.copy()))
        if gap <= tolerance:
            break
    else:
        raise RuntimeError("Benders decomposition reached max_iterations.")

    if incumbent_y is None:
        raise RuntimeError("No incumbent was generated.")

    return BendersResult(
        objective=float(incumbent),
        open_facilities=incumbent_y,
        lower_bound=float(lower_bound),
        upper_bound=float(incumbent),
        iterations=iteration,
        history=history,
    )


def solve_extensive_form(
    instance: FacilityInstance | None = None,
) -> tuple[float, np.ndarray]:
    instance = instance or example_instance()
    n, m = instance.n_customers, instance.n_facilities
    n_x = n * m
    c = np.concatenate([instance.fixed_cost, instance.assignment_cost.reshape(-1)])

    rows = []
    lb = []
    ub = []

    for i in range(n):
        row = np.zeros(m + n_x)
        row[m + i * m : m + (i + 1) * m] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    for j in range(m):
        row = np.zeros(m + n_x)
        row[j] = -instance.capacity[j]
        for i in range(n):
            row[m + i * m + j] = 1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    result = milp(
        c,
        integrality=np.concatenate([np.ones(m), np.zeros(n_x)]),
        bounds=Bounds(np.zeros(m + n_x), np.ones(m + n_x)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub)),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Extensive form failed: {result.message}")
    return float(result.fun), np.rint(result.x[:m]).astype(int)


def main() -> None:
    benders = solve_benders()
    extensive_obj, extensive_y = solve_extensive_form()
    print(
        {
            "benders_objective": benders.objective,
            "benders_open": benders.open_facilities.tolist(),
            "iterations": benders.iterations,
            "final_gap": benders.upper_bound - benders.lower_bound,
            "extensive_objective": extensive_obj,
            "extensive_open": extensive_y.tolist(),
        }
    )


if __name__ == "__main__":
    main()
