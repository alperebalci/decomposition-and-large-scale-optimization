"""Consensus ADMM for separable strongly convex quadratic optimization.

The benchmark solves

    minimize_z  sum_i (0.5 z^T Q_i z + c_i^T z)

through local copies x_i with consensus constraints x_i = z.  The resulting
two-block consensus ADMM iteration exposes the decomposition mechanics without
requiring a commercial solver.  A closed-form centralized optimum is computed
independently for verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


Array = np.ndarray


@dataclass(frozen=True)
class ConsensusQuadraticProblem:
    """Separable quadratic agents coupled only by a shared consensus vector."""

    q_matrices: Array
    c_vectors: Array

    def validated(self) -> "ConsensusQuadraticProblem":
        q = np.asarray(self.q_matrices, dtype=float)
        c = np.asarray(self.c_vectors, dtype=float)

        if q.ndim != 3:
            raise ValueError("q_matrices must have shape (n_agents, n, n)")
        if c.ndim != 2:
            raise ValueError("c_vectors must have shape (n_agents, n)")
        if q.shape[0] != c.shape[0] or q.shape[1] != q.shape[2] or q.shape[1] != c.shape[1]:
            raise ValueError("incompatible q_matrices and c_vectors shapes")
        if q.shape[0] < 2:
            raise ValueError("consensus ADMM requires at least two agents")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)):
            raise ValueError("problem data must be finite")

        for idx, matrix in enumerate(q):
            if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
                raise ValueError(f"Q[{idx}] must be symmetric")
            try:
                cho_factor(matrix, lower=True, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Q[{idx}] must be positive definite") from exc

        return ConsensusQuadraticProblem(q, c)


@dataclass(frozen=True)
class ADMMIteration:
    iteration: int
    objective: float
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float


@dataclass(frozen=True)
class ConsensusADMMResult:
    consensus: Array
    local_solutions: Array
    objective: float
    reference_solution: Array
    reference_objective: float
    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float
    history: tuple[ADMMIteration, ...]

    @property
    def objective_gap(self) -> float:
        return self.objective - self.reference_objective

    @property
    def reference_error(self) -> float:
        return float(np.linalg.norm(self.consensus - self.reference_solution))


def objective(problem: ConsensusQuadraticProblem, z: Array) -> float:
    """Evaluate the original centralized objective at a consensus vector."""
    p = problem.validated()
    z = np.asarray(z, dtype=float)
    if z.shape != (p.c_vectors.shape[1],):
        raise ValueError("z has incompatible shape")

    value = 0.0
    for q_i, c_i in zip(p.q_matrices, p.c_vectors, strict=True):
        value += 0.5 * float(z @ q_i @ z) + float(c_i @ z)
    return value


def solve_centralized(problem: ConsensusQuadraticProblem) -> tuple[Array, float]:
    """Solve the strongly convex centralized reference problem exactly."""
    p = problem.validated()
    q_sum = np.sum(p.q_matrices, axis=0)
    c_sum = np.sum(p.c_vectors, axis=0)
    z_star = np.linalg.solve(q_sum, -c_sum)
    return z_star, objective(p, z_star)


def solve_consensus_admm(
    problem: ConsensusQuadraticProblem,
    *,
    rho: float = 1.0,
    abs_tol: float = 1e-7,
    rel_tol: float = 1e-6,
    max_iterations: int = 5000,
) -> ConsensusADMMResult:
    """Solve a separable quadratic consensus problem with scaled-form ADMM.

    Each agent solves one local positive-definite linear system.  The global
    variable is then the average of the shifted local copies, followed by the
    scaled dual update.  Factorizations of Q_i + rho I are cached once.
    """
    p = problem.validated()
    if rho <= 0.0 or not np.isfinite(rho):
        raise ValueError("rho must be finite and positive")
    if abs_tol <= 0.0 or rel_tol <= 0.0:
        raise ValueError("tolerances must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    n_agents, dimension = p.c_vectors.shape
    local = np.zeros((n_agents, dimension), dtype=float)
    consensus = np.zeros(dimension, dtype=float)
    scaled_dual = np.zeros_like(local)

    identity = np.eye(dimension)
    factors = [
        cho_factor(q_i + rho * identity, lower=True, check_finite=False)
        for q_i in p.q_matrices
    ]

    history: list[ADMMIteration] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        for i in range(n_agents):
            rhs = rho * (consensus - scaled_dual[i]) - p.c_vectors[i]
            local[i] = cho_solve(factors[i], rhs, check_finite=False)

        previous_consensus = consensus.copy()
        consensus = np.mean(local + scaled_dual, axis=0)
        scaled_dual += local - consensus

        primal_residual = float(np.linalg.norm(local - consensus))
        dual_residual = float(
            rho * np.sqrt(n_agents) * np.linalg.norm(consensus - previous_consensus)
        )

        primal_tolerance = float(
            np.sqrt(n_agents * dimension) * abs_tol
            + rel_tol
            * max(
                np.linalg.norm(local),
                np.sqrt(n_agents) * np.linalg.norm(consensus),
            )
        )
        dual_tolerance = float(
            np.sqrt(n_agents * dimension) * abs_tol
            + rel_tol * rho * np.linalg.norm(scaled_dual)
        )

        history.append(
            ADMMIteration(
                iteration=iteration,
                objective=objective(p, consensus),
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                primal_tolerance=primal_tolerance,
                dual_tolerance=dual_tolerance,
            )
        )

        if primal_residual <= primal_tolerance and dual_residual <= dual_tolerance:
            converged = True
            break

    reference_solution, reference_objective = solve_centralized(p)
    final = history[-1]
    return ConsensusADMMResult(
        consensus=consensus.copy(),
        local_solutions=local.copy(),
        objective=final.objective,
        reference_solution=reference_solution,
        reference_objective=reference_objective,
        iterations=final.iteration,
        converged=converged,
        primal_residual=final.primal_residual,
        dual_residual=final.dual_residual,
        history=tuple(history),
    )


def example_problem() -> ConsensusQuadraticProblem:
    """Deterministic four-agent benchmark with coupled local quadratics."""
    q_matrices = np.array(
        [
            [[4.0, 0.6, 0.2], [0.6, 2.5, 0.1], [0.2, 0.1, 1.8]],
            [[2.8, 0.2, 0.0], [0.2, 3.6, 0.4], [0.0, 0.4, 2.2]],
            [[3.2, 0.1, 0.5], [0.1, 2.2, 0.2], [0.5, 0.2, 3.4]],
            [[2.4, 0.3, 0.1], [0.3, 3.0, 0.2], [0.1, 0.2, 2.6]],
        ],
        dtype=float,
    )
    c_vectors = np.array(
        [
            [-1.2, 0.4, -0.6],
            [-0.5, -0.8, 0.3],
            [0.2, -1.0, -0.4],
            [-0.7, 0.1, -0.9],
        ],
        dtype=float,
    )
    return ConsensusQuadraticProblem(q_matrices, c_vectors).validated()


def main() -> None:
    result = solve_consensus_admm(example_problem(), rho=1.0)
    print(f"converged={result.converged}")
    print(f"iterations={result.iterations}")
    print(f"objective={result.objective:.12f}")
    print(f"reference_objective={result.reference_objective:.12f}")
    print(f"objective_gap={result.objective_gap:.3e}")
    print(f"reference_error={result.reference_error:.3e}")
    print(f"primal_residual={result.primal_residual:.3e}")
    print(f"dual_residual={result.dual_residual:.3e}")
    print("consensus=", np.array2string(result.consensus, precision=8))


if __name__ == "__main__":
    main()
