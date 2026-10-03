"""Exact branch-and-price for a small bin-packing set-partitioning model.

The restricted master is solved by column generation. Branching uses Ryan-Foster
pair decisions so every branch restriction is enforced by the pricing oracle,
including columns that have not yet been generated. The pricing oracle is
enumerative on purpose: it keeps the example solver-independent and makes the
branch-and-price logic independently verifiable on small instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
import heapq
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class BranchAndPriceResult:
    objective: float
    patterns: np.ndarray
    pattern_usage: np.ndarray
    root_lp_bound: float
    nodes_explored: int
    columns_generated: int
    reference_objective: float


@dataclass
class _Node:
    together: frozenset[tuple[int, int]]
    separate: frozenset[tuple[int, int]]
    patterns: np.ndarray
    depth: int


@dataclass(order=True)
class _QueueItem:
    bound: float
    serial: int
    node: _Node = field(compare=False)


def example_instance() -> tuple[int, np.ndarray]:
    # Fractional LP optimum 2.25, integer optimum 3.
    return 10, np.array([2, 4, 2, 5, 8], dtype=int)


def enumerate_patterns(capacity: int, item_sizes: np.ndarray) -> np.ndarray:
    item_sizes = np.asarray(item_sizes, dtype=int)
    patterns: list[np.ndarray] = []
    for bits in product((0, 1), repeat=item_sizes.size):
        p = np.asarray(bits, dtype=int)
        if p.sum() == 0:
            continue
        if int(item_sizes @ p) <= capacity:
            patterns.append(p)
    return np.asarray(patterns, dtype=int)


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _allowed(
    pattern: np.ndarray,
    together: frozenset[tuple[int, int]],
    separate: frozenset[tuple[int, int]],
) -> bool:
    for i, j in together:
        if int(pattern[i]) != int(pattern[j]):
            return False
    for i, j in separate:
        if pattern[i] and pattern[j]:
            return False
    return True


def _components(n: int, together: frozenset[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in together:
        union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _seed_patterns(
    capacity: int,
    item_sizes: np.ndarray,
    together: frozenset[tuple[int, int]],
    separate: frozenset[tuple[int, int]],
) -> np.ndarray | None:
    n = item_sizes.size
    rows: list[np.ndarray] = []
    for comp in _components(n, together):
        p = np.zeros(n, dtype=int)
        p[comp] = 1
        if int(item_sizes @ p) > capacity or not _allowed(p, together, separate):
            return None
        rows.append(p)
    return np.asarray(rows, dtype=int)


def _deduplicate(patterns: np.ndarray) -> np.ndarray:
    seen: set[tuple[int, ...]] = set()
    rows = []
    for p in patterns:
        key = tuple(int(v) for v in p)
        if key not in seen:
            seen.add(key)
            rows.append(np.asarray(p, dtype=int))
    return np.asarray(rows, dtype=int)


def _initial_patterns(
    node: _Node,
    capacity: int,
    item_sizes: np.ndarray,
) -> np.ndarray | None:
    seed = _seed_patterns(capacity, item_sizes, node.together, node.separate)
    if seed is None:
        return None
    inherited = [
        p for p in node.patterns if _allowed(p, node.together, node.separate)
    ]
    if inherited:
        return _deduplicate(np.vstack([seed, np.asarray(inherited, dtype=int)]))
    return seed


def _solve_master(patterns: np.ndarray, n_items: int):
    result = linprog(
        np.ones(patterns.shape[0]),
        A_eq=patterns.T,
        b_eq=np.ones(n_items),
        bounds=(0.0, None),
        method="highs",
    )
    if result.status == 2:
        return None, None
    if not result.success:
        raise RuntimeError(f"Restricted master failed: {result.message}")
    duals = np.asarray(result.eqlin.marginals, dtype=float)
    return result, duals


def _pricing(
    all_patterns: np.ndarray,
    current_patterns: np.ndarray,
    duals: np.ndarray,
    together: frozenset[tuple[int, int]],
    separate: frozenset[tuple[int, int]],
) -> tuple[np.ndarray | None, float]:
    present = {tuple(int(v) for v in p) for p in current_patterns}
    best_pattern: np.ndarray | None = None
    best_rc = math.inf
    for p in all_patterns:
        key = tuple(int(v) for v in p)
        if key in present or not _allowed(p, together, separate):
            continue
        rc = 1.0 - float(duals @ p)
        if rc < best_rc:
            best_rc = rc
            best_pattern = p
    return best_pattern, float(best_rc)


def _solve_node_lp(
    node: _Node,
    capacity: int,
    item_sizes: np.ndarray,
    all_patterns: np.ndarray,
    tolerance: float,
) -> tuple[object | None, np.ndarray | None, int]:
    patterns = _initial_patterns(node, capacity, item_sizes)
    if patterns is None:
        return None, None, 0
    generated = 0
    while True:
        master, duals = _solve_master(patterns, item_sizes.size)
        if master is None:
            return None, patterns, generated
        candidate, rc = _pricing(
            all_patterns,
            patterns,
            duals,
            node.together,
            node.separate,
        )
        if candidate is None or rc >= -tolerance:
            return master, patterns, generated
        patterns = np.vstack([patterns, candidate])
        generated += 1


def _restricted_integer_solution(patterns: np.ndarray, n_items: int):
    result = milp(
        np.ones(patterns.shape[0]),
        integrality=np.ones(patterns.shape[0]),
        bounds=Bounds(np.zeros(patterns.shape[0]), np.ones(patterns.shape[0])),
        constraints=LinearConstraint(patterns.T, np.ones(n_items), np.ones(n_items)),
        options={"disp": False},
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"Restricted integer master failed: {result.message}")
    return result


def _branch_pair(
    patterns: np.ndarray,
    usage: np.ndarray,
    together: frozenset[tuple[int, int]],
    separate: frozenset[tuple[int, int]],
    tolerance: float,
) -> tuple[int, int] | None:
    n = patterns.shape[1]
    best: tuple[float, int, int] | None = None
    for i, j in combinations(range(n), 2):
        pair = _pair(i, j)
        if pair in together or pair in separate:
            continue
        coassignment = float(usage @ (patterns[:, i] * patterns[:, j]))
        if tolerance < coassignment < 1.0 - tolerance:
            score = abs(coassignment - 0.5)
            candidate = (score, i, j)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return best[1], best[2]


def full_integer_reference(
    capacity: int | None = None,
    item_sizes: np.ndarray | None = None,
) -> float:
    if capacity is None:
        capacity, item_sizes = example_instance()
    assert item_sizes is not None
    item_sizes = np.asarray(item_sizes, dtype=int)
    patterns = enumerate_patterns(capacity, item_sizes)
    result = milp(
        np.ones(patterns.shape[0]),
        integrality=np.ones(patterns.shape[0]),
        bounds=Bounds(np.zeros(patterns.shape[0]), np.ones(patterns.shape[0])),
        constraints=LinearConstraint(
            patterns.T, np.ones(item_sizes.size), np.ones(item_sizes.size)
        ),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Exact set-partitioning master failed: {result.message}")
    return float(result.fun)


def solve_branch_and_price(
    capacity: int | None = None,
    item_sizes: np.ndarray | None = None,
    tolerance: float = 1e-9,
    max_nodes: int = 1000,
) -> BranchAndPriceResult:
    if capacity is None:
        capacity, item_sizes = example_instance()
    assert item_sizes is not None
    item_sizes = np.asarray(item_sizes, dtype=int)
    if item_sizes.ndim != 1 or item_sizes.size == 0:
        raise ValueError("item_sizes must be a non-empty one-dimensional array")
    if capacity <= 0 or np.any(item_sizes <= 0) or np.any(item_sizes > capacity):
        raise ValueError("item sizes must be positive and no larger than capacity")

    all_patterns = enumerate_patterns(capacity, item_sizes)
    root_seed = _seed_patterns(capacity, item_sizes, frozenset(), frozenset())
    assert root_seed is not None
    root = _Node(frozenset(), frozenset(), root_seed, 0)

    queue: list[_QueueItem] = [_QueueItem(-math.inf, 0, root)]
    serial = 0
    incumbent = math.inf
    incumbent_patterns: np.ndarray | None = None
    incumbent_usage: np.ndarray | None = None
    root_lp_bound = math.nan
    nodes_explored = 0
    columns_generated = 0

    while queue:
        if nodes_explored >= max_nodes:
            raise RuntimeError("Branch-and-price reached max_nodes before proving optimality")
        item = heapq.heappop(queue)
        node = item.node
        master, patterns, generated = _solve_node_lp(
            node, capacity, item_sizes, all_patterns, tolerance
        )
        nodes_explored += 1
        columns_generated += generated
        if master is None or patterns is None:
            continue
        bound = float(master.fun)
        if node.depth == 0:
            root_lp_bound = bound
        if bound >= incumbent - tolerance:
            continue

        integer_master = _restricted_integer_solution(patterns, item_sizes.size)
        if integer_master is not None and float(integer_master.fun) < incumbent - tolerance:
            incumbent = float(integer_master.fun)
            incumbent_patterns = patterns.copy()
            incumbent_usage = np.rint(integer_master.x)

        usage = np.asarray(master.x, dtype=float)
        if np.all(np.abs(usage - np.rint(usage)) <= tolerance):
            integer_obj = float(np.rint(usage).sum())
            if integer_obj < incumbent - tolerance:
                incumbent = integer_obj
                incumbent_patterns = patterns.copy()
                incumbent_usage = np.rint(usage)
            continue

        branch = _branch_pair(
            patterns, usage, node.together, node.separate, tolerance
        )
        if branch is None:
            raise RuntimeError(
                "Fractional master solution had no fractional Ryan-Foster pair"
            )
        pair = _pair(*branch)

        serial += 1
        heapq.heappush(
            queue,
            _QueueItem(
                bound,
                serial,
                _Node(
                    frozenset(set(node.together) | {pair}),
                    node.separate,
                    patterns.copy(),
                    node.depth + 1,
                ),
            ),
        )
        serial += 1
        heapq.heappush(
            queue,
            _QueueItem(
                bound,
                serial,
                _Node(
                    node.together,
                    frozenset(set(node.separate) | {pair}),
                    patterns.copy(),
                    node.depth + 1,
                ),
            ),
        )

    if incumbent_patterns is None or incumbent_usage is None:
        raise RuntimeError("Branch-and-price did not find an integer solution")

    reference = full_integer_reference(capacity, item_sizes)
    return BranchAndPriceResult(
        objective=float(incumbent),
        patterns=incumbent_patterns,
        pattern_usage=incumbent_usage,
        root_lp_bound=float(root_lp_bound),
        nodes_explored=nodes_explored,
        columns_generated=columns_generated,
        reference_objective=reference,
    )


def main() -> None:
    result = solve_branch_and_price()
    print(
        {
            "objective": result.objective,
            "reference_objective": result.reference_objective,
            "root_lp_bound": result.root_lp_bound,
            "nodes_explored": result.nodes_explored,
            "columns_generated": result.columns_generated,
        }
    )


if __name__ == "__main__":
    main()
