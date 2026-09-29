import numpy as np

from decompopt.benders_facility import (
    example_instance,
    solve_benders,
    solve_extensive_form,
)
from decompopt.column_generation import full_lp_reference, solve_column_generation
from decompopt.lagrangian_knapsack import solve_lagrangian


def test_benders_matches_extensive_form():
    instance = example_instance()
    benders = solve_benders(instance)
    extensive_obj, _ = solve_extensive_form(instance)
    assert np.isclose(benders.objective, extensive_obj, atol=1e-7)
    assert benders.upper_bound + 1e-7 >= benders.lower_bound
    assert benders.iterations >= 1


def test_column_generation_matches_full_pattern_lp():
    cg = solve_column_generation()
    reference = full_lp_reference()
    assert np.isclose(cg.lp_objective, reference, atol=1e-8)
    assert cg.last_reduced_cost >= -1e-8


def test_lagrangian_bound_is_valid():
    result = solve_lagrangian()
    assert result.best_dual_bound + 1e-8 >= result.exact_objective
    assert result.best_feasible_value <= result.exact_objective + 1e-8
    assert result.duality_gap >= -1e-8


from decompopt.progressive_hedging import solve_progressive_hedging
from decompopt.column_constraint_generation import solve_ccg


def test_progressive_hedging_matches_extensive_form():
    result = solve_progressive_hedging()
    assert result.residual <= 1e-6
    assert abs(result.objective - result.reference_objective) <= 1e-5
    assert abs(result.consensus_quantity - result.reference_quantity) <= 1e-3


def test_ccg_matches_compact_robust_counterpart():
    result = solve_ccg()
    assert result.violation <= 1e-8
    assert np.isclose(result.profit, result.compact_profit, atol=1e-8)
    assert result.iterations >= 2
