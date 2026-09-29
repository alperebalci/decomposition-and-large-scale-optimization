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
