import numpy as np

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.bcs_gap import (
    bcs_reduced_gap,
    bcs_reduced_gap_squared_derivative,
)
from hydrochoerus.core.gap_cache import solve_gap_grid
from hydrochoerus.core.specific_heat import ces, ces_from_gap, ces_on_grid
from hydrochoerus.model.gap_grid import GapGrid


def test_should_be_normal_state_above_tc_whatever_the_gap():
    expected = 1.2
    assert ces_from_gap(1.2, gap_reduced=0.7, gap_squared_derivative=-3.0, alpha=ALPHA_BCS) == (
        expected
    )


def test_should_equal_ces_when_given_the_solved_gap():
    T_reduced = 0.5
    expected = ces(T_reduced, ALPHA_BCS)
    assert (
        ces_from_gap(
            T_reduced,
            bcs_reduced_gap(T_reduced),
            bcs_reduced_gap_squared_derivative(T_reduced),
            ALPHA_BCS,
        )
        == expected
    )


def test_should_use_the_gap_from_the_grid():
    T_reduced = 0.5
    grid = GapGrid(np.array([T_reduced]), np.array([0.9]), np.array([-1.0]))
    expected = [ces_from_gap(T_reduced, 0.9, -1.0, ALPHA_BCS)]
    assert ces_on_grid(grid, ALPHA_BCS).tolist() == expected


def test_should_return_one_value_per_row_in_grid_order():
    grid = GapGrid(np.array([1.5, 1.0, 1.2]), np.zeros(3), np.zeros(3))
    expected = [1.5, 1.0, 1.2]
    assert ces_on_grid(grid, ALPHA_BCS).tolist() == expected


def test_should_equal_ces_on_a_solved_grid():
    reduced_temperatures = np.array([0.4, 0.8, 1.1])
    expected = [ces(T_reduced, ALPHA_BCS) for T_reduced in reduced_temperatures]
    assert ces_on_grid(solve_gap_grid(reduced_temperatures), ALPHA_BCS).tolist() == expected
