import numpy as np

from hydrochoerus.core.bcs_gap import (
    bcs_reduced_gap,
    bcs_reduced_gap_squared_derivative,
)
from hydrochoerus.core.gap_cache import solve_gap_grid
from hydrochoerus.model.gap_grid import GapGrid


def test_should_return_gap_grid():
    grid = solve_gap_grid(np.array([0.3, 1.2]))
    expected = GapGrid
    assert isinstance(grid, expected)


def test_should_return_correct_gap_grid():
    reduced_temperatures = np.array([0.3, 0.8, 0.95, 1.0, 1.2])
    expected = (
        [0.3, 0.8, 0.95, 1.0, 1.2],
        [bcs_reduced_gap(T_reduced) for T_reduced in reduced_temperatures],
        [bcs_reduced_gap_squared_derivative(T_reduced) for T_reduced in reduced_temperatures],
    )
    grid = solve_gap_grid(reduced_temperatures)

    assert (
        grid.reduced_temperatures.tolist(),
        grid.reduced_gaps.tolist(),
        grid.gap_squared_derivatives.tolist(),
    ) == expected
