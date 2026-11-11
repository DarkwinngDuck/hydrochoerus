import numpy as np
import pytest

from hydrochoerus.model.gap_grid import GapGrid, ShapeError

mock_reduced_gaps = np.array([1.0, 0.9, 0.5, 0.0])
mock_reduced_temperatures = np.array([0.2, 0.5, 0.9, 1.1])
mock_gap_squared_derivatives = np.array([-0.1, -0.7, -2.3, 0.0])


def make_grid(
    reduced_temperatures=mock_reduced_temperatures,
    reduced_gaps=mock_reduced_gaps,
    gap_squared_derivatives=mock_gap_squared_derivatives,
):
    return GapGrid(
        reduced_temperatures,
        reduced_gaps,
        gap_squared_derivatives,
    )


def test_should_return_subset_temperatures_by_mask():
    grid = make_grid()
    expected = [0.2, 0.5]
    assert grid.subset(grid.reduced_temperatures < 0.9).reduced_temperatures.tolist() == expected


def test_should_return_subset_gaps_by_mask():
    grid = make_grid()
    expected = [1.0, 0.9]
    assert grid.subset(grid.reduced_temperatures < 0.9).reduced_gaps.tolist() == expected


def test_should_return_subset_derivatives_by_mask():
    grid = make_grid()
    expected = [-0.1, -0.7]
    assert grid.subset(grid.reduced_temperatures < 0.9).gap_squared_derivatives.tolist() == expected


def test_should_subset_gaps_in_the_middle():
    grid = make_grid()
    mask = np.array([False, True, True, False])
    expected = [0.9, 0.5]
    assert grid.subset(mask).reduced_gaps.tolist() == expected


def test_should_return_a_gap_grid():
    grid = make_grid()
    assert isinstance(grid.subset(grid.reduced_temperatures < 0.9), GapGrid)


def test_should_leave_the_original_grid_unchanged():
    grid = make_grid()
    grid.subset(grid.reduced_temperatures < 0.9)
    assert grid.reduced_temperatures.tolist() == mock_reduced_temperatures.tolist()


def test_should_raise_shape_error():
    with pytest.raises(ShapeError):
        GapGrid(np.array([0.2, 0.5]), np.array([1.0]), np.array([0.0, 0.0]))
