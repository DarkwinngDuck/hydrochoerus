from collections.abc import Callable

import numpy as np

from hydrochoerus.core.bcs_gap import (
    bcs_reduced_gap,
    bcs_reduced_gap_squared_derivative,
)
from hydrochoerus.model.gap_grid import GapGrid

type GapGridSolve = Callable[[np.ndarray], GapGrid]
type _GridKey = tuple[tuple[int, ...], bytes]


def solve_gap_grid(reduced_temperatures: np.ndarray) -> GapGrid:
    reduced_gaps = np.array([bcs_reduced_gap(T_reduced) for T_reduced in reduced_temperatures])
    gap_squared_derivatives = np.array(
        [bcs_reduced_gap_squared_derivative(T_reduced) for T_reduced in reduced_temperatures]
    )
    return GapGrid(
        reduced_temperatures=reduced_temperatures,
        reduced_gaps=reduced_gaps,
        gap_squared_derivatives=gap_squared_derivatives,
    )


class GapGridCache:
    def __init__(self, solve: GapGridSolve = solve_gap_grid) -> None:
        self._solve = solve
        self._grids: dict[_GridKey, GapGrid] = {}

    def get(self, reduced_temperatures: np.ndarray, recompute: bool = False) -> GapGrid:
        temperatures = np.array(reduced_temperatures, dtype=np.float64)
        key = (temperatures.shape, temperatures.tobytes())
        if recompute or key not in self._grids:
            self._grids[key] = self._solve(temperatures)
        return self._grids[key]
