from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ShapeError(ValueError):
    """Shapes are not equal."""


@dataclass(frozen=True)
class GapGrid:
    reduced_temperatures: NDArray[np.float64]
    reduced_gaps: NDArray[np.float64]
    gap_squared_derivatives: NDArray[np.float64]

    def __post_init__(self):
        if not (
            self.reduced_temperatures.shape
            == self.reduced_gaps.shape
            == self.gap_squared_derivatives.shape
        ):
            raise ShapeError(
                "GapGrid temperatures, gaps and derivatives must have the same shape, got "
                f"{self.reduced_temperatures.shape}, {self.reduced_gaps.shape}, "
                f"{self.gap_squared_derivatives.shape}"
            )

    def subset(self, mask: NDArray[np.bool_]) -> GapGrid:
        return GapGrid(
            reduced_temperatures=self.reduced_temperatures[mask],
            reduced_gaps=self.reduced_gaps[mask],
            gap_squared_derivatives=self.gap_squared_derivatives[mask],
        )
