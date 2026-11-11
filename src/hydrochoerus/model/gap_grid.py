from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hydrochoerus.utils.readonly_columns import ReadOnlyColumns


@dataclass(frozen=True)
class GapGrid(ReadOnlyColumns):
    reduced_temperatures: NDArray[np.float64]
    reduced_gaps: NDArray[np.float64]
    gap_squared_derivatives: NDArray[np.float64]

    def subset(self, mask: NDArray[np.bool_]) -> GapGrid:
        return GapGrid(
            reduced_temperatures=self.reduced_temperatures[mask],
            reduced_gaps=self.reduced_gaps[mask],
            gap_squared_derivatives=self.gap_squared_derivatives[mask],
        )
