from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydrochoerus.utils.readonly_columns import ReadOnlyColumns, ShapeError


@dataclass(frozen=True, eq=False)
class ExperimentalData(ReadOnlyColumns):
    """t = T/Tc, y = C_el/(gamma_n·Tc)."""

    name: str
    path: Path
    t: np.ndarray
    y: np.ndarray

    def validate(self) -> None:
        if self.t.ndim != 1:
            raise ShapeError(f"{self.name}: t and y must be 1-D, got shape {self.t.shape}")
