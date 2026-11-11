from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class ExperimentalData:
    """t [T/T_c], y [c_el(T)/(gamma_n T_c)]"""

    name: str
    path: Path
    t: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        t = np.array(self.t, dtype=float)
        y = np.array(self.y, dtype=float)

        if t.ndim != 1 or t.shape != y.shape:
            raise ValueError(f"{self.name}: t and y must have the same shape")

        t.setflags(write=False)
        y.setflags(write=False)

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "path", Path(self.path))
