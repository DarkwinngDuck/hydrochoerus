from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hydrochoerus.model.gap_grid import GapGrid
from hydrochoerus.model.models import Model


def _ignore_step(_model: Model) -> None:
    return None


def _never_cancel() -> bool:
    return False


@dataclass(frozen=True)
class FitContext:
    gap: GapGrid
    observed: np.ndarray
    fixed: frozenset[str] = frozenset()
    on_step: Callable[[Model], None] = _ignore_step
    should_cancel: Callable[[], bool] = _never_cancel
