import numpy as np
import pytest

from hydrochoerus.model.gap_grid import GapGrid


@pytest.fixture
def hand_grid() -> GapGrid:
    return GapGrid(np.array([0.5, 1.2]), np.array([0.9, 0.0]), np.array([-1.0, 0.0]))
