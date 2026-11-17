from typing import Literal

import numpy as np
from pydantic import Field

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.specific_heat import CesOnGrid, ces_on_grid
from hydrochoerus.model.gap_grid import GapGrid
from hydrochoerus.model.models import Gap, Model, ModelType


class SingleAlpha(Model):
    type: Literal[ModelType.SINGLE_ALPHA] = ModelType.SINGLE_ALPHA
    alpha: Gap = Field(ALPHA_BCS, ge=0.01, le=20)

    def curve(self, gap: GapGrid, *, ces: CesOnGrid = ces_on_grid) -> np.ndarray:
        return ces(gap, self.alpha)
