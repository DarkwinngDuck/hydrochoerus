from typing import Literal, Self

from pydantic import Field

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.model.models import Gap, Model, ModelType, Weight


class TwoAlpha(Model):
    type: Literal[ModelType.TWO_ALPHA] = ModelType.TWO_ALPHA
    alpha1: Gap = Field(ALPHA_BCS, ge=0.01, le=20)
    alpha2: Gap = Field(ALPHA_BCS, ge=0.01, le=20)
    x: Weight = Field(1, ge=0, le=1)

    def normalised(self) -> Self:
        if self.alpha1 >= self.alpha2:
            return self
        return self.model_copy(
            update={"alpha1": self.alpha2, "alpha2": self.alpha1, "x": 1.0 - self.x}
        )
