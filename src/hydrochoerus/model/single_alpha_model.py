from typing import Literal

from pydantic import Field

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.model.models import Gap, Model, ModelType


class SingleAlpha(Model):
    type: Literal[ModelType.SINGLE_ALPHA] = ModelType.SINGLE_ALPHA
    alpha: Gap = Field(ALPHA_BCS, ge=0.01, le=20)
