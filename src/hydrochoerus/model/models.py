from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Self

import numpy as np
from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict

from hydrochoerus.core.specific_heat import CesOnGrid, ces_on_grid
from hydrochoerus.model.gap_grid import GapGrid


class ParameterType(StrEnum):
    GAP = "gap"
    WEIGHT = "weight"


class ModelType(StrEnum):
    SINGLE_ALPHA = "single-alpha"
    TWO_ALPHA = "two-alpha"


Gap = Annotated[float, ParameterType.GAP]
Weight = Annotated[float, ParameterType.WEIGHT]


def _has_ptype(metadata: list) -> bool:
    return any(isinstance(m, ParameterType) for m in metadata)


class Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return tuple(name for name, f in cls.model_fields.items() if _has_ptype(f.metadata))

    @classmethod
    def bounds(cls, free: Sequence[str]) -> tuple[list[float], list[float]]:
        meta = [cls.model_fields[name].metadata for name in free]
        low = [float(next(m.ge for m in ms if isinstance(m, Ge))) for ms in meta]
        high = [float(next(m.le for m in ms if isinstance(m, Le))) for ms in meta]
        return low, high

    @classmethod
    def ptype(cls, name: str) -> ParameterType:
        return next(m for m in cls.model_fields[name].metadata if isinstance(m, ParameterType))

    def vector(self, free: Sequence[str]) -> list[float]:
        return [getattr(self, name) for name in free]

    def with_vector(self, free: Sequence[str], values: Sequence[float]) -> Self:
        update = dict(zip(free, map(float, values), strict=True))
        return self.model_validate(self.model_dump() | update)

    def curve(self, gap: GapGrid, *, ces: CesOnGrid = ces_on_grid) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no curve")

    def normalised(self) -> Self:
        return self
