from dataclasses import dataclass, fields

import numpy as np


class ShapeError(ValueError):
    """Shapes are not equal."""


@dataclass(frozen=True)
class ReadOnlyColumns:
    def __post_init__(self):
        self._lock_arrays()
        self._require_same_shape()
        self.validate()

    def validate(self):
        """Override for additional validation."""
        pass

    def _array_fields(self):
        return {
            field.name: value
            for field in fields(self)
            if isinstance(value := getattr(self, field.name), np.ndarray)
        }

    def _lock_arrays(self):
        for name, array in self._array_fields().items():
            locked = array.copy()
            locked.setflags(write=False)
            object.__setattr__(self, name, locked)

    def _require_same_shape(self):
        shapes = {name: array.shape for name, array in self._array_fields().items()}

        if len(set(shapes.values())) > 1:
            raise ShapeError(
                f"{type(self).__name__} columns must have the same shape, got {shapes}"
            )
