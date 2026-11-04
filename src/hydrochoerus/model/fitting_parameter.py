from dataclasses import dataclass


# TODO: errors
@dataclass(frozen=True)
class FittingParameter:
    """DTO for fitting parameters: anisotropy, weighting two-gap model, alpha"""

    name: str
    label: str
    default_guess: float
    bounds: tuple[float, float]
    scale_ratio: float = 1.0

    def __post_init__(self):
        bottom, top = self.bounds
        if not bottom < top:
            raise ValueError(f"{self.name}: bottom bound must be less then top one, {self.bounds}")
        if not bottom <= self.default_guess <= top:
            raise ValueError(
                f"{self.name}: default guess {self.default_guess} is outside {self.bounds}"
            )

    def scale(self, value: float) -> float:
        return value * self.scale_ratio

    def from_scaled(self, scaled_value: float) -> float:
        return scaled_value / self.scale_ratio

    def scale_bounds(self) -> tuple[float, float]:
        bottom, top = self.bounds
        return self.scale(bottom), self.scale(top)
