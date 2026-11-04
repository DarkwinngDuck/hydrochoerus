from dataclasses import dataclass, field
from enum import Enum


class InvalidTransition(Exception):
    """Can not move to the selected run status"""


class RunStatus(Enum):
    RUNNING = "running"
    DONE = "done"
    SIMULATED = "simulated"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def can_change_to(self, status: RunStatus) -> bool:
        return self is RunStatus.RUNNING and status is not RunStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        return self is not RunStatus.RUNNING


@dataclass(frozen=True)
class Metrics:
    residual: float
    r_squared: float
    r_factor: float


@dataclass(frozen=True)
class FitResult:
    parameters: dict[str, float]
    metrics: Metrics


@dataclass(frozen=True)
class SweepStep:
    x1: float
    parameters: dict[str, float]
    metrics: Metrics


@dataclass
class FitRun:
    id: int
    scan_name: str
    model_name: str
    inputs: dict[str, float]
    t_max: float
    status: RunStatus = RunStatus.RUNNING
    result: FitResult | None = None
    sweep_steps: list[SweepStep] = field(default_factory=list)
    error: str | None = None

    def finish(
        self,
        status: RunStatus,
        result: FitResult | None = None,
        error: str | None = None,
    ) -> None:
        if not self.status.can_change_to(status):
            raise InvalidTransition(f"run {self.id}: {self.status.value} -> {status.value}")
        if status in {RunStatus.DONE, RunStatus.SIMULATED} and result is None:
            raise ValueError(f"FitResult is required {self.id}")
        self.status = status
        self.result = result
        self.error = error
