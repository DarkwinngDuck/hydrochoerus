import math

import numpy as np

from hydrochoerus.model.fit_run import Metrics


def ss_res(experimental: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sum((experimental - fitted) ** 2))


def ss_tot(experimental: np.ndarray) -> float:
    return float(np.sum((experimental - experimental.mean()) ** 2))


def r_squared(residual: float, total: float) -> float:
    return 1.0 - residual / total if total > 0 else math.nan


def r_factor(experimental: np.ndarray, fitted: np.ndarray) -> float:
    experimental_sum = float(np.sum(np.abs(experimental)))
    diff_sum = float(np.sum(np.abs(experimental - fitted)))
    return diff_sum / experimental_sum if experimental_sum > 0 else math.nan


# TODO: value errors
def metrics(experimental: np.ndarray, fitted: np.ndarray) -> Metrics:
    if experimental.size == 0:
        raise ValueError("empty experimental")
    if experimental.shape != fitted.shape:
        raise ValueError(f"shapes differ, exp: {experimental.shape}  fit: {fitted.shape}")
    residual = ss_res(experimental, fitted)
    return Metrics(
        residual=residual,
        r_squared=r_squared(residual, ss_tot(experimental)),
        r_factor=r_factor(experimental, fitted),
    )
