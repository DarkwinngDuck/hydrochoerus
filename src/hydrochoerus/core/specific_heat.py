import math
from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from hydrochoerus.core.bcs_gap import (
    bcs_reduced_gap,
    bcs_reduced_gap_squared_derivative,
)
from hydrochoerus.core.fermi_dirac import fermi_dirac
from hydrochoerus.model.gap_grid import GapGrid

type CesOnGrid = Callable[[GapGrid, float], np.ndarray]

RELATIVE_TOLERANCE = 1e-10


def ces_integrand(eps_reduced, gap_reduced, gap_squared_derivative, T_reduced, alpha):
    E_reduced = math.hypot(eps_reduced, gap_reduced)
    f = fermi_dirac(E_reduced, T_reduced, alpha)
    f_prefactor = f * (1.0 - f)
    return f_prefactor * (E_reduced**2 / T_reduced - 0.5 * gap_squared_derivative)


def ces_from_gap(
    T_reduced: float, gap_reduced: float, gap_squared_derivative: float, alpha: float
) -> float:
    if T_reduced >= 1.0:
        return T_reduced
    prefactor = 6.0 * alpha**3 / (math.pi**2 * T_reduced)
    integral, _ = quad(
        ces_integrand,
        0.0,
        math.inf,
        args=(gap_reduced, gap_squared_derivative, T_reduced, alpha),
        limit=200,
        epsabs=0.0,
        epsrel=RELATIVE_TOLERANCE,
    )
    return prefactor * integral


def ces(T_reduced: float, alpha: float) -> float:
    return ces_from_gap(
        T_reduced,
        bcs_reduced_gap(T_reduced),
        bcs_reduced_gap_squared_derivative(T_reduced),
        alpha,
    )


def ces_on_grid(gap: GapGrid, alpha: float) -> np.ndarray:
    rows = zip(gap.reduced_temperatures, gap.reduced_gaps, gap.gap_squared_derivatives, strict=True)
    return np.array([ces_from_gap(t, d, dd, alpha) for t, d, dd in rows], dtype=np.float64)
