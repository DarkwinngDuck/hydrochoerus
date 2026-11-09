import math

from scipy.integrate import quad
from scipy.optimize import brentq

from hydrochoerus.consts import ALPHA_BCS

REDUCED_ENERGY_CUTOFF = 500.0
SMALLEST_REDUCED_GAP = 1e-12
LARGEST_REDUCED_GAP = 1.5


def gap_integrand(
    reduced_normal_energy: float,
    reduced_gap: float,
    reduced_temperature: float,
) -> float:
    reduced_quasiparticle_energy = math.hypot(reduced_normal_energy, reduced_gap)
    thermal_factor = math.tanh(
        ALPHA_BCS * reduced_quasiparticle_energy / (2.0 * reduced_temperature)
    )
    return thermal_factor / reduced_quasiparticle_energy


def gap_equation(reduced_gap: float, reduced_temperature: float) -> float:
    integral, _ = quad(
        gap_integrand,
        0.0,
        REDUCED_ENERGY_CUTOFF,
        args=(reduced_gap, reduced_temperature),
        limit=200,
    )
    return integral - math.log(2.0 * REDUCED_ENERGY_CUTOFF)


def bcs_reduced_gap(reduced_temperature: float) -> float:
    if reduced_temperature >= 1.0:
        return 0.0
    return brentq(
        gap_equation,
        SMALLEST_REDUCED_GAP,
        LARGEST_REDUCED_GAP,
        args=(reduced_temperature,),
    )
