import math

from scipy.integrate import quad
from scipy.optimize import brentq

from hydrochoerus.consts import ALPHA_BCS

REDUCED_ENERGY_CUTOFF = 500.0
SMALLEST_REDUCED_GAP = 1e-12
LARGEST_REDUCED_GAP = 1.5


def gap_integrand(
    eps_reduced: float,
    gap_reduced: float,
    T_reduced: float,
) -> float:
    E_reduced = math.hypot(eps_reduced, gap_reduced)
    thermal_factor = math.tanh(ALPHA_BCS * E_reduced / (2.0 * T_reduced))
    return thermal_factor / E_reduced


def gap_equation(gap_reduced: float, T_reduced: float) -> float:
    integral, _ = quad(
        gap_integrand,
        0.0,
        REDUCED_ENERGY_CUTOFF,
        args=(gap_reduced, T_reduced),
        limit=200,
    )
    return integral - math.log(2.0 * REDUCED_ENERGY_CUTOFF)


def bcs_reduced_gap(T_reduced: float) -> float:
    if T_reduced >= 1.0:
        return 0.0
    return brentq(
        gap_equation,
        SMALLEST_REDUCED_GAP,
        LARGEST_REDUCED_GAP,
        args=(T_reduced,),
    )


def calculate_g(E_reduced: float, T_reduced: float) -> float:
    return ALPHA_BCS * E_reduced / (2.0 * T_reduced)


def gap_squared_derivative_numerator_integrand(
    eps_reduced: float,
    gap_reduced: float,
    T_reduced: float,
) -> float:
    E_reduced = math.hypot(eps_reduced, gap_reduced)
    g = calculate_g(E_reduced, T_reduced)
    return sech_squared(g)


def gap_squared_derivative_denominator_integrand(
    eps_reduced: float,
    gap_reduced: float,
    T_reduced: float,
) -> float:
    E_reduced = math.hypot(eps_reduced, gap_reduced)
    g = calculate_g(E_reduced, T_reduced)
    tanh_g = math.tanh(g)
    left = T_reduced * sech_squared(g) / (2 * E_reduced**2)
    right = T_reduced**2 * tanh_g / (ALPHA_BCS * E_reduced**3)
    return left - right


def sech_squared(g: float) -> float:
    return 1.0 - math.tanh(g) ** 2


def integrate_over_energy(integrand, gap_reduced: float, T_reduced: float) -> float:
    integral, _ = quad(
        integrand,
        0.0,
        REDUCED_ENERGY_CUTOFF,
        args=(gap_reduced, T_reduced),
        points=(T_reduced, 1.0),
        limit=200,
    )
    return integral


def bcs_reduced_gap_squared_derivative(T_reduced: float) -> float:
    if T_reduced >= 1.0:
        return 0.0
    gap_reduced = bcs_reduced_gap(T_reduced)
    numerator = integrate_over_energy(
        gap_squared_derivative_numerator_integrand, gap_reduced, T_reduced
    )
    denominator = integrate_over_energy(
        gap_squared_derivative_denominator_integrand, gap_reduced, T_reduced
    )
    return numerator / denominator
