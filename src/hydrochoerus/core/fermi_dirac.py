import math

OVERFLOW = 100.0


def fermi_dirac(reduced_energy: float, reduced_temperature: float, alpha: float) -> float:
    x = alpha * reduced_energy / reduced_temperature
    if x > OVERFLOW:
        return math.exp(-x)
    return 1.0 / (1.0 + math.exp(x))
