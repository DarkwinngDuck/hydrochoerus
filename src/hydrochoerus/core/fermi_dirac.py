import math

OVERFLOW = 100.0


def fermi_dirac(reduced_energy: float, T_reduced: float, alpha: float) -> float:
    x = alpha * reduced_energy / T_reduced
    if x > OVERFLOW:
        return math.exp(-x)
    return 1.0 / (1.0 + math.exp(x))
