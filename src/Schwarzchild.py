import numpy as np


def refractive_index_schwarzschild(P, b_inf, n_at_P0=1.0, P0=6.0):
    """
    Schwarzschild scalar refractive index based on Eq. (14) of the paper:
        n(P) ∝ sqrt(b_inf^2 + 2 P^3)

    We fix the proportionality constant by imposing n(P0) = n_at_P0.
    """
    P = np.asarray(P, dtype=float)

    raw = np.sqrt(b_inf**2 + 2.0 * P**3)
    raw_P0 = np.sqrt(b_inf**2 + 2.0 * P0**3)

    scale = n_at_P0 / raw_P0
    return scale * raw