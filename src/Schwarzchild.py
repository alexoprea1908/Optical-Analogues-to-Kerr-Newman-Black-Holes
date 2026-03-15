import numpy as np

from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
from constants import DEFAULT_NUM_ANNULI, P0 as DEFAULT_P0


def refractive_index_schwarzschild(P, b_inf, n_at_P0=1.0, P0=DEFAULT_P0):
    """
    Schwarzschild scalar refractive index (Eq. 14 in the paper):
        n(P) is proportional to sqrt(b_inf^{-2} + 2 P^{-3})

    The proportionality constant is fixed so that n(P0) = n_at_P0.
    """
    P = np.asarray(P, dtype=float)

    raw = np.sqrt(b_inf**-2 + 2.0 * P**-3)
    raw_P0 = np.sqrt(b_inf**-2 + 2.0 * P0**-3)

    scale = n_at_P0 / raw_P0
    return scale * raw


def schwarzschild_annuli_profile(
    b_inf,
    P_min,
    P_max=DEFAULT_P0,
    n_annuli=DEFAULT_NUM_ANNULI,
    n_at_P0=1.0,
    P0=DEFAULT_P0,
):
    """
    Piecewise-constant annular approximation of the Schwarzschild index profile.

    Uses the repo's annulus convention: half-width end annuli and endpoint sampling.
    Returns (centers, values) ordered from outer to inner, matching annuli.py.
    """
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    centers, values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P0
    )
    return centers, values
