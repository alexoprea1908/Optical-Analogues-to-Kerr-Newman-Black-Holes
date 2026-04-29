import numpy as np

from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
from constants import DEFAULT_NUM_ANNULI, P0 as DEFAULT_P0


def refractive_index_schwarzschild(P, b_inf, n_at_P0=1.0, P0=DEFAULT_P0):
    """
    Schwarzschild scalar refractive index (Eq. 14 in the paper):
        n(P) is proportional to sqrt(b_inf^{-2} + 2 P^{-3})

    The proportionality constant is fixed so that n(P0) = n_at_P0.
    """
    P = np.asarray(P, dtype=float) #turns P into a numpy array, which allows for vectorized operations. The dtype=float ensures that the elements of the array are treated as floating-point numbers, which is important for the mathematical operations that follow.

    raw = np.sqrt(b_inf**-2 + 2.0 * P**-3)
    raw_P0 = np.sqrt(b_inf**-2 + 2.0 * P0**-3)

    scale = n_at_P0 / raw_P0
    return scale * raw # The function calculates the raw refractive index values using the formula from the paper, then computes a scaling factor to ensure that n(P0) equals n_at_P0, and finally returns the scaled refractive index values.



def schwarzschild_annuli_profile(
    b_inf, P_min,
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
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli) # This function generates the edges of the annuli based on the specified minimum and maximum radii (P_min and P_max) and the number of annuli (n_annuli). The edges are ordered from outer to inner, following the convention used in the repository.
    centers, values = sample_piecewise_constant(refractive_index_schwarzschild, edges, b_inf, n_at_P0, P0) # This function samples the refractive index at the centers of the annuli defined by the edges. It uses the refractive_index_schwarzschild function to compute the refractive index values at these centers, and it also applies the specified parameters (b_inf, n_at_P0, P0) to ensure that the profile is consistent with the desired conditions.
    return centers, values
