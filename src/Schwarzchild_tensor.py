"""
Polarization-preserving Schwarzschild analogue.

Implements paper Eq. (8): the dielectric tensor and (equal) permeability tensor
in dimensionless real-space Cartesian coordinates that EXACTLY reproduce both
the null geodesics AND the polarization of light moving in the Schwarzschild
metric (in Schwarzschild coordinates), in contrast to the scalar n(P) of
Schwarzchild.py which only reproduces trajectories.

    eps^{ij} = mu^{ij} = (2 P^i P^j - P^3 delta^{ij}) / [P^2 (2 - P)]

where P^i = (X_hat, Y_hat, Z_hat) is the dimensionless real-space radius vector
and P = ||P^i||.

Equatorial reduction (Z_hat = 0)
--------------------------------
We restrict to the equatorial plane Z_hat = 0, consistent with the rest of
the codebase. Plugging Z_hat = 0 into Eq. (8):

    eps^{XX} = (2 X^2 - P^3) / [P^2 (2 - P)]
    eps^{YY} = (2 Y^2 - P^3) / [P^2 (2 - P)]
    eps^{XY} = eps^{YX} = 2 X Y      / [P^2 (2 - P)]
    eps^{ZZ} = -P^3 / [P^2 (2 - P)] = -P / (2 - P)
    eps^{XZ} = eps^{YZ} = 0

For TM polarization in the equatorial plane (E = E_z z_hat, H in-plane), the
relevant tensor components are:

    eps_zz                   couples to E_z
    mu_xx, mu_yy, mu_xy      couple to H_x, H_y

The factor 1/(2 - P) blows up at the horizon P = 2 just like the scalar n,
so we still truncate the simulation at P_min slightly outside the horizon.

For P < 2 (inside the Schwarzschild radius), eps_zz becomes NEGATIVE: -P/(2-P)
> 0 numerator with negative denominator gives a negative real eps. This is
physically meaningful (an opaque, plasma-like medium) and is exactly the right
behaviour for an analogue of the BH interior, but we do not include this
region in the simulation domain — we cap the medium at P_min and replace the
core with the same absorbing disc the scalar code uses, for direct
comparability with the scalar Fig. 3.

Annular sampling
----------------
The radial dependence of the tensor lives entirely in 1/(2 - P); the angular
dependence is fixed by the projector P^i P^j / P^2. We therefore make the
medium piecewise-radially-constant (each annulus uses a single value of the
coefficient 1/(2 - P_center)) but still angularly varying through P^i P^j on
the actual grid. This matches the spirit of the paper's annular construction
in the scalar case while preserving the tensor's directional structure inside
each annulus.
"""

import numpy as np

from annuli import annulus_edges_with_half_ends, annulus_centers_from_edges


# ---------------------------------------------------------------------------
# Continuous tensor field on a 2D Cartesian grid
# ---------------------------------------------------------------------------

def schwarzschild_tensor_field(X, Y, P_min=None, regularize=1e-3):
    """
    Evaluate the exact equatorial Schwarzschild Eq. (8) tensor on a 2D grid
    of dimensionless coordinates (X, Y) (i.e. X = X_hat, Y = Y_hat).

    Parameters
    ----------
    X, Y : ndarray, same shape
        Dimensionless Cartesian coordinates.
    P_min : float or None
        If given, points with sqrt(X^2 + Y^2) < P_min are masked out (NaN)
        — caller fills the core separately.
    regularize : float
        Small floor to keep |2 - P| from going exactly to zero at the horizon
        in cells that happen to land there. Acts only when P_min is None.

    Returns
    -------
    Components dict with keys:
        'eps_zz', 'mu_xx', 'mu_yy', 'mu_xy', 'P', 'mask'

    where mask is True for cells inside the simulated medium.
    """
    P = np.sqrt(X**2 + Y**2)

    # Coefficient that carries the radial divergence
    denom = P**2 * (2.0 - P)

    if P_min is None:
        # Avoid exact pole at P = 2
        denom = np.where(np.abs(2.0 - P) < regularize,
                         np.sign(2.0 - P) * regularize * P**2,
                         denom)
        mask = np.ones_like(P, dtype=bool)
    else:
        mask = P >= P_min

    # In-plane components of eps = mu (Eq. 8 with Z_hat = 0)
    # Use safe denominator for arithmetic; values outside `mask` are bogus
    # and will be overwritten by the caller anyway.
    safe_denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)

    eps_xx = (2.0 * X**2 - P**3) / safe_denom
    eps_yy = (2.0 * Y**2 - P**3) / safe_denom
    eps_xy = (2.0 * X * Y)        / safe_denom

    # Out-of-plane (z) component
    # eps_zz = -P^3 / [P^2 (2 - P)] = -P / (2 - P)
    eps_zz = -P / np.where(np.abs(2.0 - P) > 1e-30, 2.0 - P, 1e-30)

    # The paper has eps^ij = mu^ij, so all four use the same components.
    return {
        'eps_zz': eps_zz,
        'mu_xx':  eps_xx,
        'mu_yy':  eps_yy,
        'mu_xy':  eps_xy,
        'P':      P,
        'mask':   mask,
    }


# ---------------------------------------------------------------------------
# Piecewise-constant (radially) annular tensor profile
# ---------------------------------------------------------------------------

def schwarzschild_tensor_annular_field(X, Y, P_min, P_max, n_annuli):
    """
    Build the equatorial Eq. (8) tensor on grid (X, Y), but with the radial
    coefficient C(P) = 1 / [P^2 (2 - P)] held PIECEWISE CONSTANT inside each
    annulus, exactly as the scalar code does for n(P).

    Each annulus uses C evaluated at its centre (with the paper's endpoint
    rule for the outermost / innermost annuli). The angular projector
    P^i P^j (and the P^3 delta term) is re-evaluated cell-by-cell so the
    tensor's directional structure is preserved within each annulus.

    The eps_zz component is also held piecewise constant per annulus, using
    -P_eval / (2 - P_eval) at the same sampling P_eval as C.

    Parameters
    ----------
    X, Y : ndarray
        Dimensionless Cartesian grid.
    P_min, P_max : float
        Inner / outer dimensionless radii of the annular system.
    n_annuli : int
        Number of annuli.

    Returns
    -------
    components : dict with same keys as schwarzschild_tensor_field, plus
                 'edges' and 'C_values' for diagnostics.

        Outside the medium (P > P_max or P < P_min), the tensor entries are
        set to free-space defaults (eps_zz = 1, mu_xx = mu_yy = 1, mu_xy = 0)
        so the caller can overlay an absorbing core / leave vacuum outside.
    """
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    centers = annulus_centers_from_edges(edges)

    # Sampling points for the radial coefficient inside each annulus
    P_eval = centers.copy()
    # Endpoint rule from the paper / annuli.py: outermost & innermost use edges
    P_eval[0]  = edges[0]    # P_max
    if len(P_eval) > 1:
        P_eval[-1] = edges[-1]   # P_min

    # Radial coefficient C = 1 / [P^2 (2 - P)] and the eps_zz value, sampled
    # once per annulus
    C_values    = 1.0 / (P_eval**2 * (2.0 - P_eval))
    epszz_values = -P_eval / (2.0 - P_eval)

    P_grid = np.sqrt(X**2 + Y**2)

    # Default: free space everywhere
    eps_zz = np.ones_like(P_grid, dtype=float)
    mu_xx  = np.ones_like(P_grid, dtype=float)
    mu_yy  = np.ones_like(P_grid, dtype=float)
    mu_xy  = np.zeros_like(P_grid, dtype=float)

    # Fill annulus by annulus
    for i, C in enumerate(C_values):
        R_outer = edges[i]
        R_inner = edges[i + 1]
        in_ann = (P_grid <= R_outer) & (P_grid > R_inner)
        if not np.any(in_ann):
            continue

        Xa = X[in_ann]
        Ya = Y[in_ann]
        Pa = P_grid[in_ann]

        # In-plane components (Eq. 8): we keep angular structure cell-by-cell
        # but multiply by the annulus-constant coefficient C. We also keep the
        # radial scalar P^3 on the diagonal piecewise constant by sampling at
        # P_eval[i] -- so the WHOLE radial dependence is annulus-wise constant.
        Pe3 = P_eval[i]**3
        eps_xx_ann = (2.0 * Xa**2 - Pe3) * C
        eps_yy_ann = (2.0 * Ya**2 - Pe3) * C
        eps_xy_ann = (2.0 * Xa * Ya)     * C

        mu_xx[in_ann] = eps_xx_ann
        mu_yy[in_ann] = eps_yy_ann
        mu_xy[in_ann] = eps_xy_ann
        eps_zz[in_ann] = epszz_values[i]

    return {
        'eps_zz':   eps_zz,
        'mu_xx':    mu_xx,
        'mu_yy':    mu_yy,
        'mu_xy':    mu_xy,
        'P':        P_grid,
        'edges':    edges,
        'centers':  centers,
        'P_eval':   P_eval,
        'C_values': C_values,
        'epszz_values': epszz_values,
    }


# ---------------------------------------------------------------------------
# Convenience: print a summary of the annular tensor profile
# ---------------------------------------------------------------------------

def summarize_tensor_profile(P_min, P_max, n_annuli):
    """Print the per-annulus C and eps_zz values; useful for sanity checking."""
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    centers = annulus_centers_from_edges(edges)
    P_eval = centers.copy()
    P_eval[0]  = edges[0]
    if len(P_eval) > 1:
        P_eval[-1] = edges[-1]

    print(f"  i  P_eval    C = 1/[P^2(2-P)]    eps_zz = -P/(2-P)")
    print("  " + "-" * 56)
    for i, P in enumerate(P_eval):
        C = 1.0 / (P**2 * (2.0 - P))
        ez = -P / (2.0 - P)
        print(f"  {i:2d}  {P:6.3f}    {C:+.4f}            {ez:+.4f}")


if __name__ == "__main__":
    print("Tensor profile, P_min=2.1, P_max=6, 16 annuli:")
    summarize_tensor_profile(2.1, 6.0, 16)
    print()
    print("Tensor profile, P_min=2.5, P_max=6, 16 annuli (well outside horizon):")
    summarize_tensor_profile(2.5, 6.0, 16)
