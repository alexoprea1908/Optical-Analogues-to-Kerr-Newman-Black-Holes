"""
Ray tracing through concentric annuli of constant scalar refractive index.

Implements the algorithm described in the Methods section of the paper:
    Phi_{i+1} - Phi_i = arcsin(B_{i+1}/R_{i+1}) - arcsin(B_{i+1}/R_i)   (Eq. 25)
with the conservation law  n_i * B_i = constant.

Convention
----------
- edges[0] = R_0 (outer edge), edges[-1] = R_N (inner edge)
- n_values[i] = scalar index in annulus i (between edges[i] and edges[i+1])
- The medium outside R_0 has index n_outside = 1.
"""

import numpy as np


def ray_trace(edges, n_values, b_inf, n_outside=1.0):
    """
    Trace a ray inward through the annular system.

    Parameters
    ----------
    edges : 1-D array, length N+1
        Radii ordered from outer to inner: edges[0] > edges[1] > ... > edges[N].
    n_values : 1-D array, length N
        Refractive index in each annulus.
    b_inf : float
        Impact parameter at infinity (free-space).
    n_outside : float
        Refractive index outside the system (default 1).

    Returns
    -------
    rho_traj : array of float
        Radii along the trajectory (at each annulus boundary crossed).
    phi_traj : array of float
        Accumulated azimuthal angle at each radius.
    """
    N = len(n_values)
    assert len(edges) == N + 1

    # Conserved quantity: n * B = const = n_outside * B_outside
    # At P0, the impact parameter is b_inf (since n_outside = 1)
    C = n_outside * b_inf

    rho_traj = [edges[0]]
    phi_traj = [0.0]

    phi = 0.0

    for i in range(N):
        R_outer = edges[i]
        R_inner = edges[i + 1]
        n_i = n_values[i]
        B_i = C / n_i  # impact parameter in this annulus

        # Check if ray can reach inner boundary
        if B_i >= R_inner:
            # Ray turns around: closest approach at R = B_i
            arg_outer = min(B_i / R_outer, 1.0)
            dphi = np.pi / 2.0 - np.arcsin(arg_outer)
            phi += dphi
            rho_traj.append(B_i)
            phi_traj.append(phi)
            break

        # Angular change traversing this annulus (Eq. 25)
        arg_inner = min(B_i / R_inner, 1.0)
        arg_outer = min(B_i / R_outer, 1.0)

        dphi = np.arcsin(arg_inner) - np.arcsin(arg_outer)
        phi += dphi
        rho_traj.append(R_inner)
        phi_traj.append(phi)

    return np.array(rho_traj), np.array(phi_traj)


def ray_trace_xy(edges, n_values, b_inf, n_outside=1.0):
    """Return Cartesian (X, Y) coordinates of the ray trajectory."""
    rho, phi = ray_trace(edges, n_values, b_inf, n_outside)
    X = rho * np.cos(phi)
    Y = rho * np.sin(phi)
    return X, Y, rho, phi


# ---------------------------------------------------------------------------
# Helpers for building annular systems for Schwarzschild / Kerr-Newman
# ---------------------------------------------------------------------------

def build_schwarzschild_annuli(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    """
    Build uniform-thickness annuli (half-width end annuli) for a
    Schwarzschild optical black hole.
    """
    from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
    from Schwarzchild import refractive_index_schwarzschild

    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P_max
    )
    return edges, n_values


def build_kn_annuli(a, rho_Q, b_inf, ell_sign, P_min, P_max, n_annuli,
                    n_at_P0=1.0):
    """
    Build uniform-thickness annuli for a Kerr-Newman optical black hole.
    """
    from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
    from Kerr_Newman import refractive_index_kn_continuous

    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_kn_continuous, edges,
        a, rho_Q, b_inf, ell_sign, P_max, n_at_P0
    )
    return edges, n_values


# ---------------------------------------------------------------------------
# Error analysis helpers (for Figs. 5 and 6)
# ---------------------------------------------------------------------------

def angular_deviation_at_horizon(edges, n_values, b_inf, rho_geo, phi_geo,
                                  P_horizon=2.0):
    """
    Compute DeltaPhi = Phi_ray - Phi_geo at a given radius.
    """
    rho_ray, phi_ray = ray_trace(edges, n_values, b_inf)

    idx = np.argmin(np.abs(rho_ray - P_horizon))
    phi_ray_h = phi_ray[idx]

    if len(rho_geo) < 2:
        return np.nan
    phi_geo_h = np.interp(rho_ray[idx], rho_geo[::-1], phi_geo[::-1])

    return np.degrees(phi_ray_h - phi_geo_h)


def deviation_vs_radius(edges, n_values, b_inf, rho_geo, phi_geo):
    """
    Compute DeltaPhi(R) = Phi_ray(R) - Phi_geo(R) at each annulus boundary.
    """
    rho_ray, phi_ray = ray_trace(edges, n_values, b_inf)
    phi_geo_interp = np.interp(rho_ray, rho_geo[::-1], phi_geo[::-1])
    delta_phi = np.degrees(phi_ray - phi_geo_interp)
    return rho_ray, delta_phi