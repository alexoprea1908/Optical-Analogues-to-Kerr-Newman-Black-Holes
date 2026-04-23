"""
Ray tracing through concentric annuli of constant scalar refractive index.

This implementation follows the ray-tracing procedure described in the
Methods section of Tinguely & Turner, *Optical analogues to the equatorial
Kerr-Newman black hole*.

For a light ray crossing from annulus i to annulus i+1,

    Phi_{i+1} - Phi_i
        = arcsin(B_{i+1} / R_{i+1}) - arcsin(B_{i+1} / R_i)    (Eq. 25)

with the conserved quantity n_i * B_i = constant.

Conventions
-----------
- edges[0] = R_0 is the outer edge of the optical black hole.
- edges[-1] = R_N is the innermost modeled radius.
- n_values[i] is the scalar refractive index in annulus i,
  i.e. between edges[i] and edges[i+1].
- The medium outside R_0 has index n_outside = 1 by default.

Important modeling choice from the paper
----------------------------------------
Only the ingoing branch is modeled. If the ray cannot reach the next inner
boundary, we stop the trajectory at its minimum radius inside the current
annulus. The outgoing branch is not traced.
"""

from __future__ import annotations

import numpy as np


def ray_trace(edges, n_values, B0, n_outside=1.0, return_status=False):
    edges = np.asarray(edges, dtype=float)
    n_values = np.asarray(n_values, dtype=float)

    if edges.ndim != 1 or n_values.ndim != 1:
        raise ValueError("edges and n_values must be 1-D arrays")
    if len(edges) != len(n_values) + 1:
        raise ValueError("len(edges) must equal len(n_values) + 1")
    if not np.all(edges[:-1] > edges[1:]):
        raise ValueError("edges must be strictly decreasing from outer to inner")
    if np.any(n_values <= 0.0):
        raise ValueError("refractive indices must be strictly positive")

    rho_traj = [edges[0]]
    phi_traj = [0.0]
    phi = 0.0

    const_nb = float(n_values[0]) * float(B0)
    status = {
        "reached_inner": False,
        "turned_before_inner": False,
        "turn_radius": None,
        "turn_annulus": None,
    }

    for i, n_i in enumerate(n_values):
        R_outer = edges[i]
        R_inner = edges[i + 1]
        B_i = const_nb / n_i

        if B_i >= R_inner:
            arg_outer = np.clip(B_i / R_outer, -1.0, 1.0)
            phi += np.pi / 2.0 - np.arcsin(arg_outer)
            rho_traj.append(B_i)
            phi_traj.append(phi)
            status["turned_before_inner"] = True
            status["turn_radius"] = float(B_i)
            status["turn_annulus"] = int(i)
            break

        arg_inner = np.clip(B_i / R_inner, -1.0, 1.0)
        arg_outer = np.clip(B_i / R_outer, -1.0, 1.0)
        phi += np.arcsin(arg_inner) - np.arcsin(arg_outer)
        rho_traj.append(R_inner)
        phi_traj.append(phi)
    else:
        status["reached_inner"] = True

    rho_traj = np.asarray(rho_traj)
    phi_traj = np.asarray(phi_traj)

    if return_status:
        return rho_traj, phi_traj, status
    return rho_traj, phi_traj


def ray_trace_with_outgoing(edges, n_values, B0, n_outside=1.0, return_status=False):
    """
    Same ingoing recursion as ray_trace, but if the ray turns inside some
    annulus k (i.e. B_k >= R_{k+1}), continue the trajectory outward along
    the symmetric outgoing branch: from rho_turn = B_k back up to R_k
    (the outer edge of annulus k), then through annulus k-1, k-2, ..., 0
    out to R_0.  Within each annulus the azimuthal sweep on the way out
    has the same magnitude as on the way in (the ray is the time-reverse
    of an ingoing ray with the same |B|), and phi keeps increasing.

    If the ray reaches the innermost edge (edges[-1]) without turning,
    behaviour is identical to ray_trace.
    """
    edges = np.asarray(edges, dtype=float)
    n_values = np.asarray(n_values, dtype=float)

    if edges.ndim != 1 or n_values.ndim != 1:
        raise ValueError("edges and n_values must be 1-D arrays")
    if len(edges) != len(n_values) + 1:
        raise ValueError("len(edges) must equal len(n_values) + 1")
    if not np.all(edges[:-1] > edges[1:]):
        raise ValueError("edges must be strictly decreasing from outer to inner")
    if np.any(n_values <= 0.0):
        raise ValueError("refractive indices must be strictly positive")

    rho_traj = [edges[0]]
    phi_traj = [0.0]
    phi = 0.0
    const_nb = float(n_values[0]) * float(B0)
    status = {
        "reached_inner": False,
        "turned_before_inner": False,
        "turn_radius": None,
        "turn_annulus": None,
    }

    turn_i = None
    for i, n_i in enumerate(n_values):
        R_outer = edges[i]
        R_inner = edges[i + 1]
        B_i = const_nb / n_i

        if B_i >= R_inner:
            arg_outer = np.clip(B_i / R_outer, -1.0, 1.0)
            phi += np.pi / 2.0 - np.arcsin(arg_outer)
            rho_traj.append(B_i)
            phi_traj.append(phi)
            status["turned_before_inner"] = True
            status["turn_radius"] = float(B_i)
            status["turn_annulus"] = int(i)
            turn_i = i
            break

        arg_inner = np.clip(B_i / R_inner, -1.0, 1.0)
        arg_outer = np.clip(B_i / R_outer, -1.0, 1.0)
        phi += np.arcsin(arg_inner) - np.arcsin(arg_outer)
        rho_traj.append(R_inner)
        phi_traj.append(phi)
    else:
        status["reached_inner"] = True

    # Outgoing branch, only if the ray turned inside some annulus.
    if turn_i is not None:
        # Leg 1: from rho_turn = B_k back up to the outer edge of annulus turn_i.
        n_k = n_values[turn_i]
        B_k = const_nb / n_k
        R_outer_k = edges[turn_i]
        arg_outer_k = np.clip(B_k / R_outer_k, -1.0, 1.0)
        phi += np.pi / 2.0 - np.arcsin(arg_outer_k)
        rho_traj.append(R_outer_k)
        phi_traj.append(phi)

        # Leg 2+: traverse annuli turn_i-1, turn_i-2, ..., 0 in outward direction.
        # The azimuthal sweep in each annulus is the same magnitude as the
        # ingoing sweep would have been, because B_j = const_nb / n_j is
        # unchanged by the reflection and phi continues to increase.
        for j in range(turn_i - 1, -1, -1):
            n_j = n_values[j]
            B_j = const_nb / n_j
            R_outer_j = edges[j]
            R_inner_j = edges[j + 1]
            arg_inner_j = np.clip(B_j / R_inner_j, -1.0, 1.0)
            arg_outer_j = np.clip(B_j / R_outer_j, -1.0, 1.0)
            phi += np.arcsin(arg_inner_j) - np.arcsin(arg_outer_j)
            rho_traj.append(R_outer_j)
            phi_traj.append(phi)

    rho_traj = np.asarray(rho_traj)
    phi_traj = np.asarray(phi_traj)

    if return_status:
        return rho_traj, phi_traj, status
    return rho_traj, phi_traj


def ray_trace_xy(edges, n_values, B0, n_outside=1.0, phi_offset=0.0, return_status=False):
    result = ray_trace(
        edges,
        n_values,
        B0,
        n_outside=n_outside,
        return_status=return_status,
    )

    if return_status:
        rho, phi, status = result
    else:
        rho, phi = result

    phi_plot = phi + phi_offset
    X = rho * np.cos(phi_plot)
    Y = rho * np.sin(phi_plot)

    if return_status:
        return X, Y, rho, phi, status
    return X, Y, rho, phi


def build_schwarzschild_annuli(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
    from Schwarzchild import refractive_index_schwarzschild

    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P_max
    )
    return edges, n_values


def build_kn_annuli(a, rho_Q, b_inf, ell_sign, P_min, P_max, n_annuli, n_at_P0=1.0):
    from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
    from Kerr_Newman import refractive_index_kn_continuous

    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_kn_continuous, edges,
        a, rho_Q, b_inf, ell_sign, P_max, n_at_P0
    )
    return edges, n_values


def angular_deviation_at_horizon(edges, n_values, B0, rho_geo, phi_geo, P_horizon=2.0):
    rho_ray, phi_ray, status = ray_trace(edges, n_values, B0, return_status=True)
    if not status["reached_inner"]:
        return np.nan

    if len(rho_geo) < 2:
        return np.nan

    phi_geo_h = np.interp(P_horizon, rho_geo[::-1], phi_geo[::-1])
    return np.degrees(phi_ray[-1] - phi_geo_h)


def deviation_vs_radius(edges, n_values, B0, rho_geo, phi_geo):
    rho_ray, phi_ray, status = ray_trace(edges, n_values, B0, return_status=True)
    phi_geo_interp = np.interp(rho_ray, rho_geo[::-1], phi_geo[::-1])
    delta_phi = np.degrees(phi_ray - phi_geo_interp)
    return rho_ray, delta_phi, status