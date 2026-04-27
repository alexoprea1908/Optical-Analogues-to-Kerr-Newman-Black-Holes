"""
Test cases for FDFD simulations of optical Schwarzschild and Kerr-Newman
black holes.

Each case carries its own `b_inf` so a single change does not break the
relationship between b_inf and the radius P_* at which the scalar refractive
index n(P) diverges (Eq. 24 of the paper).

For Kerr-Newman, P_min is computed automatically from a, rho_Q and b_inf so
that the simulated annular system always sits OUTSIDE the n-divergence radius.
This avoids the artefact in which an annulus at R < R_* gets a nonsensically
large refractive index that distorts the FDFD field.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Schwarzschild cases (Fig. 3 of the paper uses b_inf = 2, 3, 4, 5)
# ---------------------------------------------------------------------------

SCHWARZSCHILD_CASES = [
    {"name": "schwarzschild_b2", "panel": "a", "b_inf": 2.0},
    {"name": "schwarzschild_b3", "panel": "b", "b_inf": 3.0},
    {"name": "schwarzschild_b4", "panel": "c", "b_inf": 4.0},
    {"name": "schwarzschild_b5", "panel": "d", "b_inf": 5.0},
]


# ---------------------------------------------------------------------------
# Helpers for Kerr-Newman geometry
# ---------------------------------------------------------------------------

P0 = 6.0


def kn_horizon(a, rho_Q):
    """Outer event horizon: largest root of Δ̂ = ρ² − 2ρ + â² + ρ_Q² = 0."""
    disc = 1.0 - a**2 - rho_Q**2
    if disc < 0:
        return None  # naked singularity (not allowed for the cases we use)
    return 1.0 + np.sqrt(disc)


def kn_n_divergence_radius(a, rho_Q, b_inf):
    """
    Radius P_* at which the scalar refractive index n(P) diverges.
    Paper Eq. (24):  P_* = 1 - â/b̂_∞ + sqrt[(1 - â/b̂_∞)(1 - â/b̂_∞ - ρ_Q²)].
    For Schwarzschild and Reissner-Nordström this is a removable pole that
    lies on or inside the horizon. For rotating BHs with counter-rotating
    geodesics it can sit OUTSIDE the horizon, where it represents the true
    rotation-reversal point and bounds the validity of the gradient-index
    analogue from below.
    """
    inv = a / b_inf
    inner = (1.0 - inv) * (1.0 - inv - rho_Q**2)
    if inner < 0:
        return None
    return (1.0 - inv) + np.sqrt(inner)


def kn_auto_P_min(a, rho_Q, b_inf, safety=0.1):
    """
    Pick the innermost simulated radius for a Kerr-Newman case.

    This helper deliberately does not classify the ray as escaping, capturing,
    or critical. It only keeps the simulated annulus outside the radii where
    the scalar refractive-index geometry would become invalid:

      P_h : outer event horizon,
      P_* : radius where n(P) diverges, from Eq. (24).

    Thus P_min tracks b_inf automatically through P_*:

        P_min = max(P_h, P_*) + safety.

    Change B_INF_KN once, and all four Kerr-Newman panels update consistently.
    """
    P_h = kn_horizon(a, rho_Q)
    P_s = kn_n_divergence_radius(a, rho_Q, b_inf)

    candidates = []
    if P_h is not None:
        candidates.append(P_h)
    if P_s is not None:
        candidates.append(P_s)

    if not candidates:
        raise ValueError(
            f"No valid P_min for a={a}, rho_Q={rho_Q}, b_inf={b_inf}"
        )

    return max(candidates) + safety


# ---------------------------------------------------------------------------
# Kerr-Newman cases (Fig. 4 of the paper uses b_inf = 3 in all four panels)
# ---------------------------------------------------------------------------

# Single source of truth for the impact parameter of the KN figure. Change
# this in ONE place and every panel's P_min will update consistently.
B_INF_KN = 5.409

_kn_specs = [
    # panel,     name,                   a,          rho_Q,        ell_sign
    ("a",  "extremal_kerr_corot",        1.0,        0.0,           +1),
    ("b",  "extremal_rn",                0.0,        1.0,           +1),
    ("c",  "kn_corot",                   2.0 / 5.0,  4.0 / 5.0,     +1),
    ("d",  "kn_counter",                -2.0 / 5.0,  4.0 / 5.0,     +1),
]

KERR_NEWMAN_CASES = [
    {
        "panel": panel,
        "name": name,
        "a": a,
        "rho_Q": rho_Q,
        "ell_sign": ell_sign,
        "b_inf": B_INF_KN,
        "P_min": kn_auto_P_min(a, rho_Q, B_INF_KN),
        "P_max": P0,
        "n_annuli": 21,
    }
    for panel, name, a, rho_Q, ell_sign in _kn_specs
]


if __name__ == "__main__":
    # Quick sanity print: shows the radii used to choose P_min.
    print(f"Kerr-Newman cases with b_inf = {B_INF_KN}:")
    print("-" * 72)
    print(f"{'panel':<6} {'a':>8} {'rho_Q':>8} {'P_h':>10} {'P_*':>10} {'P_min':>10}")
    print("-" * 72)

    for c in KERR_NEWMAN_CASES:
        a_, rho_Q_, b_ = c["a"], c["rho_Q"], c["b_inf"]
        P_h = kn_horizon(a_, rho_Q_)
        P_s = kn_n_divergence_radius(a_, rho_Q_, b_)

        P_h_str = f"{P_h:.6f}" if P_h is not None else "None"
        P_s_str = f"{P_s:.6f}" if P_s is not None else "None"

        print(
            f"{c['panel']:<6} "
            f"{a_:>8.3f} "
            f"{rho_Q_:>8.3f} "
            f"{P_h_str:>10} "
            f"{P_s_str:>10} "
            f"{c['P_min']:>10.6f}"
        )
