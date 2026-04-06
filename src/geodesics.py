"""
Numerical integration of null geodesics for Schwarzschild and Kerr-Newman
black holes, following the paper's Eqs. (9)-(10) and (19)-(21).

All quantities are dimensionless (Planck units with c=ħ=G=4πε₀=1).
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Schwarzschild geodesics  (Eq. 10)
# ---------------------------------------------------------------------------

def schwarzschild_geodesic(b_inf, P0, P_end=2.001, n_points=4000):
    """
    Integrate dϕ/dρ = ±(ρ⁴/b∞² − ρ² + 2ρ)^{-1/2}  (Eq. 10)
    from ρ = P0 inward to ρ = P_end (or until the turning point).

    Returns arrays (rho, phi) along the ingoing trajectory.
    The sign convention is chosen so that ϕ increases as ρ decreases.
    """

    def dphidrho(rho, phi):
        arg = rho**4 / b_inf**2 - rho**2 + 2.0 * rho
        if arg <= 0:
            return 0.0
        return -1.0 / np.sqrt(arg)          # minus: ρ decreasing, ϕ increasing

    # Find turning point: ρ⁴/b∞² − ρ² + 2ρ = 0
    # This is where the radial velocity vanishes.
    # We'll let the integrator handle it by stopping when the radicand → 0.

    def radicand_zero(rho, phi):
        return rho**4 / b_inf**2 - rho**2 + 2.0 * rho - 1e-10
    radicand_zero.terminal = True
    radicand_zero.direction = -1

    rho_span = (P0, P_end)
    rho_eval = np.linspace(P0, P_end, n_points)

    sol = solve_ivp(
        dphidrho, rho_span, [0.0],
        t_eval=rho_eval, events=radicand_zero,
        rtol=1e-10, atol=1e-12, method='DOP853', max_step=0.01
    )

    rho = sol.t
    phi = sol.y[0]
    return rho, phi


def schwarzschild_geodesic_xy(b_inf, P0, P_end=2.001, n_points=4000):
    """Return (X, Y) Cartesian coordinates of the geodesic."""
    rho, phi = schwarzschild_geodesic(b_inf, P0, P_end, n_points)
    X = rho * np.cos(phi)
    Y = rho * np.sin(phi)
    return X, Y, rho, phi


# ---------------------------------------------------------------------------
# Kerr–Newman geodesics  (Eqs. 19-21)
# ---------------------------------------------------------------------------

def _delta_hat(rho, a, rho_Q):
    """Eq. (17): Δ̂ = ρ² − 2ρ + â² + ρ_Q²"""
    return rho**2 - 2.0 * rho + a**2 + rho_Q**2


def _V_pm_geo(rho, a, rho_Q, ell_sign):
    """Eq. (21): V̂±  (used in geodesic equation)."""
    D = _delta_hat(rho, a, rho_Q)
    D = max(D, 0.0)
    sqrtD = np.sqrt(D)
    numer_base = a * (2.0 * rho - rho_Q**2)
    denom = (rho**2 + a**2)**2 - a**2 * D
    if abs(denom) < 1e-30:
        return 0.0, 0.0
    Vp = (numer_base + ell_sign * rho**2 * sqrtD) / denom
    Vm = (numer_base - ell_sign * rho**2 * sqrtD) / denom
    return Vp, Vm


def kerr_newman_geodesic(a, rho_Q, b_inf, ell_sign, P0, P_end,
                          n_points=6000):
    """
    Integrate equatorial Kerr-Newman geodesic equations (Eq. 20).
    Includes both ingoing and outgoing legs if the geodesic has a
    turning point (escaping trajectory).

    Parameters
    ----------
    a       : dimensionless spin  (â)
    rho_Q   : dimensionless charge  (ρ_Q)
    b_inf   : impact parameter at infinity  (b̂∞ = ℓ̂/ε̂)
    ell_sign: sign of angular momentum (+1 or -1)
    P0      : outer radius
    P_end   : inner radius to integrate to

    Returns (rho, phi) arrays.
    """
    b_inv = 1.0 / b_inf

    def _dphi_drho(rho, phi, sign=-1):
        """Compute dφ/dρ. sign=-1 for ingoing, +1 for outgoing."""
        D = _delta_hat(rho, a, rho_Q)
        if D < 0:
            D = 0.0
        sqrtD = np.sqrt(max(D, 0.0))

        coeff_ell = (1.0 - 2.0 / rho + rho_Q**2 / rho**2)
        coeff_eps = (2.0 * a / rho - rho_Q**2 * a / rho**2)
        if abs(D) < 1e-30:
            return 0.0

        dphi_ds = (coeff_ell * b_inf + coeff_eps) / D

        Vp, Vm = _V_pm_geo(rho, a, rho_Q, ell_sign)
        prefactor = ((rho**2 + a**2)**2 - a**2 * D) / rho**4
        drho_ds_sq = b_inf**2 * prefactor * (b_inv - Vp) * (b_inv - Vm)

        if drho_ds_sq < 0:
            return 0.0

        drho_ds = sign * np.sqrt(drho_ds_sq)

        if abs(drho_ds) < 1e-30:
            return 0.0

        return dphi_ds / drho_ds

    def deriv_in(rho, y):
        return [_dphi_drho(rho, y[0], sign=-1)]

    def deriv_out(rho, y):
        return [_dphi_drho(rho, y[0], sign=+1)]

    def turning_point(rho, y):
        D = _delta_hat(rho, a, rho_Q)
        D = max(D, 0.0)
        Vp, Vm = _V_pm_geo(rho, a, rho_Q, ell_sign)
        prefactor = ((rho**2 + a**2)**2 - a**2 * D) / rho**4
        val = prefactor * (b_inv - Vp) * (b_inv - Vm)
        return val - 1e-10
    turning_point.terminal = True
    turning_point.direction = -1

    rho_eval = np.linspace(P0, P_end, n_points)

    # --- Ingoing leg ---
    sol = solve_ivp(
        deriv_in, (P0, P_end), [0.0],
        t_eval=rho_eval, events=turning_point,
        rtol=1e-10, atol=1e-12, method='DOP853', max_step=0.005
    )

    rho_in = sol.t
    phi_in = sol.y[0]

    # --- Check for turning point and integrate outgoing leg ---
    hit_turning = (sol.t_events is not None and len(sol.t_events) > 0
                   and len(sol.t_events[0]) > 0)

    if hit_turning:
        rho_turn = sol.t_events[0][0]
        phi_turn = sol.y_events[0][0, 0]

        # Outgoing: integrate from turning point back out to P0
        rho_eval_out = np.linspace(rho_turn + 1e-6, P0, n_points)

        sol_out = solve_ivp(
            deriv_out, (rho_turn + 1e-6, P0), [phi_turn],
            t_eval=rho_eval_out,
            rtol=1e-10, atol=1e-12, method='DOP853', max_step=0.005
        )

        rho = np.concatenate([rho_in, sol_out.t])
        phi = np.concatenate([phi_in, sol_out.y[0]])
    else:
        rho = rho_in
        phi = phi_in

    return rho, phi


def kerr_newman_geodesic_xy(a, rho_Q, b_inf, ell_sign, P0, P_end,
                             n_points=6000):
    """Return Cartesian coordinates and (rho, phi) of KN geodesic."""
    rho, phi = kerr_newman_geodesic(a, rho_Q, b_inf, ell_sign, P0, P_end,
                                     n_points)
    X = rho * np.cos(phi)
    Y = rho * np.sin(phi)
    return X, Y, rho, phi