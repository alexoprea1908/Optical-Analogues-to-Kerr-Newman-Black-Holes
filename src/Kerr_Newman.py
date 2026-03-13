import numpy as np


def delta_hat(P, a, rho_Q):
    """
    Eq. (17):
        Delta_hat = P^2 - 2P + a^2 + rho_Q^2
    """
    P = np.asarray(P, dtype=float)
    return P**2 - 2.0 * P + a**2 + rho_Q**2


def V_pm(P, a, rho_Q, ell_sign=+1):
    """
    Eq. (21):
        V_± = [ a (2P - rho_Q^2) ± sgn(l) P^2 sqrt(Delta_hat) ] /
              [ (P^2 + a^2)^2 - a^2 Delta_hat ]
    """
    P = np.asarray(P, dtype=float)
    D = delta_hat(P, a, rho_Q)

    if np.any(D < -1e-12):
        raise ValueError("delta_hat < 0 on the requested interval.")

    D = np.maximum(D, 0.0)
    sqrtD = np.sqrt(D)
    denom = (P**2 + a**2) ** 2 - a**2 * D

    V_plus = (a * (2.0 * P - rho_Q**2) + ell_sign * P**2 * sqrtD) / denom
    V_minus = (a * (2.0 * P - rho_Q**2) - ell_sign * P**2 * sqrtD) / denom
    return V_plus, V_minus


def b_hat_kn_exact(P, a, rho_Q, b_inf, ell_sign=+1):
    """
    Exact Eq. (22), using the notation on page 4.

    IMPORTANT:
    The numerator is P^2 * X,
    but the first term under the square root is P^2 * X^2,
    not (P^2 * X)^2.
    """
    P = np.asarray(P, dtype=float)
    D = delta_hat(P, a, rho_Q)

    if np.any(D < -1e-12):
        raise ValueError("delta_hat < 0 on the requested interval.")

    D = np.maximum(D, 0.0)
    V_plus, V_minus = V_pm(P, a, rho_Q, ell_sign)

    b_inf_inv = 1.0 / b_inf

    X = (D - a**2) + (2.0 * P - rho_Q**2) * a * b_inf_inv
    B = (P**2 + a**2) ** 2 - a**2 * D
    C = (b_inf_inv - V_plus) * (b_inf_inv - V_minus)

    numerator = P**2 * X
    radicand = P**2 * (X**2) + (D**2) * B * C

    if np.any(radicand < -1e-10):
        raise ValueError("Negative radicand encountered in Eq. (22).")

    radicand = np.maximum(radicand, 0.0)
    denom = np.sqrt(radicand)

    out = numerator / denom

    # Handle removable 0/0 cases at the horizon with a one-sided limit
    bad = ~np.isfinite(out)
    if np.any(bad):
        out = out.astype(float)
        idx = np.where(bad)[0]
        for i in idx:
            p = P[i]
            found = False
            for eps in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
                pp = p + eps
                Dp = delta_hat(np.array([pp]), a, rho_Q)
                if Dp[0] < -1e-12:
                    continue
                Dp = np.maximum(Dp, 0.0)
                Vp, Vm = V_pm(np.array([pp]), a, rho_Q, ell_sign)
                Xp = (Dp - a**2) + (2.0 * pp - rho_Q**2) * a * b_inf_inv
                Bp = (pp**2 + a**2) ** 2 - a**2 * Dp
                Cp = (b_inf_inv - Vp) * (b_inf_inv - Vm)
                Rp = pp**2 * (Xp**2) + (Dp**2) * Bp * Cp
                Rp = np.maximum(Rp, 0.0)
                val = (pp**2 * Xp) / np.sqrt(Rp)
                if np.isfinite(val[0]):
                    out[i] = val[0]
                    found = True
                    break
            if not found:
                out[i] = np.nan

    return out


def refractive_index_kn_continuous(P, a, rho_Q, b_inf, ell_sign=+1, P0=6.0, n0=1.0):
    """
    Eq. (23):
        n(P) ∝ 1 / b_hat(P)

    Normalize so that n(P0) = n0.
    """
    P = np.asarray(P, dtype=float)

    bP = b_hat_kn_exact(P, a, rho_Q, b_inf, ell_sign)
    bP0 = b_hat_kn_exact(np.array([P0], dtype=float), a, rho_Q, b_inf, ell_sign)[0]

    raw = 1.0 / bP
    raw0 = 1.0 / bP0

    scale = n0 / raw0
    return scale * raw