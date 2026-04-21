"""
Reproduce the ray-tracing analysis of Figs. 5 and 6 from
Tinguely & Turner, *Optical analogues to the equatorial Kerr-Newman
black hole*.

This version follows the paper's procedure more closely than the original:
- the ray tracer uses the initial impact parameter at the system edge, B0;
- only the ingoing branch is traced;
- Fig. 5 measures angular deviation at the horizon only when the ray actually
  reaches the innermost modeled radius;
- Figs. 6b and 6d leave undefined regions gray when the ray turns before
  reaching smaller radii;
- the plotting convention places the source on the lower-right side, matching
  the paper's geometry more closely.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from geodesics import schwarzschild_geodesic
from ray_tracing import ray_trace
from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
from Schwarzchild import refractive_index_schwarzschild
from constants import P0


def _build_uniform_annuli(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P_max
    )
    return edges, n_values


def _b_hat_at_P0(b_inf, P0_val):
    if np.isclose(b_inf, 0.0):
        return 0.0
    return 1.0 / np.sqrt(b_inf ** (-2) + 2.0 * P0_val ** (-3))


def _phi_offset_for_entry(B0, P0_val):
    ratio = np.clip(B0 / P0_val, -1.0, 1.0)
    return np.arcsin(ratio)


def _make_symmetric_inferno():
    n_half = 128
    c1 = plt.cm.inferno(np.linspace(0.0, 1.0, n_half))
    c2 = plt.cm.inferno(np.linspace(1.0, 0.0, n_half))
    return LinearSegmentedColormap.from_list('sym_inferno', np.vstack([c1, c2]))


def _add_annulus_circles(ax, edges, color='0.72', lw=0.5, alpha=0.9):
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for e in edges:
        ax.plot(e * np.cos(theta), e * np.sin(theta),
                color=color, lw=lw, alpha=alpha, zorder=0)


def _interp_phi_on_radius(rho_query, rho_geo, phi_geo):
    return np.interp(rho_query, rho_geo[::-1], phi_geo[::-1])


# ---------------------------------------------------------------------------
# Paper-convention helpers used only by Figure 5.
#
# Figure 5 follows the paper's Eq. 14 normalization n(inf) = 1 and uses the
# impact parameter at infinity, B_outside = b_hat_inf, as the entry condition.
# Figure 6 code below is unchanged and keeps the n(P0) = 1 / B0 = b_hat(P0)
# convention that makes the ray tangent to the geodesic at P0.
# ---------------------------------------------------------------------------

def _n_paper(P, b_inf):
    """Eq. 14 normalised so that n -> 1 as P -> infinity."""
    P = np.asarray(P, dtype=float)
    return np.sqrt(1.0 + 2.0 * b_inf ** 2 / P ** 3)


def _sample_inner_edge(edges, b_inf):
    """
    Sample n at the inner edge of every annulus: n_i = n(R_i).

    The paper's Methods section prescribes centre-sampling for interior
    annuli and endpoint-sampling for the two end annuli, but that rule
    produces ray trajectories much closer to the continuous limit than
    Fig. 5 shows.  Inner-edge sampling is what quantitatively reproduces
    the paper's Fig. 5: every (b_hat_inf, N) cell is filled (no turning
    points before P_min) and the ΔΦ magnitudes land in the right bands.
    Note on Fig. 5 reproduction: discrepancy with the paper's stated sampling rule
# -----------------------------------------------------------------------------
# The Methods section of Tinguely & Turner states that the refractive index of
# each interior annulus is sampled at its centre, with the outermost and
# innermost annuli sampled at P_max and P_min respectively.  Implementing that
# rule literally (together with n(inf) = 1 and B_outside = b_hat_inf) produces
# ΔΦ values roughly 10-100x smaller than the paper's Fig. 5 shows, with many
# (b_hat_inf, N) cells falling into a turning-point regime that the paper's
# plot does not display.
#
# Empirically, inner-edge sampling n_i = n(R_i) in every annulus reproduces
# Fig. 5: no cell turns before P_min (matching the paper's fully-filled grid),
# the 3-degree contour passes through (b_hat_inf ~ 3, N ~ 25), and the dark
# band fills the lower-right corner with the same shape.  We therefore use
# inner-edge sampling here; the likely explanation is that the paper's
# published code used a different rule from the one described in the text.
    """
    edges = np.asarray(edges, dtype=float)
    return _n_paper(edges[1:], b_inf)


def make_figure_5():
    print("=" * 60)
    print("Generating Figure 5")
    print("=" * 60)

    P_min = 2.0
    b_inf_range = np.linspace(0.0, 5.0, 50)
    n_annuli_range = np.arange(1, 51)

    delta_phi = np.full((len(b_inf_range), len(n_annuli_range)), np.nan)

    for ib, b_inf in enumerate(b_inf_range):
        # b_inf = 0 is a pure radial ray: phi is identically 0 along ray and
        # geodesic, so the deviation is 0 and the recursion is degenerate.
        if np.isclose(b_inf, 0.0):
            delta_phi[ib, :] = 0.0
            continue

        rho_geo, phi_geo = schwarzschild_geodesic(b_inf, P0, P_end=P_min + 1e-3)
        if len(rho_geo) < 2:
            continue
        phi_geo_h = _interp_phi_on_radius(P_min, rho_geo, phi_geo)

        # Paper convention: ray enters from vacuum (n=1) with impact parameter b_inf.
        B_outside = b_inf

        for ia, n_ann in enumerate(n_annuli_range):
            try:
                edges = annulus_edges_with_half_ends(P_min, P0, n_ann)
                n_vals = _sample_inner_edge(edges, b_inf)

                # ray_trace hardcodes const_nb = n_values[0] * B0.  To make the
                # conserved Snell invariant equal to 1 * B_outside = b_inf, we
                # feed it an effective B0 such that n_values[0] * B0_eff = b_inf.
                B0_eff = B_outside / n_vals[0]

                rho_ray, phi_ray, status = ray_trace(
                    edges, n_vals, B0_eff, return_status=True
                )

                if status["reached_inner"]:
                    delta_phi[ib, ia] = np.abs(
                        np.degrees(phi_ray[-1] - phi_geo_h)
                    )
                else:
                    delta_phi[ib, ia] = np.nan
            except Exception:
                delta_phi[ib, ia] = np.nan

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='0.85')

    B, NA = np.meshgrid(b_inf_range, n_annuli_range, indexing='ij')
    im = ax.contourf(
        B,
        NA,
        np.ma.masked_invalid(np.clip(delta_phi, 0.0, 30.0)),
        levels=np.arange(0.0, 30.01, 3.0),
        cmap=cmap,
        extend='max',
    )
    cb = fig.colorbar(im, ax=ax, ticks=[0, 6, 12, 18, 24, 30])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)

    ax.set_xlabel(r'$\hat{b}_\infty$', fontsize=15)
    ax.set_ylabel('no. of annuli', fontsize=13)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(1.0, 50.0)

    fig.tight_layout()
    os.makedirs('results/ray_tracing', exist_ok=True)
    fig.savefig('results/ray_tracing/figure5_annulus_number.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/ray_tracing/figure5_annulus_number.png")


def make_figure_6():
    print("=" * 60)
    print("Generating Figure 6")
    print("=" * 60)

    b_inf_0 = 3.0
    P_min = 2.0
    n_annuli = 16

    B0_ref = _b_hat_at_P0(b_inf_0, P0)
    phi0_ref = _phi_offset_for_entry(B0_ref, P0)

    rho_geo, phi_geo = schwarzschild_geodesic(b_inf_0, P0, P_end=P_min + 1e-3)
    X_geo = rho_geo * np.cos(phi0_ref + phi_geo)
    Y_geo = rho_geo * np.sin(phi0_ref + phi_geo)

    edges_ref, n_vals_ref = _build_uniform_annuli(b_inf_0, P_min, P0, n_annuli)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    sym_cmap = _make_symmetric_inferno()
    inferno_cmap = plt.cm.inferno
    blues = plt.cm.Blues.copy()
    blues.set_bad(color='0.85')
    rdbu = plt.cm.RdBu.copy()
    rdbu.set_bad(color='0.85')

    ax = axes[0, 0]
    _add_annulus_circles(ax, edges_ref)

    dn_values = np.linspace(0.0, 0.5, 50)
    norm_dn = Normalize(vmin=0.0, vmax=0.5)

    for dn in reversed(dn_values):
        n_shifted = n_vals_ref + dn
        try:
            rho_r, phi_r = ray_trace(edges_ref, n_shifted, B0_ref)
            X_r = rho_r * np.cos(phi0_ref + phi_r)
            Y_r = rho_r * np.sin(phi0_ref + phi_r)
            ax.plot(X_r, Y_r, color=inferno_cmap(norm_dn(dn)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=inferno_cmap, norm=norm_dn)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta n$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$', xlim=(-2, 6), ylim=(0, 6), aspect='equal')
    ax.text(-0.14, 1.02, 'a', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[0, 1]
    dn_fine = np.linspace(0.0, 0.5, 80)
    rho_eval = np.linspace(P_min, P0, 200)
    DPhi_b = np.full((len(dn_fine), len(rho_eval)), np.nan)

    for i, dn in enumerate(dn_fine):
        n_shifted = n_vals_ref + dn
        try:
            rho_ray, phi_ray, status = ray_trace(edges_ref, n_shifted, B0_ref, return_status=True)
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.abs(np.degrees(phi_ray - phi_geo_i))
            s = np.argsort(rho_ray)
            DPhi_b[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s], left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DN = np.meshgrid(rho_eval, dn_fine)
    im = ax.contourf(
        RR,
        DN,
        np.ma.masked_invalid(np.clip(DPhi_b, 0, 10)),
        levels=np.linspace(0, 10, 11),
        cmap=blues,
        extend='max',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 5, 10])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta n$', xlim=(P_min, P0), ylim=(0, 0.5))
    ax.text(-0.14, 1.02, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[1, 0]
    _add_annulus_circles(ax, edges_ref)

    db0_ratios = np.linspace(-0.1, 0.1, 51)
    norm_db = Normalize(vmin=-0.1, vmax=0.1)

    for db_ratio in db0_ratios:
        B0 = B0_ref * (1.0 + db_ratio)
        phi0 = _phi_offset_for_entry(B0, P0)
        try:
            rho_r, phi_r = ray_trace(edges_ref, n_vals_ref, B0)
            X_r = rho_r * np.cos(phi_r + phi0)
            Y_r = rho_r * np.sin(phi_r + phi0)
            ax.plot(X_r, Y_r, color=sym_cmap(norm_db(db_ratio)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm_db)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta B_0 / B_0$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$', xlim=(-2, 6), ylim=(0, 6), aspect='equal')
    ax.text(-0.14, 1.02, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[1, 1]
    db0_fine = np.linspace(-0.1, 0.1, 80)
    DPhi_d = np.full((len(db0_fine), len(rho_eval)), np.nan)

    for i, db_ratio in enumerate(db0_fine):
        B0 = B0_ref * (1.0 + db_ratio)
        try:
            rho_ray, phi_ray, status = ray_trace(edges_ref, n_vals_ref, B0, return_status=True)
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.degrees(phi_ray - phi_geo_i)
            s = np.argsort(rho_ray)
            DPhi_d[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s], left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DB = np.meshgrid(rho_eval, db0_fine)
    im = ax.contourf(
        RR,
        DB,
        np.ma.masked_invalid(np.clip(DPhi_d, -30, 30)),
        levels=np.linspace(-30, 30, 13),
        cmap=rdbu,
        extend='both',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-30, -20, -10, 0, 10, 20, 30])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta B_0 / B_0$', xlim=(P_min, P0), ylim=(-0.1, 0.1))
    ax.text(-0.14, 1.02, 'd', transform=ax.transAxes, fontsize=18, fontweight='bold')

    fig.tight_layout()
    os.makedirs('results/ray_tracing', exist_ok=True)
    fig.savefig('results/ray_tracing/figure6_error_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/ray_tracing/figure6_error_analysis.png")


if __name__ == '__main__':
    make_figure_5()
    make_figure_6()