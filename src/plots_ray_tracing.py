"""
Reproduce Figs. 5 and 6 of the paper:
    Fig. 5: Impact of annulus number on ray trajectories
            (angular deviation vs b_inf and number of annuli)
    Fig. 6: Effects of construction and experimenter errors
            (a,b: uniform Dn offset; c,d: impact parameter deviation)

All for optical Schwarzschild black hole with P0 = 6.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from geodesics import schwarzschild_geodesic
from ray_tracing import ray_trace, ray_trace_xy
from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
from Schwarzchild import refractive_index_schwarzschild
from constants import P0


# ============================================================================
# Helpers
# ============================================================================

def _build_uniform_annuli(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    """Build annuli with uniform widths (half-width ends) for Schwarzschild."""
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P_max
    )
    return edges, n_values


def _geodesic_phi_at_horizon(b_inf, P0=6.0, P_horizon=2.0):
    """Compute geodesic azimuthal angle at the horizon."""
    rho_geo, phi_geo = schwarzschild_geodesic(b_inf, P0, P_end=P_horizon + 0.001)
    if len(rho_geo) < 2:
        return np.nan
    # Interpolate to exact horizon radius
    return np.interp(P_horizon, rho_geo[::-1], phi_geo[::-1])


# ============================================================================
# Figure 5: Impact of annulus number on ray trajectories
# ============================================================================

def make_figure_5():
    """
    Fig. 5: Angular deviation at the horizon vs b_inf and number of annuli.

    Paper: b_inf in [0, 5], annuli 1-50, P0=6.
    """
    print("=" * 60)
    print("Generating Figure 5: Impact of annulus number")
    print("=" * 60)

    b_inf_range = np.linspace(0.1, 5.0, 50)
    n_annuli_range = np.arange(1, 51)

    # Compute deviation matrix
    delta_phi = np.full((len(b_inf_range), len(n_annuli_range)), np.nan)

    for ib, b_inf in enumerate(b_inf_range):
        # Compute geodesic
        rho_geo, phi_geo = schwarzschild_geodesic(b_inf, P0, P_end=2.001)
        if len(rho_geo) < 2:
            continue
        phi_geo_horizon = np.interp(2.0, rho_geo[::-1], phi_geo[::-1])

        for ia, n_ann in enumerate(n_annuli_range):
            if n_ann < 2:
                continue
            try:
                edges, n_vals = _build_uniform_annuli(b_inf, 2.0, P0, n_ann)
                rho_ray, phi_ray = ray_trace(edges, n_vals, b_inf)

                # Ray angle at horizon (or closest approach)
                idx = np.argmin(np.abs(rho_ray - 2.0))
                phi_ray_h = phi_ray[idx]

                dphi = np.abs(np.degrees(phi_ray_h - phi_geo_horizon))
                delta_phi[ib, ia] = dphi
            except Exception:
                pass

        if (ib + 1) % 10 == 0:
            print(f"  b_inf = {b_inf:.1f} done ({ib+1}/{len(b_inf_range)})")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    B, NA = np.meshgrid(b_inf_range, n_annuli_range, indexing='ij')

    # Clip to [0, 30] degrees as in paper
    delta_phi_clipped = np.clip(delta_phi, 0, 30)

    im = ax.pcolormesh(B, NA, delta_phi_clipped, cmap='viridis',
                       vmin=0, vmax=30, shading='auto')

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}$ (°)')

    ax.set_xlabel(r'$\hat{b}_\infty$', fontsize=14)
    ax.set_ylabel('no. of annuli', fontsize=12)
    ax.set_title('Impact of annulus number on ray trajectories', fontsize=13)

    fig.tight_layout()
    os.makedirs('results/ray_tracing', exist_ok=True)
    fig.savefig('results/ray_tracing/figure5_annulus_number.png', dpi=200)
    plt.close(fig)
    print("Saved results/ray_tracing/figure5_annulus_number.png")


# ============================================================================
# Figure 6: Effects of construction and experimenter errors
# ============================================================================

def make_figure_6():
    """
    Fig. 6: Error analysis for Schwarzschild b_inf=3.
        (a) Ray trajectories with uniform Dn offset
        (b) Angular deviation vs radius for Dn offset
        (c) Ray trajectories with DB0 offset
        (d) Angular deviation vs radius for DB0 offset
    """
    print("=" * 60)
    print("Generating Figure 6: Error analysis")
    print("=" * 60)

    b_inf_0 = 3.0
    P_min = 2.0
    n_annuli = 21

    # Reference geodesic
    rho_geo, phi_geo = schwarzschild_geodesic(b_inf_0, P0, P_end=P_min + 0.001)

    # Build reference annular system
    edges_ref, n_vals_ref = _build_uniform_annuli(b_inf_0, P_min, P0, n_annuli)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel (a): Ray trajectories with Dn offset ---
    ax = axes[0, 0]
    dn_values = np.linspace(0, 0.5, 11)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=0, vmax=0.5)

    # Geodesic
    X_geo = rho_geo * np.cos(phi_geo)
    Y_geo = rho_geo * np.sin(phi_geo)
    ax.plot(X_geo, Y_geo, 'k--', lw=1.5, label='geodesic')

    for dn in dn_values:
        n_vals_shifted = n_vals_ref + dn
        try:
            X_r, Y_r, rho_r, phi_r = ray_trace_xy(edges_ref, n_vals_shifted, b_inf_0)
            ax.plot(X_r, Y_r, color=cmap(norm(dn)), lw=0.8)
        except Exception:
            pass

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(r'$\Delta n$')

    ax.set_xlabel(r'$X/M$')
    ax.set_ylabel(r'$Y/M$')
    ax.set_xlim(-2, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.text(0.05, 0.92, 'a', transform=ax.transAxes,
            fontsize=14, fontweight='bold')

    # --- Panel (b): DPhi vs radius for Dn offset ---
    ax = axes[0, 1]
    dn_fine = np.linspace(0, 0.5, 30)

    for dn in dn_fine:
        n_vals_shifted = n_vals_ref + dn
        try:
            rho_ray, phi_ray = ray_trace(edges_ref, n_vals_shifted, b_inf_0)
            # Interpolate geodesic to ray radii
            phi_geo_interp = np.interp(rho_ray, rho_geo[::-1], phi_geo[::-1])
            dphi = np.degrees(phi_ray - phi_geo_interp)
            ax.plot(rho_ray, [dn] * len(rho_ray), '.', color=cmap(norm(dn)),
                    markersize=1)
        except Exception:
            pass

    # Better: make a 2D plot
    # Redo as pcolormesh
    rho_eval = np.linspace(P_min, P0, 100)
    DPhi_grid = np.full((len(dn_fine), len(rho_eval)), np.nan)

    for i, dn in enumerate(dn_fine):
        n_vals_shifted = n_vals_ref + dn
        try:
            rho_ray, phi_ray = ray_trace(edges_ref, n_vals_shifted, b_inf_0)
            phi_geo_interp = np.interp(rho_ray, rho_geo[::-1], phi_geo[::-1])
            dphi_ray = np.degrees(phi_ray - phi_geo_interp)
            # Interpolate onto rho_eval
            dphi_interp = np.interp(rho_eval, rho_ray[::-1], dphi_ray[::-1],
                                    left=np.nan, right=np.nan)
            DPhi_grid[i, :] = dphi_interp
        except Exception:
            pass

    ax.clear()
    RR, DN = np.meshgrid(rho_eval, dn_fine)
    im = ax.pcolormesh(RR, DN, np.abs(DPhi_grid), cmap='viridis',
                       vmin=0, vmax=10, shading='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$|\Phi_{\rm ray} - \Phi_{\rm geo}|$ (°)')
    ax.set_xlabel(r'$R/M$')
    ax.set_ylabel(r'$\Delta n$')
    ax.text(0.05, 0.92, 'b', transform=ax.transAxes,
            fontsize=14, fontweight='bold')

    # --- Panel (c): Ray trajectories with DB0 offset ---
    ax = axes[1, 0]
    db0_ratios = np.linspace(-0.1, 0.1, 11)
    norm_db = Normalize(vmin=-0.1, vmax=0.1)

    ax.plot(X_geo, Y_geo, 'k--', lw=1.5, label='geodesic')

    for db_ratio in db0_ratios:
        b_shifted = b_inf_0 * (1.0 + db_ratio)
        # Rebuild the annular system for original b_inf (system doesn't change)
        # but trace with shifted impact parameter
        try:
            X_r, Y_r, _, _ = ray_trace_xy(edges_ref, n_vals_ref, b_shifted)
            ax.plot(X_r, Y_r, color=cmap(norm_db(db_ratio)), lw=0.8)
        except Exception:
            pass

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_db)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(r'$\Delta B_0 / B_0$')

    ax.set_xlabel(r'$X/M$')
    ax.set_ylabel(r'$Y/M$')
    ax.set_xlim(-2, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.text(0.05, 0.92, 'c', transform=ax.transAxes,
            fontsize=14, fontweight='bold')

    # --- Panel (d): DPhi vs radius for DB0 offset ---
    ax = axes[1, 1]
    db0_fine = np.linspace(-0.1, 0.1, 30)
    DPhi_grid_b = np.full((len(db0_fine), len(rho_eval)), np.nan)

    for i, db_ratio in enumerate(db0_fine):
        b_shifted = b_inf_0 * (1.0 + db_ratio)
        try:
            rho_ray, phi_ray = ray_trace(edges_ref, n_vals_ref, b_shifted)
            phi_geo_interp = np.interp(rho_ray, rho_geo[::-1], phi_geo[::-1])
            dphi_ray = np.degrees(phi_ray - phi_geo_interp)
            dphi_interp = np.interp(rho_eval, rho_ray[::-1], dphi_ray[::-1],
                                    left=np.nan, right=np.nan)
            DPhi_grid_b[i, :] = dphi_interp
        except Exception:
            pass

    RR, DB = np.meshgrid(rho_eval, db0_fine)
    im = ax.pcolormesh(RR, DB, DPhi_grid_b, cmap='RdBu_r',
                       vmin=-30, vmax=30, shading='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}$ (°)')
    ax.set_xlabel(r'$R/M$')
    ax.set_ylabel(r'$\Delta B_0 / B_0$')
    ax.text(0.05, 0.92, 'd', transform=ax.transAxes,
            fontsize=14, fontweight='bold')

    fig.suptitle('Effects of construction and experimenter errors on ray trajectories',
                 fontsize=13)
    fig.tight_layout()

    os.makedirs('results/ray_tracing', exist_ok=True)
    fig.savefig('results/ray_tracing/figure6_error_analysis.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("Saved results/ray_tracing/figure6_error_analysis.png")


if __name__ == "__main__":
    make_figure_5()
    make_figure_6()