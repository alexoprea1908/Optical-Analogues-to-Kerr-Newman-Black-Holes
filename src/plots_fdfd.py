"""
Reproduce Figs. 3 and 4 of the paper:
    Fig. 3: FDFD simulations of optical Schwarzschild black holes
            (b_inf = 2, 3, 4, 5)
    Fig. 4: FDFD simulations of optical Kerr-Newman black holes
            (4 cases with b_inf = 3)

Paper parameters:
    lambda = 0.5 um, R_S = 2M = 5 um (M = 2.5 um), R_0 = P0*M = 15 um
    Domain: 60*lambda x 60*lambda = 30 x 30 um
    PML: lambda/5
    Beam: Gaussian with delta = lambda/2, truncated at 2*lambda
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from fdfd import simulate_schwarzschild, simulate_kerr_newman
from geodesics import schwarzschild_geodesic_xy, kerr_newman_geodesic_xy
from ray_tracing import build_schwarzschild_annuli, build_kn_annuli
from cases import SCHWARZSCHILD_CASES, KERR_NEWMAN_CASES


# ============================================================================
# Common parameters matching the paper
# ============================================================================
WAVELENGTH = 0.5       # um
M = 2.5                # um (so R_S = 5 um)
P0 = 6.0               # dimensionless outer radius
RESOLUTION = 10        # grid points per wavelength (paper uses ~25)
N_ANNULI_SCH = 16      # Schwarzschild: 16 annuli (paper)
N_ANNULI_KN = 21       # Kerr-Newman: 21 annuli (paper)


def _add_annulus_edges(ax, edges, M_scale, color='w', lw=0.3, alpha=0.5):
    """Draw annulus edges as circles."""
    theta = np.linspace(0, 2 * np.pi, 300)
    for e in edges:
        r = e * M_scale
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=color, lw=lw, alpha=alpha)


def _add_geodesic(ax, X, Y, color='w', lw=2):
    """Overlay the true geodesic trajectory."""
    ax.plot(X, Y, color=color, lw=lw, alpha=0.9)


def _add_poynting_vectors(ax, fdfd, step=8, M_scale=1.0, color='w'):
    """Overlay scaled Poynting vectors."""
    S_x, S_y = fdfd.compute_poynting()

    x_sub = fdfd.x[::step]
    y_sub = fdfd.y[::step]
    Sx_sub = S_x[::step, ::step]
    Sy_sub = S_y[::step, ::step]

    XX, YY = np.meshgrid(x_sub, y_sub, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)

    # Scale by 1/R as in paper
    Smag = np.sqrt(Sx_sub**2 + Sy_sub**2)
    scale = 1.0 / np.maximum(R, 0.5)
    Smag_scaled = Smag * scale

    # Only show vectors with significant magnitude
    thresh = 0.02 * np.max(Smag_scaled)
    mask = Smag_scaled > thresh

    if np.any(mask):
        ax.quiver(XX[mask], YY[mask],
                  Sx_sub[mask] * scale[mask],
                  Sy_sub[mask] * scale[mask],
                  color=color, alpha=0.5, scale=None,
                  headwidth=3, headlength=4, width=0.002)


def _geodesic_rotated_xy(rho, phi, b_inf, P0, M):
    """
    Rotate the geodesic so it enters the BH at x = b_inf*M from below,
    matching the beam's impact parameter.
    Prepends a straight vertical approach line from the bottom of the plot.
    """
    if len(rho) < 2:
        return np.array([]), np.array([])

    # Entry point on outer circle: (b_inf, -sqrt(P0^2 - b_inf^2))
    if b_inf >= P0:
        phi_offset = -np.pi / 2
    else:
        phi_offset = np.arctan2(-np.sqrt(P0**2 - b_inf**2), b_inf)

    X_geo = rho * np.cos(phi + phi_offset) * M
    Y_geo = rho * np.sin(phi + phi_offset) * M

    # Prepend straight vertical line from bottom of plot to entry point
    entry_x = X_geo[0]
    entry_y = Y_geo[0]
    y_bottom = -P0 * M * 1.05
    X_line = np.array([entry_x, entry_x])
    Y_line = np.array([y_bottom, entry_y])

    X = np.concatenate([X_line, X_geo])
    Y = np.concatenate([Y_line, Y_geo])
    return X, Y


# ============================================================================
# Figure 3: Schwarzschild FDFD simulations
# ============================================================================

def make_figure_3():
    """
    Fig. 3: FDFD simulations of optical Schwarzschild black holes.
    Four panels: b_inf = 2, 3, 4, 5.
    """
    print("=" * 60)
    print("Generating Figure 3: Schwarzschild FDFD simulations")
    print("=" * 60)

    b_inf_values = [2.0, 3.0, 4.0, 5.0]
    panel_labels = ['a', 'b', 'c', 'd']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    R_S = 2.0 * M  # Schwarzschild radius in um
    R0_phys = P0 * M

    for idx, (b_inf, label) in enumerate(zip(b_inf_values, panel_labels)):
        print(f"\n--- Panel {label}: b_inf = {b_inf} ---")
        ax = axes[idx]

        # Run FDFD simulation
        fdfd = simulate_schwarzschild(
            b_inf=b_inf, P_min=2.0, P0=P0,
            n_annuli=N_ANNULI_SCH, wavelength=WAVELENGTH, M=M,
            resolution=RESOLUTION, verbose=True
        )

        # Plot |E|/max|E|
        E_norm = np.abs(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
        im = ax.pcolormesh(fdfd.x, fdfd.y, E_norm.T,
                           cmap='inferno', vmin=0, vmax=0.5,
                           shading='auto', rasterized=True)

        # Draw annulus edges
        edges, _ = build_schwarzschild_annuli(b_inf, 2.0, P0, N_ANNULI_SCH)
        _add_annulus_edges(ax, edges, M)

        # Draw outer edge and Schwarzschild radius
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R0_phys * np.cos(theta), R0_phys * np.sin(theta),
                'w-', lw=1.5)
        ax.plot(R_S * np.cos(theta), R_S * np.sin(theta), 'w-', lw=1.5)
        ax.text(R_S * 0.7, R_S * 0.3, '$R_S$', color='w', fontsize=10)

        # True geodesic (rotated to match beam entry from below)
        _, _, rho_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf, P0, P_end=2.01)
        X_geo, Y_geo = _geodesic_rotated_xy(
            rho_geo, phi_geo, b_inf, P0, M)
        ax.plot(X_geo, Y_geo, 'w-', lw=2.5, alpha=0.9)

        # Poynting vectors
        _add_poynting_vectors(ax, fdfd, step=12)

        # Labels
        ax.text(0.05, 0.92, f'{label}', transform=ax.transAxes,
                fontsize=16, fontweight='bold', color='w')
        ax.text(0.05, 0.82, f'$\\hat{{b}}_\\infty = {int(b_inf)}$',
                transform=ax.transAxes, fontsize=12, color='w')

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('μm')
        ax.set_ylabel('μm')
        ax.set_aspect('equal')

        # Colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$|E|/\max|E|$')

    fig.suptitle('Numerical simulations of optical Schwarzschild black holes',
                 fontsize=14, y=0.98)
    fig.tight_layout()

    os.makedirs('results/fdfd', exist_ok=True)
    fig.savefig('results/fdfd/figure3_schwarzschild_fdfd.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("\nSaved results/fdfd/figure3_schwarzschild_fdfd.png")


# ============================================================================
# Figure 4: Kerr-Newman FDFD simulations
# ============================================================================

def make_figure_4():
    """
    Fig. 4: FDFD simulations of optical Kerr-Newman black holes.
    Four panels: extremal Kerr co-rot, extremal RN, KN co-rot, KN counter-rot.
    All with b_inf = 3.
    """
    print("=" * 60)
    print("Generating Figure 4: Kerr-Newman FDFD simulations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    R_S = 2.0 * M
    R0_phys = P0 * M
    B_INF = 3.0

    for idx, case in enumerate(KERR_NEWMAN_CASES):
        ax = axes[idx]
        label = case['panel']
        a = case['a']
        rho_Q = case['rho_Q']
        ell_sign = case['ell_sign']
        P_min = case['P_min']

        print(f"\n--- Panel {label}: a={a}, rho_Q={rho_Q} ---")

        # Run FDFD simulation
        fdfd = simulate_kerr_newman(
            a=a, rho_Q=rho_Q, b_inf=B_INF, ell_sign=ell_sign,
            P_min=P_min, P0=P0, n_annuli=N_ANNULI_KN,
            wavelength=WAVELENGTH, M=M, resolution=RESOLUTION, verbose=True
        )

        # Plot |E|/max|E|
        E_norm = np.abs(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
        im = ax.pcolormesh(fdfd.x, fdfd.y, E_norm.T,
                           cmap='inferno', vmin=0, vmax=0.5,
                           shading='auto', rasterized=True)

        # Annulus edges
        edges, _ = build_kn_annuli(a, rho_Q, B_INF, ell_sign,
                                    P_min, P0, N_ANNULI_KN)
        _add_annulus_edges(ax, edges, M)

        # Outer edge and important radii
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R0_phys * np.cos(theta), R0_phys * np.sin(theta),
                'w-', lw=1.5)

        # Horizon radius
        from Kerr_Newman import delta_hat
        # P_h where delta_hat = 0: P^2 - 2P + a^2 + rho_Q^2 = 0
        disc = 1.0 - a**2 - rho_Q**2
        if disc >= 0:
            P_h = 1.0 + np.sqrt(disc)
            R_h = P_h * M
            ax.plot(R_h * np.cos(theta), R_h * np.sin(theta), 'w--', lw=1.2)
            ax.text(R_h * 0.6, R_h * 0.3, '$R_h$', color='w', fontsize=10,
                    fontweight='bold')

        # Schwarzschild radius
        ax.plot(R_S * np.cos(theta), R_S * np.sin(theta), 'w-', lw=1.2)
        ax.text(-R_S * 0.3, R_S * 0.6, '$R_S$', color='w', fontsize=10,
                fontweight='bold')

        # Inner edge
        R_min = P_min * M
        ax.plot(R_min * np.cos(theta), R_min * np.sin(theta), 'w-', lw=1.5)

        # True geodesic (rotated to match beam entry from below)
        _, _, rho_geo, phi_geo = kerr_newman_geodesic_xy(
            a, rho_Q, B_INF, ell_sign, P0, P_min + 0.01
        )
        X_geo, Y_geo = _geodesic_rotated_xy(
            rho_geo, phi_geo, B_INF, P0, M)
        ax.plot(X_geo, Y_geo, 'w-', lw=2.5, alpha=0.9)

        # Poynting vectors
        _add_poynting_vectors(ax, fdfd, step=12)

        # Label
        ax.text(0.05, 0.92, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', color='w')

        # Subtitle with parameters
        subtitle = f'$\\hat{{a}}={a:.1f}$, $\\rho_Q={rho_Q:.1f}$'
        ax.text(0.05, 0.02, subtitle, transform=ax.transAxes,
                fontsize=10, color='w')

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('μm')
        ax.set_ylabel('μm')
        ax.set_aspect('equal')

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$|E|/\max|E|$')

    fig.suptitle('Numerical simulations of optical Kerr–Newman black holes',
                 fontsize=14, y=0.98)
    fig.tight_layout()

    os.makedirs('results/fdfd', exist_ok=True)
    fig.savefig('results/fdfd/figure4_kerr_newman_fdfd.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("\nSaved results/fdfd/figure4_kerr_newman_fdfd.png")


if __name__ == "__main__":
    make_figure_3()
    make_figure_4()