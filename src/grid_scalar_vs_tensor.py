"""
Tensor vs scalar Schwarzschild: full Fig. 3-style 2x2 grid for b_inf in
{2, 3, 4, 5}, scalar on top, tensor on bottom. The geodesic is overlaid in
each panel to verify that the polarisation-preserving tensor analogue still
reproduces the trajectory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fdfd import simulate_schwarzschild
from fdfd_tensor import simulate_schwarzschild_tensor
from geodesics import schwarzschild_geodesic_xy


WAVELENGTH = 0.5
M = 2.5
P0 = 6.0
P_MIN = 2.1
N_ANNULI = 16
RESOLUTION = 6         # coarse for speed
N_PML = 8
B_INFS = [2.0, 3.0, 4.0, 5.0]


def geodesic_xy_rotated(b_inf, P0, M):
    """Geodesic in real-space coordinates, entry from below at x = b_inf*M."""
    _, _, rho, phi = schwarzschild_geodesic_xy(b_inf, P0, P_end=P_MIN + 0.01)
    if len(rho) < 2:
        return np.array([]), np.array([])

    if b_inf >= P0:
        phi_offset = -np.pi / 2
    else:
        phi_offset = np.arctan2(-np.sqrt(P0**2 - b_inf**2), b_inf)

    X = rho * np.cos(phi + phi_offset) * M
    Y = rho * np.sin(phi + phi_offset) * M
    entry_x, entry_y = X[0], Y[0]
    Xline = np.array([entry_x, entry_x])
    Yline = np.array([-P0 * M * 1.05, entry_y])
    return np.concatenate([Xline, X]), np.concatenate([Yline, Y])


def render_panel(ax, fdfd, b_inf, title, with_geodesic=True):
    R_S = 2.0 * M
    R0_phys = P0 * M
    R_min_phys = P_MIN * M

    E_norm = np.abs(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
    im = ax.pcolormesh(fdfd.x, fdfd.y, E_norm.T,
                       cmap='magma', vmin=0, vmax=0.5,
                       shading='auto', rasterized=True)

    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(R0_phys * np.cos(theta), R0_phys * np.sin(theta),
            color='white', lw=1.0, alpha=0.85)
    ax.plot(R_S * np.cos(theta), R_S * np.sin(theta),
            color='#66ccff', lw=1.2)
    ax.plot(R_min_phys * np.cos(theta), R_min_phys * np.sin(theta),
            color='#ff6699', lw=0.9, ls='--')

    if with_geodesic:
        Xg, Yg = geodesic_xy_rotated(b_inf, P0, M)
        ax.plot(Xg, Yg, color='#88ff88', lw=1.6, alpha=0.9)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('white')
    ax.set_title(title, color='white', fontsize=10)
    return im


def main():
    fig = plt.figure(figsize=(16, 8.5), facecolor='black')
    gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.15)

    print("Running 8 simulations (4 scalar + 4 tensor)...")

    last_im = None
    for col, b_inf in enumerate(B_INFS):
        print(f"\n[col {col}] b_inf = {b_inf}")

        # Scalar (top row)
        print("  scalar...", end=" ", flush=True)
        fdfd_s = simulate_schwarzschild(
            b_inf=b_inf, P_min=P_MIN, P0=P0, n_annuli=N_ANNULI,
            wavelength=WAVELENGTH, M=M, resolution=RESOLUTION,
            N_pml=N_PML, verbose=False
        )
        print("done.")

        ax = fig.add_subplot(gs[0, col])
        title = (rf'scalar, $\hat b_\infty={b_inf:g}$' if col == 0
                 else rf'$\hat b_\infty={b_inf:g}$')
        last_im = render_panel(ax, fdfd_s, b_inf, title)
        if col > 0:
            ax.set_yticklabels([])

        # Tensor (bottom row)
        print("  tensor...", end=" ", flush=True)
        fdfd_t = simulate_schwarzschild_tensor(
            b_inf=b_inf, P_min=P_MIN, P0=P0, n_annuli=N_ANNULI,
            wavelength=WAVELENGTH, M=M, resolution=RESOLUTION,
            N_pml=N_PML, verbose=False
        )
        print("done.")

        ax = fig.add_subplot(gs[1, col])
        title = (rf'tensor, $\hat b_\infty={b_inf:g}$' if col == 0
                 else rf'$\hat b_\infty={b_inf:g}$')
        render_panel(ax, fdfd_t, b_inf, title)
        if col > 0:
            ax.set_yticklabels([])

    # Row labels on the leftmost panels
    fig.text(0.02, 0.74, 'Scalar n(P) [Eq. 14]',
             rotation=90, ha='center', va='center',
             color='white', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.30, r'Tensor $\epsilon^{ij}=\mu^{ij}$ [Eq. 8]',
             rotation=90, ha='center', va='center',
             color='white', fontsize=12, fontweight='bold')

    fig.suptitle('Optical Schwarzschild: scalar (geodesic-only) vs tensor '
                 '(geodesic + polarisation),  P_min=2.1',
                 color='white', fontsize=13, y=0.96)

    # One shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label(r'$|E|/\max|E|$', color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')

    os.makedirs('results', exist_ok=True)
    out = 'results/schwarzschild_scalar_vs_tensor_grid.png'
    fig.savefig(out, dpi=170, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
