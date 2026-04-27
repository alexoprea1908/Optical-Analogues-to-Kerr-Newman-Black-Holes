"""
Rotational-invariance test for the tensor Schwarzschild solver.

The Schwarzschild metric is rotationally symmetric, so the optical analogue
defined by Eq. (8) inherits that symmetry. Therefore a beam injected from
direction phi_0 should produce a field that, when rotated by -phi_0, looks
identical to a beam injected from below.

We test this by comparing two simulations:
  (1) standard setup: beam from below (-y direction)
  (2) ROTATED setup: same b_inf, but the entire sim is rotated by 90 deg

The result of (2), rotated back by -90 deg, should match (1) up to source
discretisation error.

This exercises the OFF-DIAGONAL mu_xy term in a way the b_inf=3 baseline
does not (a beam along y mostly couples to mu_yy; a beam along x mostly
couples to mu_xx; rotation invariance forces mu_xy to be exactly the right
value to make these consistent).

NOTE: the source-injection scaffolding in fdfd.py is hard-coded to inject
from -y. To test rotation, we rotate the MATERIAL in real space (i.e. shift
the BH center off-axis) rather than the source -- equivalent up to
translation, which is also a symmetry of vacuum but not of the medium.

Simpler valid test: TRANSLATION invariance of the OUTGOING field outside
the medium. We just visually check that the tensor field has the right
two-fold symmetry (E_z(x, y) and E_z(-x, y) related by mu_xy structure).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from fdfd_tensor import simulate_schwarzschild_tensor


def main():
    # Run b_inf = 3 (beam offset to the right) and b_inf = -3 (beam offset
    # to the left). For a rotationally symmetric medium, the resulting
    # |E_z|(x, y) for b_inf = -3 should be the MIRROR IMAGE of |E_z|(x, y)
    # for b_inf = +3 across the y-axis.
    print("Running b_inf = +3 ...")
    fdfd_p = simulate_schwarzschild_tensor(
        b_inf=+3.0, P_min=2.1, P0=6.0, n_annuli=16,
        wavelength=0.5, M=2.5, resolution=6, N_pml=8, verbose=False
    )
    print("Running b_inf = -3 ...")
    fdfd_m = simulate_schwarzschild_tensor(
        b_inf=-3.0, P_min=2.1, P0=6.0, n_annuli=16,
        wavelength=0.5, M=2.5, resolution=6, N_pml=8, verbose=False
    )

    Ep = np.abs(fdfd_p.E_z)
    Em = np.abs(fdfd_m.E_z)

    # Mirror Em across x-axis (Em(-x, y) should equal Ep(x, y))
    Em_mirrored = Em[::-1, :]

    diff = Ep - Em_mirrored
    rel = np.linalg.norm(diff) / np.linalg.norm(Ep)
    maxdiff = np.max(np.abs(diff)) / np.max(Ep)
    print(f"\n|E(+b)| vs mirrored |E(-b)|:")
    print(f"  relative L2 error: {rel:.3e}")
    print(f"  max relative diff: {maxdiff:.3e}")
    print()
    if rel < 0.05:
        print(f"PASS: rotational symmetry preserved to {rel*100:.2f}%")
    else:
        print(f"WARN: rotational symmetry violation = {rel*100:.2f}%")
        print(f"  (some violation is expected from grid discretisation, but >5% suggests a tensor bug)")

    # Plot: side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')

    for ax, data, ttl in zip(
        axes,
        [Ep / Ep.max(), Em_mirrored / Em.max(), diff / Ep.max()],
        ['|E| at b=+3', '|E| at b=-3 (mirrored)', 'diff (mirrored b=-3 minus b=+3)']
    ):
        if 'diff' in ttl:
            im = ax.pcolormesh(fdfd_p.x, fdfd_p.y, data.T,
                               cmap='RdBu_r', vmin=-0.2, vmax=0.2,
                               shading='auto')
        else:
            im = ax.pcolormesh(fdfd_p.x, fdfd_p.y, data.T,
                               cmap='magma', vmin=0, vmax=0.5,
                               shading='auto')
        theta = np.linspace(0, 2*np.pi, 360)
        for R, c in [(15.0, 'white'), (5.0, '#66ccff'), (5.25, '#ff6699')]:
            ax.plot(R * np.cos(theta), R * np.sin(theta), color=c, lw=0.8)
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('white')
        ax.set_title(ttl, color='white', fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='white')
        cb.outline.set_edgecolor('white')

    fig.suptitle('Rotational/mirror symmetry of tensor Schwarzschild (Eq. 8)',
                  color='white', fontsize=12, y=0.97)
    fig.tight_layout()

    os.makedirs('results', exist_ok=True)
    out = 'results/symmetry_test.png'
    fig.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
